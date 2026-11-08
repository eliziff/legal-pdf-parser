//! The structure stage's orchestrator: a PDF's extracted pages and lines in, its paragraphs, notes, headings and
//! structure graph out. Each stage is a crate of its own over the contract (`legal_pdf_structure_model`); this calls
//! them in order and caches what the first three read:
//!
//! 0. The pages normalized (`legal_pdf_evidence::normalize`): a transcript's line numbers parted from the lines
//!    they were read into, a table's rulings cleared from under its words, each line given its slot.
//! 1. Measured (`Evidence`): each page's words, rows and note separator; each line's type, its source's tag, its marker,
//!    note label, reference tokens, words and citations; the document's bookmarks and whether it is an instrument.
//! 2. Fenced and ordered (`Fences`): a transcript's line numbers, furniture and folios (`legal_pdf_furniture`), each
//!    page's columns, contents lists (`legal_pdf_headings`), tables, the note band under a separator, the body's type,
//!    the constituent documents (`legal_pdf_constituents`), the reading order, quotations and reading contexts, and each
//!    line's continuation of the one before.
//! 3. Numbered (`Runs`): each reading context's numbering runs, read and resolved by the engine.
//!
//!    What these stages read is cached by the input pages (`derive_cached`).
//! 4. Notes (`Notes`): each document's note band and the lanes that take more lines for notes (`legal_pdf_notes`).
//! 5. Decided (`transition::tier0`): every rule heard on every line (`Decision::hear`), one decision on each line in
//!    reading order (`settle`), the headings' levels (`Decision::finish`). Paragraph numbers are fixed input.
//! 6. Written once (`write_lines`): each line's region, note mode, block and place in its page's reading order.
//! 7. Derived from what was written: regions (`legal_pdf_layout::build_regions`), notes paired (the engine's pairer,
//!    `legal_pdf_notes::note_pairing`), propositions and cross-references (`note_context`), and the moves the decided
//!    lines make emitted as the structure graph (`transition::emit`, `legal_pdf_graph`).
//!
//! Measure a change with the grader's digest (`structure-gold digest`): every parse hashed, two builds compared.

pub use legal_pdf_structure_model::StructureIdentity;
use legal_pdf_decide as decide;
use legal_pdf_evidence::{normalize, Evidence};
use legal_pdf_fences::{decided, Fences};
use legal_pdf_structure_model::{Runs, Slots};
pub(crate) use decide::*;
use legal_pdf_diagnostics::*;
use legal_pdf_graph::graph;
pub use graph::PdfTextIndex;
use legal_pdf_layout::build_regions;
use legal_pdf_notes::{note_context, note_pairing};
use note_context::*;
/// The transition parser's walk over tier 0's reading, and its fitted readers (DOC with `encoder`).
use legal_pdf_transition as transition;
use legal_pdf_tier0 as tier0;
use legal_pdf_emit::Settled;
use legal_pdf_machine::lines::LineTable;
use std::path::Path;

use legal_pdf_core::model::{Diagnostic, Page, PdfOutlineEntry};
use legal_structure_model::StructureNode;
use legal_pdf_core::{Error, Result};
use legal_structure_model::DocumentStructure;
use std::collections::{HashMap, HashSet};

pub struct StructureOutput {
    /// The authority units, read off the rendered text (`emit`).
    pub authority_text_units: Vec<serde_json::Value>,
    pub diagnostics: Vec<Diagnostic>,
    pub structure_graph: DocumentStructure,
    /// Each heading line, and each line taken from the headings, with the rule that decided it
    /// ("+anchor", "-marginal_note").
    pub heading_sources: HashMap<String, String>,
}

/// What the measuring, fencing and numbering of a document read, cached by its input pages: the
/// evidence, the fences, contexts and reading order, the runs, and the text the runs' repair
/// rewrote.
#[derive(serde::Serialize, serde::Deserialize)]
struct Measured {
    evidence: Evidence,
    fences: Fences,
    runs: Runs,
}

/// A line written in the note band that no note holds (the pairer made no note of its label, or
/// bounded the note's body short of it) is the body's, and joins its paragraph rather than
/// standing alone: on the gold, 6,825 such lines were no notes, 2,146 were. Whether any was.
fn give_back(pages: &mut [Page], notes: &[StructureNode]) -> bool {
    let held: HashSet<&str> = notes.iter().flat_map(|note| &note.line_ids).map(String::as_str).collect();
    let mut given_back = false;
    for line in pages.iter_mut().flat_map(|page| &mut page.lines) {
        if line.region_type == "footnote" && !held.contains(line.id.as_str()) {
            line.region_type = "body".to_owned();
            line.note_region_mode.clear();
            given_back = true;
        }
    }
    given_back
}

pub fn derive(pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    derive_cached(pages, separators, outline, identity, None, None)
}

/// `derive`, reading what the measuring, fencing and numbering read from `cache` where it is there
/// (a file named for the input pages and the code that reads them) and writing it there where not.
/// The notes, the decision and everything after it are read on every run. With `trace`, the decision is also written
/// to that folder (`decide::trace`); the parse is the same.
pub fn derive_cached(pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    cache: Option<&std::path::Path>,
    trace: Option<&std::path::Path>,
) -> Result<StructureOutput> {
    use legal_pdf_core::profile::measure;
    if separators.len() != pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    // The document's type, where known (the caller's profile): an instrument numbers its provisions.
    let instrument = identity.profile == Some(legal_structure_model::DetectionProfile::Instrument);
    let (slots, normalized) = measure("derive.normalize", || normalize(pages));
    let cached = cache
        .and_then(|path| legal_pdf_core::read_gzip_json::<Measured>(path).ok())
        .filter(|measured| measured.evidence.size.len() == slots.len());
    // Each page's measures and columns, read once for every stage that reads the pages before they are written.
    let geometry = decide::Geometry::of(pages);
    let Measured { evidence, fences, runs } = match cached {
        Some(mut measured) => {
            for (slot, text, words) in &measured.runs.edits {
                let line = slots.line_mut(pages, *slot);
                line.text.clone_from(text);
                line.words.clone_from(words);
            }
            measured.evidence.items = normalized.items;
            measured
        }
        None => {
            let mut evidence = measure("derive.measure", || {
                legal_pdf_evidence::measure(pages, &slots, normalized, separators, outline, &identity.page_labels, instrument)
            });
            let mut fences = measure("derive.fences", || Fences::read(pages, &slots, &evidence, outline));
            let runs = measure("derive.runs", || legal_pdf_runs::read(pages, &slots, &mut evidence, &mut fences, &geometry))?;
            let measured = Measured { evidence, fences, runs };
            if let Some(path) = cache {
                let _ = legal_pdf_core::write_gzip_json(path, &measured);
            }
            measured
        }
    };
    // The output is tier 0's; the transition walk over its reading (the DOC tier where it is given, the scorer's fitting)
    // runs inside it, once (`tier0::output`).
    measure("derive.machine", || output(pages, &slots, &evidence, fences, &runs, &geometry, outline, &identity, trace))
}

/// The document's structure as the machine reads it, what `derive` gives where the request names `Reader::Machine`.
#[allow(clippy::too_many_arguments)]
/// Tier 0's reading of the document written: its decision (`tier0::decide`), the walk over it where DOC or the scorer's
/// fitting asks (`transition::over_tier0`), the lines written, the notes paired, the moves made (`tier0::moves_of`, DOC's
/// in their place where it is given) and emitted as the structure graph (`legal_pdf_emit`).
fn output(pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    mut fences: Fences,
    runs: &Runs,
    geometry: &decide::Geometry,
    outline: &[PdfOutlineEntry],
    identity: &StructureIdentity,
    trace: Option<&Path>,
) -> Result<StructureOutput> {
    // One line table for both tiers: tier 0 reads it, and the walk measures it for DOC.
    let table = LineTable::plain(pages, slots, evidence, &fences, &runs.before, transition::confusions());
    let views = tier0::line_views(pages, slots, evidence, &fences, &table);
    let (mut notes, mut decision) = tier0::decide(pages, slots, evidence, &fences, runs, geometry, outline, &table, &views, trace.is_some());
    // The walk over this reading where it is asked for (the DOC tier's, the scorer's fitting), read off the pages as they
    // came, before this reading writes and orders them.
    let doc = transition::over_tier0(pages, slots, evidence, &fences, geometry, outline, &identity.source_sha256, &notes, &decision, &table, &views);
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    diagnostics.append(&mut notes.diagnostics);
    diagnostics.append(&mut decision.diagnostics);
    if let Some(folder) = trace {
        write_trace(folder, &identity.source_sha256, pages, slots, evidence, &fences, runs, &notes, &decision);
    }
    let decided = decided(pages, slots, &fences);
    let heading_sources = write_lines(pages, slots, evidence, &fences, &notes, &decision);
    // The pairer reads each page's lines in reading order.
    in_reading_order(pages, slots, &fences);
    build_regions(pages);
    diagnostics.extend(text_flow_faults(pages));
    let index = PdfTextIndex::from_pages(pages);
    let citations = graph::citation_spans(pages, slots, evidence);
    let pairing = note_pairing::pair_notes(pages, &decided, &citations, &index);
    let notes = pairing.notes;
    diagnostics.extend(pairing.diagnostics);
    if give_back(pages, &notes) {
        build_regions(pages);
    }
    diagnostics.extend(unmatched_reference_diagnostics(pages, &notes, &index, &decided));
    let (moves, opened, within) = tier0::moves_of(slots, &fences, runs, &table, &views, &decision, &tier0::reading_regions(pages, &decided), &notes);
    // A tier scores the moves, the output stays this one: the walk takes the DOC tier's move for each line, but for the
    // notes, the pairer's (a line either reads as a note keeps this reading's move). DOC reads the same table, so its
    // reading of a marker names one of this reading's.
    tier0::in_slot_order(pages, slots);
    let settled = Settled { index: &index, notes, runs, opened, within };
    let (structure_graph, authority_text_units) = legal_pdf_emit::structure_graph(pages, slots, evidence, &fences, &decided, &table, &views, &moves, doc, identity, settled)?;
    diagnostics.extend(note_crossrefs(&structure_graph, &decided));
    in_reading_order(pages, slots, &fences);
    Ok(StructureOutput { authority_text_units, diagnostics, structure_graph, heading_sources })
}
