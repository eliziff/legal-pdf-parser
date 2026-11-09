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
//!    What these stages read is cached by the input pages (`derive`).
//! 4. Notes (`Notes`): each document's note band and the lanes that take more lines for notes (`legal_pdf_notes`).
//! 5. Decided (`transition::tier0`): every rule heard on every line (`Decision::hear`), one decision on each line in
//!    reading order (`settle`), the headings' levels (`Decision::finish`). Paragraph numbers are fixed input.
//! 6. Written once (`write_lines`): each line's place in its page's reading order; and read by slot, each line's zone
//!    and block (`zones`, `blocks`).
//! 7. Derived from the decision: notes paired (the engine's pairer, over `note_lines`), a note band's line no note holds
//!    given back to the body, propositions and cross-references (`note_context`), and the moves the decided lines make
//!    (`tier0::moves_of`, over `tier0::reading_regions`) emitted as the structure graph (`legal_pdf_emit`).
//!
//! Measure a change with the grader's digest (`structure-gold digest`): every parse hashed, two builds compared.

pub use legal_pdf_structure_model::StructureIdentity;
use legal_pdf_decide as decide;
use legal_pdf_evidence::{normalize, Evidence};
use legal_pdf_fences::{decided, Fences};
use legal_pdf_structure_model::{Runs, Slots};
pub(crate) use decide::*;
use legal_pdf_graph::graph;
pub use graph::PdfTextIndex;
use legal_pdf_notes::note_context;
use note_context::*;
/// The transition parser's walk over tier 0's reading, and its fitted readers (DOC with `encoder`).
use legal_pdf_transition as transition;
use legal_pdf_tier0 as tier0;
use legal_pdf_emit::Settled;
use legal_pdf_machine::lines::LineTable;
use legal_pdf_machine::machine::LineView;
use std::path::Path;

use legal_pdf_core::model::{Diagnostic, LineZone, Page, PdfOutlineEntry};
use legal_structure_model::StructureNode;
use legal_pdf_core::{Error, Result};
use legal_structure_model::DocumentStructure;
use std::collections::HashSet;

pub struct StructureOutput {
    /// The authority units, read off the rendered text (`emit`).
    pub authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
    pub diagnostics: Vec<Diagnostic>,
    pub structure_graph: DocumentStructure,
}

/// A line of the note band that no note holds (the pairer made no note of its label, or bounded the note's body short of
/// it) is the body's, and joins its paragraph rather than standing alone: on the gold, 6,825 such lines were no notes,
/// 2,146 were.
fn give_back(zones: &mut [LineZone], slots: &Slots, notes: &[StructureNode]) {
    let mut held = vec![false; zones.len()];
    for slot in notes.iter().flat_map(|note| &note.line_ids).filter_map(|id| slots.of_id(id)) {
        held[slot.index()] = true;
    }
    for (zone, held) in zones.iter_mut().zip(held) {
        if matches!(zone, LineZone::Note(_)) && !held {
            *zone = LineZone::Body;
        }
    }
}
/// The document's structure. With `trace`, the decision is also written to that folder (`decide::trace`); the parse is
/// the same.
pub fn derive(pages: &mut [Page],
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
    let (mut slots, normalized) = measure("derive.normalize", || normalize(pages));
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
                legal_pdf_evidence::measure(pages, &slots, normalized, separators, outline, instrument)
            });
            let mut fences = measure("derive.fences", || Fences::read(pages, &slots, &evidence, outline, &identity.page_labels));
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
    measure("derive.machine", || output(pages, &mut slots, &evidence, fences, &runs, &geometry, outline, &identity, trace))
}

/// The document's structure as the machine reads it, what `derive` gives where the request names `Reader::Machine`.
#[allow(clippy::too_many_arguments)]
/// Tier 0's reading of the document written: its decision (`tier0::decide`), the walk over it where DOC or the scorer's
/// fitting asks (`transition::over_tier0`), the lines written, the notes paired, the moves made (`tier0::moves_of`, DOC's
/// in their place where it is given) and emitted as the structure graph (`legal_pdf_emit`).
fn output(pages: &mut [Page],
    slots: &mut Slots,
    evidence: &Evidence,
    mut fences: Fences,
    runs: &Runs,
    geometry: &decide::Geometry,
    outline: &[PdfOutlineEntry],
    identity: &StructureIdentity,
    trace: Option<&Path>,
) -> Result<StructureOutput> {
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    let fences = fences;
    // One line table for both tiers: tier 0 reads it, and the walk measures it for DOC.
    let table = LineTable::plain(pages, slots, evidence, &fences, &runs.before, transition::confusions());
    let (mut notes, decision) = tier0::decide(pages, slots, evidence, &fences, runs, geometry, outline, &table, trace.is_some());
    // The walk over this reading where it is asked for (the DOC tier's, the scorer's fitting), read off the pages as they
    // came, before this reading writes and orders them.
    let doc = transition::over_tier0(pages, slots, evidence, &fences, geometry, outline, &identity.source_sha256, &notes, &decision, &table);
    diagnostics.append(&mut notes.diagnostics);
    if let Some(folder) = trace {
        write_trace(folder, &identity.source_sha256, pages, slots, evidence, &fences, runs, &notes, &decision);
    }
    write_lines(pages, slots, &fences, &decision);
    let blocks = blocks(pages, slots, evidence, &fences, &decision);
    let mut zones = zones(slots, evidence, &fences, &notes, &decision);
    // The lines are read in reading order from here on, the pairer's and the output's: each page's put in it once.
    slots.read_in_order(pages, &fences.order);
    let slots: &Slots = slots;
    let decided = decided(pages, slots, &fences);
    let pages: &[Page] = pages;
    // A line joins the one before it where evidence reads that line's marker standing beside it (`Evidence::beside`).
    let index = PdfTextIndex::from_pages(pages, |before, line| slots.of_id(&before.id).and_then(|at| evidence.beside[at.index()]).is_some_and(|beside| slots.of_id(&line.id) == Some(beside)));
    let citations = graph::citation_spans(pages, slots, evidence);
    let lines = note_lines(pages, slots, &fences, &notes, &decision, &zones, &blocks, &citations);
    let pairing = legal_structure_model::structure_analysis().pair_footnotes(&lines, &|at, start, end| index.global_range(lines[at].id, start, end));
    let notes = pairing.notes;
    diagnostics.extend(pairing.diagnostics);
    give_back(&mut zones, slots, &notes);
    diagnostics.extend(unmatched_reference_diagnostics(fences.order.iter().map(|&slot| (slots.line(pages, slot), zones[slot.index()])), &notes, &index, &decided));
    let views = LineView::all(slots, evidence, &fences, &table);
    let moves = tier0::moves_of(slots, &fences, runs, &views, &decision, &tier0::reading_regions(pages, slots, &fences, &zones, &blocks), &notes);
    // The writer takes DOC's move for each line where DOC read one, but for the notes, the pairer's (a line either reads
    // as a note keeps this reading's move). DOC reads the same table, so its reading of a marker names one of this
    // reading's.
    let settled = Settled { index: &index, notes, runs };
    let (structure_graph, authority_text_units) = legal_pdf_emit::structure_graph(pages, slots, evidence, &fences, &decided, &table, &moves, doc, identity, settled)?;
    Ok(StructureOutput { authority_text_units, diagnostics, structure_graph })
}
