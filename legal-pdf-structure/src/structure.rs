//! The structure stage: a PDF's extracted pages and lines in, its paragraphs, notes, headings and
//! structure graph out. Every line's role is decided once (`decide`), in stages:
//!
//! 0. The pages normalized (`normalize`): a transcript's line numbers parted from the lines they
//!    were read into, a table's rulings cleared from under its words, each line given its slot.
//! 1. Measured (`Evidence`): each page's words, rows and note separator; each line's type, its
//!    source's tag, its marker, note label, reference tokens, words and citations; the document's
//!    bookmarks and whether it is an instrument.
//! 2. Fenced and ordered (`Fences`): a transcript's line numbers, furniture and folios
//!    (`furniture`), each page's columns, contents lists (`contents_lists`), tables
//!    (`legal_pdf_tables`), the note band under a separator, the body's type, the constituent
//!    documents (`constituents`) and their own furniture, the transcripts, the reading order,
//!    quotations and reading contexts, and each line's continuation of the one before.
//! 3. Numbered (`Runs`): each reading context's numbering runs, read and resolved by the engine.
//!
//!    What these stages read is cached by the input pages (`derive_cached`).
//! 4. Notes (`Notes`): each document's note band (`notes::band`) and the lanes that take more lines
//!    for notes against it (`notes::lanes`).
//! 5. Decided (`transition::tier0`): every rule heard on every line (`Decision::hear`), one decision on each line in reading
//!    order against what is open (`settle`), the headings' levels (`Decision::finish`). Paragraph numbers are fixed input.
//! 6. Written once (`write_lines`): each line's region, note mode, block and place in its page's reading order, each page's
//!    printed label.
//! 7. Derived from what was written: regions (`crate::layout::build_regions`), notes paired (`note_pairing`, the engine's
//!    pairer), propositions and cross-references (`note_context`), and the moves the decided lines make (`moves_of`) emitted as
//!    the structure graph (`transition::emit`) with its outline assembled (`graph::assemble_outline`).
//!
//! Measure a change on the structure gold with `experiments/legal-structure-gold/grade/prove.ps1`
//! (with `STRUCTURE_GOLD_ORACLE_LAYOUT=1` as an arm, as a perfect layout model reads), the PDF
//! footnote benchmark with `experiments/footnote-gold/score.py`, and provider text with the A2AJ
//! census (`a2aj_corpus`).

mod captions;
mod constituents;
mod contents_lists;
mod furniture;
mod graph;
mod heading_ladder;
mod headings;
mod notes;
use notes::*;
mod note_pairing;
mod text;
mod joining;
use joining::*;
mod note_context;
use note_context::*;
mod diagnostics;
pub use diagnostics::*;
mod decide;
pub(crate) use decide::*;
/// The machine: tier 0 read as one decision on each line, and the transition parser's fitted readers (DOC with `encoder`).
mod transition;
/// The origin of the transition parser's nodes.
const TRANSITION_ORIGIN: &str = "legal-pdf-transition.v1";

use crate::layout::*;
use furniture::*;
use headings::*;
use text::*;
pub(crate) use legal_structure_model::text::{
    is_note_symbol, label_prefix, line_start_label_prefix, normalize_label, LabelPrefix, MAX_SYMBOL_LABEL_LEN,
};
pub use graph::PdfTextIndex;

/// The transition parser's whole reading trained on whole documents (`transition::train_whole`).
pub fn train_whole_reading(problems: &std::path::Path, out: &std::path::Path, epochs: usize) -> std::result::Result<String, String> {
    transition::train_whole(problems, out, epochs)
}
use graph::map_note_pairs;
use legal_pdf_core::model::{
    Diagnostic, Footnote, FootnoteCrossref, Line, Page, Paragraph,
    ParagraphAnchor, PdfOutlineEntry, Span,
};
use legal_pdf_support::LegalDocument;
use legal_pdf_core::{flag_share, layout_analysis, line_font_size, median, off_grid_sizes, quantile, Anchor, Error, Result, BOLD};
use heading_ladder::{enumerator_interpretations, parse_heading_runs, EnumeratorInterpretation};
use legal_structure_model::{
    utf16_len, DocumentStructure, NodeKind, ResolutionRuleV2, ScalarRange, ScalarText, StructureAnalysis,
};
use regex::Regex;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;
use pdf_inspector::markdown::has_dot_leaders as has_dot_leader;

const HARD_DIAGNOSTICS: &[&str] = &[
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
];

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub(super) type DocumentContext = (usize, bool);

/// The decision as the derivation reads it, by line id: each line's document and language, its
/// reading context, role, quotation and heading level; and the document's contents pages and
/// tables.
pub(super) struct Decided<'a> {
    slots: &'a Slots,
    fences: &'a Fences,
    /// Each quotation's id ("quotation:" and its first line's id), by its number less one.
    quotations: Vec<String>,
}

impl<'a> Decided<'a> {
    /// The decision on the pages as `write` finds them, before their lines move into reading order.
    fn of(pages: &[Page], slots: &'a Slots, fences: &'a Fences) -> Self {
        let count = fences.quotation.iter().copied().max().unwrap_or(0);
        let mut first: Vec<Option<Slot>> = vec![None; usize::from(count)];
        for &slot in &fences.order {
            let quotation = usize::from(fences.quotation[slot.index()]);
            if quotation > 0 && first[quotation - 1].is_none() {
                first[quotation - 1] = Some(slot);
            }
        }
        let quotations = first
            .into_iter()
            .map(|slot| slot.map_or_else(String::new, |slot| format!("quotation:{}", slots.line(pages, slot).id)))
            .collect();
        Self { slots, fences, quotations }
    }

    fn slot(&self, id: &str) -> Option<Slot> {
        self.slots.of_id(id)
    }

    /// The line's constituent document and language.
    pub(super) fn document_context(&self, id: &str) -> DocumentContext {
        self.slot(id).map_or((0, false), |slot| {
            (usize::from(self.fences.document[slot.index()]), self.fences.translation[slot.index()])
        })
    }

    /// The line's reading context (its document, language and quotation) and language.
    pub(super) fn reading_context(&self, id: &str) -> DocumentContext {
        self.slot(id).map_or((0, false), |slot| {
            (usize::from(self.fences.ctx[slot.index()]), self.fences.translation[slot.index()])
        })
    }

    /// The document context of a note's first line.
    pub(super) fn note_context(&self, note: &Footnote) -> DocumentContext {
        note.body_line_ids
            .first()
            .map_or((0, false), |id| self.document_context(id))
    }

    /// The line is in the French column of a parallel English and French text.
    pub(super) fn translated(&self, id: &str) -> bool {
        self.slot(id).is_some_and(|slot| self.fences.translation[slot.index()])
    }

    /// The numbered paragraphs the line's text holds: each one's language, number and UTF-16 span of the line.
    pub(super) fn held(&self, id: &str) -> &[Held] {
        self.slot(id).map_or(&[], |slot| self.fences.paragraphs.held[slot.index()].as_slice())
    }

    /// The id of the quotation holding the line.
    pub(super) fn quotation(&self, id: &str) -> Option<&str> {
        let quotation = usize::from(self.fences.quotation[self.slot(id)?.index()]);
        (quotation > 0).then(|| self.quotations[quotation - 1].as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructureIdentity {
    pub profile: Option<legal_structure_model::DetectionProfile>,
    pub document_id: String,
    pub source_sha256: String,
    /// The page labels the PDF declares (`/PageLabels`), by page index; empty where it has none.
    pub page_labels: Vec<Option<String>>,
}

pub struct StructureOutput {
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub structure_graph: DocumentStructure,
    /// Each heading line, and each line taken from the headings, with the rule that decided it
    /// ("+anchor", "-marginal_note").
    pub heading_sources: HashMap<String, String>,
}

pub(super) fn scalar_suffix(value: &str, start: usize) -> &str {
    if value.is_ascii() {
        return &value[start.min(value.len())..];
    }
    let end = value.len();
    let byte = value.char_indices().nth(start).map_or(end, |at| at.0);
    &value[byte..]
}

/// Whether a page layout model labelled the line: its region is one of the model's.
pub(super) fn layout_labelled(line: &Line) -> bool {
    line.region_id.contains("-ppdoc-r")
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
fn give_back(pages: &mut [Page], footnotes: &[Footnote]) -> bool {
    let held: HashSet<&str> = footnotes.iter().flat_map(|note| &note.body_line_ids).map(String::as_str).collect();
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

pub fn derive(
    analysis: &dyn StructureAnalysis,
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    derive_cached(analysis, pages, separators, outline, identity, None, None)
}

/// `derive`, reading what the measuring, fencing and numbering read from `cache` where it is there
/// (a file named for the input pages and the code that reads them) and writing it there where not.
/// The notes, the decision and everything after it are read on every run. With `trace`, the decision is also written
/// to that folder (`decide::trace`); the parse is the same.
pub fn derive_cached(
    analysis: &dyn StructureAnalysis,
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    cache: Option<&std::path::Path>,
    trace: Option<&std::path::Path>,
) -> Result<StructureOutput> {
    use legal_pdf_support::profile::measure;
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
                Evidence::measure(analysis, pages, &slots, normalized, separators, outline, &identity.page_labels, instrument)
            });
            let mut fences = measure("derive.fences", || Fences::read(analysis, pages, &slots, &evidence, outline));
            let runs = measure("derive.runs", || Runs::read(analysis, pages, &slots, &mut evidence, &mut fences))?;
            let measured = Measured { evidence, fences, runs };
            if let Some(path) = cache {
                let _ = legal_pdf_core::write_gzip_json(path, &measured);
            }
            measured
        }
    };
    // The transition parser: its reading the output where LEGALPDF_TRANSITION_EMIT says so, else read
    // beside the stages while it is fitted and measured.
    if transition::emitting() {
        return measure("derive.transition", || transition::output(pages, &slots, &evidence, &fences, &runs, outline, &identity));
    }
    if transition::enabled() {
        measure("derive.transition", || transition::run(pages, &slots, &evidence, &fences, &runs, outline, &identity.source_sha256));
    }
    measure("derive.machine", || transition::tier0::output(analysis, pages, &slots, &evidence, fences, &runs, outline, &identity, trace))
}
