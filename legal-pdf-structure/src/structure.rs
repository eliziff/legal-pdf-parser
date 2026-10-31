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
//! 5. Decided (`Decision`): every line's role and the unit each marker opens, the lines no rule
//!    settled weighed for headings, and the headings' levels.
//! 6. Written once (`write`): each line's region, note mode, block and place in its page's reading
//!    order, each page's printed label.
//! 7. Derived from what was written: regions (`crate::layout::build_regions`), notes paired
//!    (`note_pairing`, the engine's pairer), paragraphs joined (`joining`), propositions and
//!    cross-references (`note_context`), and the structure graph resolved on the decision's roles
//!    (`graph`) with its outline assembled (`graph::assemble_outline`).
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
mod headings;
mod notes;
use notes::*;
mod note_pairing;
mod numbered_paragraphs;
mod text;
mod joining;
use joining::*;
mod note_context;
use note_context::*;
mod diagnostics;
pub use diagnostics::*;
mod decide;
pub(crate) use decide::*;

use crate::layout::*;
use furniture::*;
use headings::*;
use text::*;
pub(crate) use legal_structure_model::text::{
    is_note_symbol, label_prefix, line_start_label_prefix, normalize_label, LabelPrefix, MAX_SYMBOL_LABEL_LEN,
};
pub use graph::PdfTextIndex;
use graph::{map_note_pairs, native_graph_parts, PdfResolutionInput};
use legal_pdf_core::model::{
    Diagnostic, Footnote, FootnoteCrossref, Line, Page, Paragraph,
    ParagraphAnchor, PdfOutlineEntry, Span,
};
use legal_pdf_support::LegalDocument;
use legal_pdf_core::{flag_share, layout_analysis, line_font_size, median, off_grid_sizes, quantile, Anchor, Error, Result, BOLD};
use legal_structure_model::{
    enumerator_interpretations, parse_heading_runs, utf16_len, DocumentStructure, EnumeratorInterpretation, NodeKind, ResolutionRuleV2, ScalarRange, ScalarText,
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
    decision: &'a Decision,
    /// Each quotation's id ("quotation:" and its first line's id), by its number less one.
    quotations: Vec<String>,
}

impl<'a> Decided<'a> {
    /// The decision on the pages as `write` finds them, before their lines move into reading order.
    fn of(pages: &[Page], slots: &'a Slots, fences: &'a Fences, decision: &'a Decision) -> Self {
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
        Self { slots, fences, decision, quotations }
    }

    fn slot(&self, id: &str) -> Option<Slot> {
        self.slots.of_id(id)
    }

    /// The role the decision gave the slot's line.
    pub(super) fn role(&self, slot: Slot) -> Role {
        self.decision.role[slot.index()]
    }

    /// The rule that gave the slot's line its role.
    pub(super) fn rule(&self, slot: Slot) -> Rule {
        self.decision.rule[slot.index()]
    }

    /// The reading of the slot's opening marker that may open its unit: its run and member.
    pub(super) fn reading(&self, slot: Slot) -> Option<(u32, u32)> {
        self.decision.reading[slot.index()]
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

    /// A contents list's line, or a line of a page wholly a contents list's.
    pub(super) fn listed(&self, id: &str) -> bool {
        self.slot(id).is_some_and(|slot| {
            self.decision.role[slot.index()] == Role::Contents || self.fences.contents_pages[self.slots.page(slot)]
        })
    }

    /// The id of the quotation holding the line.
    pub(super) fn quotation(&self, id: &str) -> Option<&str> {
        let quotation = usize::from(self.fences.quotation[self.slot(id)?.index()]);
        (quotation > 0).then(|| self.quotations[quotation - 1].as_str())
    }

    /// The heading's level, where the line is a heading the decision levelled.
    pub(super) fn heading_level(&self, id: &str) -> Option<usize> {
        let slot = self.slot(id)?.index();
        (self.decision.role[slot] == Role::Heading && self.decision.level[slot] > 0).then(|| usize::from(self.decision.level[slot]))
    }

    /// The document's tables.
    pub(super) fn tables(&self) -> &'a [legal_pdf_tables::PdfTable] {
        &self.fences.tables
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

pub fn derive(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    derive_cached(pages, separators, outline, identity, None)
}

/// `derive`, reading what the measuring, fencing and numbering read from `cache` where it is there
/// (a file named for the input pages and the code that reads them) and writing it there where not.
/// The notes, the decision and everything after it are read on every run.
pub fn derive_cached(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    cache: Option<&std::path::Path>,
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
    let Measured { evidence, mut fences, runs } = match cached {
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
                Evidence::measure(pages, &slots, normalized, separators, outline, &identity.page_labels, instrument)
            });
            let mut fences = measure("derive.fences", || Fences::read(pages, &slots, &evidence, outline));
            let runs = measure("derive.runs", || Runs::read(pages, &slots, &mut evidence, &mut fences))?;
            let measured = Measured { evidence, fences, runs };
            if let Some(path) = cache {
                let _ = legal_pdf_core::write_gzip_json(path, &measured);
            }
            measured
        }
    };
    let mut notes = measure("derive.notes", || Notes::read(pages, &slots, &evidence, &fences, &runs));
    let mut decision = measure("derive.decide", || Decision::decide(pages, &slots, &evidence, &fences, &runs, &notes, outline));
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    diagnostics.append(&mut notes.diagnostics);
    diagnostics.append(&mut decision.diagnostics);
    let decided = Decided::of(pages, &slots, &fences, &decision);
    // Traced before the decision is written: writing puts each page's lines in reading order.
    if let Some(folder) = trace_folder() {
        write_trace(folder, &identity.source_sha256, pages, &slots, &evidence, &fences, &runs, &notes, &decision);
    }
    let heading_sources = measure("derive.write", || write(pages, &slots, &evidence, &fences, &notes, &decision));
    // Regions are a view of the lines' roles, derived once they are written.
    build_regions(pages);
    diagnostics.extend(measure("derive.text_flow", || text_flow_faults(pages)));
    let resolution = measure("derive.structure_candidates", || {
        PdfResolutionInput::from_decision(pages, &slots, &evidence, &runs, &decided)
    });
    let pairing = measure("derive.pair_footnotes", || {
        note_pairing::pair_notes(pages, &decided, &resolution.citation_spans, &resolution.index)
    });
    let (note_pairs, graph_diagnostics) = map_note_pairs(&resolution.index, &pairing.pair_claims)?;
    let anchors = pairing.anchors;
    let mut footnotes = pairing.footnotes;
    diagnostics.extend(pairing.diagnostics);
    // A line written in the note band that no note holds (the pairer made no note of its label, or
    // bounded the note's body short of it) is the body's, and joins its paragraph rather than
    // standing alone: on the gold, 6,825 such lines were no notes, 2,146 were.
    let given_back = {
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
    };
    if given_back {
        build_regions(pages);
    }
    diagnostics.extend(measure("derive.unmatched_references", || {
        unmatched_reference_diagnostics(pages, &footnotes, &anchors, &decided)
    }));
    let paragraphs = measure("derive.paragraphs", || build_paragraphs_with_context(pages, &anchors, &decided));
    measure("derive.propositions", || attach_propositions(&mut footnotes, &paragraphs, &decided));
    measure("derive.crossrefs", || attach_crossrefs(&mut footnotes, &mut diagnostics, &decided));
    let mut nodes = native_graph_parts(&resolution.index, pages, &paragraphs, &decided)?;
    nodes.extend(constituents::constituents(&resolution.index, pages, &fences.starts));
    let source_fragments = resolution.index.source_fragments();
    let document_id = identity.document_id.clone();
    let mut structure_graph = measure("derive.structure_graph", || {
        legal_structure_model::structure_analysis().resolve_structure_graph_with_assembler(
            identity.document_id,
            "local-pdf".to_owned(),
            resolution.index.text(),
            Some(identity.source_sha256),
            nodes,
            &resolution.runs,
            &resolution.evidence,
            &resolution.roles,
            &note_pairs,
            graph_diagnostics,
            &mut |nodes| {
                let contexts = graph::outline_contexts(nodes, &document_id, &resolution.index, &decided);
                graph::assemble_outline(nodes, &contexts, resolution.index.text(), &|id| decided.heading_level(id));
            },
        )
    })
    .map_err(|error| Error::Message(error.to_string()))?;
    structure_graph.profile = identity.profile;
    structure_graph.source_fragments = source_fragments;
    Ok(StructureOutput {
        paragraphs,
        footnotes,
        diagnostics,
        structure_graph,
        heading_sources,
    })
}
