//! The structure stage: a PDF's extracted pages and lines in, its paragraphs, notes, headings and
//! structure graph out. It runs in three phases, each a function below; every step names the
//! module that decides it.
//!
//! **Readers** (`prepare_reader`, cached by the input pages): what each line is, read once.
//! 1. A layout model's labels in the stage's vocabulary (`layout_labelled` lines).
//! 2. A transcript's line numbers (`graph::mark_line_numbers`), markers printed apart joined to
//!    their lines (`crate::layout::join_detached_markers`).
//! 3. Each document of a packet (`constituents`) classified page by page (`classify`): furniture,
//!    parallel languages, tables (`legal_pdf_tables`), contents lists (`contents_lists`), the note
//!    band (`notes::band`), reading order, captions (`captions`); paragraphs segmented
//!    (`crate::paragraphs`).
//! 4. The note band corrected over the whole document (`notes::lanes`).
//! 5. Quotations (`quotations`), a quoted decision's numbered paragraphs (`numbered_paragraphs`),
//!    each line's reading context.
//! 6. Headings (`headings`: the page readers, the document's typography, its bookmarks and
//!    contents list).
//!
//! **Composed decisions** (`prepare_composed`): the whole-document heading rules and levels
//! (`headings::rules`), printed page labels (`furniture`), regions from the lines' roles.
//!
//! **Derivation** (`derive_prepared`): note modes (`notes`), label dropouts repaired (the engine's
//! `numbering_repair`), the text index and structure candidates (`graph`: citations, numbering
//! runs, witnesses), notes paired (`note_pairing`, the engine's pairer), paragraphs joined
//! (`joining`), propositions and cross-references (`note_context`), and the structure graph
//! resolved and its outline assembled (`graph`, `legal_pdf_witnesses`).
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
mod quotations;
mod numbered_paragraphs;
mod text;
mod classify;
use classify::*;
mod joining;
use joining::*;
mod note_context;
use note_context::*;
mod diagnostics;
pub use diagnostics::*;

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
const MAX_FONT_HEADING_CHARS: usize = 180;

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub(super) type DocumentContext = (usize, bool);

/// What the document's containers claim for the whole document, so that no heading, note or
/// paragraph reading takes a line back: its contents lists and its tables.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Claims {
    contents_pages: HashSet<usize>,
    contents_line_ids: HashSet<String>,
    /// The entries of the document's contents lists, in order (contents_lists).
    #[serde(default)]
    contents_entries: Vec<contents_lists::Entry>,
    table_cell_line_ids: HashSet<String>,
    /// Every table of the document, ruled, headed by rules, aligned by white space or read from
    /// a page's figures: one kind of table, whose lines are its cells.
    tables: Vec<legal_pdf_tables::PdfTable>,
}

impl Claims {
    /// A line a container has claimed (a table's cell, a contents entry).
    fn claimed(&self, id: &str) -> bool {
        self.contents_line_ids.contains(id) || self.table_cell_line_ids.contains(id)
    }
}

/// Which document of a packet, and which language of a parallel text, each line reads in.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct Contexts {
    /// The French column of a parallel English and French text.
    translation_line_ids: HashSet<String>,
    constituent_starts: Vec<constituents::Start>,
    document_line_ids: HashMap<String, usize>,
    #[serde(default)]
    reading_line_ids: HashMap<String, usize>,
}

impl Contexts {
    fn document_context(&self, id: &str) -> DocumentContext {
        (
            self.document_line_ids.get(id).copied().unwrap_or(0),
            self.translation_line_ids.contains(id),
        )
    }

    fn reading_lines(&self) -> &HashMap<String, usize> {
        if self.reading_line_ids.is_empty() {
            &self.document_line_ids
        } else {
            &self.reading_line_ids
        }
    }

    fn reading_context(&self, id: &str) -> DocumentContext {
        (
            self.reading_lines().get(id).copied().unwrap_or(0),
            self.translation_line_ids.contains(id),
        )
    }

    fn note_context(&self, note: &Footnote) -> DocumentContext {
        note.body_line_ids
            .first()
            .map_or((0, false), |id| self.document_context(id))
    }
}

/// The headings' levels, and the parents the document's bookmarks or contents list give them.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct HeadingReading {
    heading_levels: HashMap<String, usize>,
    #[serde(default)]
    navigation_parents: HashMap<String, Option<String>>,
    /// Each line's heading trail: every pass or rule that made it a heading ("+styled") or took
    /// that away ("-demote_body_type"), in order (`headings::mark`, `headings::stamp`).
    #[serde(default)]
    trail: HashMap<String, String>,
}

/// What the readers read of a document, each group written by one reader and read by the ones
/// after it: the source's own regions, the containers' claims, the reading contexts, the headings,
/// the quotations, and the strokes of a recognized page's bold.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
struct PdfPrimitiveEvidence {
    source_regions: Option<HashMap<String, String>>,
    claims: Claims,
    contexts: Contexts,
    headings: HeadingReading,
    #[serde(default)]
    quotations: quotations::Reading,
    /// The lines of paragraphs whose bracketed number is another text's (a quoted decision's
    /// paragraphs): quotations, and no numbered-paragraph candidates.
    #[serde(default)]
    foreign_numbered_line_ids: HashSet<String>,
    /// The page labels the PDF itself declares (its `/PageLabels`), by page index.
    #[serde(default)]
    embedded_page_labels: Vec<Option<String>>,
}
#[derive(Debug)]
struct PdfPreparation {
    diagnostics: Vec<Diagnostic>,
    primitives: PdfPrimitiveEvidence,
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
    /// Each line some heading pass or rule touched, and its heading trail ("+styled-demote_label").
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

/// The pages as the readers leave them, before any composed decision: what a document's
/// compositions never change, cached by its input pages and the readers' code.
#[derive(serde::Serialize, serde::Deserialize)]
struct ReaderState {
    pages: Vec<Page>,
    primitives: PdfPrimitiveEvidence,
    diagnostics: Vec<Diagnostic>,
}

/// Everything before the composed decisions: furniture, classification, tables, note bands and
/// the styled heading reader.
fn prepare_reader(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    page_labels: &[Option<String>],
) -> Result<(PdfPrimitiveEvidence, Vec<Diagnostic>)> {
    if separators.len() != pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    // A page layout model's labels (docling: text, paragraph_title, doc_title, footnote, header,
    // footer, table ...) in the stage's region vocabulary: its text is body, its titles headings.
    for page in pages.iter_mut() {
        for line in page.lines.iter_mut().filter(|line| layout_labelled(line)) {
            line.region_type = match line.region_type.as_str() {
                "text" => "body".to_owned(),
                "paragraph_title" | "doc_title" => "heading".to_owned(),
                other => other.to_owned(),
            };
        }
    }
    // A transcript's line numbers are claimed first, for the whole document: they locate its
    // lines and are no part of them. A numbering marker printed on its own line joins the line
    // that carries its text, on a page without line numbers.
    legal_pdf_support::profile::measure("prepare.detached_markers", || {
        graph::mark_line_numbers(pages);
        for page in pages.iter_mut() {
            if graph::line_labels(page).next().is_none() {
                join_detached_markers(page);
            }
        }
    });
    let mut primitives = PdfPrimitiveEvidence {
        embedded_page_labels: page_labels.to_vec(),
        contexts: Contexts { constituent_starts: constituents::bookmarked_starts(pages, outline), ..Contexts::default() },
        source_regions: legal_pdf_support::profile::measure("prepare.source_regions", || {
            source_region_contract(pages)
        }),
        ..PdfPrimitiveEvidence::default()
    };
    let diagnostics = legal_pdf_support::profile::measure("prepare.classify", || {
        classify_pages_with_source(pages, separators, &mut primitives)
    });
    // Each document and language keeps its own note type, ladder and continuation history.
    legal_pdf_support::profile::measure("prepare.note_regions", || {
        let separators_by_page = pages
            .iter()
            .zip(separators)
            .map(|(page, separator)| (page.index, *separator))
            .collect::<HashMap<_, _>>();
        let mut changed = 0;
        constituents::within_document_contexts(
            pages,
            &primitives.contexts.document_line_ids,
            &primitives.contexts.translation_line_ids,
            |document| {
                let separators = document
                    .iter()
                    .map(|page| separators_by_page[&page.index])
                    .collect::<Vec<_>>();
                // The generic note reader honors source exclusions: the containers' claimed lines,
                // and a contents page's body, are held out of it for this view.
                let mut excluded = Vec::new();
                for (page_slot, page) in document.iter_mut().enumerate() {
                    let contents = primitives.claims.contents_pages.contains(&page.index);
                    for (slot, line) in page.lines.iter_mut().enumerate() {
                        if !line.exclude_from_body
                            && (primitives.claims.claimed(&line.id) || contents && line.region_type == "body")
                        {
                            line.exclude_from_body = true;
                            excluded.push((page_slot, slot));
                        }
                    }
                }
                changed += notes::lanes::correct_note_regions(document, &separators);
                for (page, line) in excluded {
                    document[page].lines[line].exclude_from_body = false;
                }
            },
        );
        changed
    });
    // LEGALPDF_ABLATE=quotations leaves the quotation reader out, for its measure under the tagger.
    primitives.quotations = legal_pdf_support::profile::measure("reader.quotations", || {
        if legal_structure_model::ablated("quotations") { quotations::Reading::default() } else { quotations::read(pages, &primitives) }
    });
    for group in legal_pdf_support::profile::measure("reader.sequence", || numbered_paragraphs::foreign_numbered_paragraphs(pages, &primitives)) {
        if std::env::var_os("LEGALPDF_SEQUENCE_TRACE").is_some() {
            eprintln!("sequence\tquoted paragraph\t{}\t{} lines", group[0], group.len());
        }
        let id = format!("quotation:{}", group[0]);
        for line in &group {
            primitives.quotations.scopes.entry(line.clone()).or_insert_with(|| id.clone());
        }
        primitives.foreign_numbered_line_ids.extend(group);
    }
    primitives.contexts.reading_line_ids = legal_pdf_support::profile::measure("reader.reading_lines", || quotations::reading_lines(pages, &primitives));
    read_headings(pages, outline, &mut primitives);
    Ok((primitives, diagnostics))
}

/// The decisions that rest on the readers' pages: the headings' one decision (an instrument's numbered
/// captions among its vetoes), then the printed page labels and the regions.
fn prepare_composed(
    pages: &mut [Page],
    mut primitives: PdfPrimitiveEvidence,
    mut diagnostics: Vec<Diagnostic>,
    instrument: bool,
) -> PdfPreparation {
    legal_pdf_support::profile::measure("compose.heading_rules", || headings::rules::read(pages, &mut primitives, instrument));
    stamp(pages, &mut primitives.headings.trail, "rules");
    diagnostics.extend(legal_pdf_support::profile::measure(
        "prepare.printed_labels",
        || assign_printed_page_labels(pages),
    ));
    // Regions are a view of the lines' final roles, derived once every reader has run.
    build_regions(pages);
    PdfPreparation {
        diagnostics,
        primitives,
    }
}

fn derive_prepared(
    pages: &mut [Page],
    mut prepared: PdfPreparation,
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    legal_pdf_support::profile::measure("derive.note_regions", || {
        constituents::within_document_contexts(
            pages,
            &prepared.primitives.contexts.document_line_ids,
            &prepared.primitives.contexts.translation_line_ids,
            infer_note_region_modes,
        );
    });
    prepared
        .diagnostics
        .extend(legal_pdf_support::profile::measure(
            "derive.text_flow",
            || text_flow_faults(pages),
        ));
    // Note labels recognition cut short, restored before any text is indexed.
    constituents::within_document_contexts(
        pages,
        &prepared.primitives.contexts.document_line_ids,
        &prepared.primitives.contexts.translation_line_ids,
        |document| legal_pdf_core::layout_analysis().repair_label_dropouts(document),
    );
    let resolution = legal_pdf_support::profile::measure("derive.structure_candidates", || {
        PdfResolutionInput::from_pages(pages, &prepared.primitives)
    });
    let pairing = legal_pdf_support::profile::measure("derive.pair_footnotes", || {
        note_pairing::pair_notes(
            pages,
            &prepared.primitives.contexts,
            &resolution.citation_spans,
            &resolution.index,
        )
    });
    let (note_pairs, graph_diagnostics) = map_note_pairs(&resolution.index, &pairing.pair_claims)?;
    let anchors = pairing.anchors;
    let mut footnotes = pairing.footnotes;
    let mut diagnostics = prepared.diagnostics;
    diagnostics.extend(pairing.diagnostics);
    diagnostics.extend(legal_pdf_support::profile::measure(
        "derive.unmatched_references",
        || unmatched_reference_diagnostics(pages, &footnotes, &anchors, &prepared.primitives),
    ));
    let paragraphs = legal_pdf_support::profile::measure("derive.paragraphs", || {
        build_paragraphs_with_context(pages, &anchors, &prepared.primitives)
    });
    legal_pdf_support::profile::measure("derive.propositions", || {
        attach_propositions(&mut footnotes, &paragraphs, &prepared.primitives)
    });
    legal_pdf_support::profile::measure("derive.crossrefs", || {
        attach_crossrefs(&mut footnotes, &mut diagnostics, &prepared.primitives)
    });
    let mut nodes =
        native_graph_parts(&resolution.index, pages, &paragraphs, &prepared.primitives)?;
    nodes.extend(constituents::constituents(
        &resolution.index,
        pages,
        &prepared.primitives.contexts.constituent_starts,
    ));
    let source_fragments = resolution.index.source_fragments();
    let pages: &[Page] = pages;
    let document_id = identity.document_id.clone();
    let mut structure_graph =
        legal_pdf_support::profile::measure("derive.structure_graph", || {
            legal_structure_model::structure_analysis().resolve_structure_graph_with_assembler(
                identity.document_id,
                "local-pdf".to_owned(),
                resolution.index.text(),
                Some(identity.source_sha256),
                nodes,
                &resolution.runs,
                &resolution.evidence,
                &note_pairs,
                graph_diagnostics,
                &mut |nodes| {
                    let contexts = graph::outline_contexts(
                        nodes,
                        &document_id,
                        &prepared.primitives.quotations,
                        &resolution.index,
                        &prepared.primitives,
                    );
                    let line_range = |id: &str| resolution.index.line(id).map(|line| line.range);
                    let mut parents = prepared.primitives.headings.navigation_parents.clone();
                    parents.extend(
                        prepared.primitives.quotations.introductions.iter().map(
                            |(quote, introduction)| (quote.clone(), Some(introduction.clone())),
                        ),
                    );
                    legal_pdf_witnesses::assemble_outline(
                        nodes,
                        &contexts,
                        resolution.index.text(),
                        pages,
                        &line_range,
                        &parents,
                        &prepared.primitives.headings.heading_levels,
                    );
                },
            )
        })
        .map_err(|error| Error::Message(error.to_string()))?;
    structure_graph.profile = identity.profile;
    structure_graph.source_fragments = source_fragments;
    let mut heading_sources = std::mem::take(&mut prepared.primitives.headings.trail);
    // A step after the heading passes that made a heading of a line.
    for line in pages.iter().flat_map(|page| &page.lines).filter(|line| line.region_type == "heading") {
        let trail = heading_sources.entry(line.id.clone()).or_default();
        if !headings::heads(trail) {
            trail.push_str("+later");
        }
    }
    Ok(StructureOutput {
        paragraphs,
        footnotes,
        diagnostics,
        structure_graph,
        heading_sources,
    })
}

pub fn derive(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
) -> Result<StructureOutput> {
    derive_cached(pages, separators, outline, identity, None)
}

/// `derive`, reading the readers' state from `reader_cache` where it is there (a file named for the
/// input pages and reader dependencies) and writing it there where not. Composed decisions run
/// after restoring the cached state.
pub fn derive_cached(
    pages: &mut [Page],
    separators: &[Option<f64>],
    outline: &[PdfOutlineEntry],
    identity: StructureIdentity,
    reader_cache: Option<&std::path::Path>,
) -> Result<StructureOutput> {
    let cached = reader_cache
        .and_then(|path| legal_pdf_core::read_gzip_json::<ReaderState>(path).ok())
        .filter(|state| state.pages.len() == pages.len());
    let (primitives, diagnostics) = match cached {
        Some(state) => {
            pages.clone_from_slice(&state.pages);
            (state.primitives, state.diagnostics)
        }
        None => {
            let (primitives, diagnostics) = legal_pdf_support::profile::measure("prepare.reader", || {
                prepare_reader(pages, separators, outline, &identity.page_labels)
            })?;
            if let Some(path) = reader_cache {
                let state = ReaderState { pages: pages.to_vec(), primitives, diagnostics };
                let _ = legal_pdf_core::write_gzip_json(path, &state);
                (state.primitives, state.diagnostics)
            } else {
                (primitives, diagnostics)
            }
        }
    };
    // The document's type, where known (the caller's profile): an instrument numbers its provisions.
    let instrument = identity.profile == Some(legal_structure_model::DetectionProfile::Instrument);
    let prepared = prepare_composed(pages, primitives, diagnostics, instrument);
    derive_prepared(pages, prepared, identity)
}
