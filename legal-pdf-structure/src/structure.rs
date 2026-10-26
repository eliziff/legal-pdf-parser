//! Shared structure derivation for aligned page and line evidence.

mod bookmarks;
mod captions;
mod constituents;
mod contents;
mod contents_lists;
mod furniture;
mod graph;
mod headings;
mod heading_rules;
mod line_roles;
mod note_pairing;
mod note_regions;
mod quotations;
mod numbered_paragraphs;
mod text;

use crate::layout::*;
use furniture::*;
use headings::*;
use line_roles::*;
use text::*;
#[cfg(test)]
use graph::{contents_row, index_pages};
pub use graph::PdfTextIndex;
use graph::{
    contents_leader_re, map_note_pairs, native_graph_parts, PdfResolutionInput,
};
use legal_pdf_core::model::{
    DetachedReference, Diagnostic, Footnote, FootnoteCrossref, Line, Page, Paragraph,
    ParagraphAnchor, PdfOutlineEntry, Span, Word,
};
use legal_pdf_support::LegalDocument;
#[cfg(test)]
use legal_pdf_core::model::{NotePairClaim, NotePairKind};
use legal_pdf_core::{flag_share, layout_analysis, line_font_size, median, off_grid_sizes, quantile, Anchor, Error, Result, BOLD};
use legal_structure_model::{
    enumerator_interpretations, normalize_decimal_digit, normalize_note_symbol,
    parse_heading_ladder, utf16_len, DocumentStructure, EnumeratorInterpretation, HeadingAction,
    HeadingFamilyStats, HeadingLadderStatus, NodeKind, ResolutionRuleV2, ScalarRange, ScalarText,
};
#[cfg(test)]
use legal_structure_model::{CandidateGrammar, CandidateObservationV2};
use regex::Regex;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

const HARD_DIAGNOSTICS: &[&str] = &[
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
];
pub(super) const MAX_SYMBOL_LABEL_LEN: usize = 8;
const MAX_FONT_HEADING_CHARS: usize = 180;
const FONT_HEADING_SIZE_RATIO: f64 = 1.18;

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub(super) type DocumentContext = (usize, bool);

#[derive(Debug, Clone)]
pub(super) struct LabelPrefix {
    pub(super) label: String,
    start: usize,
    pub(super) end: usize,
}

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
    table_note_line_ids: HashSet<String>,
    /// Every table of the document, ruled, headed by rules, aligned by white space or read from
    /// a page's figures: one kind of table, whose lines are its cells.
    tables: Vec<legal_pdf_tables::PdfTable>,
}

impl Claims {
    /// A line a container has claimed (a table's cell or note, a contents entry).
    fn claimed(&self, id: &str) -> bool {
        self.contents_line_ids.contains(id)
            || self.table_cell_line_ids.contains(id)
            || self.table_note_line_ids.contains(id)
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
    /// The reader that first made each heading line one (`read_headings`' passes), for a reader
    /// after the parser to weigh each heading by where it came from.
    #[serde(default)]
    sources: HashMap<String, String>,
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
    /// The share of each measured line's characters set bold on a recognized page, read from
    /// their strokes (`mark_stroke_bold`).
    #[serde(default)]
    stroke_bold: HashMap<String, f64>,
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
    /// Each heading line the stage leaves, and the reader that made it a heading ("later" where a
    /// step after the heading readers did).
    pub heading_sources: HashMap<String, String>,
}

fn normalize_label(value: &str) -> String {
    let translated: String = value
        .trim()
        .chars()
        .map(|character| {
            normalize_decimal_digit(character).unwrap_or_else(|| normalize_note_symbol(character))
        })
        .collect();
    translated
        .parse::<u64>()
        .map_or(translated, |number| number.to_string())
}

pub(super) fn scalar_suffix(value: &str, start: usize) -> &str {
    if value.is_ascii() {
        return &value[start.min(value.len())..];
    }
    let end = value.len();
    let byte = value.char_indices().nth(start).map_or(end, |at| at.0);
    &value[byte..]
}

#[cfg(test)]
pub(super) fn char_slice(value: &str, start: usize, end: usize) -> &str {
    let length = value.chars().count();
    ScalarText::new(value)
        .slice(start.min(length)..end.min(length))
        .expect("valid scalar range")
}

pub(super) fn is_note_symbol(character: char) -> bool {
    matches!(
        character,
        '*' | '∗' | '\u{f02a}' | '†' | '‡' | '§' | '¶' | '#'
    )
}

fn line_start_label_prefix(text: &str) -> Option<LabelPrefix> {
    let leading = text
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let trimmed = text.trim_start();
    let symbols: String = trimmed
        .chars()
        .take(MAX_SYMBOL_LABEL_LEN)
        .take_while(|character| is_note_symbol(*character))
        .collect();
    let token = if symbols.is_empty() {
        trimmed
            .chars()
            .take(4)
            .take_while(char::is_ascii_digit)
            .collect()
    } else {
        symbols
    };
    if token.is_empty() {
        return None;
    }
    let token_chars = token.chars().count();
    let remainder = scalar_suffix(trimmed, token_chars);
    let embedded_endnote = format!("endnote {token}");
    let embedded_chars = embedded_endnote.chars().count();
    if remainder
        .get(..embedded_endnote.len())
        .is_some_and(|value| value.eq_ignore_ascii_case(&embedded_endnote))
    {
        return Some(LabelPrefix {
            label: normalize_label(&token),
            start: leading,
            end: leading + token_chars + embedded_chars,
        });
    }
    let mut remainder_chars = remainder.chars();
    let valid_boundary = match remainder_chars.next() {
        Some(character) if character.is_whitespace() => true,
        Some(character) if ".)],:;-".contains(character) => remainder_chars
            .next()
            .is_none_or(|next| next.is_whitespace()),
        _ => false,
    };
    if !valid_boundary {
        return None;
    }
    let label = normalize_label(&token);
    Some(LabelPrefix {
        label,
        start: leading,
        end: leading + token_chars,
    })
}

pub(super) fn label_prefix(text: &str) -> Option<LabelPrefix> {
    if let Some(prefix) = line_start_label_prefix(text) {
        return Some(prefix);
    }
    let leading = text
        .chars()
        .take_while(|character| character.is_whitespace())
        .count();
    let stripped = text.trim();
    let pure = ((1..=MAX_SYMBOL_LABEL_LEN).contains(&stripped.chars().count())
        && stripped.chars().all(is_note_symbol))
        || ((1..=4).contains(&stripped.chars().count())
            && stripped.chars().all(|character| character.is_ascii_digit()));
    pure.then(|| LabelPrefix {
        label: normalize_label(stripped),
        start: leading,
        end: leading + stripped.chars().count(),
    })
}

fn label_spans<'a>(line: &'a Line, prefix: &'a LabelPrefix) -> impl Iterator<Item = &'a Span> {
    line.spans
        .iter()
        .filter(|span| span.start < prefix.end && span.end > prefix.start)
}

/// A label set above or smaller than the rest of its own line, as a note label is.
fn label_is_raised(line: &Line, prefix: &LabelPrefix, line_size: f64) -> bool {
    let height = line.bbox[3] - line.bbox[1];
    label_spans(line, prefix).any(|span| {
        span.superscript
            || (line_size > 0.0 && span.size > 0.0 && span.size <= line_size * 0.75)
            || (line_size > 0.0
                && span.size > 0.0
                && span.size * 1.25 <= line_size
                && height > 0.0
                && span.bbox[3] <= line.bbox[3] - 0.25 * height)
    })
}

fn label_is_typographic(line: &Line, prefix: &LabelPrefix, line_size: f64, body_size: f64) -> bool {
    let label_size = label_spans(line, prefix)
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(line_size);
    label_is_raised(line, prefix, line_size) || (label_size > 0.0 && label_size <= body_size * 0.75)
}

/// Whether the line just above `index`, in its type and at its leading, ends on a counted
/// enactment's noun ("s", "c", "art") still waiting for its number.
fn reference_wrapped_onto(lines: &[Line], sizes: &[f64], index: usize) -> bool {
    let line = &lines[index];
    let height = line.bbox[3] - line.bbox[1];
    lines
        .iter()
        .zip(sizes)
        .filter(|(above, size)| {
            (**size - sizes[index]).abs() <= 0.5
                && above.bbox[3] <= line.bbox[1] + height * 0.5
                && line.bbox[1] - above.bbox[3] < height * 0.6
                && above.bbox[0] < line.bbox[2]
                && line.bbox[0] < above.bbox[2]
        })
        .max_by(|(left, _), (right, _)| left.bbox[3].total_cmp(&right.bbox[3]))
        .and_then(|(above, _)| above.text.split_whitespace().last())
        .is_some_and(|word| {
            matches!(
                word.trim_end_matches('.'),
                "c" | "ch" | "s" | "ss" | "art" | "arts"
            )
        })
}

/// Text-Fidelity's native-PDF body-font estimator: the largest rounded font
/// bucket carrying at least ten percent of the supplied character mass. Smaller
/// quote/note fonts can be common without redefining the body face, while
/// headings normally lack enough mass to become a substantial bucket.
fn body_font_size<'a>(lines: impl IntoIterator<Item = &'a Line>) -> f64 {
    let mut mass = HashMap::<i64, usize>::new();
    for line in lines {
        let size = line_font_size(line);
        if size > 0.0 {
            *mass.entry((size * 100.0).round() as i64).or_default() += line.text.chars().count();
        }
    }
    let total: usize = mass.values().sum();
    mass.into_iter()
        .filter(|(_, chars)| total > 0 && chars.saturating_mul(10) >= total)
        .map(|(size, _)| size as f64 / 100.0)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0)
}

/// Source labels survive ordering and classification. Unknown labels carry no evidence;
/// they must not erase the known labels of other lines. Ambiguous identities invalidate the map.
fn source_region_contract(pages: &[Page]) -> Option<HashMap<String, String>> {
    let mut regions = HashMap::new();
    let mut ids = HashSet::new();
    for line in pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
    {
        if line.id.is_empty() || !ids.insert(line.id.as_str()) {
            return None;
        }
        let source = line.region_type.trim();
        let tag = matches!(source, "text" | "body" | "" | "unknown" | "unknown_region")
            .then(|| {
                legal_pdf_core::tagged_heading(line, |span| span.structure_role.as_deref()).or_else(
                    || {
                        legal_pdf_core::tagged_line_role(line, |span| {
                            span.structure_role.as_deref().filter(|role| *role == "LI")
                        })
                    },
                )
            })
            .flatten();
        let region = match tag {
            Some(tag) => tag.to_owned(),
            None if legal_pdf_core::is_heading_role(source) || source == "LI" => source.to_owned(),
            None => source.to_ascii_lowercase(),
        };
        if !matches!(region.as_str(), "" | "unknown" | "unknown_region") {
            regions.insert(line.id.clone(), region);
        }
    }
    (!regions.is_empty()).then_some(regions)
}

fn source_regions_cover_pages(pages: &[Page], regions: Option<&HashMap<String, String>>) -> bool {
    regions.is_some_and(|regions| {
        pages
            .iter()
            .flat_map(|page| &page.lines)
            .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
            .all(|line| regions.contains_key(&line.id))
    })
}

#[cfg(test)]
fn source_regions_available(pages: &[Page]) -> bool {
    source_regions_cover_pages(pages, source_region_contract(pages).as_ref())
}

// A contents entry runs its title into a dot leader ("4. Land Use ........ 12"), as pdf-inspector reads one.
use pdf_inspector::markdown::has_dot_leaders as has_dot_leader;

fn classify_pages_with_source(
    pages: &mut [Page],
    separators: &[Option<f64>],
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    // Discover openings with the existing furniture/reference filters before calibrating each document.
    let source = pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| {
            (
                line.region_type.clone(),
                line.exclude_from_body,
                line.detached_references.len(),
            )
        })
        .collect::<Vec<_>>();
    legal_pdf_support::profile::measure("prepare.furniture", || mark_repeated_furniture(pages));
    legal_pdf_support::profile::measure("prepare.detached_references", || {
        associate_detached_references(pages, separators)
    });
    constituents::reconcile_starts(pages, &mut evidence.contexts.constituent_starts);
    let documents = constituents::document_lines(pages, &evidence.contexts.constituent_starts);
    let diagnostics = if documents.is_empty() {
        drop(source);
        classify_document_pages(pages, separators, &HashSet::new(), evidence)
    } else {
        for (line, (region, excluded, references)) in pages
            .iter_mut()
            .flat_map(|page| &mut page.lines)
            .zip(source)
        {
            line.region_type = region;
            line.exclude_from_body = excluded;
            line.detached_references.truncate(references);
        }
        let separators_by_page = pages
            .iter()
            .zip(separators)
            .map(|(page, separator)| (page.index, *separator))
            .collect::<HashMap<_, _>>();
        let shared_pages = pages
            .iter()
            .filter(|page| {
                let mut contexts = page
                    .lines
                    .iter()
                    .map(|line| documents.get(&line.id).copied().unwrap_or(0));
                contexts
                    .next()
                    .is_some_and(|first| contexts.any(|context| context != first))
            })
            .map(|page| page.index)
            .collect::<HashSet<_>>();
        let mut diagnostics = Vec::new();
        constituents::within_document_contexts(pages, &documents, &HashSet::new(), |document| {
            let separators = document
                .iter()
                .map(|page| separators_by_page[&page.index])
                .collect::<Vec<_>>();
            legal_pdf_support::profile::measure("prepare.furniture", || {
                mark_repeated_furniture(document)
            });
            legal_pdf_support::profile::measure("prepare.detached_references", || {
                associate_detached_references(document, &separators)
            });
            diagnostics.extend(classify_document_pages(
                document,
                &separators,
                &shared_pages,
                evidence,
            ));
        });
        for page in pages.iter_mut() {
            order_furniture(page);
        }
        evidence
            .claims.contents_pages
            .retain(|page| !shared_pages.contains(page));
        diagnostics
    };
    evidence.contexts.document_line_ids = documents;
    legal_pdf_support::profile::measure("classify.paragraphs", || {
        constituents::within_document_contexts(
            pages,
            &evidence.contexts.document_line_ids,
            &evidence.contexts.translation_line_ids,
            |document| {
                crate::paragraphs::segment_paragraphs(document);
                contents::group_rows(document, &evidence.claims);
            },
        );
        legal_pdf_tables::part_cell_paragraphs(pages, &evidence.claims.tables);
    });
    diagnostics
}

fn classify_document_pages(
    pages: &mut [Page],
    separators: &[Option<f64>],
    shared_pages: &HashSet<usize>,
    evidence: &mut PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut contents_line_ids = HashSet::new();
    let translation =
        legal_pdf_support::profile::measure("classify.translation", || translation_lines(pages));
    // A table's rulings underline none of its words.
    for page in pages.iter_mut() {
        legal_pdf_tables::clear_table_underlines(page);
    }
    let (tables, listed) = legal_pdf_support::profile::measure("classify.tables", || {
        legal_pdf_tables::document_tables(pages, &translation, &TableReadings)
    });
    // A contents list pdf-inspector read as a table is contents, beside the rows read below.
    contents_line_ids.extend(listed);
    // The document's own lists, read by their typography before any table keeps their lines: a
    // table that is a contents list or a word index is that list.
    let lists = legal_pdf_support::profile::measure("classify.contents_lists", || {
        contents_lists::read(pages, &legal_pdf_tables::table_line_ids(&tables), &evidence.embedded_page_labels)
    });
    let tables: Vec<_> = tables
        .into_iter()
        .filter(|table| {
            let ids = table.line_ids();
            let listed = ids.iter().filter(|id| lists.contents.contains(id.as_str())).count();
            if std::env::var_os("LEGALPDF_CONTENTS_TRACE").is_some() {
                eprintln!("contents\ttable of {} lines, {listed} listed: {}", ids.len(), if listed * 2 <= ids.len() { "kept" } else { "dropped" });
            }
            listed * 2 <= ids.len()
        })
        .collect();
    let table_lines = legal_pdf_tables::table_line_ids(&tables);
    contents_line_ids.extend(lists.contents);
    let mut contents_pages = lists.pages;
    evidence.claims.contents_entries.extend(lists.entries);
    // A row the extractor read across both columns parts into its two languages.
    for page in pages.iter_mut() {
        let base = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0)
            + 1;
        for line in page
            .lines
            .iter_mut()
            .filter(|line| translation.contains(&line.id))
        {
            line.block_index += base;
        }
    }
    let table_started = std::time::Instant::now();
    let table_pages: Vec<_> = pages
        .iter()
        .map(|page| {
            // The tables' cells, and the notes set under a table where it ends on this page.
            let cells: HashSet<usize> = (0..page.lines.len())
                .filter(|index| table_lines.contains(&page.lines[*index].id))
                .collect();
            let ends: Vec<f64> = tables
                .iter()
                .map(legal_pdf_tables::PdfTable::end)
                .filter(|(at, _)| *at == page.index)
                .map(|(_, bottom)| bottom)
                .collect();
            let notes = table_note_lines(&page.lines, &cells, &ends);
            (!cells.is_empty(), cells, notes)
        })
        .collect();

    if std::env::var_os("LEGALPDF_PHASE_TIMING").is_some() {
        eprintln!("phase	classify.table_pages	{}", table_started.elapsed().as_micros());
    }
    for (page, (_, table_cells, table_notes)) in pages.iter().zip(&table_pages) {
        evidence.claims.table_cell_line_ids.extend(
            table_cells
                .iter()
                .filter_map(|index| page.lines.get(*index))
                .map(|line| line.id.clone()),
        );
        evidence.claims.table_note_line_ids.extend(
            table_notes
                .iter()
                .filter_map(|index| page.lines.get(*index))
                .map(|line| line.id.clone()),
        );
    }

    // Read note history within one language; tables and physical order retain the whole page.
    if translation.is_empty() {
        diagnostics.extend(classify_line_roles(
            pages,
            separators,
            &contents_pages,
            evidence,
        ));
    } else {
        let separators_by_page = pages
            .iter()
            .zip(separators)
            .map(|(page, separator)| (page.index, *separator))
            .collect::<HashMap<_, _>>();
        constituents::within_document_contexts(pages, &HashMap::new(), &translation, |language| {
            let separators = language
                .iter()
                .map(|page| separators_by_page[&page.index])
                .collect::<Vec<_>>();
            diagnostics.extend(classify_line_roles(
                language,
                &separators,
                &contents_pages,
                evidence,
            ));
        });
    }
    legal_pdf_support::profile::measure("classify.order_pages", || {
        for (page, (table_page, _, table_notes)) in pages.iter_mut().zip(table_pages) {
            diagnostics.extend(order_page(page, table_page, contents_pages.contains(&page.index), &table_notes));
            legal_pdf_tables::order_table_lines(page, &tables);
        }
    });
    evidence.claims.tables.extend(tables);
    // Keep page-wide contents evidence precise when two documents share a physical page.
    contents_line_ids.extend(
        pages
            .iter()
            .filter(|page| {
                shared_pages.contains(&page.index) && contents_pages.contains(&page.index)
            })
            .flat_map(|page| page.lines.iter().map(|line| line.id.clone())),
    );
    // A table of authorities is no contents list, whatever its leaders and locators.
    let authority_pages: HashSet<usize> = pages
        .iter()
        .filter(|page| page.lines.iter().any(|line| lists.authorities.contains(&line.id)))
        .map(|page| page.index)
        .collect();
    contents_line_ids.retain(|id| !lists.authorities.contains(id));
    contents_pages.retain(|page| !authority_pages.contains(page));
    evidence.claims.contents_line_ids.extend(contents_line_ids);
    evidence.claims.contents_pages.extend(contents_pages);
    legal_pdf_support::profile::measure("classify.captions", || {
        constituents::within_document_contexts(pages, &HashMap::new(), &translation, |document| {
            captions::classify(document, &evidence.claims, &mut evidence.headings);
        });
    });
    evidence.contexts.translation_line_ids.extend(translation);
    diagnostics
}

#[cfg(test)]
fn classify_pages(pages: &mut [Page], separators: &[Option<f64>]) -> Vec<Diagnostic> {
    let mut evidence = PdfPrimitiveEvidence {
        source_regions: source_region_contract(pages),
        ..PdfPrimitiveEvidence::default()
    };
    let diagnostics = classify_pages_with_source(pages, separators, &mut evidence);
    read_headings(pages, &[], &mut evidence);
    build_regions(pages);
    diagnostics
}

/// A document's own dictionary for rejoining words its lines break: its words and its hyphenated
/// compounds (pdf-inspector's `document_vocabulary`).
type Vocabulary = (HashSet<String>, HashSet<String>);

fn vocabulary(pages: &[Page]) -> Vocabulary {
    let text: Vec<&str> = pages.iter().flat_map(|page| &page.lines).map(|line| line.text.trim()).collect();
    pdf_inspector::markdown::document_vocabulary(&text.join(" "))
}

/// A paragraph's lines joined into its text, with the source run of each. A word broken over two
/// lines is rejoined as pdf-inspector rejoins it: plain where the document writes it whole
/// ("de- fendant"), with its hyphen where it writes the compound or the continuation is
/// capitalized ("six- month", "Third- Party"), and otherwise plain before a lowercase continuation
/// ("judg- ment"), the paragraph's lines being one column's; a soft hyphen always joins.
fn join_lines(lines: &[&Line], vocabulary: &Vocabulary) -> (String, Vec<legal_pdf_core::model::SourceTextSpan>) {
    use pdf_inspector::markdown::{join_decision, Join};
    let mut text = String::new();
    let mut source_spans: Vec<legal_pdf_core::model::SourceTextSpan> = Vec::new();
    for line in lines {
        let value = line.text.trim();
        if value.is_empty() {
            continue;
        }
        let first: &str = value.split(|c: char| !c.is_alphabetic()).next().unwrap_or("");
        let join = match text.chars().next_back() {
            Some('\u{00ad}') if !first.is_empty() => Join::Plain,
            Some(hyphen @ ('-' | '\u{00ac}')) => {
                let before = &text[..text.len() - hyphen.len_utf8()];
                let last = before.rsplit(|c: char| !c.is_alphabetic()).next().unwrap_or("");
                let decision = if last.chars().count() >= 2 && first.chars().count() >= 2 {
                    join_decision(last, first, &vocabulary.0, &vocabulary.1)
                } else {
                    Join::Keep
                };
                // Without the document's evidence, a lowercase continuation in one column's
                // paragraph is the word going on.
                match decision {
                    Join::Keep if !last.is_empty() && first.chars().next().is_some_and(char::is_lowercase) => Join::Plain,
                    decision => decision,
                }
            }
            _ => Join::Keep,
        };
        match join {
            Join::Plain => {
                text.pop();
                if let Some(last) = source_spans.last_mut() { last.end -= 1; last.source_end -= 1; }
                source_spans.retain(|span| span.start < span.end);
            }
            Join::Hyphen => {}
            Join::Keep if !text.is_empty() => text.push(' '),
            Join::Keep => {}
        }
        let output_start = legal_structure_model::utf16_len(&text);
        let source_start = legal_structure_model::utf16_len(&line.text[..line.text.len() - line.text.trim_start().len()]);
        let length = legal_structure_model::utf16_len(value);
        source_spans.push(legal_pdf_core::model::SourceTextSpan {start:output_start,end:output_start+length,
            line_id:line.id.clone(),source_start,source_end:source_start+length});
        text.push_str(value);
    }
    (text, source_spans)
}

fn build_paragraphs_with_context(
    pages: &[Page],
    anchors: &HashMap<String, Vec<Anchor>>,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<Paragraph> {
    let vocabulary = vocabulary(pages);
    let context = |id: &str| primitives.contexts.reading_context(id);
    let mut regions: Vec<(&Page, &str, Vec<&Line>)> = Vec::new();
    let mut tails = HashMap::<DocumentContext, usize>::new();
    for page in pages {
        let line_by_id: HashMap<&str, &Line> = page
            .lines
            .iter()
            .map(|line| (line.id.as_str(), line))
            .collect();
        let mut seen = HashSet::new();
        for region in &page.regions {
            if !matches!(region.kind.as_str(), "body" | "heading" | "caption") {
                continue;
            }
            let lines: Vec<&Line> = region
                .line_ids
                .iter()
                .filter_map(|id| line_by_id.get(id.as_str()).copied())
                .filter(|line| !line.exclude_from_body)
                .collect();
            for lines in lines.chunk_by(|left, right| context(&left.id) == context(&right.id)) {
                let key = context(&lines[0].id);
                let first = seen.insert(key);
                // The other column's intervening regions do not close this context's paragraph.
                if let Some(&slot) = tails.get(&key) {
                    let (_, kind, previous) = &mut regions[slot];
                    if first
                        && region.kind == "body"
                        && *kind == "body"
                        && crate::paragraphs::continues_onto(previous, lines)
                    {
                        previous.extend_from_slice(lines);
                        continue;
                    }
                }
                tails.insert(key, regions.len());
                regions.push((page, region.kind.as_str(), lines.to_vec()));
            }
        }
    }
    let mut paragraphs = Vec::new();
    for (page, kind, lines) in regions {
        let (text, source_spans) = join_lines(&lines, &vocabulary);
        if text.is_empty() {
            continue;
        }
        let mut events = Vec::new();
        let joined_coordinates = ScalarText::new(&text);
        for line in &lines {
            let coordinates = ScalarText::new(&line.text);
            for anchor in anchors.get(&line.id).into_iter().flatten() {
                let start = coordinates.utf16(anchor.start);
                let end = coordinates.utf16(anchor.end);
                let Some(run) = source_spans.iter().find(|span| span.line_id == line.id
                    && span.source_start <= start && end <= span.source_end) else { continue; };
                let start = joined_coordinates.scalar_at_utf16(run.start + start - run.source_start).unwrap();
                let end = joined_coordinates.scalar_at_utf16(run.start + end - run.source_start).unwrap();
                events.push((start,end,anchor.pair_id.as_str(),anchor.label.as_str()));
            }
        }
        events.sort_by_key(|event| (event.0, event.1));
        let (rendered, output_anchors, source_spans) = if events.is_empty() {
            (text, Vec::new(), source_spans)
        } else {
            let coordinates = ScalarText::new(&text);
            let text_len = coordinates.len();
            let mut rendered = String::with_capacity(text.len());
            let mut output_anchors = Vec::with_capacity(events.len());
            let mut retained_spans = Vec::new();
            let (mut cursor, mut rendered_len) = (0, 0);
            for (start, end, pair_id, label) in events {
                let start = start.max(cursor).min(text_len);
                let end = end.max(start).min(text_len);
                let output_start = legal_structure_model::utf16_len(&rendered);
                retained_spans.extend(source_spans.iter().filter_map(|span| span.retain(coordinates.utf16(cursor),coordinates.utf16(start),output_start)));
                rendered.push_str(coordinates.slice(cursor..start).unwrap());
                rendered_len += start - cursor;
                let offset = rendered_len;
                rendered.push_str("⟦FN:");
                rendered.push_str(&pair_id);
                rendered.push('⟧');
                rendered_len += 5 + pair_id.chars().count();
                output_anchors.push(ParagraphAnchor {
                    pair_id: pair_id.to_owned(),
                    label: label.to_owned(),
                    offset,
                });
                cursor = end;
            }
            let output_start = legal_structure_model::utf16_len(&rendered);
            retained_spans.extend(source_spans.iter().filter_map(|span| span.retain(coordinates.utf16(cursor),coordinates.utf16(text_len),output_start)));
            rendered.push_str(coordinates.slice(cursor..text_len).unwrap());
            (rendered, output_anchors, retained_spans)
        };
        paragraphs.push(Paragraph {
            source_spans,
            id: format!("para-{:06}", paragraphs.len() + 1),
            page_index: page.index,
            region_type: kind.to_owned(),
            text: rendered,
            line_ids: lines.iter().map(|line| line.id.clone()).collect(),
            anchors: output_anchors,
        });
    }
    paragraphs
}

#[cfg(test)]
fn build_paragraphs(pages: &[Page], anchors: &HashMap<String, Vec<Anchor>>) -> Vec<Paragraph> {
    let starts = constituents::starts(pages);
    let primitives = PdfPrimitiveEvidence {
        contexts: Contexts {
            document_line_ids: constituents::document_lines(pages, &starts),
            translation_line_ids: translation_lines(pages),
            ..Contexts::default()
        },
        ..Default::default()
    };
    build_paragraphs_with_context(pages, anchors, &primitives)
}

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
}

fn clean_markers(value: &str) -> String {
    marker_re().replace_all(value, "").trim().to_owned()
}

fn sentence_boundaries(chars: &[char]) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::<(usize, usize)>::new();
    let mut index = 0;
    while index < chars.len() {
        if matches!(chars[index], '.' | '?' | '!') {
            let start = index;
            index += 1;
            while chars
                .get(index)
                .is_some_and(|character| ['"', '\'', '”', '’', ')', ']'].contains(character))
            {
                index += 1;
            }
            let marker_follows = chars
                .get(index..)
                .is_some_and(|tail| tail.starts_with(&['⟦', 'F', 'N', ':']));
            if index == chars.len()
                || chars
                    .get(index)
                    .is_some_and(|character| character.is_whitespace())
                || marker_follows
            {
                boundaries.push((start, index));
            }
            continue;
        }
        index += 1;
    }
    boundaries
}

fn sentence_at_boundaries(chars: &[char], boundaries: &[(usize, usize)], offset: usize) -> String {
    let previous = &boundaries[..boundaries.partition_point(|(_, end)| *end <= offset)];
    let (start, end) = if previous.last().is_some_and(|(_, end)| {
        chars[*end..offset.min(chars.len())]
            .iter()
            .all(|character| character.is_whitespace())
    }) {
        (
            if previous.len() > 1 {
                previous[previous.len() - 2].1
            } else {
                0
            },
            previous.last().map_or(chars.len(), |(_, end)| *end),
        )
    } else {
        let start = previous.last().map_or(0, |(_, end)| *end);
        let end = boundaries
            .iter()
            .find(|(boundary_start, _)| *boundary_start >= offset)
            .map_or(chars.len(), |(_, boundary_end)| *boundary_end);
        (start, end)
    };
    clean_markers(&chars[start..end].iter().collect::<String>())
}

#[cfg(test)]
fn sentence_at(text: &str, offset: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    sentence_at_boundaries(&chars, &sentence_boundaries(&chars), offset)
}

fn attach_propositions(
    footnotes: &mut [Footnote],
    paragraphs: &[Paragraph],
    primitives: &PdfPrimitiveEvidence,
) {
    let by_pair: HashMap<String, usize> = footnotes
        .iter()
        .enumerate()
        .map(|(index, footnote)| (footnote.pair_id.clone(), index))
        .collect();
    let mut tails = HashMap::<DocumentContext, String>::new();
    for paragraph in paragraphs {
        let context = paragraph
            .line_ids
            .first()
            .map_or((0, false), |id| primitives.contexts.document_context(id));
        let previous_tail = tails.entry(context).or_default();
        let chars: Vec<char> = paragraph.text.chars().collect();
        let boundaries = sentence_boundaries(&chars);
        let mut previous_offset = 0;
        for anchor in &paragraph.anchors {
            let Some(&index) = by_pair.get(&anchor.pair_id) else {
                continue;
            };
            let offset = anchor.offset;
            footnotes[index].sentence_proposition =
                sentence_at_boundaries(&chars, &boundaries, offset);
            let passage_text: String = chars
                [previous_offset.min(chars.len())..offset.min(chars.len())]
                .iter()
                .collect();
            let passage = clean_markers(&passage_text);
            footnotes[index].passage_since_prior_note = if passage.is_empty() {
                previous_tail.clone()
            } else {
                passage
            };
            previous_offset = offset + 5 + anchor.pair_id.chars().count();
        }
        *previous_tail = chars[previous_offset.min(chars.len())..]
            .iter()
            .collect::<String>()
            .trim()
            .to_owned();
    }
}

fn line_family(line: &Line) -> &'static str {
    match line.region_type.as_str() {
        "footnote" => "note",
        "heading" => "heading",
        _ => "body",
    }
}

fn hyphen_fragment_tail(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z]{2,}[-\u{00ac}\u{00ad}]$").unwrap())
        .is_match(text.trim_end())
}

fn hyphen_continuation(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]{2,}").unwrap())
        .is_match(text.trim_start())
}

fn text_flow_faults(pages: &[Page]) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for page in pages {
        let mut eligible = page.lines.iter().filter(|line| {
            !matches!(line.region_type.as_str(), "header" | "footer") && !line.exclude_from_body
        });
        let Some(mut previous) = eligible.next() else {
            continue;
        };
        for current in eligible {
            if hyphen_fragment_tail(&previous.text) {
                let continues = hyphen_continuation(&current.text);
                if continues && line_family(previous) != line_family(current) {
                    let mut diagnostic = Diagnostic::warning(
                        "REGION_BOUNDARY_FAULT",
                        format!(
                            "A hyphenated word spans the {}/{} boundary; either a region label or the order is wrong.",
                            line_family(previous),
                            line_family(current)
                        ),
                        Some(page.index),
                    );
                    diagnostic.line_ids = vec![previous.id.clone(), current.id.clone()];
                    diagnostics.push(diagnostic);
                } else if !continues && line_family(previous) == line_family(current) {
                    let mut diagnostic = Diagnostic::info(
                        "DANGLING_SOFT_HYPHEN",
                        "A line ends mid-word but the next eligible line does not continue it.",
                        Some(page.index),
                    );
                    diagnostic.line_ids.push(previous.id.clone());
                    diagnostics.push(diagnostic);
                }
            }
            previous = current;
        }
    }
    diagnostics
}

fn unmatched_reference_diagnostics(
    pages: &[Page],
    footnotes: &[Footnote],
    anchors: &HashMap<String, Vec<Anchor>>,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<Diagnostic> {
    let labels: HashSet<_> = footnotes
        .iter()
        .map(|note| {
            let context = primitives.contexts.note_context(note);
            (context, note.label.as_str())
        })
        .collect();
    let primary_lines: HashSet<&str> = footnotes
        .iter()
        .filter_map(|note| note.reference_line_id.as_deref())
        .collect();
    let mut diagnostics = Vec::new();
    for line in pages.iter().flat_map(|page| page.lines.iter()) {
        for detached in &line.detached_references {
            let label = normalize_label(&detached.note_id);
            let start = detached.start_offset;
            let end = detached.end_offset;
            let paired = anchors.get(&line.id).is_some_and(|entries| {
                entries.iter().any(|anchor| {
                    anchor.start == start && anchor.end == end && anchor.label == label
                })
            });
            if !paired {
                let mut diagnostic = Diagnostic::warning(
                    "FOOTNOTE_UNMATCHED_REFERENCE",
                    format!("Detached superscript '{label}' has no paired label."),
                    Some(line.page_index),
                );
                diagnostic.line_ids.push(line.id.clone());
                if !detached.source_line_id.is_empty() {
                    diagnostic.line_ids.push(detached.source_line_id.clone());
                }
                diagnostic.details.insert("label".to_owned(), json!(label));
                diagnostics.push(diagnostic);
            }
        }
        if line.exclude_from_body
            || !matches!(line.region_type.as_str(), "body" | "heading")
            || primary_lines.contains(line.id.as_str())
        {
            continue;
        }
        if line.spans.iter().any(|span| {
            span.superscript
                && labels.contains(&(
                    primitives.contexts.document_context(&line.id),
                    normalize_label(span.text.trim()).as_str(),
                ))
        }) {
            let mut diagnostic = Diagnostic::warning(
                "FOOTNOTE_UNMATCHED_REFERENCE",
                "A superscript resembling a known note label was not paired.",
                Some(line.page_index),
            );
            diagnostic.line_ids.push(line.id.clone());
            diagnostics.push(diagnostic);
        }
    }
    diagnostics
}

fn attach_crossrefs(
    footnotes: &mut [Footnote],
    diagnostics: &mut Vec<Diagnostic>,
    primitives: &PdfPrimitiveEvidence,
) {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern = RE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:(supra|infra),?\s+(?:foot)?notes?|(op)\.?\s*cit\.?,?\s+(?:foot)?notes?|(see)\s+(?:also\s+)?footnote)\s+(\d{1,3})\b",
        )
        .unwrap()
    });
    let mut by_number: HashMap<(DocumentContext, u32), Vec<(usize, String)>> = HashMap::new();
    for footnote in footnotes.iter() {
        if let Ok(number) = footnote.label.parse::<u32>() {
            by_number
                .entry((primitives.contexts.note_context(footnote), number))
                .or_default()
                .push((footnote.restart_sequence, footnote.pair_id.clone()));
        }
    }
    for footnote in footnotes {
        let context = primitives.contexts.note_context(footnote);
        let (mut prior_byte, mut prior_scalar) = (0, 0);
        for capture in pattern.captures_iter(&footnote.body) {
            let Some(found) = capture.get(0) else {
                continue;
            };
            let number = capture
                .get(4)
                .and_then(|value| value.as_str().parse::<u32>().ok())
                .unwrap_or(0);
            let candidates = by_number
                .get(&(context, number))
                .map(Vec::as_slice)
                .unwrap_or_default();
            let target_pair_id = if candidates.len() == 1 {
                candidates[0].1.clone()
            } else {
                let mut scoped = candidates
                    .iter()
                    .filter(|(restart, _)| *restart == footnote.restart_sequence);
                match (scoped.next(), scoped.next()) {
                    (Some((_, pair_id)), None) => pair_id.clone(),
                    _ => String::new(),
                }
            };
            let kind = if let Some(value) = capture.get(1) {
                value.as_str().to_lowercase()
            } else if capture.get(2).is_some() {
                "op_cit".to_owned()
            } else {
                "see_footnote".to_owned()
            };
            let start = prior_scalar + footnote.body[prior_byte..found.start()].chars().count();
            let end = start + found.as_str().chars().count();
            (prior_byte, prior_scalar) = (found.end(), end);
            let record = FootnoteCrossref {
                source_pair_id: footnote.pair_id.clone(),
                kind,
                number,
                shortform: layout_analysis().crossref_short_form(&footnote.body, found.start()),
                start,
                end,
                resolved: !target_pair_id.is_empty(),
                target_pair_id,
                target_count: candidates.len(),
            };
            if !record.resolved {
                let mut diagnostic = Diagnostic::info(
                    "NOTE_CROSSREF_UNRESOLVED",
                    if candidates.is_empty() {
                        format!(
                        "Note {} references {} note {}, which no paired note carries - a pairing-quality witness.",
                        footnote.label, record.kind, number
                    )
                    } else {
                        format!(
                            "Note {} references {} note {}, which has multiple possible targets.",
                            footnote.label, record.kind, number
                        )
                    },
                    footnote.reference_page.map(|page| page as usize),
                );
                diagnostic
                    .details
                    .insert("crossref".to_owned(), json!(&record));
                diagnostics.push(diagnostic);
            }
            footnote.crossrefs.push(record);
        }
    }
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
        for line in page.lines.iter_mut().filter(|line| line.region_id.contains("-ppdoc-r")) {
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
            if graph::line_labels(page).next().is_none() && std::env::var_os("LEGALPDF_NO_DETACHED_MARKERS").is_none() {
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
                changed += note_regions::correct_note_regions(document, &separators);
                for (page, line) in excluded {
                    document[page].lines[line].exclude_from_body = false;
                }
            },
        );
        changed
    });
    // A scan's bold, read once from its strokes after notes are known, for the quotation and
    // heading readers alike.
    primitives.stroke_bold = mark_stroke_bold(pages);
    primitives.quotations = legal_pdf_support::profile::measure("reader.quotations", || quotations::read(pages, &primitives));
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

/// The decisions that rest on the readers' pages.
fn prepare_composed(
    pages: &mut [Page],
    primitives: PdfPrimitiveEvidence,
    mut diagnostics: Vec<Diagnostic>,
) -> PdfPreparation {
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
    // The document's type, where known (the caller's profile): an instrument numbers its provisions.
    if identity.profile == Some(legal_structure_model::DetectionProfile::Instrument) {
        heading_rules::instrument_captions(pages, &mut prepared.primitives);
    }
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
    // The engine's inputs, for the structure gold to replay engine changes without parsing
    // (legal-structure's replay example).
    if let Ok(path) = std::env::var("LEGALPDF_RESOLVER_DUMP") {
        let dump = json!({
            "document_id": identity.document_id,
            "source_sha256": identity.source_sha256,
            "text": resolution.index.text(),
            "nodes": nodes,
            "runs": resolution.runs,
            "evidence": resolution.evidence,
            "note_pairs": note_pairs,
            "diagnostics": graph_diagnostics,
        });
        let _ = std::fs::write(&path, dump.to_string());
    }
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
                    let contexts = outline_contexts(
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
    let heading_sources = pages
        .iter()
        .flat_map(|page| &page.lines)
        .filter(|line| line.region_type == "heading")
        .map(|line| {
            let source = prepared.primitives.headings.sources.get(&line.id).cloned().unwrap_or_else(|| "later".to_owned());
            (line.id.clone(), source)
        })
        .collect();
    Ok(StructureOutput {
        paragraphs,
        footnotes,
        diagnostics,
        structure_graph,
        heading_sources,
    })
}

/// Settle each node's exact membership and role, and give the context its outline lives in: the
/// constituent document holding it, and its quotation, note, table cell or parallel text.
fn outline_contexts(
    nodes: &mut [legal_structure_model::StructureNode], document_id: &str,
    quotes: &quotations::Reading,
    index: &PdfTextIndex,
    primitives: &PdfPrimitiveEvidence,
) -> Vec<legal_pdf_witnesses::OutlineContext> {
    use legal_structure_model::PrimaryRole;
    let translation = &primitives.contexts.translation_line_ids;
    // Establish exact membership before comparing containers. An envelope may cross
    // intervening columns, notes, quotations, or constituent documents.
    for node in nodes.iter_mut() {
        if node.ranges.is_empty() { node.ranges.push(node.range); }
        if let Some(marker) = node.marker_range {
            node.ranges.push(marker);
        }
        node.ranges.sort_by_key(|range| (range.start, range.end));
        let mut merged: Vec<legal_structure_model::ScalarRange> = Vec::new();
        for range in node.ranges.drain(..) {
            if let Some(last) = merged.last_mut().filter(|last| range.start <= last.end) {
                last.end = last.end.max(range.end);
            } else {
                merged.push(range);
            }
        }
        node.ranges = merged;
        // Keep owned source fragments when the envelope spans intervening columns.
        if node.kind != NodeKind::Page && !node.line_ids.is_empty() {
            let mut source = node
                .line_ids
                .iter()
                .filter_map(|id| index.line(id).map(|line| line.range))
                .collect::<Vec<_>>();
            if let Some(marker) = node.marker_range {
                let ids = index.line_ids(marker);
                if let Some(first) = ids.first() {
                    let context = primitives.contexts.reading_context(first);
                    source.extend(
                        ids.iter()
                            .filter(|id| primitives.contexts.reading_context(id) == context)
                            .filter_map(|id| index.line(id))
                            .map(|line| ScalarRange {
                                start: marker.start.max(line.range.start),
                                end: marker.end.min(line.range.end),
                            })
                            .filter(|part| part.start < part.end),
                    );
                }
            }
            let source = graph::source_membership(source);
            let mut owned = Vec::new();
            let mut at = 0;
            for range in &node.ranges {
                while at < source.len() && source[at].end <= range.start {
                    at += 1;
                }
                for part in source[at..]
                    .iter()
                    .take_while(|part| part.start < range.end)
                {
                    let start = range.start.max(part.start);
                    let end = range.end.min(part.end);
                    if start < end {
                        owned.push(ScalarRange { start, end });
                    }
                }
            }
            node.ranges = owned;
        }
        let mut seen_lines = HashSet::new();
        node.line_ids = node.ranges.iter().flat_map(|range| index.line_ids(*range))
            .filter(|id| seen_lines.insert(id.clone())).collect();
        node.page_indexes = index.page_indexes_for_line_ids(&node.line_ids);
        if let (Some(first), Some(last)) = (node.ranges.first(), node.ranges.last()) {
            node.range.start = first.start;
            node.range.end = last.end;
        }
        if let Some(start) = &mut node.content_start {
            if let Some(part) = node.ranges.iter().find(|part| *start < part.end) {
                *start = (*start).max(part.start);
            }
        }
    }
    let constituents = nodes.iter().filter(|node| node.grammar.as_deref().is_some_and(|grammar| grammar.starts_with("constituent_")))
        .map(|node| (node.id.clone(), node.ranges.clone())).collect::<Vec<_>>();
    let cells = nodes.iter().filter(|node| matches!(node.kind, NodeKind::Cell | NodeKind::Footnote | NodeKind::Endnote))
        .map(|node| (node.id.clone(), node.ranges.clone())).collect::<Vec<_>>();
    let contains = |outer: &[legal_structure_model::ScalarRange], inner: &[legal_structure_model::ScalarRange]| {
        !inner.is_empty() && inner.iter().all(|part| outer.iter().any(|range|
            range.start <= part.start && part.end <= range.end))
    };
    nodes
        .iter_mut()
        .map(|node| {
            if node.kind == NodeKind::Heading {
                node.role = Some(PrimaryRole::Heading);
                node.heading_range = (node.ranges.len() == 1).then_some(node.range);
            } else if matches!(
                node.kind,
                NodeKind::Paragraph | NodeKind::ListItem | NodeKind::Prose
            ) {
                node.role.get_or_insert(PrimaryRole::Text);
            }
            let smallest = |holders: &[(String, Vec<legal_structure_model::ScalarRange>)]| {
                holders
                    .iter()
                    .filter(|(id, ranges)| id != &node.id && contains(ranges, &node.ranges))
                    .min_by_key(|(_, ranges)| {
                        ranges
                            .iter()
                            .map(|range| range.end - range.start)
                            .sum::<usize>()
                    })
                    .map(|(id, _)| id.clone())
            };
            // A page holds whatever ends on it (a blank page sits at the edge of a note) rather than
            // belonging to its quotation or note: it names only its document.
            let unit = node.kind != NodeKind::Page;
            let quotation = if node.kind == NodeKind::Quote {
                quotes
                    .introductions
                    .get(&node.id)
                    .and_then(|id| quotes.scopes.get(id))
                    .cloned()
            } else {
                node.line_ids
                    .first()
                    .filter(|_| unit)
                    .and_then(|id| quotes.scopes.get(id))
                    .filter(|id| {
                        node.line_ids
                            .iter()
                            .all(|line| quotes.scopes.get(line) == Some(*id))
                    })
                    .cloned()
            };
            let translated = unit
                && !node.line_ids.is_empty()
                && node.line_ids.iter().all(|id| translation.contains(id));
            let container = smallest(&cells)
                .filter(|_| unit)
                .map(|id| format!("object:{id}"))
                .or_else(|| translated.then(|| "translation".to_owned()));
            (
                smallest(&constituents).unwrap_or_else(|| document_id.to_owned()),
                quotation,
                container,
            )
        })
        .collect()
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
    let prepared = prepare_composed(pages, primitives, diagnostics);
    derive_prepared(pages, prepared, identity)
}

pub fn status(diagnostics: &[Diagnostic], pages: &[Page]) -> String {
    let ocr_pages: HashSet<Option<usize>> = diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "OCR_REQUIRED")
        .map(|diagnostic| diagnostic.page_index)
        .collect();
    if !pages.is_empty() && ocr_pages.len() == pages.len() {
        return "ocr_required".to_owned();
    }
    if diagnostics.iter().any(|diagnostic| {
        matches!(diagnostic.severity.as_str(), "warning" | "error")
            && (diagnostic.code == "OCR_REQUIRED"
                || HARD_DIAGNOSTICS.contains(&diagnostic.code.as_str()))
    }) {
        "degraded".to_owned()
    } else {
        "ready".to_owned()
    }
}

fn validate_page_records(pages: &[Page]) -> Result<HashSet<&str>> {
    let mut ids = HashSet::new();
    let mut span_ids = HashSet::new();
    let mut word_ids = HashSet::new();
    for page in pages {
        if page.lines.iter().any(|line| !ids.insert(line.id.as_str())) {
            return Err(Error::Message(
                "document contains duplicate line IDs".to_owned(),
            ));
        }
        let mut regions_by_line = HashMap::new();
        let duplicate_region_line = page.regions.iter().any(|region| {
            region
                .line_ids
                .iter()
                .any(|line| regions_by_line.insert(line.as_str(), region).is_some())
        });
        if duplicate_region_line
            || regions_by_line.len() != page.lines.len()
            || page
                .lines
                .iter()
                .any(|line| !regions_by_line.contains_key(line.id.as_str()))
        {
            return Err(Error::Message(format!(
                "page {} region coverage is incomplete",
                page.number
            )));
        }
        for line in &page.lines {
            if !line
                .spans
                .iter()
                .all(|span| span_ids.insert(span.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate span IDs".to_owned(),
                ));
            }
            if !line
                .words
                .iter()
                .all(|word| word_ids.insert(word.id.as_str()))
            {
                return Err(Error::Message(
                    "document contains duplicate word IDs".to_owned(),
                ));
            }
            let region = regions_by_line
                .get(line.id.as_str())
                .copied()
                .ok_or_else(|| {
                    Error::Message(format!("line {} has no containing region", line.id))
                })?;
            if line.region_id != region.id || line.region_type != region.kind {
                return Err(Error::Message(format!(
                    "page {} line/region annotations disagree for {}",
                    page.number, line.id
                )));
            }
            let mut prior_end = 0;
            let scalar_boundaries = (!line.text.is_ascii()).then(|| {
                line.text
                    .char_indices()
                    .map(|(index, _)| index)
                    .chain(std::iter::once(line.text.len()))
                    .collect::<Vec<_>>()
            });
            let scalar_len = scalar_boundaries
                .as_ref()
                .map_or(line.text.len(), |boundaries| boundaries.len() - 1);
            for word in &line.words {
                let text = scalar_boundaries.as_ref().map_or_else(
                    || line.text.get(word.start..word.end),
                    |boundaries| {
                        boundaries
                            .get(word.start)
                            .zip(boundaries.get(word.end))
                            .and_then(|(&start, &end)| line.text.get(start..end))
                    },
                );
                if word.start < prior_end
                    || word.end <= word.start
                    || word.end > scalar_len
                    || text != Some(word.text.as_str())
                {
                    return Err(Error::Message(format!(
                        "line {} contains invalid word geometry",
                        line.id
                    )));
                }
                prior_end = word.end;
            }
        }
        let printed = (
            page.printed_label.as_deref(),
            page.printed_label_source.as_deref(),
            page.printed_label_line_id.as_deref(),
        );
        if printed.0.is_some() || printed.1.is_some() || printed.2.is_some() {
            let (Some(label), Some(source), Some(line_id)) = printed else {
                return Err(Error::Message(format!(
                    "page {} has incomplete printed-label provenance",
                    page.number
                )));
            };
            let valid = page.lines.iter().any(|line| {
                line.id == line_id
                    && line.region_type == source
                    && matches!(source, "header" | "footer")
                    && printed_label(line.text.trim()).as_deref() == Some(label)
            });
            if !valid {
                return Err(Error::Message(format!(
                    "page {} has invalid printed-label provenance",
                    page.number
                )));
            }
        }
    }
    Ok(ids)
}

pub fn validate_document(document: &LegalDocument) -> Result<()> {
    if document.page_count != document.pages.len() {
        return Err(Error::Message(
            "document page_count does not match the page collection".to_owned(),
        ));
    }
    validate_pdf_components(
        &document.document_id,
        &document.source_sha256,
        &document.pages,
        &document.paragraphs,
        &document.footnotes,
        &document.structure_graph,
    )
}

pub fn validate_pdf_components(
    document_id: &str,
    source_sha256: &str,
    pages: &[Page],
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
    structure_graph: &DocumentStructure,
) -> Result<()> {
    let known_lines = validate_page_records(pages)?;
    let known_pages = pages.iter().map(|page| page.index).collect::<HashSet<_>>();
    let mut pair_ids = HashSet::new();
    for footnote in footnotes {
        if !pair_ids.insert(&footnote.pair_id) {
            return Err(Error::Message(
                "document contains duplicate footnote pair IDs".to_owned(),
            ));
        }
        if footnote
            .reference_line_id
            .as_ref()
            .is_some_and(|line| !known_lines.contains(line.as_str()))
            || footnote
                .body_line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "footnote {} contains an unknown source line",
                footnote.pair_id
            )));
        }
    }
    let mut paragraph_ids = HashSet::new();
    for paragraph in paragraphs {
        if !paragraph_ids.insert(&paragraph.id)
            || paragraph
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
        {
            return Err(Error::Message(format!(
                "paragraph {} is invalid",
                paragraph.id
            )));
        }
    }
    if structure_graph.document_id != document_id
        || structure_graph.source_sha256.as_deref() != Some(source_sha256)
    {
        return Err(Error::Message(
            "structure graph identity disagrees with the document".to_owned(),
        ));
    }
    let (text_length, line_count) = pages
        .iter()
        .flat_map(|page| &page.lines)
        .fold((0, 0_usize), |(units, count), line| {
            (units + utf16_len(&line.text), count + 1)
        });
    let text_length = text_length + line_count.saturating_sub(1);
    let mut node_ids = HashSet::new();
    for node in &structure_graph.nodes {
        if node.id.is_empty()
            || !node_ids.insert(node.id.as_str())
            || node.range.start > node.range.end
            || node.range.end > text_length
            || node
                .page_indexes
                .iter()
                .any(|page| !known_pages.contains(page))
            || node
                .line_ids
                .iter()
                .any(|line| !known_lines.contains(line.as_str()))
            || (node.kind == NodeKind::Section
                && node.grammar.as_deref() != Some("heading_section")
                && !node
                    .grammar
                    .as_deref()
                    .is_some_and(|grammar| grammar.starts_with(constituents::GRAMMAR))
                && (node.locator_kind.as_deref().is_none_or(str::is_empty)
                    || node
                        .proof
                        .as_ref()
                        .is_none_or(|proof| proof.rule != ResolutionRuleV2::HierarchySection)))
        {
            return Err(Error::Message(format!(
                "structure node {} is invalid",
                node.id
            )));
        }
    }
    if structure_graph.nodes.iter().any(|node| {
        node.parent_id
            .as_deref()
            .is_some_and(|parent| !node_ids.contains(parent))
    }) {
        return Err(Error::Message(
            "structure graph contains an unknown parent".to_owned(),
        ));
    }
    for note in &structure_graph.notes {
        let valid_range = |range: ScalarRange| range.start <= range.end && range.end <= text_length;
        if note.node_id.as_deref().is_some_and(|id| !node_ids.contains(id))
            || !valid_range(note.label_range)
            || !valid_range(note.body_range)
            || note.label_ranges.iter().chain(&note.body_ranges).any(|range| !valid_range(*range))
            || note
                .primary_reference
                .is_some_and(|range| !valid_range(range))
            || note.references.iter().any(|reference| {
                !valid_range(reference.range)
                    || reference
                        .page_indexes
                        .iter()
                        .any(|page| !known_pages.contains(page))
                    || reference
                        .line_ids
                        .iter()
                        .any(|line| !known_lines.contains(line.as_str()))
            })
        {
            return Err(Error::Message(format!(
                "structure note {} is invalid",
                note.id
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
