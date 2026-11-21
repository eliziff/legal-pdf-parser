//! A lookup by page, paragraph, section or footnote: the request, the units it returns, and how
//! each locator finds its units.

use std::collections::HashSet;
use legal_pdf_core::model::{Page, ScalarRange};
use legal_structure_model::{serialization_sha256, utf16_len, DocumentStructure, NodeKind, ScalarText, StructureNode};
use serde::{Deserialize, Serialize};
use super::{LOCATOR_KINDS, MAX_CONTEXT, MAX_RETURN_CHARS, MAX_UNITS};
use super::document::{lookup_payload_sha256, page_confidence, page_index, page_number, PdfDocument};
use legal_document_query::{note_named, DocumentQuery, Provisions};

/// A paragraph a PDF numbers in print ("[12]", or 12 in the margin): its number and its printings, the structure's
/// paragraph nodes that print it (a parallel report's columns), the first printing's first, as the structure engine
/// reads them (`paragraph_printings`).
pub type PrintedParagraph = (usize, Vec<usize>);

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS), ts(optional_fields = nullable))]
pub struct PdfLookupRequest {
    pub locator_kind: String,
    pub locator: String,
    #[serde(default)]
    pub end_locator: Option<String>,
    #[serde(default)]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<usize>", optional))]
    pub context_blocks: usize,
    #[serde(default)]
    pub page: Option<u32>,
    #[serde(default)]
    pub occurrence: Option<usize>,
    /// The instrument a provision is of, where the PDF prints several ("Canadian Charter of Rights and Freedoms" in
    /// "The Constitution Acts 1867 to 1982").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub instrument: Option<String>,
}

impl PdfLookupRequest {
    pub fn new(locator_kind: impl Into<String>, locator: impl Into<String>) -> Self {
        Self {
            locator_kind: locator_kind.into(),
            locator: locator.into(),
            end_locator: None,
            context_blocks: 0,
            page: None,
            occurrence: None,
            instrument: None,
        }
    }

    fn valid(&self) -> bool {
        LOCATOR_KINDS.contains(&self.locator_kind.as_str())
            && !self.locator.trim().is_empty()
            && utf16_len(&self.locator) <= 200
            && self
                .end_locator
                .as_deref()
                .is_none_or(|end| end.is_empty() || utf16_len(end) <= 200)
            && self.context_blocks <= MAX_CONTEXT
            && self.page.is_none_or(|page| page > 0)
            && self.occurrence.is_none_or(|occurrence| occurrence > 0)
            && self.instrument.as_deref().is_none_or(|instrument| utf16_len(instrument) <= 200)
    }
}

/// A unit a lookup returns: its node, as the model serializes it, and what no node holds: its text as the lookup reads
/// it, how far its pages' text is trusted, a numbered paragraph's other printings (a parallel report's columns), and a
/// note's count of its label's printings up to it and the page its first call stands on.
#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct PdfLookupUnit {
    pub node: StructureNode,
    pub text: String,
    /// The pages (from 1) it stands on (`page_numbers`).
    pub pages: Vec<u32>,
    /// The spans of the document's query text its node holds (`DocumentStructure::query_ranges`).
    pub ranges: Vec<ScalarRange>,
    pub confidence: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<Vec<StructureNode>>", optional))]
    pub printings: Vec<StructureNode>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub occurrence: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub reference_page: Option<u32>,
}

impl PdfLookupUnit {
    fn new(node: &StructureNode, text: String, confidence: Option<f64>) -> Self {
        Self { node: node.clone(), text, pages: Vec::new(), ranges: Vec::new(), confidence, printings: Vec::new(), occurrence: None, reference_page: None }
    }

    /// The pages (from 1) the unit stands on: its note's first call's, then its node's and its printings'.
    pub fn page_numbers(&self) -> Vec<u32> {
        let mut pages: Vec<u32> = self.reference_page.into_iter().collect();
        for page in std::iter::once(&self.node).chain(&self.printings).flat_map(|node| &node.page_indexes) {
            let page = *page as u32 + 1;
            if !pages.contains(&page) {
                pages.push(page);
            }
        }
        pages
    }
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct PdfLookupPage {
    pub page_number: u32,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum PdfLookupStatus {
    Found,
    NotFound,
    Ambiguous,
    Invalid,
    Unavailable,
}

#[derive(Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
pub struct PdfStructureLookup {
    pub requested: PdfLookupRequest,
    /// The kind of unit the request asks for: a page, a transcript's line, a paragraph, a note or a provision.
    #[cfg_attr(feature = "binding-types", ts(type = "\"page\" | \"line\" | \"paragraph\" | \"footnote\" | \"section\""))]
    pub kind: &'static str,
    pub units: Vec<PdfLookupUnit>,
    pub before: Vec<PdfLookupUnit>,
    pub after: Vec<PdfLookupUnit>,
    pub matches: Vec<String>,
    pub pages: Vec<PdfLookupPage>,
    pub status: PdfLookupStatus,
    #[serde(skip_serializing_if = "String::is_empty")]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<String>", optional))]
    pub payload_sha256: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    #[cfg_attr(feature = "binding-types", ts(as = "Option<String>", optional))]
    pub page_text_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "binding-types", ts(optional))]
    pub error: Option<String>,
}

pub(crate) fn result(
    input: &PdfLookupRequest,
    status: PdfLookupStatus,
    error: Option<String>,
) -> PdfStructureLookup {
    let mut requested = input.clone();
    if requested.end_locator.as_deref() == Some("") {
        requested.end_locator = None;
    }
    PdfStructureLookup {
        kind: lookup_kind(&input.locator_kind),
        requested,
        units: Vec::new(),
        before: Vec::new(),
        after: Vec::new(),
        matches: Vec::new(),
        pages: Vec::new(),
        status,
        payload_sha256: String::new(),
        page_text_sha256: String::new(),
        error,
    }
}

/// The kind of unit a locator asks for: a page, a transcript's line, a paragraph, a note, else a provision.
fn lookup_kind(locator_kind: &str) -> &'static str {
    match locator_kind {
        "page" => "page",
        "line" => "line",
        "paragraph" => "paragraph",
        "footnote" => "footnote",
        _ => "section",
    }
}

pub(crate) fn range_text(text: &ScalarText<'_>, ranges: &[ScalarRange]) -> String {
    ranges
        .iter()
        .filter_map(|range| text.slice_utf16(range.start..range.end))
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn rendered_slice(
    text: &ScalarText<'_>,
    document: &DocumentStructure,
    node: &StructureNode,
) -> String {
    range_text(text, &document.query_ranges(node))
}

/// A declared paragraph as a lookup unit: its first printing, with its other printings and the text of them all; trusted
/// as its first page's text is.
pub(crate) fn printed_unit(
    text: &ScalarText<'_>,
    document: &PdfDocument,
    (_, printings): &PrintedParagraph,
) -> PdfLookupUnit {
    let nodes: Vec<&StructureNode> = printings.iter().map(|&index| &document.structure.nodes[index]).collect();
    let mut ranges: Vec<ScalarRange> = nodes.iter().flat_map(|node| document.structure.query_ranges(node).into_owned()).collect();
    ScalarRange::normalize_union(&mut ranges);
    let first = nodes.iter().flat_map(|node| node.page_indexes.iter().copied()).min();
    let page = first.and_then(|index| document.page_node(index, false));
    let mut unit = PdfLookupUnit::new(nodes[0], range_text(text, &ranges),
        page.map(|page| page_confidence(&document.summary.page_text_quality, page)));
    unit.printings = nodes[1..].iter().map(|node| (*node).clone()).collect();
    unit
}

pub(crate) fn page_unit(document: &DocumentStructure, text: &ScalarText<'_>, quality: &[f64], page: &StructureNode) -> PdfLookupUnit {
    PdfLookupUnit::new(page, rendered_slice(text, document, page), Some(page_confidence(quality, page)))
}

/// A page's printed transcript lines in reading order: each line number the structure labels
/// (its `LineLabel` nodes), its node, and the words set across its row; a number on a
/// blank line is left out.
pub(crate) fn transcript_lines<'a>(
    document: &'a PdfDocument,
    page: &StructureNode,
    geometry: &Page,
) -> Vec<(usize, &'a StructureNode, String)> {
    let height = |rect: &[f64; 4]| (rect[3] - rect[1]).max(0.1);
    document
        .structure
        .nodes
        .iter()
        .filter(|node| {
            node.role == Some(legal_structure_model::PrimaryRole::LineLabel)
                && node.page_indexes.contains(&page_index(page))
        })
        .filter_map(|node| {
            let number = node.label.as_deref()?.trim().parse().ok()?;
            let label_id = node.line_ids.first()?;
            let label = geometry.lines.iter().find(|line| &line.id == label_id)?;
            let mut row = geometry
                .lines
                .iter()
                .filter(|line| {
                    line.id != label.id
                        && !line.exclude_from_body
                        && line.bbox[3].min(label.bbox[3]) - line.bbox[1].max(label.bbox[1])
                            > 0.5 * height(&line.bbox).min(height(&label.bbox))
                })
                .collect::<Vec<_>>();
            row.sort_by(|left, right| left.bbox[0].total_cmp(&right.bbox[0]));
            let text = row
                .iter()
                .map(|line| line.text.trim())
                .filter(|text| !text.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            (!text.is_empty()).then_some((number, node, text))
        })
        .collect()
}

/// The document's notes, in order.
pub(crate) fn notes(document: &PdfDocument) -> Vec<&StructureNode> {
    document.structure.nodes.iter().filter(|node| matches!(node.kind, NodeKind::Footnote | NodeKind::Endnote)).collect()
}

/// The page (counted from 1) whose text holds `offset`.
fn page_at(document: &PdfDocument, offset: usize) -> Option<u32> {
    document.structure.nodes.iter()
        .find(|node| node.kind == NodeKind::Page && node.range.start <= offset && offset < node.range.end)
        .and_then(|node| node.page_indexes.first())
        .map(|page| *page as u32 + 1)
}

/// A note as a lookup returns it: its body, the times its label was printed up to it, and the page its first call stands
/// on.
pub(crate) fn footnote_unit(text: &ScalarText<'_>, document: &PdfDocument, notes: &[&StructureNode], at: usize) -> PdfLookupUnit {
    let note = notes[at];
    let mut unit = PdfLookupUnit::new(note, rendered_slice(text, &document.structure, note).trim().to_owned(), None);
    unit.occurrence = Some(notes[..=at].iter().filter(|other| other.label == note.label).count());
    unit.reference_page = note.calls.first().and_then(|call| page_at(document, call.start));
    unit
}

/// A provision as a lookup returns it, trusted as the least trusted of its pages.
pub(crate) fn section_unit(document: &PdfDocument, text: &ScalarText<'_>, section: &StructureNode) -> PdfLookupUnit {
    // (A page's node, its last where a page has several.)
    let confidence = section.page_indexes.iter().filter_map(|&index| document.page_node(index, true))
        .map(|page| page_confidence(&document.summary.page_text_quality, page)).reduce(f64::min);
    PdfLookupUnit::new(section, rendered_slice(text, &document.structure, section).trim().to_owned(), confidence)
}

pub(crate) fn exact_footnotes(document: &PdfDocument, notes: &[&StructureNode], locator: &str, input: &PdfLookupRequest) -> Vec<usize> {
    let wanted = legal_locator::note_label(locator);
    (0..notes.len()).filter(|&at| {
        let note = notes[at];
        let label = note.label.as_deref().unwrap_or("");
        input.occurrence.is_none_or(|number| notes[..=at].iter().filter(|other| other.label.as_deref().unwrap_or("") == label).count() == number)
            && input.page.is_none_or(|number| note.calls.first().and_then(|call| page_at(document, call.start)) == Some(number)
                || note.page_indexes.iter().any(|page| *page as u32 + 1 == number))
            && note_named(note, &wanted)
    }).collect()
}

/// Both ends of a range found, the first not after the last.
fn ordered(start: Option<usize>, end: Option<usize>) -> Option<(usize, usize)> {
    start.zip(end).filter(|(start, end)| start <= end)
}

pub(crate) fn finish<F>(
    document: &PdfDocument,
    text: &ScalarText<'_>,
    input: &PdfLookupRequest,
    ordered_len: usize,
    selected_start: usize,
    selected_end: usize,
    mut unit_at: F,
) -> PdfStructureLookup
where
    F: FnMut(usize) -> PdfLookupUnit,
{
    if selected_end - selected_start + 1 > MAX_UNITS {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!("Exact ranges are limited to {MAX_UNITS} units")),
        );
    }
    // Each unit with the pages it stands on.
    let mut unit_at = |index| {
        let mut unit = unit_at(index);
        unit.pages = unit.page_numbers();
        unit.ranges = document.structure.query_ranges(&unit.node).into_owned();
        unit
    };
    let before: Vec<_> = (selected_start.saturating_sub(input.context_blocks)..selected_start)
        .map(&mut unit_at)
        .collect();
    let selected: Vec<_> = (selected_start..=selected_end).map(&mut unit_at).collect();
    let after: Vec<_> = (selected_end + 1
        ..usize::min(ordered_len, selected_end + 1 + input.context_blocks))
        .map(unit_at)
        .collect();
    if selected.iter().any(|unit| unit.text.is_empty()) {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some("The requested structural unit has no exact text".to_owned()),
        );
    }
    if before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .map(|unit| utf16_len(&unit.text))
        .sum::<usize>()
        > MAX_RETURN_CHARS
    {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!(
                "Exact result exceeds {MAX_RETURN_CHARS} characters; request a narrower range"
            )),
        );
    }
    let page_numbers: HashSet<_> = before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .flat_map(|unit| unit.pages.iter().copied())
        .collect();
    // The text of each page a unit is on, as a page lookup returns it (`PdfDocument::page_texts`).
    let pages = document
        .page_nodes_numbered(page_numbers)
        .into_iter()
        .filter_map(|page| {
            let text = rendered_slice(text, &document.structure, page);
            (!text.is_empty()).then_some(PdfLookupPage { page_number: page_number(page), text })
        })
        .collect();
    let matches = selected.iter().map(|unit| unit.node.id.clone()).collect();
    let mut lookup = result(input, PdfLookupStatus::Found, None);
    lookup.units = selected;
    lookup.before = before;
    lookup.after = after;
    lookup.matches = matches;
    lookup.pages = pages;
    let payload_sha256 = lookup_payload_sha256(&lookup);
    let page_text_sha256 = serialization_sha256(&lookup.pages);
    lookup.payload_sha256 = payload_sha256;
    lookup.page_text_sha256 = page_text_sha256;
    lookup
}

pub(crate) fn structure_lookup(document: &PdfDocument,
    query: &DocumentQuery,
    input: &PdfLookupRequest,
) -> PdfStructureLookup {
    if !input.valid() {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some("Invalid or unbounded PDF locator".to_owned()),
        );
    }
    let kind = lookup_kind(&input.locator_kind);
    let text = query.text(&document.structure);
    let end_locator = input
        .end_locator
        .as_deref()
        .filter(|value| !value.is_empty());
    // The numbers a page, paragraph or line locator runs from and to, in order.
    let numbers = || {
        let inline = legal_locator::ordinal_range(kind, &input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| legal_locator::parse_ordinal(kind, &input.locator));
        let end_number = end_locator
            .map(|end| legal_locator::parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        match (start_number, end_number) {
            (Some(start), Some(end)) if start <= end => Ok((start, end)),
            _ => Err(result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            )),
        }
    };
    if kind == "line" {
        // A transcript's printed line numbers on one page ("p. 451, ll. 12-20"): each numbered
        // line that carries words, read across the row its number opens.
        let Some(page_number) = input.page else {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("A line locator needs its page".to_owned()),
            );
        };
        let (start_number, end_number) = match numbers() {
            Ok(numbers) => numbers,
            Err(invalid) => return invalid,
        };
        // The page a reader cites is the one printed with that number; a PDF whose pages print
        // none is read by its own page numbers.
        let printed = page_number.to_string();
        let page = document
            .page_nodes()
            .find(|page| page.aliases.as_ref().is_some_and(|aliases| aliases.iter().any(|alias| alias.trim() == printed)))
            .or_else(|| document.page_nodes().find(|page| self::page_number(page) == page_number));
        let passage = document.passage_pages();
        let Some((page, geometry)) = page.and_then(|page| Some((page, passage.get(page_index(page))?))) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        let lines = transcript_lines(document, page, geometry);
        if lines.is_empty() {
            return result(
                input,
                PdfLookupStatus::Unavailable,
                Some("This page prints no line numbers".to_owned()),
            );
        }
        let at = |number| lines.iter().position(|line| line.0 == number);
        let Some((start, end)) = ordered(at(start_number), at(end_number)) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        return finish(document, &text, input, lines.len(), start, end, |index| {
            let (_, node, words) = &lines[index];
            PdfLookupUnit::new(node, words.clone(), Some(page_confidence(&document.summary.page_text_quality, page)))
        });
    }
    if kind == "page" || kind == "paragraph" {
        let (start_number, end_number) = match numbers() {
            Ok(numbers) => numbers,
            Err(invalid) => return invalid,
        };
        if kind == "page" {
            let pages: Vec<&StructureNode> = document.page_nodes().collect();
            let position = |number| {
                u32::try_from(number)
                    .ok()
                    .and_then(|number| pages.iter().position(|page| page_number(page) == number))
            };
            let Some((start, end)) = ordered(position(start_number), position(end_number)) else {
                return result(input, PdfLookupStatus::NotFound, None);
            };
            return finish(document, &text, input, pages.len(), start, end, |index| {
                page_unit(&document.structure, &text, &document.summary.page_text_quality, pages[index])
            });
        }
        if end_number - start_number >= MAX_UNITS {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Paragraph range exceeds the lookup limit".to_owned()),
            );
        }
        let (start, end) = match document.paragraph_range(start_number, end_number) {
            Ok(range) => range,
            Err(status) => return result(input, status, None),
        };
        let printed = document.printed_paragraphs();
        return finish(document, &text, input, printed.len(), start, end, |index| {
            printed_unit(&text, document, &printed[index])
        });
    }
    if kind == "footnote" {
        let notes = notes(document);
        let start = exact_footnotes(document, &notes, &input.locator, input);
        let end = end_locator.map(|locator| exact_footnotes(document, &notes, locator, input));
        let end_matches = end.as_deref().unwrap_or(&start);
        if start.len() > 1 || end_matches.len() > 1 {
            let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
            for pair_id in start
                .iter()
                .chain(end.iter().flatten())
                .map(|index| notes[*index].id.as_str())
            {
                if !lookup.matches.iter().any(|item| item == pair_id) {
                    lookup.matches.push(pair_id.to_owned());
                }
            }
            return lookup;
        }
        let Some((start, end)) = ordered(start.first().copied(), end_matches.first().copied())
        else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        return finish(document, &text, input, notes.len(), start, end, |index| footnote_unit(&text, document, &notes, index));
    }
    // The provisions the locator names, as the query reads them for every document (`DocumentQuery::provisions`).
    let kind = input.locator_kind.as_str();
    match query.provisions(&document.structure, kind, &input.locator, end_locator, input.instrument.as_deref(), true) {
        Provisions::Unavailable => {
            let kind = if kind == "rule" { "section" } else { kind };
            result(input, PdfLookupStatus::Unavailable, Some(format!("No exact {kind} identifiers exist in this source PDF")))
        }
        Provisions::NotFound => result(input, PdfLookupStatus::NotFound, None),
        Provisions::Ambiguous(nodes) => {
            let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
            lookup.matches = nodes.into_iter().map(|index| document.structure.nodes[index].id.clone()).collect();
            lookup
        }
        Provisions::Found { order, first, last } => finish(document, &text, input, order.len(), first, last, |at| {
            section_unit(document, &text, &document.structure.nodes[order[at]])
        }),
    }
}
