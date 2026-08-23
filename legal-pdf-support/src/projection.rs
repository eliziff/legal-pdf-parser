use legal_pdf_core::model::{
    Derivation, Footnote, LegalDocument, NodeKind, Page, Paragraph, PdfSourceExtent, ScalarRange,
    StructureNode,
};
use legal_pdf_core::Result;
use legal_structure::{normalize_compact_numbered_section_locator, utf16_len, ScalarText};
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const MAX_UNITS: usize = 20;
const MAX_CONTEXT: usize = 2;
const MAX_RETURN_CHARS: usize = 60_000;
const LOCATOR_KINDS: [&str; 11] = [
    "page",
    "paragraph",
    "footnote",
    "section",
    "subsection",
    "provision_paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "schedule",
    "article",
];

#[derive(Clone, Serialize)]
struct Proposition {
    sentence: String,
    passage_since_prior_note: String,
}

#[derive(Clone, Serialize)]
struct Note {
    label: String,
    occurrence: usize,
    restart_sequence: usize,
    reference_page: Option<u32>,
    body_pages: Vec<u32>,
    warnings: Vec<String>,
}

#[derive(Serialize)]
struct LookupUnit {
    id: String,
    kind: String,
    locator: String,
    text: String,
    page_numbers: Vec<u32>,
    confidence: Option<f64>,
    confidence_basis: String,
    provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    proposition: Option<Proposition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<Note>,
}

struct LookupInput<'a> {
    locator_kind: &'a str,
    locator: &'a str,
    end_locator: Option<&'a str>,
    context: usize,
    page: Option<u32>,
    occurrence: Option<usize>,
    valid: bool,
}

fn optional_positive<T: TryFrom<u64>>(value: &Value, key: &str) -> (Option<T>, bool) {
    match value.get(key) {
        None | Some(Value::Null) => (None, true),
        Some(item) => match item
            .as_u64()
            .filter(|number| *number > 0)
            .and_then(|n| T::try_from(n).ok())
        {
            Some(number) => (Some(number), true),
            None => (None, false),
        },
    }
}

impl<'a> LookupInput<'a> {
    fn new(value: &'a Value) -> Self {
        let locator_kind = value
            .get("locator_kind")
            .and_then(Value::as_str)
            .unwrap_or("");
        let locator = value.get("locator").and_then(Value::as_str).unwrap_or("");
        let end_locator = value
            .get("end_locator")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty());
        let context_value = value.get("context_blocks");
        let context = context_value
            .and_then(Value::as_u64)
            .and_then(|n| usize::try_from(n).ok())
            .unwrap_or(0);
        let context_valid =
            context_value.is_none_or(|item| item.as_u64().is_some_and(|n| n <= MAX_CONTEXT as u64));
        let (page, page_valid) = optional_positive(value, "page");
        let (occurrence, occurrence_valid) = optional_positive(value, "occurrence");
        let valid = LOCATOR_KINDS.contains(&locator_kind)
            && !locator.trim().is_empty()
            && utf16_len(locator) <= 200
            && end_locator.is_none_or(|end| utf16_len(end) <= 200)
            && context_valid
            && page_valid
            && occurrence_valid;
        Self {
            locator_kind,
            locator,
            end_locator,
            context,
            page,
            occurrence,
            valid,
        }
    }

    fn requested(&self) -> Value {
        json!({
            "locator_kind": self.locator_kind,
            "locator": self.locator,
            "end_locator": self.end_locator.filter(|value| !value.is_empty()),
            "context_blocks": self.context,
            "page": self.page,
            "occurrence": self.occurrence,
        })
    }
}

fn base(input: &LookupInput<'_>) -> Value {
    json!({
        "schema_version": "legalpdf.structure-lookup.v1",
        "requested": input.requested(),
        "units": [],
        "before": [],
        "after": [],
        "matches": [],
        "pages": [],
    })
}

fn result(input: &LookupInput<'_>, status: &str, exact: bool, error: Option<String>) -> Value {
    let mut value = base(input);
    value["status"] = Value::String(status.to_owned());
    value["exact"] = Value::Bool(exact);
    if let Some(message) = error {
        value["error"] = Value::String(message);
    }
    value
}

fn clean_text(value: &str) -> Cow<'_, str> {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    let value = value.trim();
    if !value.contains("⟦FN:") {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(
            MARKER
                .get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
                .replace_all(value, "")
                .trim()
                .to_owned(),
        )
    }
}

fn join_page_lines(page: &Page) -> String {
    let mut output = String::new();
    let mut lines: Vec<_> = page.lines.iter().collect();
    lines.sort_by_key(|line| line.reading_order);
    for line in lines {
        let text = line.text.trim();
        if text.is_empty() {
            continue;
        }
        if output.ends_with('-') && text.chars().next().is_some_and(char::is_lowercase) {
            output.pop();
        } else if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(text);
    }
    output
}

fn normal_pages(document: &LegalDocument, selected: &HashSet<u32>) -> Vec<(u32, String)> {
    let mut pages: Vec<_> = document
        .pages
        .iter()
        .filter(|page| selected.contains(&page.number))
        .map(|page| (page.number, join_page_lines(page)))
        .filter(|(_, text)| !text.is_empty())
        .collect();
    pages.sort_by_key(|(number, _)| *number);
    pages
}

fn rendered_slice<'a>(text: &ScalarText<'a>, node: &StructureNode) -> &'a str {
    node.rendered_range
        .and_then(|range| text.slice_utf16(range.start..range.end))
        .unwrap_or_default()
}

fn section_nodes(document: &LegalDocument) -> Vec<&StructureNode> {
    document
        .structure_graph
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Section)
        .collect()
}

fn nodes_by_id(document: &LegalDocument) -> HashMap<&str, &StructureNode> {
    document
        .structure_graph
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect()
}

struct SourceParagraph<'a> {
    paragraph: &'a legal_pdf_core::model::Paragraph,
    text: Cow<'a, str>,
}

fn source_paragraphs(document: &LegalDocument) -> Vec<SourceParagraph<'_>> {
    let mut paragraphs_by_page: HashMap<usize, Vec<_>> = HashMap::new();
    for paragraph in &document.paragraphs {
        paragraphs_by_page
            .entry(paragraph.page_index)
            .or_default()
            .push(paragraph);
    }
    let mut pages: Vec<_> = document.pages.iter().collect();
    pages.sort_by_key(|page| page.index);
    let mut output = Vec::new();
    for page in pages {
        for paragraph in paragraphs_by_page.remove(&page.index).unwrap_or_default() {
            let text = clean_text(&paragraph.text);
            if !text.is_empty() {
                output.push(SourceParagraph { paragraph, text });
            }
        }
    }
    output
}

fn paragraph_unit(
    text: &ScalarText<'_>,
    nodes: &HashMap<&str, &StructureNode>,
    pages: &HashMap<usize, &Page>,
    paragraph: &Paragraph,
    number: usize,
) -> LookupUnit {
    let page = pages.get(&paragraph.page_index);
    let node = nodes.get(paragraph.id.as_str()).copied();
    LookupUnit {
        id: if paragraph.id.is_empty() {
            format!("paragraph-{number}")
        } else {
            paragraph.id.clone()
        },
        kind: "paragraph".to_owned(),
        locator: format!("paragraph {number}"),
        text: node
            .map(|node| rendered_slice(text, node).to_owned())
            .unwrap_or_default(),
        page_numbers: page.map_or_else(Vec::new, |item| vec![item.number]),
        confidence: page.map(|item| item.text_quality.clamp(0.0, 1.0)),
        confidence_basis: if page.is_some() {
            "page_text_quality"
        } else {
            "unavailable"
        }
        .to_owned(),
        provenance: format!(
            "legalpdf:{}",
            if paragraph.region_type.is_empty() {
                "unknown"
            } else {
                &paragraph.region_type
            }
        ),
        proposition: None,
        note: None,
    }
}

fn page_unit(
    text: &ScalarText<'_>,
    nodes: &HashMap<&str, &StructureNode>,
    page: &Page,
) -> LookupUnit {
    let text = nodes
        .get(page.id.as_str())
        .copied()
        .map(|node| rendered_slice(text, node).to_owned())
        .unwrap_or_default();
    LookupUnit {
        id: if page.id.is_empty() {
            format!("page-{}", page.number)
        } else {
            page.id.clone()
        },
        kind: "page".to_owned(),
        locator: format!("[page {}]", page.number),
        text,
        page_numbers: vec![page.number],
        confidence: Some(page.text_quality.clamp(0.0, 1.0)),
        confidence_basis: "page_text_quality".to_owned(),
        provenance: if page.source.is_empty() {
            "unknown".to_owned()
        } else {
            page.source.clone()
        },
        proposition: None,
        note: None,
    }
}

fn footnote_unit(note: &Footnote) -> LookupUnit {
    let mut page_numbers = Vec::new();
    if let Some(page) = note.reference_page {
        page_numbers.push(page);
    }
    for page in &note.body_pages {
        if !page_numbers.contains(page) {
            page_numbers.push(*page);
        }
    }
    LookupUnit {
        id: note.pair_id.clone(),
        kind: "footnote".to_owned(),
        locator: format!("footnote {}", note.label),
        text: note.body.trim().to_owned(),
        page_numbers,
        confidence: Some(note.confidence.clamp(0.0, 1.0)),
        confidence_basis: "footnote_pairing".to_owned(),
        provenance: if note.provenance.is_empty() {
            "unknown".to_owned()
        } else {
            note.provenance.clone()
        },
        proposition: Some(Proposition {
            sentence: note.sentence_proposition.trim().to_owned(),
            passage_since_prior_note: note.passage_since_prior_note.trim().to_owned(),
        }),
        note: Some(Note {
            label: note.label.clone(),
            occurrence: note.occurrence,
            restart_sequence: note.restart_sequence,
            reference_page: note.reference_page,
            body_pages: note.body_pages.clone(),
            warnings: note.warnings.clone(),
        }),
    }
}

fn section_unit(
    text: &ScalarText<'_>,
    extents: &HashMap<&str, &PdfSourceExtent>,
    pages: &HashMap<usize, &Page>,
    section: &StructureNode,
    index: usize,
) -> LookupUnit {
    let mut page_numbers = Vec::new();
    let mut confidence: Option<f64> = None;
    let page_indexes = extents
        .get(section.id.as_str())
        .map(|extent| extent.page_indexes.as_slice())
        .unwrap_or_default();
    for page_index in page_indexes {
        if let Some(page) = pages.get(page_index) {
            if !page_numbers.contains(&page.number) {
                page_numbers.push(page.number);
            }
            let quality = page.text_quality.clamp(0.0, 1.0);
            confidence = Some(confidence.map_or(quality, |value| value.min(quality)));
        }
    }
    LookupUnit {
        id: if section.id.is_empty() {
            format!("section-{}", index + 1)
        } else {
            section.id.clone()
        },
        kind: "section".to_owned(),
        locator: section
            .label
            .as_deref()
            .unwrap_or(&section.id)
            .trim()
            .to_owned(),
        text: rendered_slice(text, section).trim().to_owned(),
        page_numbers,
        confidence,
        confidence_basis: if confidence.is_some() {
            "minimum_page_text_quality"
        } else {
            "unavailable"
        }
        .to_owned(),
        provenance: match section.source {
            Derivation::Native => "native",
            Derivation::Heuristic => "legal-structure",
            Derivation::Model => "model",
        }
        .to_owned(),
        proposition: None,
        note: None,
    }
}

pub fn parse_ordinal(kind: &str, raw: &str) -> Option<usize> {
    static PAGE: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH: OnceLock<Regex> = OnceLock::new();
    let value: String = raw.nfkc().collect();
    let captures = if kind == "page" {
        PAGE.get_or_init(|| {
            Regex::new(r"(?i)^#?\s*\[?\s*(?:(?:pages?|pp?\.?)[\s:_=-]*)?0*(\d{1,6})\s*\]?$")
                .unwrap()
        })
        .captures(value.trim())
    } else {
        PARAGRAPH.get_or_init(|| Regex::new(r"(?i)^#?\s*(?:(?:paragraphs?|paras?|pars?|¶)\.?\s*)?(?:(?:paragraph|para|par)[\s:_=-]*)?0*(\d{1,6})$" ).unwrap()).captures(value.trim())
    }?;
    captures.get(1)?.as_str().parse().ok()
}

pub fn numeric_range(kind: &str, raw: &str) -> Option<(usize, usize)> {
    static RANGE: OnceLock<Regex> = OnceLock::new();
    static PAGE_PREFIX: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH_PREFIX: OnceLock<Regex> = OnceLock::new();
    let mut value: String = raw.nfkc().collect();
    value = value
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim()
        .to_owned();
    let prefix = if kind == "page" {
        PAGE_PREFIX.get_or_init(|| Regex::new(r"(?i)^(?:pages?|pp?\.?)[\s:_=-]*").unwrap())
    } else {
        PARAGRAPH_PREFIX
            .get_or_init(|| Regex::new(r"(?i)^(?:paragraphs?|paras?|pars?)\.?[\s:_=-]*").unwrap())
    };
    let stripped = prefix.replace(&value, "");
    let captures = RANGE
        .get_or_init(|| Regex::new(r"(?i)^(\d{1,6})\s*(?:-|–|—|\.\.|to)\s*(\d{1,6})$").unwrap())
        .captures(&stripped)?;
    Some((captures[1].parse().ok()?, captures[2].parse().ok()?))
}

fn normalize_footnote(raw: &str) -> String {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value: String = raw.nfkc().collect();
    PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:footnotes?|notes?|fn)\s*[#.:_-]?\s*").unwrap())
        .replace(value.trim(), "")
        .to_lowercase()
}

fn normalized_section(raw: &str) -> Option<String> {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value: String = raw.nfkc().collect();
    let compact: String = PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:ss?\.?|sections?)\s*").unwrap())
        .replace(value.trim(), "")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let normalized = normalize_compact_numbered_section_locator(&compact);
    (!normalized.is_empty()).then_some(normalized)
}

fn section_alias(raw: &str) -> String {
    normalized_section(raw).unwrap_or_else(|| {
        let value: String = raw.nfkc().collect();
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    })
}

fn section_matches(section: &StructureNode, requested_kind: &str, requested: &str) -> bool {
    if requested_kind != "section" && section.locator_kind.as_deref() != Some(requested_kind) {
        return false;
    }
    if requested_kind == "section"
        && !section.id.is_empty()
        && section_alias(&format!("section:{}", section.id)) == requested
    {
        return true;
    }
    section
        .label
        .iter()
        .map(String::as_str)
        .chain(section.aliases.iter().flatten().map(String::as_str))
        .any(|value| section_alias(value) == requested)
}

fn exact_footnotes(document: &LegalDocument, locator: &str, input: &LookupInput<'_>) -> Vec<usize> {
    let query = normalize_footnote(locator);
    document
        .footnotes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            (normalize_footnote(&note.pair_id) == query || normalize_footnote(&note.label) == query)
                && input
                    .occurrence
                    .is_none_or(|number| note.occurrence == number)
                && input.page.is_none_or(|number| {
                    note.reference_page == Some(number) || note.body_pages.contains(&number)
                })
        })
        .map(|(index, _)| index)
        .collect()
}

fn finish<F>(
    document: &LegalDocument,
    input: &LookupInput<'_>,
    ordered_len: usize,
    selected_start: usize,
    selected_end: usize,
    mut unit_at: F,
) -> Value
where
    F: FnMut(usize) -> LookupUnit,
{
    if selected_end - selected_start + 1 > MAX_UNITS {
        return result(
            input,
            "invalid",
            false,
            Some(format!("Exact ranges are limited to {MAX_UNITS} units")),
        );
    }
    let before: Vec<_> = (selected_start.saturating_sub(input.context)..selected_start)
        .map(&mut unit_at)
        .collect();
    let selected: Vec<_> = (selected_start..=selected_end).map(&mut unit_at).collect();
    let after: Vec<_> = (selected_end + 1
        ..usize::min(ordered_len, selected_end + 1 + input.context))
        .map(unit_at)
        .collect();
    if selected.iter().any(|unit| unit.text.is_empty()) {
        return result(
            input,
            "unavailable",
            false,
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
            "invalid",
            false,
            Some(format!(
                "Exact result exceeds {MAX_RETURN_CHARS} characters; request a narrower range"
            )),
        );
    }
    let page_numbers: HashSet<_> = before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .flat_map(|unit| unit.page_numbers.iter().copied())
        .collect();
    let pages: Vec<_> = normal_pages(document, &page_numbers)
        .into_iter()
        .map(|(number, text)| json!({"page_number": number, "text": text}))
        .collect();
    let matches: Vec<_> = selected.iter().map(|unit| unit.id.as_str()).collect();
    let mut value = base(input);
    value["status"] = Value::String("found".to_owned());
    value["exact"] = Value::Bool(true);
    value["units"] = serde_json::to_value(&selected).expect("lookup units serialize");
    value["before"] = serde_json::to_value(&before).expect("lookup units serialize");
    value["after"] = serde_json::to_value(&after).expect("lookup units serialize");
    value["matches"] = serde_json::to_value(matches).expect("lookup ids serialize");
    value["pages"] = Value::Array(pages);
    value
}

pub fn structure_lookup(document: &LegalDocument, request: &Value) -> Result<Value> {
    let input = LookupInput::new(request);
    if !input.valid {
        return Ok(result(
            &input,
            "invalid",
            false,
            Some("Invalid or unbounded PDF locator".to_owned()),
        ));
    }
    let kind = match input.locator_kind {
        "page" | "paragraph" | "footnote" => input.locator_kind,
        _ => "section",
    };
    if kind == "page" || kind == "paragraph" {
        let inline = numeric_range(kind, input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| parse_ordinal(kind, input.locator));
        let end_number = input
            .end_locator
            .map(|end| parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        let (Some(start_number), Some(end_number)) = (start_number, end_number) else {
            return Ok(result(
                &input,
                "invalid",
                false,
                Some("Invalid exact range".to_owned()),
            ));
        };
        if start_number > end_number {
            return Ok(result(
                &input,
                "invalid",
                false,
                Some("Invalid exact range".to_owned()),
            ));
        }
        if kind == "page" {
            let position = |number| {
                u32::try_from(number)
                    .ok()
                    .and_then(|number| document.pages.iter().position(|page| page.number == number))
            };
            let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
                return Ok(result(&input, "not_found", false, None));
            };
            if end < start {
                return Ok(result(&input, "not_found", false, None));
            }
            let text = ScalarText::new(document.structure_graph.query_text());
            let nodes = nodes_by_id(document);
            return Ok(finish(
                document,
                &input,
                document.pages.len(),
                start,
                end,
                |index| page_unit(&text, &nodes, &document.pages[index]),
            ));
        }
        let paragraphs = source_paragraphs(document);
        let position = |number: usize| {
            number
                .checked_sub(1)
                .filter(|index| *index < paragraphs.len())
        };
        let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
            return Ok(result(&input, "not_found", false, None));
        };
        let text = ScalarText::new(document.structure_graph.query_text());
        let nodes = nodes_by_id(document);
        let pages: HashMap<_, _> = document
            .pages
            .iter()
            .map(|page| (page.index, page))
            .collect();
        return Ok(finish(
            document,
            &input,
            paragraphs.len(),
            start,
            end,
            |index| paragraph_unit(&text, &nodes, &pages, paragraphs[index].paragraph, index + 1),
        ));
    }
    if kind == "footnote" {
        let start = exact_footnotes(document, input.locator, &input);
        let end = input
            .end_locator
            .map(|locator| exact_footnotes(document, locator, &input))
            .unwrap_or_else(|| start.clone());
        let mut matches = Vec::new();
        for pair_id in start
            .iter()
            .chain(&end)
            .map(|index| document.footnotes[*index].pair_id.as_str())
        {
            if !matches.contains(&pair_id) {
                matches.push(pair_id);
            }
        }
        if start.len() > 1 || end.len() > 1 {
            let mut value = result(&input, "ambiguous", false, None);
            value["matches"] = serde_json::to_value(matches)?;
            return Ok(value);
        }
        let (Some(start), Some(end)) = (start.first().copied(), end.first().copied()) else {
            return Ok(result(&input, "not_found", false, None));
        };
        if end < start {
            return Ok(result(&input, "not_found", false, None));
        }
        return Ok(finish(
            document,
            &input,
            document.footnotes.len(),
            start,
            end,
            |index| footnote_unit(&document.footnotes[index]),
        ));
    }
    if input.end_locator.is_some() {
        return Ok(result(
            &input,
            "invalid",
            false,
            Some("Section ranges are not supported by this document contract".to_owned()),
        ));
    }
    let sections = section_nodes(document);
    if input.locator_kind != "section"
        && !sections
            .iter()
            .any(|section| section.locator_kind.as_deref() == Some(input.locator_kind))
    {
        return Ok(result(
            &input,
            "unavailable",
            false,
            Some(format!(
                "No exact {} identifiers exist in this source PDF",
                input.locator_kind
            )),
        ));
    }
    let requested = section_alias(input.locator);
    let candidates: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section_matches(section, input.locator_kind, &requested))
        .map(|(index, _)| index)
        .collect();
    if candidates.len() > 1 {
        let mut value = result(&input, "ambiguous", false, None);
        value["matches"] = serde_json::to_value(
            candidates
                .into_iter()
                .map(|index| sections[index].id.as_str())
                .collect::<Vec<_>>(),
        )?;
        return Ok(value);
    }
    let Some(index) = candidates.first().copied() else {
        return Ok(result(&input, "not_found", false, None));
    };
    let text = ScalarText::new(document.structure_graph.query_text());
    let extents: HashMap<_, _> = document
        .pdf_source_map
        .nodes
        .iter()
        .map(|extent| (extent.id.as_str(), extent))
        .collect();
    let pages: HashMap<_, _> = document
        .pages
        .iter()
        .map(|page| (page.index, page))
        .collect();
    Ok(finish(
        document,
        &input,
        sections.len(),
        index,
        index,
        |index| section_unit(&text, &extents, &pages, sections[index], index),
    ))
}

fn append(text: &mut String, position: &mut usize, value: &str) {
    text.push_str(value);
    *position += utf16_len(value);
}

fn extend_range(range: Option<ScalarRange>, next: ScalarRange) -> Option<ScalarRange> {
    Some(range.map_or(next, |range| ScalarRange {
        start: range.start.min(next.start),
        end: range.end.max(next.end),
    }))
}

pub fn project_structure(document: &mut LegalDocument) {
    let mut text = String::new();
    let mut id_ranges = HashMap::new();
    let mut line_ranges = HashMap::new();
    let mut page_ranges: HashMap<usize, ScalarRange> = HashMap::new();
    let mut position = 0;
    for item in source_paragraphs(document) {
        if !text.is_empty() {
            append(&mut text, &mut position, "\n\n");
        }
        let start = position;
        append(&mut text, &mut position, &item.text);
        let range = ScalarRange {
            start,
            end: position,
        };
        if !item.paragraph.id.is_empty() {
            id_ranges.insert(item.paragraph.id.as_str(), range);
        }
        for line_id in &item.paragraph.line_ids {
            line_ranges.insert(line_id.as_str(), range);
        }
        page_ranges
            .entry(item.paragraph.page_index)
            .and_modify(|page| {
                page.start = page.start.min(range.start);
                page.end = page.end.max(range.end);
            })
            .or_insert(range);
    }
    for page in &document.pages {
        if !page.id.is_empty() {
            if let Some(range) = page_ranges.get(&page.index) {
                id_ranges.insert(page.id.as_str(), *range);
            }
        }
    }

    for note in &document.footnotes {
        let lines_range = note
            .body_line_ids
            .iter()
            .filter_map(|id| line_ranges.get(id.as_str()))
            .copied()
            .fold(None, extend_range);
        let range = if let Some(range) = lines_range {
            Some(range)
        } else {
            let body = clean_text(&note.body);
            if body.is_empty() {
                None
            } else if let Some(start_byte) = text.find(body.as_ref()) {
                let start = utf16_len(&text[..start_byte]);
                Some(ScalarRange {
                    start,
                    end: start + utf16_len(&body),
                })
            } else {
                if !text.is_empty() {
                    append(&mut text, &mut position, "\n\n");
                }
                let start = position;
                append(&mut text, &mut position, &body);
                Some(ScalarRange {
                    start,
                    end: position,
                })
            }
        };
        if let Some(range) = range {
            id_ranges.insert(note.pair_id.as_str(), range);
            for line_id in &note.body_line_ids {
                line_ranges.entry(line_id.as_str()).or_insert(range);
            }
        }
    }

    let extents = document
        .pdf_source_map
        .nodes
        .iter()
        .map(|extent| (extent.id.as_str(), extent))
        .collect::<HashMap<_, _>>();
    let note_ranges = document
        .structure_graph
        .notes
        .iter()
        .filter_map(|note| {
            id_ranges
                .get(note.id.as_str())
                .copied()
                .map(|range| (note.node_id.as_str(), range))
        })
        .collect::<HashMap<_, _>>();
    let rendered_ranges = document
        .structure_graph
        .nodes
        .iter()
        .map(|node| {
            let lines_range = extents
                .get(node.id.as_str())
                .into_iter()
                .flat_map(|extent| &extent.line_ids)
                .filter_map(|id| line_ranges.get(id.as_str()))
                .copied()
                .fold(None, extend_range);
            note_ranges
                .get(node.id.as_str())
                .copied()
                .or_else(|| id_ranges.get(node.id.as_str()).copied())
                .or(lines_range)
                .or_else(|| {
                    extents
                        .get(node.id.as_str())
                        .and_then(|extent| extent.page_indexes.first())
                        .and_then(|page| page_ranges.get(page))
                        .copied()
                })
        })
        .collect::<Vec<_>>();
    for (node, range) in document
        .structure_graph
        .nodes
        .iter_mut()
        .zip(rendered_ranges)
    {
        node.rendered_range = range;
    }
    document.structure_graph.revision = format!("{:x}", Sha256::digest(text.as_bytes()));
    document.structure_graph.rendered_text = Some(text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{
        DocumentStructure, Paragraph, ScalarRange, PARSER_VERSION, SCHEMA_VERSION,
    };
    use serde_json::Map;

    fn structure_graph(nodes: Vec<StructureNode>) -> DocumentStructure {
        serde_json::from_value(json!({
            "schema_version": "legalpdf.document-structure.v1",
            "document_id": "doc",
            "offset_unit": "utf16",
            "provider": "local-pdf",
            "revision": "00",
            "text": "",
            "text_sha256": "00",
            "source_sha256": "00",
            "scope": {"kind": "complete"},
            "origins": [],
            "nodes": nodes,
            "diagnostics": []
        }))
        .unwrap()
    }

    fn node(id: &str, kind: NodeKind) -> StructureNode {
        StructureNode {
            id: id.to_owned(),
            kind,
            range: ScalarRange { start: 0, end: 0 },
            rendered_range: None,
            origin_id: "test".to_owned(),
            source: Derivation::Native,
            label: None,
            locator_kind: None,
            aliases: None,
            parent_id: None,
            anchor: None,
            content_start: None,
            marker_range: None,
            page_indexes: Vec::new(),
            line_ids: Vec::new(),
            grammar: None,
            proof: None,
        }
    }

    #[test]
    fn invalid_lookup_is_bounded_without_guessing() {
        let mut document = LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![Page {
                id: "page-1".to_owned(),
                index: 0,
                number: 1,
                width: 612.0,
                height: 792.0,
                lines: vec![],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![
                Paragraph {
                    id: "marker".to_owned(),
                    page_index: 0,
                    region_type: "body".to_owned(),
                    text: "⟦FN:one⟧".to_owned(),
                    line_ids: vec![],
                    anchors: vec![],
                },
                Paragraph {
                    id: "p".to_owned(),
                    page_index: 0,
                    region_type: "body".to_owned(),
                    text: "text".to_owned(),
                    line_ids: vec![],
                    anchors: vec![],
                },
            ],
            footnotes: vec![],
            tables: vec![],
            images: vec![],
            structure_graph: structure_graph(vec![
                node("page-1", NodeKind::Page),
                node("p", NodeKind::Prose),
            ]),
            pdf_source_map: Default::default(),
            pairing_audit: None,
            diagnostics: vec![],
            repairs: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        };
        assert_eq!(
            structure_lookup(&document, &json!({"locator_kind":"page","locator":""})).unwrap()
                ["status"],
            "invalid"
        );
        project_structure(&mut document);
        assert_eq!(document.structure_graph.query_text(), "text");
        assert_eq!(
            document.structure_graph.revision,
            format!("{:x}", Sha256::digest(b"text"))
        );
        assert_eq!(
            document.structure_graph.nodes[1].rendered_range,
            Some(ScalarRange { start: 0, end: 4 })
        );
        let lookup = structure_lookup(
            &document,
            &json!({"locator_kind":"paragraph","locator":"par1"}),
        )
        .unwrap();
        assert_eq!(lookup["status"], "found");
        assert_eq!(lookup["units"][0]["id"], "p");
        assert_eq!(lookup["units"][0]["text"], "text");
        let page =
            structure_lookup(&document, &json!({"locator_kind":"page","locator":"1"})).unwrap();
        assert_eq!(page["status"], "found");
        assert_eq!(page["units"][0]["text"], "text");
    }

    #[test]
    fn section_ids_are_exact_lookup_locators() {
        let section = StructureNode {
            id: "section-000001".to_owned(),
            kind: NodeKind::Section,
            range: ScalarRange { start: 0, end: 18 },
            rendered_range: None,
            origin_id: "test".to_owned(),
            source: Derivation::Heuristic,
            label: Some("A heading too long to use as a trusted locator".to_owned()),
            locator_kind: None,
            aliases: None,
            parent_id: None,
            anchor: None,
            content_start: Some(0),
            marker_range: None,
            page_indexes: Vec::new(),
            line_ids: Vec::new(),
            grammar: Some("hierarchy".to_owned()),
            proof: None,
        };
        assert!(section_matches(
            &section,
            "section",
            &section_alias("section:section-000001"),
        ));
    }

    #[test]
    fn projection_does_not_invent_section_nodes() {
        let text = "😀".repeat(MAX_RETURN_CHARS / 2 + 1);
        let mut document = LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![Page {
                id: "page-1".to_owned(),
                index: 0,
                number: 1,
                width: 612.0,
                height: 792.0,
                lines: vec![],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![Paragraph {
                id: "heading".to_owned(),
                page_index: 0,
                region_type: "heading".to_owned(),
                text: text.clone(),
                line_ids: vec![],
                anchors: vec![],
            }],
            footnotes: vec![],
            tables: vec![],
            images: vec![],
            structure_graph: structure_graph(vec![node("heading", NodeKind::Heading)]),
            pdf_source_map: Default::default(),
            pairing_audit: None,
            diagnostics: vec![],
            repairs: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        };

        project_structure(&mut document);
        assert_eq!(document.structure_graph.query_text(), text);
        assert_eq!(document.structure_graph.nodes[0].kind, NodeKind::Heading);
        assert_eq!(
            document.structure_graph.nodes[0].rendered_range,
            Some(ScalarRange {
                start: 0,
                end: utf16_len(&text),
            })
        );
        let lookup = structure_lookup(
            &document,
            &json!({"locator_kind":"section","locator":"section:section-000001"}),
        )
        .unwrap();
        assert_eq!(lookup["status"], "not_found");
    }
}
