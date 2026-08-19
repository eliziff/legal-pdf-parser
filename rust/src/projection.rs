use crate::error::Result;
use crate::model::{LegalDocument, Page, Section};
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Value};
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

#[derive(Clone, Serialize)]
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

fn utf16_len(value: &str) -> usize {
    value.encode_utf16().count()
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

fn clean_text(value: &str) -> String {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER
        .get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
        .replace_all(value, "")
        .trim()
        .to_owned()
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

fn normal_pages(document: &LegalDocument) -> Vec<(u32, String)> {
    let mut pages: Vec<_> = document
        .pages
        .iter()
        .map(|page| (page.number, join_page_lines(page)))
        .filter(|(_, text)| !text.is_empty())
        .collect();
    pages.sort_by_key(|(number, _)| *number);
    pages
}

fn paragraph_units(document: &LegalDocument) -> Vec<LookupUnit> {
    let pages: HashMap<_, _> = document
        .pages
        .iter()
        .map(|page| (page.index, page))
        .collect();
    document
        .paragraphs
        .iter()
        .enumerate()
        .map(|(index, paragraph)| {
            let page = pages.get(&paragraph.page_index);
            LookupUnit {
                id: if paragraph.id.is_empty() {
                    format!("paragraph-{}", index + 1)
                } else {
                    paragraph.id.clone()
                },
                kind: "paragraph".to_owned(),
                locator: format!("paragraph {}", index + 1),
                text: clean_text(&paragraph.text),
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
        })
        .collect()
}

fn page_units(document: &LegalDocument) -> Vec<LookupUnit> {
    document
        .pages
        .iter()
        .map(|page| {
            let body = join_page_lines(page);
            LookupUnit {
                id: if page.id.is_empty() {
                    format!("page-{}", page.number)
                } else {
                    page.id.clone()
                },
                kind: "page".to_owned(),
                locator: format!("[page {}]", page.number),
                text: if body.is_empty() {
                    String::new()
                } else {
                    format!("[page {}]\n{body}", page.number)
                },
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
        })
        .collect()
}

fn footnote_units(document: &LegalDocument) -> Vec<LookupUnit> {
    document
        .footnotes
        .iter()
        .map(|note| {
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
        })
        .collect()
}

fn section_units(document: &LegalDocument) -> Vec<LookupUnit> {
    let pages: HashMap<_, _> = document
        .pages
        .iter()
        .map(|page| (page.index, page))
        .collect();
    document
        .sections
        .iter()
        .enumerate()
        .map(|(index, section)| {
            let mut page_numbers = Vec::new();
            let mut qualities = Vec::new();
            for page_index in &section.page_indexes {
                if let Some(page) = pages.get(page_index) {
                    if !page_numbers.contains(&page.number) {
                        page_numbers.push(page.number);
                    }
                    qualities.push(page.text_quality.clamp(0.0, 1.0));
                }
            }
            let confidence = qualities.into_iter().reduce(f64::min);
            LookupUnit {
                id: if section.id.is_empty() {
                    format!("section-{}", index + 1)
                } else {
                    section.id.clone()
                },
                kind: "section".to_owned(),
                locator: section.locator.trim().to_owned(),
                text: section.text.trim().to_owned(),
                page_numbers,
                confidence,
                confidence_basis: if confidence.is_some() {
                    "minimum_page_text_quality"
                } else {
                    "unavailable"
                }
                .to_owned(),
                provenance: if section.provenance.is_empty() {
                    "heading-region".to_owned()
                } else {
                    section.provenance.clone()
                },
                proposition: None,
                note: None,
            }
        })
        .collect()
}

pub(crate) fn parse_ordinal(kind: &str, raw: &str) -> Option<usize> {
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

pub(crate) fn numeric_range(kind: &str, raw: &str) -> Option<(usize, usize)> {
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
    static NUMERIC: OnceLock<Regex> = OnceLock::new();
    static ALPHA: OnceLock<Regex> = OnceLock::new();
    let value: String = raw.nfkc().collect();
    let compact: String = PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:ss?\.?|sections?)\s*").unwrap())
        .replace(value.trim(), "")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let valid = NUMERIC
        .get_or_init(|| {
            Regex::new(r"^\d{1,8}[A-Za-z]{0,3}(?:[.-]\d{1,8}[A-Za-z]{0,3}){0,3}(?:\([^)]+\))*$")
                .unwrap()
        })
        .is_match(&compact)
        || ALPHA
            .get_or_init(|| {
                Regex::new(r"^[A-Za-z]{1,3}(?:[.-][0-9A-Za-z]{1,8}){1,3}(?:\([^)]+\))*$").unwrap()
            })
            .is_match(&compact);
    valid.then(|| format!("sec{compact}"))
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

fn section_matches(section: &Section, requested_kind: &str, requested: &str) -> bool {
    if requested_kind != "section" && section.locator_kind.as_deref() != Some(requested_kind) {
        return false;
    }
    std::iter::once(section.locator.as_str())
        .chain(section.aliases.iter().map(String::as_str))
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

fn finish(
    document: &LegalDocument,
    input: &LookupInput<'_>,
    ordered: Vec<LookupUnit>,
    selected_start: usize,
    selected_end: usize,
) -> Value {
    let selected = ordered[selected_start..=selected_end].to_vec();
    if selected.len() > MAX_UNITS {
        return result(
            input,
            "invalid",
            false,
            Some(format!("Exact ranges are limited to {MAX_UNITS} units")),
        );
    }
    if selected.iter().any(|unit| unit.text.is_empty()) {
        return result(
            input,
            "unavailable",
            false,
            Some("The requested structural unit has no exact text".to_owned()),
        );
    }
    let before = ordered[selected_start.saturating_sub(input.context)..selected_start].to_vec();
    let after = ordered
        [selected_end + 1..usize::min(ordered.len(), selected_end + 1 + input.context)]
        .to_vec();
    if before
        .iter()
        .chain(&selected)
        .chain(&after)
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
        .chain(&selected)
        .chain(&after)
        .flat_map(|unit| unit.page_numbers.iter().copied())
        .collect();
    let pages: Vec<_> = normal_pages(document)
        .into_iter()
        .filter(|(number, _)| page_numbers.contains(number))
        .map(|(number, text)| json!({"page_number": number, "text": text}))
        .collect();
    let matches: Vec<_> = selected.iter().map(|unit| unit.id.clone()).collect();
    let mut value = base(input);
    value["status"] = Value::String("found".to_owned());
    value["exact"] = Value::Bool(true);
    value["units"] = serde_json::to_value(selected).expect("lookup units serialize");
    value["before"] = serde_json::to_value(before).expect("lookup units serialize");
    value["after"] = serde_json::to_value(after).expect("lookup units serialize");
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
        let ordered = if kind == "page" {
            page_units(document)
        } else {
            paragraph_units(document)
        };
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
        let position = |number: usize| {
            ordered.iter().position(|unit| {
                if kind == "page" {
                    unit.page_numbers.first().copied() == u32::try_from(number).ok()
                } else {
                    unit.locator == format!("paragraph {number}")
                }
            })
        };
        let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
            return Ok(result(&input, "not_found", false, None));
        };
        if end < start {
            return Ok(result(&input, "not_found", false, None));
        }
        return Ok(finish(document, &input, ordered, start, end));
    }
    if kind == "footnote" {
        let ordered = footnote_units(document);
        let start = exact_footnotes(document, input.locator, &input);
        let end = input
            .end_locator
            .map(|locator| exact_footnotes(document, locator, &input))
            .unwrap_or_else(|| start.clone());
        let mut matches = Vec::new();
        for pair_id in start
            .iter()
            .chain(&end)
            .map(|index| &document.footnotes[*index].pair_id)
        {
            if !matches.contains(pair_id) {
                matches.push(pair_id.clone());
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
        return Ok(finish(document, &input, ordered, start, end));
    }
    if input.end_locator.is_some() {
        return Ok(result(
            &input,
            "invalid",
            false,
            Some("Section ranges are not supported by this document contract".to_owned()),
        ));
    }
    if input.locator_kind != "section"
        && !document
            .sections
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
    let candidates: Vec<_> = document
        .sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section_matches(section, input.locator_kind, &requested))
        .map(|(index, _)| index)
        .collect();
    if candidates.len() > 1 {
        let mut value = result(&input, "ambiguous", false, None);
        value["matches"] = Value::Array(
            candidates
                .into_iter()
                .map(|index| Value::String(document.sections[index].id.clone()))
                .collect(),
        );
        return Ok(value);
    }
    let Some(index) = candidates.first().copied() else {
        return Ok(result(&input, "not_found", false, None));
    };
    Ok(finish(
        document,
        &input,
        section_units(document),
        index,
        index,
    ))
}

fn normalized_locator(kind: &str, value: &str) -> Option<String> {
    if kind == "section" {
        return normalized_section(value);
    }
    static FOOTNOTE: OnceLock<Regex> = OnceLock::new();
    let number = if kind == "footnote" {
        let normalized: String = value.nfkc().collect();
        FOOTNOTE
            .get_or_init(|| Regex::new(r"(?i)^(?:fn|footnotes?|notes?)?[\s#.]*(\d{1,5})$").unwrap())
            .captures(normalized.trim())?
            .get(1)?
            .as_str()
            .parse()
            .ok()?
    } else {
        parse_ordinal(kind, value)?
    };
    Some(match kind {
        "page" => format!("page{number}"),
        "paragraph" => format!("par{number}"),
        _ => format!("fn{number}"),
    })
}

fn append(text: &mut String, position: &mut usize, value: &str) {
    text.push_str(value);
    *position += utf16_len(value);
}

pub fn source_doc(document: &LegalDocument, id: Option<&str>, url: Option<&str>) -> Value {
    let mut text = String::new();
    let mut blocks = Vec::new();
    let mut offsets = HashMap::new();
    let mut position = 0;
    let mut paragraph_number = 0;
    let mut paragraphs_by_page: HashMap<usize, Vec<_>> = HashMap::new();
    for paragraph in &document.paragraphs {
        paragraphs_by_page
            .entry(paragraph.page_index)
            .or_default()
            .push(paragraph);
    }
    let mut pages: Vec<_> = document.pages.iter().collect();
    pages.sort_by_key(|page| page.index);
    for page in pages {
        if !text.is_empty() {
            append(&mut text, &mut position, "\n\n");
        }
        let printed = page
            .printed_label
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let display = printed.map_or_else(|| page.number.to_string(), str::to_owned);
        let page_start = position;
        append(&mut text, &mut position, &format!("[page {display}]\n"));
        for (order, paragraph) in paragraphs_by_page
            .get(&page.index)
            .into_iter()
            .flatten()
            .enumerate()
        {
            let body = clean_text(&paragraph.text);
            if body.is_empty() {
                continue;
            }
            if order > 0 {
                append(&mut text, &mut position, "\n\n");
            }
            let start = position;
            append(&mut text, &mut position, &body);
            let end = position;
            let anchor = if paragraph.id.is_empty() {
                format!("page-{}-paragraph-{}", page.number, order + 1)
            } else {
                paragraph.id.clone()
            };
            offsets.insert(anchor.clone(), (start, end));
            paragraph_number += 1;
            blocks.push(json!({"kind":"paragraph","label":format!("par{paragraph_number}"),"start":start,"end":end,"origin":"heuristic","anchor":anchor}));
        }
        let label =
            normalized_locator("page", &display).unwrap_or_else(|| format!("page{}", page.number));
        let distinct_label = display != page.number.to_string();
        let mut aliases = vec![Value::String(page.number.to_string())];
        if distinct_label {
            aliases.push(Value::String(display));
        }
        blocks.push(json!({"kind":"page","label":label,"start":page_start,"end":position,"origin":if printed.is_some(){"heuristic"}else{"native"},"anchor":format!("page={}",page.number),"aliases":aliases}));
    }
    for (order, section) in document.sections.iter().enumerate() {
        let Some(first) = section.paragraph_ids.first().and_then(|id| offsets.get(id)) else {
            continue;
        };
        let Some(last) = section.paragraph_ids.last().and_then(|id| offsets.get(id)) else {
            continue;
        };
        if first.0 >= last.1 {
            continue;
        }
        let id = if section.id.is_empty() {
            format!("section-{}", order + 1)
        } else {
            section.id.clone()
        };
        let label = normalized_locator("section", section.locator.trim())
            .unwrap_or_else(|| format!("section:{id}"));
        let mut seen = HashSet::new();
        let aliases: Vec<_> = std::iter::once(section.locator.as_str())
            .chain(section.aliases.iter().map(String::as_str))
            .filter(|value| !value.is_empty() && seen.insert((*value).to_owned()))
            .map(|value| Value::String(value.to_owned()))
            .collect();
        blocks.push(json!({"kind":"section","label":label,"start":first.0,"end":last.1,"origin":if section.provenance.to_lowercase()=="native"{"native"}else{"heuristic"},"anchor":id,"aliases":aliases}));
    }
    for (order, note) in document.footnotes.iter().enumerate() {
        let body = clean_text(&note.body);
        if body.is_empty() {
            continue;
        }
        append(
            &mut text,
            &mut position,
            &format!(
                "\n\n[footnote {}]\n",
                if note.label.trim().is_empty() {
                    (order + 1).to_string()
                } else {
                    note.label.trim().to_owned()
                }
            ),
        );
        let start = position;
        append(&mut text, &mut position, &body);
        let end = position;
        let label = normalized_locator("footnote", note.label.trim())
            .unwrap_or_else(|| format!("fn{}", order + 1));
        let aliases: Vec<_> = [&note.label, &note.pair_id]
            .into_iter()
            .filter(|value| !value.is_empty())
            .map(|value| Value::String(value.clone()))
            .collect();
        let mut block = json!({"kind":"footnote","label":label,"start":start,"end":end,"origin":"heuristic","aliases":aliases});
        if !note.pair_id.is_empty() {
            block["anchor"] = Value::String(note.pair_id.clone());
        }
        blocks.push(block);
    }
    blocks.sort_by(|left, right| {
        left["start"]
            .as_u64()
            .cmp(&right["start"].as_u64())
            .then_with(|| left["end"].as_u64().cmp(&right["end"].as_u64()))
            .then_with(|| left["label"].as_str().cmp(&right["label"].as_str()))
    });
    json!({
        "schema_version": "legalpdf.source-doc.v1",
        "source_doc": {
            "provider": "local-pdf",
            "id": id.unwrap_or(&document.document_id),
            "url": url,
            "text": text,
            "blocks": blocks,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Paragraph, PARSER_VERSION, SCHEMA_VERSION};
    use serde_json::Map;

    #[test]
    fn invalid_lookup_is_bounded_without_guessing() {
        let document = LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 0,
            status: "ready".to_owned(),
            pages: vec![],
            paragraphs: vec![Paragraph {
                id: "p".to_owned(),
                page_index: 0,
                region_type: "body".to_owned(),
                text: "text".to_owned(),
                line_ids: vec![],
                anchors: vec![],
            }],
            sections: vec![],
            footnotes: vec![],
            tables: vec![],
            images: vec![],
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
        assert_eq!(
            source_doc(&document, Some("id"), None)["source_doc"]["id"],
            "id"
        );
    }
}
