#[cfg(feature = "structure-inference")]
use crate::{
    derive_structure_evidence, javascript_whitespace, DetectionProfile, DocumentInput, EngineError,
    NodeKind, StructureGraphV2,
};
#[cfg(feature = "structure-inference")]
use regex::Regex;
use serde::{Deserialize, Serialize};
#[cfg(feature = "structure-inference")]
use std::collections::{HashMap, HashSet};
#[cfg(feature = "structure-inference")]
use std::sync::OnceLock;

#[cfg(feature = "structure-inference")]
fn split_instrument_space_runs(text: &str) -> String {
    let characters = text.chars().collect::<Vec<_>>();
    let mut recovered = String::with_capacity(text.len());
    let mut index = 0;
    while index < characters.len() {
        if matches!(characters[index], ' ' | '\t') {
            let start = index;
            while index < characters.len() && matches!(characters[index], ' ' | '\t') {
                index += 1;
            }
            let internal_run = index - start >= 2
                && start > 0
                && index < characters.len()
                && !javascript_whitespace(characters[start - 1])
                && !javascript_whitespace(characters[index]);
            for (offset, character) in characters[start..index].iter().enumerate() {
                recovered.push(if internal_run && offset == 0 {
                    '\n'
                } else {
                    *character
                });
            }
            continue;
        }
        recovered.push(characters[index]);
        index += 1;
    }
    recovered
}

#[cfg(feature = "structure-inference")]
fn split_instrument_sentence_joins(text: &str) -> String {
    static HEAD: OnceLock<Regex> = OnceLock::new();
    let head = HEAD.get_or_init(|| {
        Regex::new(
            r"^(?:(?:ARTICLE|Article|PART|Part|DIVISION|Division|Section|SECTION|SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)[\s\u{feff}]+[IVXLCDM0-9]|[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})*[\s\u{feff}]+\S|\([A-Za-z0-9_]{1,3}\)[\s\u{feff}])",
        )
        .expect("valid instrument sentence-join grammar")
    });
    let positions = text.char_indices().collect::<Vec<_>>();
    let characters = positions
        .iter()
        .map(|(_, character)| *character)
        .collect::<Vec<_>>();
    let mut recovered = String::with_capacity(text.len());
    for (index, (byte, character)) in positions.iter().copied().enumerate() {
        let preceded_by_terminator = index > 0
            && (matches!(characters[index - 1], '.' | ';' | ':')
                || (matches!(
                    characters[index - 1],
                    ')' | ']' | '"' | '\'' | '\u{201d}' | '\u{2019}' | '\u{00bb}'
                ) && index > 1
                    && matches!(characters[index - 2], '.' | ';' | ':')));
        let after = byte + character.len_utf8();
        if matches!(character, ' ' | '\t')
            && preceded_by_terminator
            && head.is_match(&text[after..])
        {
            recovered.push('\n');
        } else {
            recovered.push(character);
        }
    }
    recovered
}

/// Offset-preserving lineation hypotheses used by the instrument structure profile.
/// The source lineation is first, so downstream selection keeps it on a tie.
#[cfg(feature = "structure-inference")]
pub fn instrument_lineation_hypotheses(text: &str) -> Vec<String> {
    let joined = split_instrument_sentence_joins(text);
    let hypotheses = [
        text.to_owned(),
        split_instrument_space_runs(text),
        joined.clone(),
        split_instrument_space_runs(&joined),
    ];
    let mut unique = Vec::with_capacity(hypotheses.len());
    for hypothesis in hypotheses {
        if !unique.contains(&hypothesis) {
            unique.push(hypothesis);
        }
    }
    unique
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
pub struct InstrumentReferenceEvidence {
    pub key: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentContentsEntry {
    pub label: String,
    pub display: String,
    pub heading: String,
    pub depth: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent_label: Option<String>,
    pub page: Option<u32>,
    pub contents_line_start: usize,
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InstrumentContentsOutline {
    pub entries: Vec<InstrumentContentsEntry>,
    pub region_start: usize,
    pub region_end: usize,
    pub pages_cited: usize,
}

#[derive(Clone, Debug, Serialize, Eq, PartialEq)]
pub struct InstrumentContentsReading {
    pub outline: Option<InstrumentContentsOutline>,
    pub refusal: Option<String>,
}

// Contents entries advertise provision labels and printed pages; they are not
// provision spans and never enter the detected node inventory.
#[cfg(feature = "structure-inference")]
const CONTENTS_MAX_ENTRY_GAP_UTF16: usize = 400;
// Measured entry gaps were 28-176 UTF-16 units across the accepted corpus;
// 200, 400, and 800 produced identical outlines on all 124 agreement texts.
#[cfg(feature = "structure-inference")]
const CONTENTS_WINDOW_UTF16: usize = 80_000;
#[cfg(feature = "structure-inference")]
const CONTENTS_MAX_ANCHORS: usize = 4;
#[cfg(feature = "structure-inference")]
const MIN_CONTENTS_ENTRIES: usize = 5;
// Accepted contents regions cite pages on 84-100% of their entries.
#[cfg(feature = "structure-inference")]
const MIN_CONTENTS_PAGE_SHARE: f64 = 0.6;
// A short pageless exhibits tail is valid; a continuing body walk is not.
#[cfg(feature = "structure-inference")]
const MAX_PAGELESS_RUN: usize = 3;

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug)]
enum InstrumentContentsHeadKind {
    Container { word: String, value: String },
    Schedule { word: String, value: String },
    Section { number: String },
}

#[cfg(feature = "structure-inference")]
#[derive(Clone, Debug)]
struct InstrumentContentsHead {
    start_byte: usize,
    end_byte: usize,
    start_utf16: usize,
    end_utf16: usize,
    kind: InstrumentContentsHeadKind,
}

#[cfg(feature = "structure-inference")]
fn utf16_boundaries(text: &str) -> Vec<(usize, usize)> {
    let mut boundaries = Vec::new();
    let mut utf16 = 0;
    boundaries.push((0, 0));
    for (byte, character) in text.char_indices() {
        utf16 += character.len_utf16();
        boundaries.push((byte + character.len_utf8(), utf16));
    }
    boundaries
}

#[cfg(feature = "structure-inference")]
fn utf16_at_byte(boundaries: &[(usize, usize)], byte: usize) -> usize {
    boundaries
        .binary_search_by_key(&byte, |(at, _)| *at)
        .map(|index| boundaries[index].1)
        .unwrap_or_else(|index| boundaries[index.saturating_sub(1)].1)
}

#[cfg(feature = "structure-inference")]
fn byte_at_utf16(boundaries: &[(usize, usize)], utf16: usize) -> usize {
    boundaries
        .binary_search_by_key(&utf16, |(_, at)| *at)
        .map(|index| boundaries[index].0)
        .unwrap_or_else(|index| boundaries[index.min(boundaries.len() - 1)].0)
}

#[cfg(feature = "structure-inference")]
fn normalize_javascript_whitespace(value: &str) -> String {
    let mut normalized = String::with_capacity(value.len());
    let mut separating = false;
    for character in value.chars() {
        if javascript_whitespace(character) {
            separating = !normalized.is_empty();
        } else {
            if separating {
                normalized.push(' ');
            }
            normalized.push(character);
            separating = false;
        }
    }
    normalized
}

#[cfg(feature = "structure-inference")]
fn instrument_contents_anchors(text: &str) -> Vec<(usize, usize)> {
    static TABLE: OnceLock<Regex> = OnceLock::new();
    static BARE: OnceLock<Regex> = OnceLock::new();
    let table = TABLE.get_or_init(|| {
        Regex::new(r"(?i:TABLE[ \t]+OF[ \t]+CONTENTS)")
            .expect("valid instrument contents anchor grammar")
    });
    let bare = BARE.get_or_init(|| {
        Regex::new(r"(?i)^(?:CONTENTS|INDEX)$")
            .expect("valid bare instrument contents anchor grammar")
    });
    let mut anchors = Vec::with_capacity(CONTENTS_MAX_ANCHORS);
    let mut start = 0;
    loop {
        let end = text[start..]
            .find(['\r', '\n'])
            .map_or(text.len(), |length| start + length);
        let line = &text[start..end];
        for found in table.find_iter(line) {
            let found_start = start + found.start();
            let found_end = start + found.end();
            let before = text[..found_start].chars().next_back();
            let after = text[found_end..].chars().next();
            if before.is_none_or(|character| matches!(character, '\r' | '\n' | '\t' | ' '))
                && after.is_none_or(|character| matches!(character, '\r' | '\n' | '\t' | ' '))
            {
                anchors.push((found_start, found_end));
                if anchors.len() == CONTENTS_MAX_ANCHORS {
                    break;
                }
            }
        }
        if anchors.len() == CONTENTS_MAX_ANCHORS {
            break;
        }
        let core = line.trim_matches([' ', '\t']);
        if bare.is_match(core) {
            anchors.push((start, end));
            if anchors.len() == CONTENTS_MAX_ANCHORS {
                break;
            }
        }
        if end == text.len() {
            break;
        }
        start = end + text[end..].chars().next().unwrap().len_utf8();
    }
    anchors
        .into_iter()
        .map(|(_, end)| (end, text[..end].encode_utf16().count()))
        .collect()
}

#[cfg(feature = "structure-inference")]
fn utf8_prefix_for_utf16(text: &str, limit: usize) -> usize {
    let mut utf16 = 0;
    for (byte, character) in text.char_indices() {
        if utf16 + character.len_utf16() > limit {
            return byte;
        }
        utf16 += character.len_utf16();
    }
    text.len()
}

#[cfg(feature = "structure-inference")]
fn instrument_contents_heads(
    region: &str,
    boundaries: &[(usize, usize)],
) -> Vec<InstrumentContentsHead> {
    // Heads, not line breaks, delimit entries because source formats variously
    // pack entries, preserve spacing, or break them mid-entry. Schedule-like
    // heads require a line boundary or two spaces because their vocabulary
    // also appears inside entry titles.
    static HEAD: OnceLock<Regex> = OnceLock::new();
    let head = HEAD.get_or_init(|| {
        Regex::new(
            r"(?:(?P<container>ARTICLE|Article|PART|Part|DIVISION|Division)[ \t]+(?P<container_value>[IVXLCDM]{1,7}|[0-9]{1,3})[.:]?|(?P<schedule>SCHEDULE|Schedule|EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix)[ \t]+(?P<schedule_value>[A-Z0-9][A-Za-z0-9_.-]{0,12}?)[.:]?|(?P<section_word>Section|SECTION)[ \t]+(?P<section>[0-9]{1,3}(?:\.[0-9]{1,3})*[A-Za-z]?)[.)]?|(?P<decimal>[0-9]{1,3}\.[0-9]{1,3}(?:\.[0-9]{1,3})*)[.)]?|(?P<integer>[0-9]{1,3})[.)])(?P<trail>[ \t\r\n]|$)",
        )
        .expect("valid instrument contents head grammar")
    });
    let mut heads = Vec::new();
    let mut search = 0;
    while search <= region.len() {
        let Some(found) = head.captures_at(region, search) else {
            break;
        };
        let whole = found.get(0).expect("contents head match");
        let trail = found.name("trail").expect("contents head trail");
        let body = found
            .name("container")
            .or_else(|| found.name("schedule"))
            .or_else(|| found.name("section_word"))
            .or_else(|| found.name("decimal"))
            .or_else(|| found.name("integer"))
            .expect("contents head body");
        let before = region[..body.start()].chars().next_back();
        let valid_lead = body.start() == 0 || before.is_some_and(javascript_whitespace);
        let schedule_lead = || {
            let mut before = region[..body.start()].chars().rev();
            match before.next() {
                Some('\r' | '\n') => true,
                Some(' ' | '\t') => before
                    .next()
                    .is_some_and(|value| matches!(value, ' ' | '\t')),
                _ => false,
            }
        };
        let kind = if !valid_lead {
            None
        } else if let (Some(word), Some(value)) =
            (found.name("container"), found.name("container_value"))
        {
            Some(InstrumentContentsHeadKind::Container {
                word: word.as_str().to_owned(),
                value: value.as_str().to_owned(),
            })
        } else if let (Some(word), Some(value)) =
            (found.name("schedule"), found.name("schedule_value"))
        {
            schedule_lead().then(|| InstrumentContentsHeadKind::Schedule {
                word: word.as_str().to_owned(),
                value: value.as_str().to_owned(),
            })
        } else {
            found
                .name("section")
                .or_else(|| found.name("decimal"))
                .or_else(|| found.name("integer"))
                .map(|number| InstrumentContentsHeadKind::Section {
                    number: number.as_str().to_owned(),
                })
        };
        if let Some(kind) = kind {
            let start_byte = body.start();
            heads.push(InstrumentContentsHead {
                start_byte,
                end_byte: trail.start(),
                start_utf16: utf16_at_byte(boundaries, start_byte),
                end_utf16: utf16_at_byte(boundaries, trail.start()),
                kind,
            });
            search = trail.start();
        } else {
            search = body.start() + body.as_str().chars().next().unwrap().len_utf8();
        }
        if search >= region.len() {
            break;
        }
        if search <= whole.start() {
            search = whole.end();
        }
    }
    heads
}

#[cfg(feature = "structure-inference")]
fn instrument_contents_unit_end(value: &str) -> Option<usize> {
    // Printed page footers occur between blank lines inside contents pages;
    // absorbing one as an entry page creates a false page decrease.
    let mut search = 0;
    while let Some(relative) = value[search..].find('\n') {
        let newline = search + relative;
        let mut start = newline;
        for (byte, character) in value[..newline].char_indices().rev() {
            if character == '\n' || !javascript_whitespace(character) {
                break;
            }
            start = byte;
        }

        let after_newline = newline + 1;
        let mut closes_blank_line = false;
        for character in value[after_newline..].chars() {
            if character == '\n' {
                closes_blank_line = true;
                break;
            }
            if !javascript_whitespace(character) {
                break;
            }
        }
        if closes_blank_line {
            return Some(start);
        }
        search = after_newline;
    }
    None
}

#[cfg(feature = "structure-inference")]
fn instrument_roman_value(value: &str) -> Option<u32> {
    let values = |character| match character {
        'I' => Some(1),
        'V' => Some(5),
        'X' => Some(10),
        'L' => Some(50),
        'C' => Some(100),
        'D' => Some(500),
        'M' => Some(1000),
        _ => None,
    };
    let characters = value.chars().collect::<Vec<_>>();
    let mut total = 0i32;
    for (index, character) in characters.iter().enumerate() {
        let current = values(*character)?;
        let next = characters
            .get(index + 1)
            .and_then(|next| values(*next))
            .unwrap_or(0);
        total += if current < next { -current } else { current };
    }
    u32::try_from(total).ok()
}

#[cfg(feature = "structure-inference")]
fn instrument_contents_region(
    text: &str,
    from_byte: usize,
    from_utf16: usize,
) -> Option<InstrumentContentsOutline> {
    let suffix = &text[from_byte..];
    let region_end = utf8_prefix_for_utf16(suffix, CONTENTS_WINDOW_UTF16);
    let region = &suffix[..region_end];
    let region_boundaries = utf16_boundaries(region);
    let region_utf16 = region_boundaries.last().map_or(0, |(_, utf16)| *utf16);
    let heads = instrument_contents_heads(region, &region_boundaries);
    if heads.is_empty() || heads[0].start_utf16 > CONTENTS_MAX_ENTRY_GAP_UTF16 {
        return None;
    }

    let mut entries = Vec::new();
    let mut by_label: HashMap<String, InstrumentContentsEntry> = HashMap::new();
    let mut container: Option<String> = None;
    let mut previous_page = 0;
    let mut pageless = 0;
    let mut pageless_from = 0;
    let mut last_head: Option<usize> = None;
    for (index, head) in heads.iter().enumerate() {
        if index > 0 && head.start_utf16 - heads[index - 1].end_utf16 > CONTENTS_MAX_ENTRY_GAP_UTF16
        {
            break;
        }
        let until_byte = if heads
            .get(index + 1)
            .is_some_and(|next| next.start_utf16 - head.end_utf16 <= CONTENTS_MAX_ENTRY_GAP_UTF16)
        {
            heads[index + 1].start_byte
        } else {
            byte_at_utf16(&region_boundaries, (head.end_utf16 + 200).min(region_utf16))
        };
        let raw = &region[head.end_byte..until_byte];
        let raw = instrument_contents_unit_end(raw).map_or(raw, |cut| &raw[..cut]);
        let unit = normalize_javascript_whitespace(raw);
        let page_match_start = unit.rfind(' ').map_or(0, |space| space);
        let page_token = if page_match_start == 0 {
            unit.as_str()
        } else {
            &unit[page_match_start + 1..]
        };
        let page = (!page_token.is_empty()
            && page_token.len() <= 4
            && page_token.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| page_token.parse::<u32>().ok())
        .flatten();
        if page.is_some_and(|page| page < previous_page) {
            break;
        }

        let (label, display, depth, parent_label, is_container) = match &head.kind {
            InstrumentContentsHeadKind::Container { word, value } => {
                let number = value
                    .parse::<u32>()
                    .ok()
                    .or_else(|| instrument_roman_value(&value.to_ascii_uppercase()));
                let Some(number) = number else { continue };
                let lower = word.to_ascii_lowercase();
                let prefix = match lower.as_str() {
                    "article" => "art",
                    "part" => "part",
                    _ => "div",
                };
                (
                    format!("{prefix}{number}"),
                    format!("{} {value}", word.to_ascii_uppercase()),
                    0,
                    None,
                    true,
                )
            }
            InstrumentContentsHeadKind::Schedule { word, value } => {
                let prefix = match word.to_ascii_lowercase().as_str() {
                    "schedule" => "sched",
                    "exhibit" => "exh",
                    "annex" => "annex",
                    _ => "app",
                };
                (
                    format!("{prefix}{}", value.to_ascii_lowercase()),
                    format!("{} {value}", word.to_ascii_uppercase()),
                    0,
                    None,
                    true,
                )
            }
            InstrumentContentsHeadKind::Section { number } => {
                let numbered_parent = number.rfind('.').and_then(|dot| {
                    by_label
                        .get(&format!("sec{}", &number[..dot]))
                        .map(|entry| (entry.label.clone(), entry.depth + 1))
                });
                let (parent, depth) = numbered_parent
                    .map(|(parent, depth)| (Some(parent), depth))
                    .unwrap_or_else(|| (container.clone(), usize::from(container.is_some())));
                (
                    format!("sec{number}"),
                    format!("Section {number}"),
                    depth,
                    parent,
                    false,
                )
            }
        };
        if by_label.contains_key(&label) {
            break;
        }
        if page.is_none() {
            if pageless == 0 {
                pageless_from = entries.len();
            }
            pageless += 1;
            if pageless > MAX_PAGELESS_RUN {
                entries.truncate(pageless_from);
                break;
            }
        } else {
            pageless = 0;
            previous_page = page.unwrap();
        }
        if is_container {
            container = Some(label.clone());
        }
        let heading_source = page.map_or(unit.as_str(), |_| &unit[..page_match_start]);
        let heading = heading_source
            .trim_end_matches(|character: char| {
                character == '.' || character == '\u{2026}' || javascript_whitespace(character)
            })
            .trim_start_matches(|character: char| {
                javascript_whitespace(character)
                    || matches!(character, '\u{2013}' | '\u{2014}' | '-' | ':' | '.')
            })
            .trim_matches(javascript_whitespace)
            .to_owned();
        let entry = InstrumentContentsEntry {
            label: label.clone(),
            display,
            heading,
            depth,
            parent_label,
            page,
            contents_line_start: from_utf16 + head.start_utf16,
        };
        by_label.insert(label, entry.clone());
        entries.push(entry);
        last_head = Some(index);
    }
    if entries.is_empty() {
        return None;
    }
    let last_head = last_head?;
    Some(InstrumentContentsOutline {
        pages_cited: entries.iter().filter(|entry| entry.page.is_some()).count(),
        region_start: from_utf16 + heads[0].start_utf16,
        region_end: from_utf16 + heads[last_head].end_utf16,
        entries,
    })
}

/// Read a document's own table of contents as a page-addressed outline. The
/// outline never claims provision spans; ambiguous inputs receive a typed refusal.
#[cfg(feature = "structure-inference")]
pub fn instrument_contents_outline(text: &str) -> InstrumentContentsReading {
    let anchors = instrument_contents_anchors(text);
    if anchors.is_empty() {
        return InstrumentContentsReading {
            outline: None,
            refusal: Some("no_contents_marker".to_owned()),
        };
    }
    let mut refusal = "no_contents_entries";
    for (from_byte, from_utf16) in anchors {
        let Some(outline) = instrument_contents_region(text, from_byte, from_utf16) else {
            continue;
        };
        if outline.entries.len() < MIN_CONTENTS_ENTRIES {
            refusal = "too_few_contents_entries";
            continue;
        }
        if outline.pages_cited as f64 / (outline.entries.len() as f64) < MIN_CONTENTS_PAGE_SHARE {
            refusal = "contents_without_page_numbers";
            continue;
        }
        return InstrumentContentsReading {
            outline: Some(outline),
            refusal: None,
        };
    }
    InstrumentContentsReading {
        outline: None,
        refusal: Some(refusal.to_owned()),
    }
}

#[cfg(feature = "structure-inference")]
fn instrument_roman(mut value: usize) -> String {
    let mut result = String::new();
    for (amount, numeral) in [
        (1000, "m"),
        (900, "cm"),
        (500, "d"),
        (400, "cd"),
        (100, "c"),
        (90, "xc"),
        (50, "l"),
        (40, "xl"),
        (10, "x"),
        (9, "ix"),
        (5, "v"),
        (4, "iv"),
        (1, "i"),
    ] {
        while value >= amount {
            result.push_str(numeral);
            value -= amount;
        }
    }
    result
}

#[cfg(feature = "structure-inference")]
fn instrument_reference_index(
    graph: &StructureGraphV2,
    scalar_to_utf16: &[usize],
) -> Result<HashMap<String, usize>, EngineError> {
    let mut index = HashMap::new();
    let mut duplicates = HashSet::new();
    for node in graph
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Section)
    {
        let Some(label) = node.label.as_deref() else {
            continue;
        };
        let start = scalar_to_utf16
            .get(node.range.start)
            .copied()
            .ok_or_else(|| EngineError::invalid("instrument node start exceeds source text"))?;
        let mut keys = vec![label.to_ascii_lowercase()];
        for (prefix, word) in [("art", "article"), ("part", "part"), ("div", "division")] {
            if let Some(value) = label
                .strip_prefix(prefix)
                .and_then(|value| value.parse().ok())
            {
                keys.push(format!("{word} {}", instrument_roman(value)));
            }
        }
        for key in keys {
            if index.insert(key.clone(), start).is_some() {
                duplicates.insert(key);
            }
        }
    }
    for key in duplicates {
        index.remove(&key);
    }
    Ok(index)
}

/// Select the instrument graph whose provision inventory is best endorsed by
/// typed references from the source text. Candidate zero wins every tie.
#[cfg(feature = "structure-inference")]
pub fn select_instrument_lineation(
    text: &str,
    graphs: &[StructureGraphV2],
    references: &[InstrumentReferenceEvidence],
) -> Result<usize, EngineError> {
    if graphs.is_empty() {
        return Err(EngineError::invalid(
            "instrument lineation selection requires a graph",
        ));
    }
    let mut scalar_to_utf16 = Vec::with_capacity(text.chars().count() + 1);
    scalar_to_utf16.push(0);
    for character in text.chars() {
        scalar_to_utf16.push(scalar_to_utf16.last().copied().unwrap() + character.len_utf16());
    }
    let text_length = *scalar_to_utf16.last().unwrap();
    let score = |graph: &StructureGraphV2| -> Result<usize, EngineError> {
        let index = instrument_reference_index(graph, &scalar_to_utf16)?;
        Ok(references
            .iter()
            .filter(|reference| {
                index
                    .get(&reference.key.to_lowercase())
                    .is_some_and(|start| *start < reference.start || *start >= reference.end)
            })
            .count())
    };
    let head_span = |graph: &StructureGraphV2| -> Result<f64, EngineError> {
        let starts = graph.nodes.iter().filter_map(|node| {
            let label = node.label.as_deref()?;
            (node.kind == NodeKind::Section && label.starts_with("sec") && !label.contains('('))
                .then_some(node.range.start)
        });
        let mut low = usize::MAX;
        let mut high = 0;
        let mut found = false;
        for start in starts {
            let start = scalar_to_utf16
                .get(start)
                .copied()
                .ok_or_else(|| EngineError::invalid("instrument node start exceeds source text"))?;
            low = low.min(start);
            high = high.max(start);
            found = true;
        }
        Ok(if found && text_length > 0 {
            (high - low) as f64 / text_length as f64
        } else {
            0.0
        })
    };

    let mut selected = 0;
    let mut best = score(&graphs[0])?;
    for (index, graph) in graphs.iter().enumerate().skip(1) {
        if head_span(graph)? < 0.05 {
            continue;
        }
        let candidate = score(graph)?;
        if candidate > best {
            selected = index;
            best = candidate;
        }
    }
    Ok(selected)
}

#[cfg(feature = "structure-inference")]
pub fn derive_instrument_structure(
    text: &str,
    documents: Vec<DocumentInput>,
    references: &[InstrumentReferenceEvidence],
) -> Result<(usize, StructureGraphV2, InstrumentContentsReading), EngineError> {
    if documents
        .iter()
        .any(|document| document.profile != DetectionProfile::Instrument)
    {
        return Err(EngineError::invalid(
            "instrument structure derivation requires instrument-profile evidence",
        ));
    }
    let graphs = documents
        .into_iter()
        .map(derive_structure_evidence)
        .collect::<Result<Vec<_>, _>>()?;
    let selected = select_instrument_lineation(text, &graphs, references)?;
    let graph = graphs
        .into_iter()
        .nth(selected)
        .ok_or_else(|| EngineError::invalid("selected instrument graph is missing"))?;
    Ok((selected, graph, instrument_contents_outline(text)))
}
