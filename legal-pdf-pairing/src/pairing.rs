use legal_pdf_core::model::{Diagnostic, Footnote, Line, Page};
use legal_pdf_core::{line_font_size, Anchor, PairingOutput};
use legal_pdf_support::pairing_support;
use legal_structure::{select_numeric_sequence, NumericSequenceCandidate, NumericSequencePolicy};
use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

const MAX_VALUE: u32 = 999;
const ENDNOTE_MIN_LABELS: usize = 8;
const SYMBOL_START_PAGE_LIMIT: u32 = 2;
const MAX_SYMBOL_RUN: usize = 8;
const SYMBOLS: &str = "*∗\u{f02a}†‡§¶#";
const SUPERSCRIPTS: &str = "⁰¹²³⁴⁵⁶⁷⁸⁹";
const QUOTES: &str = "\"'‘’“”«»";
const DASHES: &str = "–—-";
const TERMINAL_PUNCTUATION: &str = ".!?:;”\"'’";
const REF_PUNCTUATION: &str = ".,;:!?)]}";
const REF_RIGHT_PUNCTUATION: &str = ".,;:!?)]}…/¬\u{00ad}·";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Zone {
    Note,
    Body,
    Title,
    Header,
    Number,
    Visual,
    Other,
}

#[derive(Debug, Clone)]
struct PairLine {
    idx: usize,
    page: u32,
    page_index: usize,
    order: usize,
    id: String,
    region_id: String,
    region_type: String,
    text: String,
    bbox: [f64; 4],
    page_width: f64,
    page_height: f64,
    zone: Zone,
    protected_spans: Vec<(usize, usize)>,
    outline_spans: Vec<(usize, usize)>,
    note_column_fit: bool,
    small_font: bool,
    prose_like: bool,
    region_witness_demoted: bool,
    native_superscript_spans: Vec<(usize, usize)>,
    suppress_footnote_label: bool,
    exclude_from_body: bool,
    note_region_mode: String,
    note_sequence_restart: bool,
    detached_references: Vec<Value>,
}

impl PairLine {
    fn height(&self) -> f64 {
        self.bbox[3] - self.bbox[1]
    }
}

#[derive(Debug, Clone, Default)]
struct CandidateFlags {
    ref_supported: bool,
    paren_ref: bool,
    volume_split: bool,
}

#[derive(Debug, Clone)]
struct Candidate {
    line: usize,
    start: usize,
    end: usize,
    observed: String,
    value: Option<u32>,
    symbol: String,
    form: &'static str,
    score: f64,
    reason: &'static str,
    repaired: bool,
    repair_kind: &'static str,
    requires_visual_cue: bool,
    flags: CandidateFlags,
}

impl Candidate {
    fn pos(&self, lines: &[PairLine]) -> (usize, usize) {
        (lines[self.line].idx, self.start)
    }

    fn note_id(&self) -> String {
        self.value
            .map_or_else(|| self.symbol.clone(), |value| value.to_string())
    }

    fn zone_is_noteish(&self, lines: &[PairLine]) -> bool {
        let line = &lines[self.line];
        (line.zone == Zone::Note && !line.region_witness_demoted) || line.note_column_fit
    }
}

#[derive(Debug, Clone)]
struct Pair {
    label: Candidate,
    refs: Vec<Candidate>,
    primary_ref: Option<(usize, usize, usize)>,
    previous_value: Option<u32>,
    next_value: Option<u32>,
    restart_sequence: usize,
    endnote: bool,
    pair_id: String,
    provenance: String,
}

#[derive(Debug, Clone)]
struct LabelToken {
    pre_start: usize,
    start: usize,
    end: usize,
    match_end: usize,
    observed: String,
    value: Option<u32>,
    symbol: String,
    form: &'static str,
    post: String,
}

fn chars(value: &str) -> Vec<char> {
    value.chars().collect()
}

fn char_slice(value: &str, start: usize, end: usize) -> String {
    value
        .chars()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn chars_slice(values: &[char], start: usize, end: usize) -> String {
    values
        .iter()
        .skip(start)
        .take(end.saturating_sub(start))
        .collect()
}

fn is_symbol(character: char) -> bool {
    SYMBOLS.contains(character)
}

fn is_superscript(character: char) -> bool {
    SUPERSCRIPTS.contains(character)
}

fn normalized_digit(character: char) -> Option<char> {
    match character {
        '⁰' => Some('0'),
        '¹' => Some('1'),
        '²' => Some('2'),
        '³' => Some('3'),
        '⁴' => Some('4'),
        '⁵' => Some('5'),
        '⁶' => Some('6'),
        '⁷' => Some('7'),
        '⁸' => Some('8'),
        '⁹' => Some('9'),
        value if value.is_ascii_digit() => Some(value),
        _ => None,
    }
}

fn normalize_symbol(value: &str) -> String {
    value
        .chars()
        .map(|character| match character {
            '∗' | '\u{f02a}' => '*',
            other => other,
        })
        .collect()
}

fn numeric_value(value: &str) -> Option<u32> {
    let mut number = 0_u32;
    let mut found = false;
    for character in value.chars() {
        number = number
            .checked_mul(10)?
            .checked_add(normalized_digit(character)?.to_digit(10)?)?;
        found = true;
    }
    (found && (1..=MAX_VALUE).contains(&number)).then_some(number)
}

fn label_token(text: &str) -> Option<LabelToken> {
    let values = chars(text);
    let mut cursor = 0;
    while cursor < values.len()
        && cursor < 3
        && (values[cursor].is_whitespace() || "\"'‘’“”.,:;–—-".contains(values[cursor]))
    {
        cursor += 1;
    }
    let pre_start = 0;
    let start = cursor;
    let mut form = "plain";
    let mut token_start = cursor;
    let token_end: usize;
    let mut value = None;
    let mut symbol = String::new();
    if values.get(cursor) == Some(&'(') {
        cursor += 1;
        while values
            .get(cursor)
            .is_some_and(|character| character.is_whitespace())
        {
            cursor += 1;
        }
        token_start = cursor;
        while cursor < values.len() && cursor - token_start < 3 && values[cursor].is_ascii_digit() {
            cursor += 1;
        }
        token_end = cursor;
        while values
            .get(cursor)
            .is_some_and(|character| character.is_whitespace())
        {
            cursor += 1;
        }
        if token_end == token_start || values.get(cursor) != Some(&')') {
            return None;
        }
        cursor += 1;
        value = numeric_value(&char_slice(text, token_start, token_end));
        form = "paren";
    } else if values
        .get(cursor)
        .is_some_and(|character| is_superscript(*character))
    {
        while cursor < values.len() && cursor - token_start < 3 && is_superscript(values[cursor]) {
            cursor += 1;
        }
        token_end = cursor;
        value = numeric_value(&char_slice(text, token_start, token_end));
        form = "sup";
    } else if values.get(cursor).is_some_and(char::is_ascii_digit) {
        while cursor < values.len() && cursor - token_start < 3 && values[cursor].is_ascii_digit() {
            cursor += 1;
        }
        token_end = cursor;
        if values.get(cursor).is_some_and(char::is_ascii_digit) {
            return None;
        }
        value = numeric_value(&char_slice(text, token_start, token_end));
    } else if values
        .get(cursor)
        .is_some_and(|character| is_symbol(*character))
    {
        while cursor < values.len()
            && cursor - token_start < MAX_SYMBOL_RUN
            && is_symbol(values[cursor])
        {
            cursor += 1;
        }
        token_end = cursor;
        symbol = normalize_symbol(&char_slice(text, token_start, token_end));
        form = "symbol";
    } else {
        return None;
    }
    let end = if form == "paren" { cursor } else { token_end };
    let post_start = cursor;
    while cursor < values.len() && cursor - post_start < 2 && ".)],".contains(values[cursor]) {
        cursor += 1;
    }
    let observed = char_slice(text, if form == "paren" { start } else { token_start }, end);
    let embedded_endnote = format!("endnote {observed}");
    let remainder = char_slice(text, cursor, text.chars().count());
    let match_end = if form == "plain"
        && remainder
            .to_ascii_lowercase()
            .starts_with(&embedded_endnote.to_ascii_lowercase())
    {
        cursor + embedded_endnote.chars().count()
    } else {
        cursor
    };
    Some(LabelToken {
        pre_start,
        start: if form == "paren" { start } else { token_start },
        end,
        match_end,
        observed,
        value,
        symbol,
        form,
        post: char_slice(text, post_start, cursor),
    })
}

fn overlaps(spans: &[(usize, usize)], start: usize, end: usize) -> bool {
    spans
        .iter()
        .any(|(span_start, span_end)| start < *span_end && end > *span_start)
}

fn classify_zone(line: &Line) -> Zone {
    // core._pair_markers presents region_type twice (region/coarse) and a
    // derived line_type to the canonical pairer. Reconstruct that adapter
    // contract exactly rather than interpreting note_region_mode here.
    let line_type = if line.region_type == "footnote" {
        "footnote"
    } else {
        "paragraph"
    };
    let joined = format!("{} {} {line_type}", line.region_type, line.region_type).to_lowercase();
    if ["footnote", "endnote", "reference_content", "note"]
        .iter()
        .any(|token| joined.contains(token))
    {
        Zone::Note
    } else if ["page_number", "number", "folio"]
        .iter()
        .any(|token| joined.contains(token))
    {
        Zone::Number
    } else if ["header", "running"]
        .iter()
        .any(|token| joined.contains(token))
    {
        Zone::Header
    } else if [
        "image",
        "figure",
        "chart",
        "graphic",
        "formula",
        "separator",
        "table",
        "photo",
    ]
    .iter()
    .any(|token| joined.contains(token))
    {
        Zone::Visual
    } else if [
        "title",
        "heading",
        "byline",
        "abstract",
        "toc",
        "table_of_contents",
    ]
    .iter()
    .any(|token| joined.contains(token))
    {
        Zone::Title
    } else if ["body", "block_quote", "text", "content", "paragraph"]
        .iter()
        .any(|token| joined.contains(token))
    {
        Zone::Body
    } else {
        Zone::Other
    }
}

fn is_edge_folio(line: &Line) -> bool {
    matches!(
        line.region_type.as_str(),
        "header" | "footer" | "page_number" | "folio"
    ) && {
        let text = line
            .text
            .trim()
            .trim_matches(['-', '\u{2013}', '\u{2014}'])
            .trim();
        (1..=4).contains(&text.len()) && text.bytes().all(|byte| byte.is_ascii_digit())
    }
}

fn prose_like(text: &str) -> bool {
    static LOWER_WORD: OnceLock<Regex> = OnceLock::new();
    let lower_words = LOWER_WORD
        .get_or_init(|| Regex::new(r"[a-z]{2,}").unwrap())
        .find_iter(text)
        .take(2)
        .count();
    (lower_words >= 2 && text.chars().any(|character| ".,;".contains(character)))
        || (text
            .chars()
            .filter(|character| character.is_alphabetic())
            .count()
            >= 8
            && text
                .as_bytes()
                .windows(2)
                .any(|pair| pair[0].is_ascii_lowercase() && pair[1].is_ascii_digit()))
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.retain(|value| value.is_finite());
    values.sort_by(f64::total_cmp);
    values.get(values.len() / 2).copied()
}

fn core_line_label(text: &str) -> Option<String> {
    static LINE_START: OnceLock<Regex> = OnceLock::new();
    static PURE: OnceLock<Regex> = OnceLock::new();
    let line_start =
        LINE_START.get_or_init(|| Regex::new(r"^\s*(\d{1,4}|[*†‡§¶#])(?:\s|[.)\],:;-])").unwrap());
    let pure = PURE.get_or_init(|| Regex::new(r"^(?:\d{1,4}|[*†‡§¶#])$").unwrap());
    let raw = line_start
        .captures(text)
        .and_then(|capture| capture.get(1))
        .map(|value| value.as_str())
        .or_else(|| {
            let stripped = text.trim();
            pure.is_match(stripped).then_some(stripped)
        })?;
    Some(
        normalize_symbol(raw)
            .parse::<u32>()
            .map_or_else(|_| normalize_symbol(raw), |number| number.to_string()),
    )
}

fn folded_words(text: &str) -> String {
    text.to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn annotate_edge_furniture(lines: &mut [PairLine]) {
    static PAGE_LABEL: OnceLock<Regex> = OnceLock::new();
    let page_label = PAGE_LABEL
        .get_or_init(|| Regex::new(r"^\s*(?:[-–—]\s*)?(\d{1,4})(?:\s*[-–—])?\s*$").unwrap());
    let mut by_page = BTreeMap::<u32, Vec<usize>>::new();
    for (index, line) in lines.iter().enumerate() {
        by_page.entry(line.page).or_default().push(index);
    }
    let mut top_counts = HashMap::<String, usize>::new();
    for indexes in by_page.values() {
        let mut seen = HashSet::new();
        for &index in indexes.iter().take(4) {
            let key = folded_words(&lines[index].text);
            if (3..=120).contains(&key.chars().count()) {
                seen.insert(key);
            }
        }
        for key in seen {
            *top_counts.entry(key).or_default() += 1;
        }
    }
    let repeated: HashSet<String> = top_counts
        .into_iter()
        .filter_map(|(key, count)| (count >= 2).then_some(key))
        .collect();

    let mut edge_numerals = Vec::<(usize, u32)>::new();
    let mut offset_votes = HashMap::<i64, (usize, usize)>::new();
    let mut vote_order = 0;
    for (&page, indexes) in &by_page {
        for &index in indexes
            .iter()
            .take(4)
            .chain(indexes.iter().skip(indexes.len().saturating_sub(4)))
        {
            let Some(capture) = page_label.captures(&lines[index].text) else {
                continue;
            };
            let Some(value) = capture
                .get(1)
                .and_then(|value| value.as_str().parse::<u32>().ok())
            else {
                continue;
            };
            edge_numerals.push((index, value));
            let offset = i64::from(value) - i64::from(page);
            let entry = offset_votes.entry(offset).or_insert((0, vote_order));
            entry.0 += 1;
            vote_order += 1;
        }
    }
    let page_offset = offset_votes
        .into_iter()
        .max_by(|left, right| {
            left.1
                 .0
                .cmp(&right.1 .0)
                .then_with(|| right.1 .1.cmp(&left.1 .1))
        })
        .and_then(|(offset, (votes, _))| (votes >= 3).then_some(offset));

    for indexes in by_page.values() {
        for &index in indexes.iter().take(4) {
            let key = folded_words(&lines[index].text);
            if matches!(lines[index].zone, Zone::Body | Zone::Other | Zone::Title)
                && repeated.contains(&key)
            {
                lines[index].zone = Zone::Header;
            }
        }
    }
    if let Some(offset) = page_offset {
        for (index, value) in edge_numerals {
            if matches!(lines[index].zone, Zone::Body | Zone::Other | Zone::Title)
                && i64::from(value) - i64::from(lines[index].page) == offset
            {
                lines[index].zone = Zone::Number;
            }
        }
    }
}

fn build_lines(pages: &[Page]) -> Vec<PairLine> {
    let mut source: Vec<(&Page, &Line)> =
        legal_pdf_support::profile::measure("lines.collect", || {
            pages
                .iter()
                .flat_map(|page| page.lines.iter().map(move |line| (page, line)))
                .collect()
        });
    legal_pdf_support::profile::measure("lines.sort", || {
        source.sort_by(|(left_page, left_line), (right_page, right_line)| {
            left_page
                .index
                .cmp(&right_page.index)
                .then_with(|| left_line.reading_order.cmp(&right_line.reading_order))
                .then_with(|| left_line.id.cmp(&right_line.id))
        })
    });
    let mut lines: Vec<PairLine> = legal_pdf_support::profile::measure("lines.construct", || {
        source
            .into_iter()
            .enumerate()
            .filter(|(_, (_, line))| !line.exclude_from_body)
            .enumerate()
            .map(|(idx, (source_order, (page, line)))| {
                let mut detached_references = line.detached_references.clone();
                detached_references.extend(isolated_inline_references(line));
                let text_len = line.text.chars().count();
                PairLine {
                    idx,
                    page: page.number,
                    page_index: page.index,
                    order: source_order + 1,
                    id: line.id.clone(),
                    region_id: line.region_id.clone(),
                    region_type: line.region_type.clone(),
                    text: line.text.clone(),
                    bbox: line.bbox,
                    page_width: page.width,
                    page_height: page.height,
                    zone: classify_zone(line),
                    protected_spans: pairing_support::protected_citation_spans(&line.text),
                    outline_spans: Vec::new(),
                    note_column_fit: false,
                    small_font: false,
                    prose_like: prose_like(&line.text),
                    region_witness_demoted: false,
                    native_superscript_spans: line
                        .spans
                        .iter()
                        .filter(|span| {
                            span.superscript && span.start < span.end && span.end <= text_len
                        })
                        .map(|span| (span.start, span.end))
                        .collect(),
                    suppress_footnote_label: line.suppress_footnote_label || is_edge_folio(line),
                    exclude_from_body: line.exclude_from_body,
                    note_region_mode: line.note_region_mode.clone(),
                    note_sequence_restart: false,
                    detached_references,
                }
            })
            .collect()
    });
    legal_pdf_support::profile::measure("lines.annotate", || {
        let mut page_local_one_pages = HashSet::new();
        for line in &mut lines {
            let page_local_one = line.region_type == "footnote"
                && line.note_region_mode == "footnote"
                && core_line_label(&line.text).as_deref() == Some("1");
            if page_local_one {
                line.note_sequence_restart =
                    !page_local_one_pages.is_empty() && !page_local_one_pages.contains(&line.page);
                page_local_one_pages.insert(line.page);
            }
        }
        let body_median = median(
            lines
                .iter()
                .filter(|line| line.zone == Zone::Body && line.height() > 0.0)
                .map(PairLine::height)
                .collect(),
        );
        let note_column = median(
            lines
                .iter()
                .filter(|line| line.zone == Zone::Note && label_token(&line.text).is_some())
                .map(|line| line.bbox[0])
                .collect(),
        );
        for line in &mut lines {
            if body_median
                .is_some_and(|height| line.height() > 0.0 && line.height() <= 0.92 * height)
            {
                line.small_font = true;
            }
            if note_column.is_some_and(|column| {
                line.page_width > 0.0 && (line.bbox[0] - column).abs() <= 0.025 * line.page_width
            }) {
                line.note_column_fit = true;
            }
        }
    });
    legal_pdf_support::profile::measure("lines.edge_furniture", || {
        annotate_edge_furniture(&mut lines)
    });
    legal_pdf_support::profile::measure("lines.outlines", || {
        annotate_hierarchical_outlines(&mut lines)
    });
    lines
}

/// Preserve the old extractor's detached-marker evidence when the faster PDF
/// backend emits the same raised run inline. A visibly separated line-final
/// digit is the exact case PyMuPDF exposed as its own row; contiguous native
/// superscripts remain on the monotone pairing path.
fn isolated_inline_references(line: &Line) -> Vec<Value> {
    let text_len = line.text.chars().count();
    line.spans
        .iter()
        .filter_map(|marker| {
            let selected = marker.text.trim();
            if !marker.superscript
                || marker.end != text_len
                || marker.start >= marker.end
                || selected.is_empty()
                || selected.len() > 4
                || !selected.chars().all(|character| character.is_ascii_digit())
            {
                return None;
            }
            let host = line
                .spans
                .iter()
                .filter(|span| !span.superscript && span.end == marker.start)
                .max_by_key(|span| span.end)?;
            let gap = marker.bbox[0] - host.bbox[2];
            (gap >= host.size.max(1.0) * 0.25).then(|| {
                json!({
                    "note_id": selected,
                    "selected_text": selected,
                    "start_offset": marker.start,
                    "end_offset": marker.end,
                    "source_line_id": line.id,
                })
            })
        })
        .collect()
}

fn annotate_hierarchical_outlines(lines: &mut [PairLine]) {
    static OUTLINE: OnceLock<Regex> = OnceLock::new();
    let regex = OUTLINE
        .get_or_init(|| Regex::new(r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})([.)]?)(\s+)(\S.*)$").unwrap());
    type OutlineCandidate = (usize, Vec<u32>, usize, usize);
    let mut by_page: HashMap<u32, Vec<OutlineCandidate>> = HashMap::new();
    for (index, line) in lines.iter().enumerate() {
        let Some(captures) = regex.captures(&line.text) else {
            continue;
        };
        let body = captures.get(4).map_or("", |value| value.as_str());
        if !pairing_support::heading_text_plausible(body) {
            continue;
        }
        let raw = captures.get(1).expect("outline value").as_str();
        let parts: Vec<u32> = raw
            .split('.')
            .filter_map(|value| value.parse().ok())
            .collect();
        if parts.len() == 1
            && captures
                .get(2)
                .map_or("", |value| value.as_str())
                .is_empty()
        {
            continue;
        }
        let value_match = captures.get(1).expect("outline value");
        let end_match = captures
            .get(2)
            .filter(|value| !value.as_str().is_empty())
            .unwrap_or(value_match);
        let start = line.text[..value_match.start()].chars().count();
        let end = line.text[..end_match.end()].chars().count();
        by_page
            .entry(line.page)
            .or_default()
            .push((index, parts, start, end));
    }
    for values in by_page.values() {
        if values.len() < 4
            || values
                .iter()
                .filter(|(_, parts, _, _)| parts.len() > 1)
                .count()
                < 2
        {
            continue;
        }
        if values.windows(2).any(|pair| pair[1].1 <= pair[0].1) {
            continue;
        }
        let mut seen = HashSet::new();
        if values.iter().any(|(_, parts, _, _)| {
            let missing = parts.len() > 1 && !seen.contains(&parts[..parts.len() - 1]);
            seen.insert(parts.clone());
            missing
        }) {
            continue;
        }
        let grammar = values
            .iter()
            .map(|(index, _, _, _)| {
                let captures = regex
                    .captures(&lines[*index].text)
                    .expect("accepted outline capture");
                let raw = captures.get(1).expect("outline value").as_str();
                let punct = captures.get(2).map_or("", |value| value.as_str());
                let effective_punct = if punct.is_empty() { "." } else { punct };
                json!({
                    "line_index": lines[*index].idx,
                    "kind": "enumerator",
                    "joined": false,
                    "value_text": raw,
                    "punct": effective_punct,
                    "text": captures.get(4).map_or("", |value| value.as_str()),
                    "interpretations": pairing_support::enumerator_interpretations(raw, effective_punct),
                })
            })
            .collect::<Vec<_>>();
        if pairing_support::parse_heading_ladder(&grammar)
            .get("status")
            .and_then(Value::as_str)
            != Some("parsed_clean")
        {
            continue;
        }
        let page_width = values
            .iter()
            .map(|(index, _, _, _)| lines[*index].page_width)
            .find(|width| *width != 0.0)
            .unwrap_or(800.0);
        let tolerance = 6.0_f64.max(page_width * 0.015);
        let mut x0_by_depth = HashMap::<usize, Vec<f64>>::new();
        for (index, parts, _, _) in values {
            x0_by_depth
                .entry(parts.len())
                .or_default()
                .push(lines[*index].bbox[0]);
        }
        if x0_by_depth.values().any(|positions| {
            let minimum = positions.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = positions.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            maximum - minimum > tolerance
        }) {
            continue;
        }
        for (index, _, start, end) in values {
            lines[*index].protected_spans.push((*start, *end));
            lines[*index].outline_spans.push((*start, *end));
        }
    }
}

fn ref_zone_score(line: &PairLine) -> Option<(f64, bool)> {
    match line.zone {
        Zone::Body | Zone::Other => Some((0.6, false)),
        Zone::Title => Some((0.1, false)),
        Zone::Visual => Some((-0.4, false)),
        Zone::Note => Some((-0.4, true)),
        Zone::Header | Zone::Number if line.prose_like => Some((-0.5, true)),
        _ => None,
    }
}

fn ref_left(character: char) -> bool {
    character.is_alphabetic() || REF_PUNCTUATION.contains(character) || QUOTES.contains(character)
}

fn ref_right(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\u{00a0}')
        || REF_RIGHT_PUNCTUATION.contains(character)
        || QUOTES.contains(character)
        || DASHES.contains(character)
}

fn chars_at_line_end(values: &[char], end: usize) -> bool {
    let tail = values.get(end..).unwrap_or_default();
    let start = tail
        .iter()
        .position(|character| !character.is_whitespace())
        .unwrap_or(tail.len());
    let end = tail
        .iter()
        .rposition(|character| !character.is_whitespace())
        .map_or(start, |index| index + 1);
    tail[start..end]
        .iter()
        .all(|character| REF_PUNCTUATION.contains(*character) || QUOTES.contains(*character))
}

fn at_line_end(text: &str, end: usize) -> bool {
    chars_at_line_end(&chars(text), end)
}

fn preceding_word(values: &[char], start: usize) -> String {
    let end = start.min(values.len());
    let start = values[..end]
        .iter()
        .rposition(|character| !character.is_ascii_alphabetic())
        .map_or(0, |index| index + 1);
    values[start..end].iter().collect()
}

fn spaced_preceding_word(values: &[char], start: usize) -> String {
    let mut end = start.min(values.len());
    while end > 0 && matches!(values[end - 1], ' ' | '\t') {
        end -= 1;
    }
    preceding_word(values, end)
}

fn counter_noun(word: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)^(?:notes?|supra|infra|pages?|pp|paras?|paragraphs?|secs?|sections?|arts?|articles?|vols?|volumes?|nos?|numbers?|chapters?|parts?|clauses?|rules?|regs?|schedules?|appendix|appendices|tables?|figures?|figs?|charts?|columns?|cols?|books?|editions?|amend)$").unwrap()).is_match(word)
}

fn abbreviation_pinpoint(prefix: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:^|[\s(\[])(?:pp?|paras?|ss?|secs?|arts?|vols?|nos?|c|ch|cc|pts?|eds?|figs?|tabs?|apps?|cls?)\.$",
        )
        .unwrap()
    })
    .is_match(prefix)
}

fn date_day_site(values: &[char], start: usize, end: usize) -> bool {
    static MONTH_BEFORE: OnceLock<Regex> = OnceLock::new();
    static MONTH_AFTER: OnceLock<Regex> = OnceLock::new();
    static COMMA_YEAR: OnceLock<Regex> = OnceLock::new();
    static DAY_LIST: OnceLock<Regex> = OnceLock::new();
    let month = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)";
    let prefix = chars_slice(values, 0, start);
    let tail = chars_slice(values, end, values.len());
    MONTH_BEFORE
        .get_or_init(|| Regex::new(&format!(r"(?i)\b{month}\.?[ \t]+$")).unwrap())
        .is_match(&prefix)
        || MONTH_AFTER
            .get_or_init(|| Regex::new(&format!(r"(?i)^[ \t]+{month}\b[.,]?")).unwrap())
            .is_match(&tail)
        || COMMA_YEAR
            .get_or_init(|| Regex::new(r"^,\s*(?:1[6-9]|20)\d\d\b").unwrap())
            .is_match(&tail)
        || DAY_LIST
            .get_or_init(|| {
                Regex::new(&format!(
                    r"(?i)^\s*\d{{1,2}}(?:\s*,\s*\d{{1,2}})+\s+{month}\b"
                ))
                .unwrap()
            })
            .is_match(&chars_slice(values, start, values.len()))
}

fn ref_site_penalty(
    line: &PairLine,
    values: &[char],
    start: usize,
    end: usize,
    form: &str,
) -> Option<f64> {
    if form == "sup" {
        return Some(0.0);
    }
    let left = start
        .checked_sub(1)
        .and_then(|index| values.get(index))
        .copied();
    let right = values.get(end).copied();
    let mut penalty = 0.0;
    if let Some(right) = right {
        if !ref_right(right) {
            let tail = chars_slice(values, end, (end + 2).min(values.len())).to_lowercase();
            if ["th", "st", "nd", "rd", "am", "pm"].contains(&tail.as_str())
                && values
                    .get(end + 2)
                    .is_none_or(|value| !value.is_alphabetic())
            {
                return None;
            }
            if right.is_uppercase() {
                penalty += 0.4;
            } else if right.is_lowercase()
                && values
                    .get(end + 1)
                    .is_none_or(|value| value.is_whitespace())
            {
                penalty += 0.7;
            } else if right.is_lowercase() {
                penalty += 1.0;
            } else {
                return None;
            }
        }
        if DASHES.contains(right) && values.get(end + 1).is_some_and(char::is_ascii_digit) {
            return None;
        }
        if right == '.' && values.get(end + 1).is_some_and(char::is_ascii_digit) {
            let left_char = left.unwrap_or_default();
            let following_end = values[end + 1..]
                .iter()
                .position(|value| !value.is_ascii_digit())
                .map_or(values.len(), |offset| end + 1 + offset);
            if !(left_char.is_alphabetic()
                && values.get(end + 1..following_end) == values.get(start..end))
            {
                return None;
            }
        }
        if right == ','
            && values
                .get(end + 1..end + 4)
                .is_some_and(|tail| tail.iter().all(char::is_ascii_digit))
            && values
                .get(end + 4)
                .is_none_or(|value| !value.is_ascii_digit())
        {
            return None;
        }
    }
    if left.is_some_and(|value| DASHES.contains(value))
        && start >= 2
        && values[start - 2].is_ascii_digit()
    {
        return None;
    }
    if left == Some('.') && start >= 2 && values[start - 2] == '.' {
        let mut run_start = start - 2;
        while run_start > 0 && values[run_start - 1] == '.' {
            run_start -= 1;
        }
        if run_start == 0 || !values[run_start - 1].is_alphabetic() {
            return None;
        }
        penalty += 0.4;
    }
    if left == Some('.') && start >= 2 && values[start - 2].is_ascii_digit() {
        penalty += 0.8;
    }
    if values[start.saturating_sub(3)..start].contains(&'$') {
        return None;
    }
    if overlaps(&line.outline_spans, start, end) {
        return None;
    }
    if overlaps(&line.protected_spans, start, end) {
        let tail_glued = left.is_some_and(|value| ".,".contains(value))
            && start >= 2
            && values[start - 2].is_ascii_digit()
            && line
                .protected_spans
                .iter()
                .any(|(span_start, span_end)| end == *span_end && start > *span_start);
        if !tail_glued {
            return None;
        }
        penalty += 1.2;
    }
    let word = preceding_word(values, start);
    if counter_noun(&word) {
        if line.zone == Zone::Note {
            return None;
        }
        penalty += 1.1;
    }
    let prefix = chars_slice(values, 0, start);
    if abbreviation_pinpoint(&prefix) {
        return None;
    }
    if matches!(form, "spaced_eol" | "spaced_mid")
        && counter_noun(&spaced_preceding_word(values, start))
    {
        return None;
    }
    if form == "spaced_mid" {
        static MEASURE: OnceLock<Regex> = OnceLock::new();
        if MEASURE
            .get_or_init(|| Regex::new(r"(?i)^[ \t]+(?:years?|days?|months?|weeks?|hours?|minutes?|seconds?|per|percent|p\.c\b|cents?|dollars?|pounds?|shillings?|pence|feet|foot|acres?|miles?|inches?|yards?|tons?|o.?clock)\b").unwrap())
            .is_match(&chars_slice(values, end, values.len()))
        {
            return None;
        }
        if date_day_site(values, start, end) {
            return None;
        }
    }
    Some(penalty)
}

fn digit_runs(values: &[char]) -> Vec<(usize, usize)> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < values.len() {
        if !values[index].is_ascii_digit() {
            index += 1;
            continue;
        }
        let start = index;
        while index < values.len() && values[index].is_ascii_digit() {
            index += 1;
        }
        if index - start <= 7 {
            result.push((start, index));
        }
    }
    result
}

fn ascii_number(values: &[char], start: usize, end: usize) -> u32 {
    values[start..end]
        .iter()
        .fold(0, |number, digit| number * 10 + digit.to_digit(10).unwrap())
}

fn superscript_runs(values: &[char]) -> Vec<(usize, usize, String)> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < values.len() {
        if !is_superscript(values[index]) {
            index += 1;
            continue;
        }
        let start = index;
        while index < values.len() && is_superscript(values[index]) && index - start < 4 {
            index += 1;
        }
        result.push((start, index, values[start..index].iter().collect()));
    }
    result
}

fn symbol_runs(values: &[char]) -> Vec<(usize, usize, String)> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < values.len() {
        if !is_symbol(values[index]) {
            index += 1;
            continue;
        }
        let start = index;
        while index < values.len() && is_symbol(values[index]) && index - start < MAX_SYMBOL_RUN {
            index += 1;
        }
        result.push((start, index, values[start..index].iter().collect()));
    }
    result
}

fn extract_refs(lines: &[PairLine]) -> Vec<Candidate> {
    let mut candidates = Vec::new();
    let mut previous_body = None;
    let mut previous_page = None;
    let mut previous_by_line = Vec::with_capacity(lines.len());
    static PAREN_REF: OnceLock<Regex> = OnceLock::new();
    static SECTION_BEFORE_PAREN: OnceLock<Regex> = OnceLock::new();
    let paren_ref = PAREN_REF.get_or_init(|| Regex::new(r"\(\s*(\d{1,3})\s*\)").unwrap());
    let section_before = SECTION_BEFORE_PAREN
        .get_or_init(|| Regex::new(r"(?i)\b(?:ss?|sub-?ss?|arts?|paras?|cls?)\.?\s*$").unwrap());
    for (line_index, line) in lines.iter().enumerate() {
        if previous_page != Some(line.page) {
            previous_body = None;
            previous_page = Some(line.page);
        }
        previous_by_line.push(previous_body);
        if line.zone == Zone::Body && !line.text.trim().is_empty() {
            previous_body = Some(line_index);
        }
    }
    for (line_index, line) in lines.iter().enumerate() {
        let zone = ref_zone_score(line);
        let sup_zone_score = zone.map(|(score, _)| score).or_else(|| {
            (matches!(line.zone, Zone::Header | Zone::Number)
                && line.text.trim().chars().count() >= 6)
                .then_some(0.1)
        });
        let Some(sup_zone_score) = sup_zone_score else {
            continue;
        };
        let label_end = if line.zone == Zone::Note {
            label_token(&line.text).map_or(0, |token| token.match_end)
        } else {
            0
        };
        let line_chars = chars(&line.text);
        let trimmed_len = line_chars
            .iter()
            .rposition(|character| !character.is_whitespace())
            .map_or(0, |index| index + 1);
        let digit_runs = digit_runs(&line_chars);
        let mut seen = HashSet::new();
        for (start, end, observed) in superscript_runs(&line_chars) {
            if start < label_end || overlaps(&line.protected_spans, start, end) {
                continue;
            }
            let Some(value) = numeric_value(&observed) else {
                continue;
            };
            seen.insert((start, end));
            candidates.push(Candidate {
                line: line_index,
                start,
                end,
                observed,
                value: Some(value),
                symbol: String::new(),
                form: "sup",
                score: 1.6 + sup_zone_score,
                reason: "superscript_marker",
                repaired: false,
                repair_kind: "",
                requires_visual_cue: line.zone == Zone::Visual,
                flags: CandidateFlags::default(),
            });
        }
        for &(start, end) in &line.native_superscript_spans {
            if seen.contains(&(start, end))
                || start < label_end
                || overlaps(&line.protected_spans, start, end)
            {
                continue;
            }
            let observed = chars_slice(&line_chars, start, end);
            let value = numeric_value(&observed);
            let symbol = value.map_or_else(
                || {
                    if (1..=MAX_SYMBOL_RUN).contains(&observed.chars().count())
                        && observed.chars().all(is_symbol)
                    {
                        normalize_symbol(&observed)
                    } else {
                        String::new()
                    }
                },
                |_| String::new(),
            );
            if value.is_none() && symbol.is_empty() {
                continue;
            };
            seen.insert((start, end));
            candidates.push(Candidate {
                line: line_index,
                start,
                end,
                observed,
                value,
                symbol,
                form: if value.is_some() { "sup" } else { "symbol" },
                score: 1.6 + sup_zone_score,
                reason: "native_superscript_span",
                repaired: false,
                repair_kind: "",
                requires_visual_cue: line.zone == Zone::Visual,
                flags: CandidateFlags::default(),
            });
        }
        let Some((zone_score, _)) = zone else {
            continue;
        };
        let stripped = line.text.trim();
        if stripped.chars().all(|character| character.is_ascii_digit())
            && (1..=3).contains(&stripped.chars().count())
            && line.zone == Zone::Body
            && (line.bbox[1] + line.bbox[3]) / 2.0 / line.page_height > 0.05
        {
            if let Some(value) = numeric_value(stripped) {
                let start = line
                    .text
                    .chars()
                    .position(|character| character.is_ascii_digit())
                    .unwrap_or(0);
                candidates.push(Candidate {
                    line: line_index,
                    start,
                    end: start + stripped.chars().count(),
                    observed: stripped.to_owned(),
                    value: Some(value),
                    symbol: String::new(),
                    form: "standalone",
                    score: 0.5 + zone_score,
                    reason: "standalone_marker_line",
                    repaired: false,
                    repair_kind: "",
                    requires_visual_cue: false,
                    flags: CandidateFlags::default(),
                });
                continue;
            }
        }
        for &(start, end) in digit_runs
            .iter()
            .filter(|(start, end)| (1..=3).contains(&(end - start)))
        {
            if seen.contains(&(start, end)) || start < label_end {
                continue;
            }
            let value = ascii_number(&line_chars, start, end);
            if !(1..=MAX_VALUE).contains(&value) {
                continue;
            }
            let left = start
                .checked_sub(1)
                .and_then(|index| line_chars.get(index))
                .copied();
            let at_eol = chars_at_line_end(&line_chars, end);
            let form = if start == 0 {
                let previous = previous_by_line[line_index].map(|index| &lines[index]);
                let follow_ok = line_chars
                    .get(end)
                    .is_none_or(|value| value.is_whitespace());
                if line.zone == Zone::Note
                    || !follow_ok
                    || previous.is_none_or(|value| {
                        value
                            .text
                            .trim_end()
                            .ends_with(|value| TERMINAL_PUNCTUATION.contains(value))
                    })
                {
                    continue;
                }
                "line_start"
            } else if left.is_some_and(ref_left) {
                "tight"
            } else if left.is_some_and(char::is_whitespace) && at_eol && line.zone != Zone::Note {
                "spaced_eol"
            } else if left.is_some_and(char::is_whitespace) && line.zone == Zone::Body {
                "spaced_mid"
            } else if left.is_some_and(|value| value.is_lowercase()) {
                let word = preceding_word(&line_chars, start);
                if word.len() < 3
                    || counter_noun(&word)
                    || line_chars.get(end).is_some_and(|value| !ref_right(*value))
                {
                    continue;
                }
                "letter_glued"
            } else {
                continue;
            };
            let Some(penalty) = ref_site_penalty(line, &line_chars, start, end, form) else {
                continue;
            };
            let mut score = zone_score - penalty
                + match form {
                    "tight" => 0.9,
                    "line_start" => -0.3,
                    "spaced_eol" => 0.1,
                    "spaced_mid" | "letter_glued" => -0.6,
                    _ => 0.0,
                };
            if end == trimmed_len {
                score += 0.15;
            }
            if matches!(line.zone, Zone::Title | Zone::Visual) && form != "tight" {
                continue;
            }
            candidates.push(Candidate {
                line: line_index,
                start,
                end,
                observed: chars_slice(&line_chars, start, end),
                value: Some(value),
                symbol: String::new(),
                form,
                score,
                reason: match form {
                    "tight" => "attached_digit_marker",
                    "line_start" => "line_start_marker",
                    "spaced_eol" => "spaced_end_of_line_marker",
                    "spaced_mid" => "spaced_mid_line_marker",
                    _ => "word_glued_marker",
                },
                repaired: false,
                repair_kind: "",
                requires_visual_cue: line.zone == Zone::Visual,
                flags: CandidateFlags::default(),
            });
        }
        for &(start, end) in digit_runs
            .iter()
            .filter(|(start, end)| (5..=7).contains(&(end - start)))
        {
            if start < label_end || matches!(line.zone, Zone::Title | Zone::Visual) {
                continue;
            }
            let year = ascii_number(&line_chars, start, start + 4);
            let tail = ascii_number(&line_chars, start + 4, end);
            let left = start
                .checked_sub(1)
                .and_then(|index| line_chars.get(index))
                .copied();
            if !(1600..=2069).contains(&year)
                || tail == 0
                || left.is_some_and(|value| !value.is_whitespace() && !ref_left(value))
                || line_chars[start.saturating_sub(3)..start].contains(&'$')
            {
                continue;
            }
            let sub_start = start + 4;
            let Some(penalty) = ref_site_penalty(line, &line_chars, sub_start, end, "tight") else {
                continue;
            };
            let mut score = zone_score - penalty;
            if end == trimmed_len {
                score += 0.15;
            }
            candidates.push(Candidate {
                line: line_index,
                start: sub_start,
                end,
                observed: chars_slice(&line_chars, sub_start, end),
                value: Some(tail),
                symbol: String::new(),
                form: "year_glued",
                score,
                reason: "year_glued_marker",
                repaired: false,
                repair_kind: "",
                requires_visual_cue: false,
                flags: CandidateFlags::default(),
            });
        }
        for captures in paren_ref.captures_iter(&line.text) {
            let whole = captures.get(0).expect("paren ref");
            let digits = captures.get(1).expect("paren digits");
            let open = line.text[..whole.start()].chars().count();
            let start = line.text[..digits.start()].chars().count();
            let end = line.text[..digits.end()].chars().count();
            if start < label_end
                || (open > 0 && line_chars[open - 1].is_alphanumeric())
                || section_before.is_match(&char_slice(&line.text, 0, open))
                || overlaps(&line.protected_spans, start, end)
            {
                continue;
            }
            let Some(value) = numeric_value(digits.as_str()) else {
                continue;
            };
            let Some(penalty) = ref_site_penalty(line, &line_chars, start, end, "tight") else {
                continue;
            };
            candidates.push(Candidate {
                line: line_index,
                start,
                end,
                observed: digits.as_str().to_owned(),
                value: Some(value),
                symbol: String::new(),
                form: "paren",
                score: zone_score - penalty + 0.4,
                reason: "paren_marker",
                repaired: false,
                repair_kind: "",
                requires_visual_cue: line.zone == Zone::Visual,
                flags: CandidateFlags {
                    paren_ref: true,
                    ..CandidateFlags::default()
                },
            });
        }
        if line.zone != Zone::Note {
            for (start, end, observed) in symbol_runs(&line_chars) {
                if start == 0
                    || seen.contains(&(start, end))
                    || start < label_end
                    || overlaps(&line.protected_spans, start, end)
                {
                    continue;
                }
                let left = line_chars[start - 1];
                let mut spaced = left.is_whitespace() && line.zone == Zone::Title;
                if left.is_whitespace() && line.zone == Zone::Body && !spaced {
                    let prefix = chars_slice(&line_chars, 0, start);
                    let tail = chars_slice(&line_chars, end, line_chars.len());
                    let run_head = prefix
                        .trim_end()
                        .chars()
                        .last()
                        .is_none_or(|character| !is_symbol(character));
                    let starts_with_run =
                        line.text.trim_start().chars().next().is_some_and(is_symbol);
                    let more_symbols = tail.trim_start().chars().next().is_some_and(is_symbol);
                    let prose = line_chars
                        .iter()
                        .filter(|character| character.is_alphabetic())
                        .count()
                        >= 12;
                    spaced = run_head && !starts_with_run && more_symbols && prose;
                }
                let glued_byline = left.is_alphabetic() && line.zone == Zone::Title;
                if !ref_left(left) && !left.is_ascii_digit() && !spaced && !glued_byline {
                    continue;
                }
                if line_chars.get(end).is_some_and(|right| !ref_right(*right)) {
                    continue;
                }
                candidates.push(Candidate {
                    line: line_index,
                    start,
                    end,
                    observed: observed.clone(),
                    value: None,
                    symbol: normalize_symbol(&observed),
                    form: "symbol",
                    score: 0.9 + zone_score - if spaced { 0.3 } else { 0.0 },
                    reason: "attached_symbol_marker",
                    repaired: false,
                    repair_kind: "",
                    requires_visual_cue: line.zone == Zone::Visual,
                    flags: CandidateFlags::default(),
                });
            }
        }
    }
    candidates
}

fn heading_shaped(body: &str) -> bool {
    let letters: Vec<char> = body
        .trim()
        .chars()
        .take(40)
        .filter(|character| character.is_alphabetic())
        .collect();
    letters.len() >= 4 && letters.iter().all(|character| character.is_uppercase())
}

fn paren_style(lines: &[PairLine]) -> bool {
    let mut paren = 0;
    let mut plain = 0;
    for line in lines.iter().filter(|line| line.zone == Zone::Note) {
        let Some(token) = label_token(&line.text) else {
            continue;
        };
        if token.form == "paren" || (token.form == "plain" && token.post.contains(')')) {
            paren += 1;
        } else if matches!(token.form, "plain" | "sup") {
            plain += 1;
        }
    }
    paren >= 3 && paren > 2 * plain
}

fn spaced_symbol_separator(text: &str) -> bool {
    let parts: Vec<&str> = text.split_whitespace().collect();
    parts.len() >= 3
        && parts
            .iter()
            .all(|part| part.chars().count() == 1 && part.chars().all(is_symbol))
}

fn day_list_date(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)^\s*\d{1,2}(?:\s*,\s*\d{1,2})+\s+(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b",
        )
        .unwrap()
    })
    .is_match(text)
}

fn volume_cite_start(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let Some(found) = RE
        .get_or_init(|| {
        Regex::new(
            r"^\d{1,3}\s+[A-Z][A-Za-z]{0,5}\.(?:(?:\s+(?:&\s*)?|&\s*)[A-Z][A-Za-z]{0,5}\.?|[A-Z]\.)*\s*(?:\(\d{1,4}[a-z]{0,2}\)\s*)?(?:(?:c|ch|ss?)\.\s*)?\d{1,4}",
        )
        .unwrap()
    })
        .find(text)
    else {
        return false;
    };
    text[found.end()..]
        .chars()
        .next()
        .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
}

fn bracket_year_body(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\[(?:1[5-9]|20)\d\d\]").unwrap())
        .is_match(text)
}

fn extract_labels(lines: &[PairLine], refs: &[Candidate]) -> Vec<Candidate> {
    let mut ref_pages: HashMap<u32, HashSet<u32>> = HashMap::new();
    for candidate in refs
        .iter()
        .filter(|candidate| !candidate.repaired && !candidate.flags.paren_ref)
    {
        if let Some(value) = candidate.value {
            ref_pages
                .entry(value)
                .or_default()
                .insert(lines[candidate.line].page);
        }
    }
    let paren_style = paren_style(lines);
    let mut candidates = Vec::new();
    for (line_index, line) in lines.iter().enumerate() {
        if line.suppress_footnote_label {
            continue;
        }
        let Some(token) = label_token(&line.text) else {
            continue;
        };
        if overlaps(&line.protected_spans, token.start, token.end) {
            continue;
        }
        let body = char_slice(&line.text, token.match_end, line.text.chars().count());
        let follow = body.chars().next();
        if token.form == "plain"
            && token.value.is_some()
            && day_list_date(&char_slice(
                &line.text,
                token.start,
                line.text.chars().count(),
            ))
        {
            continue;
        }
        if token.post.is_empty()
            && follow.is_some_and(|value| {
                !value.is_whitespace()
                    && !value.is_uppercase()
                    && !QUOTES.contains(value)
                    && !(value == '[' && bracket_year_body(&body))
            })
        {
            continue;
        }
        if token.symbol.is_empty() && token.value.is_none() {
            continue;
        }
        if token.form == "plain" && follow.is_some_and(|value| value.is_lowercase()) {
            continue;
        }
        if token.form == "symbol" {
            if spaced_symbol_separator(&line.text) {
                continue;
            }
            let bare = body.trim().is_empty();
            if !(bare && line.page <= SYMBOL_START_PAGE_LIMIT) && body.trim().chars().count() < 4 {
                continue;
            }
        }
        let body_stripped = body.trim();
        if token.value.is_some() && line.zone != Zone::Note && heading_shaped(body_stripped) {
            continue;
        }
        if token.value.is_some()
            && token.post.contains(',')
            && line.zone != Zone::Note
            && body_stripped
                .chars()
                .next()
                .is_some_and(|character| character.is_lowercase())
        {
            continue;
        }
        let mut score = match line.zone {
            Zone::Note => 3.0,
            Zone::Body => 0.2,
            Zone::Title => -2.0,
            Zone::Header | Zone::Number => -1.2,
            Zone::Visual => -1.5,
            Zone::Other => 0.6,
        };
        if line.zone == Zone::Note && line.region_witness_demoted {
            score = 0.6;
        }
        if matches!(line.zone, Zone::Header | Zone::Number)
            && !line.prose_like
            && body_stripped.is_empty()
        {
            continue;
        }
        if token.form == "sup" {
            score += 1.0;
        } else if token.form == "paren" || (token.form == "plain" && token.post.contains(')')) {
            score += if paren_style { 0.4 } else { -1.8 };
        } else if !token.post.is_empty() || follow.is_some_and(char::is_whitespace) {
            score += 0.8;
        }
        if !body_stripped.is_empty() {
            if pairing_support::has_legal_citation_cue(body_stripped) {
                score += 0.7;
            } else if body_stripped.chars().count() >= 8 {
                score += 0.4;
            }
            if pairing_support::is_legal_citation_continuation(body_stripped) {
                score -= 1.2;
            }
        } else if !(token.form == "symbol" && line.page <= SYMBOL_START_PAGE_LIMIT) {
            score -= 0.9;
        }
        if line.note_column_fit {
            score += 0.7;
        }
        if line.small_font {
            score += 0.4;
        }
        let junk_prefix = char_slice(&line.text, token.pre_start, token.start)
            .trim()
            .to_owned();
        if !junk_prefix.is_empty() {
            score -= 0.6;
        }
        let ref_supported = token.value.is_some_and(|value| {
            ref_pages
                .get(&value)
                .is_some_and(|pages| pages.iter().any(|page| line.page.abs_diff(*page) <= 1))
        });
        if ref_supported {
            score += 0.9;
        }
        let candidate = Candidate {
            line: line_index,
            start: token.start,
            end: token.end,
            observed: token.observed.clone(),
            value: token.value,
            symbol: token.symbol,
            form: token.form,
            score,
            reason: "line_start_label",
            repaired: !junk_prefix.is_empty(),
            repair_kind: if junk_prefix.is_empty() {
                ""
            } else {
                "junk_prefix_stripped"
            },
            requires_visual_cue: false,
            flags: CandidateFlags {
                ref_supported,
                ..CandidateFlags::default()
            },
        };
        candidates.push(candidate.clone());
        if token.form == "plain" && token.post.is_empty() && token.observed.chars().count() >= 2 {
            for cut in 1..token.observed.chars().count() {
                let head = char_slice(&token.observed, 0, cut);
                let rest = format!(
                    "{}{}",
                    char_slice(&token.observed, cut, token.observed.chars().count()),
                    body
                );
                let Some(head_value) = numeric_value(&head) else {
                    continue;
                };
                if !volume_cite_start(&rest) {
                    continue;
                }
                let mut split = candidate.clone();
                split.end = split.start + cut;
                split.observed = head;
                split.value = Some(head_value);
                split.score -= 1.1;
                split.reason = "glued_volume_split";
                split.flags.volume_split = true;
                candidates.push(split);
                break;
            }
        }
    }
    let mut by_value: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, candidate) in candidates.iter().enumerate() {
        if !candidate.flags.volume_split {
            if let Some(value) = candidate.value {
                by_value.entry(value).or_default().push(index);
            }
        }
    }
    for indexes in by_value.values().filter(|indexes| indexes.len() >= 2) {
        for &index in indexes {
            let candidate = &candidates[index];
            let continuation = char_slice(
                &lines[candidate.line].text,
                candidate.start,
                lines[candidate.line].text.chars().count(),
            );
            if volume_cite_start(&continuation)
                && indexes.iter().any(|other| {
                    *other != index
                        && lines[candidates[*other].line]
                            .idx
                            .abs_diff(lines[candidate.line].idx)
                            <= 3
                })
            {
                candidates[index].score -= 1.2;
            }
        }
    }
    candidates
}

fn select_backbone(candidates: &[Candidate], lines: &[PairLine]) -> (Vec<Candidate>, Value) {
    let numeric = candidates
        .iter()
        .enumerate()
        .filter_map(|(index, candidate)| {
            Some(NumericSequenceCandidate {
                index,
                value: candidate.value?,
                position: candidate.pos(lines),
                page: lines[candidate.line].page,
                score: candidate.score,
                start_supported: candidate.flags.ref_supported,
            })
        })
        .collect::<Vec<_>>();
    let candidate_count = numeric.len();
    if candidate_count == 0 {
        return (Vec::new(), json!({"candidate_count": 0}));
    }
    let selected = select_numeric_sequence(numeric, NumericSequencePolicy::FootnoteBackbone);
    let chain = selected
        .indices
        .into_iter()
        .map(|index| candidates[index].clone())
        .collect::<Vec<_>>();
    (
        chain.clone(),
        json!({
            "candidate_count": candidate_count,
            "selected_count": chain.len(),
            "chain_score": (selected.score * 1_000.0).round_ties_even() / 1_000.0,
            "first_value": chain.first().and_then(|candidate| candidate.value),
            "last_value": chain.last().and_then(|candidate| candidate.value),
        }),
    )
}

fn trim_unsupported_tail(mut segment: Vec<Candidate>) -> Vec<Candidate> {
    while segment.len() >= 2 {
        let tail = &segment[segment.len() - 1];
        let prior = &segment[segment.len() - 2];
        let gap = tail
            .value
            .unwrap_or(0)
            .saturating_sub(prior.value.unwrap_or(0));
        if gap > 20 && tail.score < 3.2 && !tail.flags.ref_supported {
            segment.pop();
        } else {
            break;
        }
    }
    segment
}

fn select_segments(candidates: &[Candidate], lines: &[PairLine]) -> (Vec<Vec<Candidate>>, Value) {
    let mut remaining: Vec<Candidate> = candidates
        .iter()
        .filter(|candidate| candidate.value.is_some())
        .cloned()
        .collect();
    let mut segments: Vec<Vec<Candidate>> = Vec::new();
    let mut diagnostics = Vec::new();
    for _ in 0..6 {
        let (chain, diagnostic) = select_backbone(&remaining, lines);
        if chain.is_empty() {
            break;
        }
        if !segments.is_empty() {
            let average =
                chain.iter().map(|candidate| candidate.score).sum::<f64>() / chain.len() as f64;
            let first = chain[0].value.unwrap_or(0);
            let span = (
                lines[chain[0].line].idx,
                lines[chain.last().unwrap().line].idx,
            );
            let overlaps = segments.iter().any(|segment| {
                let other = (
                    lines[segment[0].line].idx,
                    lines[segment.last().unwrap().line].idx,
                );
                !(span.1 < other.0 || other.1 < span.0)
            });
            let strong_short = first == 1
                && average >= 3.2
                && (chain.len() >= 2
                    || (lines[chain[0].line].note_sequence_restart
                        && chain[0].flags.ref_supported));
            if (chain.len() < 4 && !strong_short) || average < 2.0 || first > 3 || overlaps {
                break;
            }
        }
        let used: HashSet<usize> = chain.iter().map(|candidate| candidate.line).collect();
        remaining.retain(|candidate| !used.contains(&candidate.line));
        diagnostics.push(diagnostic);
        segments.push(chain);
    }
    segments.sort_by_key(|segment| lines[segment[0].line].idx);
    let count = segments.len();
    (
        segments,
        json!({"segments": diagnostics, "segment_count": count}),
    )
}

fn confusable_variants(value: &str) -> HashSet<u32> {
    if value.is_empty() || value.chars().count() > 3 {
        return HashSet::new();
    }
    let mut result = String::new();
    for character in value.chars() {
        let mapped = normalized_digit(character).or(match character {
            'l' | 'I' | 'i' | '|' | '!' | 'í' | 't' => Some('1'),
            'o' | 'O' | '°' | 'º' | 'ð' | 'D' | 'Q' => Some('0'),
            's' | 'S' => Some('5'),
            'B' => Some('8'),
            'G' | 'b' => Some('6'),
            'Z' | 'z' => Some('2'),
            'g' | 'q' => Some('9'),
            'A' => Some('4'),
            'T' | '?' | 'n' => Some('7'),
            'J' => Some('3'),
            _ => None,
        });
        let Some(mapped) = mapped else {
            return HashSet::new();
        };
        result.push(mapped);
    }
    result
        .parse::<u32>()
        .ok()
        .filter(|value| (1..=MAX_VALUE).contains(value))
        .into_iter()
        .collect()
}

fn gap_line_score(line: &PairLine, body: &str) -> f64 {
    let mut score = if line.zone == Zone::Note && !line.region_witness_demoted {
        3.0
    } else {
        0.6
    };
    if line.note_column_fit {
        score += 0.7;
    }
    if pairing_support::has_legal_citation_cue(body) {
        score += 0.7;
    }
    score
}

fn gap_repair_tokens(line: &PairLine) -> Vec<(String, usize, usize)> {
    let mut tokens = Vec::new();
    if let Some(token) = label_token(&line.text) {
        if token.value.is_some() && matches!(token.form, "plain" | "sup") {
            tokens.push((token.observed, token.start, token.end));
        }
    }
    let values = chars(&line.text);
    let lead = values
        .iter()
        .take_while(|value| value.is_whitespace())
        .count();
    let mut end = lead;
    while end < values.len()
        && end - lead < 3
        && (values[end].is_ascii_digit() || "lIi|!íoO°ºDQsSBGbZzgqATnJt?".contains(values[end]))
    {
        end += 1;
    }
    if end > lead {
        let observed: String = values[lead..end].iter().collect();
        let has_digit = observed.chars().any(|value| value.is_ascii_digit());
        let follow_ok = values
            .get(end)
            .is_none_or(|value| value.is_whitespace() || *value == '.');
        if has_digit
            && follow_ok
            && !tokens
                .iter()
                .any(|(_, start, stop)| *start == lead && *stop == end)
        {
            tokens.push((observed, lead, end));
        }
    }
    tokens
}

fn gap_repair_claim(
    line_index: usize,
    line: &PairLine,
    unclaimed: &HashSet<u32>,
    last_value: u32,
) -> Option<Candidate> {
    for (observed, start, end) in gap_repair_tokens(line) {
        let exact = numeric_value(&observed)
            .filter(|value| unclaimed.contains(value) && *value > last_value);
        let mut variants: Vec<u32> = confusable_variants(&observed)
            .into_iter()
            .filter(|value| unclaimed.contains(value) && *value > last_value)
            .collect();
        variants.sort_unstable();
        let (value, repair_kind) = if let Some(value) = exact {
            (value, "weak_form_promoted")
        } else {
            (variants.first().copied()?, "confusable_value_repair")
        };
        let body = char_slice(&line.text, end, line.text.chars().count())
            .trim_start_matches(|character: char| " .)]".contains(character))
            .to_owned();
        let score = gap_line_score(line, &body);
        if body.chars().count() < 4 || score < 1.2 {
            continue;
        }
        return Some(Candidate {
            line: line_index,
            start,
            end,
            observed,
            value: Some(value),
            symbol: String::new(),
            form: "plain",
            score,
            reason: "sequence_gap_glyph_repair",
            repaired: true,
            repair_kind,
            requires_visual_cue: false,
            flags: CandidateFlags::default(),
        });
    }
    None
}

fn sequence_position_rekey(
    window: &[usize],
    residual: &[u32],
    lines: &[PairLine],
    used_lines: &mut HashSet<usize>,
) -> Vec<Candidate> {
    if residual.is_empty() || residual.len() > 8 {
        return Vec::new();
    }
    let residual_set: HashSet<u32> = residual.iter().copied().collect();
    let mut qualifying = Vec::<(usize, String, usize, usize)>::new();
    for &line_index in window {
        if used_lines.contains(&line_index) {
            continue;
        }
        let line = &lines[line_index];
        let Some(token) = label_token(&line.text) else {
            continue;
        };
        if !matches!(token.form, "plain" | "sup") {
            continue;
        }
        let Some(value) = token.value else {
            continue;
        };
        if residual_set.contains(&value) || value >= residual[0] {
            return Vec::new();
        }
        let body = char_slice(&line.text, token.end, line.text.chars().count())
            .trim_start_matches(|character: char| " .)]".contains(character))
            .to_owned();
        if body.chars().count() < 4 || pairing_support::is_legal_citation_continuation(&body) {
            continue;
        }
        if gap_line_score(line, &body) < 1.2 {
            continue;
        }
        qualifying.push((line_index, token.observed, token.start, token.end));
    }
    if qualifying.len() != residual.len() {
        return Vec::new();
    }
    residual
        .iter()
        .copied()
        .zip(qualifying)
        .map(|(value, (line, observed, start, end))| {
            used_lines.insert(line);
            Candidate {
                line,
                start,
                end,
                observed,
                value: Some(value),
                symbol: String::new(),
                form: "plain",
                score: if lines[line].zone == Zone::Note && !lines[line].region_witness_demoted {
                    3.0
                } else {
                    0.6
                },
                reason: "sequence_position_rekey",
                repaired: true,
                repair_kind: "sequence_position_rekey",
                requires_visual_cue: false,
                flags: CandidateFlags::default(),
            }
        })
        .collect()
}

fn repair_gaps(
    chain: Vec<Candidate>,
    lines: &[PairLine],
    used_lines: &mut HashSet<usize>,
) -> (Vec<Candidate>, Vec<Value>) {
    if chain.is_empty() {
        return (chain, Vec::new());
    }
    let mut windows = Vec::new();
    let mut holes = Vec::new();
    if chain[0].value.is_some_and(|value| (2..=9).contains(&value)) {
        windows.push((0, lines[chain[0].line].idx, 1, chain[0].value.unwrap() - 1));
    } else if chain[0].value.is_some_and(|value| value > 9) {
        holes.push(json!({
            "values_before_first": chain[0].value.unwrap() - 1,
            "reason": "chain_starts_above_one",
        }));
    }
    for pair in chain.windows(2) {
        let left = pair[0].value.unwrap();
        let right = pair[1].value.unwrap();
        if right > left + 1 {
            windows.push((
                lines[pair[0].line].idx + 1,
                lines[pair[1].line].idx,
                left + 1,
                right - 1,
            ));
        }
    }
    let mut repairs = Vec::new();
    for (start, end, low, high) in windows {
        let window: Vec<usize> = lines
            .iter()
            .enumerate()
            .skip(start)
            .take(end.saturating_sub(start))
            .filter(|(line_index, line)| {
                !used_lines.contains(line_index)
                    && ((line.zone == Zone::Note && !line.region_witness_demoted)
                        || ((line.note_column_fit
                            || matches!(line.zone, Zone::Body | Zone::Other))
                            && pairing_support::has_legal_citation_cue(&line.text)))
            })
            .map(|(line_index, _)| line_index)
            .collect();
        let mut unclaimed: HashSet<u32> = (low..=high).collect();
        let mut last_value = low.saturating_sub(1);
        let mut claims = Vec::<(usize, Candidate)>::new();
        for (position, &line_index) in window.iter().enumerate() {
            if unclaimed.is_empty() {
                break;
            }
            if let Some(candidate) =
                gap_repair_claim(line_index, &lines[line_index], &unclaimed, last_value)
            {
                let value = candidate.value.expect("gap value");
                claims.push((position, candidate.clone()));
                repairs.push(candidate);
                unclaimed.remove(&value);
                last_value = value;
                used_lines.insert(line_index);
            }
        }
        let mut previous_position = 0;
        let mut previous_value = low.saturating_sub(1);
        let mut intervals = Vec::<(Vec<usize>, u32, u32)>::new();
        for (position, candidate) in claims {
            let value = candidate.value.expect("gap value");
            if value > previous_value + 1 {
                intervals.push((
                    window[previous_position..position].to_vec(),
                    previous_value + 1,
                    value - 1,
                ));
            }
            previous_position = position + 1;
            previous_value = value;
        }
        if previous_value < high {
            intervals.push((
                window[previous_position..].to_vec(),
                previous_value + 1,
                high,
            ));
        }
        for (interval, interval_low, interval_high) in intervals {
            let residual: Vec<u32> = (interval_low..=interval_high)
                .filter(|value| unclaimed.contains(value))
                .collect();
            let rekeyed = sequence_position_rekey(&interval, &residual, lines, used_lines);
            if rekeyed.is_empty() {
                for value in residual {
                    holes.push(json!({
                        "value": value,
                        "reason": "no_confusable_glyph_in_window",
                    }));
                }
            } else {
                for candidate in &rekeyed {
                    if let Some(value) = candidate.value {
                        unclaimed.remove(&value);
                    }
                }
            }
            repairs.extend(rekeyed);
        }
    }
    let mut result = chain;
    result.extend(repairs);
    result.sort_by_key(|candidate| candidate.value);
    (result, holes)
}

fn note_zone_column_split(lines: &[PairLine], page: u32) -> Option<f64> {
    let candidates: Vec<&PairLine> = lines
        .iter()
        .filter(|line| {
            line.page == page
                && line.zone == Zone::Note
                && line.page_width > 0.0
                && line.page_height > 0.0
                && (line.bbox[2] - line.bbox[0]) / line.page_width <= 0.55
        })
        .collect();
    let boxed = lines
        .iter()
        .filter(|line| line.page == page && line.zone == Zone::Note)
        .count();
    if candidates.len() < 6 || boxed == 0 || 1.0 - candidates.len() as f64 / boxed as f64 > 0.40 {
        return None;
    }
    let mut centers: Vec<f64> = candidates
        .iter()
        .map(|line| (line.bbox[0] + line.bbox[2]) / 2.0 / line.page_width)
        .collect();
    centers.sort_by(f64::total_cmp);
    let (left, right) = centers
        .windows(2)
        .map(|pair| (pair[0], pair[1]))
        .max_by(|left, right| (left.1 - left.0).total_cmp(&(right.1 - right.0)))?;
    let split = (left + right) / 2.0;
    if right - left < 0.12 || !(0.25..=0.75).contains(&split) {
        return None;
    }
    let left_lines: Vec<&&PairLine> = candidates
        .iter()
        .filter(|line| (line.bbox[0] + line.bbox[2]) / 2.0 / line.page_width < split)
        .collect();
    let right_lines: Vec<&&PairLine> = candidates
        .iter()
        .filter(|line| (line.bbox[0] + line.bbox[2]) / 2.0 / line.page_width >= split)
        .collect();
    if left_lines.len() < 3 || right_lines.len() < 3 {
        return None;
    }
    let left_y0 = left_lines
        .iter()
        .map(|line| line.bbox[1] / line.page_height)
        .min_by(f64::total_cmp)?;
    let left_y1 = left_lines
        .iter()
        .map(|line| line.bbox[3] / line.page_height)
        .max_by(f64::total_cmp)?;
    let right_y0 = right_lines
        .iter()
        .map(|line| line.bbox[1] / line.page_height)
        .min_by(f64::total_cmp)?;
    let right_y1 = right_lines
        .iter()
        .map(|line| line.bbox[3] / line.page_height)
        .max_by(f64::total_cmp)?;
    let span = left_y1.max(right_y1) - left_y0.min(right_y0);
    let overlap = left_y1.min(right_y1) - left_y0.max(right_y0);
    if span <= 0.0 || overlap.max(0.0) / span < 0.30 {
        return None;
    }
    let mut left_widths: Vec<f64> = left_lines
        .iter()
        .map(|line| (line.bbox[2] - line.bbox[0]) / line.page_width)
        .collect();
    let mut right_widths: Vec<f64> = right_lines
        .iter()
        .map(|line| (line.bbox[2] - line.bbox[0]) / line.page_width)
        .collect();
    left_widths.sort_by(f64::total_cmp);
    right_widths.sort_by(f64::total_cmp);
    let left_width = left_widths[left_widths.len() / 2];
    let right_width = right_widths[right_widths.len() / 2];
    let wide = left_width.max(right_width);
    if wide > 0.0 && left_width.min(right_width) / wide < 0.60 {
        None
    } else {
        Some(split)
    }
}

fn recover_out_of_order_labels(
    mut chain: Vec<Candidate>,
    label_candidates: &[Candidate],
    used_lines: &mut HashSet<usize>,
    lines: &[PairLine],
) -> Vec<Candidate> {
    if chain.len() < 2 {
        return chain;
    }
    let have: HashSet<u32> = chain
        .iter()
        .filter_map(|candidate| candidate.value)
        .collect();
    let Some(low) = have.iter().min().copied() else {
        return chain;
    };
    let high = have.iter().max().copied().unwrap_or(low);
    let min_page = chain
        .iter()
        .map(|candidate| lines[candidate.line].page)
        .min()
        .unwrap_or(0);
    let max_page = chain
        .iter()
        .map(|candidate| lines[candidate.line].page)
        .max()
        .unwrap_or(0);
    let mut by_value: HashMap<u32, Vec<Candidate>> = HashMap::new();
    for candidate in label_candidates {
        let Some(value) = candidate.value else {
            continue;
        };
        let line = &lines[candidate.line];
        let noteish = (line.zone == Zone::Note && !line.region_witness_demoted)
            || ((line.note_column_fit || matches!(line.zone, Zone::Body | Zone::Other))
                && pairing_support::has_legal_citation_cue(&line.text));
        if candidate.symbol.is_empty()
            && candidate.form == "plain"
            && !have.contains(&value)
            && low < value
            && value < high
            && !used_lines.contains(&candidate.line)
            && (min_page..=max_page).contains(&line.page)
            && noteish
            && candidate.score >= 1.2
        {
            by_value.entry(value).or_default().push(candidate.clone());
        }
    }
    let mut restored = Vec::new();
    for options in by_value.values_mut() {
        options.sort_by(|left, right| right.score.total_cmp(&left.score));
        if options.len() > 1 && options[0].score - options[1].score < 0.3 {
            continue;
        }
        let candidate = options[0].clone();
        let Some(split) = note_zone_column_split(lines, lines[candidate.line].page) else {
            continue;
        };
        let mut page_labels: Vec<&Candidate> = chain
            .iter()
            .filter(|entry| lines[entry.line].page == lines[candidate.line].page)
            .collect();
        page_labels.push(&candidate);
        page_labels.sort_by_key(|entry| entry.value);
        let mut prior: Option<(usize, f64)> = None;
        let consistent = page_labels.into_iter().all(|entry| {
            let line = &lines[entry.line];
            let rank = (
                usize::from((line.bbox[0] + line.bbox[2]) / 2.0 / line.page_width >= split),
                line.bbox[1],
            );
            let valid = prior
                .is_none_or(|value| rank.0 > value.0 || (rank.0 == value.0 && rank.1 > value.1));
            prior = Some(rank);
            valid
        });
        if consistent {
            let mut candidate = candidate;
            candidate.repaired = true;
            candidate.repair_kind = "out_of_order_label";
            used_lines.insert(candidate.line);
            restored.push(candidate);
        }
    }
    chain.extend(restored);
    chain.sort_by_key(|candidate| candidate.value);
    chain
}

fn endnote_mode(chain: &[Candidate], lines: &[PairLine]) -> bool {
    if chain.len() < ENDNOTE_MIN_LABELS {
        return false;
    }
    let max_page = chain
        .iter()
        .map(|candidate| lines[candidate.line].page)
        .max()
        .unwrap_or(0);
    max_page > 1
        && chain
            .iter()
            .filter(|candidate| lines[candidate.line].page as f64 > 0.75 * max_page as f64)
            .count() as f64
            / chain.len() as f64
            >= 0.7
}

fn visual_label_cue(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:diagram|figure|chart|table|graph|map|illustration|image|photo)")
            .unwrap()
    })
    .is_match(text)
}

#[derive(Debug, Clone, Copy)]
struct RefTail {
    value: u32,
    reference: usize,
    parent: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
struct RefState {
    position: Option<(usize, usize)>,
    score: f64,
    tail: Option<usize>,
}

fn select_refs(
    backbone: &[Candidate],
    refs: &[Candidate],
    lines: &[PairLine],
    endnote: bool,
) -> (HashMap<u32, Candidate>, BTreeMap<String, usize>) {
    let mut drop_reasons = BTreeMap::<String, usize>::new();
    let labels: HashMap<u32, &Candidate> = backbone
        .iter()
        .filter_map(|candidate| candidate.value.map(|value| (value, candidate)))
        .collect();
    let label_spans: HashSet<(usize, usize)> = backbone
        .iter()
        .map(|candidate| (candidate.line, candidate.start))
        .collect();
    let mut candidates_by_value: HashMap<u32, Vec<usize>> = HashMap::new();
    for (index, candidate) in refs.iter().enumerate() {
        let Some(value) = candidate.value else {
            continue;
        };
        if label_spans.contains(&(candidate.line, candidate.start)) {
            *drop_reasons
                .entry("is_selected_label_span".to_owned())
                .or_default() += 1;
            continue;
        }
        let Some(label) = labels.get(&value) else {
            *drop_reasons
                .entry("no_selected_label".to_owned())
                .or_default() += 1;
            continue;
        };
        let table_superscript = candidate.form == "sup"
            && lines[candidate.line]
                .region_type
                .to_lowercase()
                .contains("table");
        if candidate.requires_visual_cue && !table_superscript {
            let label_body = char_slice(
                &lines[label.line].text,
                label.end,
                lines[label.line].text.chars().count(),
            );
            if !visual_label_cue(&label_body) {
                *drop_reasons
                    .entry("visual_zone_without_visual_label_cue".to_owned())
                    .or_default() += 1;
                continue;
            }
        }
        candidates_by_value.entry(value).or_default().push(index);
    }
    let mut values: Vec<u32> = labels.keys().copied().collect();
    values.sort_unstable();
    let max_pool_index = candidates_by_value
        .values()
        .flatten()
        .map(|index| lines[refs[*index].line].idx)
        .max()
        .unwrap_or(0)
        + 1;
    type MatchChoice = (f64, (usize, usize), usize);
    let mut by_value: HashMap<u32, Vec<MatchChoice>> = HashMap::new();
    for value in &values {
        let label = labels[value];
        let mut choices = Vec::new();
        for index in candidates_by_value.get(value).into_iter().flatten() {
            let candidate = &refs[*index];
            let page_delta =
                i64::from(lines[label.line].page) - i64::from(lines[candidate.line].page);
            if !endnote && labels.len() >= ENDNOTE_MIN_LABELS && page_delta.unsigned_abs() >= 2 {
                continue;
            }
            let proximity = if endnote {
                0.08 * lines[candidate.line].idx as f64 / max_pool_index as f64
            } else if page_delta == 0 {
                1.0
            } else if page_delta == 1 {
                0.35
            } else if page_delta == -1 && lines[candidate.line].order <= 10 {
                -0.2
            } else if page_delta < 0 {
                -2.0 - 0.45 * (-page_delta - 1) as f64
            } else {
                -0.45 * (page_delta - 1) as f64
            };
            choices.push((candidate.score + proximity, candidate.pos(lines), *index));
        }
        choices.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        by_value.insert(*value, choices);
    }
    let mut tails = Vec::<RefTail>::new();
    let mut states = vec![RefState {
        position: None,
        score: 0.0,
        tail: None,
    }];
    for value in values {
        let mut next = states.clone();
        for state in &states {
            for (option_score, position, reference) in
                by_value.get(&value).into_iter().flatten().take(6)
            {
                if *option_score <= -1.0
                    || state.position.is_some_and(|previous| *position <= previous)
                {
                    continue;
                }
                let tail = tails.len();
                tails.push(RefTail {
                    value,
                    reference: *reference,
                    parent: state.tail,
                });
                next.push(RefState {
                    position: Some(*position),
                    score: state.score + option_score + 0.5,
                    tail: Some(tail),
                });
            }
        }
        next.sort_by(|left, right| {
            left.position
                .cmp(&right.position)
                .then_with(|| right.score.total_cmp(&left.score))
        });
        let mut best = f64::NEG_INFINITY;
        let mut pruned = Vec::new();
        for state in next {
            if state.score > best + 1e-9 {
                best = state.score;
                pruned.push(state);
            }
        }
        if pruned.len() > 400 {
            pruned.drain(..pruned.len() - 400);
        }
        states = pruned;
    }
    let mut result = HashMap::new();
    let mut final_state = &states[0];
    for state in &states[1..] {
        if state.score > final_state.score {
            final_state = state;
        }
    }
    let mut tail = final_state.tail;
    while let Some(index) = tail {
        let value = tails[index];
        result.insert(value.value, refs[value.reference].clone());
        tail = value.parent;
    }
    for value in labels.keys() {
        if !result.contains_key(value) {
            if let Some(count) = candidates_by_value.get(value).map(Vec::len) {
                *drop_reasons
                    .entry("window_conflict".to_owned())
                    .or_default() += count;
            }
        }
    }
    (result, drop_reasons)
}

fn truncated_value_match(observed: &str, expected: u32) -> bool {
    let Some(digits) = observed
        .chars()
        .map(normalized_digit)
        .collect::<Option<String>>()
    else {
        return false;
    };
    let expected = expected.to_string();
    digits != expected && (expected.starts_with(&digits) || expected.ends_with(&digits))
}

fn repair_missing_refs(
    chosen: &mut HashMap<u32, Candidate>,
    backbone: &[Candidate],
    refs: &[Candidate],
    lines: &[PairLine],
    endnote: bool,
) -> BTreeMap<String, usize> {
    let mut repair_counts = BTreeMap::<String, usize>::new();
    let labels: HashMap<u32, &Candidate> = backbone
        .iter()
        .filter_map(|candidate| candidate.value.map(|value| (value, candidate)))
        .collect();
    let mut values: Vec<u32> = labels.keys().copied().collect();
    values.sort_unstable();
    let mut taken: HashSet<(usize, usize, usize)> = chosen
        .values()
        .map(|candidate| (candidate.line, candidate.start, candidate.end))
        .collect();
    let label_spans: HashSet<(usize, usize)> = backbone
        .iter()
        .map(|candidate| (candidate.line, candidate.start))
        .collect();
    for (index, value) in values.iter().copied().enumerate() {
        if chosen.contains_key(&value) {
            continue;
        }
        let floor = values[..index]
            .iter()
            .rev()
            .find_map(|prior| chosen.get(prior).map(|candidate| candidate.pos(lines)));
        let ceiling = values[index + 1..]
            .iter()
            .find_map(|next| chosen.get(next).map(|candidate| candidate.pos(lines)));
        let in_window = |candidate: &Candidate| {
            let position = candidate.pos(lines);
            floor.is_none_or(|bound| position > bound)
                && ceiling.is_none_or(|bound| position < bound)
        };
        let label = labels[&value];
        let immediate_window = index > 0
            && index + 1 < values.len()
            && values[index - 1] == value - 1
            && values[index + 1] == value + 1
            && chosen.contains_key(&(value - 1))
            && chosen.contains_key(&(value + 1));
        let substitution_sites: Vec<&Candidate> = if immediate_window {
            refs.iter()
                .filter(|candidate| {
                    candidate.symbol.is_empty()
                        && candidate.value.is_some()
                        && !candidate.requires_visual_cue
                        && !taken.contains(&(candidate.line, candidate.start, candidate.end))
                        && !label_spans.contains(&(candidate.line, candidate.start))
                        && in_window(candidate)
                        && (endnote
                            || lines[label.line].page.abs_diff(lines[candidate.line].page) <= 1)
                        && at_line_end(&lines[candidate.line].text, candidate.end)
                })
                .collect()
        } else {
            Vec::new()
        };
        let sole_substitution = (substitution_sites.len() == 1).then(|| {
            let candidate = substitution_sites[0];
            (candidate.line, candidate.start, candidate.end)
        });
        let mut best: Option<(f64, Candidate, &'static str)> = None;
        for candidate in refs {
            if !candidate.symbol.is_empty()
                || candidate.value.is_none()
                || candidate.requires_visual_cue
                || taken.contains(&(candidate.line, candidate.start, candidate.end))
                || label_spans.contains(&(candidate.line, candidate.start))
                || !in_window(candidate)
                || (!endnote && lines[label.line].page.abs_diff(lines[candidate.line].page) > 1)
            {
                continue;
            }
            let left = candidate
                .start
                .checked_sub(1)
                .and_then(|position| chars(&lines[candidate.line].text).get(position).copied());
            let repair_kind = if candidate.value == Some(value) {
                "window_rescued"
            } else if confusable_variants(&candidate.observed).contains(&value) {
                "confusable_value_repair"
            } else if truncated_value_match(&candidate.observed, value)
                && (chosen.contains_key(&value.saturating_sub(1))
                    || labels.contains_key(&value.saturating_sub(1))
                    || value == values[0])
                && (chosen.contains_key(&(value + 1))
                    || labels.contains_key(&(value + 1))
                    || value == *values.last().unwrap())
                && (at_line_end(&lines[candidate.line].text, candidate.end)
                    || left.is_some_and(|character| "\"'’”)]»".contains(character)))
                && !left.is_some_and(|character| character.is_uppercase())
            {
                "truncated_value_repair"
            } else if sole_substitution == Some((candidate.line, candidate.start, candidate.end)) {
                "sequence_window_substitution_repair"
            } else {
                continue;
            };
            let score = candidate.score - 0.8
                + if lines[candidate.line].page == lines[label.line].page {
                    1.0
                } else {
                    0.0
                };
            if score <= 0.0 || best.as_ref().is_some_and(|prior| prior.0 >= score) {
                continue;
            }
            let mut repaired = candidate.clone();
            if repair_kind != "window_rescued" {
                repaired.value = Some(value);
                repaired.repaired = true;
                repaired.repair_kind = repair_kind;
                repaired.score -= 0.8;
            }
            best = Some((score, repaired, repair_kind));
        }
        if let Some((_, candidate, repair_kind)) = best {
            taken.insert((candidate.line, candidate.start, candidate.end));
            chosen.insert(value, candidate);
            *repair_counts.entry(repair_kind.to_owned()).or_default() += 1;
        }
    }
    repair_counts
}

fn rekey_same_value_ref_runs(
    chosen: &mut HashMap<u32, Candidate>,
    backbone: &[Candidate],
    refs: &[Candidate],
    lines: &[PairLine],
    endnote: bool,
) -> BTreeMap<String, usize> {
    let mut repair_counts = BTreeMap::<String, usize>::new();
    let labels: HashMap<u32, &Candidate> = backbone
        .iter()
        .filter_map(|candidate| candidate.value.map(|value| (value, candidate)))
        .collect();
    let mut values: Vec<u32> = labels.keys().copied().collect();
    values.sort_unstable();
    let mut taken: HashMap<(usize, usize, usize), u32> = chosen
        .iter()
        .map(|(value, candidate)| ((candidate.line, candidate.start, candidate.end), *value))
        .collect();
    for (w_index, w) in values.iter().copied().enumerate() {
        if !chosen.contains_key(&w) {
            continue;
        }
        let mut run = Vec::new();
        let mut cursor = w_index;
        while cursor > 0 {
            let candidate = values[cursor - 1];
            if candidate != w - run.len() as u32 - 1 || chosen.contains_key(&candidate) {
                break;
            }
            run.push(candidate);
            cursor -= 1;
        }
        if run.is_empty() {
            continue;
        }
        run.reverse();
        let floor = if cursor > 0 {
            chosen
                .get(&values[cursor - 1])
                .map(|candidate| candidate.pos(lines))
        } else {
            None
        };
        if floor.is_none() && run[0] != values[0] {
            continue;
        }
        let ceiling = values[w_index + 1..]
            .iter()
            .find_map(|value| chosen.get(value).map(|candidate| candidate.pos(lines)));
        let mut sites = Vec::new();
        let mut valid = true;
        for candidate in refs {
            if candidate.value != Some(w)
                || !candidate.symbol.is_empty()
                || candidate.requires_visual_cue
                || !matches!(candidate.form, "tight" | "sup")
                || candidate.score <= 0.0
                || floor.is_some_and(|bound| candidate.pos(lines) <= bound)
                || ceiling.is_some_and(|bound| candidate.pos(lines) >= bound)
            {
                continue;
            }
            if taken
                .get(&(candidate.line, candidate.start, candidate.end))
                .is_some_and(|owner| *owner != w)
            {
                valid = false;
                break;
            }
            sites.push(candidate.clone());
        }
        if !valid || sites.len() != run.len() + 1 {
            continue;
        }
        sites.sort_by_key(|candidate| candidate.pos(lines));
        let mut targets = run;
        targets.push(w);
        if !endnote
            && targets.iter().zip(&sites).any(|(value, candidate)| {
                lines[labels[value].line]
                    .page
                    .abs_diff(lines[candidate.line].page)
                    > 1
            })
        {
            continue;
        }
        for (value, mut candidate) in targets.into_iter().zip(sites) {
            let key = (candidate.line, candidate.start, candidate.end);
            if chosen
                .get(&value)
                .is_some_and(|selected| (selected.line, selected.start, selected.end) == key)
            {
                continue;
            }
            candidate.repaired = true;
            candidate.repair_kind = "sequence_position_rekey";
            candidate.value = Some(value);
            chosen.insert(value, candidate);
            taken.insert(key, value);
            *repair_counts
                .entry("sequence_position_rekey".to_owned())
                .or_default() += 1;
        }
    }
    repair_counts
}

fn repeated_refs(
    chosen: &HashMap<u32, Candidate>,
    refs: &[Candidate],
    lines: &[PairLine],
) -> HashMap<u32, Vec<Candidate>> {
    let primary: HashSet<(usize, usize, usize)> = chosen
        .values()
        .map(|candidate| (candidate.line, candidate.start, candidate.end))
        .collect();
    let mut result: HashMap<u32, Vec<Candidate>> = HashMap::new();
    for candidate in refs {
        let Some(value) = candidate.value else {
            continue;
        };
        let Some(selected) = chosen.get(&value) else {
            continue;
        };
        if candidate.form != "sup"
            || primary.contains(&(candidate.line, candidate.start, candidate.end))
            || candidate.pos(lines) <= selected.pos(lines)
        {
            continue;
        }
        result.entry(value).or_default().push(candidate.clone());
    }
    result
}

fn label_only_supported(backbone: &[Candidate], position: usize, lines: &[PairLine]) -> bool {
    let label = &backbone[position];
    label.zone_is_noteish(lines)
        && (label.score >= 1.5
            || [
                position.checked_sub(1),
                (position + 1 < backbone.len()).then_some(position + 1),
            ]
            .into_iter()
            .flatten()
            .any(|neighbor| {
                let neighbor = &backbone[neighbor];
                label.value.unwrap().abs_diff(neighbor.value.unwrap()) == 1
                    && lines[label.line].page.abs_diff(lines[neighbor.line].page) <= 1
            }))
}

fn build_pairs(pages: &[Page]) -> (Vec<Pair>, Vec<PairLine>, Value) {
    let lines = legal_pdf_support::profile::measure("pair.build_lines", || build_lines(pages));
    let refs = legal_pdf_support::profile::measure("pair.extract_refs", || extract_refs(&lines));
    let labels = legal_pdf_support::profile::measure("pair.extract_labels", || {
        extract_labels(&lines, &refs)
    });
    let (segments, backbone_summary) =
        legal_pdf_support::profile::measure("pair.select_segments", || {
            select_segments(&labels, &lines)
        });
    let mut segments: Vec<Vec<Candidate>> = segments
        .into_iter()
        .map(trim_unsupported_tail)
        .filter(|segment| !segment.is_empty())
        .collect();
    let mut used: HashSet<usize> = segments
        .iter()
        .flatten()
        .map(|candidate| candidate.line)
        .collect();
    let mut sequence_holes = Vec::<Value>::new();
    for segment in &mut segments {
        let (repaired, holes) = repair_gaps(std::mem::take(segment), &lines, &mut used);
        sequence_holes.extend(holes);
        *segment = recover_out_of_order_labels(repaired, &labels, &mut used, &lines);
    }
    segments.retain(|segment| !segment.is_empty());
    let endnote = segments
        .first()
        .is_some_and(|segment| endnote_mode(segment, &lines));

    let mut pools = Vec::<Vec<Candidate>>::new();
    let mut previous_boundary = None;
    for (segment_index, segment) in segments.iter().enumerate() {
        let last_page = segment
            .last()
            .map(|candidate| lines[candidate.line].page)
            .unwrap_or(0);
        let boundary = if segment_index + 1 == segments.len() {
            usize::MAX
        } else {
            lines
                .iter()
                .filter(|line| line.page <= last_page)
                .map(|line| line.idx)
                .max()
                .unwrap_or(0)
        };
        let label_pages: HashMap<u32, HashSet<u32>> =
            segment.iter().fold(HashMap::new(), |mut pages, candidate| {
                if let Some(value) = candidate.value {
                    pages
                        .entry(value)
                        .or_insert_with(HashSet::new)
                        .insert(lines[candidate.line].page);
                }
                pages
            });
        let pool = refs
            .iter()
            .filter(|candidate| {
                let position = lines[candidate.line].idx;
                previous_boundary.is_none_or(|prior| position > prior)
                    && position <= boundary
                    && (!candidate.flags.paren_ref
                        || candidate.value.is_some_and(|value| {
                            label_pages
                                .get(&value)
                                .is_some_and(|pages| pages.contains(&lines[candidate.line].page))
                        }))
            })
            .cloned()
            .collect();
        pools.push(pool);
        previous_boundary = Some(boundary);
    }

    let mut chosen_by_segment = Vec::<HashMap<u32, Candidate>>::new();
    let mut repeated_by_segment = Vec::<HashMap<u32, Vec<Candidate>>>::new();
    let mut suppressed = HashSet::<usize>::new();
    let mut suppressed_reasons = HashMap::<usize, &'static str>::new();
    let mut numbered_diagnostics = Vec::<Value>::new();
    let mut ref_drop_reasons = BTreeMap::<String, usize>::new();
    let mut ref_repair_counts = BTreeMap::<String, usize>::new();
    for (segment_index, (segment, pool)) in segments.iter().zip(&pools).enumerate() {
        let (mut chosen, drops) = select_refs(segment, pool, &lines, endnote);
        for (reason, count) in drops {
            *ref_drop_reasons.entry(reason).or_default() += count;
        }
        for (kind, count) in repair_missing_refs(&mut chosen, segment, pool, &lines, endnote) {
            *ref_repair_counts.entry(kind).or_default() += count;
        }
        for (kind, count) in rekey_same_value_ref_runs(&mut chosen, segment, pool, &lines, endnote)
        {
            *ref_repair_counts.entry(kind).or_default() += count;
        }

        let paren_count = segment
            .iter()
            .filter(|candidate| candidate.form == "paren")
            .count();
        if !segment.is_empty() && paren_count * 2 >= segment.len() {
            let label_pages: HashMap<u32, HashSet<u32>> =
                segment.iter().fold(HashMap::new(), |mut pages, candidate| {
                    if let Some(value) = candidate.value {
                        pages
                            .entry(value)
                            .or_insert_with(HashSet::new)
                            .insert(lines[candidate.line].page);
                    }
                    pages
                });
            let same_page = chosen.iter().any(|(value, candidate)| {
                label_pages
                    .get(value)
                    .is_some_and(|pages| pages.contains(&lines[candidate.line].page))
            });
            if !same_page {
                suppressed.insert(segment_index);
                suppressed_reasons.insert(segment_index, "paren_list_segment");
            }
        }

        if !suppressed.contains(&segment_index) && segment.len() >= 20 {
            let noteish = segment
                .iter()
                .filter(|candidate| {
                    lines[candidate.line].zone == Zone::Note
                        && !lines[candidate.line].region_witness_demoted
                })
                .count();
            let small = segment
                .iter()
                .filter(|candidate| lines[candidate.line].small_font)
                .count();
            let segment_pages: HashSet<u32> = segment
                .iter()
                .map(|candidate| lines[candidate.line].page)
                .collect();
            let article_pages: HashSet<u32> = lines.iter().map(|line| line.page).collect();
            let fired = chosen.len() <= (segment.len() / 50).max(1)
                && noteish * 5 < segment.len()
                && small * 5 < segment.len()
                && segment_pages.len() * 10 >= article_pages.len() * 6;
            numbered_diagnostics.push(json!({
                "segment_index": segment_index,
                "labels": segment.len(),
                "chosen_refs": chosen.len(),
                "noteish_labels": noteish,
                "small_font_labels": small,
                "segment_pages": segment_pages.len(),
                "article_pages": article_pages.len(),
                "suppressed": fired,
            }));
            if fired {
                suppressed.insert(segment_index);
                suppressed_reasons.insert(segment_index, "numbered_paragraph_segment");
            }
        }
        repeated_by_segment.push(repeated_refs(&chosen, pool, &lines));
        chosen_by_segment.push(chosen);
    }

    let mut ref_bearing_values = HashSet::<u32>::new();
    for (segment_index, (segment, chosen)) in segments.iter().zip(&chosen_by_segment).enumerate() {
        if !suppressed.contains(&segment_index)
            && !segment.is_empty()
            && chosen.len() * 2 >= segment.len()
        {
            ref_bearing_values.extend(segment.iter().filter_map(|candidate| candidate.value));
        }
    }
    let mut duplicate_diagnostics = Vec::<Value>::new();
    for (segment_index, (segment, chosen)) in segments.iter().zip(&chosen_by_segment).enumerate() {
        if suppressed.contains(&segment_index) || segment.len() < 10 || !chosen.is_empty() {
            continue;
        }
        let values: HashSet<u32> = segment
            .iter()
            .filter_map(|candidate| candidate.value)
            .collect();
        let segment_pages: HashSet<u32> = segment
            .iter()
            .map(|candidate| lines[candidate.line].page)
            .collect();
        let page_lines = lines
            .iter()
            .filter(|line| segment_pages.contains(&line.page))
            .count();
        let fired = !values.is_empty()
            && segment_pages.len() <= 2
            && values.is_subset(&ref_bearing_values)
            && segment.len() * 5 >= page_lines * 2;
        duplicate_diagnostics.push(json!({
            "segment_index": segment_index,
            "labels": segment.len(),
            "segment_pages": segment_pages.len(),
            "page_lines": page_lines,
            "values_covered": values.intersection(&ref_bearing_values).count(),
            "values_total": values.len(),
            "suppressed": fired,
        }));
        if fired {
            suppressed.insert(segment_index);
            suppressed_reasons.insert(segment_index, "duplicate_valueset_zero_ref_segment");
        }
    }

    let mut body_restart_diagnostics = Vec::<Value>::new();
    for (segment_index, (segment, chosen)) in segments.iter().zip(&chosen_by_segment).enumerate() {
        if segment_index == 0 || suppressed.contains(&segment_index) || segment.len() < 6 {
            continue;
        }
        let values: HashSet<u32> = segment
            .iter()
            .filter_map(|candidate| candidate.value)
            .collect();
        let noteish = segment
            .iter()
            .filter(|candidate| {
                lines[candidate.line].zone == Zone::Note || lines[candidate.line].note_column_fit
            })
            .count();
        let fired = !values.is_empty()
            && values.is_subset(&ref_bearing_values)
            && chosen.len() * 3 <= segment.len()
            && (segment.len() - noteish) * 5 >= segment.len() * 3;
        body_restart_diagnostics.push(json!({
            "segment_index": segment_index,
            "labels": segment.len(),
            "chosen_refs": chosen.len(),
            "noteish_labels": noteish,
            "values_covered": values.intersection(&ref_bearing_values).count(),
            "values_total": values.len(),
            "suppressed": fired,
        }));
        if fired {
            suppressed.insert(segment_index);
            suppressed_reasons.insert(segment_index, "body_zone_restart_segment");
        }
    }

    let mut pairs = Vec::new();
    let mut pair_number = 0;
    let mut paired_count = 0;
    let mut skipped = HashMap::<&'static str, usize>::new();
    for (segment_index, ((backbone, chosen), repeated)) in segments
        .iter()
        .zip(&chosen_by_segment)
        .zip(&repeated_by_segment)
        .enumerate()
    {
        if suppressed.contains(&segment_index) {
            *skipped
                .entry(
                    suppressed_reasons
                        .get(&segment_index)
                        .copied()
                        .unwrap_or("paren_list_segment"),
                )
                .or_default() += backbone.len();
            continue;
        }
        for (position, label) in backbone.iter().enumerate() {
            let value = label.value.expect("numeric backbone");
            let primary = chosen.get(&value).cloned();
            if primary.is_none() && !label_only_supported(backbone, position, &lines) {
                *skipped.entry("unsupported_label_only").or_default() += 1;
                continue;
            }
            pair_number += 1;
            let primary_ref = primary
                .as_ref()
                .map(|reference| (reference.line, reference.start, reference.end));
            let mut refs = primary.into_iter().collect::<Vec<_>>();
            if !refs.is_empty() {
                paired_count += 1;
                refs.extend(repeated.get(&value).into_iter().flatten().cloned());
            }
            pairs.push(Pair {
                label: label.clone(),
                refs,
                primary_ref,
                previous_value: position
                    .checked_sub(1)
                    .and_then(|index| backbone[index].value),
                next_value: backbone
                    .get(position + 1)
                    .and_then(|candidate| candidate.value),
                restart_sequence: segment_index + 1,
                endnote,
                pair_id: format!("fnv2-pair-LEGALPDF-document-{pair_number:06}"),
                provenance: if label.repaired && label.repair_kind == "confusable_value_repair" {
                    "article_sequence_gap_glyph_repair"
                } else {
                    "article_sequence_line_start_label"
                }
                .to_owned(),
            });
        }
    }

    let mut used_symbols = HashSet::new();
    for label in labels
        .iter()
        .filter(|candidate| !candidate.symbol.is_empty())
    {
        if label.score < 0.4 {
            continue;
        }
        let mut options: Vec<&Candidate> = refs
            .iter()
            .filter(|reference| {
                reference.symbol == label.symbol
                    && !used_symbols.contains(&reference.pos(&lines))
                    && matches!(
                        i64::from(lines[label.line].page) - i64::from(lines[reference.line].page),
                        0 | 1
                    )
            })
            .collect();
        options.sort_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.pos(&lines).cmp(&right.pos(&lines)))
        });
        let reference = options.first().copied().cloned();
        if reference.is_none()
            && !(lines[label.line].page <= SYMBOL_START_PAGE_LIMIT && label.zone_is_noteish(&lines))
        {
            continue;
        }
        pair_number += 1;
        let primary_ref = reference
            .as_ref()
            .map(|candidate| (candidate.line, candidate.start, candidate.end));
        if let Some(reference) = &reference {
            paired_count += 1;
            used_symbols.insert(reference.pos(&lines));
        }
        pairs.push(Pair {
            label: label.clone(),
            refs: reference.into_iter().collect(),
            primary_ref,
            previous_value: None,
            next_value: None,
            restart_sequence: 1,
            endnote: false,
            pair_id: format!("fnv2-pair-LEGALPDF-document-{pair_number:06}"),
            provenance: "article_start_custom_symbol_label".to_owned(),
        });
    }

    pairs.sort_by_key(|pair| pair.label.pos(&lines));
    let marker_count = pairs.len() + pairs.iter().map(|pair| pair.refs.len()).sum::<usize>();
    let label_only = pairs.len().saturating_sub(paired_count);
    let summary = json!({
        "schema_version": "oajd.footnote_pairing_v2.v1",
        "engine": "tools.footnotes.footnote_pairing_v2",
        "line_count": lines.len(),
        "marker_count": marker_count,
        "safe_marker_count": marker_count,
        "role_counts": {
            "fn_label": pairs.len(),
            "fn_ref": pairs.iter().map(|pair| pair.refs.len()).sum::<usize>()
        },
        "pair_count": paired_count,
        "materialized_pair_count": paired_count,
        "materialized_label_only_count": label_only,
        "pair_status_counts": {"paired": paired_count, "label_only": label_only},
        "label_candidate_count": labels.len(),
        "ref_candidate_count": refs.len(),
        "article_footnote_pair_materialization": {
            "materialized_marker_count": marker_count,
            "materialized_pair_count": paired_count,
            "materialized_label_only_count": label_only,
            "endnote_mode": endnote,
            "segment_count": segments.len(),
            "skipped_marker_counts": skipped,
            "numbered_paragraph_guard": numbered_diagnostics,
            "duplicate_valueset_guard": duplicate_diagnostics,
            "body_zone_restart_guard": body_restart_diagnostics,
            "label_backbone": backbone_summary,
            "sequence_holes": sequence_holes,
            "monotone_ref_sequence": {"drop_reason_counts": ref_drop_reasons},
            "ref_repair_counts": ref_repair_counts,
        },
        "workflow_stage_summary": {
            "stages": [
                "extract_candidates",
                "label_backbone_dp",
                "gap_glyph_repair",
                "out_of_order_label_recovery",
                "monotone_ref_assignment",
                "ref_glyph_repair",
                "false_positive_suppression",
                "custom_symbol_pairs",
                "materialize"
            ]
        },
    });
    (pairs, lines, summary)
}

fn merge_detached_references(pairs: &mut [Pair], lines: &[PairLine]) -> usize {
    let mut existing: HashSet<(usize, usize, usize, String)> = pairs
        .iter()
        .flat_map(|pair| pair.refs.iter())
        .map(|reference| {
            (
                reference.line,
                reference.start,
                reference.end,
                reference.note_id(),
            )
        })
        .collect();
    let mut added = 0;
    for (line_index, line) in lines.iter().enumerate() {
        for detached in &line.detached_references {
            let Some(raw) = detached.get("note_id").and_then(Value::as_str) else {
                continue;
            };
            let value = numeric_value(raw);
            let symbol = value.map_or_else(|| normalize_symbol(raw), |_| String::new());
            if value.is_none() && symbol.is_empty() {
                continue;
            }
            let note_id = value.map_or_else(|| symbol.clone(), |number| number.to_string());
            let start = detached
                .get("start_offset")
                .and_then(Value::as_u64)
                .and_then(|offset| usize::try_from(offset).ok())
                .unwrap_or(0);
            let end = detached
                .get("end_offset")
                .and_then(Value::as_u64)
                .and_then(|offset| usize::try_from(offset).ok())
                .unwrap_or(start);
            let key = (line_index, start, end, note_id.clone());
            if existing.contains(&key) {
                continue;
            }
            let selected = pairs
                .iter()
                .enumerate()
                .filter(|(_, pair)| {
                    pair.label.note_id() == note_id
                        && (lines[pair.label.line].page.abs_diff(line.page) <= 1 || pair.endnote)
                })
                .min_by_key(|(_, pair)| {
                    let label_line = &lines[pair.label.line];
                    (
                        label_line.page.abs_diff(line.page),
                        label_line.order < line.order,
                        label_line.order.abs_diff(line.order),
                    )
                })
                .map(|(index, _)| index);
            let Some(pair_index) = selected else {
                continue;
            };
            pairs[pair_index].refs.push(Candidate {
                line: line_index,
                start,
                end,
                observed: detached
                    .get("selected_text")
                    .and_then(Value::as_str)
                    .unwrap_or(raw)
                    .to_owned(),
                value,
                symbol,
                form: "detached",
                score: 1.5,
                reason: "detached_pdf_superscript",
                repaired: false,
                repair_kind: "",
                requires_visual_cue: false,
                flags: CandidateFlags::default(),
            });
            pairs[pair_index]
                .refs
                .sort_by_key(|reference| reference.pos(lines));
            existing.insert(key);
            added += 1;
        }
    }
    added
}

fn candidate_key(candidate: &Candidate) -> (usize, usize, usize) {
    (candidate.line, candidate.start, candidate.end)
}

fn marker_confidence(value: f64) -> f64 {
    (value * 1_000.0).round_ties_even() / 1_000.0
}

fn normalized_observed(value: &str) -> String {
    value
        .trim()
        .chars()
        .filter_map(|character| {
            normalized_digit(character).or_else(|| {
                (!character.is_whitespace()).then_some(match character {
                    '\u{2217}' | '\u{f02a}' => '*',
                    other => other,
                })
            })
        })
        .collect::<String>()
        .trim_end_matches(['.', ')', ':', ']'])
        .trim_start_matches(['(', '['])
        .to_owned()
}

fn canonical_marker_row(
    candidate: &Candidate,
    lines: &[PairLine],
    role: &str,
    marker_id: String,
    strategy: &str,
    confidence: f64,
) -> Map<String, Value> {
    let line = &lines[candidate.line];
    let note_id = candidate.note_id();
    let source_note_id = {
        let normalized = normalized_observed(&candidate.observed);
        if normalized.is_empty() {
            note_id.clone()
        } else {
            normalized
        }
    };
    let mut row = Map::new();
    row.insert(
        "schema_version".to_owned(),
        json!("oajd.footnote_pairing_v2_marker.v1"),
    );
    row.insert("marker_id".to_owned(), json!(marker_id));
    row.insert("role".to_owned(), json!(role));
    row.insert("safe_to_use".to_owned(), json!(true));
    row.insert("note_id".to_owned(), json!(note_id));
    row.insert("source_note_id".to_owned(), json!(source_note_id));
    row.insert("selected_text".to_owned(), json!(candidate.observed));
    row.insert("visible_marker_value".to_owned(), json!(candidate.observed));
    row.insert(
        "article_context_note_id_repaired".to_owned(),
        json!(candidate.repaired && candidate.repair_kind == "confusable_value_repair"),
    );
    row.insert(
        "repair_lane_status".to_owned(),
        json!(if candidate.repaired && !candidate.repair_kind.is_empty() {
            format!("repaired_{}", candidate.repair_kind)
        } else {
            "visible_glyph".to_owned()
        }),
    );
    row.insert("image_filename".to_owned(), json!(""));
    row.insert("dataset".to_owned(), json!("LEGALPDF"));
    row.insert("article_id".to_owned(), json!("document"));
    row.insert("pdf_page".to_owned(), json!(line.page));
    row.insert("line_id".to_owned(), json!(line.id));
    row.insert("region_id".to_owned(), json!(line.region_id));
    row.insert("reading_order_index".to_owned(), json!(line.order));
    row.insert("start_offset".to_owned(), json!(candidate.start));
    row.insert("end_offset".to_owned(), json!(candidate.end));
    row.insert("line_text".to_owned(), json!(line.text));
    row.insert("region_type".to_owned(), json!(line.region_type));
    row.insert(
        "line_type".to_owned(),
        json!(if line.region_type == "footnote" {
            "footnote"
        } else {
            "paragraph"
        }),
    );
    row.insert(
        "candidate_confidence".to_owned(),
        json!(marker_confidence(confidence)),
    );
    row.insert("candidate_reason".to_owned(), json!(candidate.reason));
    row.insert("pairing_strategy_family".to_owned(), json!(strategy));
    row.insert(
        "label_sequence_guard_status".to_owned(),
        json!(if role == "fn_label" {
            "passed"
        } else {
            "not_applicable"
        }),
    );
    row.insert("protected_span_guard_status".to_owned(), json!("passed"));
    row.insert(
        "materialization_source".to_owned(),
        json!("tools.footnotes.footnote_pairing_v2"),
    );
    row
}

#[allow(clippy::too_many_arguments)]
fn insert_shared_marker_fields(
    row: &mut Map<String, Value>,
    pair_id: &str,
    note_id: &str,
    label_marker_id: &str,
    ref_marker_ids: &[String],
    status: &str,
    same_page: bool,
    sequence_context: &Value,
) {
    row.insert("materialized_pair_id".to_owned(), json!(pair_id));
    row.insert("materialized_pair_status".to_owned(), json!(status));
    row.insert("materialized_note_id".to_owned(), json!(note_id));
    row.insert(
        "materialized_ref_count".to_owned(),
        json!(ref_marker_ids.len()),
    );
    row.insert("materialized_label_count".to_owned(), json!(1));
    row.insert(
        "materialized_pair_scope".to_owned(),
        json!("full_article_sequence_context"),
    );
    row.insert(
        "materialized_label_marker_id".to_owned(),
        json!(label_marker_id),
    );
    row.insert(
        "materialized_ref_marker_ids".to_owned(),
        json!(ref_marker_ids),
    );
    row.insert(
        "materialized_label_same_page_as_ref".to_owned(),
        json!(same_page),
    );
    row.insert(
        "article_sequence_context".to_owned(),
        sequence_context.clone(),
    );
}

fn pair_markers(pairs: &[Pair], lines: &[PairLine]) -> Vec<Value> {
    let mut ordered_pairs: Vec<&Pair> = pairs.iter().collect();
    ordered_pairs.sort_by(|left, right| left.pair_id.cmp(&right.pair_id));
    let mut rows = Vec::<Value>::new();
    let mut marker_sequence = 0;
    for pair in ordered_pairs {
        marker_sequence += 1;
        let label_marker_id = format!("fnv2-label-LEGALPDF-document-{marker_sequence:06}");
        let symbol_pair = !pair.label.symbol.is_empty();
        let label_strategy = if symbol_pair {
            "article_start_custom_symbol_label"
        } else if pair.label.repaired && pair.label.repair_kind == "confusable_value_repair" {
            "article_sequence_gap_glyph_repair"
        } else {
            "article_sequence_line_start_label"
        };
        let label_confidence = if symbol_pair {
            0.9
        } else {
            (0.7 + 0.05 * pair.label.score.max(0.0)).min(0.98)
        };
        let mut label_row = canonical_marker_row(
            &pair.label,
            lines,
            "fn_label",
            label_marker_id.clone(),
            label_strategy,
            label_confidence,
        );
        let mut canonical_refs: Vec<&Candidate> = pair
            .refs
            .iter()
            .filter(|candidate| candidate.form != "detached")
            .collect();
        canonical_refs.sort_by_key(|candidate| {
            (
                pair.primary_ref != Some(candidate_key(candidate)),
                candidate.pos(lines),
            )
        });
        let mut ref_rows = Vec::<Map<String, Value>>::new();
        for (ref_index, reference) in canonical_refs.iter().enumerate() {
            marker_sequence += 1;
            let marker_id = format!("fnv2-ref-LEGALPDF-document-{marker_sequence:06}");
            let primary = pair.primary_ref == Some(candidate_key(reference));
            let strategy = if symbol_pair {
                "article_context_custom_marker_ref_same_page_label"
            } else if !primary {
                "article_context_repeated_superscript_ref"
            } else if reference.requires_visual_cue {
                "article_context_visual_region_ref_same_page_label"
            } else if reference.repaired
                && matches!(
                    reference.repair_kind,
                    "confusable_value_repair" | "truncated_value_repair"
                )
            {
                "article_sequence_ref_value_repair"
            } else if lines[reference.line].page == lines[pair.label.line].page {
                "article_context_body_ref_same_page_label_sequence"
            } else {
                "article_context_body_ref_cross_page_label_sequence"
            };
            let confidence = if symbol_pair {
                0.9
            } else if !primary {
                0.85
            } else {
                (0.65 + 0.06 * reference.score.max(0.0)).min(0.98)
            };
            let mut row =
                canonical_marker_row(reference, lines, "fn_ref", marker_id, strategy, confidence);
            row.insert("valid_repeated_ref".to_owned(), json!(ref_index > 0));
            ref_rows.push(row);
        }
        let label_line = &lines[pair.label.line];
        let same_page = ref_rows.first().is_some_and(|row| {
            row.get("pdf_page").and_then(Value::as_u64) == Some(u64::from(label_line.page))
        });
        let sequence_context = if symbol_pair {
            json!({
                "value": pair.label.symbol,
                "selected_label_image_filename": "",
                "selected_label_pdf_page": label_line.page,
                "same_page_as_selected_label": same_page,
            })
        } else {
            json!({
                "value": pair.label.value,
                "previous_value": pair.previous_value,
                "next_value": pair.next_value,
                "endnote_mode": pair.endnote,
                "segment_index": pair.restart_sequence - 1,
                "selected_label_image_filename": "",
                "selected_label_pdf_page": label_line.page,
                "same_page_as_selected_label": same_page,
            })
        };
        let status = if ref_rows.is_empty() {
            "label_only"
        } else {
            "paired"
        };
        let ref_marker_ids: Vec<String> = ref_rows
            .iter()
            .filter_map(|row| {
                row.get("marker_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        insert_shared_marker_fields(
            &mut label_row,
            &pair.pair_id,
            &pair.label.note_id(),
            &label_marker_id,
            &ref_marker_ids,
            status,
            same_page,
            &sequence_context,
        );
        rows.push(Value::Object(label_row));
        for mut row in ref_rows {
            insert_shared_marker_fields(
                &mut row,
                &pair.pair_id,
                &pair.label.note_id(),
                &label_marker_id,
                &ref_marker_ids,
                status,
                same_page,
                &sequence_context,
            );
            rows.push(Value::Object(row));
        }
    }

    let detached_pairs: HashMap<(usize, usize, usize, String), &Pair> = pairs
        .iter()
        .flat_map(|pair| {
            pair.refs
                .iter()
                .filter(|candidate| candidate.form == "detached")
                .map(move |candidate| {
                    (
                        (
                            candidate.line,
                            candidate.start,
                            candidate.end,
                            candidate.note_id(),
                        ),
                        pair,
                    )
                })
        })
        .collect();
    let mut detached_count = 0;
    for (line_index, line) in lines.iter().enumerate() {
        for detached in &line.detached_references {
            let Some(raw) = detached.get("note_id").and_then(Value::as_str) else {
                continue;
            };
            let note_id =
                numeric_value(raw).map_or_else(|| normalize_symbol(raw), |n| n.to_string());
            let start = detached
                .get("start_offset")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(0);
            let end = detached
                .get("end_offset")
                .and_then(Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .unwrap_or(start);
            let Some(pair) = detached_pairs.get(&(line_index, start, end, note_id.clone())) else {
                continue;
            };
            detached_count += 1;
            rows.push(json!({
                "schema_version": "oajd.footnote_pairing_v2_marker.v1",
                "marker_id": format!("legalpdf-detached-ref-{detached_count:06}"),
                "role": "fn_ref",
                "safe_to_use": true,
                "note_id": note_id,
                "selected_text": detached.get("selected_text").and_then(Value::as_str).unwrap_or(raw),
                "line_id": line.id,
                "region_id": line.region_id,
                "region_type": line.region_type,
                "pdf_page": line.page,
                "reading_order_index": line.order,
                "start_offset": start,
                "end_offset": end,
                "confidence": 0.84,
                "pairing_strategy_family": "detached_pdf_superscript",
                "materialized_pair_id": pair.pair_id,
                "materialized_note_id": pair.label.note_id(),
                "materialized_pair_status": "paired",
                "restart_sequence": 1,
            }));
        }
    }

    let pair_indexes: HashMap<&str, usize> = pairs
        .iter()
        .enumerate()
        .map(|(index, pair)| (pair.pair_id.as_str(), index))
        .collect();
    let mut grouped_rows = vec![(None, Vec::new()); pairs.len()];
    for (row_index, row) in rows.iter().enumerate() {
        let Some(pair_index) = row
            .get("materialized_pair_id")
            .and_then(Value::as_str)
            .and_then(|pair_id| pair_indexes.get(pair_id))
            .copied()
        else {
            continue;
        };
        match row.get("role").and_then(Value::as_str) {
            Some("fn_label") => {
                grouped_rows[pair_index].0.get_or_insert(row_index);
            }
            Some("fn_ref") => grouped_rows[pair_index].1.push(row_index),
            _ => {}
        }
    }
    for (label_index, ref_indexes) in grouped_rows {
        let Some(label_index) = label_index else {
            continue;
        };
        let label_page = rows[label_index]
            .get("pdf_page")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        let status = if ref_indexes.is_empty() {
            "label_only"
        } else {
            "paired"
        };
        let ref_ids: Vec<String> = ref_indexes
            .iter()
            .filter_map(|&index| {
                rows[index]
                    .get("marker_id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        let same_page = ref_indexes
            .iter()
            .any(|&index| rows[index].get("pdf_page").and_then(Value::as_u64) == Some(label_page));
        for index in std::iter::once(label_index).chain(ref_indexes) {
            if let Some(row) = rows[index].as_object_mut() {
                row.insert("materialized_pair_status".to_owned(), json!(status));
                row.insert("materialized_ref_count".to_owned(), json!(ref_ids.len()));
                row.insert("materialized_ref_marker_ids".to_owned(), json!(ref_ids));
                row.insert(
                    "materialized_label_same_page_as_ref".to_owned(),
                    json!(same_page),
                );
            }
        }
    }
    rows.sort_by_key(|row| {
        (
            row.get("reading_order_index")
                .and_then(Value::as_u64)
                .unwrap_or(0),
            row.get("start_offset").and_then(Value::as_u64).unwrap_or(0),
        )
    });
    rows
}

fn refresh_summary(summary: &mut Value, markers: &[Value]) {
    let mut role_counts = BTreeMap::<String, usize>::new();
    let mut pair_statuses = BTreeMap::<String, usize>::new();
    let mut marker_statuses = BTreeMap::<String, usize>::new();
    let mut cross_page_ids = HashSet::<String>::new();
    for marker in markers {
        let role = marker
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_owned();
        *role_counts.entry(role.clone()).or_default() += 1;
        let status = marker
            .get("materialized_pair_status")
            .and_then(Value::as_str)
            .unwrap_or("label_only")
            .to_owned();
        *marker_statuses.entry(status.clone()).or_default() += 1;
        if role == "fn_label" {
            *pair_statuses.entry(status.clone()).or_default() += 1;
            if status == "paired" {
                if let Some(pair_id) = marker.get("materialized_pair_id").and_then(Value::as_str) {
                    if marker
                        .get("materialized_label_same_page_as_ref")
                        .and_then(Value::as_bool)
                        == Some(false)
                    {
                        cross_page_ids.insert(pair_id.to_owned());
                    }
                }
            }
        }
    }
    let paired = pair_statuses.get("paired").copied().unwrap_or(0);
    let label_only = pair_statuses.get("label_only").copied().unwrap_or(0);
    let has_lines = summary
        .get("line_count")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        > 0;
    summary["dataset"] = json!(if has_lines { "LEGALPDF" } else { "" });
    summary["article_id"] = json!(if has_lines { "document" } else { "" });
    summary["selected_image_count"] = json!(usize::from(has_lines));
    summary["marker_count"] = json!(markers.len());
    summary["safe_marker_count"] = json!(markers.len());
    summary["role_counts"] = json!(role_counts);
    summary["safe_role_counts"] = json!(role_counts);
    summary["pair_count"] = json!(paired);
    summary["pair_status_counts"] = json!(pair_statuses);
    summary["materialized_marker_count"] = json!(markers.len());
    summary["materialized_pair_count"] = json!(paired);
    summary["materialized_label_only_count"] = json!(label_only);
    summary["materialized_marker_status_counts"] = json!(marker_statuses);
    summary["materialized_pair_status_counts"] = json!(pair_statuses);
    summary["article_footnote_pair_materialization"]["materialized_marker_count"] =
        json!(markers.len());
    summary["article_footnote_pair_materialization"]["materialized_pair_count"] = json!(paired);
    summary["article_footnote_pair_materialization"]["materialized_label_only_count"] =
        json!(label_only);
    summary["article_footnote_pair_materialization"]["cross_page_pair_count"] =
        json!(cross_page_ids.len());
    summary["article_footnote_pair_materialization"]["synthesized_label_marker_count"] =
        json!(markers
            .iter()
            .filter(|marker| {
                marker.get("role").and_then(Value::as_str) == Some("fn_label")
                    && marker
                        .get("article_context_note_id_repaired")
                        .and_then(Value::as_bool)
                        == Some(true)
            })
            .count());
    summary["workflow_stage_summary"] = json!({
        "stages": [
            "extract_candidates",
            "label_backbone_dp",
            "gap_glyph_repair",
            "monotone_ref_assignment",
            "custom_symbol_pairs",
            "materialize"
        ]
    });
}

fn materialize(
    pairs: Vec<Pair>,
    lines: &[PairLine],
    pages: &[Page],
    summary: Value,
) -> PairingOutput {
    let original: HashMap<&str, &Line> = pages
        .iter()
        .flat_map(|page| page.lines.iter())
        .map(|line| (line.id.as_str(), line))
        .collect();
    let mut occurrences: HashMap<String, usize> = HashMap::new();
    let continuation_pages: HashSet<u32> = lines
        .iter()
        .filter(|line| line.note_region_mode == "footnote_continuation")
        .map(|line| line.page)
        .collect();
    let mut footnotes = Vec::new();
    let mut diagnostics = Vec::new();
    let mut anchors: HashMap<String, Vec<Anchor>> = HashMap::new();
    for (index, pair) in pairs.iter().enumerate() {
        let label_line = &lines[pair.label.line];
        let next = pairs.get(index + 1);
        let next_page = next.map(|value| lines[value.label.line].page);
        let mut allowed_pages = BTreeSet::new();
        allowed_pages.insert(label_line.page);
        if pair.endnote {
            let last_page =
                next_page.unwrap_or_else(|| lines.last().map_or(label_line.page, |line| line.page));
            allowed_pages.extend(label_line.page..=last_page);
        } else {
            if next_page == Some(label_line.page + 1) {
                allowed_pages.insert(label_line.page + 1);
            }
            let mut continuation = label_line.page + 1;
            while continuation_pages.contains(&continuation) {
                allowed_pages.insert(continuation);
                continuation += 1;
            }
        }
        let stop = next
            .filter(|value| pair.endnote || allowed_pages.contains(&lines[value.label.line].page))
            .map_or(usize::MAX, |value| lines[value.label.line].idx);
        let mut body_lines: Vec<&PairLine> = lines[label_line.idx..stop.min(lines.len())]
            .iter()
            .filter(|line| {
                allowed_pages.contains(&line.page)
                    && line.region_type == "footnote"
                    && !line.exclude_from_body
            })
            .collect();
        if !pair.endnote && !label_line.region_id.is_empty() {
            let mut accepted = HashSet::from([label_line.region_id.as_str()]);
            for line in &body_lines {
                if line.page == label_line.page
                    && !line.region_id.is_empty()
                    && line.bbox[1] <= label_line.bbox[3]
                    && line.bbox[3] >= label_line.bbox[1]
                {
                    accepted.insert(line.region_id.as_str());
                }
            }
            let mut bounded = Vec::new();
            let mut prior: Option<&PairLine> = None;
            let mut blocked = false;
            for line in body_lines {
                if line.page != label_line.page {
                    bounded.push(line);
                    continue;
                }
                let mut include = accepted.contains(line.region_id.as_str());
                if !include && !blocked {
                    if let Some(previous) = prior {
                        let height = previous.height().max(1.0);
                        let gap = line.bbox[1] - previous.bbox[3];
                        let previous_size = original
                            .get(previous.id.as_str())
                            .map_or(0.0, |line| line_font_size(line));
                        let size = original
                            .get(line.id.as_str())
                            .map_or(0.0, |line| line_font_size(line));
                        include = gap <= (3.0_f64).max(height * 0.5)
                            && previous_size > 0.0
                            && (previous_size * 0.75..=previous_size * 1.25).contains(&size);
                    }
                }
                if include {
                    if !line.region_id.is_empty() {
                        accepted.insert(line.region_id.as_str());
                    }
                    prior = Some(line);
                    bounded.push(line);
                } else {
                    blocked = true;
                }
            }
            body_lines = bounded;
        }
        let mut parts = Vec::new();
        for line in &body_lines {
            let mut text = line.text.clone();
            if line.id == label_line.id {
                let body_start = label_token(&text)
                    .filter(|token| {
                        token.start == pair.label.start && token.value == pair.label.value
                    })
                    .map_or(pair.label.end, |token| token.match_end);
                text = char_slice(&text, body_start, text.chars().count())
                    .trim_start_matches(|character: char| {
                        character.is_whitespace() || ".)],:;-".contains(character)
                    })
                    .to_owned();
            }
            if text.is_empty() {
                continue;
            }
            parts.push(text);
        }
        let display = pair.label.note_id();
        let body = parts.join(" ").trim().to_owned();
        let body = if body.is_empty() {
            display.clone()
        } else {
            body
        };
        let occurrence = occurrences.entry(display.clone()).or_insert(0);
        *occurrence += 1;
        for reference in &pair.refs {
            let line = &lines[reference.line];
            anchors.entry(line.id.clone()).or_default().push(Anchor {
                pair_id: pair.pair_id.clone(),
                label: display.clone(),
                start: reference.start,
                end: reference.end,
            });
        }
        let primary = pair.refs.first();
        let mut warnings = Vec::new();
        if primary.is_none() {
            warnings.push("label_only".to_owned());
            let mut diagnostic = Diagnostic::warning(
                "FOOTNOTE_UNMATCHED_LABEL",
                format!("Footnote label '{display}' has no paired reference."),
                Some(label_line.page_index),
            );
            diagnostic.line_ids.push(label_line.id.clone());
            diagnostic
                .details
                .insert("pair_id".to_owned(), json!(pair.pair_id));
            diagnostic
                .details
                .insert("label".to_owned(), json!(display));
            diagnostics.push(diagnostic);
        }
        footnotes.push(Footnote {
            pair_id: pair.pair_id.clone(),
            label: display,
            occurrence: *occurrence,
            restart_sequence: pair.restart_sequence,
            reference_page: primary.map(|reference| lines[reference.line].page),
            body_pages: body_lines
                .iter()
                .map(|line| line.page)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect(),
            reference_line_id: primary.map(|reference| lines[reference.line].id.clone()),
            body_line_ids: body_lines.iter().map(|line| line.id.clone()).collect(),
            body,
            sentence_proposition: String::new(),
            passage_since_prior_note: String::new(),
            // The Python materializer reads the marker row's `confidence`
            // field. Canonical pairer rows expose `candidate_confidence`
            // instead, so the established public value is the 0.75 default.
            confidence: 0.75,
            provenance: pair.provenance.clone(),
            warnings,
            crossrefs: Vec::new(),
        });
    }
    PairingOutput {
        footnotes,
        diagnostics,
        anchors,
        markers: Vec::new(),
        summary,
    }
}

pub fn pair_footnotes(pages: &[Page]) -> PairingOutput {
    let (mut pairs, lines, mut summary) =
        legal_pdf_support::profile::measure("pair.build_pairs", || build_pairs(pages));
    let detached = legal_pdf_support::profile::measure("pair.merge_detached", || {
        merge_detached_references(&mut pairs, &lines)
    });
    summary["detached_reference_count"] = json!(detached);
    let markers =
        legal_pdf_support::profile::measure("pair.markers", || pair_markers(&pairs, &lines));
    legal_pdf_support::profile::measure("pair.refresh_summary", || {
        refresh_summary(&mut summary, &markers)
    });
    let mut output = legal_pdf_support::profile::measure("pair.materialize", || {
        materialize(pairs, &lines, pages, summary)
    });
    output.markers = markers;
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_keeps_the_oracle_empty_identity() {
        let output = pair_footnotes(&[]);

        assert_eq!(output.summary["dataset"], "");
        assert_eq!(output.summary["article_id"], "");
        assert_eq!(output.summary["selected_image_count"], 0);
    }

    #[test]
    fn label_token_covers_plain_paren_superscript_and_symbol_forms() {
        assert_eq!(label_token("12 Smith").unwrap().value, Some(12));
        assert_eq!(label_token("(7) Smith").unwrap().value, Some(7));
        assert_eq!(label_token("⁴ Smith").unwrap().value, Some(4));
        assert_eq!(label_token("* Author").unwrap().symbol, "*");
        assert_eq!(label_token("**** Author").unwrap().symbol, "****");
        let embedded = label_token("2endnote 2This is the note").unwrap();
        assert_eq!(embedded.value, Some(2));
        assert_eq!(embedded.end, 1);
        assert_eq!(
            char_slice("2endnote 2This is the note", embedded.match_end, 27),
            "This is the note"
        );
    }

    #[test]
    fn native_superscript_symbol_runs_are_reference_candidates() {
        let line = PairLine {
            idx: 0,
            page: 1,
            page_index: 0,
            order: 1,
            id: "byline".to_owned(),
            region_id: String::new(),
            region_type: "heading".to_owned(),
            text: "AUTHOR ****".to_owned(),
            bbox: [0.0, 0.0, 100.0, 10.0],
            page_width: 100.0,
            page_height: 100.0,
            zone: Zone::Title,
            protected_spans: Vec::new(),
            outline_spans: Vec::new(),
            note_column_fit: false,
            small_font: false,
            prose_like: false,
            region_witness_demoted: false,
            native_superscript_spans: vec![(7, 11)],
            suppress_footnote_label: false,
            exclude_from_body: false,
            note_region_mode: String::new(),
            note_sequence_restart: false,
            detached_references: Vec::new(),
        };

        let candidates = extract_refs(&[line]);

        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].symbol, "****");
        assert_eq!(candidates[0].reason, "native_superscript_span");
    }

    #[test]
    fn reference_right_boundary_matches_the_oracle_character_set() {
        assert!(ref_right(' '));
        assert!(ref_right('\t'));
        assert!(ref_right('\u{00a0}'));
        assert!(!ref_right('\u{2009}'));
    }

    #[test]
    fn pairing_indexes_stay_contiguous_after_excluded_lines() {
        let line = |id: &str, order: usize, excluded: bool| {
            json!({
                "id": id,
                "page_index": 0,
                "page_number": 1,
                "source_index": order,
                "reading_order": order,
                "block_index": 1,
                "text": id,
                "bbox": [0.0, order as f64, 10.0, order as f64 + 1.0],
                "exclude_from_body": excluded,
                "region_type": "body"
            })
        };
        let page: Page = serde_json::from_value(json!({
            "id": "p0001",
            "index": 0,
            "number": 1,
            "width": 100.0,
            "height": 100.0,
            "lines": [line("first", 1, false), line("excluded", 2, true), line("last", 3, false)],
            "regions": []
        }))
        .unwrap();

        let lines = build_lines(&[page]);

        assert_eq!(
            lines.iter().map(|line| line.idx).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(
            lines
                .iter()
                .map(|line| line.id.as_str())
                .collect::<Vec<_>>(),
            vec!["first", "last"]
        );
        assert_eq!(
            lines.iter().map(|line| line.order).collect::<Vec<_>>(),
            vec![1, 3]
        );
    }

    #[test]
    fn footer_folios_never_become_note_labels() {
        let page: Page = serde_json::from_value(json!({
            "id": "p0030",
            "index": 29,
            "number": 30,
            "width": 600.0,
            "height": 800.0,
            "lines": [{
                "id": "folio",
                "page_index": 29,
                "page_number": 30,
                "source_index": 1,
                "reading_order": 1,
                "block_index": 1,
                "text": "30",
                "bbox": [300.0, 768.0, 314.0, 780.0],
                "region_type": "footer"
            }],
            "regions": []
        }))
        .unwrap();
        let lines = build_lines(&[page]);
        let refs = extract_refs(&lines);

        assert_eq!(lines[0].zone, Zone::Body);
        assert!(lines[0].suppress_footnote_label);
        assert!(extract_labels(&lines, &refs).is_empty());
    }

    #[test]
    fn body_number_does_not_materialize_an_unmatched_label_only_note() {
        let line = |id: &str, order: usize, text: &str, height: f64| {
            json!({
                "id": id,
                "page_index": 0,
                "page_number": 1,
                "source_index": order,
                "reading_order": order,
                "block_index": 1,
                "text": text,
                "bbox": [60.0, order as f64 * 12.0, 500.0, order as f64 * 12.0 + height],
                "spans": [{
                    "id": format!("{id}-span"),
                    "text": text,
                    "bbox": [60.0, order as f64 * 12.0, 500.0, order as f64 * 12.0 + height],
                    "size": height
                }],
                "region_type": "body"
            })
        };
        let page: Page = serde_json::from_value(json!({
            "id": "p0001",
            "index": 0,
            "number": 1,
            "width": 600.0,
            "height": 800.0,
            "lines": [
                line("body-1", 1, "Ordinary body prose establishes the body size.", 10.0),
                line("body-2", 2, "More ordinary body prose establishes the body size.", 10.0),
                line("number", 3, "19 pandemic, grounds their policy analysis.", 8.0),
                line("body-3", 4, "The paragraph continues without any footnote.", 10.0)
            ],
            "regions": []
        }))
        .unwrap();

        let output = pair_footnotes(&[page]);

        assert!(output.footnotes.is_empty());
    }

    #[test]
    fn materialization_keeps_the_rest_of_an_accepted_continuation_region() {
        let line = |id: &str, order: usize, text: &str, region: &str, top: f64| {
            json!({
                "id": id,
                "page_index": 0,
                "page_number": 1,
                "source_index": order,
                "reading_order": order,
                "block_index": 1,
                "text": text,
                "bbox": [0.0, top, 100.0, top + 10.0],
                "spans": [{
                    "id": format!("{id}-span"),
                    "text": text,
                    "bbox": [0.0, top, 100.0, top + 10.0],
                    "size": 8.0
                }],
                "region_id": region,
                "region_type": "footnote"
            })
        };
        let page: Page = serde_json::from_value(json!({
            "id": "p0001",
            "index": 0,
            "number": 1,
            "width": 100.0,
            "height": 100.0,
            "lines": [
                line("label", 1, "1 First line.", "r1", 0.0),
                line("continuation", 2, "Continued.", "r2", 10.5),
                line("tail", 3, "Tail.", "r2", 21.0),
                line("next-label", 4, "2 Next note.", "r3", 50.0)
            ],
            "regions": []
        }))
        .unwrap();
        let lines = build_lines(std::slice::from_ref(&page));
        let candidate = |line: usize, value: u32| Candidate {
            line,
            start: 0,
            end: 1,
            observed: value.to_string(),
            value: Some(value),
            symbol: String::new(),
            form: "plain",
            score: 1.0,
            reason: "test",
            repaired: false,
            repair_kind: "",
            requires_visual_cue: false,
            flags: CandidateFlags::default(),
        };
        let pair = |line: usize, value: u32| Pair {
            label: candidate(line, value),
            refs: Vec::new(),
            primary_ref: None,
            previous_value: value.checked_sub(1),
            next_value: Some(value + 1),
            restart_sequence: 1,
            endnote: false,
            pair_id: format!("pair-{value}"),
            provenance: "test".to_owned(),
        };

        let output = materialize(vec![pair(0, 1), pair(3, 2)], &lines, &[page], json!({}));

        assert_eq!(
            output.footnotes[0].body_line_ids,
            ["label", "continuation", "tail"]
        );
        assert_eq!(output.footnotes[0].body, "First line. Continued. Tail.");
    }

    #[test]
    fn monotone_backbone_ignores_a_high_scoring_out_of_sequence_number() {
        let lines = ["1 One", "99 Noise", "2 Two", "3 Three"]
            .into_iter()
            .enumerate()
            .map(|(idx, text)| PairLine {
                idx,
                page: 1,
                page_index: 0,
                order: idx + 1,
                id: format!("l{idx}"),
                region_id: String::new(),
                region_type: "footnote".to_owned(),
                text: text.to_owned(),
                bbox: [0.0, idx as f64, 10.0, idx as f64 + 1.0],
                page_width: 100.0,
                page_height: 100.0,
                zone: Zone::Note,
                protected_spans: Vec::new(),
                outline_spans: Vec::new(),
                note_column_fit: true,
                small_font: true,
                prose_like: false,
                region_witness_demoted: false,
                native_superscript_spans: Vec::new(),
                suppress_footnote_label: false,
                exclude_from_body: false,
                note_region_mode: "footnote".to_owned(),
                note_sequence_restart: false,
                detached_references: Vec::new(),
            })
            .collect::<Vec<_>>();
        let candidates = extract_labels(&lines, &[]);
        let (chain, _) = select_backbone(&candidates, &lines);
        assert_eq!(
            chain
                .iter()
                .filter_map(|value| value.value)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn reporter_volume_prefix_requires_the_oracle_word_boundary() {
        assert!(volume_cite_start("12 S.C.R. 345, remainder"));
        assert!(!volume_cite_start("12 S.C.R. 345A remainder"));
        assert!(!volume_cite_start("12 S.C.R. 34567 remainder"));
    }

    #[test]
    fn only_visibly_isolated_inline_superscripts_keep_detached_evidence() {
        use legal_pdf_core::model::Span;

        let line = |gap: f64| Line {
            id: "line".to_owned(),
            page_index: 0,
            page_number: 1,
            source_index: 1,
            reading_order: 1,
            block_index: 1,
            text: "patients.7".to_owned(),
            bbox: [0.0, 0.0, 60.0 + gap, 10.0],
            spans: vec![
                Span {
                    id: "host".to_owned(),
                    text: "patients.".to_owned(),
                    bbox: [0.0, 0.0, 50.0, 10.0],
                    font: String::new(),
                    size: 10.0,
                    flags: 0,
                    superscript: false,
                    start: 0,
                    end: 9,
                },
                Span {
                    id: "marker".to_owned(),
                    text: "7".to_owned(),
                    bbox: [50.0 + gap, 2.0, 54.0 + gap, 7.0],
                    font: String::new(),
                    size: 5.0,
                    flags: 0,
                    superscript: true,
                    start: 9,
                    end: 10,
                },
            ],
            words: Vec::new(),
            detached_references: Vec::new(),
            exclude_from_body: false,
            suppress_footnote_label: false,
            note_region_mode: String::new(),
            region_id: String::new(),
            region_type: "paragraph".to_owned(),
            source: "native".to_owned(),
        };

        assert!(isolated_inline_references(&line(0.0)).is_empty());
        let evidence = isolated_inline_references(&line(4.0));
        assert_eq!(evidence.len(), 1);
        assert_eq!(evidence[0]["note_id"], "7");
        assert_eq!(evidence[0]["start_offset"], 9);
        assert_eq!(evidence[0]["end_offset"], 10);
    }
}
