use crate::grammar_tables::{compile_table_entry, find_table_matches};
use crate::{Error, Result};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

const REPORTER_IDS: &[&str] = &[
    "cite.reporter.splitter",
    "cite.us.reporter.full",
    "cite.us.reporter.short",
    "cite.us.reporter.custom.full",
    "cite.us.reporter.custom.short",
];
const STATUTE_IDS: &[&str] = &[
    "cite.statute.splitter",
    "cite.us.law.full",
    "cite.us.law.short",
];
const JOURNAL_IDS: &[&str] = &[
    "cite.journal.splitter",
    "cite.us.journal.full",
    "cite.us.journal.short",
];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicPart {
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub anchors: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicSplit {
    pub status: String,
    #[serde(default)]
    pub parts: Vec<DeterministicPart>,
    #[serde(default)]
    pub delimiters: Vec<(usize, usize, String)>,
    #[serde(default)]
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeterministicFields {
    pub status: String,
    pub corrected: String,
    pub kind: String,
    pub link_candidate: String,
    pub pinpoint_fragments: Vec<String>,
    pub page_pinpoints: Vec<u32>,
    pub bare_citation: String,
    pub citation_with_style: String,
    pub short_form: String,
    #[serde(default)]
    pub reasons: Vec<String>,
}

fn patterns() -> &'static Mutex<HashMap<&'static str, Arc<Regex>>> {
    static PATTERNS: OnceLock<Mutex<HashMap<&'static str, Arc<Regex>>>> = OnceLock::new();
    PATTERNS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pattern(id: &'static str) -> Result<Arc<Regex>> {
    if let Some(regex) = patterns().lock().expect("pattern cache").get(id).cloned() {
        return Ok(regex);
    }
    let compiled = Arc::new(compile_table_entry(id)?);
    let mut cache = patterns().lock().expect("pattern cache");
    Ok(cache.entry(id).or_insert_with(|| compiled.clone()).clone())
}

fn find_matches<'a>(id: &'static str, text: &'a str) -> Result<Vec<fancy_regex::Match<'a>>> {
    let regex = pattern(id)?;
    find_table_matches(id, &regex, text)
}

fn is_full_match(id: &'static str, text: &str) -> Result<bool> {
    Ok(find_matches(id, text)?
        .first()
        .is_some_and(|matched| matched.start() == 0 && matched.end() == text.len()))
}

fn masked_ranges(text: &str) -> Result<Vec<(usize, usize)>> {
    Ok(find_matches("cite.url", text)?
        .into_iter()
        .map(|matched| (matched.start(), matched.end()))
        .collect())
}

fn top_level_indices(text: &str) -> Result<HashSet<usize>> {
    let masked = masked_ranges(text)?;
    let mut masked_index = 0;
    let mut round_depth = 0;
    let mut square_depth = 0;
    let mut curly_depth = 0;
    let mut smart_quote = false;
    let mut straight_quote = false;
    let mut positions = HashSet::new();
    for (index, character) in text.char_indices() {
        while masked_index < masked.len() && index >= masked[masked_index].1 {
            masked_index += 1;
        }
        if masked_index < masked.len()
            && masked[masked_index].0 <= index
            && index < masked[masked_index].1
        {
            continue;
        }
        if !smart_quote
            && !straight_quote
            && round_depth == 0
            && square_depth == 0
            && curly_depth == 0
        {
            positions.insert(index);
        }
        match character {
            '“' => smart_quote = true,
            '”' => smart_quote = false,
            '"' => straight_quote = !straight_quote,
            '(' if !smart_quote && !straight_quote => round_depth += 1,
            ')' if !smart_quote && !straight_quote && round_depth > 0 => round_depth -= 1,
            '[' if !smart_quote && !straight_quote => square_depth += 1,
            ']' if !smart_quote && !straight_quote && square_depth > 0 => square_depth -= 1,
            '{' if !smart_quote && !straight_quote => curly_depth += 1,
            '}' if !smart_quote && !straight_quote && curly_depth > 0 => curly_depth -= 1,
            _ => {}
        }
    }
    Ok(positions)
}

fn top_level_semicolons(text: &str) -> Result<Vec<usize>> {
    let top = top_level_indices(text)?;
    Ok(text
        .char_indices()
        .filter_map(|(index, character)| {
            (character == ';' && top.contains(&index)).then_some(index)
        })
        .collect())
}

fn top_level_signals(text: &str) -> Result<Vec<usize>> {
    let top = top_level_indices(text)?;
    let mut positions = Vec::new();
    let regex = pattern("signal.source")?;
    for captures in regex.captures_iter(text) {
        let captures = captures.map_err(|error| Error::Message(error.to_string()))?;
        let matched = captures
            .name("sentence")
            .or_else(|| captures.name("inline"))
            .expect("signal grammar has a named branch");
        if top.contains(&matched.start()) {
            positions.push(matched.start());
        }
    }
    Ok(positions)
}

fn anchors(text: &str) -> Result<Vec<(usize, usize, String)>> {
    let mut found = Vec::new();
    for (kind, id) in [
        ("neutral", "cite.neutral"),
        ("reporter", REPORTER_IDS[0]),
        ("reporter", REPORTER_IDS[1]),
        ("reporter", REPORTER_IDS[2]),
        ("reporter", REPORTER_IDS[3]),
        ("reporter", REPORTER_IDS[4]),
        ("statute", STATUTE_IDS[0]),
        ("statute", STATUTE_IDS[1]),
        ("statute", STATUTE_IDS[2]),
        ("journal", JOURNAL_IDS[0]),
        ("journal", JOURNAL_IDS[1]),
        ("journal", JOURNAL_IDS[2]),
        ("book", "frame.book"),
        ("url", "cite.url"),
    ] {
        found.extend(
            find_matches(id, text)?
                .into_iter()
                .map(|matched| (matched.start(), matched.end(), kind.to_owned())),
        );
    }
    found.sort();
    let mut deduped: Vec<(usize, usize, String)> = Vec::new();
    for item in found {
        if deduped.last().is_some_and(|prior| item.0 < prior.1) {
            let prior = deduped.last_mut().expect("prior anchor");
            if item.1 - item.0 > prior.1 - prior.0 {
                *prior = item;
            }
            continue;
        }
        deduped.push(item);
    }
    Ok(deduped)
}

fn one_anchor_cluster(text: &str, anchors: &[(usize, usize, String)]) -> Result<bool> {
    if anchors.is_empty() {
        return Ok(false);
    }
    for pair in anchors.windows(2) {
        let previous = &pair[0];
        let current = &pair[1];
        let gap = &text[previous.1..current.0];
        if Regex::new(r"^\s*,\s*$")
            .expect("literal regex")
            .is_match(gap)
            .map_err(|error| Error::Message(error.to_string()))?
        {
            continue;
        }
        if current.2 == "url" && is_full_match("attach.link", gap)? {
            continue;
        }
        return Ok(false);
    }
    Ok(true)
}

fn trim_bounds(text: &str, mut start: usize, mut end: usize) -> (usize, usize) {
    while start < end {
        let character = text[start..].chars().next().expect("character");
        if !character.is_whitespace() {
            break;
        }
        start += character.len_utf8();
    }
    while end > start {
        let character = text[..end].chars().next_back().expect("character");
        if !character.is_whitespace() {
            break;
        }
        end -= character.len_utf8();
    }
    (start, end)
}

fn byte_to_char(text: &str, byte: usize) -> usize {
    text[..byte].chars().count()
}

fn back_chars(text: &str, end: usize, count: usize) -> usize {
    text[..end]
        .char_indices()
        .rev()
        .nth(count.saturating_sub(1))
        .map_or(0, |(index, _)| index)
}

fn clause(start: usize, end: usize, text: &str) -> Result<Option<DeterministicPart>> {
    let (start, end) = trim_bounds(text, start, end);
    if start >= end {
        return Ok(None);
    }
    let value = &text[start..end];
    if is_full_match("ref.pure.splitter", value)? {
        return Ok(Some(DeterministicPart {
            start: byte_to_char(text, start),
            end: byte_to_char(text, end),
            text: value.to_owned(),
            anchors: vec!["reference".to_owned()],
        }));
    }
    let values = anchors(value)?;
    if !one_anchor_cluster(value, &values)? {
        return Ok(None);
    }
    Ok(Some(DeterministicPart {
        start: byte_to_char(text, start),
        end: byte_to_char(text, end),
        text: value.to_owned(),
        anchors: values.into_iter().map(|item| item.2).collect(),
    }))
}

fn strip_signals(text: &str) -> Result<String> {
    let mut value = text.trim().to_owned();
    let regex = pattern("signal.prefix.splitter")?;
    for _ in 0..3 {
        let Some(matched) = regex
            .find(&value)
            .map_err(|error| Error::Message(error.to_string()))?
        else {
            break;
        };
        if matched.start() != 0 {
            break;
        }
        let stripped = value[matched.end()..].to_owned();
        if stripped == value {
            break;
        }
        value = stripped.trim().to_owned();
    }
    Ok(value)
}

fn pin_values(value: &str, expand_ranges: bool) -> Vec<String> {
    static ITEMS: OnceLock<regex::Regex> = OnceLock::new();
    static NUMBERS: OnceLock<regex::Regex> = OnceLock::new();
    let items = ITEMS
        .get_or_init(|| regex::Regex::new(r"(?i)\s*(?:,\s*(?:and\s+)?|\band\b|&)\s*").unwrap());
    let numbers = NUMBERS.get_or_init(|| regex::Regex::new(r"\d+(?:\.\d+)?").unwrap());
    let mut result = Vec::new();
    for item in items.split(value) {
        let numbers = numbers
            .find_iter(item)
            .map(|matched| matched.as_str())
            .collect::<Vec<_>>();
        let Some(first) = numbers.first() else {
            continue;
        };
        result.push((*first).to_owned());
        if !expand_ranges || numbers.len() < 2 || first.contains('.') || numbers[1].contains('.') {
            continue;
        }
        let Ok(start) = first.parse::<u32>() else {
            continue;
        };
        let end_text = numbers[1];
        let end = if end_text.len() < first.len() {
            let magnitude = 10_u32.pow(end_text.len() as u32);
            let mut end = start - start % magnitude + end_text.parse::<u32>().unwrap_or(start);
            if end < start {
                end += magnitude;
            }
            end
        } else {
            end_text.parse::<u32>().unwrap_or(start)
        };
        if end > start && end - start <= 100 {
            result.extend((start + 1..=end).map(|number| number.to_string()));
        }
    }
    result
}

fn provision_values(value: &str) -> Vec<String> {
    static ITEMS: OnceLock<regex::Regex> = OnceLock::new();
    static PROVISION: OnceLock<regex::Regex> = OnceLock::new();
    let items = ITEMS
        .get_or_init(|| regex::Regex::new(r"(?i)\s*(?:,\s*(?:and\s+)?|\band\b|&)\s*").unwrap());
    let provision = PROVISION.get_or_init(|| {
        regex::Regex::new(r"\d+(?:\.\d+)*(?:\s*\([A-Za-z0-9]+\))*(?:\s*[-–]\s*\d+(?:\.\d+)*)?")
            .unwrap()
    });
    items
        .split(value)
        .filter_map(|item| provision.find(item))
        .map(|matched| {
            matched
                .as_str()
                .split_whitespace()
                .collect::<String>()
                .replace('–', "-")
        })
        .collect()
}

fn named_match<'a>(id: &'static str, text: &'a str, name: &str) -> Result<Option<&'a str>> {
    let Some(captures) = pattern(id)?
        .captures(text)
        .map_err(|error| Error::Message(error.to_string()))?
    else {
        return Ok(None);
    };
    Ok(captures.name(name).map(|matched| matched.as_str()))
}

fn pinpoints(text: &str, kind: &str) -> Result<(Vec<String>, Vec<u32>)> {
    let source_kind = kind.to_lowercase();
    let case_source = matches!(source_kind.as_str(), "case" | "unreported");
    let law_source = matches!(
        source_kind.as_str(),
        "statute" | "regulation" | "legislation"
    );
    let unresolved = matches!(source_kind.as_str(), "" | "other");
    if case_source || unresolved {
        if let Some(values) = named_match("pinpoint.para.splitter", text, "values")? {
            return Ok((
                pin_values(values, false)
                    .into_iter()
                    .map(|value| format!("par{value}"))
                    .collect(),
                vec![],
            ));
        }
    }
    if law_source || unresolved {
        let reporter_spans = REPORTER_IDS
            .iter()
            .map(|id| find_matches(id, text))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .map(|matched| (matched.start(), matched.end()))
            .collect::<Vec<_>>();
        let regex = pattern("pinpoint.section.splitter")?;
        for captures in regex.captures_iter(text) {
            let captures = captures.map_err(|error| Error::Message(error.to_string()))?;
            let whole = captures.get(0).expect("whole section pinpoint");
            if reporter_spans
                .iter()
                .any(|(start, end)| *start <= whole.start() && whole.start() < *end)
            {
                continue;
            }
            let values = captures.name("values").expect("values group").as_str();
            return Ok((
                provision_values(values)
                    .into_iter()
                    .map(|value| format!("sec{value}"))
                    .collect(),
                vec![],
            ));
        }
    }
    if !law_source {
        if let Some(values) = named_match("pinpoint.page.splitter", text, "values")? {
            return Ok((
                vec![],
                pin_values(values, true)
                    .into_iter()
                    .filter_map(|value| value.parse().ok())
                    .collect(),
            ));
        }
    }
    Ok((vec![], vec![]))
}

fn trim_chars(value: &str) -> &str {
    value.trim_matches(&[' ', ',', ';', ':', '.'][..])
}

fn short_form(text: &str, kind: &str) -> Result<String> {
    if let Some(captures) = pattern("shortform.splitter")?
        .captures(text)
        .map_err(|error| Error::Message(error.to_string()))?
    {
        let value = captures.get(1).expect("short-form capture").as_str().trim();
        let editorial = pattern("bracket.editorial")?
            .find(value)
            .map_err(|error| Error::Message(error.to_string()))?
            .is_some_and(|matched| matched.start() == 0);
        if !value.chars().all(|character| character.is_ascii_digit()) && !editorial {
            return Ok(value.to_owned());
        }
    }
    if pattern("ref.token")?
        .is_match(text)
        .map_err(|error| Error::Message(error.to_string()))?
    {
        if find_matches("ref.token", text)?
            .into_iter()
            .any(|matched| matched.as_str().eq_ignore_ascii_case("ibid"))
        {
            return Ok("Ibid".to_owned());
        }
        let prefix = find_matches("ref.token", text)?
            .into_iter()
            .find(|matched| matched.as_str().eq_ignore_ascii_case("supra"))
            .map_or(text, |matched| &text[..matched.start()]);
        return Ok(trim_chars(strip_signals(prefix)?.as_str()).to_owned());
    }
    if matches!(kind, "journal" | "book" | "essay_collection" | "report") {
        let quote = [text.find('"'), text.find('“')].into_iter().flatten().min();
        let prefix = if let Some(position) = quote {
            let mut prefix = text[..position].trim_matches(&[' ', ','][..]);
            if let Some(colon) = prefix.rfind(':') {
                prefix = prefix[colon + 1..].trim();
            }
            prefix
        } else {
            text.split_once(',')
                .map_or(text, |(prefix, _)| prefix)
                .trim()
        };
        let prefix = strip_signals(prefix)?;
        static AUTHORS: OnceLock<regex::Regex> = OnceLock::new();
        static TOKENS: OnceLock<regex::Regex> = OnceLock::new();
        let authors =
            AUTHORS.get_or_init(|| regex::Regex::new(r"(?i)\s*(?:,|&|\band\b)\s*").unwrap());
        let tokens = TOKENS
            .get_or_init(|| regex::Regex::new(r"[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’.-]*").unwrap());
        let mut surnames = Vec::new();
        for author in authors.split(&prefix) {
            let values = tokens
                .find_iter(author)
                .map(|matched| matched.as_str())
                .filter(|token| {
                    !matches!(
                        token.to_ascii_lowercase().as_str(),
                        "et" | "al" | "kc" | "qc" | "eds" | "ed"
                    )
                })
                .collect::<Vec<_>>();
            if let Some(value) = values.last() {
                surnames.push(value.trim_matches('.').to_owned());
            }
        }
        if !surnames.is_empty() {
            return Ok(surnames.join(" and "));
        }
    }
    Ok(String::new())
}

fn has_embedded_source_signal(text: &str) -> Result<bool> {
    static SIGNAL: OnceLock<regex::Regex> = OnceLock::new();
    let signal = SIGNAL.get_or_init(|| {
        regex::Regex::new(
            r"(?i)(?:\.\s*(?:see(?:\s+also|\s+generally)?|cf\.?|compare)|\b(?:citing|quoted?\s+in|quoting|discuss(?:ed|ing)\s+in)\b)",
        )
        .unwrap()
    });
    for matched in signal.find_iter(text) {
        if matched.start() < 3 {
            continue;
        }
        let tail_end = text
            .char_indices()
            .skip_while(|(index, _)| *index < matched.end())
            .nth(320)
            .map_or(text.len(), |(index, _)| index);
        let tail = &text[matched.end()..tail_end];
        for id in ["ref.token", "cite.neutral", "frame.book"]
            .into_iter()
            .chain(REPORTER_IDS.iter().copied())
            .chain(STATUTE_IDS.iter().copied())
            .chain(JOURNAL_IDS.iter().copied())
        {
            if pattern(id)?
                .is_match(tail)
                .map_err(|error| Error::Message(error.to_string()))?
            {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn inside_quotes(text: &str, position: usize) -> bool {
    let mut inside = false;
    for character in text[..position].chars() {
        match character {
            '“' => inside = true,
            '”' => inside = false,
            '"' => inside = !inside,
            _ => {}
        }
    }
    inside
}

fn inside_square_brackets(text: &str, position: usize) -> bool {
    let prefix = &text[..position];
    prefix.rfind('[') > prefix.rfind(']')
}

fn source_evidence(text: &str) -> Result<bool> {
    if !anchors(text)?.is_empty() {
        return Ok(true);
    }
    for id in [
        "ref.cross-reference",
        "cite.quoted",
        "cite.secondary",
        "title.legal.splitter",
        "title.named-code",
    ] {
        if pattern(id)?
            .is_match(text)
            .map_err(|error| Error::Message(error.to_string()))?
        {
            return Ok(true);
        }
    }
    static PROVISION: OnceLock<regex::Regex> = OnceLock::new();
    Ok(PROVISION
        .get_or_init(|| {
            regex::Regex::new(r"(?i)^\s*(?:s(?:ection)?|r(?:ule)?|art(?:icle)?)\.?\s*\d").unwrap()
        })
        .is_match(text))
}

fn sentence_starts(text: &str) -> Result<Vec<usize>> {
    let mut starts = Vec::new();
    static ABBREVIATION: OnceLock<regex::Regex> = OnceLock::new();
    static CORPORATE: OnceLock<regex::Regex> = OnceLock::new();
    static VERSUS: OnceLock<regex::Regex> = OnceLock::new();
    let abbreviation = ABBREVIATION.get_or_init(|| {
        regex::Regex::new(r"(?i)(?:\be\.g|\bi\.e|\bcf|\bno|\bv|\bpara|\bart|\b[A-Z])\.$").unwrap()
    });
    let corporate = CORPORATE
        .get_or_init(|| regex::Regex::new(r"(?i)\b(?:Ltd|Inc|Corp|Co|LLC|LLP)\.$").unwrap());
    let versus = VERSUS.get_or_init(|| regex::Regex::new(r"(?i)^v\.?(?:\s|$)").unwrap());
    for matched in find_matches("boundary.sentence.splitter", text)? {
        if inside_quotes(text, matched.start()) {
            continue;
        }
        let prefix = &text[..matched.start() + 1];
        if abbreviation.is_match(prefix) {
            continue;
        }
        let suffix = &text[matched.end()..];
        if corporate.is_match(prefix) && versus.is_match(suffix) {
            continue;
        }
        starts.push(matched.end());
    }
    Ok(starts)
}

fn segment_start(boundaries: &[(usize, usize, String)], position: usize) -> usize {
    boundaries
        .iter()
        .filter(|(left, right, _)| *left < position && *right <= position)
        .map(|(_, right, _)| *right)
        .max()
        .unwrap_or(0)
}

fn segment_end(boundaries: &[(usize, usize, String)], position: usize, length: usize) -> usize {
    boundaries
        .iter()
        .filter(|(left, _, _)| *left > position)
        .map(|(left, _, _)| *left)
        .min()
        .unwrap_or(length)
}

fn case_starts(text: &str) -> Result<Vec<usize>> {
    static CASE_START: OnceLock<std::result::Result<Regex, String>> = OnceLock::new();
    let regex = CASE_START
        .get_or_init(|| {
            Regex::new(
                r"(?<!\w)(?:R\.?\s+v\.?|Reference\s+re|In\s+re|[A-Z][A-Za-z'’().& -]{1,70}\s+(?:v\.?|c))\s+",
            )
            .map_err(|error| error.to_string())
        })
        .as_ref()
        .map_err(|message| Error::Message(message.clone()))?;
    regex
        .find_iter(text)
        .map(|matched| {
            matched
                .map(|value| value.start())
                .map_err(|error| Error::Message(error.to_string()))
        })
        .collect()
}

fn recall_boundaries(text: &str) -> Result<Vec<(usize, usize, String)>> {
    let mut boundaries: Vec<(usize, usize, String)> = text
        .char_indices()
        .filter_map(|(index, character)| {
            (character == ';').then_some((index, index + 1, "semicolon".to_owned()))
        })
        .collect();

    let sentence_starts = sentence_starts(text)?;
    let mut sentence_ends = sentence_starts.iter().skip(1).copied().collect::<Vec<_>>();
    sentence_ends.push(text.len());
    for (&start, end) in sentence_starts.iter().zip(sentence_ends) {
        if !text[..start].trim().is_empty() && source_evidence(&text[start..end])? {
            boundaries.push((start, start, "new_citation_sentence".to_owned()));
        }
    }

    let mut hard_positions = vec![0, text.len()];
    hard_positions.extend(
        boundaries
            .iter()
            .flat_map(|(left, right, _)| [*left, *right]),
    );
    hard_positions.sort_unstable();
    hard_positions.dedup();
    for matched in find_matches("signal.aggressive", text)? {
        if inside_quotes(text, matched.start()) {
            continue;
        }
        let left = hard_positions
            .iter()
            .copied()
            .filter(|position| *position <= matched.start())
            .max()
            .unwrap_or(0);
        let right = hard_positions
            .iter()
            .copied()
            .filter(|position| *position >= matched.end())
            .min()
            .unwrap_or(text.len());
        if source_evidence(&text[left..matched.start()])?
            && source_evidence(&text[matched.start()..right])?
        {
            boundaries.push((matched.start(), matched.start(), "source_signal".to_owned()));
        }
    }

    let case_starts = case_starts(text)?;
    static CONJUNCTION: OnceLock<regex::Regex> = OnceLock::new();
    let conjunction =
        CONJUNCTION.get_or_init(|| regex::Regex::new(r"(?i)(?:\b(?:and|or)|&)\s*$").unwrap());
    for &position in case_starts.iter().skip(1) {
        let prefix = &text[back_chars(text, position, 12)..position];
        if text[..position].ends_with('(') || conjunction.is_match(prefix) {
            continue;
        }
        let start = segment_start(&boundaries, position);
        let end = segment_end(&boundaries, position, text.len());
        if source_evidence(&text[start..position])? && source_evidence(&text[position..end])? {
            boundaries.push((position, position, "new_case_frame".to_owned()));
        }
    }

    let author_starts = find_matches("ref.quoted-work-author", text)?
        .into_iter()
        .map(|matched| matched.start())
        .collect::<Vec<_>>();
    for position in author_starts.into_iter().skip(1) {
        let start = segment_start(&boundaries, position);
        let end = segment_end(&boundaries, position, text.len());
        if source_evidence(&text[start..position])? && source_evidence(&text[position..end])? {
            boundaries.push((position, position, "new_author_title_frame".to_owned()));
        }
    }

    let note_starts = find_matches("ref.note-reference", text)?
        .into_iter()
        .map(|matched| matched.start())
        .collect::<Vec<_>>();
    for position in note_starts.into_iter().skip(1) {
        let start = segment_start(&boundaries, position);
        if source_evidence(&text[start..position])? {
            boundaries.push((position, position, "new_note_reference".to_owned()));
        }
    }

    if anchors(text)?.is_empty()
        && regex::Regex::new(r"(?i)^\s*(?:see|compare|cf\.?|contra)\b")
            .unwrap()
            .is_match(text)
    {
        let bare =
            regex::Regex::new(r"(?i)\band\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’.-]{2,50}\s*\.\s*$").unwrap();
        if let Some(matched) = bare.find(text) {
            boundaries.push((
                matched.start(),
                matched.start(),
                "conjoined_short_form".to_owned(),
            ));
        }
    }

    for matched in find_matches("boundary.conjunction", text)? {
        if inside_quotes(text, matched.start()) {
            continue;
        }
        let start = segment_start(&boundaries, matched.start());
        let end = segment_end(&boundaries, matched.start(), text.len());
        if source_evidence(&text[start..matched.start()])?
            && source_evidence(&text[matched.start()..end])?
        {
            boundaries.push((
                matched.start(),
                matched.start(),
                "conjoined_citation".to_owned(),
            ));
        }
    }

    let legal_starts = find_matches("title.legal.splitter", text)?
        .into_iter()
        .filter(|matched| {
            !inside_quotes(text, matched.start()) && !inside_square_brackets(text, matched.start())
        })
        .map(|matched| matched.start())
        .collect::<Vec<_>>();
    for &position in &legal_starts {
        let start = segment_start(&boundaries, position);
        if !legal_starts
            .iter()
            .any(|prior| start <= *prior && *prior < position)
        {
            continue;
        }
        let end = segment_end(&boundaries, position, text.len());
        if source_evidence(&text[start..position])? && source_evidence(&text[position..end])? {
            boundaries.push((position, position, "new_legal_source_frame".to_owned()));
        }
    }

    let semicolons = boundaries
        .iter()
        .filter(|(left, right, _)| right > left)
        .map(|(left, _, _)| *left)
        .collect::<HashSet<_>>();
    let mut deduped = BTreeMap::new();
    boundaries.sort();
    for (left, right, reason) in boundaries {
        if left == right
            && (semicolons.contains(&left)
                || left.checked_sub(1).is_some_and(|p| semicolons.contains(&p)))
        {
            continue;
        }
        deduped.entry((left, right)).or_insert(reason);
    }
    Ok(deduped
        .into_iter()
        .map(|((left, right), reason)| (left, right, reason))
        .collect())
}

fn distinct_reasons(boundaries: &[(usize, usize, String)]) -> Vec<String> {
    let mut seen = HashSet::new();
    boundaries
        .iter()
        .filter_map(|(_, _, reason)| seen.insert(reason.clone()).then_some(reason.clone()))
        .collect()
}

pub fn split_footnote_recall_first(text: &str) -> Result<DeterministicSplit> {
    if text.trim().is_empty() {
        return Ok(DeterministicSplit {
            status: "abstain".to_owned(),
            parts: vec![],
            delimiters: vec![],
            reasons: vec!["empty".to_owned()],
        });
    }
    let boundaries = recall_boundaries(text)?;
    let starts = std::iter::once(0).chain(boundaries.iter().map(|(_, right, _)| *right));
    let ends = boundaries
        .iter()
        .map(|(left, _, _)| *left)
        .chain(std::iter::once(text.len()));
    let mut parts = Vec::new();
    for (start, end) in starts.zip(ends) {
        let (start, end) = trim_bounds(text, start, end);
        if start >= end {
            continue;
        }
        let value = &text[start..end];
        let mut kinds = anchors(value)?.into_iter().map(|item| item.2).collect();
        if is_full_match("ref.pure.splitter", value)? {
            kinds = vec!["reference".to_owned()];
        }
        parts.push(DeterministicPart {
            start: byte_to_char(text, start),
            end: byte_to_char(text, end),
            text: value.to_owned(),
            anchors: kinds,
        });
    }
    if parts.is_empty() {
        return Ok(DeterministicSplit {
            status: "abstain".to_owned(),
            parts,
            delimiters: vec![],
            reasons: vec!["empty_parts".to_owned()],
        });
    }
    let delimiters = parts
        .windows(2)
        .map(|pair| {
            let byte_start = text
                .char_indices()
                .nth(pair[0].end)
                .map_or(text.len(), |(index, _)| index);
            let byte_end = text
                .char_indices()
                .nth(pair[1].start)
                .map_or(text.len(), |(index, _)| index);
            (
                pair[0].end,
                pair[1].start,
                text[byte_start..byte_end].to_owned(),
            )
        })
        .collect();
    let reasons = distinct_reasons(&boundaries);
    Ok(DeterministicSplit {
        status: "deterministic_complete".to_owned(),
        parts,
        delimiters,
        reasons: if reasons.is_empty() {
            vec!["single_citation_or_prose".to_owned()]
        } else {
            reasons
        },
    })
}

pub fn split_footnote(text: &str) -> Result<DeterministicSplit> {
    if text.trim().is_empty() {
        return Ok(DeterministicSplit {
            status: "abstain".to_owned(),
            parts: vec![],
            delimiters: vec![],
            reasons: vec!["empty".to_owned()],
        });
    }
    let mut boundaries = top_level_semicolons(text)?
        .into_iter()
        .map(|position| (position, position + 1, "top_level_semicolon".to_owned()))
        .collect::<Vec<_>>();
    for position in top_level_signals(text)? {
        boundaries.sort();
        let segment_start = boundaries
            .iter()
            .filter(|(_, right, _)| *right <= position)
            .map(|(_, right, _)| *right)
            .max()
            .unwrap_or(0);
        let segment_end = boundaries
            .iter()
            .filter(|(left, _, _)| *left >= position)
            .map(|(left, _, _)| *left)
            .min()
            .unwrap_or(text.len());
        if clause(segment_start, position, text)?.is_some()
            && clause(position, segment_end, text)?.is_some()
        {
            boundaries.push((position, position, "explicit_source_signal".to_owned()));
        }
    }
    boundaries.sort();
    if boundaries.is_empty() {
        if is_full_match("ref.pure.splitter", text)? {
            return Ok(DeterministicSplit {
                status: "deterministic_complete".to_owned(),
                parts: vec![clause(0, text.len(), text)?.expect("pure reference clause")],
                delimiters: vec![],
                reasons: vec!["pure_reference".to_owned()],
            });
        }
        return Ok(DeterministicSplit {
            status: "abstain".to_owned(),
            parts: vec![],
            delimiters: vec![],
            reasons: vec!["no_supported_boundary".to_owned()],
        });
    }
    let starts = std::iter::once(0).chain(boundaries.iter().map(|(_, right, _)| *right));
    let ends = boundaries
        .iter()
        .map(|(left, _, _)| *left)
        .chain(std::iter::once(text.len()));
    let mut parts = Vec::new();
    for (start, end) in starts.zip(ends) {
        let Some(part) = clause(start, end, text)? else {
            return Ok(DeterministicSplit {
                status: "abstain".to_owned(),
                parts: vec![],
                delimiters: vec![],
                reasons: vec!["unconsumed_or_ambiguous_clause".to_owned()],
            });
        };
        parts.push(part);
    }
    let delimiters = parts
        .windows(2)
        .map(|pair| {
            let byte_start = text
                .char_indices()
                .nth(pair[0].end)
                .map_or(text.len(), |(index, _)| index);
            let byte_end = text
                .char_indices()
                .nth(pair[1].start)
                .map_or(text.len(), |(index, _)| index);
            (
                pair[0].end,
                pair[1].start,
                text[byte_start..byte_end].to_owned(),
            )
        })
        .collect();
    let used = boundaries
        .iter()
        .map(|(_, _, reason)| reason.as_str())
        .collect::<HashSet<_>>();
    Ok(DeterministicSplit {
        status: "deterministic_complete".to_owned(),
        parts,
        delimiters,
        reasons: ["top_level_semicolon", "explicit_source_signal"]
            .into_iter()
            .filter(|reason| used.contains(reason))
            .map(str::to_owned)
            .collect(),
    })
}

fn kind(text: &str, anchors: &[String]) -> Result<String> {
    if is_full_match("ref.pure.splitter", text)? {
        return Ok("other".to_owned());
    }
    if pattern("frame.book")?
        .is_match(text)
        .map_err(|e| Error::Message(e.to_string()))?
    {
        static ESSAY: OnceLock<regex::Regex> = OnceLock::new();
        return Ok(if ESSAY
            .get_or_init(|| regex::Regex::new(r#"["\u{201c}].+?["\u{201d}]\s+in\b"#).unwrap())
            .is_match(text)
        {
            "essay_collection"
        } else {
            "book"
        }
        .to_owned());
    }
    if JOURNAL_IDS
        .iter()
        .map(|id| pattern(id))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .any(|regex| regex.is_match(text).unwrap_or(false))
        && (text.contains('"') || text.contains('“'))
    {
        return Ok("journal".to_owned());
    }
    if anchors.iter().any(|value| value == "statute") {
        Ok("statute".to_owned())
    } else if anchors
        .iter()
        .any(|value| matches!(value.as_str(), "neutral" | "reporter"))
    {
        Ok("case".to_owned())
    } else if anchors.iter().any(|value| value == "journal") {
        Ok("journal".to_owned())
    } else {
        Ok("other".to_owned())
    }
}

fn bare_citation(text: &str, kind: &str) -> Result<String> {
    let mut value = text.trim().trim_end_matches('.').trim().to_owned();
    if let Some(matched) = pattern("shortform.splitter")?
        .find(&value)
        .map_err(|error| Error::Message(error.to_string()))?
    {
        value.replace_range(matched.start()..matched.end(), "");
        value = value.trim().trim_end_matches('.').trim().to_owned();
    }
    if pattern("ref.token")?
        .is_match(&value)
        .map_err(|e| Error::Message(e.to_string()))?
    {
        return Ok(value);
    }
    let ids = match kind {
        "case" => std::iter::once("cite.neutral")
            .chain(REPORTER_IDS.iter().copied())
            .collect::<Vec<_>>(),
        "statute" => STATUTE_IDS.to_vec(),
        "journal" => JOURNAL_IDS.to_vec(),
        _ => Vec::new(),
    };
    let start = ids
        .iter()
        .filter_map(|id| {
            pattern(id)
                .ok()?
                .find(&value)
                .ok()
                .flatten()
                .map(|matched| matched.start())
        })
        .min();
    Ok(start.map_or(value.clone(), |index| value[index..].to_owned()))
}

pub fn extract_fields(part: &DeterministicPart) -> Result<DeterministicFields> {
    let text = part.text.trim().to_owned();
    let kind = kind(&text, &part.anchors)?;
    let styled = strip_signals(&text)?;
    let (pinpoint_fragments, page_pinpoints) = pinpoints(&styled, &kind)?;
    let direct_link = find_matches("cite.url", &text)?.into_iter().next();
    let link = direct_link.map_or_else(
        || "other".to_owned(),
        |matched| {
            matched
                .as_str()
                .trim_matches(&['<', '>', '.', ',', ';', ' '][..])
                .to_owned()
        },
    );
    let mut reasons = Vec::new();
    if has_embedded_source_signal(&styled)? {
        reasons.push("embedded_second_source".to_owned());
    }
    if styled.is_empty() {
        reasons.push("missing_citation_surface".to_owned());
    }
    let bare = if styled.is_empty() {
        String::new()
    } else {
        bare_citation(&styled, &kind)?
    };
    if bare.is_empty() {
        reasons.push("missing_bare_citation".to_owned());
    }
    let short_form = short_form(&styled, &kind)?;
    Ok(DeterministicFields {
        status: if reasons.is_empty() {
            "complete"
        } else {
            "partial"
        }
        .to_owned(),
        corrected: text,
        kind,
        link_candidate: link,
        pinpoint_fragments,
        page_pinpoints,
        bare_citation: bare,
        citation_with_style: styled,
        short_form,
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corporate_suffix_and_parallel_reporters_match_the_oracle_vectors() {
        let corporate = "1068490 Ontario Ltd. V. Marlin Center Mobile Homes Inc. and Howard Geisler, 2001 CarswellOnt 4564, at para. 21 (Book of Authorities TAB 17)";
        let split = split_footnote_recall_first(corporate).unwrap();
        assert_eq!(
            split
                .parts
                .iter()
                .map(|part| &part.text)
                .collect::<Vec<_>>(),
            [corporate]
        );
        let fields = extract_fields(&split.parts[0]).unwrap();
        assert_eq!(fields.kind, "case");
        assert!(fields.bare_citation.starts_with("2001 CarswellOnt 4564"));

        let parallel = "Groia v Law Society, 2018 SCC 27, [2018] 1 SCR 772 at paras 64–67.";
        assert_eq!(
            split_footnote_recall_first(parallel).unwrap().parts.len(),
            1
        );
    }

    #[test]
    fn us_reporter_and_statute_are_shared_corpus_anchors() {
        let text = "Roe v Wade, 410 U.S. 113; claim under 42 U.S.C. § 1983.";
        let split = split_footnote_recall_first(text).unwrap();
        assert_eq!(
            split
                .parts
                .iter()
                .map(|part| part.anchors.clone())
                .collect::<Vec<_>>(),
            [vec!["reporter".to_owned()], vec!["statute".to_owned()]]
        );
    }

    #[test]
    fn delimiters_are_lossless_and_not_discarded() {
        let text = "2018 SCC 27; 2019 SCC 1";
        let split = split_footnote_recall_first(text).unwrap();
        assert_eq!(split.delimiters, [(11, 13, "; ".to_owned())]);
        let rebuilt = format!(
            "{}{}{}",
            split.parts[0].text, split.delimiters[0].2, split.parts[1].text
        );
        assert_eq!(rebuilt, text);
    }
}
