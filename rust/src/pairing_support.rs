use fancy_regex::Regex as FancyRegex;
use regex::{Regex, RegexSet};
use serde_json::{json, Map, Value};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::OnceLock;
use unicode_normalization::UnicodeNormalization;

const MAX_COUNTER_VALUE: u32 = 200;
const MAX_OUTLINE_DEPTH: usize = 4;
const FOOTNOTE_SUSPECT_MIN_VALUE: u32 = 15;
const ALL_CAPS_MIN_RATIO: f64 = 0.85;
const TITLECASE_MIN_RATIO: f64 = 0.6;

const COURT_CODE_PATTERN: &str = "SCC|FCA|FC|TCC|CMAC|BCCA|BCSC|BCPC|ABCA|ABQB|ABKB|ABPC|SKCA|SKQB|SKKB|SKPC|MBCA|MBQB|ONCA|ONSC|ONCJ|QCCA|QCCS|QCCQ|NBCA|NBQB|NSSC|NSCA|PECA|PESC|NLCA|NLSC|YKCA|YKSC|NWTCA|NWTSC|NUCA|NUCJ";
const REPORTER_TOKEN_PATTERN: &str = r"S\.?\s*C\.?\s*R\.?|D\.?\s*L\.?\s*R\.?|C\.?\s*C\.?\s*C\.?|O\.?\s*R\.?|W\.?\s*W\.?\s*R\.?|C\.?\s*R\.?|All\s+E\.?\s*R\.?|A\.?\s*C\.?|K\.?\s*B\.?|Q\.?\s*B\.?|Q\.?\s*B\.?\s*D\.?|Ch(?:\s+D)?\.?|App\s+Cas|W\.?\s*L\.?\s*R\.?|E\.?\s*R\.?|T\.?\s*L\.?\s*R\.?|Cox\s+C\.?\s*C\.?|Cr\s+App\s+R\.?|Ex\.?|Eq\.?|H\.?\s*L\.?\s*Cas\.?";
// Canadian statute sources only; add canonical codes for each supported jurisdiction.
const STATUTE_PATTERN: &str =
    "RSC|RSO|RSA|RSBC|RSM|RSNB|RSNS|RSPEI|CQLR|CCSM|SC|SO|SA|SBC|SM|SNB|SNS|SS|SY|SNWT|SNu|RLRQ";

fn citation_cue_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)\b(?:ibid|id\.?|ibidem|supra|infra|op\s+cit|note|notes|para\.?|paras\.?|paragraphs?|pp?\.?|pages?|ss?\.?|secs?\.?|sections?|art\.?|arts\.?|at|see|cf\.?|e\.?g\.?|accord|contra|R\.?\s*v\.?|Rex|Regina|v\.?|vs\.?|CanLII|SCC|SCR|DLR|(?:{COURT_CODE_PATTERN})|(?:{REPORTER_TOKEN_PATTERN}))\b|\[(?:17|18|19|20)\d{{2}}\]|\((?:17|18|19|20)\d{{2}}\)"
        ))
        .expect("frozen legal citation cue regex")
    })
}

fn citation_continuation_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)^\s*(?:{REPORTER_TOKEN_PATTERN}|U\.?\s*T\.?\s*L\.?|A\.?\s*L\.?\s*R\.?|N\.?\s*R\.?|S\.?\s*E\.?|N\.?\s*E\.?|P\.?\s*\d+d)\b"
        ))
        .expect("frozen legal citation continuation regex")
    })
}

pub(crate) fn has_legal_citation_cue(text: &str) -> bool {
    citation_cue_re().is_match(text)
}

pub(crate) fn is_legal_citation_continuation(text: &str) -> bool {
    citation_continuation_re().is_match(text)
}

fn protected_re(index: usize) -> &'static Regex {
    static RES: OnceLock<[OnceLock<Regex>; 5]> = OnceLock::new();
    RES.get_or_init(|| std::array::from_fn(|_| OnceLock::new()))[index].get_or_init(|| {
        let pattern = match index {
            0 => format!(r"(?i)(?P<span>\[(?:17|18|19|20)\d{{2}}\]\s+(?:\d{{1,4}}\s+)?(?:{REPORTER_TOKEN_PATTERN})\s+\d{{1,4}}|\((?:17|18|19|20)\d{{2}}\)\s+\d{{1,4}}\s+(?:{REPORTER_TOKEN_PATTERN})\s+\d{{1,4}}|\b\d{{1,4}}\s+(?:{REPORTER_TOKEN_PATTERN})\s*(?:\(\d{{1,4}}[a-z]{{0,2}}\))?\s+\d{{1,4}})(?:[^\d]|$)"),
            1 => format!(r"(?i)(?P<span>\b(?:17|18|19|20)\d{{2}}\s+(?:CanLII\s+\d{{1,4}}|(?:{COURT_CODE_PATTERN})\s+\d{{1,4}})\b)"),
            2 => format!(r"(?i)(?P<span>\b(?:{STATUTE_PATTERN})\s+(?:17|18|19|20)\d{{2}},?\s+c(?:h)?\.?\s+[A-Z0-9][A-Z0-9.\-]*)"),
            3 => r"(?P<span>\((?:17|18|19|20)\d{2}\)\s+\d{1,4}(?::\d{1,4})?\s+[A-Z][A-Za-z&.'\-\s]{2,60}\s+\d{1,4}\b)".to_owned(),
            4 => r"(?i)(?:^|[^A-Za-z0-9])(?P<span>(?:at\s+|pp?\.?\s+|pages?\s+|paras?\.?\s+|ss?\.?\s+)\d{1,4}[A-Za-z]?(?:\.\d{1,4})?(?:\s*(?:,|and|-|to|–)\s*\d{1,4}[A-Za-z]?(?:\.\d{1,4})?)*)(?:[^\d]|$)".to_owned(),
            _ => unreachable!(),
        };
        Regex::new(&pattern).expect("frozen protected citation regex")
    })
}

fn has_pinpoint_prefix(text: &str) -> bool {
    if !text.is_ascii() {
        return true;
    }
    const CUES: [(&str, bool); 9] = [
        ("at", false),
        ("p", true),
        ("pp", true),
        ("page", true),
        ("pages", true),
        ("para", true),
        ("paras", true),
        ("s", true),
        ("ss", true),
    ];
    text.char_indices().any(|(index, character)| {
        if !character.is_ascii_alphabetic()
            || (index > 0
                && text[..index]
                    .chars()
                    .next_back()
                    .is_some_and(|previous| previous.is_ascii_alphanumeric()))
        {
            return false;
        }
        CUES.iter().any(|(cue, allows_dot)| {
            let rest = &text[index..];
            let Some(prefix) = rest.get(..cue.len()) else {
                return false;
            };
            if !prefix.eq_ignore_ascii_case(cue) {
                return false;
            }
            let mut tail = &rest[cue.len()..];
            if *allows_dot && tail.starts_with('.') {
                tail = &tail[1..];
            }
            let mut characters = tail.chars();
            characters.next().is_some_and(char::is_whitespace)
                && characters
                    .find(|next| !next.is_whitespace())
                    .is_some_and(char::is_numeric)
        })
    })
}

pub(crate) fn protected_citation_spans(text: &str) -> Vec<(usize, usize)> {
    if !text.chars().any(|character| character.is_numeric()) {
        return Vec::new();
    }
    let mut digit_runs = 0;
    let mut inside_digits = false;
    for character in text.chars() {
        if character.is_numeric() {
            if !inside_digits {
                digit_runs += 1;
            }
            inside_digits = true;
        } else {
            inside_digits = false;
        }
    }
    let statute_source = !text.is_ascii()
        || text
            .split(|character: char| !character.is_ascii_alphanumeric())
            .any(|token| {
                STATUTE_PATTERN
                    .split('|')
                    .any(|source| token.eq_ignore_ascii_case(source))
            });
    let pinpoint_prefix = has_pinpoint_prefix(text);
    let mut spans = Vec::new();
    for index in 0..5 {
        if (index < 2 && digit_runs < 2)
            || (index == 2 && !statute_source)
            || (index == 3 && digit_runs < 3)
            || (index == 4 && !pinpoint_prefix)
        {
            continue;
        }
        let regex = protected_re(index);
        let mut offset = 0;
        while let Some(captures) = regex.captures_at(text, offset) {
            let found = captures.name("span").expect("protected span capture");
            spans.push(if text.is_ascii() {
                (found.start(), found.end())
            } else {
                (
                    text[..found.start()].chars().count(),
                    text[..found.end()].chars().count(),
                )
            });
            offset = found.end();
        }
    }
    spans
}

fn citation_signal_re() -> &'static RegexSet {
    static RE: OnceLock<RegexSet> = OnceLock::new();
    RE.get_or_init(|| {
        RegexSet::new([
            format!(r"\b\d{{4}}\s+(?:CanLII\s+\d+|(?:{COURT_CODE_PATTERN})\s+\d+)\b"),
            r"(?i)\[\d{4}\]\s+\d+\s+S\.?\s*C\.?\s*R\.?\s+\d+\b".to_owned(),
            format!(r"(?i)(?:\[\d{{4}}\]\s+(?:\d+\s+)?(?:{REPORTER_TOKEN_PATTERN})\s+\d+|\(\d{{4}}\)\s+\d+\s+(?:{REPORTER_TOKEN_PATTERN})\s+\d+|\b\d+\s+(?:{REPORTER_TOKEN_PATTERN})\s*(?:\(\d+[a-z]{{0,2}}\))?\s+\d+)"),
            r"\b[A-Z][A-Za-z'.\-]+(?:\s+(?:[A-Z][A-Za-z'.\-]+|of|and|the|for|to|du|de|des|la|le)){0,6}\s+(?i:v(?:s)?\.?)\s+[A-Z][A-Za-z'.\-]+(?:\s+(?:[A-Z][A-Za-z'.\-]+|of|and|the|for|to|du|de|des|la|le)){0,6}\b".to_owned(),
            format!(r"(?i)\b(?:{STATUTE_PATTERN})\s+\d{{4}},?\s+c(?:h)?\.?\s+[A-Z0-9][A-Z0-9.\-]*"),
            r"(?i)(?:^|[^A-Za-z0-9])(?:s|ss|sec|secs|section|sections|silcrow|§+)\.?\s+\d+[A-Za-z]?(?:\.\d+)*(?:\([A-Za-z0-9]+\))?(?:\s*(?:,|and|-|to)\s*\d+[A-Za-z]?(?:\.\d+)*(?:\([A-Za-z0-9]+\))?)*".to_owned(),
            r"(?i)(?:^|[^A-Za-z0-9])(?:paras?|paragraphs?|pilcrow|¶)\.?\s+\d+(?:\([A-Za-z0-9]+\))?(?:\s*(?:,|and|-|to)\s*\d+(?:\([A-Za-z0-9]+\))?)*".to_owned(),
            r"(?i)(?:^|[^A-Za-z0-9])(?:at\s+|à\s+la\s+|aux\s+)(?:p{1,2}|pages?)\.?\s+\d{1,4}[A-Za-z]?(?:\.\d{1,4})?(?:\s*(?:,|and|et|-|to|à|–)\s*\d{1,4}[A-Za-z]?(?:\.\d{1,4})?)*(?:[^\d]|$)".to_owned(),
            r"(?i)\b(?:supra\s+(?:note|n\.?|nn\.?)\s+\d+|ibid(?:em)?\.?)(?:\W|$)".to_owned(),
            r"\(\d{4}\)\s+\d+(?::\d+)?\s+[A-Z][A-Za-z&.\-\s]{2,45}\s+\d+\b".to_owned(),
        ])
        .expect("frozen citation signal regex set")
    })
}

fn python_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\u{000b}\u{000c}".contains(character) {
            escaped.push('\\');
        }
        escaped.push(character);
    }
    escaped
}

fn reporter_abbreviation_regex(abbreviation: &str) -> String {
    let mut result = String::new();
    let characters: Vec<char> = abbreviation.chars().collect();
    let mut index = 0;
    while index < characters.len() {
        let character = characters[index];
        if character.is_whitespace() {
            while index < characters.len() && characters[index].is_whitespace() {
                index += 1;
            }
            result.push_str(r"\s+");
        } else if character.is_alphanumeric() {
            let start = index;
            while index < characters.len() && characters[index].is_alphanumeric() {
                index += 1;
            }
            let token: String = characters[start..index].iter().collect();
            if token.chars().count() > 1 && token.chars().all(char::is_uppercase) {
                result.push_str(
                    &token
                        .chars()
                        .map(|value| format!("{}\\.?", python_escape(&value.to_string())))
                        .collect::<Vec<_>>()
                        .join(r"\s*"),
                );
            } else {
                result.push_str(&python_escape(&token));
            }
        } else if character == '(' {
            let end = characters[index + 1..]
                .iter()
                .position(|value| *value == ')')
                .map_or(characters.len(), |offset| index + offset + 1);
            let inner: String = characters[index + 1..end].iter().collect();
            result.push_str(r"\(\s*");
            result.push_str(&python_escape(&inner).replace(r"\ ", r"\s+"));
            result.push_str(r"\s*\)");
            index = (end + 1).min(characters.len());
        } else {
            match character {
                '&' => result.push_str(r"\s*&\s*"),
                '-' | '/' => result.push_str(r"\s*[-/]\s*"),
                '\'' | '\u{2019}' => result.push_str("['\u{2019}]"),
                '.' => result.push_str(r"\.?"),
                _ => result.push_str(&python_escape(&character.to_string())),
            }
            index += 1;
        }
    }
    result
}

fn mcgill_reporter_citation_re(first: u8) -> Option<&'static Regex> {
    static RES: OnceLock<[OnceLock<Regex>; 26]> = OnceLock::new();
    static ABBREVIATIONS: OnceLock<Vec<String>> = OnceLock::new();
    let index = first.checked_sub(b'A')? as usize;
    let slot = RES
        .get_or_init(|| std::array::from_fn(|_| OnceLock::new()))
        .get(index)?;
    let abbreviations = ABBREVIATIONS.get_or_init(|| {
        serde_json::from_str(include_str!(
            "../../src/legalpdf/data/mcgill_reporters.json"
        ))
        .expect("frozen McGill reporter inventory")
    });
    abbreviations
        .iter()
        .any(|value| value.as_bytes().first() == Some(&first))
        .then(|| {
            slot.get_or_init(|| {
                let reporters = abbreviations
                    .iter()
                    .filter(|value| value.as_bytes().first() == Some(&first))
                    .map(|value| reporter_abbreviation_regex(value))
                    .collect::<Vec<_>>()
                    .join("|");
                Regex::new(&format!(
                    r"(?:\[\d{{4}}\]\s+(?:\d+\s+)?(?:{reporters})\s+\d+|\(\d{{4}}\)\s+\d+\s+(?:{reporters})\s+\d+|\b\d+\s+(?:{reporters})\s*(?:\(\d+[A-Za-z]{{0,3}}\))?\s+\d+)"
                ))
                .expect("frozen McGill reporter citation regex")
            })
        })
}

fn has_mcgill_reporter_citation(text: &str) -> bool {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let prefix = PREFIX.get_or_init(|| {
        Regex::new(r"(?:\[\d{4}\]\s+(?:\d+\s+)?|\(\d{4}\)\s+\d+\s+|\b\d+\s+)([A-Z])")
            .expect("frozen McGill reporter prefix regex")
    });
    let mut tried = 0_u32;
    prefix.captures_iter(text).any(|captures| {
        let first = captures[1].as_bytes()[0];
        let bit = 1 << (first - b'A');
        if tried & bit != 0 {
            return false;
        }
        tried |= bit;
        mcgill_reporter_citation_re(first).is_some_and(|regex| regex.is_match(text))
    })
}

fn normalize_citation_text(text: &str) -> Cow<'_, str> {
    if text.is_ascii() {
        return Cow::Borrowed(text);
    }
    Cow::Owned(
        text.nfkc()
            .filter_map(|character| match character {
                '\u{2018}' | '\u{2019}' | '\u{201a}' | '\u{201b}' => Some('\''),
                '\u{201c}' | '\u{201d}' | '\u{201e}' => Some('"'),
                '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' => Some('-'),
                '\u{00a0}' => Some(' '),
                '\u{feff}' => None,
                _ => Some(character),
            })
            .collect(),
    )
}

pub(crate) fn has_citation_signal(text: &str) -> bool {
    let normalized = normalize_citation_text(text);
    if citation_signal_re().is_match(&normalized) {
        return true;
    }
    static DIGIT_RUN: OnceLock<Regex> = OnceLock::new();
    if DIGIT_RUN
        .get_or_init(|| Regex::new(r"\d+").expect("digit run regex"))
        .find_iter(&normalized)
        .take(2)
        .count()
        < 2
    {
        return false;
    }
    has_mcgill_reporter_citation(&normalized)
}

pub(crate) fn heading_text_plausible(value: &str) -> bool {
    let text = value.trim();
    if text.is_empty() || text.chars().count() > 100 {
        return false;
    }
    let Some(first) = text.chars().next() else {
        return false;
    };
    if !first.is_alphabetic() || !first.is_uppercase() {
        return false;
    }
    static TRAILING_DIGIT: OnceLock<Regex> = OnceLock::new();
    if TRAILING_DIGIT
        .get_or_init(|| Regex::new(r"\d\s*[.,;]?\s*$").expect("trailing digit regex"))
        .is_match(text)
    {
        return false;
    }
    static POSSESSIVE: OnceLock<FancyRegex> = OnceLock::new();
    let citation_text = POSSESSIVE
        .get_or_init(|| {
            FancyRegex::new(r"(?i)(?<=[A-Za-z])['’]s\b").expect("possessive suffix regex")
        })
        .replace_all(text, "")
        .into_owned();
    if has_legal_citation_cue(&citation_text) || has_citation_signal(&citation_text) {
        return false;
    }
    let letters = text
        .chars()
        .filter(|character| character.is_alphabetic())
        .collect::<Vec<_>>();
    let all_caps = letters.len() >= 4
        && letters
            .iter()
            .filter(|character| character.is_uppercase())
            .count() as f64
            / letters.len() as f64
            >= ALL_CAPS_MIN_RATIO;
    let words = text
        .split_whitespace()
        .filter(|word| word.chars().any(|character| character.is_alphabetic()))
        .collect::<Vec<_>>();
    let titlecase = !words.is_empty()
        && words
            .iter()
            .filter(|word| word.chars().next().is_some_and(char::is_uppercase))
            .count() as f64
            / words.len() as f64
            >= TITLECASE_MIN_RATIO;
    all_caps || titlecase
}

fn roman_to_int(value: &str) -> Option<u32> {
    let mut total = 0;
    let mut prior = 0;
    for character in value.to_uppercase().chars().rev() {
        let current = match character {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if current < prior {
            total -= current;
        } else {
            total += current;
            prior = current;
        }
    }
    (total > 0 && total <= MAX_COUNTER_VALUE).then_some(total)
}

pub(crate) fn enumerator_interpretations(value: &str, punct: &str) -> Vec<Value> {
    static LEGAL_NUMERIC: OnceLock<Regex> = OnceLock::new();
    static ROMAN: OnceLock<Regex> = OnceLock::new();
    let mut result = Vec::new();
    if LEGAL_NUMERIC
        .get_or_init(|| Regex::new(r"^\d{1,2}(?:\.\d{1,2}){1,3}$").unwrap())
        .is_match(value)
    {
        let parts = value.split('.').collect::<Vec<_>>();
        let tail = parts.last().and_then(|part| part.parse::<u32>().ok());
        if tail.is_some_and(|number| (1..=MAX_COUNTER_VALUE).contains(&number)) {
            result.push(json!([format!("legal_numeric_{punct}"), tail, parts.len()]));
        }
        return result;
    }
    if value.chars().all(|character| character.is_ascii_digit()) {
        if value
            .parse::<u32>()
            .ok()
            .is_some_and(|number| (1..=MAX_COUNTER_VALUE).contains(&number))
        {
            result.push(json!([
                format!("numeric_{punct}"),
                value.parse::<u32>().unwrap(),
                0
            ]));
        }
        return result;
    }
    let upper = value.to_uppercase();
    if value == upper
        && ROMAN
            .get_or_init(|| Regex::new(r"^[IVXLCDM]{2,7}$").unwrap())
            .is_match(&upper)
    {
        if let Some(number) = roman_to_int(value) {
            result.push(json!([format!("roman_{punct}"), number, 0]));
        }
        return result;
    }
    if value.chars().count() == 1 && value.chars().all(char::is_alphabetic) {
        let family = if value.chars().all(char::is_uppercase) {
            "upper_alpha"
        } else {
            "lower_alpha"
        };
        if value == upper && "IVXLCDM".contains(&upper) {
            if let Some(number) = roman_to_int(value) {
                result.push(json!([format!("roman_{punct}"), number, 0]));
            }
        }
        let character = upper.chars().next().unwrap();
        let number = u32::from(character).wrapping_sub(u32::from('A')) + 1;
        result.push(json!([format!("{family}_{punct}"), number, 0]));
    }
    result
}

#[derive(Clone)]
struct Frame {
    family: String,
    value: u32,
}

#[derive(Default)]
struct FamilyStats {
    count: usize,
    max_value: u32,
    level_votes: BTreeMap<usize, usize>,
    violations: usize,
    gaps: usize,
}

fn interpretations(candidate: &Value) -> Vec<(String, u32, usize)> {
    candidate
        .get("interpretations")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let values = item.as_array()?;
            Some((
                values.first()?.as_str()?.to_owned(),
                u32::try_from(values.get(1)?.as_u64()?).ok()?,
                usize::try_from(values.get(2)?.as_u64()?).ok()?,
            ))
        })
        .collect()
}

fn assigned(candidate: &Value, family: &str, value: Value, level: Value, action: &str) -> Value {
    let mut result = candidate.as_object().cloned().unwrap_or_default();
    result.insert("family".to_owned(), Value::String(family.to_owned()));
    result.insert("value".to_owned(), value);
    result.insert("level".to_owned(), level);
    result.insert("action".to_owned(), Value::String(action.to_owned()));
    Value::Object(result)
}

pub(crate) fn parse_heading_ladder(candidates: &[Value]) -> Value {
    let mut stack: Vec<Frame> = Vec::new();
    let mut assignments = Vec::new();
    let mut stats: BTreeMap<String, FamilyStats> = BTreeMap::new();
    let mut violations = 0_usize;
    let mut gaps = 0_usize;

    for candidate in candidates {
        if candidate.get("kind").and_then(Value::as_str) != Some("enumerator") {
            assignments.push(assigned(
                candidate,
                "",
                Value::Null,
                Value::Null,
                "caps_observed",
            ));
            continue;
        }
        let choices = interpretations(candidate);
        let mut chosen: Option<(String, u32, &'static str, usize)> = None;
        for (family, value, _) in &choices {
            if let Some(index) = stack.iter().rposition(|frame| &frame.family == family) {
                if stack[index].value + 1 == *value {
                    stack.truncate(index + 1);
                    stack[index].value = *value;
                    chosen = Some((family.clone(), *value, "increment", index + 1));
                    break;
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in &choices {
                if *value == 1
                    && !stack.iter().any(|frame| &frame.family == family)
                    && stack.len() < MAX_OUTLINE_DEPTH
                {
                    stack.push(Frame {
                        family: family.clone(),
                        value: 1,
                    });
                    chosen = Some((family.clone(), *value, "open_level", stack.len()));
                    break;
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in &choices {
                if let Some(index) = stack.iter().rposition(|frame| &frame.family == family) {
                    if *value == 1 {
                        stack.truncate(index + 1);
                        stack[index].value = 1;
                        chosen = Some((family.clone(), *value, "illegal_restart", index + 1));
                        break;
                    }
                    if *value > stack[index].value + 1 {
                        stack.truncate(index + 1);
                        stack[index].value = *value;
                        chosen = Some((family.clone(), *value, "jump_forward", index + 1));
                        break;
                    }
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in &choices {
                if !stack.iter().any(|frame| &frame.family == family)
                    && stack.len() < MAX_OUTLINE_DEPTH
                {
                    stack.push(Frame {
                        family: family.clone(),
                        value: *value,
                    });
                    chosen = Some((family.clone(), *value, "open_midcounter", stack.len()));
                    break;
                }
            }
        }
        let Some((family, value, action, level)) = chosen else {
            let (family, value) = choices
                .first()
                .map(|(family, value, _)| (family.clone(), Some(*value)))
                .unwrap_or_else(|| ("unknown".to_owned(), None));
            violations += 1;
            stats.entry(family.clone()).or_default().violations += 1;
            assignments.push(assigned(
                candidate,
                &family,
                value.map_or(Value::Null, |number| json!(number)),
                Value::Null,
                "violation",
            ));
            continue;
        };
        let family_stats = stats.entry(family.clone()).or_default();
        if action == "illegal_restart" {
            violations += 1;
            family_stats.violations += 1;
        } else if matches!(action, "jump_forward" | "open_midcounter") {
            gaps += 1;
            family_stats.gaps += 1;
        }
        family_stats.count += 1;
        family_stats.max_value = family_stats.max_value.max(value);
        *family_stats.level_votes.entry(level).or_default() += 1;
        assignments.push(assigned(
            candidate,
            &family,
            json!(value),
            json!(level),
            action,
        ));
    }

    let families = stats
        .into_iter()
        .map(|(family, stats)| {
            let footnote_suspect = (family.starts_with("numeric_")
                || family.starts_with("legal_numeric_"))
                && stats.max_value >= FOOTNOTE_SUSPECT_MIN_VALUE
                && stats.level_votes.len() == 1;
            let level_votes = stats
                .level_votes
                .into_iter()
                .map(|(level, count)| (level.to_string(), json!(count)))
                .collect::<Map<_, _>>();
            (
                family,
                json!({
                    "count": stats.count,
                    "max_value": stats.max_value,
                    "violations": stats.violations,
                    "gaps": stats.gaps,
                    "level_votes": level_votes,
                    "footnote_suspect": footnote_suspect,
                }),
            )
        })
        .collect::<Map<_, _>>();
    let enumerator_count = assignments
        .iter()
        .filter(|row| row.get("action").and_then(Value::as_str) != Some("caps_observed"))
        .count();
    let status = if enumerator_count == 0 {
        "no_enumerators"
    } else if violations == 0 {
        "parsed_clean"
    } else if violations <= (enumerator_count / 5).max(1) {
        "parsed_with_violations"
    } else {
        "unparseable"
    };
    json!({
        "assignments": assignments,
        "violations": violations,
        "gaps": gaps,
        "families": families,
        "status": status,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_vectors_include_the_full_reporter_inventory() {
        let vectors = [
            ("Legal Principles", true),
            ("Background and Overview", true),
            ("Member's Right To Fair Treatment", true),
            ("D. Tax Cas. 1088.", false),
            (
                "Introduction 1 Ont Liquor Licence App Trib Dec 2 Analysis",
                false,
            ),
            ("Background 2026 Overview", true),
        ];
        for (value, expected) in vectors {
            assert_eq!(heading_text_plausible(value), expected, "{value}");
        }
    }

    #[test]
    fn statute_citations_are_protected_from_footnote_pairing() {
        let text = "Criminal Code, RSC 1985, c C-46, s 7";
        let protected = protected_citation_spans(text)
            .into_iter()
            .map(|(start, end)| {
                text.chars()
                    .skip(start)
                    .take(end - start)
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert!(protected.iter().any(|span| span == "RSC 1985, c C-46"));
    }

    #[test]
    fn citation_signal_boundaries_survive_the_linear_regex_set() {
        for text in ["under s. 7", "see paras 12-14", "at p. 123", "supra note 4"] {
            assert!(has_citation_signal(text), "{text}");
        }
        for text in ["class 7", "xss. 12", "at p. 12345"] {
            assert!(!has_citation_signal(text), "{text}");
        }
    }

    #[test]
    fn mcgill_tenth_inventory_preserves_journal_punctuation() {
        let abbreviations: Vec<String> = serde_json::from_str(include_str!(
            "../../src/legalpdf/data/mcgill_reporters.json"
        ))
        .unwrap();
        assert_eq!(abbreviations.len(), 2_110);
        for abbreviation in [
            "Alta L Rev",
            "Nat'l J Sexual Orientation L",
            "Actualités-Justice",
            "Res Communes: Vermont's J Env't",
            "J Energy, Nat'l Res & Envtl L",
        ] {
            let pattern = Regex::new(&format!(
                "^(?:{})$",
                reporter_abbreviation_regex(abbreviation)
            ))
            .unwrap();
            assert!(pattern.is_match(abbreviation), "{abbreviation}");
            assert!(
                has_citation_signal(&format!("1 {abbreviation} 2")),
                "{abbreviation}"
            );
        }
    }
}
