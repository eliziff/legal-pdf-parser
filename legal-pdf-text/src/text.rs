//! The shape of a line's text, as the readers test it: the enumerator opening it and the title
//! after, capitals, a sentence's end, title case, brackets, a contents list's own title.

use regex::Regex;
use std::sync::OnceLock;
pub use legal_pdf_core::model::sentence_ended;

/// A heading's enumerator, alone on its line ("IV.", "(a)" aside, "b)", "12."), or a decimal
/// section number ("1.1", "2.3.1") with or without a period after it.
pub fn heading_standalone_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:([IVXLCDM]{1,7}|[A-Za-z](?:\.\d+)?|\d{1,3})([.)])|(\d{1,2}(?:\.\d{1,2}){1,3})([.)]?))\s*$",
        )
        .unwrap()
    })
}

/// A heading's enumerator with its title on the line: as [`heading_standalone_re`], the title after
/// it ("IV. Analysis", "1.1 Definitions").
pub fn heading_inline_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:([IVXLCDM]{1,7}|[A-Za-z](?:\.\d+)?|\d{1,3})([.)])|(\d{1,2}(?:\.\d{1,2}){1,3})([.)]?))\s+(\S.*)$",
        )
        .unwrap()
    })
}

/// (value, punctuation, title) of a heading enumerator match from either lenient regex.
pub fn heading_enumerator<'t>(capture: &regex::Captures<'t>) -> (&'t str, &'t str, &'t str) {
    let (value, punct) = match (capture.get(1), capture.get(3)) {
        (Some(value), _) => (value.as_str(), capture.get(2).map_or("", |m| m.as_str())),
        (None, Some(value)) => (value.as_str(), capture.get(4).map_or("", |m| m.as_str())),
        (None, None) => ("", ""),
    };
    (value, punct, capture.get(5).map_or("", |m| m.as_str()).trim())
}

/// A party's role set as a label, on its own line or past a column gap after the party's name ("—
/// Respondent", "ACME LTD.      Plaintiff", "First and Second Defendants").
pub fn party_label() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        let ordinal = r"(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|\d+(?:st|nd|rd|th))";
        let roles = legal_pdf_core::model::PARTY_ROLES;
        let role = format!(r"(?:{ordinal}(?:\s*(?:,|and|&|to)\s*{ordinal})*\s+)?(?:named\s+)?(?:{roles})");
        Regex::new(&format!(
            r"(?i)(?:^[\-\u{{2013}}\u{{2014}}\u{{2015}}\u{{2026}}.\s]*|\s{{3,}})(?:the\s+)?{role}(?:\s*(?:/|,|and|&|\()\s*(?:the\s+)?{role})*[\s.,:;)]*$"
        ))
        .expect("party label")
    })
}

/// Whether a line opens on a heading's enumerator, alone or before its title.
pub fn enumerated(text: &str) -> bool {
    heading_inline_re().is_match(text) || heading_standalone_re().is_match(text)
}

/// "[12]" alone on its line; four digits would be a bracketed year.
pub fn standalone_paragraph_number(text: &str) -> bool {
    text.trim()
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(|number| {
            (1..=3).contains(&number.trim().len())
                && number.trim().bytes().all(|byte| byte.is_ascii_digit())
        })
}

pub fn all_caps(text: &str) -> bool {
    let mut letters = text.chars().filter(|character| character.is_alphabetic());
    letters.next().is_some_and(|first| {
        first.is_uppercase() && letters.all(|character| character.is_uppercase())
    })
}

/// A line of six words or more, set in lower case, ending on a word no sentence ends on, so that
/// its sentence goes on at the next body line whatever that opens with: an article, preposition,
/// conjunction, pronoun or auxiliary, English or French ("of the", "à la", "and"), a citation's
/// abbreviation its number completes ("s.", "para.", "R. v."), or a title its name completes ("Mr.",
/// "Mme").
pub fn dangling_end(text: &str) -> bool {
    const WORDS: &[&str] = &[
        "the", "a", "an", "of", "to", "in", "on", "at", "by", "for", "with", "from", "into", "onto", "upon", "and", "or",
        "nor", "but", "that", "which", "who", "whom", "whose", "is", "are", "was", "were", "be", "been", "being", "as",
        "than", "its", "their", "his", "her", "this", "these", "those", "such", "if", "whether", "not", "under", "between",
        "within", "without", "against", "during", "before", "after", "about", "over", "per", "via", "le", "la", "les", "un",
        "une", "des", "du", "de", "à", "au", "aux", "et", "ou", "en", "par", "pour", "sur", "dans", "que", "qui", "dont",
        "avec", "sans", "sous", "entre", "ce", "cette", "ces", "son", "sa", "ses", "leur", "leurs",
    ];
    const ABBREVIATIONS: &[&str] = &[
        "s.", "ss.", "art.", "arts.", "para.", "paras.", "p.", "pp.", "no.", "nos.", "c.", "ch.", "v.", "vol.", "r.", "rr.",
        "cl.", "sch.", "reg.", "regs.", "sec.", "secs.", "mr.", "mrs.", "ms.", "messrs.", "dr.", "hon.", "prof.", "st.",
        "m.", "mme", "mme.", "mmes",
    ];
    let words: Vec<&str> = text.split_whitespace().collect();
    if words.len() < 6 || !text.chars().any(char::is_lowercase) {
        return false;
    }
    let last = words[words.len() - 1].to_lowercase();
    WORDS.contains(&last.as_str()) || ABBREVIATIONS.contains(&last.as_str())
}

pub fn bracket_excess(text: &str) -> usize {
    let mut stack = Vec::new();
    for character in text.chars() {
        match character {
            '(' => stack.push(')'),
            '[' => stack.push(']'),
            '{' => stack.push('}'),
            '\u{201c}' => stack.push('\u{201d}'),
            ')' | ']' | '}' | '\u{201d}' if stack.last() == Some(&character) => {
                stack.pop();
            }
            _ => {}
        }
    }
    stack.len()
}

pub fn closes_before_opening(text: &str) -> bool {
    text.chars()
        .take(48)
        .find_map(|character| match character {
            ')' | ']' | '}' | '\u{201d}' => Some(true),
            '(' | '[' | '{' | '\u{201c}' => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

pub fn starts_note_or_list(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:(?:[-*•])|(?:\(?\d{1,4}[.)])|(?:\(?[A-Za-z][.)])|(?:[IVXLCDMivxlcdm]{1,8}[.)])|(?:[*†‡§]))\s+\S",
        )
        .unwrap()
    })
    .is_match(text)
}

pub fn contents_heading(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "contents"
            | "table of contents"
            | "table des matières"
            | "table of provisions"
            | "table analytique"
    )
}

/// A bare page number: digits alone on the line, a dash on either side at most ("- 12 -").
pub fn arabic_page_number(value: &str) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\s*-?\s*(\d{1,4})\s*-?\s*$").unwrap())
        .captures(value)
        .and_then(|capture| capture.get(1))
        .and_then(|value| value.as_str().parse().ok())
}

/// A line ending on a word broken by a hyphen.
pub fn hyphen_fragment_tail(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[A-Za-z]{2,}[-\u{00ac}\u{00ad}]$").unwrap())
        .is_match(text.trim_end())
}

/// A line opening on the rest of a broken word.
pub fn hyphen_continuation(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^[A-Za-z]{2,}").unwrap())
        .is_match(text.trim_start())
}

pub fn scalar_suffix(value: &str, start: usize) -> &str {
    if value.is_ascii() {
        return &value[start.min(value.len())..];
    }
    let end = value.len();
    let byte = value.char_indices().nth(start).map_or(end, |at| at.0);
    &value[byte..]
}
