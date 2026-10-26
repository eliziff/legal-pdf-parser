//! The shape of a line's text, as the readers test it: the enumerator opening it and the title
//! after, capitals, a sentence's end, title case, brackets, a contents list's own title.

use super::*;

/// A heading's enumerator, alone on its line ("IV.", "(a)" aside, "b)", "12."), or a decimal
/// section number ("1.1", "2.3.1") with or without a period after it.
pub(super) fn heading_standalone_re() -> &'static Regex {
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
pub(super) fn heading_inline_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:([IVXLCDM]{1,7}|[A-Za-z](?:\.\d+)?|\d{1,3})([.)])|(\d{1,2}(?:\.\d{1,2}){1,3})([.)]?))\s+(\S.*)$",
        )
        .unwrap()
    })
}

/// (value, punctuation, title) of a heading enumerator match from either lenient regex.
pub(super) fn heading_enumerator<'t>(capture: &regex::Captures<'t>) -> (&'t str, &'t str, &'t str) {
    let (value, punct) = match (capture.get(1), capture.get(3)) {
        (Some(value), _) => (value.as_str(), capture.get(2).map_or("", |m| m.as_str())),
        (None, Some(value)) => (value.as_str(), capture.get(4).map_or("", |m| m.as_str())),
        (None, None) => ("", ""),
    };
    (value, punct, capture.get(5).map_or("", |m| m.as_str()).trim())
}

/// Whether a line opens on a heading's enumerator, alone or before its title.
pub(super) fn enumerated(text: &str) -> bool {
    heading_inline_re().is_match(text) || heading_standalone_re().is_match(text)
}

/// The title after a leading enumerator, or the whole text.
pub(super) fn title_after_enumerator(text: &str) -> &str {
    heading_inline_re()
        .captures(text.trim())
        .map(|capture| heading_enumerator(&capture).2)
        .unwrap_or(text.trim())
}

/// A title set in capitals: the typographic evidence a recognized page keeps when it loses bold
/// and size.
pub(super) fn caps_title(text: &str) -> bool {
    let title = title_after_enumerator(text);
    title.chars().filter(|c| c.is_alphabetic()).count() >= 4 && all_caps(title) && !sentence_ended(title)
}

/// "[12]" alone on its line; four digits would be a bracketed year.
pub(super) fn standalone_paragraph_number(text: &str) -> bool {
    text.trim()
        .strip_prefix('[')
        .and_then(|rest| rest.strip_suffix(']'))
        .is_some_and(|number| {
            (1..=3).contains(&number.trim().len())
                && number.trim().bytes().all(|byte| byte.is_ascii_digit())
        })
}

pub(super) fn all_caps(text: &str) -> bool {
    let mut letters = text.chars().filter(|character| character.is_alphabetic());
    letters.next().is_some_and(|first| {
        first.is_uppercase() && letters.all(|character| character.is_uppercase())
    })
}

/// A sentence's end: a full stop, question or exclamation mark or an ellipsis, before any closing
/// quotes and brackets.
pub(super) fn sentence_ended(text: &str) -> bool {
    text.trim_end()
        .trim_end_matches(|c: char| "\"'\u{2019}\u{201d}\u{bb})]}".contains(c))
        .ends_with(['.', '?', '!', '\u{2026}'])
}

pub(super) fn bracket_excess(text: &str) -> usize {
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

pub(super) fn closes_before_opening(text: &str) -> bool {
    text.chars()
        .take(48)
        .find_map(|character| match character {
            ')' | ']' | '}' | '\u{201d}' => Some(true),
            '(' | '[' | '{' | '\u{201c}' => Some(false),
            _ => None,
        })
        .unwrap_or(false)
}

pub(super) fn starts_note_or_list(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^\s*(?:(?:[-*•])|(?:\(?\d{1,4}[.)])|(?:\(?[A-Za-z][.)])|(?:[IVXLCDMivxlcdm]{1,8}[.)])|(?:[*†‡§]))\s+\S",
        )
        .unwrap()
    })
    .is_match(text)
}

pub(super) fn titlecase_ratio(text: &str) -> f64 {
    let mut words = 0;
    let titlecase = text
        .split_whitespace()
        .filter(|word| word.chars().any(char::is_alphabetic))
        .inspect(|_| words += 1)
        .filter(|word| word.chars().next().is_some_and(char::is_uppercase))
        .count();
    if words == 0 {
        0.0
    } else {
        titlecase as f64 / words as f64
    }
}

pub(super) fn contents_heading(text: &str) -> bool {
    matches!(
        text.trim().to_lowercase().as_str(),
        "contents"
            | "table of contents"
            | "table des matières"
            | "table of provisions"
            | "table analytique"
    )
}
