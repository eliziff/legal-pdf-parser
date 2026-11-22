//! A contents list's words: its title, a word index's letter, a locator's number.

use legal_pdf_core::model::Line;
use legal_pdf_core::regex;

pub fn value(token: &str) -> u32 {
    token
        .parse()
        .ok()
        .or_else(|| legal_pdf_core::roman_value(token))
        .unwrap_or(0)
}

/// A page label's number: its figures, or its roman numeral's value.
pub fn page_numeral(label: &str) -> Option<u32> {
    let label = label.trim();
    (label.chars().all(|c| c.is_ascii_digit()) || legal_pdf_core::roman_value(label).is_some())
        .then(|| value(label))
        .filter(|n| *n > 0)
}

/// A page's or a list's own title: the parser's contents titles ("TABLE OF CONTENTS", "TABLE OF
/// PROVISIONS"), the same continued ("CONTENTS (cont'd)"), and an index's or a summary's ("INDEX
/// OF EXHIBITS", "I N D E X", "SOMMAIRE").
pub fn contents_title(text: &str) -> bool {
    crate::text::text::contents_heading(text)
        || regex!(r"(?i)^\s*(?:(?:table\s+of\s+)?contents|table\s+des\s+mati[e\u{00e8}]res|index(?:\s+of\s+\w+)?|sommaire|i\s*n\s*d\s*e\s*x)\b").is_match(text)
}

/// A word index's letter: "< A >", "- B -"; a heading over its entries, not an entry.
pub fn index_letter(text: &str) -> bool {
    regex!(r"^\s*(?:<\s*[A-Z0-9]\s*>|-\s*[A-Z]\s*-)\s*$").is_match(text)
}

/// A locator alone, its value and its place as printed ("12", "xiv", a range "18-43"): "12", "xiv",
/// "p. 3", "[18] - [43]", "1 to 27".
pub fn locator(text: &str) -> Option<(u32, String)> {
    regex!(r"(?i)^\s*(?:p(?:age|p?\.|ara\.?)?\s*)?\[?(\d{1,4}|[ivxlc]{1,7})\]?(?:\s*(?:[-\u{2013}]|to)\s*\[?(\d{1,4})\]?)?\s*$")
        .captures(text)
        .map(|found| (value(&found[1]), place(&found)))
}

/// A locator after a leader at a title's end ("Overview ........ 3"), or after a tab or a wide
/// space the title is typeset with ("Introduction\u{2003}3"): where the title ends, and the locator.
pub fn leader_locator(text: &str) -> Option<(usize, u32, String)> {
    regex!(r"(?:(?:\.\s?){2,}|_{3,}|\u{2026}+|[\t\u{2002}\u{2003}\u{2007}])\s*(?:p(?:age|p?\.)?\s*)?\[?(\d{1,4}|[ivxlc]{1,6})\]?(?:\s*(?:[-\u{2013}]|to)\s*\[?(\d{1,4})\]?)?\s*$")
        .captures(text)
        .map(|found| (found.get(0).map_or(text.len(), |all| all.start()), value(&found[1]), place(&found)))
}

/// A locator's place as printed: its first value, and the last where it gives a range ("18-43").
pub fn place(found: &regex::Captures) -> String {
    found.get(2).map_or_else(
        || found[1].to_owned(),
        |last| format!("{}-{}", &found[1], last.as_str()),
    )
}

/// One row of a page: its lines left to right, its text, its title (the text its locator aside),
/// and the locator it closes on (value, right edge, as printed).
pub struct Row<'a> {
    pub lines: Vec<&'a Line>,
    pub text: String,
    pub title: String,
    pub locator: Option<(u32, f64, String)>,
    /// Its locator was printed alone on the row below it (`paragraph_range`).
    pub stacked: bool,
}

/// A row read from the lines sharing its baseline band, left to right: its locator alone in the
/// row's last cell, after a leader or a wide space, or set apart by a gap of three line heights.
pub fn row_of(lines: Vec<&Line>) -> Row<'_> {
    let text = lines
        .iter()
        .map(|line| line.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    let last = lines[lines.len() - 1];
    let join = |part: &[&Line]| {
        part.iter()
            .map(|line| line.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    };
    if lines.len() > 1 {
        if let Some((value, token)) = locator(&last.text) {
            let title = join(&lines[..lines.len() - 1]);
            return Row {
                stacked: false,
                locator: Some((value, last.bbox[2], token)),
                title,
                text,
                lines,
            };
        }
    }
    if let Some((end, value, token)) = leader_locator(&text) {
        let title = text[..end]
            .trim_end_matches(|c: char| {
                c == '.' || c == '_' || c == '\u{2026}' || c.is_whitespace()
            })
            .to_owned();
        return Row {
            stacked: false,
            locator: Some((value, last.bbox[2], token)),
            title,
            text,
            lines,
        };
    }
    let height = (last.bbox[3] - last.bbox[1]).max(1.0);
    if let [.., before, word] = last.words.as_slice() {
        if word.bbox[0] - before.bbox[2] >= 3.0 * height {
            if let (Some((value, token)), Some(title)) =
                (locator(&word.text), text.strip_suffix(word.text.trim()))
            {
                let title = title.trim_end().to_owned();
                return Row {
                    stacked: false,
                    locator: Some((value, last.bbox[2], token)),
                    title,
                    text,
                    lines,
                };
            }
        }
    }
    let alone = (lines.len() == 1).then(|| locator(&last.text)).flatten();
    Row {
        stacked: false,
        locator: alone.map(|(value, token)| (value, last.bbox[2], token)),
        title: if lines.len() == 1 && locator(&text).is_some() {
            String::new()
        } else {
            text.clone()
        },
        text,
        lines,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_citations as _;
    use legal_citations_data as _;

    #[test]
    fn contents_titles_and_en_dash_locators_match_utf8_properly() {
        assert!(contents_title("Table des mati\u{00e8}res"));
        assert!(contents_title("TABLE OF CONTENTS"));
        assert_eq!(
            leader_locator("I. INTRODUCTION ............ 444").map(|(_, value, token)| (value, token)),
            Some((444, "444".to_owned()))
        );
        assert_eq!(locator("18\u{2013}43"), Some((18, "18-43".to_owned())));
    }
}
