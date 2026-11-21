//! A line's opening marker as the numbering grammar reads it, and the readings the stages ask of it.

use crate::model::MarkerReading;

/// The words a heading's number may follow, its number counting under the word ("PART II", "Article 5", "SCHEDULE A").
pub const HEADING_WORDS: [&str; 15] = [
    "part", "section", "article", "chapter", "schedule", "division", "title", "book", "appendix",
    "annex", "exhibit", "count", "partie", "chapitre", "annexe",
];

/// A heading's number as it opens its line, a stop or a bracket after it ("4.", "4.1", "iv.", "B)"), alone or before its
/// title.
pub fn enumerated(text: &str) -> bool {
    enumerated_in(&marker_readings(text, false))
}

/// `enumerated`, of a line's readings.
pub fn enumerated_in(readings: &[MarkerReading]) -> bool {
    readings.iter().any(|reading| {
        !reading.worded()
            && (reading.family.starts_with(".:")
                || reading.family.starts_with("):")
                || reading.family.contains("1.1"))
    })
}

/// A line's text with the space a recognizer leaves inside its opening bracket closed up ("[ 13] Having ...").
pub fn closed_up(text: &str) -> std::borrow::Cow<'_, str> {
    match text.trim_start().strip_prefix("[ ") {
        Some(rest) => format!("[{}", rest.trim_start()).into(),
        None => text.into(),
    }
}

/// Every reading of a line's opening token by the numbering grammar: read alone and, where a single
/// letter is also a roman numeral, as the letter after the letter before it ("(i)" as 1 and after
/// "(h)"); and, where `division` asks and the line opens on a heading's word (`HEADING_WORDS`), the number after the word.
pub fn marker_readings(text: &str, division: bool) -> Vec<MarkerReading> {
    let text = closed_up(text);
    let mut tokens = text.split_whitespace();
    let Some(first) = tokens.next().filter(|token| token.chars().count() <= 12) else {
        return Vec::new();
    };
    let read = |token: &str, word: Option<&str>| {
        let alone = legal_structure_model::layout::numbering::numbering_marker(token, None);
        let letter = alone
            .as_ref()
            .filter(|m| m.family.ends_with(":i") || m.family.ends_with(":I"))
            .and_then(|m| {
                let core = token.trim_matches(|c: char| !c.is_ascii_alphabetic());
                let c = core.chars().next().filter(|_| core.chars().count() == 1)?;
                let family = format!(
                    "{}{}",
                    &m.family[..m.family.len() - 1],
                    if c.is_ascii_uppercase() { 'A' } else { 'a' }
                );
                let ordinal = u32::from(c.to_ascii_lowercase()) - u32::from('a');
                legal_structure_model::layout::numbering::numbering_marker(
                    token,
                    Some(&legal_structure_model::NumberingMarker {
                        family,
                        ordinal,
                        prefix: Vec::new(),
                    }),
                )
            });
        alone
            .into_iter()
            .chain(letter)
            .map(|m| MarkerReading {
                family: word.map_or(m.family.clone(), |word| {
                    format!("{}:{}", word.to_lowercase(), m.family)
                }),
                value: m.ordinal,
                prefix: m.prefix,
            })
            .collect::<Vec<_>>()
    };
    let mut found = read(first, None);
    if division && HEADING_WORDS.contains(&first.to_lowercase().as_str()) {
        if let Some(number) = tokens.next() {
            found.extend(read(
                number.trim_end_matches(['.', ':', '\u{2014}', '-']),
                Some(first),
            ));
        }
    }
    found.dedup();
    found
}

/// A line's heading numbers (`MarkerReading::heading`), a heading word's among them.
pub fn heading_numbers(text: &str) -> Vec<MarkerReading> {
    marker_readings(text, true)
        .into_iter()
        .filter(MarkerReading::heading)
        .collect()
}
