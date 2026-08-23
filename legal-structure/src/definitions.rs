use crate::{EngineError, ScalarRange};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DefinitionParagraph {
    pub range: ScalarRange,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub source_paragraph_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_artifact_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DefinitionOccurrence {
    pub range: ScalarRange,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub source_paragraph_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_artifact_id: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DefinedTerm {
    pub term: String,
    pub definitions: Vec<DefinitionOccurrence>,
    pub uses: Vec<DefinitionOccurrence>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DefinitionsResult {
    pub terms: Vec<DefinedTerm>,
}

type DetectedDefinition = (String, usize, usize);
type PendingTerm = (String, Vec<(usize, DefinitionOccurrence)>);

fn parenthetical_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"\(([^()]*)\)").unwrap())
}

fn quoted_term_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r#""([A-Z][A-Za-z0-9&'\- ]{0,79})""#).unwrap())
}

fn list_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(concat!(
            r#"^"([A-Z][A-Za-z0-9&'\- ]{0,79})""#,
            r"[\u{0009}-\u{000D}\u{0020}\u{00A0}\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}\u{FEFF}]+",
            r"(?:means|shall mean|has the meaning|shall have the meaning)",
        ))
        .unwrap()
    })
}

fn parenthetical_definitions(text: &str) -> Vec<DetectedDefinition> {
    parenthetical_pattern()
        .captures_iter(text)
        .filter_map(|captures| captures.get(1))
        .filter(|content| (1..=200).contains(&content.as_str().encode_utf16().count()))
        .flat_map(|content| {
            quoted_term_pattern()
                .captures_iter(content.as_str())
                .filter_map(move |captures| {
                    let term = captures.get(1)?;
                    Some((
                        term.as_str().to_owned(),
                        content.start() + term.start(),
                        content.start() + term.end(),
                    ))
                })
        })
        .collect()
}

fn list_definition(text: &str) -> Option<DetectedDefinition> {
    let captures = list_pattern().captures(text)?;
    let after = captures.get(0)?.end();
    if text[after..]
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    let term = captures.get(1)?;
    Some((term.as_str().to_owned(), term.start(), term.end()))
}

fn occurrence(
    document: &crate::text::ScalarText<'_>,
    paragraph: &DefinitionParagraph,
    start_byte: usize,
    end_byte: usize,
) -> DefinitionOccurrence {
    DefinitionOccurrence {
        range: ScalarRange {
            start: document.scalar(start_byte),
            end: document.scalar(end_byte),
        },
        node_id: paragraph.node_id.clone(),
        source_paragraph_id: paragraph.source_paragraph_id.clone(),
        source_artifact_id: paragraph.source_artifact_id.clone(),
    }
}

fn left_boundary(character: Option<char>) -> bool {
    !character.is_some_and(|value| value.is_ascii_alphanumeric())
}

fn right_boundary(character: Option<char>) -> bool {
    !character.is_some_and(|value| value.is_ascii_lowercase() || value.is_ascii_digit())
}

fn use_ranges<'a>(text: &'a str, pattern: &'a Regex) -> impl Iterator<Item = (usize, usize)> + 'a {
    pattern.find_iter(text).filter_map(|found| {
        let (start, end) = (found.start(), found.end());
        (left_boundary(text[..start].chars().next_back())
            && right_boundary(text[end..].chars().next()))
        .then_some((start, end))
    })
}

pub fn derive_definitions(
    text: &str,
    paragraphs: &[DefinitionParagraph],
) -> Result<DefinitionsResult, EngineError> {
    let document = crate::text::ScalarText::new(text);
    let mut previous_end = 0;
    for paragraph in paragraphs {
        if !paragraph.range.valid(document.len()) || paragraph.range.start < previous_end {
            return Err(EngineError::invalid(
                "definition paragraphs must be ordered, non-overlapping scalar ranges",
            ));
        }
        if paragraph.source_paragraph_id.is_empty()
            || paragraph.source_artifact_id.as_deref() == Some("")
        {
            return Err(EngineError::invalid(
                "definition source identifiers must be non-empty",
            ));
        }
        previous_end = paragraph.range.end;
    }
    let mut terms = Vec::<PendingTerm>::new();
    let mut term_indexes = HashMap::<String, usize>::new();
    for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
        let paragraph_byte = document.byte(paragraph.range.start);
        let paragraph_text = document
            .slice(paragraph.range.start..paragraph.range.end)
            .expect("validated definition paragraph range");
        let mut detected = parenthetical_definitions(paragraph_text);
        if let Some(list) = list_definition(paragraph_text) {
            detected.push(list);
        }
        let mut seen = HashSet::new();
        for (term, start, end) in detected {
            if !seen.insert(term.clone()) {
                continue;
            }
            let next = terms.len();
            let index = *term_indexes.entry(term.clone()).or_insert(next);
            if index == next {
                terms.push((term, Vec::new()));
            }
            terms[index].1.push((
                paragraph_index,
                occurrence(
                    &document,
                    paragraph,
                    paragraph_byte + start,
                    paragraph_byte + end,
                ),
            ));
        }
    }
    let mut result = Vec::with_capacity(terms.len());
    for (term, definitions) in terms {
        let defined_in = definitions
            .iter()
            .map(|(paragraph, _)| *paragraph)
            .collect::<HashSet<_>>();
        let mut variants = vec![term.clone()];
        let variant = term
            .strip_suffix('s')
            .map_or_else(|| format!("{term}s"), str::to_owned);
        if variant != term {
            variants.push(variant);
        }
        let patterns = variants
            .iter()
            .map(|variant| Regex::new(&regex::escape(variant)).unwrap())
            .collect::<Vec<_>>();
        let mut uses = Vec::new();
        for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
            if defined_in.contains(&paragraph_index) {
                continue;
            }
            let paragraph_byte = document.byte(paragraph.range.start);
            let paragraph_text = document
                .slice(paragraph.range.start..paragraph.range.end)
                .expect("validated definition paragraph range");
            let mut matches = patterns
                .iter()
                .flat_map(|pattern| use_ranges(paragraph_text, pattern))
                .collect::<Vec<_>>();
            matches.sort_unstable();
            matches.dedup();
            uses.extend(matches.into_iter().map(|(start, end)| {
                occurrence(
                    &document,
                    paragraph,
                    paragraph_byte + start,
                    paragraph_byte + end,
                )
            }));
        }
        result.push(DefinedTerm {
            term,
            definitions: definitions
                .into_iter()
                .map(|(_, occurrence)| occurrence)
                .collect(),
            uses,
        });
    }
    Ok(DefinitionsResult { terms: result })
}
