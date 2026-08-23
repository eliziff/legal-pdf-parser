use crate::{text::javascript_whitespace, EngineError, ScalarRange};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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

struct DetectedDefinition {
    term: String,
    start_byte: usize,
    end_byte: usize,
}

struct PendingTerm {
    term: String,
    definitions: Vec<(usize, DefinitionOccurrence)>,
}

fn accepted_term_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'&' | b'\'' | b'-' | b' ')
}

fn quoted_term_at(text: &str, quote: usize) -> Option<(usize, usize, usize)> {
    let bytes = text.as_bytes();
    if bytes.get(quote) != Some(&b'"') {
        return None;
    }
    let start = quote + 1;
    if !bytes.get(start).is_some_and(u8::is_ascii_uppercase) {
        return None;
    }
    let mut end = start + 1;
    while end < bytes.len() && end - start < 80 && accepted_term_byte(bytes[end]) {
        end += 1;
    }
    (bytes.get(end) == Some(&b'"')).then_some((start, end, end + 1))
}

fn quoted_terms(text: &str, start: usize, end: usize) -> Vec<DetectedDefinition> {
    let mut found = Vec::new();
    let mut cursor = start;
    while cursor < end {
        let Some(relative) = text[cursor..end].find('"') else {
            break;
        };
        let quote = cursor + relative;
        if let Some((term_start, term_end, after)) = quoted_term_at(text, quote) {
            if after <= end {
                found.push(DetectedDefinition {
                    term: text[term_start..term_end].to_owned(),
                    start_byte: term_start,
                    end_byte: term_end,
                });
                cursor = after;
                continue;
            }
        }
        cursor = quote + 1;
    }
    found
}

fn parenthetical_definitions(text: &str) -> Vec<DetectedDefinition> {
    let mut found = Vec::new();
    let mut cursor = 0;
    while cursor < text.len() {
        let Some(relative) = text[cursor..].find('(') else {
            break;
        };
        let open = cursor + relative;
        let content_start = open + 1;
        let next = text[content_start..]
            .char_indices()
            .find(|(_, character)| matches!(character, '(' | ')'));
        let Some((relative_end, delimiter)) = next else {
            break;
        };
        let close = content_start + relative_end;
        if delimiter == ')' && (1..=200).contains(&text[content_start..close].chars().count()) {
            found.extend(quoted_terms(text, content_start, close));
            cursor = close + 1;
        } else if delimiter == '(' {
            cursor = close;
        } else {
            cursor = close + 1;
        }
    }
    found
}

fn list_definition(text: &str) -> Option<DetectedDefinition> {
    let (term_start, term_end, mut cursor) = quoted_term_at(text, 0)?;
    let mut whitespace = false;
    while let Some(character) = text[cursor..].chars().next() {
        if !javascript_whitespace(character) {
            break;
        }
        whitespace = true;
        cursor += character.len_utf8();
    }
    if !whitespace {
        return None;
    }
    const VERBS: [&str; 4] = [
        "means",
        "shall mean",
        "has the meaning",
        "shall have the meaning",
    ];
    let verb = VERBS
        .into_iter()
        .find(|verb| text[cursor..].starts_with(verb))?;
    let after = cursor + verb.len();
    if text[after..]
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return None;
    }
    Some(DetectedDefinition {
        term: text[term_start..term_end].to_owned(),
        start_byte: term_start,
        end_byte: term_end,
    })
}

fn occurrence(
    document: &crate::text::ScalarText<'_>,
    paragraph: &DefinitionParagraph,
    node_id: &Option<String>,
    start_byte: usize,
    end_byte: usize,
) -> DefinitionOccurrence {
    DefinitionOccurrence {
        range: ScalarRange {
            start: document.scalar(start_byte),
            end: document.scalar(end_byte),
        },
        node_id: node_id.clone(),
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

fn use_ranges(text: &str, variant: &str) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut cursor = 0;
    while cursor <= text.len() {
        let Some(relative) = text[cursor..].find(variant) else {
            break;
        };
        let start = cursor + relative;
        let end = start + variant.len();
        if left_boundary(text[..start].chars().next_back())
            && right_boundary(text[end..].chars().next())
        {
            found.push((start, end));
        }
        cursor = end;
    }
    found
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
        for definition in detected {
            if !seen.insert(definition.term.clone()) {
                continue;
            }
            let next = terms.len();
            let index = *term_indexes.entry(definition.term.clone()).or_insert(next);
            if index == next {
                terms.push(PendingTerm {
                    term: definition.term,
                    definitions: Vec::new(),
                });
            }
            terms[index].definitions.push((
                paragraph_index,
                occurrence(
                    &document,
                    paragraph,
                    &paragraph.node_id,
                    paragraph_byte + definition.start_byte,
                    paragraph_byte + definition.end_byte,
                ),
            ));
        }
    }
    let mut result = Vec::with_capacity(terms.len());
    for pending in terms {
        let defined_in = pending
            .definitions
            .iter()
            .map(|(paragraph, _)| *paragraph)
            .collect::<HashSet<_>>();
        let mut variants = vec![pending.term.clone()];
        let variant = pending
            .term
            .strip_suffix('s')
            .map_or_else(|| format!("{}s", pending.term), str::to_owned);
        if variant != pending.term {
            variants.push(variant);
        }
        let mut uses = Vec::new();
        for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
            if defined_in.contains(&paragraph_index) {
                continue;
            }
            let paragraph_byte = document.byte(paragraph.range.start);
            let paragraph_text = document
                .slice(paragraph.range.start..paragraph.range.end)
                .expect("validated definition paragraph range");
            let mut matches = variants
                .iter()
                .flat_map(|variant| use_ranges(paragraph_text, variant))
                .collect::<Vec<_>>();
            matches.sort_unstable();
            matches.dedup();
            uses.extend(matches.into_iter().map(|(start, end)| {
                occurrence(
                    &document,
                    paragraph,
                    &paragraph.node_id,
                    paragraph_byte + start,
                    paragraph_byte + end,
                )
            }));
        }
        result.push(DefinedTerm {
            term: pending.term,
            definitions: pending
                .definitions
                .into_iter()
                .map(|(_, occurrence)| occurrence)
                .collect(),
            uses,
        });
    }
    Ok(DefinitionsResult { terms: result })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_legacy_definition_and_use_semantics_in_scalar_coordinates() {
        let text = "😀 (the \"Business Day\" and \"Business Day\")\n\"Units\"\u{feff}shall mean shares\nBusiness Days _Business Day Business DayX Units unit\n\"Units\" means duplicates";
        let scalar = crate::text::ScalarText::new(text);
        let paragraphs = scalar
            .lines()
            .iter()
            .enumerate()
            .map(|(index, [_start, end, scalar_start])| DefinitionParagraph {
                range: ScalarRange {
                    start: *scalar_start,
                    end: scalar.scalar_at_byte(*end).unwrap(),
                },
                node_id: Some(if index < 2 { "outer" } else { "inner" }.to_owned()),
                source_paragraph_id: format!("p{index}"),
                source_artifact_id: Some("artifact".to_owned()),
            })
            .collect::<Vec<_>>();
        let result = derive_definitions(text, &paragraphs).unwrap();
        assert_eq!(
            result
                .terms
                .iter()
                .map(|term| (
                    term.term.as_str(),
                    term.definitions.len(),
                    term.uses
                        .iter()
                        .map(|use_| scalar.slice(use_.range.start..use_.range.end).unwrap())
                        .collect::<Vec<_>>(),
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    "Business Day",
                    1,
                    vec!["Business Days", "Business Day", "Business Day"]
                ),
                ("Units", 2, vec!["Units"]),
            ]
        );
        assert_eq!(
            result.terms[0].definitions[0].range,
            ScalarRange { start: 8, end: 20 }
        );
        assert_eq!(result.terms[0].uses[0].node_id.as_deref(), Some("inner"));
    }
}
