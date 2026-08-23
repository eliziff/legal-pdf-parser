use crate::{text::ScalarText, ScalarRange};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::LazyLock,
};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DefinitionOccurrence {
    pub range: ScalarRange,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub node_id: Option<String>,
    pub source_paragraph_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_artifact_id: Option<String>,
}

pub type DefinitionParagraph = DefinitionOccurrence;

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

static PAREN: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\(([^()]*)\)").unwrap());
static QUOTED: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([A-Z][A-Za-z0-9&'\- ]{0,79})""#).unwrap());
static LIST: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r#"^"([A-Z][A-Za-z0-9&'\- ]{0,79})""#,
        r"[\u{0009}-\u{000D}\u{0020}\u{00A0}\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}\u{FEFF}]+",
        r"(?:means|shall mean|has the meaning|shall have the meaning)",
    ))
    .unwrap()
});

fn occurrence(
    document: &ScalarText<'_>,
    paragraph: &DefinitionParagraph,
    start: usize,
    end: usize,
) -> DefinitionOccurrence {
    DefinitionOccurrence {
        range: ScalarRange {
            start: document.scalar(start),
            end: document.scalar(end),
        },
        node_id: paragraph.node_id.clone(),
        source_paragraph_id: paragraph.source_paragraph_id.clone(),
        source_artifact_id: paragraph.source_artifact_id.clone(),
    }
}

#[rustfmt::skip]
pub fn derive_definitions(text: &str, paragraphs: &[DefinitionParagraph]) -> DefinitionsResult {
    let document = ScalarText::new(text);
    let slices = paragraphs.iter().map(|p| {
        let base = document.byte(p.range.start);
        (base, document.slice(p.range.start..p.range.end).unwrap())
    }).collect::<Vec<_>>();
    let mut terms = Vec::<(String, Vec<(usize, DefinitionOccurrence)>)>::new();
    let mut indexes = HashMap::<String, usize>::new();
    for (paragraph_index, ((base, value), paragraph)) in slices.iter().zip(paragraphs).enumerate() {
        let mut detected = Vec::new();
        for content in PAREN.captures_iter(value).filter_map(|c| c.get(1))
            .filter(|m| (1..=200).contains(&m.as_str().encode_utf16().count())) {
            detected.extend(QUOTED.captures_iter(content.as_str()).filter_map(|c| c.get(1))
                .map(|m| (m.as_str().to_owned(), content.start() + m.start(), content.start() + m.end())));
        }
        if let Some(captures) = LIST.captures(value).filter(|c| !value[c.get(0).unwrap().end()..]
            .chars().next().is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')) {
            let term = captures.get(1).unwrap();
            detected.push((term.as_str().to_owned(), term.start(), term.end()));
        }
        let mut seen = HashSet::new();
        for (term, start, end) in detected.into_iter().filter(|(term, _, _)| seen.insert(term.clone())) {
            let next = terms.len();
            let index = *indexes.entry(term.clone()).or_insert(next);
            if index == next { terms.push((term, Vec::new())); }
            terms[index].1.push((paragraph_index, occurrence(&document, paragraph, base + start, base + end)));
        }
    }
    let mut result = Vec::with_capacity(terms.len());
    for (term, definitions) in terms {
        let defined_in = definitions.iter().map(|(p, _)| *p).collect::<HashSet<_>>();
        let variants = [term.clone(), term.strip_suffix('s').map_or_else(|| format!("{term}s"), str::to_owned)];
        let patterns = variants.map(|variant| Regex::new(&regex::escape(&variant)).unwrap());
        let mut uses = Vec::new();
        for (paragraph_index, ((base, value), paragraph)) in slices.iter().zip(paragraphs).enumerate() {
            if defined_in.contains(&paragraph_index) { continue; }
            let mut matches = patterns.iter().flat_map(|pattern| pattern.find_iter(value))
                .filter(|m| !value[..m.start()].chars().next_back().is_some_and(|c| c.is_ascii_alphanumeric())
                    && !value[m.end()..].chars().next().is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit()))
                .map(|m| (m.start(), m.end())).collect::<Vec<_>>();
            matches.sort_unstable();
            matches.dedup();
            uses.extend(matches.into_iter().map(|(start, end)| occurrence(&document, paragraph, base + start, base + end)));
        }
        result.push(DefinedTerm { term, definitions: definitions.into_iter().map(|(_, occurrence)| occurrence).collect(), uses });
    }
    DefinitionsResult { terms: result }
}
