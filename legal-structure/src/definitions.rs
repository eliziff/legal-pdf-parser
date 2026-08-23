use crate::{text::ScalarText, ScalarRange};
use regex::Regex as R;
use serde::{Deserialize, Serialize};
use std::{collections::HashSet, sync::LazyLock};

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

static PAREN: LazyLock<R> = LazyLock::new(|| R::new(r"\(([^()]*)\)").unwrap());
static QUOTED: LazyLock<R> = LazyLock::new(|| R::new(r#""([A-Z][A-Za-z0-9&' -]{0,79})""#).unwrap());
static LIST: LazyLock<R> = LazyLock::new(|| {
    R::new(r#"^"([A-Z][A-Za-z0-9&'\- ]{0,79})"[\u{0009}-\u{000D}\u{0020}\u{00A0}\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}\u{FEFF}]+(?:means|shall mean|has the meaning|shall have the meaning)"#).unwrap()
});

impl DefinitionOccurrence {
    fn at(&self, document: &ScalarText<'_>, start: usize, end: usize) -> Self {
        let mut hit = self.clone();
        (hit.range.start, hit.range.end) = (document.scalar(start), document.scalar(end));
        hit
    }
}

fn bounded(text: &str, start: usize, end: usize) -> bool {
    let bytes = text.as_bytes();
    let left = bytes.get(start.wrapping_sub(1)).copied().unwrap_or(b' ');
    let right = bytes.get(end).copied().unwrap_or(b' ');
    !left.is_ascii_alphanumeric() && !right.is_ascii_lowercase() && !right.is_ascii_digit()
}

pub fn derive_definitions(text: &str, paragraphs: &[DefinitionParagraph]) -> DefinitionsResult {
    let document = ScalarText::new(text);
    let slices = paragraphs
        .iter()
        .map(|p| {
            let base = document.byte(p.range.start);
            (base, document.slice(p.range.start..p.range.end).unwrap())
        })
        .collect::<Vec<_>>();
    let mut terms = Vec::<(String, Vec<(usize, DefinitionOccurrence)>)>::new();
    for (paragraph_index, ((base, text), paragraph)) in slices.iter().zip(paragraphs).enumerate() {
        let mut found = Vec::new();
        for content in PAREN.captures_iter(text).filter_map(|c| c.get(1)) {
            if !(1..=200).contains(&content.as_str().encode_utf16().count()) {
                continue;
            }
            found.extend(QUOTED.captures_iter(content.as_str()).map(|c| {
                let term = c.get(1).unwrap();
                (
                    term.as_str(),
                    content.start() + term.start(),
                    content.start() + term.end(),
                )
            }));
        }
        if let Some(c) = LIST.captures(text).filter(|c| bounded(text, 0, c[0].len())) {
            let term = c.get(1).unwrap();
            found.push((term.as_str(), term.start(), term.end()));
        }
        let mut seen = HashSet::new();
        for (term, start, end) in found.into_iter().filter(|(term, _, _)| seen.insert(*term)) {
            let hit = paragraph.at(&document, base + start, base + end);
            if let Some((_, definitions)) = terms.iter_mut().find(|(known, _)| known == term) {
                definitions.push((paragraph_index, hit));
            } else {
                terms.push((term.to_owned(), vec![(paragraph_index, hit)]));
            }
        }
    }
    let terms = terms
        .into_iter()
        .map(|(term, definitions)| {
            let variants = [
                term.clone(),
                term.strip_suffix('s')
                    .map_or_else(|| format!("{term}s"), str::to_owned),
            ];
            let mut uses = Vec::new();
            for (index, ((base, text), paragraph)) in slices.iter().zip(paragraphs).enumerate() {
                if definitions.iter().any(|(defined, _)| *defined == index) {
                    continue;
                }
                let mut hits = variants
                    .iter()
                    .flat_map(|variant| text.match_indices(variant))
                    .map(|(start, found)| (start, start + found.len()))
                    .filter(|(start, end)| bounded(text, *start, *end))
                    .collect::<Vec<_>>();
                hits.sort_unstable();
                uses.extend(
                    hits.into_iter()
                        .map(|(start, end)| paragraph.at(&document, base + start, base + end)),
                );
            }
            DefinedTerm {
                term,
                definitions: definitions.into_iter().map(|(_, hit)| hit).collect(),
                uses,
            }
        })
        .collect();
    DefinitionsResult { terms }
}
