use crate::{text::ScalarText, ScalarRange};
use aho_corasick::AhoCorasick;
use regex::Regex as R;
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

static PAREN: LazyLock<R> = LazyLock::new(|| R::new(r"\(([^()]*)\)").unwrap());
static QUOTED: LazyLock<R> = LazyLock::new(|| R::new(r#""([A-Z][A-Za-z0-9&' -]{0,79})""#).unwrap());
static LIST: LazyLock<R> = LazyLock::new(|| {
    R::new(r#"^"([A-Z][A-Za-z0-9&'\- ]{0,79})"[\u{0009}-\u{000D}\u{0020}\u{00A0}\u{1680}\u{2000}-\u{200A}\u{2028}\u{2029}\u{202F}\u{205F}\u{3000}\u{FEFF}]+(?:means|shall mean|has the meaning|shall have the meaning)"#).unwrap()
});

impl DefinitionOccurrence {
    fn at(&self, document: &ScalarText<'_>, start: usize, end: usize) -> Self {
        let mut hit = self.clone();
        (hit.range.start, hit.range.end) = (
            document.utf16_at_byte(start).unwrap(),
            document.utf16_at_byte(end).unwrap(),
        );
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
    derive_definitions_indexed(&document, paragraphs)
}

pub(crate) fn derive_definitions_indexed(
    document: &ScalarText<'_>,
    paragraphs: &[DefinitionParagraph],
) -> DefinitionsResult {
    let text = document.value;
    let slices = paragraphs
        .iter()
        .map(|p| {
            let base = document.byte_at_utf16(p.range.start).unwrap();
            let end = document.byte_at_utf16(p.range.end).unwrap();
            (base, &text[base..end])
        })
        .collect::<Vec<_>>();
    let mut terms = Vec::<(String, Vec<(usize, DefinitionOccurrence)>)>::new();
    let mut term_indices = HashMap::<&str, usize>::new();
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
            let hit = paragraph.at(document, base + start, base + end);
            if let Some(&term_index) = term_indices.get(term) {
                terms[term_index].1.push((paragraph_index, hit));
            } else {
                term_indices.insert(term, terms.len());
                terms.push((term.to_owned(), vec![(paragraph_index, hit)]));
            }
        }
    }

    let patterns = terms
        .iter()
        .flat_map(|(term, _)| {
            [
                term.clone(),
                term.strip_suffix('s')
                    .map_or_else(|| format!("{term}s"), str::to_owned),
            ]
        })
        .collect::<Vec<_>>();

    let mut uses = vec![Vec::<(usize, usize, usize)>::new(); terms.len()];
    if !patterns.is_empty() {
        let matcher = AhoCorasick::new(&patterns).unwrap();
        let mut pattern_ends = vec![0; patterns.len()];
        for (paragraph_index, (_, text)) in slices.iter().enumerate() {
            pattern_ends.fill(0);
            for hit in matcher.find_overlapping_iter(text) {
                let pattern_index = hit.pattern().as_usize();
                if hit.start() < pattern_ends[pattern_index] {
                    continue;
                }
                pattern_ends[pattern_index] = hit.end();
                if !bounded(text, hit.start(), hit.end()) {
                    continue;
                }
                let term_index = pattern_index / 2;
                if terms[term_index]
                    .1
                    .binary_search_by_key(&paragraph_index, |(defined, _)| *defined)
                    .is_err()
                {
                    uses[term_index].push((paragraph_index, hit.start(), hit.end()));
                }
            }
        }
    }

    let terms = terms
        .into_iter()
        .zip(uses)
        .map(|((term, definitions), mut uses)| {
            uses.sort_unstable();
            DefinedTerm {
                term,
                definitions: definitions.into_iter().map(|(_, hit)| hit).collect(),
                uses: uses
                    .into_iter()
                    .map(|(paragraph_index, start, end)| {
                        let (base, _) = slices[paragraph_index];
                        paragraphs[paragraph_index].at(document, base + start, base + end)
                    })
                    .collect(),
            }
        })
        .collect();
    DefinitionsResult { terms }
}
