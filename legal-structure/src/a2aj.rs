use crate::{
    compose_trusted, Coverage, CoverageState, DetectionProfile, DocumentInput, EngineError,
    EvidenceKind, NativeClaim, Origin, ParagraphBreak, Scope, ScopeKind, SourceDoc, SourceDocBlock,
    SourceDocKind, SourceDocOrigin, SourceDocType, EVIDENCE_SCHEMA,
};
use regex::Regex;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum A2ajSourceKind {
    Cases,
    Laws,
}

/// A section map is ordered provider data, not a generic JSON object.
pub type A2ajSectionMap = Vec<(String, String)>;

#[derive(Deserialize)]
pub struct A2ajInput {
    pub citation: String,
    pub source_kind: A2ajSourceKind,
    pub text: String,
    pub id: Option<String>,
    pub url: Option<String>,
    pub dataset: Option<String>,
    pub name: Option<String>,
    pub alternate_citation: Option<String>,
    pub section_map: Option<A2ajSectionMap>,
    pub excerpt_of: Option<String>,
}

impl A2ajInput {
    pub fn new(
        citation: impl Into<String>,
        source_kind: A2ajSourceKind,
        text: impl Into<String>,
    ) -> Self {
        Self {
            citation: citation.into(),
            source_kind,
            text: text.into(),
            id: None,
            url: None,
            dataset: None,
            name: None,
            alternate_citation: None,
            section_map: None,
            excerpt_of: None,
        }
    }
}

fn provision_label(value: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^(?:\d{1,8}[A-Za-z]{0,3}(?:[.-]\d{1,8}[A-Za-z]{0,3}){0,3}|[A-Za-z]{1,3}(?:[.-][0-9A-Za-z]{1,8}){1,3})$").unwrap())
        .is_match(value)
}

fn compare_labels(left: &str, right: &str, fraction: bool) -> Ordering {
    crate::recovery::compare_labels(left, right, fraction)
}

fn dotted_order(source: &[(usize, &str, &str)]) -> Option<bool> {
    let labels = source
        .iter()
        .map(|(_, label, _)| label)
        .filter(|label| label.contains('.') && !label.contains('-'))
        .collect::<Vec<_>>();
    let inversions = |order| {
        labels
            .windows(2)
            .filter(|pair| compare_labels(pair[0], pair[1], order).is_gt())
            .count()
    };
    let (component, fraction) = (inversions(false), inversions(true));
    if component != fraction {
        return Some(fraction < component);
    }
    (!labels.windows(2).any(|pair| {
        compare_labels(pair[0], pair[1], false) != compare_labels(pair[0], pair[1], true)
    }))
    .then_some(false)
}

fn validate_section_map(map: &A2ajSectionMap) -> Result<(), EngineError> {
    let mut seen = std::collections::HashSet::new();
    if map.iter().any(|(key, _)| !seen.insert(key)) {
        return Err(EngineError::source("duplicate A2AJ section-map key"));
    }
    Ok(())
}

fn object_entries(map: &A2ajSectionMap) -> Result<Vec<(usize, &str, &str)>, EngineError> {
    validate_section_map(map)?;
    let mut entries = map
        .iter()
        .enumerate()
        .map(|(index, (key, value))| (index, key.as_str(), value.as_str()))
        .collect::<Vec<_>>();
    entries.sort_by_key(|(index, key, _)| {
        let integer = key
            .parse::<u32>()
            .ok()
            .filter(|number| number.to_string() == *key && *number < u32::MAX);
        (integer.is_none(), integer.unwrap_or_default(), *index)
    });
    Ok(entries)
}

fn ordered_sections(map: &A2ajSectionMap) -> Result<Vec<(&str, &str)>, EngineError> {
    let mut entries = object_entries(map)?;
    let order = dotted_order(&entries);
    entries.sort_by(|left, right| {
        let (a, b) = (left.1.trim(), right.1.trim());
        let preamble =
            |value: &str| matches!(value.to_lowercase().as_str(), "preamble" | "préambule");
        preamble(b)
            .cmp(&preamble(a))
            .then_with(|| provision_label(b).cmp(&provision_label(a)))
            .then_with(|| {
                if !provision_label(a) {
                    Ordering::Equal
                } else if let Some(order) = order {
                    compare_labels(a, b, order)
                } else {
                    let component = compare_labels(a, b, false);
                    let fraction = compare_labels(a, b, true);
                    if component == fraction {
                        component
                    } else {
                        left.0.cmp(&right.0)
                    }
                }
            })
    });
    Ok(entries
        .into_iter()
        .map(|(_, label, value)| (label, value))
        .collect())
}

#[cfg(test)]
fn utf16_at(text: &str, byte: usize) -> usize {
    text[..byte].encode_utf16().count()
}

fn provider_source(mut text: String, entries: &[(&str, &str)]) -> (String, Vec<SourceDocBlock>) {
    if !text.trim().is_empty() || entries.is_empty() {
        return (text, Vec::new());
    }
    text.clear();
    let mut blocks = Vec::new();
    let mut utf16 = 0;
    for (label, value) in entries.iter().filter(|(_, value)| {
        !value.trim().is_empty() && !value.trim().eq_ignore_ascii_case("[blank]")
    }) {
        if !text.is_empty() {
            text.push('\n');
            utf16 += 1;
        }
        let start = utf16;
        text.push_str(value);
        utf16 += value.encode_utf16().count();
        blocks.push(SourceDocBlock::new(
            SourceDocKind::Section,
            format!("sec{}", label.trim()),
            start,
            utf16,
            SourceDocOrigin::Native,
        ));
    }
    (text, blocks)
}

fn provider_claims(text: &str, map: &A2ajSectionMap) -> Vec<SourceDocBlock> {
    static PRINTED: OnceLock<Regex> = OnceLock::new();
    let utf16 = |byte| text[..byte].encode_utf16().count();
    map.iter()
        .filter_map(|(raw_label, value)| {
            let label = raw_label.trim();
            if label.is_empty()
                || value.trim().is_empty()
                || value.trim().eq_ignore_ascii_case("[blank]")
            {
                return None;
            }
            let mut matches = text.match_indices(value).map(|(start, _)| start);
            let start = matches.next()?;
            if matches.next().is_some() {
                return None;
            }
            let line_start = text[..start].rfind('\n').map_or(0, |at| at + 1);
            let prefix = text[line_start..start].trim();
            let printed = PRINTED
                .get_or_init(|| Regex::new(r"^([^\s.)]+)[.)]?$").unwrap())
                .captures(prefix)
                .map(|capture| capture[1].to_owned());
            if printed.is_some() {
                return None;
            }
            let end = start + value.len();
            Some(SourceDocBlock::new(
                SourceDocKind::Section,
                format!("sec{label}"),
                utf16(start),
                utf16(end),
                SourceDocOrigin::Native,
            ))
        })
        .collect()
}

fn evidence(
    input: &A2ajInput,
    text: String,
    blocks: Vec<SourceDocBlock>,
) -> Result<DocumentInput, EngineError> {
    const ORIGIN: &str = "provider-adapter";
    let offsets = crate::source_doc::utf16_offsets(&text);
    let scalar = |offset: usize| {
        offsets
            .binary_search(&offset)
            .map_err(|_| EngineError::source("provider UTF-16 range splits a Unicode scalar"))
    };
    let mut originals = HashMap::new();
    let claims = blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| {
            let id = format!("native-{:06}", index + 1);
            let claim = NativeClaim {
                id: id.clone(),
                kind: EvidenceKind::Section,
                label: Some(block.label.clone()),
                aliases: block.aliases.clone(),
                range: crate::ScalarRange {
                    start: scalar(block.start)?,
                    end: scalar(block.end)?,
                },
                provider_order: index,
                origin_id: ORIGIN.to_owned(),
                parent_label: block.parent_label.clone(),
                anchor: block.anchor.clone(),
            };
            originals.insert(id, block);
            Ok(claim)
        })
        .collect::<Result<Vec<_>, EngineError>>()?;
    let scalar_end = offsets.len() - 1;
    let coverage = [
        EvidenceKind::Paragraph,
        EvidenceKind::Prose,
        EvidenceKind::Page,
        EvidenceKind::Section,
        EvidenceKind::Heading,
        EvidenceKind::Footnote,
        EvidenceKind::Endnote,
    ]
    .into_iter()
    .map(|kind| Coverage {
        kind,
        range: crate::ScalarRange {
            start: 0,
            end: scalar_end,
        },
        state: if kind == EvidenceKind::Section && !claims.is_empty() {
            CoverageState::Augment
        } else {
            CoverageState::Absent
        },
        reason: "shared-engine recovery lane".to_owned(),
        origin_id: (kind == EvidenceKind::Section && !claims.is_empty()).then(|| ORIGIN.to_owned()),
    })
    .collect();
    let source_kind = input.source_kind;
    let profile = if source_kind == A2ajSourceKind::Cases {
        DetectionProfile::CaseRootedComplete
    } else {
        DetectionProfile::Legislation
    };
    let report_start_page = report_start(input);
    let require_report_start = source_kind == A2ajSourceKind::Cases
        && input
            .dataset
            .as_deref()
            .is_some_and(|value| value.eq_ignore_ascii_case("SCC"));
    let allow_hyphenated_sections = source_kind == A2ajSourceKind::Laws
        && input.name.as_deref().is_some_and(|value| {
            static RE: OnceLock<Regex> = OnceLock::new();
            RE.get_or_init(|| {
                Regex::new(r"(?iu)\b(?:rules?|regulations?|r[eè]glements?)\b").unwrap()
            })
            .is_match(value)
        });
    let text_sha256 = format!("{:x}", Sha256::digest(text.as_bytes()));
    Ok(DocumentInput {
        schema_version: EVIDENCE_SCHEMA.to_owned(),
        document_id: input.id.clone().unwrap_or_else(|| input.citation.clone()),
        provider: "a2aj".to_owned(),
        url: input.url.clone(),
        doc_type: Some(if source_kind == A2ajSourceKind::Cases {
            SourceDocType::Cases
        } else {
            SourceDocType::Laws
        }),
        provider_revision: "a2aj-adapter-v1".to_owned(),
        profile,
        report_start_page,
        require_report_start,
        allow_hyphenated_sections,
        text,
        text_sha256,
        source_sha256: None,
        offset_unit: "unicode-scalar".to_owned(),
        scope: Scope {
            kind: if input.excerpt_of.is_some() {
                ScopeKind::Excerpt
            } else {
                ScopeKind::Complete
            },
            excerpt_of: input.excerpt_of.clone(),
        },
        origins: vec![Origin {
            id: ORIGIN.to_owned(),
            producer: "a2aj".to_owned(),
            representation: "provider-rendered-text".to_owned(),
            revision: "a2aj-adapter-v1".to_owned(),
            authority: "provider-native-claims".to_owned(),
        }],
        units: Vec::new(),
        native_claims: claims,
        coverage,
        exclusions: Vec::new(),
        paragraph_breaks: Vec::<ParagraphBreak>::new(),
        original_claims: originals,
    })
}

fn report_start(input: &A2ajInput) -> Option<u32> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE
        .get_or_init(|| Regex::new(r"(?iu)\b(?:S\.?C\.?R\.?|R\.?C\.?S\.?)\s+(\d{1,4})\b").unwrap());
    std::iter::once(input.citation.as_str())
        .chain(input.alternate_citation.as_deref())
        .find_map(|value| {
            re.captures(value)
                .and_then(|capture| capture[1].parse().ok())
        })
}

fn words(text: &str) -> Vec<(String, usize, usize, usize, usize)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let mut previous = 0;
    let mut utf16 = 0;
    RE.get_or_init(|| Regex::new(r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*").unwrap())
        .find_iter(text)
        .map(|item| {
            utf16 += text[previous..item.start()].encode_utf16().count();
            let start = utf16;
            utf16 += item.as_str().encode_utf16().count();
            previous = item.end();
            (
                item.as_str().to_lowercase(),
                start,
                utf16,
                item.start(),
                item.end(),
            )
        })
        .collect()
}

fn apply_provider_section_evidence(
    text: &str,
    blocks: &mut Vec<SourceDocBlock>,
    map: &A2ajSectionMap,
) {
    let tokens = words(text);
    let mut postings = HashMap::<&str, Vec<usize>>::new();
    for (index, (word, ..)) in tokens.iter().enumerate() {
        postings.entry(word).or_default().push(index);
    }
    let mut top_sections = HashMap::<String, Vec<usize>>::new();
    for (index, block) in blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.kind == SourceDocKind::Section && block.parent_label.is_none())
    {
        for label in std::iter::once(&block.label).chain(&block.aliases) {
            let candidates = top_sections.entry(label.to_lowercase()).or_default();
            if candidates.last() != Some(&index) {
                candidates.push(index);
            }
        }
    }
    let mut counts = HashMap::new();
    for (label, _) in map {
        *counts.entry(label.trim().to_lowercase()).or_insert(0) += 1;
    }
    for (label, provider_text) in map {
        let label = label.trim();
        if label.is_empty()
            || counts.get(&label.to_lowercase()) != Some(&1)
            || provider_text.trim().is_empty()
            || provider_text.trim().eq_ignore_ascii_case("[blank]")
        {
            continue;
        }
        let phrase = words(provider_text);
        if phrase.is_empty() {
            continue;
        }
        let Some((anchor_offset, anchor_word)) = phrase
            .iter()
            .enumerate()
            .min_by_key(|(_, word)| postings.get(word.0.as_str()).map_or(0, Vec::len))
        else {
            continue;
        };
        let mut spans = Vec::new();
        for &position in postings.get(anchor_word.0.as_str()).into_iter().flatten() {
            let Some(start) = position.checked_sub(anchor_offset) else {
                continue;
            };
            if start + phrase.len() <= tokens.len()
                && tokens[start..start + phrase.len()]
                    .iter()
                    .map(|token| &token.0)
                    .eq(phrase.iter().map(|token| &token.0))
            {
                spans.push((start, tokens[start].1, tokens[start + phrase.len() - 1].2));
                if spans.len() == 2 {
                    break;
                }
            }
        }
        if spans.len() != 1 {
            continue;
        }
        let first_token = spans[0].0;
        let last_token = first_token + phrase.len() - 1;
        let body = tokens[first_token]
            .3
            .checked_sub(phrase[0].3)
            .and_then(|body_start| {
                body_start
                    .checked_add(provider_text.len())
                    .filter(|&body_end| text.get(body_start..body_end) == Some(provider_text))
                    .map(|body_end| {
                        let body_utf16_start = tokens[first_token].1 - phrase[0].1;
                        let body_utf16_end = tokens[last_token].2
                            + provider_text[phrase[phrase.len() - 1].4..]
                                .encode_utf16()
                                .count();
                        (body_start, body_end, body_utf16_start, body_utf16_end)
                    })
            });
        let provider_label = format!("sec{label}");
        let key = provider_label.to_lowercase();
        let candidates = top_sections.get(&key).map(Vec::as_slice).unwrap_or(&[]);
        if candidates.len() == 1 {
            let index = candidates[0];
            if spans[0].1 < blocks[index].start || spans[0].2 > blocks[index].end {
                continue;
            }
            blocks[index].origin = SourceDocOrigin::Native;
        } else if candidates.is_empty() {
            let Some((body_start, _, body_utf16_start, body_utf16_end)) = body else {
                continue;
            };
            let line_start = text[..body_start].rfind('\n').map_or(0, |at| at + 1);
            let prefix = &text[line_start..body_start];
            let lead = prefix.len() - prefix.trim_start_matches([' ', '\t']).len();
            let printed = prefix[lead..].trim_end_matches([' ', '\t']);
            let printed = std::iter::once(printed)
                .chain(
                    printed
                        .strip_suffix(['.', ')', ':', '-', '–', '—'])
                        .map(str::trim_end),
                )
                .find(|printed| printed.eq_ignore_ascii_case(label));
            let start = printed.map_or(body_utf16_start, |_| {
                body_utf16_start - text[line_start + lead..body_start].encode_utf16().count()
            });
            blocks.push(SourceDocBlock::new(
                SourceDocKind::Section,
                provider_label,
                start,
                body_utf16_end,
                SourceDocOrigin::Native,
            ));
        }
    }
    let mut seen = std::collections::HashSet::new();
    blocks.retain(|block| seen.insert((block.label.clone(), block.start, block.end)));
    blocks.sort_by_key(|block| (block.start, block.parent_label.is_some()));
}

pub fn a2aj_source_doc(mut input: A2ajInput) -> Result<SourceDoc, EngineError> {
    let has_text = !input.text.trim().is_empty();
    let ordered = match (&input.section_map, has_text) {
        (Some(map), true) => {
            validate_section_map(map)?;
            Vec::new()
        }
        (Some(map), false) => ordered_sections(map)?,
        (None, _) => Vec::new(),
    };
    let (text, mut blocks) = provider_source(std::mem::take(&mut input.text), &ordered);
    if has_text {
        if let Some(map) = &input.section_map {
            blocks = provider_claims(&text, map);
        }
    }
    let mut document = compose_trusted(evidence(&input, text, blocks)?)?;
    if input.source_kind == A2ajSourceKind::Laws && has_text {
        if let Some(map) = &input.section_map {
            apply_provider_section_evidence(&document.text, &mut document.blocks, map);
            document = crate::create_source_doc(
                document.provider,
                document.id,
                document.url,
                document.doc_type,
                document.text,
                document.blocks,
            );
        }
    }
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_rendering_and_provider_evidence_match_a2aj() {
        let mut mapped = A2ajInput::new("fixture", A2ajSourceKind::Laws, "");
        mapped.section_map = Some(
            ["1", "2", "4", "4.1", "4.2", "5", "Schedule 2", "Schedule 1"]
                .into_iter()
                .map(|label| (label.into(), format!("Provision {label}.")))
                .collect(),
        );
        let mapped = a2aj_source_doc(mapped).unwrap();
        assert_eq!(
            mapped
                .blocks
                .iter()
                .filter(|block| block.kind == SourceDocKind::Section)
                .map(|block| block.label.as_str())
                .collect::<Vec<_>>(),
            [
                "sec1",
                "sec2",
                "sec4",
                "sec4.1",
                "sec4.2",
                "sec5",
                "secSchedule 2",
                "secSchedule 1"
            ]
        );

        let text = "1 First full-text provision.\n2 Second full-text provision.\n3 Third full-text provision.";
        let mut promoted = A2ajInput::new("fixture", A2ajSourceKind::Laws, text);
        promoted.section_map = Some(vec![("2".into(), "Second full-text provision.".into())]);
        let promoted = a2aj_source_doc(promoted).unwrap();
        assert_eq!(
            promoted
                .blocks
                .iter()
                .filter(|block| block.kind == SourceDocKind::Section)
                .map(|block| (&*block.label, block.origin))
                .collect::<Vec<_>>(),
            [
                ("sec1", SourceDocOrigin::Heuristic),
                ("sec2", SourceDocOrigin::Native),
                ("sec3", SourceDocOrigin::Heuristic)
            ]
        );
        assert_eq!(
            promoted
                .blocks
                .iter()
                .find(|block| block.label == "sec2")
                .unwrap()
                .start,
            utf16_at(text, text.find("2 Second").unwrap())
        );

        let text = "Preamble.\n99 Provider-only provision.";
        let mut missing = A2ajInput::new("fixture", A2ajSourceKind::Laws, text);
        missing.section_map = Some(vec![("99".into(), "Provider-only provision.".into())]);
        let missing = a2aj_source_doc(missing).unwrap();
        let added = missing
            .blocks
            .iter()
            .find(|block| block.label == "sec99")
            .unwrap();
        assert_eq!(added.origin, SourceDocOrigin::Native);
        assert_eq!(
            added.start,
            utf16_at(text, text.find("99 Provider-only").unwrap())
        );

        let mut sole = A2ajInput::new("fixture", A2ajSourceKind::Laws, "1 Sole provision.");
        sole.section_map = Some(vec![("1".into(), "Sole provision.".into())]);
        let sole = a2aj_source_doc(sole).unwrap();
        assert_eq!(
            sole.blocks
                .iter()
                .filter(|block| block.kind == SourceDocKind::Section && block.parent_label.is_none())
                .map(|block| (&*block.label, block.start, block.origin))
                .collect::<Vec<_>>(),
            [("sec1", 0, SourceDocOrigin::Native)]
        );

        let mut printed = A2ajInput::new("fixture", A2ajSourceKind::Laws, "");
        printed.section_map = Some(vec![(
            "34".into(),
            "34(1) Parent provision.\n(a) Child paragraph.".into(),
        )]);
        let printed = a2aj_source_doc(printed).unwrap();
        let subsection = printed
            .blocks
            .iter()
            .find(|block| block.label == "sec34(1)")
            .unwrap();
        assert_eq!(subsection.start, 4);
        assert_eq!(
            printed
                .blocks
                .iter()
                .find(|block| block.label == "sec34(1)(a)")
                .unwrap()
                .parent_label
                .as_deref(),
            Some("sec34")
        );

        let text = "1 (1) Parent provision.\n(a) Child.\n\n### Next\n2 Next provision.";
        let mut bounded = A2ajInput::new("fixture", A2ajSourceKind::Laws, text);
        bounded.section_map = Some(vec![(
            "1".into(),
            "(1) Parent provision.\n(a) Child.".into(),
        )]);
        let bounded = a2aj_source_doc(bounded).unwrap();
        let child = bounded
            .blocks
            .iter()
            .find(|block| block.label == "sec1(1)(a)")
            .unwrap();
        assert_eq!(&bounded.text[child.start..child.start + 3], "(a)");
        assert_eq!(
            child.end,
            bounded
                .blocks
                .iter()
                .find(|block| block.label == "sec2")
                .unwrap()
                .start
        );
    }
}
