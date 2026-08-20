use crate::source_doc::utf16_offsets;
use crate::{
    compose, Coverage, CoverageState, DetectionProfile, DocumentInput, EngineError, EvidenceKind,
    NativeClaim, Origin, ScalarRange, Scope, ScopeKind, SourceDoc, SourceDocBlock, SourceDocKind,
    SourceDocOrigin, SourceDocType, EVIDENCE_SCHEMA,
};
use regex::Regex;
use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::OnceLock;

const ORIGIN: &str = "provider-adapter";

#[derive(Clone, Copy, Eq, PartialEq)]
pub enum A2ajSourceKind {
    Cases,
    Laws,
}

/// A section map is ordered provider data, not a generic JSON object.
pub type A2ajSectionMap = Vec<(String, String)>;

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

fn object_entries(map: &A2ajSectionMap) -> Result<Vec<(usize, &str, &str)>, EngineError> {
    let mut seen = std::collections::HashSet::new();
    if map.iter().any(|(key, _)| !seen.insert(key)) {
        return Err(EngineError::source("duplicate A2AJ section-map key"));
    }
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
    let source = object_entries(map)?;
    let order = dotted_order(&source);
    let mut entries = source.clone();
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

struct SectionObject<'a>(&'a [(usize, &'a str, &'a str)]);
impl Serialize for SectionObject<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (_, key, value) in self.0 {
            map.serialize_entry(key, value)?;
        }
        map.end()
    }
}

fn hash(value: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(value))
}

fn utf16_at(text: &str, byte: usize) -> usize {
    text[..byte].encode_utf16().count()
}

fn provider_source(input: &A2ajInput, entries: &[(&str, &str)]) -> (String, Vec<SourceDocBlock>) {
    if input.text.trim().is_empty() && !entries.is_empty() {
        let mut text = String::new();
        let mut blocks = Vec::new();
        for (label, value) in entries.iter().filter(|(_, value)| {
            !value.trim().is_empty() && !value.trim().eq_ignore_ascii_case("[blank]")
        }) {
            if !text.is_empty() {
                text.push('\n');
            }
            let start = text.encode_utf16().count();
            text.push_str(value);
            blocks.push(SourceDocBlock::new(
                SourceDocKind::Section,
                format!("sec{}", label.trim()),
                start,
                text.encode_utf16().count(),
                SourceDocOrigin::Native,
            ));
            if provision_label(label.trim()) {
                blocks.extend(crate::recovery::source_doc_children(
                    label.trim(),
                    value,
                    start,
                ));
            }
        }
        return (text, blocks);
    }
    let mut blocks = Vec::new();
    for (label, value) in entries {
        let label = label.trim();
        if label.is_empty()
            || value.trim().is_empty()
            || value.trim().eq_ignore_ascii_case("[blank]")
        {
            continue;
        }
        let Some(start) = input.text.find(value) else {
            continue;
        };
        let after_first = start + input.text[start..].chars().next().unwrap().len_utf8();
        if input.text[after_first..].contains(value) {
            continue;
        }
        let line_start = input.text[..start].rfind('\n').map_or(0, |at| at + 1);
        let prefix = &input.text[line_start..start];
        static PRINTED: OnceLock<Regex> = OnceLock::new();
        let printed = PRINTED
            .get_or_init(|| Regex::new(r"^([^\s.)]+)[.)]?$").unwrap())
            .captures(prefix.trim())
            .map(|capture| capture[1].to_owned());
        if printed
            .as_deref()
            .is_some_and(|mark| provision_label(mark) && !mark.eq_ignore_ascii_case(label))
            || printed.is_some()
        {
            continue;
        }
        blocks.push(SourceDocBlock::new(
            SourceDocKind::Section,
            format!("sec{label}"),
            utf16_at(&input.text, start),
            utf16_at(&input.text, start + value.len()),
            SourceDocOrigin::Native,
        ));
    }
    (input.text.clone(), blocks)
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

fn document_input(
    input: &A2ajInput,
    text: String,
    blocks: Vec<SourceDocBlock>,
    map_json: &str,
    source_sha: String,
) -> DocumentInput {
    let scalar_end = text.chars().count();
    let utf16 = utf16_offsets(&text);
    let native_claims = blocks
        .iter()
        .enumerate()
        .map(|(index, block)| NativeClaim {
            id: format!("native-{:06}", index + 1),
            kind: EvidenceKind::Section,
            label: Some(block.label.clone()),
            aliases: vec![],
            range: ScalarRange {
                start: utf16.binary_search(&block.start).unwrap(),
                end: utf16.binary_search(&block.end).unwrap(),
            },
            provider_order: index,
            origin_id: ORIGIN.into(),
            parent_label: None,
            anchor: None,
        })
        .collect::<Vec<_>>();
    let original_claims = blocks
        .into_iter()
        .enumerate()
        .map(|(index, block)| (format!("native-{:06}", index + 1), block))
        .collect();
    let section_native = !native_claims.is_empty();
    let kinds = [
        EvidenceKind::Paragraph,
        EvidenceKind::Prose,
        EvidenceKind::Page,
        EvidenceKind::Section,
        EvidenceKind::Heading,
        EvidenceKind::Footnote,
        EvidenceKind::Endnote,
    ];
    let representation = hash(if input.text.is_empty() {
        map_json.as_bytes()
    } else {
        input.text.as_bytes()
    });
    DocumentInput {
        schema_version: EVIDENCE_SCHEMA.into(),
        document_id: input.id.clone().unwrap_or_else(|| input.citation.clone()),
        provider: "a2aj".into(),
        url: input.url.clone(),
        doc_type: Some(if input.source_kind == A2ajSourceKind::Cases {
            SourceDocType::Cases
        } else {
            SourceDocType::Laws
        }),
        provider_revision: "a2aj-adapter-v1".into(),
        profile: if input.source_kind == A2ajSourceKind::Cases {
            DetectionProfile::CaseRootedComplete
        } else {
            DetectionProfile::Legislation
        },
        report_start_page: report_start(input),
        require_report_start: input.source_kind == A2ajSourceKind::Cases
            && input
                .dataset
                .as_deref()
                .is_some_and(|value| value.eq_ignore_ascii_case("SCC")),
        allow_hyphenated_sections: input.source_kind == A2ajSourceKind::Laws
            && input.name.as_deref().is_some_and(|value| {
                static RE: OnceLock<Regex> = OnceLock::new();
                RE.get_or_init(|| {
                    Regex::new(r"(?iu)\b(?:rules?|regulations?|r[eè]glements?)\b").unwrap()
                })
                .is_match(value)
            }),
        text_sha256: hash(text.as_bytes()),
        text,
        source_sha256: Some(source_sha),
        offset_unit: "unicode-scalar".into(),
        scope: Scope {
            kind: if input.excerpt_of.is_some() {
                ScopeKind::Excerpt
            } else {
                ScopeKind::Complete
            },
            excerpt_of: input.excerpt_of.clone(),
        },
        origins: vec![Origin {
            id: ORIGIN.into(),
            producer: "a2aj".into(),
            representation: "provider-rendered-text".into(),
            revision: representation,
            authority: "provider-native-claims".into(),
        }],
        units: vec![],
        native_claims,
        coverage: kinds
            .into_iter()
            .map(|kind| {
                let augment = kind == EvidenceKind::Section && section_native;
                Coverage {
                    kind,
                    range: ScalarRange {
                        start: 0,
                        end: scalar_end,
                    },
                    state: if augment {
                        CoverageState::Augment
                    } else {
                        CoverageState::Absent
                    },
                    reason: "shared-engine recovery lane".into(),
                    origin_id: augment.then(|| ORIGIN.into()),
                }
            })
            .collect(),
        exclusions: vec![],
        paragraph_breaks: vec![],
        original_claims,
    }
}

fn words(text: &str) -> Vec<(String, usize, usize)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"[\p{L}\p{N}]+(?:['’][\p{L}\p{N}]+)*").unwrap())
        .find_iter(text)
        .map(|item| {
            (
                item.as_str().to_lowercase(),
                utf16_at(text, item.start()),
                utf16_at(text, item.end()),
            )
        })
        .collect()
}

fn promote(text: &str, blocks: &mut Vec<SourceDocBlock>, map: &A2ajSectionMap) {
    let tokens = words(text);
    let mut postings = HashMap::<&str, Vec<usize>>::new();
    for (index, (word, _, _)) in tokens.iter().enumerate() {
        postings.entry(word).or_default().push(index);
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
        let phrase = words(provider_text)
            .into_iter()
            .map(|(word, _, _)| word)
            .collect::<Vec<_>>();
        if phrase.is_empty() {
            continue;
        }
        let Some(anchor) = phrase
            .iter()
            .enumerate()
            .min_by_key(|(_, word)| postings.get(word.as_str()).map_or(0, Vec::len))
        else {
            continue;
        };
        let mut spans = Vec::new();
        for &position in postings.get(anchor.1.as_str()).into_iter().flatten() {
            let Some(start) = position.checked_sub(anchor.0) else {
                continue;
            };
            if start + phrase.len() <= tokens.len()
                && tokens[start..start + phrase.len()]
                    .iter()
                    .map(|token| &token.0)
                    .eq(phrase.iter())
            {
                spans.push((tokens[start].1, tokens[start + phrase.len() - 1].2));
                if spans.len() == 2 {
                    break;
                }
            }
        }
        if spans.len() != 1 {
            continue;
        }
        let provider_label = format!("sec{label}");
        let key = provider_label.to_lowercase();
        let candidates = blocks
            .iter()
            .enumerate()
            .filter(|(_, block)| {
                block.kind == SourceDocKind::Section
                    && block.parent_label.is_none()
                    && std::iter::once(&block.label)
                        .chain(&block.aliases)
                        .any(|value| value.to_lowercase() == key)
            })
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let exact = text.match_indices(provider_text).collect::<Vec<_>>();
        let exact = (exact.len() == 1).then(|| utf16_at(text, exact[0].0));
        if candidates.len() == 1 {
            let index = candidates[0];
            if spans[0].0 < blocks[index].start || spans[0].1 > blocks[index].end {
                continue;
            }
            blocks[index].origin = SourceDocOrigin::Native;
            if let Some(start) = exact.filter(|_| blocks[index].label.eq_ignore_ascii_case(&key)) {
                let outer = (blocks[index].start, blocks[index].end);
                let descendant = format!("{key}(");
                blocks.retain(|block| {
                    !block.label.to_lowercase().starts_with(&descendant)
                        || block.start < outer.0
                        || block.end > outer.1
                });
                blocks.extend(crate::recovery::source_doc_children(
                    label,
                    provider_text,
                    start,
                ));
            }
        } else if candidates.is_empty() {
            let Some(start) = exact else { continue };
            let end = start + provider_text.encode_utf16().count();
            blocks.push(SourceDocBlock::new(
                SourceDocKind::Section,
                provider_label,
                start,
                end,
                SourceDocOrigin::Native,
            ));
            if provision_label(label) {
                blocks.extend(crate::recovery::source_doc_children(
                    label,
                    provider_text,
                    start,
                ));
            }
        }
    }
    blocks.sort_by_key(|block| (block.start, block.parent_label.is_some()));
}

pub fn a2aj_source_doc(input: A2ajInput) -> Result<SourceDoc, EngineError> {
    let entries = input
        .section_map
        .as_ref()
        .map(object_entries)
        .transpose()?
        .unwrap_or_default();
    let map_json = input
        .section_map
        .as_ref()
        .map(|_| serde_json::to_string(&SectionObject(&entries)).unwrap())
        .unwrap_or_else(|| "null".into());
    let source_sha = hash(format!(
        "[{},{}]",
        serde_json::to_string(&input.text).unwrap(),
        map_json
    ));
    let ordered = input
        .section_map
        .as_ref()
        .map(ordered_sections)
        .transpose()?
        .unwrap_or_default();
    let source_order = entries
        .iter()
        .map(|(_, label, value)| (*label, *value))
        .collect::<Vec<_>>();
    let provider_entries = if input.text.trim().is_empty() {
        &ordered
    } else {
        &source_order
    };
    let (text, blocks) = provider_source(&input, provider_entries);
    let mut doc = compose(document_input(&input, text, blocks, &map_json, source_sha))?;
    if input.source_kind == A2ajSourceKind::Laws && !input.text.trim().is_empty() {
        if let Some(map) = &input.section_map {
            promote(&doc.text, &mut doc.blocks, map);
            doc = crate::create_source_doc(
                doc.provider,
                doc.id,
                doc.url,
                doc.doc_type,
                doc.text,
                doc.blocks,
            );
        }
    }
    Ok(doc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_rendering_and_native_promotion_match_a2aj() {
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
        assert_eq!(&bounded.text[child.start..child.end], "(a) Child.");

        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let fixtures = [
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-case-scc-2026scc16-toc.json", "7cd3e33314102bb6c0c7d36d28e500239c9eca80117a644220e793702921b7ef"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-case-scc-1986scr103-dot.json", "fdbce1588eb521ce19ef618d3249260367eea341729d2dba1ed251d2a3ad73a7"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-case-scc-2014scc53-bracket.json", "27b4cfbfcd318107c5cc3e36a5630a0a6a643bcb40d2adca37f5ad196d5d307c"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-case-scc-2020scc45-bracket.json", "8df96490a7842d451e1a74d3f1ac633c01512225e25404219a8d2b9af6e30c20"),
            ("backend/experiments/source-structure-port-oracle/inputs/a2aj-citt-pr-2014-016a-endnotes.json", "8f9e459e17b8fe089467f1467cafe4efe1ceb907aaea10cccc0ce7548fc3a5d2"),
            ("backend/experiments/source-structure-port-oracle/inputs/a2aj-onca-2024-468-heading-join.json", "5fc036eada3b9be140ca65e1e8f025ab849b3cbf7dfd7ab32fc98211d8f53c91"),
            // The canonical ladder keeps dotted/restarted nodes, immediate parents, and
            // parent-covering spans instead of reproducing the old flat child projection.
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-laws-fed-criminalcode-s231.json", "04c603e4e05ea33208c88bc0567f65a05c8f130f509f3d0ee4042e3a0f11b411"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-laws-fed-criminalcode-sectionmap.json", "34a6c532cfea1273acdf2a181e6df9be44dad02e222ec1e41efc9da72c9d5750"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-regs-on-oreg267-03.json", "82166813931e2c30638623ef7f87bfb8d73d098cc4e216b40a8646801606634b"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-regs-fed-crc870-a01.json", "4f1f04471ef0e84c633241f29b9328cb26d13fbf240350373d4e17e08087cf33"),
            ("backend/src/lib/__tests__/fixtures/sourcedoc/a2aj-laws-ab-abc-benefits-s8.json", "bf8f3d6e7efddc6d15f9d0c2a3717e1746952a20ddfb660c768b3086957551fb"),
        ];
        let mut mismatches = Vec::new();
        for (path, expected) in fixtures {
            let fixture: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(root.join(path)).unwrap()).unwrap();
            let mut captured = A2ajInput::new(
                fixture["citation"].as_str().unwrap(),
                if fixture["docType"] == "laws" {
                    A2ajSourceKind::Laws
                } else {
                    A2ajSourceKind::Cases
                },
                fixture["text"].as_str().unwrap(),
            );
            captured.id = fixture["id"].as_str().map(str::to_owned);
            captured.dataset = fixture["dataset"].as_str().map(str::to_owned);
            captured.name = fixture["name"].as_str().map(str::to_owned);
            captured.url = fixture["url"].as_str().map(str::to_owned);
            captured.alternate_citation = fixture["alternateCitation"].as_str().map(str::to_owned);
            captured.section_map = fixture["sectionMap"].as_object().map(|map| {
                map.iter()
                    .map(|(label, text)| (label.clone(), text.as_str().unwrap().to_owned()))
                    .collect()
            });
            let doc = a2aj_source_doc(captured).unwrap();
            let actual = hash(serde_json::to_vec(&doc).unwrap());
            if actual != expected {
                mismatches.push(format!(
                    "{path}: {actual} != {expected}\n{}",
                    serde_json::to_string(
                        &doc.blocks
                            .iter()
                            .filter(|block| block.kind == SourceDocKind::Section)
                            .collect::<Vec<_>>()
                    )
                    .unwrap()
                ));
            }
        }
        assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    }
}
