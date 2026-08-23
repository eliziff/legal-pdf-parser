use crate::{derive_docx_numbering, javascript_whitespace, DocxNumberAnchor};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

const LABELS: [&str; 5] = ["Schedule", "Exhibit", "Appendix", "Annex", "Annexure"];
const JS_WS: &str = r"\x{0009}-\x{000D}\x{0020}\x{00A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}\x{FEFF}";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DocxCrossReferenceStatus {
    Resolved,
    SkippedExternal,
    MissingRomanArticle,
    MissingSibling { parent: String },
    MissingTopLevel,
    Abstained,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DocxCrossReference {
    pub paragraph_index: usize,
    pub subject: String,
    pub value: String,
    pub status: DocxCrossReferenceStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum DocxAttachmentReferenceStatus {
    Resolved,
    Missing { included: Vec<String> },
    AbstainedNoAnchor,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DocxAttachmentReference {
    pub paragraph_index: usize,
    pub label: String,
    pub id: String,
    pub subject: String,
    pub status: DocxAttachmentReferenceStatus,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct DocxStructureFacts {
    pub numbering: crate::DocxNumberingResult,
    pub cross_references: Vec<DocxCrossReference>,
    pub attachments: Vec<DocxAttachmentReference>,
}

fn reference_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(&format!(
        r"(?-u:\b)(Section|Sections|Clause|Clauses|Article|Articles|Paragraph|Paragraphs)[{JS_WS}]+(\d{{1,3}}(?:\.\d{{1,3}})*|[IVXLCDM]+)(?-u:\b)"
    )).unwrap())
}

fn external_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| {
        Regex::new(&format!(
            r"(?i)^[{JS_WS}]*(?:\([a-z0-9]{{1,4}}\)[{JS_WS}]*)?(?:of|to|under)[{JS_WS}]+((?-u:\w+))"
        ))
        .unwrap()
    })
}

fn is_external(following: &str) -> bool {
    external_pattern()
        .captures(following)
        .and_then(|captures| captures.get(1))
        .is_some_and(|owner| !owner.as_str().eq_ignore_ascii_case("this"))
}

fn cross_references(
    paragraphs: &[String],
    numbers: &[DocxNumberAnchor],
    romans: &[DocxNumberAnchor],
) -> Vec<DocxCrossReference> {
    let anchors = numbers
        .iter()
        .map(|anchor| anchor.number.as_str())
        .collect::<HashSet<_>>();
    let roman_anchors = romans
        .iter()
        .map(|anchor| anchor.number.as_str())
        .collect::<HashSet<_>>();
    let child_depths = anchors
        .iter()
        .filter_map(|anchor| {
            let (parent, _) = anchor.rsplit_once('.')?;
            Some((parent, anchor.matches('.').count() + 1))
        })
        .collect::<HashSet<_>>();
    let top_levels = anchors
        .iter()
        .filter(|anchor| !anchor.contains('.'))
        .count();
    let mut facts = Vec::new();
    for (paragraph_index, paragraph) in paragraphs.iter().enumerate() {
        for captures in reference_pattern().captures_iter(paragraph) {
            let whole = captures.get(0).unwrap();
            let value = captures.get(2).unwrap().as_str();
            let roman = value.bytes().all(|byte| b"IVXLCDM".contains(&byte));
            let status = if is_external(&paragraph[whole.end()..]) {
                DocxCrossReferenceStatus::SkippedExternal
            } else if roman {
                if roman_anchors.is_empty() {
                    DocxCrossReferenceStatus::Abstained
                } else if roman_anchors.contains(value) {
                    DocxCrossReferenceStatus::Resolved
                } else {
                    DocxCrossReferenceStatus::MissingRomanArticle
                }
            } else if anchors.contains(value)
                || anchors
                    .iter()
                    .any(|anchor| anchor.starts_with(&format!("{value}.")))
            {
                DocxCrossReferenceStatus::Resolved
            } else if let Some((parent, _)) = value.rsplit_once('.') {
                if child_depths.contains(&(parent, value.matches('.').count() + 1)) {
                    DocxCrossReferenceStatus::MissingSibling {
                        parent: parent.into(),
                    }
                } else {
                    DocxCrossReferenceStatus::Abstained
                }
            } else if top_levels >= 3 {
                DocxCrossReferenceStatus::MissingTopLevel
            } else {
                DocxCrossReferenceStatus::Abstained
            };
            facts.push(DocxCrossReference {
                paragraph_index,
                subject: whole.as_str().into(),
                value: value.into(),
                status,
            });
        }
    }
    facts
}

fn attachment_pattern(label: &str, anchor: bool) -> Regex {
    let prefix = if anchor { "(?i)^" } else { r"(?-u:\b)" };
    let plural = if anchor { "" } else { "s?" };
    Regex::new(&format!(
        r"{prefix}{label}{plural}[{JS_WS}]+(\d{{1,3}}|[A-Z]{{1,3}})(?-u:\b)"
    ))
    .unwrap()
}

fn heading_like(text: &str, end: usize) -> bool {
    text.encode_utf16().count() <= 80
        || text[..end].to_uppercase() == text[..end]
        || text[end..]
            .trim_start_matches(javascript_whitespace)
            .chars()
            .next()
            .is_some_and(|character| matches!(character, '-' | '–' | '—' | ':' | '.'))
}

fn attachments(paragraphs: &[String]) -> Vec<DocxAttachmentReference> {
    static PATTERNS: OnceLock<Vec<(&'static str, Regex, Regex)>> = OnceLock::new();
    let mut anchors = HashMap::<&str, HashSet<String>>::new();
    let mut found = Vec::<DocxAttachmentReference>::new();
    let patterns = PATTERNS.get_or_init(|| {
        LABELS
            .into_iter()
            .map(|label| {
                (
                    label,
                    attachment_pattern(label, true),
                    attachment_pattern(label, false),
                )
            })
            .collect()
    });
    for (paragraph_index, text) in paragraphs.iter().enumerate() {
        for (label, anchor_pattern, reference_pattern) in patterns {
            let anchor = anchor_pattern.captures(text);
            if let Some(anchor) =
                anchor.filter(|capture| heading_like(text, capture.get(0).unwrap().end()))
            {
                anchors
                    .entry(*label)
                    .or_default()
                    .insert(anchor[1].to_uppercase());
                continue;
            }
            for capture in reference_pattern.captures_iter(text) {
                let whole = capture.get(0).unwrap();
                if whole.start() == 0 || is_external(&text[whole.end()..]) {
                    continue;
                }
                found.push(DocxAttachmentReference {
                    paragraph_index,
                    label: (*label).into(),
                    id: capture[1].to_uppercase(),
                    subject: whole.as_str().into(),
                    status: DocxAttachmentReferenceStatus::AbstainedNoAnchor,
                });
            }
        }
    }
    let mut label_order = Vec::<String>::new();
    for reference in &found {
        if !label_order.contains(&reference.label) {
            label_order.push(reference.label.clone());
        }
    }
    found.sort_by_key(|reference| {
        label_order
            .iter()
            .position(|label| label == &reference.label)
    });
    for reference in &mut found {
        let Some(included) = anchors
            .get(reference.label.as_str())
            .filter(|set| !set.is_empty())
        else {
            continue;
        };
        reference.status = if included.contains(&reference.id) {
            DocxAttachmentReferenceStatus::Resolved
        } else {
            let mut included = included.iter().cloned().collect::<Vec<_>>();
            included.sort();
            DocxAttachmentReferenceStatus::Missing { included }
        };
    }
    found
}

pub fn derive_docx_lint_facts(paragraphs: &[String]) -> DocxStructureFacts {
    let numbering = derive_docx_numbering(paragraphs);
    DocxStructureFacts {
        cross_references: cross_references(
            paragraphs,
            &numbering.number_anchors,
            &numbering.roman_article_anchors,
        ),
        attachments: attachments(paragraphs),
        numbering,
    }
}

#[cfg(feature = "structure-inference")]
pub fn analyze_docx(
    document_id: String,
    paragraphs: Vec<String>,
    table_cells: &[crate::AuthoritativeTableCell],
) -> Result<crate::DocumentStructure, crate::EngineError> {
    let text = paragraphs.join("\n");
    let mut structure = crate::analyze_instrument(&text, document_id, table_cells, true)?;
    structure.provider = "docx".to_owned();
    structure.docx = Some(derive_docx_lint_facts(&paragraphs));
    Ok(structure)
}
