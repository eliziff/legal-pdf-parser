use crate::{EvidenceKind, NativeClaim, NodeKind, StructureGraphV2};
use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Deserialize, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};

const MAX_MISSING: usize = 64;

#[derive(Clone, Copy, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDocKind {
    Paragraph,
    Page,
    Section,
    Footnote,
}

#[derive(Clone, Copy, Serialize)]
pub enum SourceDocProvider {
    #[serde(rename = "a2aj")]
    A2aj,
    #[serde(rename = "courtlistener")]
    CourtListener,
    #[serde(rename = "tna")]
    Tna,
    #[serde(rename = "govinfo")]
    GovInfo,
    #[serde(rename = "govuk-et")]
    GovUkEt,
    #[serde(rename = "journal")]
    Journal,
    #[serde(rename = "local-pdf")]
    LocalPdf,
}

impl SourceDocProvider {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::A2aj => "a2aj",
            Self::CourtListener => "courtlistener",
            Self::Tna => "tna",
            Self::GovInfo => "govinfo",
            Self::GovUkEt => "govuk-et",
            Self::Journal => "journal",
            Self::LocalPdf => "local-pdf",
        }
    }

    pub(crate) fn from_name(value: &str) -> Option<Self> {
        Some(match value {
            "a2aj" => Self::A2aj,
            "courtlistener" => Self::CourtListener,
            "tna" => Self::Tna,
            "govinfo" => Self::GovInfo,
            "govuk-et" => Self::GovUkEt,
            "journal" => Self::Journal,
            "local-pdf" => Self::LocalPdf,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceDocType {
    Cases,
    Laws,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceDocOrigin {
    Native,
    Heuristic,
}

#[derive(Clone, Copy, Default)]
pub(crate) enum BlockFieldOrder {
    #[default]
    Projected,
    Native,
    NativeWithAliases,
    AliasesBeforeOrigin,
    AliasesAnchorBeforeOrigin,
    EndLast,
}

#[derive(Clone, Deserialize)]
pub struct SourceDocBlock {
    pub kind: SourceDocKind,
    pub label: String,
    pub start: usize,
    pub end: usize,
    pub origin: SourceDocOrigin,
    pub anchor: Option<String>,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(rename = "parentLabel")]
    pub parent_label: Option<String>,
    #[serde(skip)]
    pub(crate) field_order: BlockFieldOrder,
}

impl SourceDocBlock {
    pub fn new(
        kind: SourceDocKind,
        label: impl Into<String>,
        start: usize,
        end: usize,
        origin: SourceDocOrigin,
    ) -> Self {
        Self {
            kind,
            label: label.into(),
            start,
            end,
            origin,
            anchor: None,
            aliases: Vec::new(),
            parent_label: None,
            field_order: BlockFieldOrder::Projected,
        }
    }

    pub(crate) fn with_field_order(mut self, field_order: BlockFieldOrder) -> Self {
        self.field_order = field_order;
        self
    }

    pub fn preserve_field_order(&mut self, fields: &[String]) {
        let position = |name: &str| fields.iter().position(|field| field == name);
        self.field_order = match (
            position("end"),
            position("origin"),
            position("aliases"),
            position("anchor"),
        ) {
            (Some(_), Some(origin), Some(aliases), Some(anchor))
                if anchor < aliases && aliases < origin =>
            {
                BlockFieldOrder::Native
            }
            (Some(_), Some(origin), Some(aliases), Some(anchor))
                if aliases < anchor && anchor < origin =>
            {
                BlockFieldOrder::AliasesAnchorBeforeOrigin
            }
            (Some(end), Some(origin), Some(aliases), _) if aliases < origin && end < aliases => {
                BlockFieldOrder::AliasesBeforeOrigin
            }
            (Some(end), Some(origin), _, _) if origin < end => BlockFieldOrder::EndLast,
            _ => BlockFieldOrder::Projected,
        };
    }

    fn fields(&self) -> usize {
        5 + usize::from(self.anchor.is_some())
            + usize::from(!self.aliases.is_empty())
            + usize::from(self.parent_label.is_some())
    }
}

impl Serialize for SourceDocBlock {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut row = serializer.serialize_struct("SourceDocBlock", self.fields())?;
        row.serialize_field("kind", &self.kind)?;
        row.serialize_field("label", &self.label)?;
        row.serialize_field("start", &self.start)?;
        match self.field_order {
            BlockFieldOrder::Projected => {
                row.serialize_field("end", &self.end)?;
                row.serialize_field("origin", &self.origin)?;
                if !self.aliases.is_empty() {
                    row.serialize_field("aliases", &self.aliases)?;
                }
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
            }
            BlockFieldOrder::Native => {
                row.serialize_field("end", &self.end)?;
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
                if !self.aliases.is_empty() {
                    row.serialize_field("aliases", &self.aliases)?;
                }
                row.serialize_field("origin", &self.origin)?;
            }
            BlockFieldOrder::NativeWithAliases => {
                row.serialize_field("end", &self.end)?;
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
                row.serialize_field("aliases", &self.aliases)?;
                row.serialize_field("origin", &self.origin)?;
            }
            BlockFieldOrder::AliasesBeforeOrigin => {
                row.serialize_field("end", &self.end)?;
                if !self.aliases.is_empty() {
                    row.serialize_field("aliases", &self.aliases)?;
                }
                row.serialize_field("origin", &self.origin)?;
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
            }
            BlockFieldOrder::AliasesAnchorBeforeOrigin => {
                row.serialize_field("end", &self.end)?;
                if !self.aliases.is_empty() {
                    row.serialize_field("aliases", &self.aliases)?;
                }
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
                row.serialize_field("origin", &self.origin)?;
            }
            BlockFieldOrder::EndLast => {
                if let Some(anchor) = &self.anchor {
                    row.serialize_field("anchor", anchor)?;
                }
                if !self.aliases.is_empty() {
                    row.serialize_field("aliases", &self.aliases)?;
                }
                row.serialize_field("origin", &self.origin)?;
                row.serialize_field("end", &self.end)?;
            }
        }
        if let Some(parent) = &self.parent_label {
            row.serialize_field("parentLabel", parent)?;
        }
        row.end()
    }
}

#[derive(Serialize)]
pub struct SourceDocRange {
    kind: SourceDocKind,
    count: usize,
    first: Option<String>,
    last: Option<String>,
    missing: Vec<String>,
    #[serde(rename = "missingTruncated")]
    missing_truncated: bool,
}

#[derive(Serialize)]
pub struct SourceDocRanges {
    paragraph: SourceDocRange,
    page: SourceDocRange,
    section: SourceDocRange,
    footnote: SourceDocRange,
}

#[derive(Default)]
pub struct SourceDocIndex(HashMap<String, usize>);

impl SourceDocIndex {
    pub fn get(&self, label: &str) -> Option<usize> {
        self.0.get(&label.to_lowercase()).copied()
    }

    pub fn entries(&self) -> Vec<(String, usize)> {
        self.0
            .iter()
            .map(|(label, position)| (label.clone(), *position))
            .collect()
    }
}

// JavaScript serializes its in-memory Map as `{}`. Keep the useful lookup map
// in Rust while preserving the existing public SourceDoc byte contract.
impl Serialize for SourceDocIndex {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_map(Some(0))?.end()
    }
}

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum SourceDocStatus {
    Usable,
    Unavailable,
}

#[derive(Serialize)]
pub struct SourceDoc {
    pub provider: Option<SourceDocProvider>,
    pub id: String,
    pub url: Option<String>,
    pub revision: String,
    #[serde(rename = "docType")]
    pub doc_type: Option<SourceDocType>,
    pub status: SourceDocStatus,
    pub text: String,
    pub blocks: Vec<SourceDocBlock>,
    pub index: SourceDocIndex,
    pub ranges: SourceDocRanges,
}

#[derive(Serialize)]
struct StoredSourceDoc<'a> {
    provider: Option<SourceDocProvider>,
    id: &'a str,
    url: &'a Option<String>,
    revision: &'a str,
    #[serde(rename = "docType")]
    doc_type: Option<SourceDocType>,
    status: &'a SourceDocStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<&'a str>,
    blocks: &'a [SourceDocBlock],
    index: &'a SourceDocIndex,
    ranges: &'a SourceDocRanges,
}

impl SourceDoc {
    fn stored(&self, include_text: bool) -> StoredSourceDoc<'_> {
        StoredSourceDoc {
            provider: self.provider,
            id: &self.id,
            url: &self.url,
            revision: &self.revision,
            doc_type: self.doc_type,
            status: &self.status,
            text: include_text.then_some(&self.text),
            blocks: &self.blocks,
            index: &self.index,
            ranges: &self.ranges,
        }
    }

    pub fn json_value(&self, include_text: bool) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self.stored(include_text))
    }

    pub fn json_bytes(&self, include_text: bool) -> serde_json::Result<Vec<u8>> {
        serde_json::to_vec(&self.stored(include_text))
    }
}

fn number(kind: SourceDocKind, label: &str) -> Option<usize> {
    let lower = label.to_lowercase();
    if kind == SourceDocKind::Section {
        let rest = lower.strip_prefix("sec")?;
        let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
        let valid = (1..=8).contains(&digits)
            && rest[digits..]
                .bytes()
                .next()
                .is_none_or(|byte| matches!(byte, b'.' | b'-' | b'('));
        return valid.then(|| rest[..digits].parse().ok()).flatten();
    }
    let value = ["page=", "page", "par", "fn"]
        .into_iter()
        .find_map(|prefix| lower.strip_prefix(prefix))?;
    ((1..=6).contains(&value.len()) && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn range(kind: SourceDocKind, all: &[SourceDocBlock]) -> SourceDocRange {
    let blocks = all
        .iter()
        .filter(|block| block.kind == kind)
        .collect::<Vec<_>>();
    let physical = blocks
        .iter()
        .map(|block| block.label.to_lowercase())
        .collect::<HashSet<_>>();
    let aliases = blocks
        .iter()
        .flat_map(|block| block.aliases.iter().map(|label| label.to_lowercase()))
        .filter(|label| !physical.contains(label))
        .collect::<HashSet<_>>();
    let count = blocks.len() + aliases.len();
    let spine = blocks
        .iter()
        .copied()
        .filter(|block| kind != SourceDocKind::Section || !block.label.contains('('))
        .collect::<Vec<_>>();
    if spine.is_empty() {
        return SourceDocRange {
            kind,
            count,
            first: None,
            last: None,
            missing: Vec::new(),
            missing_truncated: false,
        };
    }
    let mut seen = HashSet::new();
    let labels = spine
        .iter()
        .flat_map(|block| std::iter::once(&block.label).chain(block.aliases.iter()))
        .filter(|label| seen.insert((*label).clone()))
        .collect::<Vec<_>>();
    let numbered = labels
        .iter()
        .filter_map(|label| number(kind, label).map(|value| ((*label).clone(), value)))
        .collect::<Vec<_>>();
    if numbered.len() != labels.len() {
        return SourceDocRange {
            kind,
            count,
            first: Some(spine[0].label.clone()),
            last: Some(spine.last().unwrap().label.clone()),
            missing: Vec::new(),
            missing_truncated: false,
        };
    }
    let mut lowest = &numbered[0];
    let mut highest = &numbered[0];
    for entry in &numbered[1..] {
        if entry.1 < lowest.1 {
            lowest = entry;
        }
        if entry.1 > highest.1 {
            highest = entry;
        }
    }
    let present = numbered
        .iter()
        .map(|(_, value)| *value)
        .collect::<HashSet<_>>();
    let mut missing = Vec::new();
    let mut missing_truncated = false;
    for value in lowest.1 + 1..highest.1 {
        if present.contains(&value) {
            continue;
        }
        if missing.len() == MAX_MISSING {
            missing_truncated = true;
            break;
        }
        let prefix = match kind {
            SourceDocKind::Section => "sec",
            SourceDocKind::Paragraph => "par",
            SourceDocKind::Page => "page",
            SourceDocKind::Footnote => "fn",
        };
        missing.push(format!("{prefix}{value}"));
    }
    SourceDocRange {
        kind,
        count,
        first: Some(lowest.0.clone()),
        last: Some(highest.0.clone()),
        missing,
        missing_truncated,
    }
}

pub fn create_source_doc(
    provider: Option<SourceDocProvider>,
    id: String,
    url: Option<String>,
    doc_type: Option<SourceDocType>,
    text: String,
    blocks: Vec<SourceDocBlock>,
) -> SourceDoc {
    let revision = format!("{:x}", Sha256::digest(text.as_bytes()));
    create_source_doc_with_revision(provider, id, url, doc_type, text, revision, blocks)
}

fn create_source_doc_with_revision(
    provider: Option<SourceDocProvider>,
    id: String,
    url: Option<String>,
    doc_type: Option<SourceDocType>,
    text: String,
    revision: String,
    blocks: Vec<SourceDocBlock>,
) -> SourceDoc {
    let ranges = SourceDocRanges {
        paragraph: range(SourceDocKind::Paragraph, &blocks),
        page: range(SourceDocKind::Page, &blocks),
        section: range(SourceDocKind::Section, &blocks),
        footnote: range(SourceDocKind::Footnote, &blocks),
    };
    let mut index = HashMap::new();
    let mut duplicates = HashSet::new();
    for (position, block) in blocks.iter().enumerate() {
        let mut labels = HashSet::new();
        for label in std::iter::once(&block.label)
            .chain(block.aliases.iter())
            .chain(block.anchor.iter())
            .filter(|label| labels.insert((*label).clone()))
        {
            let key = label.to_lowercase();
            if index.insert(key.clone(), position).is_some() {
                duplicates.insert(key);
            }
        }
    }
    index.retain(|label, _| !duplicates.contains(label));
    SourceDoc {
        provider,
        id,
        url,
        revision,
        doc_type,
        status: if blocks.is_empty() {
            SourceDocStatus::Unavailable
        } else {
            SourceDocStatus::Usable
        },
        text,
        blocks,
        index: SourceDocIndex(index),
        ranges,
    }
}

#[derive(Clone, Copy)]
pub enum ProjectionOrder {
    Case,
    Legislation,
    Position,
    StablePosition,
    Native,
}

pub(crate) fn project_graph(
    provider: Option<SourceDocProvider>,
    id: String,
    url: Option<String>,
    doc_type: Option<SourceDocType>,
    text: String,
    revision: String,
    originals: &HashMap<String, SourceDocBlock>,
    graph: StructureGraphV2,
    order: ProjectionOrder,
) -> SourceDoc {
    let scalar_to_utf16 = utf16_offsets(&text);
    let labels = graph
        .nodes
        .iter()
        .filter_map(|node| {
            node.label
                .as_ref()
                .map(|label| (node.id.as_str(), label.as_str()))
        })
        .collect::<HashMap<_, _>>();
    let mut prose = 0;
    let mut blocks = graph
        .nodes
        .iter()
        .filter_map(|node| {
            if let Some(original) = originals.get(&node.id) {
                return Some(original.clone());
            }
            let kind = match node.kind {
                NodeKind::Paragraph | NodeKind::Prose => SourceDocKind::Paragraph,
                NodeKind::Page => SourceDocKind::Page,
                NodeKind::Section => SourceDocKind::Section,
                NodeKind::Footnote => SourceDocKind::Footnote,
                NodeKind::Heading
                | NodeKind::Endnote
                | NodeKind::List
                | NodeKind::ListItem
                | NodeKind::Navigation => return None,
            };
            let label = if node.kind == NodeKind::Prose {
                prose += 1;
                format!("par{prose}")
            } else {
                node.label.clone()?
            };
            let mut block = SourceDocBlock::new(
                kind,
                label,
                scalar_to_utf16[node.range.start],
                scalar_to_utf16[node.range.end],
                SourceDocOrigin::Heuristic,
            );
            if matches!(order, ProjectionOrder::StablePosition) && node.kind != NodeKind::Prose {
                block.field_order = BlockFieldOrder::EndLast;
            }
            block.aliases = node.aliases.clone().unwrap_or_default();
            block.anchor = node.anchor.clone();
            block.parent_label = node
                .parent_id
                .as_deref()
                .and_then(|parent| labels.get(parent).copied())
                .map(str::to_owned);
            Some(block)
        })
        .collect::<Vec<_>>();
    if matches!(order, ProjectionOrder::Legislation) {
        let public_label = |value: &str| {
            let mut result = String::with_capacity(value.len());
            let mut rest = value;
            while let Some(at) = rest.find('@') {
                result.push_str(&rest[..at]);
                let digits = rest[at + 1..]
                    .bytes()
                    .take_while(u8::is_ascii_digit)
                    .count();
                if digits == 0 {
                    result.push('@');
                    rest = &rest[at + 1..];
                } else {
                    rest = &rest[at + 1 + digits..];
                }
            }
            result.push_str(rest);
            result
        };
        for block in &mut blocks {
            block.label = public_label(&block.label);
            block.parent_label = block.parent_label.as_deref().map(public_label);
        }
        let parents = blocks
            .iter()
            .map(|block| (block.label.clone(), block.parent_label.clone()))
            .collect::<HashMap<_, _>>();
        for block in &mut blocks {
            let mut parent = block.parent_label.clone();
            let mut seen = HashSet::new();
            while let Some(label) = parent.clone() {
                if !seen.insert(label.clone()) {
                    break;
                }
                match parents.get(&label).cloned().flatten() {
                    Some(next) => parent = Some(next),
                    None => break,
                }
            }
            block.parent_label = parent;
        }
    }
    match order {
        ProjectionOrder::StablePosition => {
            blocks.sort_by_key(|block| (block.start, block.end));
        }
        ProjectionOrder::Position => {
            blocks.sort_by(|left, right| {
                (left.start, left.end, &left.label).cmp(&(right.start, right.end, &right.label))
            });
        }
        ProjectionOrder::Legislation => blocks.sort_by(|left, right| {
            left.start
                .cmp(&right.start)
                .then_with(|| right.end.cmp(&left.end))
                .then_with(|| left.label.cmp(&right.label))
        }),
        ProjectionOrder::Case | ProjectionOrder::Native => {}
    }
    create_source_doc_with_revision(provider, id, url, doc_type, text, revision, blocks)
}

pub(crate) fn native_blocks(
    text: &str,
    claims: &[NativeClaim],
    mut originals: HashMap<String, SourceDocBlock>,
) -> HashMap<String, SourceDocBlock> {
    if claims.iter().all(|claim| originals.contains_key(&claim.id)) {
        return originals;
    }
    let scalar_to_utf16 = utf16_offsets(text);
    for claim in claims {
        if originals.contains_key(&claim.id) {
            continue;
        }
        let kind = match claim.kind {
            EvidenceKind::Paragraph => SourceDocKind::Paragraph,
            EvidenceKind::Page => SourceDocKind::Page,
            EvidenceKind::Section => SourceDocKind::Section,
            EvidenceKind::Footnote => SourceDocKind::Footnote,
            EvidenceKind::Prose
            | EvidenceKind::Heading
            | EvidenceKind::Endnote
            | EvidenceKind::List
            | EvidenceKind::Navigation => continue,
        };
        let Some(label) = &claim.label else { continue };
        let mut block = SourceDocBlock::new(
            kind,
            label,
            scalar_to_utf16[claim.range.start],
            scalar_to_utf16[claim.range.end],
            SourceDocOrigin::Native,
        );
        block.aliases.clone_from(&claim.aliases);
        block.anchor.clone_from(&claim.anchor);
        block.parent_label.clone_from(&claim.parent_label);
        originals.insert(claim.id.clone(), block);
    }
    originals
}

pub(crate) fn utf16_offsets(text: &str) -> Vec<usize> {
    std::iter::once(0)
        .chain(text.chars().scan(0, |used, character| {
            *used += character.len_utf16();
            Some(*used)
        }))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_doc_bytes_and_lookup_match_the_javascript_contract() {
        let text = "1 First\n3 Third".to_owned();
        let blocks = vec![
            SourceDocBlock::new(
                SourceDocKind::Section,
                "sec1",
                0,
                7,
                SourceDocOrigin::Native,
            ),
            SourceDocBlock::new(
                SourceDocKind::Section,
                "sec3",
                8,
                15,
                SourceDocOrigin::Heuristic,
            ),
        ];
        let doc = create_source_doc(
            Some(SourceDocProvider::A2aj),
            "fixture".into(),
            None,
            Some(SourceDocType::Laws),
            text,
            blocks,
        );
        assert_eq!(doc.index.get("SEC1"), Some(0));
        assert_eq!(
            serde_json::to_string(&doc).unwrap(),
            concat!(
                r#"{"provider":"a2aj","id":"fixture","url":null,"revision":"4963dc1b70ed75305b19feedabe3e6e7a8f5fcd639a2dedf2905dcc44e7fb54b","docType":"laws","status":"usable","text":"1 First\n3 Third","blocks":[{"kind":"section","label":"sec1","start":0,"end":7,"origin":"native"},{"kind":"section","label":"sec3","start":8,"end":15,"origin":"heuristic"}],"index":{},"ranges":{"paragraph":{"kind":"paragraph","count":0,"first":null,"last":null,"missing":[],"missingTruncated":false},"page":{"kind":"page","count":0,"first":null,"last":null,"missing":[],"missingTruncated":false},"section":{"kind":"section","count":2,"first":"sec1","last":"sec3","missing":["sec2"],"missingTruncated":false},"footnote":{"kind":"footnote","count":0,"first":null,"last":null,"missing":[],"missingTruncated":false}}}"#
            )
        );
    }
}
