pub use legal_structure::{Derivation, DocumentStructure, NodeKind, ScalarRange, StructureNode};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub const SCHEMA_VERSION: &str = "legalpdf.document.v4";
pub const PARSER_VERSION: &str = "0.3.0";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub id: String,
    pub text: String,
    pub bbox: [f64; 4],
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    pub id: String,
    pub text: String,
    pub bbox: [f64; 4],
    #[serde(default)]
    pub font: String,
    #[serde(default)]
    pub size: f64,
    #[serde(default)]
    pub flags: u32,
    #[serde(default)]
    pub superscript: bool,
    #[serde(default)]
    pub start: usize,
    #[serde(default)]
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetachedReference {
    pub note_id: String,
    pub selected_text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub source_line_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub id: String,
    pub page_index: usize,
    pub page_number: u32,
    pub source_index: usize,
    pub reading_order: usize,
    pub block_index: usize,
    pub text: String,
    pub bbox: [f64; 4],
    #[serde(default)]
    pub spans: Vec<Span>,
    #[serde(default)]
    pub words: Vec<Word>,
    #[serde(default)]
    pub detached_references: Vec<DetachedReference>,
    #[serde(default)]
    pub exclude_from_body: bool,
    #[serde(default)]
    pub suppress_footnote_label: bool,
    #[serde(default)]
    pub note_region_mode: String,
    #[serde(default)]
    pub region_id: String,
    #[serde(default = "unknown_region")]
    pub region_type: String,
    #[serde(default = "native_source")]
    pub source: String,
}

fn unknown_region() -> String {
    "unknown".to_owned()
}

fn native_source() -> String {
    "native".to_owned()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: String,
    pub page_index: usize,
    #[serde(rename = "type")]
    pub kind: String,
    pub line_ids: Vec<String>,
    pub bbox: [f64; 4],
    pub reading_order: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub index: usize,
    pub number: u32,
    pub width: f64,
    pub height: f64,
    pub lines: Vec<Line>,
    pub regions: Vec<Region>,
    #[serde(default = "native_source")]
    pub source: String,
    #[serde(default = "one")]
    pub text_quality: f64,
    #[serde(default)]
    pub printed_label: Option<String>,
    #[serde(default)]
    pub printed_label_source: Option<String>,
    #[serde(default)]
    pub printed_label_line_id: Option<String>,
}

const fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParagraphAnchor {
    pub pair_id: String,
    pub label: String,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paragraph {
    pub id: String,
    pub page_index: usize,
    pub region_type: String,
    pub text: String,
    pub line_ids: Vec<String>,
    #[serde(default)]
    pub anchors: Vec<ParagraphAnchor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FootnoteCrossref {
    pub source_pair_id: String,
    pub kind: String,
    pub number: u32,
    pub shortform: String,
    pub start: usize,
    pub end: usize,
    pub resolved: bool,
    pub target_pair_id: String,
    pub target_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Footnote {
    pub pair_id: String,
    pub label: String,
    pub occurrence: usize,
    pub restart_sequence: usize,
    pub reference_page: Option<u32>,
    pub body_pages: Vec<u32>,
    pub reference_line_id: Option<String>,
    pub body_line_ids: Vec<String>,
    pub body: String,
    pub sentence_proposition: String,
    pub passage_since_prior_note: String,
    pub confidence: f64,
    pub provenance: String,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub crossrefs: Vec<FootnoteCrossref>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: String,
    pub message: String,
    #[serde(default)]
    pub page_index: Option<usize>,
    #[serde(default)]
    pub line_ids: Vec<String>,
    #[serde(default)]
    pub details: Map<String, Value>,
}

impl Diagnostic {
    pub fn warning(code: &str, message: impl Into<String>, page_index: Option<usize>) -> Self {
        Self {
            code: code.to_owned(),
            severity: "warning".to_owned(),
            message: message.into(),
            page_index,
            line_ids: Vec::new(),
            details: Map::new(),
        }
    }

    pub fn info(code: &str, message: impl Into<String>, page_index: Option<usize>) -> Self {
        Self {
            code: code.to_owned(),
            severity: "info".to_owned(),
            message: message.into(),
            page_index,
            line_ids: Vec::new(),
            details: Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PdfSourceSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PdfPairingAudit {
    pub markers: Vec<Value>,
    pub pairing_summary: Value,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdfExtractionMetadata {
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegalDocument {
    pub document_id: String,
    pub source_name: String,
    pub source_sha256: String,
    pub page_count: usize,
    pub status: String,
    pub pages: Vec<Page>,
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub structure_graph: DocumentStructure,
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub provenance: Map<String, Value>,
    #[serde(default = "schema_version")]
    pub schema_version: String,
    #[serde(default = "parser_version")]
    pub parser_version: String,
}

fn schema_version() -> String {
    SCHEMA_VERSION.to_owned()
}

fn parser_version() -> String {
    PARSER_VERSION.to_owned()
}

impl LegalDocument {
    pub fn line_count(&self) -> usize {
        self.pages.iter().map(|page| page.lines.len()).sum()
    }
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub pair_id: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotePairKind {
    Footnote,
    Endnote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAnchor {
    pub line_id: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotePairClaim {
    pub pair_id: String,
    pub label: String,
    pub kind: NotePairKind,
    pub label_anchor: SourceAnchor,
    pub reference_anchors: Vec<SourceAnchor>,
    pub body_line_ids: Vec<String>,
}

pub struct PairingOutput {
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub anchors: HashMap<String, Vec<Anchor>>,
    pub pair_claims: Vec<NotePairClaim>,
    pub pairing_audit: Option<PdfPairingAudit>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_field_names_match_the_public_contract() {
        let region = Region {
            id: "p0001-r0001".to_owned(),
            page_index: 0,
            kind: "body".to_owned(),
            line_ids: vec![],
            bbox: [0.0; 4],
            reading_order: 1,
        };
        let value = serde_json::to_value(region).unwrap();
        assert_eq!(value["type"], "body");
        assert!(value.get("kind").is_none());
        assert_eq!(SCHEMA_VERSION, "legalpdf.document.v4");
    }
}
