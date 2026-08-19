use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

pub const SCHEMA_VERSION: &str = "legalpdf.document.v2";
pub const GEOMETRY_SCHEMA_VERSION: &str = "legalpdf.geometry.v1";
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
    pub detached_references: Vec<Value>,
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
pub struct TableBlock {
    pub id: String,
    pub page_index: usize,
    pub page_number: u32,
    pub bbox: [f64; 4],
    pub cells: Vec<Vec<String>>,
    pub provenance: String,
    pub confidence: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageBlock {
    pub id: String,
    pub page_index: usize,
    pub page_number: u32,
    pub bbox: [f64; 4],
    pub source_name: String,
    pub area_ratio: f64,
    pub route: String,
    pub route_reason: String,
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
pub struct Paragraph {
    pub id: String,
    pub page_index: usize,
    pub region_type: String,
    pub text: String,
    pub line_ids: Vec<String>,
    #[serde(default)]
    pub anchors: Vec<Value>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Section {
    pub id: String,
    pub heading_paragraph_id: String,
    pub heading: String,
    pub locator: String,
    pub locator_kind: Option<String>,
    pub aliases: Vec<String>,
    pub text: String,
    pub paragraph_ids: Vec<String>,
    pub page_indexes: Vec<usize>,
    pub line_ids: Vec<String>,
    #[serde(default = "heading_provenance")]
    pub provenance: String,
}

fn heading_provenance() -> String {
    "heading-region".to_owned()
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
    pub crossrefs: Vec<Value>,
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RepairRecord {
    pub page_index: usize,
    pub status: String,
    pub model: String,
    pub effort: String,
    pub prompt_version: String,
    pub cache_key: String,
    pub attempts: usize,
    pub elapsed_seconds: f64,
    pub input_line_hash: String,
    #[serde(default)]
    pub output_hash: String,
    #[serde(default)]
    pub token_usage: Map<String, Value>,
    #[serde(default)]
    pub error: String,
    #[serde(default)]
    pub scope_pages: Vec<usize>,
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
    pub sections: Vec<Section>,
    pub footnotes: Vec<Footnote>,
    pub tables: Vec<TableBlock>,
    pub images: Vec<ImageBlock>,
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub repairs: Vec<RepairRecord>,
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

    pub fn text(&self) -> String {
        self.paragraphs
            .iter()
            .map(|paragraph| paragraph.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    }

    pub fn manifest(&self, compact_pages: bool) -> Value {
        let mut value = json!({
            "schema_version": self.schema_version,
            "parser_version": self.parser_version,
            "document_id": self.document_id,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "page_count": self.page_count,
            "status": self.status,
            "metadata": self.metadata,
            "provenance": self.provenance,
            "counts": {
                "pages": self.pages.len(),
                "lines": self.line_count(),
                "paragraphs": self.paragraphs.len(),
                "sections": self.sections.len(),
                "footnotes": self.footnotes.len(),
                "tables": self.tables.len(),
                "images": self.images.len(),
                "diagnostics": self.diagnostics.len(),
                "repairs": self.repairs.len(),
            },
            "artifacts": {
                "pages": "pages.jsonl",
                "paragraphs": "paragraphs.jsonl",
                "sections": "sections.jsonl",
                "footnotes": "footnotes.jsonl",
                "tables": "tables.jsonl",
                "images": "images.jsonl",
                "diagnostics": "diagnostics.jsonl",
                "repairs": "repairs.jsonl",
            },
        });
        if compact_pages {
            value["artifact_profile"] = Value::String("compact-source".to_owned());
        }
        value
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FootnoteLookup {
    pub status: String,
    pub query: String,
    pub matches: Vec<String>,
    #[serde(default)]
    pub footnote: Option<Footnote>,
    #[serde(default = "sentence_mode")]
    pub proposition_mode: String,
    #[serde(default)]
    pub proposition: String,
    #[serde(default)]
    pub context: String,
}

fn sentence_mode() -> String {
    "sentence".to_owned()
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
        assert_eq!(SCHEMA_VERSION, "legalpdf.document.v2");
    }
}
