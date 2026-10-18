//! A parsed PDF: its pages, paragraphs and notes, and the structure graph the engine reads from
//! them.

use legal_pdf_core::model::{Diagnostic, Footnote, Page, Paragraph, PARSER_VERSION, SCHEMA_VERSION};
use legal_structure_model::DocumentStructure;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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
