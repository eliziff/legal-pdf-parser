#[cfg(feature = "pdf")]
use crate::structure::{replay, status, validate_document, StructureIdentity};
#[cfg(feature = "pdf")]
use legal_pdf_core::model::{
    Diagnostic, Footnote, LegalDocument, Page, Paragraph, PARSER_VERSION, SCHEMA_VERSION,
};
#[cfg(feature = "pdf")]
use serde_json::{Map, Value};

pub use legal_structure::{DocumentInput, EngineError as EvidenceError, StructureGraphV2};

#[cfg(feature = "pdf")]
#[derive(Debug)]
pub(crate) struct PdfReplayProjection {
    pub source_sha256: String,
    pub status: String,
    pub validation: &'static str,
    pub derived_pages: Vec<Page>,
    pub prepared_pages: Vec<Page>,
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub markers: Vec<Value>,
    pub marker_summary: Value,
    pub pairing_summary: Value,
    pub structure_graph: StructureGraphV2,
}

#[cfg(feature = "pdf")]
pub(crate) fn derive_pdf_pages(
    document_id: String,
    source_sha256: String,
    mut pages: Vec<Page>,
    separators: Vec<Option<f64>>,
) -> Result<PdfReplayProjection, EvidenceError> {
    if document_id.is_empty() || source_sha256.len() != 64 || pages.len() != separators.len() {
        return Err(EvidenceError {
            code: "invalid_evidence",
            message: "invalid Page-backed evidence".to_owned(),
        });
    }
    let structure = replay(
        &mut pages,
        &separators,
        StructureIdentity {
            document_id: document_id.clone(),
            source_sha256: source_sha256.clone(),
        },
    )
    .map_err(|error| EvidenceError {
        code: "invalid_evidence",
        message: error.to_string(),
    })?;
    let document = LegalDocument {
        document_id,
        source_name: String::new(),
        source_sha256: source_sha256.clone(),
        page_count: pages.len(),
        status: status(&structure.derived.diagnostics, &pages),
        pages,
        paragraphs: structure.derived.paragraphs,
        footnotes: structure.derived.footnotes,
        tables: Vec::new(),
        images: Vec::new(),
        structure_graph: structure.derived.structure_graph,
        diagnostics: structure.derived.diagnostics,
        repairs: Vec::new(),
        metadata: Map::new(),
        provenance: Map::new(),
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    validate_document(&document).map_err(|error| EvidenceError {
        code: "invalid_evidence",
        message: error.to_string(),
    })?;
    Ok(PdfReplayProjection {
        source_sha256,
        status: document.status,
        validation: "ok",
        derived_pages: document.pages,
        prepared_pages: structure.prepared_pages,
        paragraphs: document.paragraphs,
        footnotes: document.footnotes,
        diagnostics: document.diagnostics,
        markers: structure.derived.markers,
        marker_summary: structure.derived.marker_summary,
        pairing_summary: structure.derived.pairing_summary,
        structure_graph: document.structure_graph,
    })
}
