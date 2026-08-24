#[cfg(feature = "pdf")]
use legal_pdf_core::model::{
    Diagnostic, Footnote, LegalDocument, Page, Paragraph, PdfPairingAudit, PARSER_VERSION,
    SCHEMA_VERSION,
};
#[cfg(feature = "pdf")]
use legal_pdf_structure::{replay, status, validate_document, StructureIdentity, StructureReplay};
#[cfg(feature = "pdf")]
use serde_json::Map;

use legal_structure::{DocumentStructure, EngineError};

#[cfg(feature = "pdf")]
#[derive(Debug)]
pub(crate) struct PdfReplayProjection {
    pub source_sha256: String,
    pub status: String,
    pub derived_pages: Vec<Page>,
    pub prepared_pages: Vec<Page>,
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub pairing_audit: PdfPairingAudit,
    pub structure_graph: DocumentStructure,
}

#[cfg(feature = "pdf")]
pub(crate) fn derive_pdf_pages(
    document_id: String,
    source_sha256: String,
    mut pages: Vec<Page>,
    separators: Vec<Option<f64>>,
) -> Result<PdfReplayProjection, EngineError> {
    if document_id.is_empty() || source_sha256.len() != 64 || pages.len() != separators.len() {
        return Err(EngineError {
            code: "invalid_evidence",
            message: "invalid Page-backed evidence".to_owned(),
        });
    }
    let StructureReplay {
        prepared_pages,
        derived,
    } = replay(
        &mut pages,
        &separators,
        StructureIdentity {
            document_id: document_id.clone(),
            source_sha256: source_sha256.clone(),
        },
    )
    .map_err(|error| EngineError {
        code: "invalid_evidence",
        message: error.to_string(),
    })?;
    let pairing_audit = derived
        .pairing_audit
        .expect("replay includes pairing audit");
    let document = LegalDocument {
        document_id,
        source_name: String::new(),
        source_sha256,
        page_count: pages.len(),
        status: status(&derived.diagnostics, &pages),
        pages,
        paragraphs: derived.paragraphs,
        footnotes: derived.footnotes,
        structure_graph: derived.structure_graph,
        diagnostics: derived.diagnostics,
        metadata: Map::new(),
        provenance: Map::new(),
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    validate_document(&document).map_err(|error| EngineError {
        code: "invalid_evidence",
        message: error.to_string(),
    })?;
    Ok(PdfReplayProjection {
        source_sha256: document.source_sha256,
        status: document.status,
        derived_pages: document.pages,
        prepared_pages,
        paragraphs: document.paragraphs,
        footnotes: document.footnotes,
        diagnostics: document.diagnostics,
        pairing_audit,
        structure_graph: document.structure_graph,
    })
}
