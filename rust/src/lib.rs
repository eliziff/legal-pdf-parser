#[cfg(feature = "pdf")]
mod contract;
#[cfg(feature = "pdf")]
mod engine;
#[cfg(feature = "pdf")]
pub mod structure;
mod structure_engine;

#[cfg(all(feature = "profiling", feature = "fast-allocator"))]
compile_error!("profiling and fast-allocator cannot select two global allocators");

#[cfg(feature = "pdf")]
pub use contract::{
    derive_pdf_document, pdf_document_snapshot, pdf_document_summary, query_pdf_document,
    PdfDocumentResult,
};
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use engine::{corpus_check_cached_extraction, digest_cached_extraction};
#[cfg(feature = "pdf")]
pub use legal_pdf_core::model::*;
#[cfg(feature = "pdf")]
pub use legal_pdf_core::{Error, Result};
#[cfg(feature = "pdf")]
pub use legal_pdf_extraction::pdf_page_count;
#[cfg(feature = "language")]
pub use legal_pdf_language::{
    analyze_docx_bytes, apply_docx_links, assess_docx_route, compile_table_entry,
    deterministic_docx_intents, extract_docx_gold, fix_docx_supra_cross_references,
    has_docx_supra_references, load_tables, plan_docx_links, plan_footnotes, run_grammar_vectors,
    validate_docx_response, DocxPlanOptions, DocxSupraCleanup, MAX_DOCX_SUPRA_BYTES,
};
#[cfg(feature = "language")]
pub use legal_pdf_language::{
    extract_citation_fields, split_citations, split_citations_recall_first, DeterministicFields,
    DeterministicPart, DeterministicSplit,
};
#[cfg(feature = "kraken")]
pub use legal_pdf_ocr::{
    KrakenBackend, KrakenBatchDiagnostics, KrakenBatchPerformance, KrakenImageDiagnostics,
    KrakenLayout, KrakenOcr, KrakenOptions, KrakenTier,
};
#[cfg(feature = "ocr")]
pub use legal_pdf_ocr::{
    OcrLine, OcrOptions, OcrProvider, OcrWord, TesseractOcr, TesseractOptions,
};
#[cfg(feature = "pdf")]
pub use legal_pdf_support::{
    lookup_footnote, project_structure, structure_lookup, to_alr_payload, to_toa_text_units,
};
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
pub use legal_pdf_support::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions};
pub use structure_engine::{DocumentInput, DocumentStructure, EvidenceError};
