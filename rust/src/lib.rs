#[cfg(feature = "pdf")]
mod contract;
#[cfg(feature = "pdf")]
mod engine;
mod structure_engine;

#[cfg(all(feature = "allocation-profiling", feature = "fast-allocator"))]
compile_error!("profiling and fast-allocator cannot select two global allocators");

#[cfg(feature = "pdf")]
pub use contract::{
    derive_pdf_document, pdf_document_summary, prepare_pdf_document, query_pdf_document,
    restore_pdf_document, PdfDocument, PdfRequest, PdfSummary,
};
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use engine::{corpus_check_cached_extraction, digest_cached_extraction};
#[cfg(feature = "pdf")]
pub use legal_pdf_core::model::*;
#[cfg(feature = "pdf")]
pub use legal_pdf_core::{Error, Result};
#[cfg(feature = "language")]
pub use legal_pdf_language::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, docx_to_toa_text_units,
    fix_docx_supra_cross_references, has_docx_supra_references, DocxSupraCleanup,
    MAX_DOCX_SUPRA_BYTES,
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
    to_alr_payload, to_toa_text_units, PdfLookupNote, PdfLookupPage, PdfLookupProposition,
    PdfLookupRequest, PdfLookupStatus, PdfLookupUnit, PdfStructureLookup,
};
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
pub use legal_pdf_support::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions};
