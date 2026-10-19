#[cfg(feature = "pdf")]
mod contract;
#[cfg(feature = "pdf")]
mod engine;
#[cfg(feature = "pdf")]
mod supplied_ocr;

#[cfg(all(feature = "allocation-profiling", feature = "fast-allocator"))]
compile_error!("profiling and fast-allocator cannot select two global allocators");

#[cfg(feature = "pdf")]
pub use contract::{
    derive_pdf_document, pdf_document_summary, prepare_pdf_document,
    prepare_pdf_document_reporting, query_pdf_document, restore_pdf_document, PdfDocument,
    PdfRequest, PdfSummary,
};
#[cfg(feature = "pdf")]
pub use engine::RecognitionProgress;
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use engine::corpus_check_cached_extraction;
#[cfg(feature = "pdf")]
pub use legal_pdf_core::model::*;
#[cfg(feature = "pdf")]
pub use legal_pdf_core::{Error, Result};

/// Supplies the structure engine to the parsers: its document reading, and its services on
/// laid-out text. The first one supplied stays.
pub fn install_structure_analysis(analysis: &'static dyn legal_structure_model::StructureAnalysis) {
    legal_structure_model::install_structure_analysis(analysis);
    legal_pdf_core::install_layout_analysis(analysis);
}
#[cfg(feature = "language")]
pub use legal_pdf_language::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, docx_to_toa_text_units,
    fix_docx_supra_cross_references, has_docx_supra_references, DocxSupraCleanup,
    MAX_DOCX_SUPRA_BYTES,
};
#[cfg(feature = "indra")]
pub use legal_pdf_ocr::{
    IndraBackend, IndraBatchDiagnostics, IndraBatchPerformance, IndraImageDiagnostics,
    IndraLayout, IndraOcr, IndraOptions, IndraTier,
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

