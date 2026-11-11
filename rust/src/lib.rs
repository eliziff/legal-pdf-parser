#[cfg(feature = "pdf")]
mod contract;
#[cfg(feature = "pdf")]
mod engine;
#[cfg(feature = "pdf")]
mod page_map;
#[cfg(feature = "pdf")]
mod supplied_ocr;

#[cfg(feature = "pdf")]
pub use contract::{read_pdf, PdfDocument, PdfRequest, PdfSummary};
#[cfg(feature = "pdf")]
pub use engine::{pages_needing_ocr, pdf_cache_key};
/// Recognized pages so far, and pages to recognize, as recognition proceeds.
pub type RecognitionProgress<'a> = Option<&'a (dyn Fn(usize, usize) + Sync)>;
#[cfg(feature = "pdf")]
pub use legal_pdf_pages::{reread, Bands};
#[cfg(feature = "pdf")]
pub use legal_pdf_extraction_processor::{pdf_opening, PdfOpening, PdfOpeningPage};
#[cfg(feature = "pdf")]
pub use legal_pdf_extraction_processor::{compose, write_annotations, AnnotationFragment, AnnotationMark};
#[cfg(feature = "pdf")]
pub use page_map::{page_evidence, page_map, EdgeLine, PageEvidence, PdfPageBinding};
#[cfg(feature = "ocr")]
pub use legal_pdf_ocr::read_bands;
#[cfg(feature = "ocr")]
pub use page_map::page_evidence_read;
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use engine::{corpus_check_cached_extraction, gold_inputs_identity, gold_structure_document};
#[cfg(feature = "ocr")]
#[doc(hidden)]
pub use engine::gold_structure_inputs;
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use legal_pdf_extraction_processor::ExtractedPdf;
#[cfg(feature = "pdf")]
#[doc(hidden)]
pub use legal_pdf_core::model::*;
#[cfg(feature = "pdf")]
pub use legal_pdf_core::{AuthorityTextUnit, Error, FailureKind, Result};

#[cfg(feature = "language")]
pub use legal_pdf_language::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, docx_to_toa_text_units,
    fix_docx_supra_cross_references, DocxSupraCleanup,
};
#[cfg(feature = "ocr")]
pub use legal_pdf_ocr::{OcrLine, OcrOptions, OcrProvider, OcrWord};
#[cfg(feature = "pdf")]
pub use legal_pdf_support::{
    brief_occurrence_pages, BriefOccurrence, BriefUnit, PassageGeometry, PassageLine, PassageTarget, PlacedPassage, PdfLookupPage, PdfLookupRequest, PdfLookupStatus,
    PdfLookupUnit, PdfStructureLookup,
};
#[cfg(feature = "ppdoc")]
pub use legal_pdf_support::{PPDocDetection, PPDocLayout, PPDocOptions};
