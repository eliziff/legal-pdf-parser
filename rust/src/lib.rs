#[cfg(feature = "pdf-lite")]
mod contract;
#[cfg(feature = "pdf-lite")]
mod engine;
#[cfg(feature = "pdf-lite")]
mod page_map;
#[cfg(feature = "pdf-lite")]
mod supplied_ocr;

#[cfg(feature = "pdf-lite")]
pub use contract::{read_pdf, PdfDocument, PdfRequest, PdfSummary};
#[cfg(feature = "pdf-lite")]
pub use engine::{pages_needing_ocr, pdf_cache_key};
/// Recognized pages so far, and pages to recognize, as recognition proceeds.
pub type RecognitionProgress<'a> = Option<&'a (dyn Fn(usize, usize) + Sync)>;
#[cfg(feature = "pdf-lite")]
pub use legal_pdf_pages::{reread, Bands};
#[cfg(feature = "pdf-lite")]
pub use legal_pdf_extraction_processor::{pdf_opening, PdfOpening, PdfOpeningPage};
#[cfg(feature = "pdf-lite")]
pub use legal_pdf_extraction_processor::{compose, write_annotations, AnnotationFragment, AnnotationMark};
#[cfg(feature = "pdf-lite")]
pub use page_map::{page_evidence, page_map, EdgeLine, PageEvidence, PdfPageBinding};
#[cfg(feature = "ocr")]
pub use legal_pdf_ocr::read_bands;
#[cfg(feature = "ocr")]
pub use legal_pdf_core::render::page_images;
#[cfg(feature = "pdf-lite")]
#[doc(hidden)]
pub use engine::{corpus_check_cached_extraction, gold_inputs_file, gold_inputs_identity, gold_structure_document};
#[cfg(feature = "pdf-lite")]
#[doc(hidden)]
pub use legal_pdf_core::{read_gzip_packed, unpack, write_gzip_packed};
#[cfg(feature = "ocr")]
#[doc(hidden)]
pub use engine::{gold_structure_inputs, gold_structure_resume};
#[cfg(feature = "pdf-lite")]
#[doc(hidden)]
pub use legal_pdf_extraction_processor::ExtractedPdf;
#[cfg(feature = "pdf-lite")]
#[doc(hidden)]
pub use legal_pdf_structure::{stream_of, REGIONS};
#[cfg(feature = "pdf")]
#[doc(hidden)]
// A document's dump entry and the resume of the parses it keeps (the structure gold's grade).
pub use legal_pdf_structure::{doc_targets, tensors, Entry, Keep, PARSE_VERSION};
#[cfg(feature = "pdf-lite")]
#[doc(hidden)]
pub use legal_pdf_core::model::*;
#[cfg(feature = "pdf-lite")]
pub use legal_pdf_core::{AuthorityTextUnit, Error, FailureKind, Result};

#[cfg(feature = "language")]
pub use legal_pdf_language::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, docx_to_toa_text_units,
    apply_table_of_authorities, fix_docx_supra_cross_references, write_docx_spans, DocxAuthorityMark,
    DocxLinkedAuthority, DocxTableDelivery, PinpointLink, DocxEdit, DocxSpanWrite, DocxSpansWritten, DocxSupraCleanup, DocxUnit,
    apply_tracked_edits, docx_body_text, docx_tracked_change_ids, insert_tracked_blocks, resolve_tracked_changes,
    apply_text_ops, AppliedChange, BlockInsert, BlockPosition, EditDiffSegment, EditError, EditInput, EditMode, TextOpNote, TextOpParams,
    TextOpReport, TextOpRequest, TextOpsApplied, TextScope, TextSpan, TrackedEdits, compare_docx_versions, CompareAbstention,
    CompareChange, CompareKind, ComparedVersions,
};
#[cfg(feature = "ocr")]
pub use legal_pdf_ocr::{OcrLine, OcrOptions, OcrProvider, OcrWord};
#[cfg(feature = "pdf-lite")]
pub use legal_pdf_support::{
    brief_occurrence_pages, BriefOccurrence, BriefUnit, PassageGeometry, PassageLine, PassageTarget, PlacedPassage, PdfLookupPage, PdfLookupRequest, PdfLookupStatus,
    PdfLookupUnit, PdfStructureLookup,
};
#[cfg(feature = "ppdoc")]
pub use legal_pdf_support::{PPDocDetection, PPDocLayout, PPDocOptions};
