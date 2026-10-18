mod adapters;
mod document;
pub mod pairing_support;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_openvino;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_postprocess;
pub use legal_pdf_core::profile;
mod projection;

pub use adapters::{to_alr_payload, to_toa_text_units};
pub use document::LegalDocument;
pub use pairing_support::{
    has_citation_signal, has_legal_citation_cue, heading_text_plausible,
    is_legal_citation_continuation, protected_citation_spans, styled_heading_text_plausible,
};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
pub use ppdoc::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions, PreparedPPDoc};
pub use projection::{
    numeric_range, parse_ordinal, PdfDocument, PdfLookupNote, PdfLookupPage, PdfLookupProposition,
    PdfLookupRequest, PdfLookupStatus, PdfLookupUnit, PdfStructureLookup, PdfSummary, PdfTextLine,
    PdfTextPage, PdfTextWord, PrintedParagraph,
};
