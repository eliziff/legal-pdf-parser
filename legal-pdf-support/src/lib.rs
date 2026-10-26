mod adapters;
mod document;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_openvino;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
mod ppdoc_postprocess;
pub use legal_pdf_core::profile;
mod projection;

pub use document::LegalDocument;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
pub use ppdoc::{PPDocBackend, PPDocDetection, PPDocLayout, PPDocOptions, PreparedPPDoc};
pub use projection::{
    numeric_range, parse_ordinal, PdfDocument, PdfLookupNote, PdfLookupPage, PdfLookupProposition,
    PdfLookupRequest, PdfLookupStatus, PdfLookupUnit, PdfStructureLookup, PdfSummary, PdfTextLine,
    PdfTextPage, PdfTextWord, PrintedParagraph,
};
