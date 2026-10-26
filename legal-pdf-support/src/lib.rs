mod adapters;
mod document;
#[cfg(feature = "ppdoc")]
mod ppdoc;
#[cfg(feature = "ppdoc")]
mod ppdoc_openvino;
#[cfg(feature = "ppdoc")]
mod ppdoc_postprocess;
pub use legal_pdf_core::profile;
mod projection;

pub use document::LegalDocument;
#[cfg(feature = "ppdoc")]
pub use ppdoc::{PPDocDetection, PPDocLayout, PPDocOptions, PreparedPPDoc};
pub use projection::{
    numeric_range, parse_ordinal, PdfDocument, PdfLookupNote, PdfLookupPage, PdfLookupProposition,
    PdfLookupRequest, PdfLookupStatus, PdfLookupUnit, PdfStructureLookup, PdfSummary, PdfTextLine,
    PdfTextPage, PdfTextWord, PrintedParagraph,
};
