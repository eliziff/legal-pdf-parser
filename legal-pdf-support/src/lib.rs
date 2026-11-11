#[cfg(feature = "ppdoc")]
mod ppdoc;
#[cfg(feature = "ppdoc")]
mod ppdoc_openvino;
#[cfg(feature = "ppdoc")]
mod ppdoc_postprocess;
pub use legal_pdf_core::profile;
mod projection;

#[cfg(feature = "ppdoc")]
pub use ppdoc::{PPDocDetection, PPDocLayout, PPDocOptions};
pub use projection::{
    brief_occurrence_pages, BriefUnit, PassageGeometry, PassageLine, PassageTarget, PlacedPassage, PdfDocument, PdfLookupPage,
    PdfLookupRequest, PdfLookupStatus, PdfLookupUnit, PdfStatus, PdfStructureLookup, PdfSummary, PrintedParagraph,
};
