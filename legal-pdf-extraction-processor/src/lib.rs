#[cfg(feature = "full")]
mod annotate;
#[cfg(feature = "full")]
pub mod compose;
mod extract;
#[cfg(feature = "full")]
mod outline;
#[cfg(feature = "full")]
mod page_labels;
mod pdf;
#[cfg(feature = "full")]
mod reread;
#[cfg(feature = "full")]
mod paragraph_recovery;
#[cfg(feature = "full")]
pub use annotate::{write_annotations, AnnotationFragment, AnnotationMark};
#[cfg(feature = "full")]
pub use outline::embedded_outline;
#[cfg(feature = "full")]
pub use page_labels::embedded_page_labels;

#[cfg(feature = "full")]
pub use extract::{extract_pdf, pdf_opening, PdfOpening, PdfOpeningPage};
pub use extract::extract_pdf_text_only;
pub use pdf::{
    add_rect_regions, assemble_pdf, mark_rule_underlines, load_extraction_document, page_geometries, ExtractedPdf,
};
#[cfg(feature = "full")]
pub use pdf::recognize_pdf;
