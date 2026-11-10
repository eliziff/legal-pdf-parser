mod annotate;
mod extract;
mod outline;
mod page_labels;
mod pdf;
mod reread;
mod paragraph_recovery;
pub use annotate::{write_annotations, AnnotationFragment, AnnotationMark};
pub use outline::embedded_outline;
pub use page_labels::embedded_page_labels;

pub use extract::{extract_pdf, pdf_opening, PdfOpening, PdfOpeningPage};
pub use pdf::{
    add_rect_regions, assemble_pdf, mark_rule_underlines, load_extraction_document, page_geometries, recognize_pdf, ExtractedPdf,
};
