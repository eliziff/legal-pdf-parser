mod error;
mod outline;
mod page_labels;
mod pdf;
pub use outline::embedded_outline;
pub use page_labels::embedded_page_labels;

pub use error::{Error, Result};
pub use pdf::{
    assemble_pdf, load_extraction_document, page_geometries, recognize_pdf, ExtractedPdf,
};
