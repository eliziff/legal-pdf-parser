mod error;
mod pdf;

pub use error::{Error, Result};
pub use legal_pdf_core::union_bbox;
pub use pdf::{
    assemble_pdf, load_extraction_document, page_dimensions, project_table, ExtractedPdf,
};
