mod docx;
mod error;
mod process;

pub use docx::{
    analyze_docx_bytes, analyze_docx_drafting_bytes, docx_text, fix_docx_supra_cross_references,
    has_docx_supra_references, DocxSupraCleanup, MAX_DOCX_SUPRA_BYTES,
};
pub use error::{Error, Result};
