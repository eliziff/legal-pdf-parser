pub use pdf_inspector_loader::{
    load_document_from_mem, load_document_from_path, validate_pdf_bytes, validate_pdf_file,
    PdfError, OCR_REASON_NO_TEXT, OCR_REASON_SCANNED, OCR_REASON_SUSPECTED_GARBLED_TEXT,
    OCR_REASON_VECTOR_TEXT,
};

pub mod detector;

pub use detector::*;
