pub use pdf_inspector_detector::detector;
pub use pdf_inspector_detector::{DetectionConfig, PdfType, PdfTypeResult, ScanStrategy};
pub use pdf_inspector_fonts::{adobe_korea1, glyph_names, tounicode};
pub use pdf_inspector_loader::{
    load_document_from_mem, load_document_from_mem_with_password, load_document_from_path,
    load_document_from_path_with_password, validate_pdf_bytes, validate_pdf_file, PdfError,
};

pub use pdf_inspector_core::{text_utils, types};

pub mod tables;

pub mod extractor;

pub use extractor::extract_fidelity_from_doc;
pub use types::{FidelityGlyph, FidelityTextInfo, PdfLine, PdfRect, TextItem};
