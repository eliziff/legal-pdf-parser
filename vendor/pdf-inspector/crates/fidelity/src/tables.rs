pub use pdf_inspector_tables::*;

#[path = "structured_detection.rs"]
mod structured_detection;

pub use structured_detection::{detect_structured_tables, DetectedTable};
