#[cfg(feature = "ocr")]
mod indra;
#[cfg(feature = "ocr")]
mod line_detector;
#[cfg(feature = "ocr")]
mod model_pack;
mod ocr;
#[cfg(feature = "ocr")]
mod orientation;
#[cfg(feature = "ocr")]
mod planning;
mod separator;
#[cfg(feature = "ocr")]
mod uncovered;

#[cfg(feature = "ocr")]
pub use indra::{
    IndraBackend, IndraBatchDiagnostics, IndraBatchPerformance, IndraImageDiagnostics, IndraLayout,
    IndraOcr, IndraOptions, IndraTier,
};
pub use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider};
pub use ocr::{raster_page_rules, OcrOptions, OcrProvider, PreparedOcrProvider, RasterRules};
#[cfg(feature = "ocr")]
pub use planning::RuntimePlan as IndraRuntimePlan;
