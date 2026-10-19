#[cfg(feature = "indra")]
mod indra;
#[cfg(feature = "indra")]
mod line_detector;
#[cfg(feature = "indra")]
mod model_pack;
mod ocr;
#[cfg(feature = "indra")]
mod orientation;
#[cfg(feature = "indra")]
mod planning;
mod separator;
#[cfg(feature = "indra")]
mod tesseract_layout;
#[cfg(feature = "indra")]
mod uncovered;

#[cfg(feature = "indra")]
pub use indra::{
    IndraBackend, IndraBatchDiagnostics, IndraBatchPerformance, IndraImageDiagnostics, IndraLayout,
    IndraOcr, IndraOptions, IndraTier,
};
pub use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider};
pub use ocr::{
    raster_page_rules, OcrOptions, OcrProvider, PreparedOcrProvider, RasterRules, TesseractOcr,
    TesseractOptions,
};
#[cfg(feature = "indra")]
pub use planning::RuntimePlan as IndraRuntimePlan;
