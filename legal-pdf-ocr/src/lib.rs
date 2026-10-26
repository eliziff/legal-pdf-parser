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
mod raster;
mod separator;
#[cfg(feature = "ocr")]
mod uncovered;

#[cfg(feature = "ocr")]
pub use indra::{
    IndraBackend, IndraBatchDiagnostics, IndraBatchPerformance, IndraImageDiagnostics, IndraLayout,
    IndraOcr, IndraOptions, IndraTier,
};
pub use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider, RasterReading};
pub use ocr::{OcrOptions, OcrProvider, PreparedOcrProvider};
#[cfg(feature = "ocr")]
pub use ocr::{render_gray_page, GrayRender};
pub use raster::{raster_page_rules, render_gray_pages};
#[cfg(feature = "ocr")]
pub use planning::RuntimePlan as IndraRuntimePlan;
#[cfg(feature = "attribution")]
pub use indra::attribution::ProbeBox;
