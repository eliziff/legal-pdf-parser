pub mod detector_native;
mod gutter;
mod indra;
pub mod indra_native;
mod line_detector;
mod model_pack;
mod ocr;
mod orientation;
mod planning;
mod raster;
mod separator;
mod simd_kernel;
mod uncovered;

pub use indra::{IndraTier, OcrOptions, OcrProvider, PreparedOcrProvider};
pub use indra_native::{IndraNative, IndraScratch};
pub use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, OrtBackend, PdfOcrProvider, RasterReading};
pub use ocr::{read_bands, render_gray_page, GrayRender, REREAD_SCALE};
pub use planning::RuntimePlan;
pub use raster::raster_page_rules;
#[cfg(feature = "attribution")]
pub use indra::attribution::ProbeBox;
