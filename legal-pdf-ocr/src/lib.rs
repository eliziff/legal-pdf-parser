#[cfg(feature = "kraken")]
mod kraken;
#[cfg(feature = "kraken")]
mod kraken_process;
mod ocr;
mod separator;
#[cfg(feature = "kraken")]
mod tesseract_layout;

#[cfg(feature = "kraken")]
pub use kraken::{
    KrakenBackend, KrakenBatchDiagnostics, KrakenBatchPerformance, KrakenImageDiagnostics,
    KrakenLayout, KrakenOcr, KrakenOptions, KrakenTier,
};
pub use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord, PdfOcrProvider};
pub use ocr::{OcrOptions, OcrProvider, TesseractOcr, TesseractOptions};
