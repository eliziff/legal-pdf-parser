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
pub use ocr::{OcrOptions, OcrProvider, PreparedOcrProvider, TesseractOcr, TesseractOptions};

/// Scan a grayscale raster for the same footnote separator used by native OCR.
/// Coordinates are returned in the caller's page units.
pub fn raster_separator_y_from_gray(
    gray: &[u8],
    width: usize,
    height: usize,
    page_height: f64,
) -> Option<f64> {
    if width == 0
        || height == 0
        || width.checked_mul(height) != Some(gray.len())
        || !page_height.is_finite()
        || page_height <= 0.0
    {
        return None;
    }
    let record = separator::scan_gray_page(gray, width, height);
    if !matches!(record.separator_status, Some("found" | "found_two_column")) {
        return None;
    }
    record
        .separators
        .and_then(|rules| rules.first().map(|rule| rule.y_center_ratio * page_height))
}
