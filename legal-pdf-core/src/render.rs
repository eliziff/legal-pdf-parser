//! A PDF's pages drawn on white, for what reads their images: recognition, the layout model and
//! the raster measures.
use crate::{Error, FailureKind, Result};
use hayro::hayro_interpret::InterpreterSettings;
use hayro::hayro_syntax::LoadPdfError;
pub use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
pub use hayro::vello_cpu::Pixmap;
use hayro::{RenderCache, RenderSettings};

/// `bytes` opened for drawing: password-protected or corrupt where it cannot be.
pub fn open_pdf(bytes: &[u8]) -> Result<Pdf> {
    Pdf::new(bytes.to_vec()).map_err(|error| {
        let kind = match error {
            LoadPdfError::Decryption(_) => FailureKind::Encrypted,
            LoadPdfError::Invalid => FailureKind::Corrupt,
        };
        Error::Message(format!("PDF renderer could not open the PDF: {error:?}")).failure(kind)
    })
}

/// A PDF's pages drawn at one resolution, sharing one render cache, so the fonts and images they
/// share are decoded once. The cache is not `Send`: a thread draws with its own.
pub struct PageRenderer<'a> {
    pdf: &'a Pdf,
    cache: RenderCache<'a>,
    interpreter: InterpreterSettings,
    settings: RenderSettings,
}

impl<'a> PageRenderer<'a> {
    pub fn new(pdf: &'a Pdf, dpi: f64) -> Self {
        let scale = (dpi / 72.0) as f32;
        Self {
            pdf,
            cache: RenderCache::new(),
            interpreter: InterpreterSettings::default(),
            settings: RenderSettings { x_scale: scale, y_scale: scale, bg_color: WHITE, ..Default::default() },
        }
    }

    /// Page `index` drawn; none where the document has no such page or it draws to nothing.
    pub fn render(&self, index: usize) -> Option<Pixmap> {
        let page = self.pdf.pages().get(index)?;
        let pixmap = hayro::render(page, &self.cache, &self.interpreter, &self.settings);
        (pixmap.width() > 0 && pixmap.height() > 0).then_some(pixmap)
    }
}

/// Every page of `bytes` drawn so its long side is `long_side` pixels, at most `dpi_cap` dpi (whole
/// dpi, rounded down), as PNG: each page's dpi and image, in page order. What a model reading the
/// pages is shown beside the lines this engine extracted from them.
pub fn page_images(bytes: &[u8], long_side: f64, dpi_cap: u32) -> Result<Vec<(u32, Vec<u8>)>> {
    let pdf = open_pdf(bytes)?;
    let mut renderer: Option<(u32, PageRenderer)> = None;
    (0..pdf.pages().len())
        .map(|index| {
            let (width, height) = pdf.pages()[index].render_dimensions();
            let dpi = dpi_cap.min((long_side * 72.0 / f64::from(width.max(height).max(1.0))) as u32);
            if renderer.as_ref().is_none_or(|(drawn, _)| *drawn != dpi) {
                renderer = Some((dpi, PageRenderer::new(&pdf, f64::from(dpi))));
            }
            let pixmap = renderer.as_ref().and_then(|(_, renderer)| renderer.render(index));
            let png = pixmap
                .ok_or_else(|| Error::Message(format!("page {} draws to nothing", index + 1)))?
                .into_png()
                .map_err(|error| Error::Message(format!("page {} could not be written as PNG: {error}", index + 1)))?;
            Ok((dpi, png))
        })
        .collect()
}
