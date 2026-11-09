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
