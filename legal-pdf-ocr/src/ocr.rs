//! The recognizer as the parser reads a PDF with it: its pages, the separators read from their
//! images, and parts of a page read again from a render.
use crate::indra::OcrProvider;
use legal_pdf_core::render::{open_pdf, PageRenderer, Pixmap};
use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, PdfOcrProvider, Result};

impl PdfOcrProvider for OcrProvider {
    fn extract_pages(&mut self, pdf: &[u8], requests: &[OcrPageRequest]) -> Result<Vec<OcrPageResult>> {
        self.recognize_pdf(pdf, requests)
    }

    fn separators(&mut self, pdf: &[u8], requests: &[OcrPageRequest]) -> Vec<Option<f64>> {
        crate::raster::raster_page_rules(pdf, requests).into_iter().map(|reading| reading.separator).collect()
    }

    fn read_region(&mut self, pdf: &[u8], page_index: usize, region: [f64; 4], scale: f64) -> Result<Vec<OcrLine>> {
        match render_gray_page(pdf, page_index, scale * 72.0) {
            Some(render) => self.read_rendered_region(&render, region, scale),
            None => Ok(Vec::new()),
        }
    }
}

/// A page rendered to 8-bit grey: (width, height, pixels).
pub type GrayRender = (usize, usize, Vec<u8>);

impl OcrProvider {
    /// `region` of a page already rendered at `scale` times its points, read: a caller reading
    /// several regions of one page renders it once.
    pub fn read_rendered_region(&mut self, render: &GrayRender, region: [f64; 4], scale: f64) -> Result<Vec<OcrLine>> {
        let (width, height, gray) = (render.0, render.1, &render.2);
        let x0 = ((region[0] * scale).floor().max(0.0) as usize).min(width);
        let x1 = ((region[2] * scale).ceil().max(0.0) as usize).min(width);
        let y0 = ((region[1] * scale).floor().max(0.0) as usize).min(height);
        let y1 = ((region[3] * scale).ceil().max(0.0) as usize).min(height);
        if x1 <= x0 + 4 || y1 <= y0 + 4 {
            return Ok(Vec::new());
        }
        let mut crop = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for y in y0..y1 {
            crop.extend_from_slice(&gray[y * width + x0..y * width + x1]);
        }
        let Some(image) = image::GrayImage::from_raw((x1 - x0) as u32, (y1 - y0) as u32, crop) else {
            return Ok(Vec::new());
        };
        let lines = self.recognize_images(std::slice::from_ref(&image))?.pop().unwrap_or_default();
        Ok(lines
            .into_iter()
            .map(|mut line| {
                let place = |b: [f64; 4]| {
                    [(b[0] + x0 as f64) / scale, (b[1] + y0 as f64) / scale, (b[2] + x0 as f64) / scale, (b[3] + y0 as f64) / scale]
                };
                line.bbox = place(line.bbox);
                for word in &mut line.words {
                    word.bbox = place(word.bbox);
                }
                line
            })
            .collect())
    }
}

/// The scale a page's head and foot are read again at: the recognizer's own (Indra reads at 144 dpi).
pub const REREAD_SCALE: f64 = 2.0;

/// `indices`' heads and feet (`legal_pdf_pages::EDGE_BAND`) read by `ocr` from their images, the
/// document parsed once and each page rendered as it is read.
pub fn read_bands(pdf: &[u8], indices: &[usize], ocr: &mut OcrProvider) -> Result<Vec<(usize, legal_pdf_pages::Bands)>> {
    use legal_pdf_pages::{Bands, EdgeLine, EDGE_BAND};
    let Ok(document) = open_pdf(pdf) else { return Ok(Vec::new()) };
    let pages = PageRenderer::new(&document, REREAD_SCALE * 72.0);
    let mut out = Vec::new();
    for &index in indices {
        let Some(render) = pages.render(index).map(|pixmap| gray_render(&pixmap)) else { continue };
        let (width, height) = (render.0 as f64 / REREAD_SCALE, render.1 as f64 / REREAD_SCALE);
        let mut read = |band: [f64; 4]| -> Result<Vec<EdgeLine>> {
            Ok(ocr.read_rendered_region(&render, band, REREAD_SCALE)?.into_iter().filter(|line| !line.text.trim().is_empty())
                .map(|line| EdgeLine { text: line.text, rect: line.bbox }).collect())
        };
        let head = read([0.0, 0.0, width, height * EDGE_BAND])?;
        let foot = read([0.0, height * (1.0 - EDGE_BAND), width, height])?;
        out.push((index, Bands { width, height, head, foot }));
    }
    Ok(out)
}

/// One page of `pdf` rendered to 8-bit grey at `dpi`.
pub fn render_gray_page(pdf: &[u8], page_index: usize, dpi: f64) -> Option<GrayRender> {
    let document = open_pdf(pdf).ok()?;
    PageRenderer::new(&document, dpi).render(page_index).map(|pixmap| gray_render(&pixmap))
}

fn gray_render(pixmap: &Pixmap) -> GrayRender {
    (usize::from(pixmap.width()), usize::from(pixmap.height()), gray_of(pixmap))
}

/// A render's pixels in 8-bit grey: what every reader of a page's image reads.
pub(crate) fn gray_of(pixmap: &Pixmap) -> Vec<u8> {
    pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .map(|pixel| ((u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29) / 256) as u8)
        .collect()
}
