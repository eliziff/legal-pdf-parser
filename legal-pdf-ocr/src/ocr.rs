use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, PdfOcrProvider, Result};

#[derive(Debug, Clone)]
pub enum OcrOptions {
    #[cfg(feature = "ocr")]
    Indra(crate::indra::IndraOptions),
}

#[cfg(feature = "ocr")]
impl From<crate::indra::IndraOptions> for OcrOptions {
    fn from(options: crate::indra::IndraOptions) -> Self {
        Self::Indra(options)
    }
}

pub enum OcrProvider {
    #[cfg(feature = "ocr")]
    Indra(crate::indra::IndraOcr),
}

pub struct PreparedOcrProvider {
    provider: PreparedProvider,
}

enum PreparedProvider {
    #[cfg(feature = "ocr")]
    Indra(crate::indra::PreparedIndra),
}

impl OcrProvider {
    pub fn new(options: &OcrOptions) -> Result<Self> {
        Self::from_prepared(options, Self::prepare(options)?)
    }

    pub fn prepare(options: &OcrOptions) -> Result<PreparedOcrProvider> {
        match options {
            #[cfg(feature = "ocr")]
            OcrOptions::Indra(options) => {
                let provider = crate::indra::IndraOcr::prepare(options)?;
                Ok(PreparedOcrProvider {
                    provider: PreparedProvider::Indra(provider),
                })
            }
        }
    }

    pub fn from_prepared(options: &OcrOptions, prepared: PreparedOcrProvider) -> Result<Self> {
        match (options, prepared.provider) {
            #[cfg(feature = "ocr")]
            (OcrOptions::Indra(options), PreparedProvider::Indra(prepared)) => {
                crate::indra::IndraOcr::from_prepared(options, prepared).map(Self::Indra)
            }
        }
    }

    pub fn identity(&self) -> &str {
        match self {
            #[cfg(feature = "ocr")]
            Self::Indra(provider) => provider.identity(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            #[cfg(feature = "ocr")]
            Self::Indra(provider) => provider.name(),
        }
    }

    fn extract_pages_inner(
        &mut self,
        pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        match self {
            #[cfg(feature = "ocr")]
            Self::Indra(provider) => provider.extract_pages(pdf, requests),
        }
    }
}

impl PreparedOcrProvider {
    pub fn identity(&self) -> &str {
        match &self.provider {
            #[cfg(feature = "ocr")]
            PreparedProvider::Indra(provider) => provider.identity(),
        }
    }

    pub fn name(&self) -> &str {
        match &self.provider {
            #[cfg(feature = "ocr")]
            PreparedProvider::Indra(provider) => provider.name(),
        }
    }
}

impl PdfOcrProvider for OcrProvider {
    fn extract_pages(
        &mut self,
        pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        self.extract_pages_inner(pdf, requests)
    }

    fn separators(&mut self, pdf: &[u8], requests: &[OcrPageRequest]) -> Vec<Option<f64>> {
        crate::raster::raster_page_rules(pdf, requests).into_iter().map(|reading| reading.separator).collect()
    }

    fn read_region(&mut self, pdf: &[u8], page_index: usize, region: [f64; 4], scale: f64) -> Result<Vec<OcrLine>> {
        #[cfg(feature = "ocr")]
        {
            match render_gray_page(pdf, page_index, scale * 72.0) {
                Some(render) => self.read_rendered_region(&render, region, scale),
                None => Ok(Vec::new()),
            }
        }
        #[cfg(not(feature = "ocr"))]
        {
            let _ = (pdf, page_index, region, scale);
            Ok(Vec::new())
        }
    }
}

/// A page rendered to 8-bit grey: (width, height, pixels).
#[cfg(feature = "ocr")]
pub type GrayRender = (usize, usize, Vec<u8>);

#[cfg(feature = "ocr")]
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
        let lines = match self {
            Self::Indra(provider) => provider.recognize_gray_image(&image)?,
        };
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

/// One page of `pdf` rendered to 8-bit grey at `dpi`.
#[cfg(feature = "ocr")]
pub fn render_gray_page(pdf: &[u8], page_index: usize, dpi: f64) -> Option<GrayRender> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::vello_cpu::color::palette::css::WHITE;
    use hayro::{render, RenderCache, RenderSettings};

    let document = Pdf::new(pdf.to_vec()).ok()?;
    let cache = RenderCache::new();
    let interpreter = InterpreterSettings::default();
    let scale = (dpi / 72.0) as f32;
    let settings = RenderSettings { x_scale: scale, y_scale: scale, bg_color: WHITE, ..Default::default() };
    let pages = document.pages();
    let page = pages.iter().nth(page_index)?;
    let pixmap = render(page, &cache, &interpreter, &settings);
    (pixmap.width() > 0 && pixmap.height() > 0).then(|| (usize::from(pixmap.width()), usize::from(pixmap.height()), gray_of(&pixmap)))
}

#[cfg(feature = "ocr")]
pub(crate) fn gray_of(pixmap: &hayro::vello_cpu::Pixmap) -> Vec<u8> {
    pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .map(|pixel| ((u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29) / 256) as u8)
        .collect()
}

#[cfg(feature = "ocr")]
pub(crate) fn raster_separator_y(pixmap: &hayro::vello_cpu::Pixmap, page_height: f64) -> Option<f64> {
    let gray = gray_of(pixmap);
    raster_separator_gray(&gray, usize::from(pixmap.width()), usize::from(pixmap.height()), page_height)
}

#[cfg(feature = "ocr")]
pub(crate) fn raster_separator_gray(gray: &[u8], width: usize, height: usize, page_height: f64) -> Option<f64> {
    let record = crate::separator::scan_gray_page(gray, width, height);
    if !matches!(record.separator_status, Some("found" | "found_two_column")) {
        return None;
    }
    record.separators.and_then(|separators| {
        separators
            .first()
            .map(|rule| rule.y_center_ratio * page_height)
    })
}
