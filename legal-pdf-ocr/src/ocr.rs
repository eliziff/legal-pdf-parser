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
        raster_page_rules(pdf, requests).into_iter().map(|(separator, _, _, _)| separator).collect()
    }

    fn read_region(&mut self, pdf: &[u8], page_index: usize, region: [f64; 4], scale: f64) -> Result<Vec<OcrLine>> {
        #[cfg(feature = "ocr")]
        {
            let Some((width, height, gray)) = render_gray_page(pdf, page_index, scale * 72.0) else {
                return Ok(Vec::new());
            };
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
        #[cfg(not(feature = "ocr"))]
        {
            let _ = (pdf, page_index, region, scale);
            Ok(Vec::new())
        }
    }
}

/// One page of `pdf` rendered to 8-bit grey at `dpi`: (width, height, pixels).
#[cfg(feature = "ocr")]
fn render_gray_page(pdf: &[u8], page_index: usize, dpi: f64) -> Option<(usize, usize, Vec<u8>)> {
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

/// The rule printed above each requested page's notes, read from the page rendered at 150 dpi
/// and scanned as the recognizers scan their own renders: for a scan whose text layer is kept,
/// whose image is read for nothing else.
/// Each requested page rendered once: the note band's separator rule, every thin horizontal
/// stroke on it ([x0, y, x1, y] in page points), which underlines and rules read, the dark
/// share of each requested line box, and each box's stroke weight.
pub type RasterRules = (Option<f64>, Vec<[f64; 4]>, Vec<f64>, Vec<f64>);

pub fn raster_page_rules(pdf: &[u8], requests: &[OcrPageRequest]) -> Vec<RasterRules> {
    #[cfg(feature = "ocr")]
    {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};

        let Ok(document) = Pdf::new(pdf.to_vec()) else {
            return vec![(None, Vec::new(), Vec::new(), Vec::new()); requests.len()];
        };
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        let scale = 150.0 / 72.0;
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: WHITE,
            ..Default::default()
        };
        let pages = document.pages();
        requests
            .iter()
            .map(|request| {
                let Some(page) = pages.iter().nth(request.page_index) else {
                    return (None, Vec::new(), Vec::new(), Vec::new());
                };
                let pixmap = render(page, &cache, &interpreter, &settings);
                if pixmap.width() == 0 || pixmap.height() == 0 {
                    return (None, Vec::new(), Vec::new(), Vec::new());
                }
                let gray = gray_of(&pixmap);
                let (width, height) = (usize::from(pixmap.width()), usize::from(pixmap.height()));
                let (sx, sy) = (request.width / width as f64, request.height / height as f64);
                (
                    raster_separator_gray(&gray, width, height, request.height),
                    thin_strokes(&gray, width, height, sx, sy),
                    line_inks(&gray, width, height, sx, sy, &request.lines),
                    line_weights(&gray, width, height, sx, sy, &request.lines),
                )
            })
            .collect()
    }
    #[cfg(not(feature = "ocr"))]
    {
        vec![(None, Vec::new(), Vec::new(), Vec::new()); requests.len()]
    }
}

/// Each box's stroke weight: the median horizontal run of dark pixels through it, over the box's
/// height in pixels (boxes [x0, y0, x1, y1] in points, `sx` and `sy` points per pixel). Horizontal
/// runs cross the letters' stems, so the median reads the stem width, which bold type sets about
/// half again as wide; a measured box reads at least 0.0001, an empty one 0.
#[cfg(feature = "ocr")]
fn line_weights(gray: &[u8], width: usize, height: usize, sx: f64, sy: f64, boxes: &[[f64; 4]]) -> Vec<f64> {
    const DARK: u8 = 128;
    boxes
        .iter()
        .map(|b| {
            let x0 = ((b[0] / sx).floor().max(0.0) as usize).min(width);
            let x1 = ((b[2] / sx).ceil().max(0.0) as usize).min(width);
            let y0 = ((b[1] / sy).floor().max(0.0) as usize).min(height);
            let y1 = ((b[3] / sy).ceil().max(0.0) as usize).min(height);
            if x1 <= x0 || y1 <= y0 {
                return 0.0;
            }
            let mut runs: Vec<u32> = Vec::new();
            for y in y0..y1 {
                let mut run = 0u32;
                for &p in &gray[y * width + x0..y * width + x1] {
                    if p < DARK {
                        run += 1;
                    } else if run > 0 {
                        runs.push(run);
                        run = 0;
                    }
                }
                if run > 0 {
                    runs.push(run);
                }
            }
            if runs.is_empty() {
                return 0.0001;
            }
            runs.sort_unstable();
            (f64::from(runs[runs.len() / 2]) / (y1 - y0) as f64).max(0.0001)
        })
        .collect()
}

/// Every page of `pdf` rendered to 8-bit grey at `dpi`: (width, height, pixels) per page, in page
/// order. The structure gold's synthetic scans are made of these renders.
pub fn render_gray_pages(pdf: &[u8], dpi: f64) -> Vec<(usize, usize, Vec<u8>)> {
    #[cfg(feature = "ocr")]
    {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};

        let Ok(document) = Pdf::new(pdf.to_vec()) else {
            return Vec::new();
        };
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        let scale = (dpi / 72.0) as f32;
        let settings = RenderSettings { x_scale: scale, y_scale: scale, bg_color: WHITE, ..Default::default() };
        document
            .pages()
            .iter()
            .map(|page| {
                let pixmap = render(page, &cache, &interpreter, &settings);
                (usize::from(pixmap.width()), usize::from(pixmap.height()), gray_of(&pixmap))
            })
            .collect()
    }
    #[cfg(not(feature = "ocr"))]
    {
        let _ = (pdf, dpi);
        Vec::new()
    }
}

/// Each box's share of dark pixels in the render (boxes [x0, y0, x1, y1] in points, `sx` and `sy`
/// points per pixel); a measured box reads at least 0.0001, an empty one 0.
#[cfg(feature = "ocr")]
fn line_inks(gray: &[u8], width: usize, height: usize, sx: f64, sy: f64, boxes: &[[f64; 4]]) -> Vec<f64> {
    const DARK: u8 = 128;
    boxes
        .iter()
        .map(|b| {
            let x0 = ((b[0] / sx).floor().max(0.0) as usize).min(width);
            let x1 = ((b[2] / sx).ceil().max(0.0) as usize).min(width);
            let y0 = ((b[1] / sy).floor().max(0.0) as usize).min(height);
            let y1 = ((b[3] / sy).ceil().max(0.0) as usize).min(height);
            if x1 <= x0 || y1 <= y0 {
                return 0.0;
            }
            let dark: usize = (y0..y1)
                .map(|y| gray[y * width + x0..y * width + x1].iter().filter(|&&p| p < DARK).count())
                .sum();
            (dark as f64 / ((x1 - x0) * (y1 - y0)) as f64).max(0.0001)
        })
        .collect()
}

#[cfg(feature = "ocr")]
fn gray_of(pixmap: &hayro::vello_cpu::Pixmap) -> Vec<u8> {
    pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .map(|pixel| ((u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29) / 256) as u8)
        .collect()
}

/// Thin horizontal strokes: runs of dark pixels (gaps of two bridged) at least ten pixels long,
/// at most four rows thick, with light rows three pixels above and below across most of the run.
#[cfg(feature = "ocr")]
fn thin_strokes(gray: &[u8], width: usize, height: usize, sx: f64, sy: f64) -> Vec<[f64; 4]> {
    const DARK: u8 = 110;
    let dark = |x: usize, y: usize| gray[y * width + x] < DARK;
    let light_share = |y: usize, a: usize, b: usize| (a..b).filter(|&x| !dark(x, y)).count() * 10 >= (b - a) * 7;
    let mut strokes: Vec<(usize, usize, usize, usize)> = Vec::new();
    for y in 3..height.saturating_sub(3) {
        let mut x = 0;
        while x < width {
            if !dark(x, y) {
                x += 1;
                continue;
            }
            let start = x;
            let mut gap = 0;
            let mut end = x;
            while x < width && gap <= 2 {
                if dark(x, y) { end = x; gap = 0 } else { gap += 1 }
                x += 1;
            }
            if end + 1 - start >= 10 && light_share(y - 3, start, end + 1) && light_share(y + 3, start, end + 1) {
                // A row just under one already read is the same stroke, thicker.
                if let Some(last) = strokes.iter_mut().rev().take(64).find(|s| s.3 + 1 == y && s.0 < end && start < s.1) {
                    last.0 = last.0.min(start); last.1 = last.1.max(end); last.3 = y;
                } else {
                    strokes.push((start, end, y, y));
                }
            }
        }
    }
    strokes
        .into_iter()
        .filter(|s| s.3 - s.2 <= 3)
        .map(|(a, b, top, bottom)| {
            let y = (top + bottom) as f64 / 2.0 * sy;
            [a as f64 * sx, y, (b + 1) as f64 * sx, y]
        })
        .collect()
}

#[cfg(feature = "ocr")]
pub(crate) fn raster_separator_y(pixmap: &hayro::vello_cpu::Pixmap, page_height: f64) -> Option<f64> {
    let gray = gray_of(pixmap);
    raster_separator_gray(&gray, usize::from(pixmap.width()), usize::from(pixmap.height()), page_height)
}

#[cfg(feature = "ocr")]
fn raster_separator_gray(gray: &[u8], width: usize, height: usize, page_height: f64) -> Option<f64> {
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
