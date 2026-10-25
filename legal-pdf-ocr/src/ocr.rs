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
        raster_page_rules(pdf, requests).into_iter().map(|reading| reading.separator).collect()
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

/// Each requested page rendered once, at 300 dpi: the resolution scans are made at, where a bold
/// stem stands two pixels wider than a plain one rather than one (on the structure gold's scans,
/// stem weight told headings from text at an AUC of 0.89 at 300 dpi against 0.83 at 150, and read
/// body lines five times as steadily). The note band's separator and the thin strokes are read at
/// 150 dpi, the 300 dpi render's pixels averaged in fours, as their pixel bounds were set; every
/// requested line's ink, stroke weight and capital height, and its words' stroke weights, at 300.
/// For a scan whose text layer is kept, whose image is read for nothing else.
pub fn raster_page_rules(pdf: &[u8], requests: &[OcrPageRequest]) -> Vec<legal_pdf_core::RasterReading> {
    #[cfg(feature = "ocr")]
    {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};
        use legal_pdf_core::RasterReading;

        let Ok(document) = Pdf::new(pdf.to_vec()) else {
            return vec![RasterReading::default(); requests.len()];
        };
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        let scale = 300.0 / 72.0;
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
                    return RasterReading::default();
                };
                let pixmap = render(page, &cache, &interpreter, &settings);
                if pixmap.width() < 2 || pixmap.height() < 2 {
                    return RasterReading::default();
                }
                let gray = gray_of(&pixmap);
                let (width, height) = (usize::from(pixmap.width()), usize::from(pixmap.height()));
                let (sx, sy) = (request.width / width as f64, request.height / height as f64);
                let (half, half_width, half_height) = halved(&gray, width, height);
                let (hx, hy) = (request.width / half_width as f64, request.height / half_height as f64);
                let render = Render { gray: &gray, width, height, sx, sy };
                RasterReading {
                    separator: raster_separator_gray(&half, half_width, half_height, request.height),
                    strokes: thin_strokes(&half, half_width, half_height, hx, hy),
                    inks: request.lines.iter().map(|b| render.ink(b)).collect(),
                    weights: request.lines.iter().map(|b| render.weight(b)).collect(),
                    cap_heights: request.lines.iter().map(|b| render.cap_height(b)).collect(),
                    word_weights: request.words.iter().map(|words| words.iter().map(|b| render.weight(b)).collect()).collect(),
                }
            })
            .collect()
    }
    #[cfg(not(feature = "ocr"))]
    {
        vec![legal_pdf_core::RasterReading::default(); requests.len()]
    }
}

/// A grey render with the points each of its pixels spans (`sx`, `sy`).
#[cfg(feature = "ocr")]
struct Render<'a> {
    gray: &'a [u8],
    width: usize,
    height: usize,
    sx: f64,
    sy: f64,
}

#[cfg(feature = "ocr")]
impl Render<'_> {
    const DARK: u8 = 128;

    /// A box [x0, y0, x1, y1] in points as the pixel rows and columns it covers, if any.
    fn pixels(&self, b: &[f64; 4]) -> Option<(usize, usize, usize, usize)> {
        let x0 = ((b[0] / self.sx).floor().max(0.0) as usize).min(self.width);
        let x1 = ((b[2] / self.sx).ceil().max(0.0) as usize).min(self.width);
        let y0 = ((b[1] / self.sy).floor().max(0.0) as usize).min(self.height);
        let y1 = ((b[3] / self.sy).ceil().max(0.0) as usize).min(self.height);
        (x1 > x0 && y1 > y0).then_some((x0, x1, y0, y1))
    }

    fn row(&self, y: usize, x0: usize, x1: usize) -> &[u8] {
        &self.gray[y * self.width + x0..y * self.width + x1]
    }

    /// The box's share of dark pixels; a measured box reads at least 0.0001, an empty one 0.
    fn ink(&self, b: &[f64; 4]) -> f64 {
        let Some((x0, x1, y0, y1)) = self.pixels(b) else {
            return 0.0;
        };
        let dark: usize = (y0..y1).map(|y| self.row(y, x0, x1).iter().filter(|&&p| p < Self::DARK).count()).sum();
        (dark as f64 / ((x1 - x0) * (y1 - y0)) as f64).max(0.0001)
    }

    /// The box's capital height in points: from its row profile, the baseline is the lowest row of
    /// the dense core (half the busiest row's dark pixels or more) and the top the highest row with
    /// an eighth of them, so descenders and stray specks count for nothing; 0 for an empty box.
    fn cap_height(&self, b: &[f64; 4]) -> f64 {
        let Some((x0, x1, y0, y1)) = self.pixels(b) else {
            return 0.0;
        };
        let profile = (y0..y1)
            .map(|y| self.row(y, x0, x1).iter().filter(|&&p| p < Self::DARK).count())
            .collect::<Vec<_>>();
        let peak = profile.iter().copied().max().unwrap_or(0);
        let base = profile.iter().rposition(|&count| peak > 0 && count * 2 >= peak);
        let top = profile.iter().position(|&count| peak > 0 && count * 8 >= peak);
        match (top, base) {
            (Some(top), Some(base)) if base >= top => (base + 1 - top) as f64 * self.sy,
            _ => 0.0,
        }
    }

    /// The box's stroke weight: the median horizontal run of dark pixels through it, over the box's
    /// height in pixels. Horizontal runs cross the letters' stems, so the median reads the stem
    /// width, which bold type sets about half again as wide; a measured box reads at least 0.0001,
    /// an empty one 0.
    fn weight(&self, b: &[f64; 4]) -> f64 {
        let Some((x0, x1, y0, y1)) = self.pixels(b) else {
            return 0.0;
        };
        let mut runs: Vec<u32> = Vec::new();
        for y in y0..y1 {
            let mut run = 0u32;
            for &p in self.row(y, x0, x1) {
                if p < Self::DARK {
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
    }
}

/// A grey render at half its resolution, each pixel the mean of the four it covers.
#[cfg(feature = "ocr")]
fn halved(gray: &[u8], width: usize, height: usize) -> (Vec<u8>, usize, usize) {
    let (half_width, half_height) = (width / 2, height / 2);
    let mut out = vec![255u8; half_width * half_height];
    for y in 0..half_height {
        for x in 0..half_width {
            let at = |dx: usize, dy: usize| u32::from(gray[(2 * y + dy) * width + 2 * x + dx]);
            out[y * half_width + x] = ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1)) / 4) as u8;
        }
    }
    (out, half_width, half_height)
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
