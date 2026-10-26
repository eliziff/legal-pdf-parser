//! What a page's render shows beyond its text, measured for the structure stage: the note band's
//! separator, thin strokes (rules and underlines), and each line's ink, stroke weight and capital
//! height. Nothing here reads text, so it is keyed apart from recognition.
use legal_pdf_core::OcrPageRequest;
#[cfg(feature = "ocr")]
use crate::ocr::{gray_of, raster_separator_gray};

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
