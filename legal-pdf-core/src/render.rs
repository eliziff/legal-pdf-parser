//! A PDF's pages drawn on white, for what reads their images: recognition, the layout model and
//! the raster measures.
use crate::{Error, FailureKind, Result};
use hayro::hayro_interpret::font::Glyph;
use hayro::hayro_interpret::{
    interpret_page, BlendMode, ClipPath, Context, Device, GlyphDrawMode, Image, ImageData,
    InterpreterCache, InterpreterSettings, Paint, PathDrawMode, SoftMask, TransformExt,
};
use hayro::hayro_syntax::LoadPdfError;
pub use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::vello_cpu::kurbo::{Affine, BezPath, Rect, Shape};
pub use hayro::vello_cpu::Pixmap;
use hayro::{RenderCache, RenderSettings};

/// A single-image page decoded and scaled directly to 8-bit luma (`gray`) plus optional color
/// mean (`ink`) without full RGBA compositing.
pub struct RenderedGray {
    pub width: u16,
    pub height: u16,
    pub gray: Vec<u8>,
    pub ink: Option<Vec<u8>>,
}

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
    scan_cache: InterpreterCache<'a>,
    interpreter: InterpreterSettings,
    settings: RenderSettings,
}

impl<'a> PageRenderer<'a> {
    pub fn new(pdf: &'a Pdf, dpi: f64) -> Self {
        let scale = (dpi / 72.0) as f32;
        Self {
            pdf,
            cache: RenderCache::new(),
            scan_cache: InterpreterCache::new(),
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

    /// Direct grayscale render for a plain full-page upright raster scan (single image, no paths,
    /// glyphs, soft masks, blending or visible annotations). Returns `None` for any other page so
    /// the caller can fall back to `render(index)`.
    pub fn render_scan_gray(&self, index: usize) -> Option<RenderedGray> {
        let page = self.pdf.pages().get(index)?;
        let (x_scale, y_scale) = (self.settings.x_scale, self.settings.y_scale);
        let (width, height) = page.render_dimensions();
        let pix_w = self.settings.width.unwrap_or(((width * x_scale) as f64).floor() as u16);
        let pix_h = self.settings.height.unwrap_or(((height * y_scale) as f64).floor() as u16);
        if pix_w == 0 || pix_h == 0 {
            return None;
        }
        let initial_transform =
            Affine::scale_non_uniform(f64::from(x_scale), f64::from(y_scale)) * page.initial_transform(true).to_kurbo();
        let mut context = Context::new(
            initial_transform,
            Rect::new(0.0, 0.0, f64::from(pix_w), f64::from(pix_h)),
            &self.scan_cache,
            page.xref(),
            self.interpreter.clone(),
        );
        let mut device = ScanDevice { pix_w, pix_h, aborted: false, result: None };
        interpret_page(page, &mut context, &mut device);
        if device.aborted {
            None
        } else {
            device.result
        }
    }
}

struct ScanDevice {
    pix_w: u16,
    pix_h: u16,
    aborted: bool,
    result: Option<RenderedGray>,
}

impl<'a> Device<'a> for ScanDevice {
    fn set_soft_mask(&mut self, mask: Option<SoftMask<'a>>) {
        if mask.is_some() {
            self.aborted = true;
        }
    }

    fn set_blend_mode(&mut self, blend_mode: BlendMode) {
        if blend_mode != BlendMode::Normal {
            self.aborted = true;
        }
    }

    fn draw_path(&mut self, _: &BezPath, _: Affine, _: &Paint<'a>, _: &PathDrawMode) {
        self.aborted = true;
    }

    fn draw_rect(&mut self, _: &Rect, _: Affine, _: &Paint<'a>, _: &PathDrawMode) {
        self.aborted = true;
    }

    fn push_clip_path(&mut self, clip_path: &ClipPath) {
        let covers_canvas = clip_path.path.as_rect().is_some_and(|rect| {
            rect.x0 <= 0.5
                && rect.y0 <= 0.5
                && rect.x1 >= f64::from(self.pix_w) - 0.5
                && rect.y1 >= f64::from(self.pix_h) - 0.5
        });
        if !covers_canvas {
            self.aborted = true;
        }
    }

    fn push_transparency_group(&mut self, opacity: f32, mask: Option<SoftMask<'a>>, blend_mode: BlendMode) {
        if (opacity - 1.0).abs() > 1e-4 || mask.is_some() || blend_mode != BlendMode::Normal {
            self.aborted = true;
        }
    }

    fn draw_glyph(&mut self, _: &Glyph<'a>, _: Affine, _: Affine, _: &Paint<'a>, _: &GlyphDrawMode) {
        self.aborted = true;
    }

    fn draw_image(&mut self, image: Image<'a, '_>, transform: Affine) {
        if self.aborted || self.result.is_some() {
            self.aborted = true;
            return;
        }
        let Image::Raster(raster) = image else {
            self.aborted = true;
            return;
        };
        let [a, b, c, d, tx, ty] = transform.as_coeffs();
        let rw = f64::from(raster.width());
        let rh = f64::from(raster.height());
        if rw <= 0.0
            || rh <= 0.0
            || b.abs() > 1e-3
            || c.abs() > 1e-3
            || a <= 0.0
            || d <= 0.0
            || tx.abs() > 1.0
            || ty.abs() > 1.0
            || (tx + a * rw - f64::from(self.pix_w)).abs() > 1.0
            || (ty + d * rh - f64::from(self.pix_h)).abs() > 1.0
        {
            self.aborted = true;
            return;
        }
        let dst_w = usize::from(self.pix_w);
        let dst_h = usize::from(self.pix_h);
        let mut decoded = None;
        raster.with_rgba(
            |image_data, alpha| {
                if alpha.is_some() {
                    return;
                }
                decoded = convert_and_resize_scan(image_data, dst_w, dst_h).map(|(gray, ink)| RenderedGray {
                    width: self.pix_w,
                    height: self.pix_h,
                    gray,
                    ink,
                });
            },
            Some((u32::from(self.pix_w), u32::from(self.pix_h))),
        );
        if decoded.is_some() {
            self.result = decoded;
        } else {
            self.aborted = true;
        }
    }

    fn pop_clip_path(&mut self) {}
    fn pop_transparency_group(&mut self) {}
}

fn convert_and_resize_scan(image_data: ImageData, dst_w: usize, dst_h: usize) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
    match image_data {
        ImageData::Luma(luma) => {
            let (src_w, src_h) = (luma.width as usize, luma.height as usize);
            if src_w == 0 || src_h == 0 || luma.data.len() != src_w.saturating_mul(src_h) {
                return None;
            }
            let gray = if src_w == dst_w && src_h == dst_h {
                luma.data
            } else {
                resize_u8_triangle(&luma.data, src_w, src_h, dst_w, dst_h)
            };
            Some((gray, None))
        }
        ImageData::Rgb(rgb) => {
            let (src_w, src_h) = (rgb.width as usize, rgb.height as usize);
            let pixels = src_w.saturating_mul(src_h);
            if src_w == 0 || src_h == 0 || rgb.data.len() != pixels.saturating_mul(3) {
                return None;
            }
            let (src_gray, src_ink) = rgb_to_gray_and_ink(&rgb.data, pixels);
            if src_w == dst_w && src_h == dst_h {
                Some((src_gray, src_ink))
            } else {
                let gray = resize_u8_triangle(&src_gray, src_w, src_h, dst_w, dst_h);
                let ink = src_ink.map(|buf| resize_u8_triangle(&buf, src_w, src_h, dst_w, dst_h));
                Some((gray, ink))
            }
        }
    }
}

fn rgb_to_gray_and_ink(data: &[u8], pixels: usize) -> (Vec<u8>, Option<Vec<u8>>) {
    let mut gray = Vec::with_capacity(pixels);
    for (idx, px) in data.chunks_exact(3).enumerate() {
        let (r, g, b) = (px[0], px[1], px[2]);
        if r != g || g != b {
            let mut ink = Vec::with_capacity(pixels);
            ink.extend_from_slice(&gray);
            for rest in data[idx * 3..].chunks_exact(3) {
                let (rr, gg, bb) = (u32::from(rest[0]), u32::from(rest[1]), u32::from(rest[2]));
                gray.push(((rr * 77 + gg * 150 + bb * 29) >> 8) as u8);
                ink.push(((rr + gg + bb + 1) / 3) as u8);
            }
            return (gray, Some(ink));
        }
        gray.push(r);
    }
    (gray, None)
}

fn build_triangle_q12(src_len: usize, dst_len: usize) -> (Vec<(usize, usize, usize)>, Vec<u16>, usize) {
    let mut ranges = Vec::with_capacity(dst_len);
    let mut weights = Vec::with_capacity(dst_len * 4);
    let mut raw = Vec::new();
    let mut max_taps = 1usize;
    let ratio = src_len as f64 / dst_len as f64;
    let support = ratio.max(1.0);
    let inv_support = 1.0 / support;
    for d in 0..dst_len {
        let center = (d as f64 + 0.5) * ratio - 0.5;
        let left = (center - support).ceil().max(0.0) as usize;
        let right = ((center + support).floor() as isize).min(src_len as isize - 1).max(left as isize) as usize;
        raw.clear();
        let mut sum = 0.0f64;
        for s in left..=right {
            let w = (1.0 - (s as f64 - center).abs() * inv_support).max(0.0);
            raw.push(w);
            sum += w;
        }
        let len = raw.len();
        max_taps = max_taps.max(len);
        let w_off = weights.len();
        if sum > 0.0 {
            let scale = 4096.0 / sum;
            let mut qsum = 0i32;
            let mut peak_idx = 0usize;
            let mut peak_val = -1.0f64;
            for (i, &w) in raw.iter().enumerate() {
                if w > peak_val {
                    peak_val = w;
                    peak_idx = i;
                }
                let q = (w * scale).round() as u16;
                qsum += i32::from(q);
                weights.push(q);
            }
            let diff = 4096 - qsum;
            let corrected = i32::from(weights[w_off + peak_idx]) + diff;
            weights[w_off + peak_idx] = corrected.max(0) as u16;
        } else {
            weights.resize(w_off + len, 0);
            if !weights.is_empty() {
                weights[w_off] = 4096;
            }
        }
        ranges.push((left, len, w_off));
    }
    (ranges, weights, max_taps)
}

fn horiz_filter_row_q4(src_row: &[u8], x_ranges: &[(usize, usize, usize)], x_weights: &[u16], dst_row: &mut [u16]) {
    for (dx, &(x0, len, wo)) in x_ranges.iter().enumerate() {
        let mut sum = 0u32;
        for i in 0..len {
            sum += u32::from(src_row[x0 + i]) * u32::from(x_weights[wo + i]);
        }
        dst_row[dx] = ((sum + 128) >> 8) as u16;
    }
}

fn resize_u8_triangle(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let (x_ranges, x_weights, _) = build_triangle_q12(src_w, dst_w);
    let (y_ranges, y_weights, max_y_taps) = build_triangle_q12(src_h, dst_h);
    let ring_cap = max_y_taps.max(1);
    let mut ring = vec![0u16; ring_cap * dst_w];
    let mut cached_sy = vec![usize::MAX; ring_cap];
    let mut acc = vec![0u32; dst_w];
    let mut out = vec![0u8; dst_w * dst_h];

    for (dy, &(y0, len, wo)) in y_ranges.iter().enumerate() {
        acc.fill(0);
        for (i, &wy) in y_weights[wo..wo + len].iter().enumerate() {
            if wy == 0 {
                continue;
            }
            let sy = y0 + i;
            let slot = sy % ring_cap;
            if cached_sy[slot] != sy {
                let src_row = &src[sy * src_w..(sy + 1) * src_w];
                let ring_row = &mut ring[slot * dst_w..(slot + 1) * dst_w];
                horiz_filter_row_q4(src_row, &x_ranges, &x_weights, ring_row);
                cached_sy[slot] = sy;
            }
            let row = &ring[slot * dst_w..(slot + 1) * dst_w];
            let w = u32::from(wy);
            for (a, &h) in acc.iter_mut().zip(row) {
                *a += u32::from(h) * w;
            }
        }
        let dst_row = &mut out[dy * dst_w..(dy + 1) * dst_w];
        for (d, &a) in dst_row.iter_mut().zip(&acc) {
            *d = ((a + (1 << 15)) >> 16).min(255) as u8;
        }
    }
    out
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
