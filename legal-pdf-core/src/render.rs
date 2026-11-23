//! A PDF's pages drawn on white, for what reads their images: recognition, the layout model and
//! the raster measures.
use crate::{Error, FailureKind, Result};
use hayro::hayro_interpret::font::Glyph;
use hayro::hayro_interpret::{
    interpret_page, BlendMode, ClipPath, Context, Device, GlyphDrawMode, Image, ImageData,
    InterpreterCache, InterpreterSettings, Paint, PathDrawMode, RasterImage, SoftMask,
    TransformExt,
};
use hayro::hayro_syntax::object::dict::keys::*;
use hayro::hayro_syntax::object::{Name, Object};
use hayro::hayro_syntax::LoadPdfError;
pub use hayro::hayro_syntax::Pdf;
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::vello_cpu::kurbo::{Affine, BezPath, Rect, Shape};
pub use hayro::vello_cpu::Pixmap;
use hayro::{RenderCache, RenderSettings};
use std::io::Read;

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
        let mut decoded = try_decode_flate_scan(&raster, dst_w, dst_h).map(|(gray, ink)| RenderedGray {
            width: self.pix_w,
            height: self.pix_h,
            gray,
            ink,
        });
        if decoded.is_none() {
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
        }
        if decoded.is_some() {
            self.result = decoded;
        } else {
            self.aborted = true;
        }
    }

    fn pop_clip_path(&mut self) {}
    fn pop_transparency_group(&mut self) {}
}

fn single_name<'a>(obj: Object<'a>) -> Option<Name<'a>> {
    match obj {
        Object::Name(name) => Some(name),
        Object::Array(arr) => {
            let mut raw = arr.raw_iter();
            if raw.next().is_some() && raw.next().is_none() {
                arr.iter::<Name<'_>>().next()
            } else {
                None
            }
        }
        _ => None,
    }
}

fn try_decode_flate_scan(
    raster: &RasterImage<'_>,
    dst_w: usize,
    dst_h: usize,
) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
    let src_w = usize::try_from(raster.width()).ok()?;
    let src_h = usize::try_from(raster.height()).ok()?;
    if src_w == 0 || src_h == 0 {
        return None;
    }
    let stream = raster.stream();
    let dict = stream.dict();
    let filter_obj = dict.get::<Object<'_>>(F).or_else(|| dict.get::<Object<'_>>(FILTER))?;
    let filter_name = single_name(filter_obj)?;
    if !matches!(filter_name.as_ref(), FLATE_DECODE | FLATE_DECODE_ABBREVIATION) {
        return None;
    }
    if dict.contains_key(DP)
        || dict.contains_key(DECODE_PARMS)
        || dict.contains_key(F_DECODE_PARMS)
        || dict.contains_key(SMASK)
        || dict.contains_key(SMASK_IN_DATA)
        || dict.contains_key(MASK)
        || dict.contains_key(DECODE)
        || dict.contains_key(D)
    {
        return None;
    }
    if dict.get::<bool>(IM).or_else(|| dict.get::<bool>(IMAGE_MASK)).unwrap_or(false) {
        return None;
    }
    let bpc = dict.get::<u8>(BPC).or_else(|| dict.get::<u8>(BITS_PER_COMPONENT))?;
    if bpc != 8 {
        return None;
    }
    let cs_obj = dict.get::<Object<'_>>(CS).or_else(|| dict.get::<Object<'_>>(COLORSPACE))?;
    let cs_name = single_name(cs_obj)?;
    let channels = match cs_name.as_ref() {
        DEVICE_RGB | RGB => 3usize,
        DEVICE_GRAY | G => 1usize,
        _ => return None,
    };
    let pixels = src_w.checked_mul(src_h)?;
    let expected_len = pixels.checked_mul(channels)?;
    let raw = stream.raw_data();
    let mut data = Vec::with_capacity(expected_len);
    let mut decoder = flate2::read::ZlibDecoder::new(raw.as_ref()).take(expected_len as u64 + 1);
    if decoder.read_to_end(&mut data).is_err() || data.len() != expected_len {
        return None;
    }
    if channels == 1 {
        Some(finish_luma_scan(data, src_w, src_h, dst_w, dst_h))
    } else {
        Some(finish_rgb_scan(&data, src_w, src_h, dst_w, dst_h))
    }
}

fn finish_luma_scan(
    data: Vec<u8>,
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let gray = if src_w == dst_w && src_h == dst_h {
        data
    } else {
        resize_u8_triangle(&data, src_w, src_h, dst_w, dst_h)
    };
    (gray, None)
}

fn finish_rgb_scan(
    data: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> (Vec<u8>, Option<Vec<u8>>) {
    let pixels = src_w * src_h;
    let (src_gray, src_ink) = rgb_to_gray_and_ink(data, pixels);
    if src_w == dst_w && src_h == dst_h {
        (src_gray, src_ink)
    } else {
        let gray = resize_u8_triangle(&src_gray, src_w, src_h, dst_w, dst_h);
        let ink = src_ink.map(|buf| resize_u8_triangle(&buf, src_w, src_h, dst_w, dst_h));
        (gray, ink)
    }
}

fn convert_and_resize_scan(image_data: ImageData, dst_w: usize, dst_h: usize) -> Option<(Vec<u8>, Option<Vec<u8>>)> {
    match image_data {
        ImageData::Luma(luma) => {
            let (src_w, src_h) = (luma.width as usize, luma.height as usize);
            if src_w == 0 || src_h == 0 || luma.data.len() != src_w.saturating_mul(src_h) {
                return None;
            }
            Some(finish_luma_scan(luma.data, src_w, src_h, dst_w, dst_h))
        }
        ImageData::Rgb(rgb) => {
            let (src_w, src_h) = (rgb.width as usize, rgb.height as usize);
            let pixels = src_w.saturating_mul(src_h);
            if src_w == 0 || src_h == 0 || rgb.data.len() != pixels.saturating_mul(3) {
                return None;
            }
            Some(finish_rgb_scan(&rgb.data, src_w, src_h, dst_w, dst_h))
        }
    }
}

fn rgb_to_gray_and_ink(data: &[u8], pixels: usize) -> (Vec<u8>, Option<Vec<u8>>) {
    assert!(data.len() >= pixels.saturating_mul(3));
    let mut gray = vec![0u8; pixels];
    let mut idx = 0usize;
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        idx = unsafe { rgb_to_gray_mono_prefix_avx2(data, &mut gray) };
    }
    while idx < pixels {
        let off = idx * 3;
        let (r, g, b) = (data[off], data[off + 1], data[off + 2]);
        if r != g || g != b {
            let mut ink = vec![0u8; pixels];
            ink[..idx].copy_from_slice(&gray[..idx]);
            #[cfg(target_arch = "x86_64")]
            if is_x86_feature_detected!("avx2") {
                idx = unsafe { rgb_to_gray_and_ink_color_avx2(data, idx, &mut gray, &mut ink) };
            }
            for k in idx..pixels {
                let p = k * 3;
                let (rr, gg, bb) = (u32::from(data[p]), u32::from(data[p + 1]), u32::from(data[p + 2]));
                gray[k] = ((rr * 77 + gg * 150 + bb * 29) >> 8) as u8;
                ink[k] = ((rr + gg + bb + 1) / 3) as u8;
            }
            return (gray, Some(ink));
        }
        gray[idx] = r;
        idx += 1;
    }
    (gray, None)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn deinterleave_rgb16_avx2(
    src_ptr: *const u8,
) -> (std::arch::x86_64::__m128i, std::arch::x86_64::__m128i, std::arch::x86_64::__m128i) {
    use std::arch::x86_64::*;
    let mr0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mr1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
    let mr2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
    let mg0 = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mg1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
    let mg2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
    let mb0 = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    let mb1 = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
    let mb2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
    let p0 = _mm_loadu_si128(src_ptr as *const __m128i);
    let p1 = _mm_loadu_si128(src_ptr.add(16) as *const __m128i);
    let p2 = _mm_loadu_si128(src_ptr.add(32) as *const __m128i);
    let r = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(p0, mr0), _mm_shuffle_epi8(p1, mr1)), _mm_shuffle_epi8(p2, mr2));
    let g = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(p0, mg0), _mm_shuffle_epi8(p1, mg1)), _mm_shuffle_epi8(p2, mg2));
    let b = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(p0, mb0), _mm_shuffle_epi8(p1, mb1)), _mm_shuffle_epi8(p2, mb2));
    (r, g, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_gray_mono_prefix_avx2(data: &[u8], gray: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let mut i = 0usize;
    let n = gray.len();
    let ptr = data.as_ptr();
    let gptr = gray.as_mut_ptr();
    while i + 16 <= n {
        let (r, g, b) = deinterleave_rgb16_avx2(ptr.add(i * 3));
        let eq = _mm_and_si128(_mm_cmpeq_epi8(r, g), _mm_cmpeq_epi8(g, b));
        if _mm_movemask_epi8(eq) != 0xFFFF {
            break;
        }
        _mm_storeu_si128(gptr.add(i) as *mut __m128i, r);
        i += 16;
    }
    i
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rgb_to_gray_and_ink_color_avx2(data: &[u8], start: usize, gray: &mut [u8], ink: &mut [u8]) -> usize {
    use std::arch::x86_64::*;
    let w_r = _mm256_set1_epi16(77);
    let w_g = _mm256_set1_epi16(150);
    let w_b = _mm256_set1_epi16(29);
    let ones = _mm256_set1_epi16(1);
    let div3_mul = _mm256_set1_epi16(21846);
    let n = gray.len();
    let mut i = start;
    let ptr = data.as_ptr();
    let gptr = gray.as_mut_ptr();
    let iptr = ink.as_mut_ptr();
    while i + 16 <= n {
        let (r, g, b) = deinterleave_rgb16_avx2(ptr.add(i * 3));
        let r16 = _mm256_cvtepu8_epi16(r);
        let g16 = _mm256_cvtepu8_epi16(g);
        let b16 = _mm256_cvtepu8_epi16(b);

        let gray16 = _mm256_srli_epi16::<8>(_mm256_add_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(r16, w_r), _mm256_mullo_epi16(g16, w_g)),
            _mm256_mullo_epi16(b16, w_b),
        ));
        let gray8 = _mm_packus_epi16(_mm256_castsi256_si128(gray16), _mm256_extracti128_si256::<1>(gray16));
        _mm_storeu_si128(gptr.add(i) as *mut __m128i, gray8);

        let sum16 = _mm256_add_epi16(_mm256_add_epi16(r16, g16), _mm256_add_epi16(b16, ones));
        let ink16 = _mm256_mulhi_epu16(sum16, div3_mul);
        let ink8 = _mm_packus_epi16(_mm256_castsi256_si128(ink16), _mm256_extracti128_si256::<1>(ink16));
        _mm_storeu_si128(iptr.add(i) as *mut __m128i, ink8);

        i += 16;
    }
    i
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

fn horiz_filter_row_q4(
    src_row: &[u8],
    x_ranges: &[(usize, usize, usize)],
    x_weights: &[u16],
    x8_packed: Option<&[i16]>,
    avx_end: usize,
    dst_row: &mut [u16],
) {
    #[cfg(target_arch = "x86_64")]
    if let Some(w8) = x8_packed {
        if is_x86_feature_detected!("avx2") {
            unsafe { horiz_filter_row_q4_avx2(src_row, x_ranges, w8, avx_end, dst_row) };
            for (dx, &(x0, len, wo)) in x_ranges.iter().enumerate().skip(avx_end) {
                let mut sum = 0u32;
                for i in 0..len {
                    sum += u32::from(src_row[x0 + i]) * u32::from(x_weights[wo + i]);
                }
                dst_row[dx] = ((sum + 128) >> 8) as u16;
            }
            return;
        }
    }
    let _ = (x8_packed, avx_end);
    for (dx, &(x0, len, wo)) in x_ranges.iter().enumerate() {
        let mut sum = 0u32;
        for i in 0..len {
            sum += u32::from(src_row[x0 + i]) * u32::from(x_weights[wo + i]);
        }
        dst_row[dx] = ((sum + 128) >> 8) as u16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horiz_filter_row_q4_avx2(
    src_row: &[u8],
    x_ranges: &[(usize, usize, usize)],
    w8: &[i16],
    avx_end: usize,
    dst_row: &mut [u16],
) {
    use std::arch::x86_64::*;
    let sptr = src_row.as_ptr();
    let wptr = w8.as_ptr();
    let dptr = dst_row.as_mut_ptr();
    for dx in 0..avx_end {
        let x0 = x_ranges.get_unchecked(dx).0;
        let bytes = _mm_loadl_epi64(sptr.add(x0) as *const __m128i);
        let px16 = _mm_cvtepu8_epi16(bytes);
        let wt16 = _mm_loadu_si128(wptr.add(dx * 8) as *const __m128i);
        let prod = _mm_madd_epi16(px16, wt16);
        let h1 = _mm_hadd_epi32(prod, prod);
        let sum = _mm_cvtsi128_si32(h1) + _mm_extract_epi32::<1>(h1);
        *dptr.add(dx) = ((sum + 128) >> 8) as u16;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn accum_row_q4_avx2(acc: &mut [u32], row: &[u16], w: u16) {
    use std::arch::x86_64::*;
    let vw = _mm256_set1_epi32(i32::from(w));
    let len = acc.len();
    let mut i = 0usize;
    let aptr = acc.as_mut_ptr();
    let rptr = row.as_ptr();
    while i + 8 <= len {
        let vh = _mm256_cvtepu16_epi32(_mm_loadu_si128(rptr.add(i) as *const __m128i));
        let va = _mm256_loadu_si256(aptr.add(i) as *const __m256i);
        let next = _mm256_add_epi32(va, _mm256_mullo_epi32(vh, vw));
        _mm256_storeu_si256(aptr.add(i) as *mut __m256i, next);
        i += 8;
    }
    let wu = u32::from(w);
    while i < len {
        *aptr.add(i) += u32::from(*rptr.add(i)) * wu;
        i += 1;
    }
}

#[inline]
fn accum_row_q4(acc: &mut [u32], row: &[u16], w: u16) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { accum_row_q4_avx2(acc, row, w) };
        return;
    }
    let wu = u32::from(w);
    for (a, &h) in acc.iter_mut().zip(row) {
        *a += u32::from(h) * wu;
    }
}

fn resize_u8_triangle(src: &[u8], src_w: usize, src_h: usize, dst_w: usize, dst_h: usize) -> Vec<u8> {
    let (x_ranges, x_weights, max_x_taps) = build_triangle_q12(src_w, dst_w);
    let (y_ranges, y_weights, max_y_taps) = build_triangle_q12(src_h, dst_h);
    let mut x8_packed_buf = Vec::new();
    let mut avx_end = 0usize;
    let x8_packed = if max_x_taps <= 8 && src_w >= 8 {
        x8_packed_buf.resize(dst_w * 8, 0i16);
        for (dx, &(x0, len, wo)) in x_ranges.iter().enumerate() {
            for i in 0..len {
                x8_packed_buf[dx * 8 + i] = x_weights[wo + i] as i16;
            }
            if x0 + 8 <= src_w && avx_end == dx {
                avx_end = dx + 1;
            }
        }
        Some(x8_packed_buf.as_slice())
    } else {
        None
    };

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
                horiz_filter_row_q4(src_row, &x_ranges, &x_weights, x8_packed, avx_end, ring_row);
                cached_sy[slot] = sy;
            }
            let row = &ring[slot * dst_w..(slot + 1) * dst_w];
            accum_row_q4(&mut acc, row, wy);
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

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::ZlibEncoder;
    use flate2::Compression;
    use std::io::Write;

    #[test]
    fn rgb_to_gray_and_ink_avx2_matches_scalar_exact() {
        for n in 0u32..=766 {
            assert_eq!((n * 21846) >> 16, n / 3, "div3 mismatch at n={n}");
        }

        let pixels = 73usize;
        let mut rgb = Vec::with_capacity(pixels * 3);
        // 19-pixel monochrome prefix (crosses 16-byte SIMD boundary + 3 scalar mono pixels) then color tail.
        for i in 0..pixels {
            if i < 19 {
                let v = ((i * 13) % 256) as u8;
                rgb.extend_from_slice(&[v, v, v]);
            } else {
                let r = ((i * 73 + 17) % 256) as u8;
                let g = ((i * 151 + 29) % 256) as u8;
                let b = ((i * 211 + 5) % 256) as u8;
                rgb.extend_from_slice(&[r, g, b]);
            }
        }
        // Overwrite a few pixels with extreme RGB corners to test u16 overflow/shift edge cases.
        for (idx, (r, g, b)) in [
            (20, (255u8, 255, 254)),
            (21, (255, 0, 0)),
            (22, (0, 255, 0)),
            (23, (0, 0, 255)),
            (24, (1, 0, 0)),
            (72, (254, 255, 255)),
        ] {
            rgb[idx * 3] = r;
            rgb[idx * 3 + 1] = g;
            rgb[idx * 3 + 2] = b;
        }

        let (gray, ink) = rgb_to_gray_and_ink(&rgb, pixels);
        let ink = ink.expect("color pixels should produce ink channel");
        for k in 0..pixels {
            let (rr, gg, bb) = (u32::from(rgb[k * 3]), u32::from(rgb[k * 3 + 1]), u32::from(rgb[k * 3 + 2]));
            let exp_gray = ((rr * 77 + gg * 150 + bb * 29) >> 8) as u8;
            let exp_ink = ((rr + gg + bb + 1) / 3) as u8;
            assert_eq!(gray[k], exp_gray, "gray mismatch at pixel {k}");
            assert_eq!(ink[k], exp_ink, "ink mismatch at pixel {k}");
        }

        let mono: Vec<u8> = (0..pixels).flat_map(|i| {
            let v = (i * 7 % 256) as u8;
            [v, v, v]
        }).collect();
        let (mono_gray, mono_ink) = rgb_to_gray_and_ink(&mono, pixels);
        assert!(mono_ink.is_none(), "monochrome image should skip ink buffer");
        for k in 0..pixels {
            assert_eq!(mono_gray[k], mono[k * 3]);
        }
    }

    #[test]
    fn flate_scan_fast_path_decodes_devicergb_scan() {
        let (w, h) = (32usize, 24usize);
        let mut raw_rgb = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                let v = (((x + y) * 9) % 256) as u8;
                raw_rgb.extend_from_slice(&[v, v.saturating_add(3), v]);
            }
        }
        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&raw_rgb).unwrap();
        let compressed = enc.finish().unwrap();

        let content = b"q 32 0 0 24 0 0 cm /Im0 Do Q\n";
        let mut pdf_bytes = Vec::new();
        pdf_bytes.extend_from_slice(b"%PDF-1.4\n");
        let mut offsets = vec![0usize];
        let mut push_obj = |id: usize, body: &[u8]| {
            offsets.push(pdf_bytes.len());
            pdf_bytes.extend_from_slice(format!("{id} 0 obj\n").as_bytes());
            pdf_bytes.extend_from_slice(body);
            pdf_bytes.extend_from_slice(b"\nendobj\n");
        };
        push_obj(1, b"<< /Type /Catalog /Pages 2 0 R >>");
        push_obj(2, b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
        push_obj(
            3,
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 32 24] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>",
        );
        push_obj(
            4,
            format!("<< /Length {} >>\nstream\n{}endstream", content.len(), String::from_utf8_lossy(content)).as_bytes(),
        );
        let mut img_obj = format!(
            "<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /FlateDecode /Length {} >>\nstream\n",
            compressed.len()
        ).into_bytes();
        img_obj.extend_from_slice(&compressed);
        img_obj.extend_from_slice(b"\nendstream");
        push_obj(5, &img_obj);

        let xref_pos = pdf_bytes.len();
        pdf_bytes.extend_from_slice(format!("xref\n0 {}\n0000000000 65535 f \n", offsets.len()).as_bytes());
        for off in &offsets[1..] {
            pdf_bytes.extend_from_slice(format!("{off:010} 00000 n \n").as_bytes());
        }
        pdf_bytes.extend_from_slice(format!("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n", offsets.len()).as_bytes());

        let pdf = open_pdf(&pdf_bytes).expect("open synthetic flate scan PDF");
        let renderer = PageRenderer::new(&pdf, 72.0);
        let scan = renderer.render_scan_gray(0).expect("should decode via flate scan fast path");
        assert_eq!((usize::from(scan.width), usize::from(scan.height)), (w, h));
        let (exp_gray, exp_ink) = rgb_to_gray_and_ink(&raw_rgb, w * h);
        assert_eq!(scan.gray, exp_gray);
        assert_eq!(scan.ink, exp_ink);
    }
}
