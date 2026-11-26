use image::{imageops, GrayImage};
use legal_pdf_core::OcrLine;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PageProfile {
    pub(crate) turns_90: u8,
    pub(crate) skew_deg: f32,
    pub(crate) white_frac: f32,
    pub(crate) mid_frac: f32,
    pub(crate) line_energy: f32,
}

#[inline]
pub(crate) fn u8_to_norm_f32_row(src: &[u8], dst: &mut [f32]) {
    debug_assert_eq!(src.len(), dst.len());
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { u8_to_norm_f32_row_avx2(src, dst) };
        return;
    }
    const INV255: f32 = 1.0 / 255.0;
    for (&v, out) in src.iter().zip(dst.iter_mut()) {
        *out = 1.0 - f32::from(v) * INV255;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn u8_to_norm_f32_row_avx2(src: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = src.len();
    let one = _mm256_set1_ps(1.0);
    let inv255 = _mm256_set1_ps(1.0 / 255.0);
    let mut i = 0;
    while i + 8 <= len {
        let bytes = _mm_loadl_epi64(src.as_ptr().add(i) as *const __m128i);
        let ints = _mm256_cvtepu8_epi32(bytes);
        let floats = _mm256_cvtepi32_ps(ints);
        let res = _mm256_fnmadd_ps(floats, inv255, one);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), res);
        i += 8;
    }
    const INV255: f32 = 1.0 / 255.0;
    while i < len {
        *dst.get_unchecked_mut(i) = 1.0 - f32::from(*src.get_unchecked(i)) * INV255;
        i += 1;
    }
}

/// Finds the tight bounding box `(x0, y0, x1, y1)` relative to `(left, top)` of pixels `< thresh`.
#[inline]
pub(crate) fn find_ink_bbox(
    raw: &[u8],
    stride: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    thresh: u8,
) -> Option<(usize, usize, usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { find_ink_bbox_avx2(raw, stride, left, top, width, height, thresh) };
    }
    find_ink_bbox_scalar(raw, stride, left, top, width, height, thresh)
}

fn find_ink_bbox_scalar(
    raw: &[u8],
    stride: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    thresh: u8,
) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0usize, 0usize);
    for y in 0..height {
        let row = &raw[(top + y) * stride + left..(top + y) * stride + left + width];
        for (x, &v) in row.iter().enumerate() {
            if v < thresh {
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x + 1);
                y1 = y1.max(y + 1);
            }
        }
    }
    (x1 > x0).then_some((x0, y0, x1, y1))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_ink_bbox_avx2(
    raw: &[u8],
    stride: usize,
    left: usize,
    top: usize,
    width: usize,
    height: usize,
    thresh: u8,
) -> Option<(usize, usize, usize, usize)> {
    use std::arch::x86_64::*;
    let (mut x0, mut y0, mut x1, mut y1) = (width, height, 0usize, 0usize);
    let bias = _mm256_set1_epi8(-128i8);
    let t_biased = _mm256_set1_epi8((thresh as i16 - 128) as i8);
    for y in 0..height {
        let row_ptr = raw.as_ptr().add((top + y) * stride + left);
        let mut x = 0usize;
        let mut row_has_ink = false;
        while x + 32 <= width {
            let v = _mm256_loadu_si256(row_ptr.add(x) as *const __m256i);
            let vb = _mm256_xor_si256(v, bias);
            let lt = _mm256_cmpgt_epi8(t_biased, vb);
            let mask = _mm256_movemask_epi8(lt) as u32;
            if mask != 0 {
                row_has_ink = true;
                let first = mask.trailing_zeros() as usize;
                let last = 31 - mask.leading_zeros() as usize;
                x0 = x0.min(x + first);
                x1 = x1.max(x + last + 1);
            }
            x += 32;
        }
        while x < width {
            if *row_ptr.add(x) < thresh {
                row_has_ink = true;
                x0 = x0.min(x);
                x1 = x1.max(x + 1);
            }
            x += 1;
        }
        if row_has_ink {
            y0 = y0.min(y);
            y1 = y1.max(y + 1);
        }
    }
    (x1 > x0).then_some((x0, y0, x1, y1))
}

/// Zero-allocation bilinear resize directly from page `(raw, stride)` with virtual 255-white padding `(px, py)`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resize_cropped_padded_bilinear(
    raw: &[u8],
    stride: usize,
    src_x0: usize,
    src_y0: usize,
    cw: usize,
    ch: usize,
    px: usize,
    py: usize,
    dst_w: usize,
    dst_h: usize,
    scratch: &mut DownsampleScratch,
) -> Vec<u8> {
    let mut out = vec![255u8; dst_w * dst_h];
    if cw == 0 || ch == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }
    let pw = cw + 2 * px;
    let ph = ch + 2 * py;
    let sx_scale = pw as f32 / dst_w as f32;
    let sy_scale = ph as f32 / dst_h as f32;

    scratch.bilin_rx.clear();
    scratch.bilin_rx.reserve(dst_w);
    scratch.x_weights.clear();
    scratch.x_weights.reserve(dst_w);
    let px_i = px as isize;
    let cw_i = cw as isize;
    let mut left_end = dst_w;
    let mut right_start = dst_w;
    for dx in 0..dst_w {
        let fx = (dx as f32 + 0.5) * sx_scale - 0.5;
        let ix = fx.floor() as isize;
        let rx = ix - px_i;
        let wx = fx - ix as f32;
        scratch.bilin_rx.push(rx as i32);
        scratch.x_weights.push(wx);
        if rx >= 0 && rx + 1 < cw_i {
            if left_end == dst_w {
                left_end = dx;
            }
        } else if left_end < dst_w && right_start == dst_w {
            right_start = dx;
        }
    }
    if left_end > right_start {
        right_start = left_end;
    }

    // 2-slot horizontal row cache: row indices increase monotonically so each in-bounds `ry` is filtered once.
    scratch.horiz.resize(2 * dst_w, 255.0);
    let mut cached_ry = [isize::MIN; 2];

    let rx_slice = &scratch.bilin_rx[..dst_w];
    let wx_slice = &scratch.x_weights[..dst_w];
    let filter_row = |ry: isize, dst_h_row: &mut [f32]| {
        if ry < 0 || ry >= ch as isize {
            dst_h_row.fill(255.0);
            return;
        }
        let row = &raw[(src_y0 + ry as usize) * stride + src_x0..(src_y0 + ry as usize) * stride + src_x0 + cw];
        let sample_x = |rx: isize| -> f32 {
            if rx < 0 || rx >= cw_i {
                255.0
            } else {
                f32::from(row[rx as usize])
            }
        };
        for dx in 0..left_end {
            let rx = rx_slice[dx] as isize;
            let wx = wx_slice[dx];
            dst_h_row[dx] = (1.0 - wx) * sample_x(rx) + wx * sample_x(rx + 1);
        }
        for dx in left_end..right_start {
            // SAFETY: by construction above, `0 <= rx && rx + 1 < cw` for every `dx` in `left_end..right_start`.
            let (rx, wx, p0, p1) = unsafe {
                let rx = *rx_slice.get_unchecked(dx) as usize;
                let wx = *wx_slice.get_unchecked(dx);
                let p0 = f32::from(*row.get_unchecked(rx));
                let p1 = f32::from(*row.get_unchecked(rx + 1));
                (rx, wx, p0, p1)
            };
            let _ = rx;
            dst_h_row[dx] = (1.0 - wx) * p0 + wx * p1;
        }
        for dx in right_start..dst_w {
            let rx = rx_slice[dx] as isize;
            let wx = wx_slice[dx];
            dst_h_row[dx] = (1.0 - wx) * sample_x(rx) + wx * sample_x(rx + 1);
        }
    };

    #[cfg(target_arch = "x86_64")]
    let has_avx2_fma = is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2_fma = false;

    let py_i = py as isize;
    for dy in 0..dst_h {
        let fy = (dy as f32 + 0.5) * sy_scale - 0.5;
        let iy = fy.floor() as isize;
        let wy = fy - iy as f32;
        let wy0 = 1.0 - wy;
        let ry0 = iy - py_i;
        let ry1 = ry0 + 1;
        if (ry0 < 0 || ry0 >= ch as isize) && (ry1 < 0 || ry1 >= ch as isize) {
            continue;
        }
        let s0 = (ry0.rem_euclid(2)) as usize;
        let s1 = 1 - s0;
        if cached_ry[s0] != ry0 {
            let h0 = &mut scratch.horiz[s0 * dst_w..(s0 + 1) * dst_w];
            filter_row(ry0, h0);
            cached_ry[s0] = ry0;
        }
        if cached_ry[s1] != ry1 {
            let h1 = &mut scratch.horiz[s1 * dst_w..(s1 + 1) * dst_w];
            filter_row(ry1, h1);
            cached_ry[s1] = ry1;
        }
        let (h0, h1) = if s0 == 0 {
            let (a, b) = scratch.horiz.split_at(dst_w);
            (&a[..dst_w], &b[..dst_w])
        } else {
            let (a, b) = scratch.horiz.split_at(dst_w);
            (&b[..dst_w], &a[..dst_w])
        };
        let out_row = &mut out[dy * dst_w..(dy + 1) * dst_w];
        if has_avx2_fma {
            #[cfg(target_arch = "x86_64")]
            unsafe { blend_bilin_row_avx2(h0, h1, wy0, wy, out_row) };
        } else {
            for dx in 0..dst_w {
                let v = wy0 * h0[dx] + wy * h1[dx];
                out_row[dx] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn blend_bilin_row_avx2(h0: &[f32], h1: &[f32], wy0: f32, wy1: f32, out_row: &mut [u8]) {
    use std::arch::x86_64::*;
    let w0 = _mm256_set1_ps(wy0);
    let w1 = _mm256_set1_ps(wy1);
    let zero = _mm256_setzero_ps();
    let max255 = _mm256_set1_ps(255.0);
    let half = _mm256_set1_ps(0.5);
    let p0 = h0.as_ptr();
    let p1 = h1.as_ptr();
    let dp = out_row.as_mut_ptr();
    let len = out_row.len();
    let mut i = 0usize;
    while i + 8 <= len {
        let a = _mm256_loadu_ps(p0.add(i));
        let b = _mm256_loadu_ps(p1.add(i));
        let v = _mm256_fmadd_ps(w1, b, _mm256_mul_ps(w0, a));
        let clamped = _mm256_min_ps(_mm256_max_ps(v, zero), max255);
        // Since clamped >= 0, floor(clamped + 0.5) matches f32::round() away from zero.
        let ints = _mm256_cvttps_epi32(_mm256_add_ps(clamped, half));
        let p16 = _mm_packus_epi32(_mm256_castsi256_si128(ints), _mm256_extracti128_si256::<1>(ints));
        let p8 = _mm_packus_epi16(p16, p16);
        _mm_storel_epi64(dp.add(i) as *mut __m128i, p8);
        i += 8;
    }
    while i < len {
        let v = wy0 * *p0.add(i) + wy1 * *p1.add(i);
        *dp.add(i) = v.round().clamp(0.0, 255.0) as u8;
        i += 1;
    }
}

fn line_energy_of_grid(bin: &[f32], w: usize, h: usize, angle_deg: f32) -> f32 {
    if angle_deg.abs() < 0.05 {
        let mut prev = 0.0f32;
        let mut sum_sq = 0.0f32;
        let mut sum_r = 0.0f32;
        for y in 0..h {
            let row = &bin[y * w..(y + 1) * w];
            let s: f32 = row.iter().sum();
            sum_r += s;
            if y > 0 {
                let d = s - prev;
                sum_sq += d * d;
            }
            prev = s;
        }
        let mean_r = sum_r / h.max(1) as f32 + 1e-3;
        return (sum_sq / (h.saturating_sub(1)).max(1) as f32) / (mean_r * mean_r);
    }
    let rad = angle_deg.to_radians();
    let (sin_a, cos_a) = (rad.sin(), rad.cos());
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    let mut prev = 0.0f32;
    let mut sum_sq = 0.0f32;
    let mut sum_r = 0.0f32;
    for y in 0..h {
        let dy = y as f32 - cy;
        let mut s = 0.0f32;
        for x in 0..w {
            let dx = x as f32 - cx;
            let sx = (cx + dx * cos_a + dy * sin_a).round() as isize;
            let sy = (cy - dx * sin_a + dy * cos_a).round() as isize;
            if sx >= 0 && sy >= 0 && (sx as usize) < w && (sy as usize) < h {
                s += bin[sy as usize * w + sx as usize];
            }
        }
        sum_r += s;
        if y > 0 {
            let d = s - prev;
            sum_sq += d * d;
        }
        prev = s;
    }
    let mean_r = sum_r / h.max(1) as f32 + 1e-3;
    (sum_sq / (h.saturating_sub(1)).max(1) as f32) / (mean_r * mean_r)
}

/// Zero-allocation grid sample checking whether a PDF page matches crisp black-on-white print.
pub(crate) fn is_crisp_white_page(gray: &GrayImage) -> bool {
    let (w, h) = gray.dimensions();
    if w == 0 || h == 0 {
        return true;
    }
    let sc = 224.0 / w.max(h).max(1) as f32;
    let sw = (w as f32 * sc).round().max(32.0) as usize;
    let sh = (h as f32 * sc).round().max(32.0) as usize;
    let raw = gray.as_raw();
    let stride = w as usize;
    let mut wf_cnt = 0usize;
    let mut mf_cnt = 0usize;
    for y in 0..sh {
        let sy = (((y as f32 + 0.5) * h as f32 / sh as f32).floor() as usize).min(h as usize - 1);
        let row = &raw[sy * stride..];
        for x in 0..sw {
            let sx = (((x as f32 + 0.5) * w as f32 / sw as f32).floor() as usize).min(w as usize - 1);
            let v = row[sx];
            if v >= 215 {
                wf_cnt += 1;
            } else if v > 120 {
                mf_cnt += 1;
            }
        }
    }
    let total = (sw * sh).max(1) as f32;
    (wf_cnt as f32 / total) >= 0.78 && (mf_cnt as f32 / total) <= 0.08
}

/// One-pass 224px analysis: upright/90/270 orientation + +/-12 deg deskew + router features.
pub(crate) fn analyze_gray_page(gray: &GrayImage) -> PageProfile {
    let (w, h) = gray.dimensions();
    let sc = 224.0 / w.max(h).max(1) as f32;
    let sw = (w as f32 * sc).round().max(32.0) as usize;
    let sh = (h as f32 * sc).round().max(32.0) as usize;
    let raw = gray.as_raw();
    let stride = w as usize;

    let mut small = vec![0.0f32; sw * sh];
    let mut wf_cnt = 0usize;
    let mut mf_cnt = 0usize;
    for y in 0..sh {
        let sy = ((y as f32 + 0.5) * h as f32 / sh as f32).floor() as usize;
        let sy = sy.min(h as usize - 1);
        for x in 0..sw {
            let sx = ((x as f32 + 0.5) * w as f32 / sw as f32).floor() as usize;
            let sx = sx.min(w as usize - 1);
            let v = raw[sy * stride + sx];
            small[y * sw + x] = f32::from(v);
            if v >= 215 {
                wf_cnt += 1;
            } else if v > 120 {
                mf_cnt += 1;
            }
        }
    }
    let total = (sw * sh).max(1) as f32;
    let white_frac = wf_cnt as f32 / total;
    let mid_frac = mf_cnt as f32 / total;

    // 7x7 separable box high-pass -> binary ink grid
    let mut tmp = vec![0.0f32; sw * sh];
    for y in 0..sh {
        let r = &small[y * sw..(y + 1) * sw];
        for x in 0..sw {
            let x0 = x.saturating_sub(3);
            let x1 = (x + 4).min(sw);
            let sum: f32 = r[x0..x1].iter().sum();
            tmp[y * sw + x] = sum / (x1 - x0) as f32;
        }
    }
    let mut bin = vec![0.0f32; sw * sh];
    for y in 0..sh {
        let y0 = y.saturating_sub(3);
        let y1 = (y + 4).min(sh);
        let denom = (y1 - y0) as f32;
        for x in 0..sw {
            let mut sum = 0.0f32;
            for yy in y0..y1 {
                sum += tmp[yy * sw + x];
            }
            bin[y * sw + x] = if (sum / denom - small[y * sw + x]) > 12.0 { 1.0 } else { 0.0 };
        }
    }

    // Compare 0 deg vs 90 deg (CCW 1 turn) horizontal line energy
    let e0 = line_energy_of_grid(&bin, sw, sh, 0.0);
    let mut bin90 = vec![0.0f32; sw * sh];
    for y in 0..sh {
        for x in 0..sw {
            // rotate 270 CW = 90 CCW: (x, y) -> (y, sw - 1 - x)
            bin90[(sw - 1 - x) * sh + y] = bin[y * sw + x];
        }
    }
    let e90 = line_energy_of_grid(&bin90, sh, sw, 0.0);

    let (turns_90, active_bin, aw, ah, base_e) = if e90 > e0 * 1.35 {
        (1u8, bin90, sh, sw, e90)
    } else {
        (0u8, bin, sw, sh, e0)
    };

    // Coarse +/-4, +/-8, +/-12 deg sweep, then +/-1, +/-2 deg refine
    let mut best_ang = 0.0f32;
    let mut best_e = base_e;
    for &ang in &[-12.0f32, -8.0, -4.0, 4.0, 8.0, 12.0] {
        let e = line_energy_of_grid(&active_bin, aw, ah, ang);
        if e > best_e {
            best_e = e;
            best_ang = ang;
        }
    }
    for &d in &[-2.0f32, -1.0, 1.0, 2.0] {
        let ang = best_ang + d;
        if ang.abs() <= 13.0 {
            let e = line_energy_of_grid(&active_bin, aw, ah, ang);
            if e > best_e {
                best_e = e;
                best_ang = ang;
            }
        }
    }
    let skew_deg = if best_ang.abs() >= 2.0 && best_e > base_e * 1.30 { best_ang } else { 0.0 };
    PageProfile {
        turns_90,
        skew_deg,
        white_frac,
        mid_frac,
        line_energy: best_e,
    }
}

/// Rotates `gray` by `turns_90` (0 or 1 CCW) and `skew_deg` (bilinear, white background) only if needed.
pub(crate) fn upright_and_deskew(gray: &GrayImage, profile: PageProfile) -> Option<GrayImage> {
    if profile.turns_90 == 0 && profile.skew_deg.abs() < 1.0 {
        return None;
    }
    let rotated = match profile.turns_90 % 4 {
        1 => imageops::rotate270(gray),
        2 => imageops::rotate180(gray),
        3 => imageops::rotate90(gray),
        _ => gray.clone(),
    };
    if profile.skew_deg.abs() < 1.0 {
        return Some(rotated);
    }
    let (w, h) = rotated.dimensions();
    let rad = profile.skew_deg.to_radians();
    let (sin_a, cos_a) = (rad.sin(), rad.cos());
    let cx = (w as f32 - 1.0) * 0.5;
    let cy = (h as f32 - 1.0) * 0.5;
    let src = rotated.as_raw();
    let wi = w as isize;
    let hi = h as isize;
    let mut dst = vec![255u8; (w * h) as usize];
    for y in 0..h as usize {
        let dy = y as f32 - cy;
        let row = &mut dst[y * w as usize..(y + 1) * w as usize];
        for x in 0..w as usize {
            let dx = x as f32 - cx;
            let sx = cx + dx * cos_a + dy * sin_a;
            let sy = cy - dx * sin_a + dy * cos_a;
            let ix = sx.floor() as isize;
            let iy = sy.floor() as isize;
            if ix >= 0 && iy >= 0 && ix + 1 < wi && iy + 1 < hi {
                let fx = sx - ix as f32;
                let fy = sy - iy as f32;
                let idx = iy as usize * w as usize + ix as usize;
                let p00 = f32::from(src[idx]);
                let p10 = f32::from(src[idx + 1]);
                let p01 = f32::from(src[idx + w as usize]);
                let p11 = f32::from(src[idx + w as usize + 1]);
                let v = (1.0 - fy) * ((1.0 - fx) * p00 + fx * p10) + fy * ((1.0 - fx) * p01 + fx * p11);
                row[x] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    GrayImage::from_raw(w, h, dst)
}

const STOP_WORDS: &[&str] = &[
    "the", "of", "and", "to", "in", "is", "for", "that", "by", "on", "as", "with", "be", "or",
    "this", "not", "are", "from", "at", "was", "it", "an", "have", "has", "had", "which", "were",
    "will", "may", "shall", "court", "order", "section", "page", "date", "all", "any", "no",
];

fn count_stop_and_alpha(lines: &[OcrLine]) -> (usize, usize, f64) {
    let (mut toks, mut stops, mut conf_sum) = (0usize, 0usize, 0.0f64);
    for line in lines {
        conf_sum += line.confidence;
        for word in line.text.split_whitespace() {
            let trimmed = word.trim_matches(|c: char| !c.is_ascii_alphanumeric());
            if trimmed.len() >= 2 && trimmed.bytes().all(|b| b.is_ascii_alphabetic()) {
                toks += 1;
                let lower = trimmed.to_ascii_lowercase();
                if STOP_WORDS.contains(&lower.as_str()) {
                    stops += 1;
                }
            }
        }
    }
    let mean_conf = if lines.is_empty() { 1.0 } else { conf_sum / lines.len() as f64 };
    (toks, stops, mean_conf)
}

/// Returns true when recognized text has low confidence or lacks common English stop-words (candidate for 180 deg flip).
pub(crate) fn needs_180_check(lines: &[OcrLine]) -> bool {
    let (toks, stops, mean_conf) = count_stop_and_alpha(lines);
    (lines.len() >= 4 && mean_conf < 0.78)
        || (mean_conf < 0.86 && toks >= 8 && (stops as f64 / toks as f64) < 0.08)
}

/// Quality score used to pick between upright and 180-deg rotated OCR passes.
pub(crate) fn text_quality_score(lines: &[OcrLine]) -> f64 {
    let (toks, stops, mean_conf) = count_stop_and_alpha(lines);
    let stop_ratio = if toks == 0 { 0.0 } else { stops as f64 / toks as f64 };
    stops as f64 * 2.0 + stop_ratio * 20.0 + mean_conf * 10.0
}

#[derive(Default)]
pub(crate) struct DownsampleScratch {
    horiz: Vec<f32>,
    acc_row: Vec<f32>,
    src_f32: Vec<f32>,
    bilin_rx: Vec<i32>,
    x0_packed: Vec<usize>,
    w8_packed: Vec<f32>,
    x_ranges: Vec<(usize, usize)>,
    x_weights: Vec<f32>,
    x_active: Vec<(usize, usize, usize, f32)>,
    y_ranges: Vec<(usize, usize)>,
    y_weights: Vec<f32>,
}

fn fill_triangle_taps(src_len: usize, dst_len: usize, ranges: &mut Vec<(usize, usize)>, weights: &mut Vec<f32>) {
    ranges.clear();
    weights.clear();
    let ratio = src_len as f32 / dst_len as f32;
    let support = ratio.max(1.0);
    let inv_support = 1.0 / support;
    for d in 0..dst_len {
        let center = (d as f32 + 0.5) * ratio - 0.5;
        let left = (center - support).ceil().max(0.0) as usize;
        let right = ((center + support).floor() as isize).min(src_len as isize - 1).max(left as isize) as usize;
        let start = weights.len();
        let mut wsum = 0.0f32;
        for s in left..=right {
            let w = (1.0 - (s as f32 - center).abs() * inv_support).max(0.0);
            weights.push(w);
            wsum += w;
        }
        if wsum > 0.0 {
            let inv = 1.0 / wsum;
            for w in &mut weights[start..] {
                *w *= inv;
            }
        }
        ranges.push((left, start));
    }
    ranges.push((0, weights.len()));
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fmadd_row_avx2(acc: &mut [f32], src: &[f32], w: f32) {
    use std::arch::x86_64::*;
    let len = acc.len();
    let vw = _mm256_set1_ps(w);
    let mut i = 0;
    while i + 8 <= len {
        let va = _mm256_loadu_ps(acc.as_ptr().add(i));
        let vs = _mm256_loadu_ps(src.as_ptr().add(i));
        let vr = _mm256_fmadd_ps(vs, vw, va);
        _mm256_storeu_ps(acc.as_mut_ptr().add(i), vr);
        i += 8;
    }
    while i < len {
        *acc.get_unchecked_mut(i) += *src.get_unchecked(i) * w;
        i += 1;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn norm_f32_inplace_avx2(row: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = row.len();
    let one = _mm256_set1_ps(1.0);
    let inv255 = _mm256_set1_ps(1.0 / 255.0);
    let mut i = 0;
    while i + 8 <= len {
        let v = _mm256_loadu_ps(row.as_ptr().add(i));
        let r = _mm256_fnmadd_ps(v, inv255, one);
        _mm256_storeu_ps(row.as_mut_ptr().add(i), r);
        i += 8;
    }
    const INV255: f32 = 1.0 / 255.0;
    while i < len {
        let p = row.get_unchecked_mut(i);
        *p = 1.0 - *p * INV255;
        i += 1;
    }
}

#[inline]
fn fmadd_row(acc: &mut [f32], src: &[f32], w: f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { fmadd_row_avx2(acc, src, w) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        fmadd_row_wasm(acc, src, w);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for (a, &s) in acc.iter_mut().zip(src) {
        *a += s * w;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn fmadd_row_wasm(acc: &mut [f32], src: &[f32], w: f32) {
    use core::arch::wasm32::*;
    let len = acc.len().min(src.len());
    let vw = f32x4_splat(w);
    let mut i = 0usize;
    while i + 4 <= len {
        unsafe {
            let va = v128_load(acc.as_ptr().add(i) as *const v128);
            let vs = v128_load(src.as_ptr().add(i) as *const v128);
            v128_store(acc.as_mut_ptr().add(i) as *mut v128, f32x4_add(va, f32x4_mul(vs, vw)));
        }
        i += 4;
    }
    while i < len {
        acc[i] += src[i] * w;
        i += 1;
    }
}

/// Zero-allocation separable Triangle (tent) downscaler directly from page `(raw, stride)` with
/// virtual 255-white padding `(px, py)` into `dst_w * dst_h` `u8` pixels.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resize_cropped_padded_triangle_u8(
    raw: &[u8],
    stride: usize,
    src_x0: usize,
    src_y0: usize,
    cw: usize,
    ch: usize,
    px: usize,
    py: usize,
    dst_w: usize,
    dst_h: usize,
    scratch: &mut DownsampleScratch,
) -> Vec<u8> {
    let pw = cw + 2 * px;
    let ph = ch + 2 * py;
    let mut out = vec![255u8; dst_w * dst_h];
    if cw == 0 || ch == 0 || dst_w == 0 || dst_h == 0 {
        return out;
    }

    fill_triangle_taps(pw, dst_w, &mut scratch.x_ranges, &mut scratch.x_weights);
    fill_triangle_taps(ph, dst_h, &mut scratch.y_ranges, &mut scratch.y_weights);

    scratch.x_active.clear();
    scratch.x_active.reserve(dst_w);
    for dx in 0..dst_w {
        let (x0, w_start) = scratch.x_ranges[dx];
        let w_end = scratch.x_ranges[dx + 1].1;
        let x_len = w_end - w_start;
        let act_start = x0.max(px);
        let act_end = (x0 + x_len).min(px + cw);
        let mut white_w = 0.0_f32;
        for k in 0..x_len {
            let sx = x0 + k;
            if sx < act_start || sx >= act_end {
                white_w += scratch.x_weights[w_start + k];
            }
        }
        let act_len = act_end.saturating_sub(act_start);
        let src_off = act_start.saturating_sub(px);
        let w_off = w_start + act_start.saturating_sub(x0);
        scratch.x_active.push((src_off, w_off, act_len, white_w * 255.0));
    }

    scratch.horiz.resize(ch * dst_w, 0.0);
    for ry in 0..ch {
        let src_row = &raw[(src_y0 + ry) * stride + src_x0..(src_y0 + ry) * stride + src_x0 + cw];
        let h_row = &mut scratch.horiz[ry * dst_w..(ry + 1) * dst_w];
        for (dx, &(src_off, w_off, act_len, white_bias)) in scratch.x_active.iter().enumerate() {
            let mut acc = white_bias;
            if act_len > 0 {
                let src_slice = &src_row[src_off..src_off + act_len];
                let w_slice = &scratch.x_weights[w_off..w_off + act_len];
                for (&s, &w) in src_slice.iter().zip(w_slice) {
                    acc += f32::from(s) * w;
                }
            }
            h_row[dx] = acc;
        }
    }

    scratch.acc_row.resize(dst_w, 0.0);
    for dy in 0..dst_h {
        let (y0, w_start) = scratch.y_ranges[dy];
        let w_end = scratch.y_ranges[dy + 1].1;
        let y_len = w_end - w_start;
        let act_start = y0.max(py);
        let act_end = (y0 + y_len).min(py + ch);
        let mut white_w = 0.0_f32;
        for k in 0..y_len {
            let sy = y0 + k;
            if sy < act_start || sy >= act_end {
                white_w += scratch.y_weights[w_start + k];
            }
        }
        scratch.acc_row[..dst_w].fill(white_w * 255.0);
        for sy in act_start..act_end {
            let ry = sy - py;
            let wy = scratch.y_weights[w_start + (sy - y0)];
            let h_row = &scratch.horiz[ry * dst_w..(ry + 1) * dst_w];
            fmadd_row(&mut scratch.acc_row[..dst_w], h_row, wy);
        }
        let out_row = &mut out[dy * dst_w..(dy + 1) * dst_w];
        for (dst_byte, &val) in out_row.iter_mut().zip(&scratch.acc_row[..dst_w]) {
            *dst_byte = val.round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horiz_8tap_row_avx2(
    src_row: &[u8],
    src_f32: &mut [f32],
    x0_packed: &[usize],
    w8_packed: &[f32],
    h_row: &mut [f32],
) {
    use std::arch::x86_64::*;
    let src_w = src_row.len();
    let dst_w = h_row.len();
    let sp = src_row.as_ptr();
    let fp = src_f32.as_mut_ptr();
    let mut i = 0usize;
    while i + 8 <= src_w {
        let bytes = _mm_loadl_epi64(sp.add(i) as *const __m128i);
        _mm256_storeu_ps(fp.add(i), _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(bytes)));
        i += 8;
    }
    while i < src_w {
        *fp.add(i) = f32::from(*sp.add(i));
        i += 1;
    }

    let x0p = x0_packed.as_ptr();
    let wp = w8_packed.as_ptr();
    let dp = h_row.as_mut_ptr();
    let mut dx = 0usize;
    while dx + 4 <= dst_w {
        let p0 = _mm256_mul_ps(_mm256_loadu_ps(fp.add(*x0p.add(dx))), _mm256_loadu_ps(wp.add(dx * 8)));
        let p1 = _mm256_mul_ps(_mm256_loadu_ps(fp.add(*x0p.add(dx + 1))), _mm256_loadu_ps(wp.add((dx + 1) * 8)));
        let p2 = _mm256_mul_ps(_mm256_loadu_ps(fp.add(*x0p.add(dx + 2))), _mm256_loadu_ps(wp.add((dx + 2) * 8)));
        let p3 = _mm256_mul_ps(_mm256_loadu_ps(fp.add(*x0p.add(dx + 3))), _mm256_loadu_ps(wp.add((dx + 3) * 8)));
        let h01 = _mm256_hadd_ps(p0, p1);
        let h23 = _mm256_hadd_ps(p2, p3);
        let h = _mm256_hadd_ps(h01, h23);
        let out4 = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h));
        _mm_storeu_ps(dp.add(dx), out4);
        dx += 4;
    }
    while dx < dst_w {
        let p = _mm256_mul_ps(_mm256_loadu_ps(fp.add(*x0p.add(dx))), _mm256_loadu_ps(wp.add(dx * 8)));
        let h = _mm256_hadd_ps(p, p);
        let h = _mm256_hadd_ps(h, h);
        let s = _mm_add_ss(_mm256_castps256_ps128(h), _mm256_extractf128_ps::<1>(h));
        _mm_store_ss(dp.add(dx), s);
        dx += 1;
    }
}

/// Zero-alloc separable Triangle (tent) downsample directly from `image` into `f32` `dst` with `1.0 - px/255.0` normalization and 0.0 right/bottom padding.
pub(crate) fn downsample_to_norm_f32(
    image: &GrayImage,
    dst_w: usize,
    dst_h: usize,
    dst_stride: usize,
    dst_h_padded: usize,
    dst: &mut Vec<f32>,
    scratch: &mut DownsampleScratch,
) {
    let (src_w, src_h) = (image.width() as usize, image.height() as usize);
    let raw = image.as_raw();
    dst.resize(dst_stride * dst_h_padded, 0.0);
    if dst_h < dst_h_padded {
        dst[dst_h * dst_stride..dst_h_padded * dst_stride].fill(0.0);
    }

    fill_triangle_taps(src_w, dst_w, &mut scratch.x_ranges, &mut scratch.x_weights);
    fill_triangle_taps(src_h, dst_h, &mut scratch.y_ranges, &mut scratch.y_weights);

    #[cfg(target_arch = "x86_64")]
    let has_avx2 = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2 = false;
    #[cfg(target_arch = "x86_64")]
    let has_avx2_fma = has_avx2 && is_x86_feature_detected!("fma");
    #[cfg(not(target_arch = "x86_64"))]
    let has_avx2_fma = false;

    let max_taps = (0..dst_w)
        .map(|dx| scratch.x_ranges[dx + 1].1 - scratch.x_ranges[dx].1)
        .max()
        .unwrap_or(0);

    scratch.horiz.resize(src_h * dst_w, 0.0);
    if has_avx2 && max_taps <= 8 {
        scratch.x0_packed.resize(dst_w, 0);
        scratch.w8_packed.resize(dst_w * 8, 0.0);
        scratch.w8_packed.fill(0.0);
        for dx in 0..dst_w {
            let (x0, w_start) = scratch.x_ranges[dx];
            let w_end = scratch.x_ranges[dx + 1].1;
            scratch.x0_packed[dx] = x0;
            scratch.w8_packed[dx * 8..dx * 8 + (w_end - w_start)].copy_from_slice(&scratch.x_weights[w_start..w_end]);
        }
        scratch.src_f32.resize(src_w + 8, 0.0);
        scratch.src_f32[src_w..src_w + 8].fill(0.0);
        for sy in 0..src_h {
            let src_row = &raw[sy * src_w..(sy + 1) * src_w];
            let h_row = &mut scratch.horiz[sy * dst_w..(sy + 1) * dst_w];
            #[cfg(target_arch = "x86_64")]
            unsafe {
                horiz_8tap_row_avx2(src_row, &mut scratch.src_f32, &scratch.x0_packed, &scratch.w8_packed, h_row);
            }
        }
    } else {
        for sy in 0..src_h {
            let src_row = &raw[sy * src_w..(sy + 1) * src_w];
            let h_row = &mut scratch.horiz[sy * dst_w..(sy + 1) * dst_w];
            for dx in 0..dst_w {
                let (x0, w_start) = scratch.x_ranges[dx];
                let w_end = scratch.x_ranges[dx + 1].1;
                let mut acc = 0.0f32;
                for (k, &w) in scratch.x_weights[w_start..w_end].iter().enumerate() {
                    acc += f32::from(src_row[x0 + k]) * w;
                }
                h_row[dx] = acc;
            }
        }
    }

    const INV255: f32 = 1.0 / 255.0;
    for dy in 0..dst_h {
        let (y0, w_start) = scratch.y_ranges[dy];
        let w_end = scratch.y_ranges[dy + 1].1;
        let out_row = &mut dst[dy * dst_stride..(dy + 1) * dst_stride];
        out_row[..dst_w].fill(0.0);
        for (k, &w) in scratch.y_weights[w_start..w_end].iter().enumerate() {
            let h_row = &scratch.horiz[(y0 + k) * dst_w..(y0 + k + 1) * dst_w];
            if has_avx2_fma {
                #[cfg(target_arch = "x86_64")]
                unsafe { fmadd_row_avx2(&mut out_row[..dst_w], h_row, w) };
            } else {
                for dx in 0..dst_w {
                    out_row[dx] += h_row[dx] * w;
                }
            }
        }
        if has_avx2_fma {
            #[cfg(target_arch = "x86_64")]
            unsafe { norm_f32_inplace_avx2(&mut out_row[..dst_w]) };
        } else {
            for dx in 0..dst_w {
                out_row[dx] = 1.0 - out_row[dx] * INV255;
            }
        }
        if dst_w < dst_stride {
            out_row[dst_w..dst_stride].fill(0.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_resizer_matches_scalar_exact() {
        let stride = 400usize;
        let raw: Vec<u8> = (0..120 * stride).map(|i| ((i * 29 + 13) % 256) as u8).collect();
        let (src_x0, src_y0, cw, ch) = (17usize, 11usize, 311usize, 25usize);
        let (px, py, dst_w, dst_h) = (9usize, 5usize, 373usize, 32usize);

        let pw = cw + 2 * px;
        let ph = ch + 2 * py;
        let sx_scale = pw as f32 / dst_w as f32;
        let sy_scale = ph as f32 / dst_h as f32;
        let sample = |sx: isize, sy: isize| -> f32 {
            let rx = sx - px as isize;
            let ry = sy - py as isize;
            if rx < 0 || ry < 0 || rx >= cw as isize || ry >= ch as isize {
                255.0
            } else {
                f32::from(raw[(src_y0 + ry as usize) * stride + (src_x0 + rx as usize)])
            }
        };
        let mut expected = vec![255u8; dst_w * dst_h];
        for dy in 0..dst_h {
            let fy = (dy as f32 + 0.5) * sy_scale - 0.5;
            let iy = fy.floor() as isize;
            let wy = fy - iy as f32;
            let wy0 = 1.0 - wy;
            for dx in 0..dst_w {
                let fx = (dx as f32 + 0.5) * sx_scale - 0.5;
                let ix = fx.floor() as isize;
                let wx = fx - ix as f32;
                let wx0 = 1.0 - wx;
                let v = wy0 * (wx0 * sample(ix, iy) + wx * sample(ix + 1, iy))
                    + wy * (wx0 * sample(ix, iy + 1) + wx * sample(ix + 1, iy + 1));
                expected[dy * dst_w + dx] = v.round().clamp(0.0, 255.0) as u8;
            }
        }

        let mut scratch = DownsampleScratch::default();
        let actual = resize_cropped_padded_bilinear(raw.as_slice(), stride, src_x0, src_y0, cw, ch, px, py, dst_w, dst_h, &mut scratch);
        assert_eq!(actual.len(), expected.len());
        let max_diff = actual.iter().zip(&expected).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
        assert!(max_diff <= 1, "max bilinear diff {max_diff} > 1");
    }
}
