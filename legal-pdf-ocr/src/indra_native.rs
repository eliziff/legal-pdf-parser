//! Pure-Rust SIMD inference for Indra CTC line recognition (AVX2+FMA on x86_64, SIMD128 on wasm32,
//! scalar fallback elsewhere), reading weights directly from Indra ONNX checkpoints with zero ORT dependency.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

const HARD_SWISH_SIXTH: f32 = 1.0 / 6.0;
const BLANK_LABEL: usize = 0;

/// Minimal decoded ONNX float tensor.
struct OnnxTensor {
    name: String,
    dims: Vec<usize>,
    data: Vec<f32>,
}

/// Reusable per-worker scratch buffers so per-line inference allocates zero heap memory after warmup.
#[derive(Default)]
pub struct IndraScratch {
    patches: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    exp: Vec<f32>,
    se_mean: Vec<f32>,
    se_mid: Vec<f32>,
    se_gate: Vec<f32>,
    logits_block: Vec<f32>,
}

struct PackedLinear {
    /// `[out_pad / 16, inp, 16]` layout so both AVX2 (`2 x f32x8`) and Wasm SIMD128 (`4 x f32x4`)
    /// load contiguous output-channel vectors with zero horizontal reductions.
    w: Vec<f32>,
    b: Vec<f32>,
    out_pad: usize,
    inp: usize,
}

impl PackedLinear {
    fn pack(raw_w: &[f32], raw_b: &[f32], out_dim: usize, inp: usize) -> Self {
        let out_pad = out_dim.div_ceil(16) * 16;
        let tiles = out_pad / 16;
        let mut w = vec![0.0_f32; tiles * inp * 16];
        for tile in 0..tiles {
            for lane in 0..16 {
                let oc = tile * 16 + lane;
                if oc < out_dim {
                    let src_row = &raw_w[oc * inp..(oc + 1) * inp];
                    for (k, &val) in src_row.iter().enumerate() {
                        w[(tile * inp + k) * 16 + lane] = val;
                    }
                }
            }
        }
        let mut b = vec![0.0_f32; out_pad];
        b[..out_dim].copy_from_slice(&raw_b[..out_dim]);
        Self { w, b, out_pad, inp }
    }
}

struct BlockWeights {
    /// Transposed from `[C, 9]` into `[9, C]` for contiguous SIMD FMA across channels.
    dw_w: Vec<f32>,
    dw_b: Vec<f32>,
    expand: PackedLinear,
    fc1: PackedLinear,
    fc2: PackedLinear,
    project: PackedLinear,
    expand_channels: usize,
    se_channels: usize,
}

/// Loaded Indra 1D-CNN CTC recognizer model ready for zero-allocation per-line SIMD inference.
pub struct IndraNative {
    channels: usize,
    vocab_size: usize,
    /// Fused `Conv2d(1 -> C, k=[32, 4])` + optional `BatchNormalization`, packed `[C_pad/16, 128, 16]`.
    stem: PackedLinear,
    blocks: Vec<BlockWeights>,
    head: PackedLinear,
}

impl IndraNative {
    /// Load an Indra ONNX model file directly from disk (not compiled on `wasm32`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_onnx_path(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read Indra ONNX model {}: {err}", path.display()))?;
        Self::from_onnx_bytes(&bytes)
    }

    /// Parse initializers directly from an Indra ONNX protobuf byte slice, fuse `stem_bn`, and pack weights.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, String> {
        let mut tensors = parse_onnx_initializers(bytes)?;

        let take = |map: &mut std::collections::HashMap<String, OnnxTensor>, name: &str| -> Result<OnnxTensor, String> {
            map.remove(name)
                .ok_or_else(|| format!("Indra ONNX model is missing initializer '{name}'"))
        };

        let stem_w_t = take(&mut tensors, "model.stem.weight")?;
        if stem_w_t.dims.len() != 4 || stem_w_t.dims[1] != 1 || stem_w_t.dims[2] != 32 || stem_w_t.dims[3] != 4 {
            return Err(format!("unexpected model.stem.weight shape {:?}", stem_w_t.dims));
        }
        let channels = stem_w_t.dims[0];
        if channels == 0 || !channels.is_multiple_of(16) {
            return Err(format!("unsupported stem channel count {channels}"));
        }
        let mut stem_w = stem_w_t.data;
        let mut stem_b = take(&mut tensors, "model.stem.bias")?.data;
        if stem_b.len() != channels {
            return Err(format!("unexpected model.stem.bias len {} for C={channels}", stem_b.len()));
        }

        // Fuse optional BatchNorm (`model.stem_bn.*`) into `stem_w` and `stem_b`.
        if let (Some(mean), Some(var)) = (
            tensors.remove("model.stem_bn.running_mean"),
            tensors.remove("model.stem_bn.running_var"),
        ) {
            let gamma = tensors.remove("model.stem_bn.weight");
            let beta = tensors.remove("model.stem_bn.bias");
            if mean.data.len() != channels || var.data.len() != channels {
                return Err("Indra stem_bn shape mismatch".to_owned());
            }
            for c in 0..channels {
                let g = gamma.as_ref().map_or(1.0, |t| t.data[c]);
                let b = beta.as_ref().map_or(0.0, |t| t.data[c]);
                let scale = g / (var.data[c] + 1e-5_f32).sqrt();
                let w_row = &mut stem_w[c * 128..(c + 1) * 128];
                for v in w_row {
                    *v *= scale;
                }
                stem_b[c] = (stem_b[c] - mean.data[c]) * scale + b;
            }
        }
        let stem = PackedLinear::pack(&stem_w, &stem_b, channels, 128);

        let mut blocks = Vec::new();
        for idx in 0..64 {
            let dw_name = format!("model.blocks.{idx}.dw.weight");
            if !tensors.contains_key(&dw_name) {
                break;
            }
            let dw_raw = take(&mut tensors, &dw_name)?;
            if dw_raw.data.len() != channels * 9 || dw_raw.dims.first().copied() != Some(channels) {
                return Err(format!("unexpected {dw_name} shape {:?}", dw_raw.dims));
            }
            let mut dw_w = vec![0.0_f32; 9 * channels];
            for c in 0..channels {
                for k in 0..9 {
                    dw_w[k * channels + c] = dw_raw.data[c * 9 + k];
                }
            }
            let dw_b = take(&mut tensors, &format!("model.blocks.{idx}.dw.bias"))?.data;
            if dw_b.len() != channels {
                return Err(format!("unexpected blocks.{idx}.dw.bias len {}", dw_b.len()));
            }

            let expand_w_t = take(&mut tensors, &format!("model.blocks.{idx}.expand.weight"))?;
            let expand_channels = expand_w_t.dims.first().copied().unwrap_or(0);
            if expand_channels == 0
                || !expand_channels.is_multiple_of(16)
                || expand_w_t.data.len() != expand_channels * channels
            {
                return Err(format!("unexpected blocks.{idx}.expand.weight shape {:?}", expand_w_t.dims));
            }
            let expand_b = take(&mut tensors, &format!("model.blocks.{idx}.expand.bias"))?.data;
            if expand_b.len() != expand_channels {
                return Err(format!("unexpected blocks.{idx}.expand.bias len {}", expand_b.len()));
            }

            let fc1_w_t = take(&mut tensors, &format!("model.blocks.{idx}.fc1.weight"))?;
            let se_channels = fc1_w_t.dims.first().copied().unwrap_or(0);
            if se_channels == 0
                || !se_channels.is_multiple_of(16)
                || fc1_w_t.data.len() != se_channels * expand_channels
            {
                return Err(format!("unexpected blocks.{idx}.fc1.weight shape {:?}", fc1_w_t.dims));
            }
            let fc1_b = take(&mut tensors, &format!("model.blocks.{idx}.fc1.bias"))?.data;
            if fc1_b.len() != se_channels {
                return Err(format!("unexpected blocks.{idx}.fc1.bias len {}", fc1_b.len()));
            }

            let fc2_w_t = take(&mut tensors, &format!("model.blocks.{idx}.fc2.weight"))?;
            if fc2_w_t.dims.first().copied() != Some(expand_channels)
                || fc2_w_t.data.len() != expand_channels * se_channels
            {
                return Err(format!("unexpected blocks.{idx}.fc2.weight shape {:?}", fc2_w_t.dims));
            }
            let fc2_b = take(&mut tensors, &format!("model.blocks.{idx}.fc2.bias"))?.data;
            if fc2_b.len() != expand_channels {
                return Err(format!("unexpected blocks.{idx}.fc2.bias len {}", fc2_b.len()));
            }

            let project_w_t = take(&mut tensors, &format!("model.blocks.{idx}.project.weight"))?;
            if project_w_t.dims.first().copied() != Some(channels)
                || project_w_t.data.len() != channels * expand_channels
            {
                return Err(format!("unexpected blocks.{idx}.project.weight shape {:?}", project_w_t.dims));
            }
            let project_b = take(&mut tensors, &format!("model.blocks.{idx}.project.bias"))?.data;
            if project_b.len() != channels {
                return Err(format!("unexpected blocks.{idx}.project.bias len {}", project_b.len()));
            }

            blocks.push(BlockWeights {
                dw_w,
                dw_b,
                expand: PackedLinear::pack(&expand_w_t.data, &expand_b, expand_channels, channels),
                fc1: PackedLinear::pack(&fc1_w_t.data, &fc1_b, se_channels, expand_channels),
                fc2: PackedLinear::pack(&fc2_w_t.data, &fc2_b, expand_channels, se_channels),
                project: PackedLinear::pack(&project_w_t.data, &project_b, channels, expand_channels),
                expand_channels,
                se_channels,
            });
        }
        if blocks.is_empty() {
            return Err("Indra ONNX model has zero residual blocks".to_owned());
        }

        let head_w_t = take(&mut tensors, "model.head.weight")?;
        let vocab_size = head_w_t.dims.first().copied().unwrap_or(0);
        if vocab_size == 0 || head_w_t.data.len() != vocab_size * channels {
            return Err(format!("unexpected model.head.weight shape {:?}", head_w_t.dims));
        }
        let head_b = take(&mut tensors, "model.head.bias")?.data;
        if head_b.len() != vocab_size {
            return Err(format!("unexpected model.head.bias len {}", head_b.len()));
        }
        let head = PackedLinear::pack(&head_w_t.data, &head_b, vocab_size, channels);

        Ok(Self {
            channels,
            vocab_size,
            stem,
            blocks,
            head,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Run full forward pass over a `[32, width]` row-major `u8` line image, returning
    /// `(timesteps, step_labels, ctc_spans)` without materializing `[timesteps, vocab_size]`.
    /// When `need_confidence` is false, per-emission scores are `1.0` (matching `greedy_ids`);
    /// when `need_confidence` is true, per-emission scores match `greedy_ctc` max softmax probability.
    pub fn recognize_u8(
        &self,
        pixels: &[u8],
        width: usize,
        need_confidence: bool,
        scratch: &mut IndraScratch,
    ) -> (usize, Vec<Option<usize>>, Vec<(usize, f32)>) {
        if width == 0 || pixels.len() < 32 * width {
            return (0, Vec::new(), Vec::new());
        }
        let timesteps = width.div_ceil(4);
        extract_patches_u8(pixels, width, timesteps, &mut scratch.patches);
        self.run_backbone_and_head(timesteps, need_confidence, scratch)
    }

    /// Exact full-logit forward pass from already-normalized `[32, stride]` `f32` pixels (`0..width`
    /// valid, `width..stride` right-padded with `0.0`), primarily for numerical parity tests against ORT.
    pub fn forward_logits_f32(
        &self,
        norm_pixels: &[f32],
        stride: usize,
        width: usize,
        scratch: &mut IndraScratch,
    ) -> (usize, Vec<f32>) {
        if width == 0 || stride < width || norm_pixels.len() < 32 * stride {
            return (0, Vec::new());
        }
        let timesteps = width.div_ceil(4);
        extract_patches_f32(norm_pixels, stride, width, timesteps, &mut scratch.patches);
        self.run_backbone(timesteps, scratch);
        let head_stride = self.head.out_pad;
        scratch.logits_block.resize(timesteps * head_stride, 0.0);
        linear_packed::<false>(&scratch.a, timesteps, &self.head, &mut scratch.logits_block);
        if head_stride == self.vocab_size {
            return (timesteps, scratch.logits_block[..timesteps * self.vocab_size].to_vec());
        }
        let mut logits = Vec::with_capacity(timesteps * self.vocab_size);
        for t in 0..timesteps {
            let row = &scratch.logits_block[t * head_stride..t * head_stride + self.vocab_size];
            logits.extend_from_slice(row);
        }
        (timesteps, logits)
    }

    fn run_backbone(&self, timesteps: usize, scratch: &mut IndraScratch) {
        let c = self.channels;
        scratch.a.resize(timesteps * c, 0.0);
        scratch.b.resize(timesteps * c, 0.0);
        linear_packed::<false>(&scratch.patches, timesteps, &self.stem, &mut scratch.a);
        hardswish_inplace(&mut scratch.a[..timesteps * c]);

        for block in &self.blocks {
            let e = block.expand_channels;
            let s = block.se_channels;
            scratch.exp.resize(timesteps * e, 0.0);
            scratch.se_mean.resize(e, 0.0);
            scratch.se_mid.resize(s, 0.0);
            scratch.se_gate.resize(e, 0.0);

            depthwise_conv1d_k9(
                &scratch.a[..timesteps * c],
                timesteps,
                c,
                &block.dw_w,
                &block.dw_b,
                &mut scratch.b[..timesteps * c],
            );

            linear_packed::<false>(&scratch.b[..timesteps * c], timesteps, &block.expand, &mut scratch.exp[..timesteps * e]);
            hardswish_and_channel_sum(
                &mut scratch.exp[..timesteps * e],
                timesteps,
                e,
                &mut scratch.se_mean[..e],
            );

            let inv_t = (timesteps.max(1) as f32).recip();
            for v in &mut scratch.se_mean[..e] {
                *v *= inv_t;
            }
            linear_packed::<false>(&scratch.se_mean[..e], 1, &block.fc1, &mut scratch.se_mid[..s]);
            for v in &mut scratch.se_mid[..s] {
                *v = v.max(0.0);
            }
            linear_packed::<false>(&scratch.se_mid[..s], 1, &block.fc2, &mut scratch.se_gate[..e]);
            for v in &mut scratch.se_gate[..e] {
                *v = 1.0 / (1.0 + (-*v).exp());
            }

            scale_channels_inplace(&mut scratch.exp[..timesteps * e], timesteps, e, &scratch.se_gate[..e]);

            linear_packed::<true>(&scratch.exp[..timesteps * e], timesteps, &block.project, &mut scratch.a[..timesteps * c]);
        }
    }

    fn run_backbone_and_head(
        &self,
        timesteps: usize,
        need_confidence: bool,
        scratch: &mut IndraScratch,
    ) -> (usize, Vec<Option<usize>>, Vec<(usize, f32)>) {
        self.run_backbone(timesteps, scratch);

        let c = self.channels;
        let v = self.vocab_size;
        let head_stride = self.head.out_pad;
        let chunk_rows = 24.min(timesteps);
        scratch.logits_block.resize(chunk_rows * head_stride, 0.0);

        let mut step_labels = Vec::with_capacity(timesteps);
        let mut ctc_spans: Vec<(usize, f32)> = Vec::new();
        let mut previous = usize::MAX;

        let mut t = 0usize;
        while t < timesteps {
            let rows = (timesteps - t).min(chunk_rows);
            let x_slice = &scratch.a[t * c..(t + rows) * c];
            let out_slice = &mut scratch.logits_block[..rows * head_stride];
            linear_packed::<false>(x_slice, rows, &self.head, out_slice);

            for row_idx in 0..rows {
                let row = &out_slice[row_idx * head_stride..row_idx * head_stride + v];
                let (best_label, best_val) = argmax_f32(row);
                step_labels.push((best_label != BLANK_LABEL).then_some(best_label));

                if best_label != BLANK_LABEL {
                    if best_label != previous {
                        let conf = if need_confidence {
                            row_max_prob(row, best_val)
                        } else {
                            1.0
                        };
                        ctc_spans.push((best_label, conf));
                    } else if need_confidence {
                        if let Some((_, score)) = ctc_spans.last_mut() {
                            *score = score.max(row_max_prob(row, best_val));
                        }
                    }
                }
                previous = best_label;
            }
            t += rows;
        }
        (timesteps, step_labels, ctc_spans)
    }
}

#[inline]
fn row_max_prob(row: &[f32], max_val: f32) -> f32 {
    let denom: f32 = row.iter().map(|&val| (val - max_val).exp()).sum();
    1.0 / denom.max(f32::MIN_POSITIVE)
}

#[inline]
fn argmax_f32(row: &[f32]) -> (usize, f32) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        return unsafe { argmax_f32_avx2(row) };
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in row.iter().enumerate() {
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    (best_idx, best_val)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn argmax_f32_avx2(row: &[f32]) -> (usize, f32) {
    use std::arch::x86_64::*;
    let len = row.len();
    let ptr = row.as_ptr();
    let mut i = 0usize;
    let mut best_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    while i + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(i));
        best_vec = _mm256_max_ps(best_vec, v);
        i += 8;
    }
    // Reduce 8 lanes -> scalar max
    let hi = _mm256_extractf128_ps::<1>(best_vec);
    let lo = _mm256_castps256_ps128(best_vec);
    let m128 = _mm_max_ps(lo, hi);
    let m64 = _mm_max_ps(m128, _mm_movehl_ps(m128, m128));
    let m32 = _mm_max_ss(m64, _mm_shuffle_ps::<1>(m64, m64));
    let mut max_val = _mm_cvtss_f32(m32);
    for tail in i..len {
        let v = *ptr.add(tail);
        if v > max_val {
            max_val = v;
        }
    }
    // Find first index matching `max_val`
    let target = _mm256_set1_ps(max_val);
    let mut j = 0usize;
    while j + 8 <= len {
        let v = _mm256_loadu_ps(ptr.add(j));
        let cmp = _mm256_cmp_ps::<_CMP_EQ_OQ>(v, target);
        let mask = _mm256_movemask_ps(cmp) as u32;
        if mask != 0 {
            return (j + mask.trailing_zeros() as usize, max_val);
        }
        j += 8;
    }
    while j < len {
        if *ptr.add(j) == max_val {
            return (j, max_val);
        }
        j += 1;
    }
    (0, max_val)
}

fn extract_patches_u8(pixels: &[u8], width: usize, timesteps: usize, patches: &mut Vec<f32>) {
    const INV255: f32 = 1.0 / 255.0;
    patches.resize(timesteps * 128, 0.0);
    let full_steps = width / 4;
    for t in 0..full_steps {
        let dst = &mut patches[t * 128..(t + 1) * 128];
        let col0 = t * 4;
        for r in 0..32 {
            let src = &pixels[r * width + col0..r * width + col0 + 4];
            let d = &mut dst[r * 4..r * 4 + 4];
            d[0] = 1.0 - f32::from(src[0]) * INV255;
            d[1] = 1.0 - f32::from(src[1]) * INV255;
            d[2] = 1.0 - f32::from(src[2]) * INV255;
            d[3] = 1.0 - f32::from(src[3]) * INV255;
        }
    }
    if full_steps < timesteps {
        let t = full_steps;
        let col0 = t * 4;
        let rem = width - col0;
        let dst = &mut patches[t * 128..(t + 1) * 128];
        for r in 0..32 {
            let src_off = r * width + col0;
            let d = &mut dst[r * 4..r * 4 + 4];
            for dx in 0..4 {
                d[dx] = if dx < rem {
                    1.0 - f32::from(pixels[src_off + dx]) * INV255
                } else {
                    0.0
                };
            }
        }
    }
}

fn extract_patches_f32(
    norm_pixels: &[f32],
    stride: usize,
    width: usize,
    timesteps: usize,
    patches: &mut Vec<f32>,
) {
    patches.resize(timesteps * 128, 0.0);
    let full_steps = width / 4;
    for t in 0..full_steps {
        let dst = &mut patches[t * 128..(t + 1) * 128];
        let col0 = t * 4;
        for r in 0..32 {
            let src = &norm_pixels[r * stride + col0..r * stride + col0 + 4];
            dst[r * 4..r * 4 + 4].copy_from_slice(src);
        }
    }
    if full_steps < timesteps {
        let t = full_steps;
        let col0 = t * 4;
        let rem = width - col0;
        let dst = &mut patches[t * 128..(t + 1) * 128];
        for r in 0..32 {
            let src_off = r * stride + col0;
            let d = &mut dst[r * 4..r * 4 + 4];
            for dx in 0..4 {
                d[dx] = if dx < rem { norm_pixels[src_off + dx] } else { 0.0 };
            }
        }
    }
}

#[inline]
fn hardswish_scalar(x: f32) -> f32 {
    x * ((x + 3.0).clamp(0.0, 6.0) * HARD_SWISH_SIXTH)
}

fn hardswish_inplace(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { hardswish_inplace_avx2(data) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        hardswish_inplace_wasm(data);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for v in data {
        *v = hardswish_scalar(*v);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hardswish_inplace_avx2(data: &mut [f32]) {
    use std::arch::x86_64::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let three = _mm256_set1_ps(3.0);
    let six = _mm256_set1_ps(6.0);
    let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);
    let mut i = 0usize;
    while i + 8 <= len {
        let x = _mm256_loadu_ps(ptr.add(i));
        let t = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(x, three), zero), six);
        let y = _mm256_mul_ps(x, _mm256_mul_ps(t, sixth));
        _mm256_storeu_ps(ptr.add(i), y);
        i += 8;
    }
    while i < len {
        let p = ptr.add(i);
        *p = hardswish_scalar(*p);
        i += 1;
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn hardswish_inplace_wasm(data: &mut [f32]) {
    use core::arch::wasm32::*;
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let zero = f32x4_splat(0.0);
    let three = f32x4_splat(3.0);
    let six = f32x4_splat(6.0);
    let sixth = f32x4_splat(HARD_SWISH_SIXTH);
    let mut i = 0usize;
    while i + 4 <= len {
        // SAFETY: i + 4 <= len.
        unsafe {
            let x = v128_load(ptr.add(i) as *const v128);
            let t = f32x4_min(f32x4_max(f32x4_add(x, three), zero), six);
            let y = f32x4_mul(x, f32x4_mul(t, sixth));
            v128_store(ptr.add(i) as *mut v128, y);
        }
        i += 4;
    }
    while i < len {
        data[i] = hardswish_scalar(data[i]);
        i += 1;
    }
}

fn hardswish_and_channel_sum(data: &mut [f32], timesteps: usize, channels: usize, sum: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { hardswish_and_channel_sum_avx2(data, timesteps, channels, sum) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        hardswish_and_channel_sum_wasm(data, timesteps, channels, sum);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        sum[..channels].fill(0.0);
        for row in data[..timesteps * channels].chunks_exact_mut(channels) {
            for (v, acc) in row.iter_mut().zip(&mut sum[..channels]) {
                let y = hardswish_scalar(*v);
                *v = y;
                *acc += y;
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hardswish_and_channel_sum_avx2(data: &mut [f32], timesteps: usize, channels: usize, sum: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(data.len() >= timesteps * channels && sum.len() >= channels);
    sum[..channels].fill(0.0);
    let zero = _mm256_setzero_ps();
    let three = _mm256_set1_ps(3.0);
    let six = _mm256_set1_ps(6.0);
    let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);
    let sp = sum.as_mut_ptr();
    for t in 0..timesteps {
        let dp = data.as_mut_ptr().add(t * channels);
        let mut c = 0usize;
        while c + 8 <= channels {
            let x = _mm256_loadu_ps(dp.add(c));
            let t_clamp = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(x, three), zero), six);
            let y = _mm256_mul_ps(x, _mm256_mul_ps(t_clamp, sixth));
            _mm256_storeu_ps(dp.add(c), y);
            let cur = _mm256_loadu_ps(sp.add(c));
            _mm256_storeu_ps(sp.add(c), _mm256_add_ps(cur, y));
            c += 8;
        }
        while c < channels {
            let y = hardswish_scalar(*dp.add(c));
            *dp.add(c) = y;
            *sp.add(c) += y;
            c += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn hardswish_and_channel_sum_wasm(data: &mut [f32], timesteps: usize, channels: usize, sum: &mut [f32]) {
    use core::arch::wasm32::*;
    assert!(data.len() >= timesteps * channels && sum.len() >= channels);
    sum[..channels].fill(0.0);
    let zero = f32x4_splat(0.0);
    let three = f32x4_splat(3.0);
    let six = f32x4_splat(6.0);
    let sixth = f32x4_splat(HARD_SWISH_SIXTH);
    let sp = sum.as_mut_ptr();
    for t in 0..timesteps {
        let dp = unsafe { data.as_mut_ptr().add(t * channels) };
        let mut c = 0usize;
        while c + 4 <= channels {
            unsafe {
                let x = v128_load(dp.add(c) as *const v128);
                let t_clamp = f32x4_min(f32x4_max(f32x4_add(x, three), zero), six);
                let y = f32x4_mul(x, f32x4_mul(t_clamp, sixth));
                v128_store(dp.add(c) as *mut v128, y);
                let cur = v128_load(sp.add(c) as *const v128);
                v128_store(sp.add(c) as *mut v128, f32x4_add(cur, y));
            }
            c += 4;
        }
        while c < channels {
            let y = hardswish_scalar(data[t * channels + c]);
            data[t * channels + c] = y;
            sum[c] += y;
            c += 1;
        }
    }
}

fn scale_channels_inplace(data: &mut [f32], timesteps: usize, channels: usize, gate: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { scale_channels_inplace_avx2(data, timesteps, channels, gate) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        scale_channels_inplace_wasm(data, timesteps, channels, gate);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for row in data[..timesteps * channels].chunks_exact_mut(channels) {
        for (v, &g) in row.iter_mut().zip(&gate[..channels]) {
            *v *= g;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_channels_inplace_avx2(data: &mut [f32], timesteps: usize, channels: usize, gate: &[f32]) {
    use std::arch::x86_64::*;
    assert!(data.len() >= timesteps * channels && gate.len() >= channels);
    let gp = gate.as_ptr();
    for t in 0..timesteps {
        let dp = data.as_mut_ptr().add(t * channels);
        let mut c = 0usize;
        while c + 8 <= channels {
            let x = _mm256_loadu_ps(dp.add(c));
            let g = _mm256_loadu_ps(gp.add(c));
            _mm256_storeu_ps(dp.add(c), _mm256_mul_ps(x, g));
            c += 8;
        }
        while c < channels {
            *dp.add(c) *= *gp.add(c);
            c += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn scale_channels_inplace_wasm(data: &mut [f32], timesteps: usize, channels: usize, gate: &[f32]) {
    use core::arch::wasm32::*;
    assert!(data.len() >= timesteps * channels && gate.len() >= channels);
    let gp = gate.as_ptr();
    for t in 0..timesteps {
        let dp = unsafe { data.as_mut_ptr().add(t * channels) };
        let mut c = 0usize;
        while c + 4 <= channels {
            unsafe {
                let x = v128_load(dp.add(c) as *const v128);
                let g = v128_load(gp.add(c) as *const v128);
                v128_store(dp.add(c) as *mut v128, f32x4_mul(x, g));
            }
            c += 4;
        }
        while c < channels {
            data[t * channels + c] *= gate[c];
            c += 1;
        }
    }
}

fn depthwise_conv1d_k9(
    src: &[f32],
    timesteps: usize,
    channels: usize,
    weights: &[f32],
    bias: &[f32],
    dst: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { depthwise_conv1d_k9_avx2(src, timesteps, channels, weights, bias, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        depthwise_conv1d_k9_wasm(src, timesteps, channels, weights, bias, dst);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        assert!(src.len() >= timesteps * channels && dst.len() >= timesteps * channels);
        assert!(weights.len() >= 9 * channels && bias.len() >= channels);
        for t in 0..timesteps {
            let out_row = &mut dst[t * channels..(t + 1) * channels];
            out_row.copy_from_slice(&bias[..channels]);
            for k in 0..9usize {
                let st = t as isize + k as isize - 4;
                if st >= 0 && (st as usize) < timesteps {
                    let s_row = &src[st as usize * channels..(st as usize + 1) * channels];
                    let w_row = &weights[k * channels..(k + 1) * channels];
                    for c in 0..channels {
                        out_row[c] += s_row[c] * w_row[c];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn depthwise_conv1d_k9_avx2(
    src: &[f32],
    timesteps: usize,
    channels: usize,
    weights: &[f32],
    bias: &[f32],
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    assert!(src.len() >= timesteps * channels && dst.len() >= timesteps * channels);
    assert!(weights.len() >= 9 * channels && bias.len() >= channels);
    let sp = src.as_ptr();
    let wp = weights.as_ptr();
    let bp = bias.as_ptr();
    let dp = dst.as_mut_ptr();

    for t in 0..timesteps {
        let out_row = dp.add(t * channels);
        let k_start = 4usize.saturating_sub(t);
        let k_end = (timesteps + 4 - t).min(9);

        let mut c = 0usize;
        // Keep 4 AVX2 accumulators (32 channels) in registers across all 9 kernel taps.
        while c + 32 <= channels {
            let mut a0 = _mm256_loadu_ps(bp.add(c));
            let mut a1 = _mm256_loadu_ps(bp.add(c + 8));
            let mut a2 = _mm256_loadu_ps(bp.add(c + 16));
            let mut a3 = _mm256_loadu_ps(bp.add(c + 24));
            for k in k_start..k_end {
                let st = t + k - 4;
                let sr = sp.add(st * channels + c);
                let wr = wp.add(k * channels + c);
                a0 = _mm256_fmadd_ps(_mm256_loadu_ps(sr), _mm256_loadu_ps(wr), a0);
                a1 = _mm256_fmadd_ps(_mm256_loadu_ps(sr.add(8)), _mm256_loadu_ps(wr.add(8)), a1);
                a2 = _mm256_fmadd_ps(_mm256_loadu_ps(sr.add(16)), _mm256_loadu_ps(wr.add(16)), a2);
                a3 = _mm256_fmadd_ps(_mm256_loadu_ps(sr.add(24)), _mm256_loadu_ps(wr.add(24)), a3);
            }
            _mm256_storeu_ps(out_row.add(c), a0);
            _mm256_storeu_ps(out_row.add(c + 8), a1);
            _mm256_storeu_ps(out_row.add(c + 16), a2);
            _mm256_storeu_ps(out_row.add(c + 24), a3);
            c += 32;
        }
        while c + 8 <= channels {
            let mut acc = _mm256_loadu_ps(bp.add(c));
            for k in k_start..k_end {
                let st = t + k - 4;
                acc = _mm256_fmadd_ps(
                    _mm256_loadu_ps(sp.add(st * channels + c)),
                    _mm256_loadu_ps(wp.add(k * channels + c)),
                    acc,
                );
            }
            _mm256_storeu_ps(out_row.add(c), acc);
            c += 8;
        }
        while c < channels {
            let mut acc = *bp.add(c);
            for k in k_start..k_end {
                let st = t + k - 4;
                acc += *sp.add(st * channels + c) * *wp.add(k * channels + c);
            }
            *out_row.add(c) = acc;
            c += 1;
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn depthwise_conv1d_k9_wasm(
    src: &[f32],
    timesteps: usize,
    channels: usize,
    weights: &[f32],
    bias: &[f32],
    dst: &mut [f32],
) {
    use core::arch::wasm32::*;
    assert!(src.len() >= timesteps * channels && dst.len() >= timesteps * channels);
    assert!(weights.len() >= 9 * channels && bias.len() >= channels);
    let sp = src.as_ptr();
    let wp = weights.as_ptr();
    let bp = bias.as_ptr();
    let dp = dst.as_mut_ptr();

    for t in 0..timesteps {
        let out_row = unsafe { dp.add(t * channels) };
        let k_start = 4usize.saturating_sub(t);
        let k_end = (timesteps + 4 - t).min(9);

        let mut c = 0usize;
        while c + 16 <= channels {
            unsafe {
                let mut a0 = v128_load(bp.add(c) as *const v128);
                let mut a1 = v128_load(bp.add(c + 4) as *const v128);
                let mut a2 = v128_load(bp.add(c + 8) as *const v128);
                let mut a3 = v128_load(bp.add(c + 12) as *const v128);
                for k in k_start..k_end {
                    let st = t + k - 4;
                    let sr = sp.add(st * channels + c);
                    let wr = wp.add(k * channels + c);
                    a0 = f32x4_add(a0, f32x4_mul(v128_load(sr as *const v128), v128_load(wr as *const v128)));
                    a1 = f32x4_add(a1, f32x4_mul(v128_load(sr.add(4) as *const v128), v128_load(wr.add(4) as *const v128)));
                    a2 = f32x4_add(a2, f32x4_mul(v128_load(sr.add(8) as *const v128), v128_load(wr.add(8) as *const v128)));
                    a3 = f32x4_add(a3, f32x4_mul(v128_load(sr.add(12) as *const v128), v128_load(wr.add(12) as *const v128)));
                }
                v128_store(out_row.add(c) as *mut v128, a0);
                v128_store(out_row.add(c + 4) as *mut v128, a1);
                v128_store(out_row.add(c + 8) as *mut v128, a2);
                v128_store(out_row.add(c + 12) as *mut v128, a3);
            }
            c += 16;
        }
        while c + 4 <= channels {
            unsafe {
                let mut acc = v128_load(bp.add(c) as *const v128);
                for k in k_start..k_end {
                    let st = t + k - 4;
                    acc = f32x4_add(
                        acc,
                        f32x4_mul(
                            v128_load(sp.add(st * channels + c) as *const v128),
                            v128_load(wp.add(k * channels + c) as *const v128),
                        ),
                    );
                }
                v128_store(out_row.add(c) as *mut v128, acc);
            }
            c += 4;
        }
        while c < channels {
            let mut acc = bias[c];
            for k in k_start..k_end {
                let st = t + k - 4;
                acc += src[st * channels + c] * weights[k * channels + c];
            }
            dst[t * channels + c] = acc;
            c += 1;
        }
    }
}

/// Outer-product GEMM over weights pre-packed as `[out_pad / 16, inp, 16]` with output stride `lin.out_pad`.
fn linear_packed<const ADD_TO_DST: bool>(x: &[f32], rows: usize, lin: &PackedLinear, dst: &mut [f32]) {
    if rows == 0 || lin.out_pad == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { linear_packed_avx2::<ADD_TO_DST>(x, rows, lin, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        linear_packed_wasm::<ADD_TO_DST>(x, rows, lin, dst);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let inp = lin.inp;
        let out_pad = lin.out_pad;
        let tiles = out_pad / 16;
        assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);
        for tile in 0..tiles {
            let c = tile * 16;
            let w_tile = &lin.w[tile * inp * 16..(tile + 1) * inp * 16];
            let b_tile = &lin.b[c..c + 16];
            for r in 0..rows {
                let xr = &x[r * inp..(r + 1) * inp];
                let dr = &mut dst[r * out_pad + c..r * out_pad + c + 16];
                if ADD_TO_DST {
                    for lane in 0..16 {
                        dr[lane] += b_tile[lane];
                    }
                } else {
                    dr.copy_from_slice(b_tile);
                }
                for (k, &xv) in xr.iter().enumerate() {
                    let w_k = &w_tile[k * 16..k * 16 + 16];
                    for lane in 0..16 {
                        dr[lane] += xv * w_k[lane];
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_packed_avx2<const ADD_TO_DST: bool>(
    x: &[f32],
    rows: usize,
    lin: &PackedLinear,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let inp = lin.inp;
    let out_pad = lin.out_pad;
    let tiles = out_pad / 16;
    assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let b_base = lin.b.as_ptr();
    let d_base = dst.as_mut_ptr();

    for tile in 0..tiles {
        let c = tile * 16;
        let w_tile = w_base.add(tile * inp * 16);
        let bv0 = _mm256_loadu_ps(b_base.add(c));
        let bv1 = _mm256_loadu_ps(b_base.add(c + 8));

        let mut r = 0usize;
        // 6-row x 16-col outer product: 12 YMM accumulators + 2 weight loads + 1 broadcast, 0 hsums.
        while r + 6 <= rows {
            let xp0 = x_base.add(r * inp);
            let xp1 = x_base.add((r + 1) * inp);
            let xp2 = x_base.add((r + 2) * inp);
            let xp3 = x_base.add((r + 3) * inp);
            let xp4 = x_base.add((r + 4) * inp);
            let xp5 = x_base.add((r + 5) * inp);
            let dp0 = d_base.add(r * out_pad + c);
            let dp1 = d_base.add((r + 1) * out_pad + c);
            let dp2 = d_base.add((r + 2) * out_pad + c);
            let dp3 = d_base.add((r + 3) * out_pad + c);
            let dp4 = d_base.add((r + 4) * out_pad + c);
            let dp5 = d_base.add((r + 5) * out_pad + c);

            let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31, mut a40, mut a41, mut a50, mut a51) =
                if ADD_TO_DST {
                    (
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp0)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp0.add(8))),
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp1)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp1.add(8))),
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp2)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp2.add(8))),
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp3)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp3.add(8))),
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp4)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp4.add(8))),
                        _mm256_add_ps(bv0, _mm256_loadu_ps(dp5)),
                        _mm256_add_ps(bv1, _mm256_loadu_ps(dp5.add(8))),
                    )
                } else {
                    (bv0, bv1, bv0, bv1, bv0, bv1, bv0, bv1, bv0, bv1, bv0, bv1)
                };

            for k in 0..inp {
                let wp = w_tile.add(k * 16);
                let wv0 = _mm256_loadu_ps(wp);
                let wv1 = _mm256_loadu_ps(wp.add(8));
                let x0 = _mm256_broadcast_ss(&*xp0.add(k));
                a00 = _mm256_fmadd_ps(x0, wv0, a00);
                a01 = _mm256_fmadd_ps(x0, wv1, a01);
                let x1 = _mm256_broadcast_ss(&*xp1.add(k));
                a10 = _mm256_fmadd_ps(x1, wv0, a10);
                a11 = _mm256_fmadd_ps(x1, wv1, a11);
                let x2 = _mm256_broadcast_ss(&*xp2.add(k));
                a20 = _mm256_fmadd_ps(x2, wv0, a20);
                a21 = _mm256_fmadd_ps(x2, wv1, a21);
                let x3 = _mm256_broadcast_ss(&*xp3.add(k));
                a30 = _mm256_fmadd_ps(x3, wv0, a30);
                a31 = _mm256_fmadd_ps(x3, wv1, a31);
                let x4 = _mm256_broadcast_ss(&*xp4.add(k));
                a40 = _mm256_fmadd_ps(x4, wv0, a40);
                a41 = _mm256_fmadd_ps(x4, wv1, a41);
                let x5 = _mm256_broadcast_ss(&*xp5.add(k));
                a50 = _mm256_fmadd_ps(x5, wv0, a50);
                a51 = _mm256_fmadd_ps(x5, wv1, a51);
            }

            _mm256_storeu_ps(dp0, a00);
            _mm256_storeu_ps(dp0.add(8), a01);
            _mm256_storeu_ps(dp1, a10);
            _mm256_storeu_ps(dp1.add(8), a11);
            _mm256_storeu_ps(dp2, a20);
            _mm256_storeu_ps(dp2.add(8), a21);
            _mm256_storeu_ps(dp3, a30);
            _mm256_storeu_ps(dp3.add(8), a31);
            _mm256_storeu_ps(dp4, a40);
            _mm256_storeu_ps(dp4.add(8), a41);
            _mm256_storeu_ps(dp5, a50);
            _mm256_storeu_ps(dp5.add(8), a51);
            r += 6;
        }

        while r + 2 <= rows {
            let xp0 = x_base.add(r * inp);
            let xp1 = x_base.add((r + 1) * inp);
            let dp0 = d_base.add(r * out_pad + c);
            let dp1 = d_base.add((r + 1) * out_pad + c);
            let (mut a00, mut a01, mut a10, mut a11) = if ADD_TO_DST {
                (
                    _mm256_add_ps(bv0, _mm256_loadu_ps(dp0)),
                    _mm256_add_ps(bv1, _mm256_loadu_ps(dp0.add(8))),
                    _mm256_add_ps(bv0, _mm256_loadu_ps(dp1)),
                    _mm256_add_ps(bv1, _mm256_loadu_ps(dp1.add(8))),
                )
            } else {
                (bv0, bv1, bv0, bv1)
            };
            for k in 0..inp {
                let wp = w_tile.add(k * 16);
                let wv0 = _mm256_loadu_ps(wp);
                let wv1 = _mm256_loadu_ps(wp.add(8));
                let x0 = _mm256_broadcast_ss(&*xp0.add(k));
                let x1 = _mm256_broadcast_ss(&*xp1.add(k));
                a00 = _mm256_fmadd_ps(x0, wv0, a00);
                a01 = _mm256_fmadd_ps(x0, wv1, a01);
                a10 = _mm256_fmadd_ps(x1, wv0, a10);
                a11 = _mm256_fmadd_ps(x1, wv1, a11);
            }
            _mm256_storeu_ps(dp0, a00);
            _mm256_storeu_ps(dp0.add(8), a01);
            _mm256_storeu_ps(dp1, a10);
            _mm256_storeu_ps(dp1.add(8), a11);
            r += 2;
        }

        if r < rows {
            let xp0 = x_base.add(r * inp);
            let dp0 = d_base.add(r * out_pad + c);
            let (mut a0, mut a1) = if ADD_TO_DST {
                (
                    _mm256_add_ps(bv0, _mm256_loadu_ps(dp0)),
                    _mm256_add_ps(bv1, _mm256_loadu_ps(dp0.add(8))),
                )
            } else {
                (bv0, bv1)
            };
            for k in 0..inp {
                let wp = w_tile.add(k * 16);
                let x0 = _mm256_broadcast_ss(&*xp0.add(k));
                a0 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(wp), a0);
                a1 = _mm256_fmadd_ps(x0, _mm256_loadu_ps(wp.add(8)), a1);
            }
            _mm256_storeu_ps(dp0, a0);
            _mm256_storeu_ps(dp0.add(8), a1);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn linear_packed_wasm<const ADD_TO_DST: bool>(x: &[f32], rows: usize, lin: &PackedLinear, dst: &mut [f32]) {
    use core::arch::wasm32::*;
    let inp = lin.inp;
    let out_pad = lin.out_pad;
    let tiles = out_pad / 16;
    assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let b_base = lin.b.as_ptr();
    let d_base = dst.as_mut_ptr();

    for tile in 0..tiles {
        let c = tile * 16;
        let w_tile = unsafe { w_base.add(tile * inp * 16) };
        let (bv0, bv1, bv2, bv3) = unsafe {
            (
                v128_load(b_base.add(c) as *const v128),
                v128_load(b_base.add(c + 4) as *const v128),
                v128_load(b_base.add(c + 8) as *const v128),
                v128_load(b_base.add(c + 12) as *const v128),
            )
        };

        let mut r = 0usize;
        // 4-row x 8-col sub-tiles (0..8 then 8..16) to keep live accumulators in registers with 0 hsums.
        while r + 4 <= rows {
            for (half, b0, b1) in [(0usize, bv0, bv1), (8usize, bv2, bv3)] {
                unsafe {
                    let xp0 = x_base.add(r * inp);
                    let xp1 = x_base.add((r + 1) * inp);
                    let xp2 = x_base.add((r + 2) * inp);
                    let xp3 = x_base.add((r + 3) * inp);
                    let dp0 = d_base.add(r * out_pad + c + half);
                    let dp1 = d_base.add((r + 1) * out_pad + c + half);
                    let dp2 = d_base.add((r + 2) * out_pad + c + half);
                    let dp3 = d_base.add((r + 3) * out_pad + c + half);
                    let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = if ADD_TO_DST {
                        (
                            f32x4_add(b0, v128_load(dp0 as *const v128)),
                            f32x4_add(b1, v128_load(dp0.add(4) as *const v128)),
                            f32x4_add(b0, v128_load(dp1 as *const v128)),
                            f32x4_add(b1, v128_load(dp1.add(4) as *const v128)),
                            f32x4_add(b0, v128_load(dp2 as *const v128)),
                            f32x4_add(b1, v128_load(dp2.add(4) as *const v128)),
                            f32x4_add(b0, v128_load(dp3 as *const v128)),
                            f32x4_add(b1, v128_load(dp3.add(4) as *const v128)),
                        )
                    } else {
                        (b0, b1, b0, b1, b0, b1, b0, b1)
                    };
                    for k in 0..inp {
                        let wp = w_tile.add(k * 16 + half);
                        let wv0 = v128_load(wp as *const v128);
                        let wv1 = v128_load(wp.add(4) as *const v128);
                        let x0 = f32x4_splat(*xp0.add(k));
                        let x1 = f32x4_splat(*xp1.add(k));
                        let x2 = f32x4_splat(*xp2.add(k));
                        let x3 = f32x4_splat(*xp3.add(k));
                        a00 = f32x4_add(a00, f32x4_mul(x0, wv0));
                        a01 = f32x4_add(a01, f32x4_mul(x0, wv1));
                        a10 = f32x4_add(a10, f32x4_mul(x1, wv0));
                        a11 = f32x4_add(a11, f32x4_mul(x1, wv1));
                        a20 = f32x4_add(a20, f32x4_mul(x2, wv0));
                        a21 = f32x4_add(a21, f32x4_mul(x2, wv1));
                        a30 = f32x4_add(a30, f32x4_mul(x3, wv0));
                        a31 = f32x4_add(a31, f32x4_mul(x3, wv1));
                    }
                    v128_store(dp0 as *mut v128, a00);
                    v128_store(dp0.add(4) as *mut v128, a01);
                    v128_store(dp1 as *mut v128, a10);
                    v128_store(dp1.add(4) as *mut v128, a11);
                    v128_store(dp2 as *mut v128, a20);
                    v128_store(dp2.add(4) as *mut v128, a21);
                    v128_store(dp3 as *mut v128, a30);
                    v128_store(dp3.add(4) as *mut v128, a31);
                }
            }
            r += 4;
        }

        while r < rows {
            unsafe {
                let xp0 = x_base.add(r * inp);
                let dp0 = d_base.add(r * out_pad + c);
                let (mut a0, mut a1, mut a2, mut a3) = if ADD_TO_DST {
                    (
                        f32x4_add(bv0, v128_load(dp0 as *const v128)),
                        f32x4_add(bv1, v128_load(dp0.add(4) as *const v128)),
                        f32x4_add(bv2, v128_load(dp0.add(8) as *const v128)),
                        f32x4_add(bv3, v128_load(dp0.add(12) as *const v128)),
                    )
                } else {
                    (bv0, bv1, bv2, bv3)
                };
                for k in 0..inp {
                    let wp = w_tile.add(k * 16);
                    let x0 = f32x4_splat(*xp0.add(k));
                    a0 = f32x4_add(a0, f32x4_mul(x0, v128_load(wp as *const v128)));
                    a1 = f32x4_add(a1, f32x4_mul(x0, v128_load(wp.add(4) as *const v128)));
                    a2 = f32x4_add(a2, f32x4_mul(x0, v128_load(wp.add(8) as *const v128)));
                    a3 = f32x4_add(a3, f32x4_mul(x0, v128_load(wp.add(12) as *const v128)));
                }
                v128_store(dp0 as *mut v128, a0);
                v128_store(dp0.add(4) as *mut v128, a1);
                v128_store(dp0.add(8) as *mut v128, a2);
                v128_store(dp0.add(12) as *mut v128, a3);
            }
            r += 1;
        }
    }
}

// ============================================================================
// Minimal zero-dependency ONNX Protobuf initializer reader
// ============================================================================

struct ProtoReader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn read_varint(&mut self) -> Result<u64, String> {
        let mut out = 0u64;
        let mut shift = 0u32;
        while self.pos < self.buf.len() {
            let byte = self.buf[self.pos];
            self.pos += 1;
            out |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(out);
            }
            shift += 7;
            if shift >= 64 {
                return Err("malformed protobuf varint".to_owned());
            }
        }
        Err("unexpected EOF reading protobuf varint".to_owned())
    }

    fn read_bytes(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint()? as usize;
        let end = self.pos.checked_add(len).ok_or_else(|| "protobuf length overflow".to_owned())?;
        if end > self.buf.len() {
            return Err("unexpected EOF reading protobuf bytes".to_owned());
        }
        let slice = &self.buf[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn skip_field(&mut self, wire_type: u64) -> Result<(), String> {
        match wire_type {
            0 => {
                self.read_varint()?;
                Ok(())
            }
            1 => {
                let end = self.pos.checked_add(8).ok_or_else(|| "protobuf offset overflow".to_owned())?;
                if end > self.buf.len() {
                    return Err("unexpected EOF skipping 64-bit field".to_owned());
                }
                self.pos = end;
                Ok(())
            }
            2 => {
                self.read_bytes()?;
                Ok(())
            }
            5 => {
                let end = self.pos.checked_add(4).ok_or_else(|| "protobuf offset overflow".to_owned())?;
                if end > self.buf.len() {
                    return Err("unexpected EOF skipping 32-bit field".to_owned());
                }
                self.pos = end;
                Ok(())
            }
            _ => Err(format!("unsupported protobuf wire type {wire_type}")),
        }
    }
}

fn parse_onnx_initializers(bytes: &[u8]) -> Result<std::collections::HashMap<String, OnnxTensor>, String> {
    let mut model = ProtoReader::new(bytes);
    let mut graph_bytes = None;
    while !model.is_eof() {
        let tag = model.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        if field == 7 && wire == 2 {
            graph_bytes = Some(model.read_bytes()?);
            break;
        }
        model.skip_field(wire)?;
    }
    let graph_bytes = graph_bytes.ok_or_else(|| "ONNX ModelProto missing graph field".to_owned())?;

    let mut graph = ProtoReader::new(graph_bytes);
    let mut out = std::collections::HashMap::new();
    while !graph.is_eof() {
        let tag = graph.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        if field == 5 && wire == 2 {
            let tensor = parse_tensor_proto(graph.read_bytes()?)?;
            if !tensor.name.is_empty() {
                out.insert(tensor.name.clone(), tensor);
            }
        } else {
            graph.skip_field(wire)?;
        }
    }
    Ok(out)
}

fn parse_tensor_proto(bytes: &[u8]) -> Result<OnnxTensor, String> {
    let mut reader = ProtoReader::new(bytes);
    let mut dims = Vec::new();
    let mut data_type = 0i32;
    let mut float_data = Vec::new();
    let mut name = String::new();
    let mut raw_data: &[u8] = &[];
    let mut data_location = 0i32;

    while !reader.is_eof() {
        let tag = reader.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 0) => dims.push(reader.read_varint()? as usize),
            (1, 2) => {
                let mut packed = ProtoReader::new(reader.read_bytes()?);
                while !packed.is_eof() {
                    dims.push(packed.read_varint()? as usize);
                }
            }
            (2, 0) => data_type = reader.read_varint()? as i32,
            (4, 5) => {
                let b = reader.buf.get(reader.pos..reader.pos + 4).ok_or_else(|| "EOF in float_data".to_owned())?;
                reader.pos += 4;
                float_data.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
            }
            (4, 2) => {
                let bytes = reader.read_bytes()?;
                for chunk in bytes.chunks_exact(4) {
                    float_data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (8, 2) => {
                let s = reader.read_bytes()?;
                name = std::str::from_utf8(s)
                    .map_err(|_| "non-UTF8 tensor name in ONNX model".to_owned())?
                    .to_owned();
            }
            (9, 2) => raw_data = reader.read_bytes()?,
            (14, 0) => data_location = reader.read_varint()? as i32,
            _ => reader.skip_field(wire)?,
        }
    }

    if data_location == 1 {
        return Err(format!("ONNX tensor '{name}' uses unsupported external data"));
    }

    let expected_len: usize = dims.iter().product();
    let data = if !raw_data.is_empty() {
        match data_type {
            1 => {
                if raw_data.len() != expected_len * 4 {
                    return Err(format!("ONNX FLOAT tensor '{name}' size mismatch"));
                }
                raw_data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            }
            10 => {
                if raw_data.len() != expected_len * 2 {
                    return Err(format!("ONNX FLOAT16 tensor '{name}' size mismatch"));
                }
                raw_data
                    .chunks_exact(2)
                    .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                    .collect()
            }
            other => {
                if !name.is_empty() && expected_len > 0 && other != 1 && other != 10 {
                    // Non-float initializers (e.g. int64 constants) are unused for Indra CNN weights.
                    Vec::new()
                } else {
                    Vec::new()
                }
            }
        }
    } else if !float_data.is_empty() {
        float_data
    } else {
        Vec::new()
    };

    Ok(OnnxTensor { name, dims, data })
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1f);
    let mant = u32::from(bits & 0x3ff);
    let out = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let shift = mant.leading_zeros() - 21;
            let m = (mant << shift) & 0x3ff;
            let e = 127 - 14 - shift;
            sign | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        sign | (0xff << 23) | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

#[cfg(target_arch = "wasm32")]
mod wasm_abi {
    use super::{IndraNative, IndraScratch};
    use std::alloc::{alloc as heap_alloc, dealloc as heap_dealloc, Layout};

    struct WasmIndra {
        model: IndraNative,
        scratch: IndraScratch,
    }

    #[no_mangle]
    pub extern "C" fn alloc(len: usize) -> *mut u8 {
        let layout = Layout::from_size_align(len.max(1), 16).unwrap();
        // SAFETY: layout size is >= 1 and 16-byte aligned.
        unsafe { heap_alloc(layout) }
    }

    #[no_mangle]
    pub extern "C" fn dealloc(ptr: *mut u8, len: usize) {
        if !ptr.is_null() {
            // SAFETY: matches Layout passed to alloc(len).
            let layout = unsafe { Layout::from_size_align_unchecked(len.max(1), 16) };
            unsafe { heap_dealloc(ptr, layout) };
        }
    }

    #[no_mangle]
    pub extern "C" fn indra_load(ptr: *const u8, len: usize) -> usize {
        if ptr.is_null() || len == 0 {
            return 0;
        }
        // SAFETY: caller provides `len` readable bytes allocated in linear memory.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        match IndraNative::from_onnx_bytes(bytes) {
            Ok(model) => Box::into_raw(Box::new(WasmIndra {
                model,
                scratch: IndraScratch::default(),
            })) as usize,
            Err(_) => 0,
        }
    }

    #[no_mangle]
    pub extern "C" fn indra_free(handle: usize) {
        if handle != 0 {
            // SAFETY: handle was created by Box::into_raw in indra_load.
            unsafe { drop(Box::from_raw(handle as *mut WasmIndra)) };
        }
    }

    #[no_mangle]
    pub extern "C" fn indra_recognize_u8(
        handle: usize,
        gray_ptr: *const u8,
        width: usize,
        out_ids_ptr: *mut u32,
        out_cap: usize,
    ) -> usize {
        if handle == 0 || gray_ptr.is_null() || width == 0 {
            return 0;
        }
        // SAFETY: handle stays live until indra_free; caller slices live in WASM memory.
        let state = unsafe { &mut *(handle as *mut WasmIndra) };
        let gray = unsafe { std::slice::from_raw_parts(gray_ptr, width.saturating_mul(32)) };
        let (steps, step_labels, _) =
            state.model.recognize_u8(gray, width, false, &mut state.scratch);
        if !out_ids_ptr.is_null() && out_cap > 0 {
            let count = steps.min(out_cap);
            let out = unsafe { std::slice::from_raw_parts_mut(out_ids_ptr, count) };
            for (dst, lbl) in out.iter_mut().zip(&step_labels[..count]) {
                *dst = lbl.unwrap_or(0) as u32;
            }
        }
        steps
    }
}
