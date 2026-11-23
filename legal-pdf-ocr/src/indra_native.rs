//! Pure-Rust SIMD inference for Indra CTC line recognition (AVX2+FMA on x86_64, SIMD128 on wasm32,
//! scalar fallback elsewhere), reading weights directly from Indra ONNX checkpoints with zero ORT dependency.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

pub(crate) const HARD_SWISH_SIXTH: f32 = 1.0 / 6.0;
const BLANK_LABEL: usize = 0;

/// Minimal decoded ONNX float tensor.
pub(crate) struct OnnxTensor {
    pub(crate) name: String,
    pub(crate) dims: Vec<usize>,
    pub(crate) data: Vec<f32>,
}

/// Minimal decoded ONNX graph node metadata (op type, input/output names, optional tensor attribute).
pub(crate) struct OnnxNode {
    pub(crate) op_type: String,
    pub(crate) inputs: Vec<String>,
    pub(crate) outputs: Vec<String>,
    pub(crate) attr_t: Option<OnnxTensor>,
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

#[cfg(target_arch = "x86_64")]
pub(crate) struct PackedQ8 {
    /// Per-4-ic per-8-oc interleaved symmetric `i8` weights:
    /// `[tiles16, inp / 4, 2, 8, 4]` followed by optional `[inp / 4, 8, 4]` tail8.
    w_q: Vec<i8>,
    /// Per-output-channel weight scale (`max_abs / 127.0`), padded to `out_pad`.
    scale_w: Vec<f32>,
    /// Pre-scaled column sum (`scale_w[oc] * sum_k(w_q[oc, k])`), for zero-point correction.
    w_sum_scaled: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
impl PackedQ8 {
    fn pack(raw_w: &[f32], out_dim: usize, out_pad: usize, inp: usize) -> Option<Self> {
        if inp < 64 || !inp.is_multiple_of(4) || out_pad < 64 {
            return None;
        }
        let k4_count = inp / 4;
        let tiles16 = out_pad / 16;
        let has_tail8 = (out_pad & 8) != 0;
        let mut w_q = vec![0_i8; out_pad * inp];
        let mut scale_w = vec![0.0_f32; out_pad];
        let mut w_sum_scaled = vec![0.0_f32; out_pad];

        let mut encode_channel = |oc: usize, dst_oc_fn: &mut dyn FnMut(usize, i8)| {
            if oc >= out_dim {
                return;
            }
            let src_row = &raw_w[oc * inp..(oc + 1) * inp];
            let max_abs = src_row.iter().fold(0.0_f32, |m, &v| m.max(v.abs()));
            if max_abs == 0.0 {
                return;
            }
            let sw = max_abs / 127.0;
            let inv_sw = 127.0 / max_abs;
            scale_w[oc] = sw;
            let mut sum_q = 0_i32;
            for (k, &v) in src_row.iter().enumerate() {
                let q = (v * inv_sw).round().clamp(-127.0, 127.0) as i8;
                sum_q += i32::from(q);
                dst_oc_fn(k, q);
            }
            w_sum_scaled[oc] = sw * (sum_q as f32);
        };

        for tile in 0..tiles16 {
            let base = tile * k4_count * 64;
            for half in 0..2 {
                for lane in 0..8 {
                    let oc = tile * 16 + half * 8 + lane;
                    encode_channel(oc, &mut |k, q| {
                        let k4 = k >> 2;
                        let r4 = k & 3;
                        w_q[base + k4 * 64 + half * 32 + lane * 4 + r4] = q;
                    });
                }
            }
        }
        if has_tail8 {
            let base = tiles16 * k4_count * 64;
            for lane in 0..8 {
                let oc = tiles16 * 16 + lane;
                encode_channel(oc, &mut |k, q| {
                    let k4 = k >> 2;
                    let r4 = k & 3;
                    w_q[base + k4 * 32 + lane * 4 + r4] = q;
                });
            }
        }

        Some(Self {
            w_q,
            scale_w,
            w_sum_scaled,
        })
    }
}

pub(crate) struct PackedLinear {
    /// `[tiles16, inp, 16]` followed by an optional `[inp, 8]` tail tile when `out_pad % 16 == 8`,
    /// so AVX2 (`2 x f32x8` / `1 x f32x8`) and Wasm SIMD128 (`4 x f32x4` / `2 x f32x4`) load
    /// contiguous output-channel vectors with zero horizontal reductions.
    pub(crate) w: Vec<f32>,
    pub(crate) b: Vec<f32>,
    pub(crate) out_pad: usize,
    pub(crate) inp: usize,
    #[cfg(target_arch = "x86_64")]
    q8: Option<PackedQ8>,
}

impl PackedLinear {
    pub(crate) fn pack_fp32(raw_w: &[f32], raw_b: &[f32], out_dim: usize, inp: usize) -> Self {
        Self::pack_with_q8(raw_w, raw_b, out_dim, inp, false)
    }

    pub(crate) fn pack_with_q8(
        raw_w: &[f32],
        raw_b: &[f32],
        out_dim: usize,
        inp: usize,
        allow_q8: bool,
    ) -> Self {
        let out_pad = out_dim.div_ceil(8) * 8;
        let tiles16 = out_pad / 16;
        let has_tail8 = (out_pad & 8) != 0;
        let mut w = vec![0.0_f32; out_pad * inp];
        for tile in 0..tiles16 {
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
        if has_tail8 {
            let base = tiles16 * inp * 16;
            for lane in 0..8 {
                let oc = tiles16 * 16 + lane;
                if oc < out_dim {
                    let src_row = &raw_w[oc * inp..(oc + 1) * inp];
                    for (k, &val) in src_row.iter().enumerate() {
                        w[base + k * 8 + lane] = val;
                    }
                }
            }
        }
        let mut b = vec![0.0_f32; out_pad];
        b[..out_dim].copy_from_slice(&raw_b[..out_dim]);
        #[cfg(target_arch = "x86_64")]
        let q8 = allow_q8.then(|| PackedQ8::pack(raw_w, out_dim, out_pad, inp)).flatten();
        #[cfg(not(target_arch = "x86_64"))]
        let _ = allow_q8;
        Self {
            w,
            b,
            out_pad,
            inp,
            #[cfg(target_arch = "x86_64")]
            q8,
        }
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

struct ShortlistHead {
    fast: PackedLinear,
    fast_ids: Vec<usize>,
    tail: PackedLinear,
    tail_ids: Vec<usize>,
}

/// Loaded Indra 1D-CNN CTC recognizer model ready for zero-allocation per-line SIMD inference.
pub struct IndraNative {
    channels: usize,
    vocab_size: usize,
    /// Fused `Conv2d(1 -> C, k=[32, 4])` + optional `BatchNormalization`, packed `[C_pad/16, 128, 16]`.
    stem: PackedLinear,
    blocks: Vec<BlockWeights>,
    head: PackedLinear,
    shortlist: Option<ShortlistHead>,
}

impl IndraNative {
    /// Load an Indra ONNX model file directly from disk (not compiled on `wasm32`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_onnx_path(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read Indra ONNX model {}: {err}", path.display()))?;
        Self::from_onnx_bytes(&bytes)
    }

    #[cfg(all(test, not(target_arch = "wasm32")))]
    pub(crate) fn from_onnx_path_fp32(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read Indra ONNX model {}: {err}", path.display()))?;
        Self::from_onnx_bytes_with_q8(&bytes, false)
    }

    /// Parse initializers directly from an Indra ONNX protobuf byte slice, fuse `stem_bn`, and pack weights.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, String> {
        #[cfg(target_arch = "x86_64")]
        let allow_q8 = use_int8_gemm();
        #[cfg(not(target_arch = "x86_64"))]
        let allow_q8 = false;
        Self::from_onnx_bytes_with_q8(bytes, allow_q8)
    }

    fn from_onnx_bytes_with_q8(bytes: &[u8], allow_q8: bool) -> Result<Self, String> {
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
        let stem = PackedLinear::pack_with_q8(&stem_w, &stem_b, channels, 128, allow_q8);

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
                expand: PackedLinear::pack_with_q8(&expand_w_t.data, &expand_b, expand_channels, channels, allow_q8),
                fc1: PackedLinear::pack_fp32(&fc1_w_t.data, &fc1_b, se_channels, expand_channels),
                fc2: PackedLinear::pack_fp32(&fc2_w_t.data, &fc2_b, expand_channels, se_channels),
                project: PackedLinear::pack_with_q8(&project_w_t.data, &project_b, channels, expand_channels, allow_q8),
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
        let head = PackedLinear::pack_with_q8(&head_w_t.data, &head_b, vocab_size, channels, allow_q8);
        const PUNCT_EXT: [usize; 8] = [834, 835, 838, 839, 842, 843, 848, 892];
        let shortlist = if vocab_size > 256 {
            let mut in_fast = vec![false; vocab_size];
            let mut fast_ids = Vec::with_capacity(264);
            for id in 0..256 {
                in_fast[id] = true;
                fast_ids.push(id);
            }
            for &id in &PUNCT_EXT {
                if id < vocab_size && !in_fast[id] {
                    in_fast[id] = true;
                    fast_ids.push(id);
                }
            }
            let tail_ids: Vec<usize> = (0..vocab_size).filter(|&id| !in_fast[id]).collect();
            let pack_subset = |ids: &[usize]| -> PackedLinear {
                let mut w_sub = Vec::with_capacity(ids.len() * channels);
                let mut b_sub = Vec::with_capacity(ids.len());
                for &oc in ids {
                    w_sub.extend_from_slice(&head_w_t.data[oc * channels..(oc + 1) * channels]);
                    b_sub.push(head_b[oc]);
                }
                PackedLinear::pack_with_q8(&w_sub, &b_sub, ids.len(), channels, allow_q8)
            };
            Some(ShortlistHead {
                fast: pack_subset(&fast_ids),
                fast_ids,
                tail: pack_subset(&tail_ids),
                tail_ids,
            })
        } else {
            None
        };

        Ok(Self {
            channels,
            vocab_size,
            stem,
            blocks,
            head,
            shortlist,
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
        linear_packed::<false, false>(&scratch.a, timesteps, &self.head, &mut scratch.logits_block);
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
        linear_packed::<false, true>(&scratch.patches, timesteps, &self.stem, &mut scratch.a);

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

            linear_packed::<false, false>(&scratch.b[..timesteps * c], timesteps, &block.expand, &mut scratch.exp[..timesteps * e]);
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
            linear_packed::<false, false>(&scratch.se_mean[..e], 1, &block.fc1, &mut scratch.se_mid[..s]);
            for v in &mut scratch.se_mid[..s] {
                *v = v.max(0.0);
            }
            linear_packed::<false, false>(&scratch.se_mid[..s], 1, &block.fc2, &mut scratch.se_gate[..e]);
            for v in &mut scratch.se_gate[..e] {
                *v = 1.0 / (1.0 + (-*v).exp());
            }

            scale_channels_inplace(&mut scratch.exp[..timesteps * e], timesteps, e, &scratch.se_gate[..e]);

            linear_packed::<true, false>(&scratch.exp[..timesteps * e], timesteps, &block.project, &mut scratch.a[..timesteps * c]);
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
        let chunk_rows = 24.min(timesteps);
        let mut step_labels = Vec::with_capacity(timesteps);
        let mut ctc_spans: Vec<(usize, f32)> = Vec::new();
        let mut previous = usize::MAX;

        if !need_confidence {
            if let Some(sl) = &self.shortlist {
                let fast_stride = sl.fast.out_pad;
                let fast_len = sl.fast_ids.len();
                let tail_stride = sl.tail.out_pad;
                let tail_len = sl.tail_ids.len();
                scratch.logits_block.resize(chunk_rows * fast_stride, 0.0);
                scratch.b.resize(tail_stride, 0.0);

                let mut t = 0usize;
                while t < timesteps {
                    let rows = (timesteps - t).min(chunk_rows);
                    let x_slice = &scratch.a[t * c..(t + rows) * c];
                    let out_slice = &mut scratch.logits_block[..rows * fast_stride];
                    linear_packed::<false, false>(x_slice, rows, &sl.fast, out_slice);

                    for row_idx in 0..rows {
                        let row = &out_slice[row_idx * fast_stride..row_idx * fast_stride + fast_len];
                        let (fast_pos, best_val) = argmax_f32(row);
                        let mut best_label = sl.fast_ids[fast_pos];
                        if best_val < 2.0 && tail_len > 0 {
                            let x_row = &x_slice[row_idx * c..(row_idx + 1) * c];
                            linear_packed::<false, false>(x_row, 1, &sl.tail, &mut scratch.b[..tail_stride]);
                            let (tail_pos, tail_val) = argmax_f32(&scratch.b[..tail_len]);
                            if tail_val > best_val {
                                best_label = sl.tail_ids[tail_pos];
                            }
                        }
                        step_labels.push((best_label != BLANK_LABEL).then_some(best_label));
                        if best_label != BLANK_LABEL && best_label != previous {
                            ctc_spans.push((best_label, 1.0));
                        }
                        previous = best_label;
                    }
                    t += rows;
                }
                return (timesteps, step_labels, ctc_spans);
            }
        }

        let head_stride = self.head.out_pad;
        scratch.logits_block.resize(chunk_rows * head_stride, 0.0);

        let mut t = 0usize;
        while t < timesteps {
            let rows = (timesteps - t).min(chunk_rows);
            let x_slice = &scratch.a[t * c..(t + rows) * c];
            let out_slice = &mut scratch.logits_block[..rows * head_stride];
            linear_packed::<false, false>(x_slice, rows, &self.head, out_slice);

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

#[cfg(target_arch = "x86_64")]
#[inline]
fn use_int8_gemm() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("LEGALPDF_OCR_INT8")
            .ok()
            .map(|v| !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "off" | "no"))
            .unwrap_or(true)
    })
}

/// Outer-product GEMM over weights pre-packed as `[tiles16, inp, 16]` (+ optional `[inp, 8]`)
/// with stride `lin.out_pad`, optionally fusing residual addition and/or HardSwish before storing.
pub(crate) fn linear_packed<const ADD_TO_DST: bool, const HARDSWISH: bool>(
    x: &[f32],
    rows: usize,
    lin: &PackedLinear,
    dst: &mut [f32],
) {
    if rows == 0 || lin.out_pad == 0 {
        return;
    }
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        if let Some(ref q8) = lin.q8 {
            unsafe { linear_packed_avx2_q8::<ADD_TO_DST, HARDSWISH>(x, rows, lin, q8, dst) };
            return;
        }
        unsafe { linear_packed_avx2::<ADD_TO_DST, HARDSWISH>(x, rows, lin, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        linear_packed_wasm::<ADD_TO_DST, HARDSWISH>(x, rows, lin, dst);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let inp = lin.inp;
        let out_pad = lin.out_pad;
        let tiles = out_pad / 16;
        let has_tail8 = (out_pad & 8) != 0;
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
                if HARDSWISH {
                    for lane in 0..16 {
                        dr[lane] = hardswish_scalar(dr[lane]);
                    }
                }
            }
        }
        if has_tail8 {
            let c = tiles * 16;
            let w_tail = &lin.w[tiles * inp * 16..];
            let b_tail = &lin.b[c..c + 8];
            for r in 0..rows {
                let xr = &x[r * inp..(r + 1) * inp];
                let dr = &mut dst[r * out_pad + c..r * out_pad + c + 8];
                if ADD_TO_DST {
                    for lane in 0..8 {
                        dr[lane] += b_tail[lane];
                    }
                } else {
                    dr.copy_from_slice(b_tail);
                }
                for (k, &xv) in xr.iter().enumerate() {
                    let w_k = &w_tail[k * 8..k * 8 + 8];
                    for lane in 0..8 {
                        dr[lane] += xv * w_k[lane];
                    }
                }
                if HARDSWISH {
                    for lane in 0..8 {
                        dr[lane] = hardswish_scalar(dr[lane]);
                    }
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn store_avx2<const HARDSWISH: bool>(
    dp: *mut f32,
    v: std::arch::x86_64::__m256,
    zero: std::arch::x86_64::__m256,
    three: std::arch::x86_64::__m256,
    six: std::arch::x86_64::__m256,
    sixth: std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let out = if HARDSWISH {
        let gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(v, three), zero), six);
        _mm256_mul_ps(v, _mm256_mul_ps(gate, sixth))
    } else {
        v
    };
    _mm256_storeu_ps(dp, out);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_packed_avx2<const ADD_TO_DST: bool, const HARDSWISH: bool>(
    x: &[f32],
    rows: usize,
    lin: &PackedLinear,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let inp = lin.inp;
    let out_pad = lin.out_pad;
    let tiles = out_pad / 16;
    let has_tail8 = (out_pad & 8) != 0;
    assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let b_base = lin.b.as_ptr();
    let d_base = dst.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let three = _mm256_set1_ps(3.0);
    let six = _mm256_set1_ps(6.0);
    let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);

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

            store_avx2::<HARDSWISH>(dp0, a00, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp0.add(8), a01, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp1, a10, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp1.add(8), a11, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp2, a20, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp2.add(8), a21, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp3, a30, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp3.add(8), a31, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp4, a40, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp4.add(8), a41, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp5, a50, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp5.add(8), a51, zero, three, six, sixth);
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
            store_avx2::<HARDSWISH>(dp0, a00, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp0.add(8), a01, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp1, a10, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp1.add(8), a11, zero, three, six, sixth);
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
            store_avx2::<HARDSWISH>(dp0, a0, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp0.add(8), a1, zero, three, six, sixth);
        }
    }

    if has_tail8 {
        let c = tiles * 16;
        let w_tail = w_base.add(tiles * inp * 16);
        let bv = _mm256_loadu_ps(b_base.add(c));
        let mut r = 0usize;
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
            let (mut a0, mut a1, mut a2, mut a3, mut a4, mut a5) = if ADD_TO_DST {
                (
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp0)),
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp1)),
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp2)),
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp3)),
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp4)),
                    _mm256_add_ps(bv, _mm256_loadu_ps(dp5)),
                )
            } else {
                (bv, bv, bv, bv, bv, bv)
            };
            for k in 0..inp {
                let wv = _mm256_loadu_ps(w_tail.add(k * 8));
                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp0.add(k)), wv, a0);
                a1 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp1.add(k)), wv, a1);
                a2 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp2.add(k)), wv, a2);
                a3 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp3.add(k)), wv, a3);
                a4 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp4.add(k)), wv, a4);
                a5 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp5.add(k)), wv, a5);
            }
            store_avx2::<HARDSWISH>(dp0, a0, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp1, a1, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp2, a2, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp3, a3, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp4, a4, zero, three, six, sixth);
            store_avx2::<HARDSWISH>(dp5, a5, zero, three, six, sixth);
            r += 6;
        }
        while r < rows {
            let xp0 = x_base.add(r * inp);
            let dp0 = d_base.add(r * out_pad + c);
            let mut a0 = if ADD_TO_DST { _mm256_add_ps(bv, _mm256_loadu_ps(dp0)) } else { bv };
            for k in 0..inp {
                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(&*xp0.add(k)), _mm256_loadu_ps(w_tail.add(k * 8)), a0);
            }
            store_avx2::<HARDSWISH>(dp0, a0, zero, three, six, sixth);
            r += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
struct QuantRowScratch {
    x_q: Vec<u8>,
    x_scale: Vec<f32>,
    x_min: Vec<f32>,
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hmin_max_256(vmin: std::arch::x86_64::__m256, vmax: std::arch::x86_64::__m256) -> (f32, f32) {
    use std::arch::x86_64::*;
    let lo_min = _mm256_castps256_ps128(vmin);
    let hi_min = _mm256_extractf128_ps::<1>(vmin);
    let m128_min = _mm_min_ps(lo_min, hi_min);
    let m64_min = _mm_min_ps(m128_min, _mm_movehl_ps(m128_min, m128_min));
    let m32_min = _mm_min_ss(m64_min, _mm_shuffle_ps::<1>(m64_min, m64_min));

    let lo_max = _mm256_castps256_ps128(vmax);
    let hi_max = _mm256_extractf128_ps::<1>(vmax);
    let m128_max = _mm_max_ps(lo_max, hi_max);
    let m64_max = _mm_max_ps(m128_max, _mm_movehl_ps(m128_max, m128_max));
    let m32_max = _mm_max_ss(m64_max, _mm_shuffle_ps::<1>(m64_max, m64_max));
    (_mm_cvtss_f32(m32_min), _mm_cvtss_f32(m32_max))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_rows_u7_avx2(
    x: &[f32],
    rows: usize,
    inp: usize,
    scratch: &mut QuantRowScratch,
) {
    use std::arch::x86_64::*;
    scratch.x_q.resize(rows * inp, 0);
    scratch.x_scale.resize(rows, 0.0);
    scratch.x_min.resize(rows, 0.0);
    let zero_ps = _mm256_setzero_ps();
    let max127_ps = _mm256_set1_ps(127.0);

    for r in 0..rows {
        let xr = x.as_ptr().add(r * inp);
        let qr = scratch.x_q.as_mut_ptr().add(r * inp);
        let mut vmin = _mm256_set1_ps(f32::INFINITY);
        let mut vmax = _mm256_set1_ps(f32::NEG_INFINITY);
        let mut k = 0usize;
        while k + 8 <= inp {
            let v = _mm256_loadu_ps(xr.add(k));
            vmin = _mm256_min_ps(vmin, v);
            vmax = _mm256_max_ps(vmax, v);
            k += 8;
        }
        let (mut min_v, mut max_v) = hmin_max_256(vmin, vmax);
        while k < inp {
            let v = *xr.add(k);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            k += 1;
        }
        let range = max_v - min_v;
        scratch.x_min[r] = min_v;
        if range < 1e-12 {
            scratch.x_scale[r] = 0.0;
            std::ptr::write_bytes(qr, 0, inp);
            continue;
        }
        scratch.x_scale[r] = range / 127.0;
        let inv_scale = 127.0 / range;
        let vmin_ps = _mm256_set1_ps(min_v);
        let vinv_ps = _mm256_set1_ps(inv_scale);
        let mut k = 0usize;
        while k + 8 <= inp {
            let v = _mm256_loadu_ps(xr.add(k));
            let scaled = _mm256_min_ps(
                _mm256_max_ps(_mm256_mul_ps(_mm256_sub_ps(v, vmin_ps), vinv_ps), zero_ps),
                max127_ps,
            );
            let i32v = _mm256_cvtps_epi32(scaled);
            let p16 = _mm_packus_epi32(
                _mm256_castsi256_si128(i32v),
                _mm256_extracti128_si256::<1>(i32v),
            );
            let p8 = _mm_packus_epi16(p16, p16);
            (qr.add(k) as *mut i64).write_unaligned(_mm_cvtsi128_si64(p8));
            k += 8;
        }
        while k < inp {
            let q = ((*xr.add(k) - min_v) * inv_scale).round().clamp(0.0, 127.0) as u8;
            *qr.add(k) = q;
            k += 1;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn dequant_store_avx2<const ADD_TO_DST: bool, const HARDSWISH: bool>(
    dp: *mut f32,
    acc: std::arch::x86_64::__m256i,
    tot_scale: std::arch::x86_64::__m256,
    bias_adj: std::arch::x86_64::__m256,
    zero: std::arch::x86_64::__m256,
    three: std::arch::x86_64::__m256,
    six: std::arch::x86_64::__m256,
    sixth: std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;
    let res = _mm256_fmadd_ps(_mm256_cvtepi32_ps(acc), tot_scale, bias_adj);
    let val = if ADD_TO_DST {
        _mm256_add_ps(res, _mm256_loadu_ps(dp))
    } else {
        res
    };
    store_avx2::<HARDSWISH>(dp, val, zero, three, six, sixth);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_packed_avx2_q8<const ADD_TO_DST: bool, const HARDSWISH: bool>(
    x: &[f32],
    rows: usize,
    lin: &PackedLinear,
    q8: &PackedQ8,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    use std::cell::RefCell;
    thread_local! {
        static SCRATCH: RefCell<QuantRowScratch> = const {
            RefCell::new(QuantRowScratch {
                x_q: Vec::new(),
                x_scale: Vec::new(),
                x_min: Vec::new(),
            })
        };
    }

    let inp = lin.inp;
    let out_pad = lin.out_pad;
    let k4_count = inp / 4;
    let tiles16 = out_pad / 16;
    let has_tail8 = (out_pad & 8) != 0;
    assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);

    SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        quantize_rows_u7_avx2(x, rows, inp, &mut scratch);

        let xq_base = scratch.x_q.as_ptr();
        let x_scale = scratch.x_scale.as_ptr();
        let x_min = scratch.x_min.as_ptr();
        let w_base = q8.w_q.as_ptr();
        let sw_base = q8.scale_w.as_ptr();
        let ws_base = q8.w_sum_scaled.as_ptr();
        let b_base = lin.b.as_ptr();
        let d_base = dst.as_mut_ptr();

        let ones16 = _mm256_set1_epi16(1);
        let zero = _mm256_setzero_ps();
        let three = _mm256_set1_ps(3.0);
        let six = _mm256_set1_ps(6.0);
        let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);

        for tile in 0..tiles16 {
            let c = tile * 16;
            let w_tile = w_base.add(tile * k4_count * 64);
            let sw0 = _mm256_loadu_ps(sw_base.add(c));
            let sw1 = _mm256_loadu_ps(sw_base.add(c + 8));
            let ws0 = _mm256_loadu_ps(ws_base.add(c));
            let ws1 = _mm256_loadu_ps(ws_base.add(c + 8));
            let bv0 = _mm256_loadu_ps(b_base.add(c));
            let bv1 = _mm256_loadu_ps(b_base.add(c + 8));

            let mut r = 0usize;
            while r + 4 <= rows {
                let xp0 = xq_base.add(r * inp);
                let xp1 = xq_base.add((r + 1) * inp);
                let xp2 = xq_base.add((r + 2) * inp);
                let xp3 = xq_base.add((r + 3) * inp);
                let mut a00 = _mm256_setzero_si256();
                let mut a01 = _mm256_setzero_si256();
                let mut a10 = _mm256_setzero_si256();
                let mut a11 = _mm256_setzero_si256();
                let mut a20 = _mm256_setzero_si256();
                let mut a21 = _mm256_setzero_si256();
                let mut a30 = _mm256_setzero_si256();
                let mut a31 = _mm256_setzero_si256();

                for k4 in 0..k4_count {
                    let wp = w_tile.add(k4 * 64);
                    let w0 = _mm256_loadu_si256(wp as *const __m256i);
                    let w1 = _mm256_loadu_si256(wp.add(32) as *const __m256i);
                    let xv0 = _mm256_set1_epi32((xp0.add(k4 * 4) as *const i32).read_unaligned());
                    a00 = _mm256_add_epi32(a00, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w0), ones16));
                    a01 = _mm256_add_epi32(a01, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w1), ones16));
                    let xv1 = _mm256_set1_epi32((xp1.add(k4 * 4) as *const i32).read_unaligned());
                    a10 = _mm256_add_epi32(a10, _mm256_madd_epi16(_mm256_maddubs_epi16(xv1, w0), ones16));
                    a11 = _mm256_add_epi32(a11, _mm256_madd_epi16(_mm256_maddubs_epi16(xv1, w1), ones16));
                    let xv2 = _mm256_set1_epi32((xp2.add(k4 * 4) as *const i32).read_unaligned());
                    a20 = _mm256_add_epi32(a20, _mm256_madd_epi16(_mm256_maddubs_epi16(xv2, w0), ones16));
                    a21 = _mm256_add_epi32(a21, _mm256_madd_epi16(_mm256_maddubs_epi16(xv2, w1), ones16));
                    let xv3 = _mm256_set1_epi32((xp3.add(k4 * 4) as *const i32).read_unaligned());
                    a30 = _mm256_add_epi32(a30, _mm256_madd_epi16(_mm256_maddubs_epi16(xv3, w0), ones16));
                    a31 = _mm256_add_epi32(a31, _mm256_madd_epi16(_mm256_maddubs_epi16(xv3, w1), ones16));
                }

                for (ri, (acc0, acc1)) in [(0, (a00, a01)), (1, (a10, a11)), (2, (a20, a21)), (3, (a30, a31))] {
                    let row = r + ri;
                    let dp = d_base.add(row * out_pad + c);
                    let sx = _mm256_set1_ps(*x_scale.add(row));
                    let mx = _mm256_set1_ps(*x_min.add(row));
                    dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                        dp,
                        acc0,
                        _mm256_mul_ps(sx, sw0),
                        _mm256_fmadd_ps(mx, ws0, bv0),
                        zero,
                        three,
                        six,
                        sixth,
                    );
                    dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                        dp.add(8),
                        acc1,
                        _mm256_mul_ps(sx, sw1),
                        _mm256_fmadd_ps(mx, ws1, bv1),
                        zero,
                        three,
                        six,
                        sixth,
                    );
                }
                r += 4;
            }

            while r < rows {
                let xp0 = xq_base.add(r * inp);
                let dp = d_base.add(r * out_pad + c);
                let mut a0 = _mm256_setzero_si256();
                let mut a1 = _mm256_setzero_si256();
                for k4 in 0..k4_count {
                    let wp = w_tile.add(k4 * 64);
                    let w0 = _mm256_loadu_si256(wp as *const __m256i);
                    let w1 = _mm256_loadu_si256(wp.add(32) as *const __m256i);
                    let xv0 = _mm256_set1_epi32((xp0.add(k4 * 4) as *const i32).read_unaligned());
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w0), ones16));
                    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w1), ones16));
                }
                let sx = _mm256_set1_ps(*x_scale.add(r));
                let mx = _mm256_set1_ps(*x_min.add(r));
                dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                    dp,
                    a0,
                    _mm256_mul_ps(sx, sw0),
                    _mm256_fmadd_ps(mx, ws0, bv0),
                    zero,
                    three,
                    six,
                    sixth,
                );
                dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                    dp.add(8),
                    a1,
                    _mm256_mul_ps(sx, sw1),
                    _mm256_fmadd_ps(mx, ws1, bv1),
                    zero,
                    three,
                    six,
                    sixth,
                );
                r += 1;
            }
        }

        if has_tail8 {
            let c = tiles16 * 16;
            let w_tail = w_base.add(tiles16 * k4_count * 64);
            let sw = _mm256_loadu_ps(sw_base.add(c));
            let ws = _mm256_loadu_ps(ws_base.add(c));
            let bv = _mm256_loadu_ps(b_base.add(c));

            let mut r = 0usize;
            while r + 4 <= rows {
                let xp0 = xq_base.add(r * inp);
                let xp1 = xq_base.add((r + 1) * inp);
                let xp2 = xq_base.add((r + 2) * inp);
                let xp3 = xq_base.add((r + 3) * inp);
                let mut a0 = _mm256_setzero_si256();
                let mut a1 = _mm256_setzero_si256();
                let mut a2 = _mm256_setzero_si256();
                let mut a3 = _mm256_setzero_si256();
                for k4 in 0..k4_count {
                    let w0 = _mm256_loadu_si256(w_tail.add(k4 * 32) as *const __m256i);
                    let xv0 = _mm256_set1_epi32((xp0.add(k4 * 4) as *const i32).read_unaligned());
                    let xv1 = _mm256_set1_epi32((xp1.add(k4 * 4) as *const i32).read_unaligned());
                    let xv2 = _mm256_set1_epi32((xp2.add(k4 * 4) as *const i32).read_unaligned());
                    let xv3 = _mm256_set1_epi32((xp3.add(k4 * 4) as *const i32).read_unaligned());
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w0), ones16));
                    a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(_mm256_maddubs_epi16(xv1, w0), ones16));
                    a2 = _mm256_add_epi32(a2, _mm256_madd_epi16(_mm256_maddubs_epi16(xv2, w0), ones16));
                    a3 = _mm256_add_epi32(a3, _mm256_madd_epi16(_mm256_maddubs_epi16(xv3, w0), ones16));
                }
                for (ri, acc) in [(0, a0), (1, a1), (2, a2), (3, a3)] {
                    let row = r + ri;
                    let dp = d_base.add(row * out_pad + c);
                    let sx = _mm256_set1_ps(*x_scale.add(row));
                    let mx = _mm256_set1_ps(*x_min.add(row));
                    dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                        dp,
                        acc,
                        _mm256_mul_ps(sx, sw),
                        _mm256_fmadd_ps(mx, ws, bv),
                        zero,
                        three,
                        six,
                        sixth,
                    );
                }
                r += 4;
            }
            while r < rows {
                let xp0 = xq_base.add(r * inp);
                let dp = d_base.add(r * out_pad + c);
                let mut a0 = _mm256_setzero_si256();
                for k4 in 0..k4_count {
                    let w0 = _mm256_loadu_si256(w_tail.add(k4 * 32) as *const __m256i);
                    let xv0 = _mm256_set1_epi32((xp0.add(k4 * 4) as *const i32).read_unaligned());
                    a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(_mm256_maddubs_epi16(xv0, w0), ones16));
                }
                let sx = _mm256_set1_ps(*x_scale.add(r));
                let mx = _mm256_set1_ps(*x_min.add(r));
                dequant_store_avx2::<ADD_TO_DST, HARDSWISH>(
                    dp,
                    a0,
                    _mm256_mul_ps(sx, sw),
                    _mm256_fmadd_ps(mx, ws, bv),
                    zero,
                    three,
                    six,
                    sixth,
                );
                r += 1;
            }
        }
    });
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn store_wasm<const HARDSWISH: bool>(
    dp: *mut f32,
    v: core::arch::wasm32::v128,
    zero: core::arch::wasm32::v128,
    three: core::arch::wasm32::v128,
    six: core::arch::wasm32::v128,
    sixth: core::arch::wasm32::v128,
) {
    use core::arch::wasm32::*;
    let out = if HARDSWISH {
        let gate = f32x4_min(f32x4_max(f32x4_add(v, three), zero), six);
        f32x4_mul(v, f32x4_mul(gate, sixth))
    } else {
        v
    };
    v128_store(dp as *mut v128, out);
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn linear_packed_wasm<const ADD_TO_DST: bool, const HARDSWISH: bool>(
    x: &[f32],
    rows: usize,
    lin: &PackedLinear,
    dst: &mut [f32],
) {
    use core::arch::wasm32::*;
    let inp = lin.inp;
    let out_pad = lin.out_pad;
    let tiles = out_pad / 16;
    let has_tail8 = (out_pad & 8) != 0;
    assert!(x.len() >= rows * inp && dst.len() >= rows * out_pad);
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let b_base = lin.b.as_ptr();
    let d_base = dst.as_mut_ptr();
    let zero = f32x4_splat(0.0);
    let three = f32x4_splat(3.0);
    let six = f32x4_splat(6.0);
    let sixth = f32x4_splat(HARD_SWISH_SIXTH);

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
                    store_wasm::<HARDSWISH>(dp0, a00, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp0.add(4), a01, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp1, a10, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp1.add(4), a11, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp2, a20, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp2.add(4), a21, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp3, a30, zero, three, six, sixth);
                    store_wasm::<HARDSWISH>(dp3.add(4), a31, zero, three, six, sixth);
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
                store_wasm::<HARDSWISH>(dp0, a0, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp0.add(4), a1, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp0.add(8), a2, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp0.add(12), a3, zero, three, six, sixth);
            }
            r += 1;
        }
    }

    if has_tail8 {
        let c = tiles * 16;
        let w_tail = unsafe { w_base.add(tiles * inp * 16) };
        let (bv0, bv1) = unsafe {
            (
                v128_load(b_base.add(c) as *const v128),
                v128_load(b_base.add(c + 4) as *const v128),
            )
        };
        let mut r = 0usize;
        while r + 4 <= rows {
            unsafe {
                let xp0 = x_base.add(r * inp);
                let xp1 = x_base.add((r + 1) * inp);
                let xp2 = x_base.add((r + 2) * inp);
                let xp3 = x_base.add((r + 3) * inp);
                let dp0 = d_base.add(r * out_pad + c);
                let dp1 = d_base.add((r + 1) * out_pad + c);
                let dp2 = d_base.add((r + 2) * out_pad + c);
                let dp3 = d_base.add((r + 3) * out_pad + c);
                let (mut a00, mut a01, mut a10, mut a11, mut a20, mut a21, mut a30, mut a31) = if ADD_TO_DST {
                    (
                        f32x4_add(bv0, v128_load(dp0 as *const v128)),
                        f32x4_add(bv1, v128_load(dp0.add(4) as *const v128)),
                        f32x4_add(bv0, v128_load(dp1 as *const v128)),
                        f32x4_add(bv1, v128_load(dp1.add(4) as *const v128)),
                        f32x4_add(bv0, v128_load(dp2 as *const v128)),
                        f32x4_add(bv1, v128_load(dp2.add(4) as *const v128)),
                        f32x4_add(bv0, v128_load(dp3 as *const v128)),
                        f32x4_add(bv1, v128_load(dp3.add(4) as *const v128)),
                    )
                } else {
                    (bv0, bv1, bv0, bv1, bv0, bv1, bv0, bv1)
                };
                for k in 0..inp {
                    let wp = w_tail.add(k * 8);
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
                store_wasm::<HARDSWISH>(dp0, a00, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp0.add(4), a01, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp1, a10, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp1.add(4), a11, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp2, a20, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp2.add(4), a21, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp3, a30, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp3.add(4), a31, zero, three, six, sixth);
            }
            r += 4;
        }
        while r < rows {
            unsafe {
                let xp0 = x_base.add(r * inp);
                let dp0 = d_base.add(r * out_pad + c);
                let (mut a0, mut a1) = if ADD_TO_DST {
                    (
                        f32x4_add(bv0, v128_load(dp0 as *const v128)),
                        f32x4_add(bv1, v128_load(dp0.add(4) as *const v128)),
                    )
                } else {
                    (bv0, bv1)
                };
                for k in 0..inp {
                    let wp = w_tail.add(k * 8);
                    let x0 = f32x4_splat(*xp0.add(k));
                    a0 = f32x4_add(a0, f32x4_mul(x0, v128_load(wp as *const v128)));
                    a1 = f32x4_add(a1, f32x4_mul(x0, v128_load(wp.add(4) as *const v128)));
                }
                store_wasm::<HARDSWISH>(dp0, a0, zero, three, six, sixth);
                store_wasm::<HARDSWISH>(dp0.add(4), a1, zero, three, six, sixth);
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
    parse_onnx_graph(bytes, false).map(|(tensors, _)| tensors)
}

pub(crate) fn parse_onnx_model(
    bytes: &[u8],
) -> Result<(std::collections::HashMap<String, OnnxTensor>, Vec<OnnxNode>), String> {
    parse_onnx_graph(bytes, true)
}

fn parse_onnx_graph(
    bytes: &[u8],
    with_nodes: bool,
) -> Result<(std::collections::HashMap<String, OnnxTensor>, Vec<OnnxNode>), String> {
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
    let mut nodes = Vec::new();
    while !graph.is_eof() {
        let tag = graph.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        if field == 5 && wire == 2 {
            let tensor = parse_tensor_proto(graph.read_bytes()?)?;
            if !tensor.name.is_empty() {
                out.insert(tensor.name.clone(), tensor);
            }
        } else if with_nodes && field == 1 && wire == 2 {
            nodes.push(parse_node_proto(graph.read_bytes()?)?);
        } else {
            graph.skip_field(wire)?;
        }
    }
    Ok((out, nodes))
}

fn parse_node_proto(bytes: &[u8]) -> Result<OnnxNode, String> {
    let mut reader = ProtoReader::new(bytes);
    let mut op_type = String::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut attr_t = None;
    while !reader.is_eof() {
        let tag = reader.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 2) => inputs.push(
                std::str::from_utf8(reader.read_bytes()?)
                    .map_err(|_| "non-UTF8 node input in ONNX model".to_owned())?
                    .to_owned(),
            ),
            (2, 2) => outputs.push(
                std::str::from_utf8(reader.read_bytes()?)
                    .map_err(|_| "non-UTF8 node output in ONNX model".to_owned())?
                    .to_owned(),
            ),
            (4, 2) => {
                op_type = std::str::from_utf8(reader.read_bytes()?)
                    .map_err(|_| "non-UTF8 node op_type in ONNX model".to_owned())?
                    .to_owned();
            }
            (5, 2) => {
                if let Some(tensor) = parse_attribute_tensor(reader.read_bytes()?)? {
                    attr_t = Some(tensor);
                }
            }
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(OnnxNode { op_type, inputs, outputs, attr_t })
}

fn parse_attribute_tensor(bytes: &[u8]) -> Result<Option<OnnxTensor>, String> {
    let mut reader = ProtoReader::new(bytes);
    let mut attr_name = "";
    let mut tensor_bytes = None;
    while !reader.is_eof() {
        let tag = reader.read_varint()?;
        let field = tag >> 3;
        let wire = tag & 7;
        match (field, wire) {
            (1, 2) => {
                attr_name = std::str::from_utf8(reader.read_bytes()?)
                    .map_err(|_| "non-UTF8 attr name in ONNX model".to_owned())?;
            }
            (5, 2) => tensor_bytes = Some(reader.read_bytes()?),
            _ => reader.skip_field(wire)?,
        }
    }
    if attr_name == "value" {
        if let Some(tb) = tensor_bytes {
            return parse_tensor_proto(tb).map(Some);
        }
    }
    Ok(None)
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
pub mod wasm_abi {
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

#[cfg(all(test, target_arch = "x86_64"))]
mod q8_tests {
    use super::*;

    #[test]
    fn q8_gemm_tracks_fp32_across_tiles_and_modes() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        let rows = 7usize;
        let inp = 64usize;
        let out_dim = 69usize;
        let raw_w: Vec<f32> = (0..out_dim * inp)
            .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.03)
            .collect();
        let raw_b: Vec<f32> = (0..out_dim).map(|i| (i as f32 - 10.0) * 0.05).collect();
        let lin = PackedLinear::pack_with_q8(&raw_w, &raw_b, out_dim, inp, true);
        let q8 = lin.q8.as_ref().expect("PackedQ8 should be populated");
        let mut x: Vec<f32> = (0..rows * inp)
            .map(|i| ((i * 13 % 31) as f32 - 15.0) * 0.08)
            .collect();
        // Exercise constant-row fast path too.
        x[6 * inp..7 * inp].fill(-0.37);

        let initial: Vec<f32> = (0..rows * lin.out_pad)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.1)
            .collect();
        for (add, hswish) in [(false, false), (true, false), (false, true), (true, true)] {
            let mut ref_dst = initial.clone();
            let mut q8_dst = initial.clone();
            unsafe {
                match (add, hswish) {
                    (false, false) => {
                        linear_packed_avx2::<false, false>(&x, rows, &lin, &mut ref_dst);
                        linear_packed_avx2_q8::<false, false>(&x, rows, &lin, q8, &mut q8_dst);
                    }
                    (true, false) => {
                        linear_packed_avx2::<true, false>(&x, rows, &lin, &mut ref_dst);
                        linear_packed_avx2_q8::<true, false>(&x, rows, &lin, q8, &mut q8_dst);
                    }
                    (false, true) => {
                        linear_packed_avx2::<false, true>(&x, rows, &lin, &mut ref_dst);
                        linear_packed_avx2_q8::<false, true>(&x, rows, &lin, q8, &mut q8_dst);
                    }
                    (true, true) => {
                        linear_packed_avx2::<true, true>(&x, rows, &lin, &mut ref_dst);
                        linear_packed_avx2_q8::<true, true>(&x, rows, &lin, q8, &mut q8_dst);
                    }
                }
            }
            for r in 0..rows {
                for c in 0..out_dim {
                    let idx = r * lin.out_pad + c;
                    let diff = (ref_dst[idx] - q8_dst[idx]).abs();
                    assert!(
                        diff < 0.05,
                        "add={add} hswish={hswish} r={r} c={c}: ref={} q8={} diff={diff}",
                        ref_dst[idx],
                        q8_dst[idx]
                    );
                }
            }
        }
    }
}
