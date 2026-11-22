//! Pure-Rust inference for Detect-Order-Construct (`doc.safetensors`): row-wise INT8 linear layers,
//! block-local spatial RoPE self-attention, stylesheet template encoding, Order pair scoring, and
//! Construct tree node encoding.

use super::tensors::Tensor;
use std::collections::HashMap;

/// Model architecture hyperparameters and runtime decoding tables stored in `doc.safetensors` metadata.
#[derive(serde::Deserialize)]
pub struct Meta {
    pub width: usize,
    pub text_dim: usize,
    pub text_parts: usize,
    pub d: usize,
    pub classes: usize,
    pub row_width: usize,
    pub line_window: usize,
    pub line_far: [f32; 2],
    pub line_block: usize,
    pub order_block: usize,
    pub link_dim: usize,
    pub order_window: usize,
    pub order_link_dim: usize,
    #[serde(default)]
    pub order_far: HashMap<String, f32>,
    pub normalize: bool,
    pub vocab_size: usize,
    pub unknown: usize,
    pub trees: HashMap<String, super::trees::Tables>,
}

/// Outputs of Detect + Templates + spatial line-pair scoring for one document.
pub struct DetectOut {
    pub states: Vec<f32>,
    pub role: Vec<f32>,
    pub flags: Vec<f32>,
    pub succ_q: Vec<f32>,
    pub succ_k: Vec<f32>,
    pub pred_q: Vec<f32>,
    pub pred_k: Vec<f32>,
    pub table_col: Vec<f32>,
    pub speaker: Vec<f32>,
}

/// Outputs of Order for one document (`padded` region rows).
pub struct OrderOut {
    pub regions: Vec<f32>,
    pub goes_on_q: Vec<f32>,
    pub goes_on_k: Vec<f32>,
    pub goes_on_r: Vec<f32>,
    pub note_of_q: Vec<f32>,
    pub note_of_k: Vec<f32>,
    pub note_of_r: Vec<f32>,
    pub order_q: Vec<f32>,
    pub order_k: Vec<f32>,
    pub order_r: Vec<f32>,
    pub region_logits: Vec<f32>,
    pub region_start: Vec<f32>,
    pub region_doc_kind: Vec<f32>,
}

/// Outputs of Construct (`headings` or `bodies`) for `h` nodes.
pub struct ConstructOut {
    pub parent_q: Vec<f32>,
    pub parent_k: Vec<f32>,
    pub sibling_q: Vec<f32>,
    pub sibling_k: Vec<f32>,
    pub gap: Vec<f32>,
}

struct Linear {
    out: usize,
    inp: usize,
    w: Vec<f32>,
    b: Option<Vec<f32>>,
}

struct LayerNorm {
    w: Vec<f32>,
    b: Vec<f32>,
}

struct Embedding {
    rows: usize,
    dim: usize,
    w: Vec<f32>,
}

struct TransformerLayer {
    heads: usize,
    head_dim: usize,
    norm1: LayerNorm,
    norm2: LayerNorm,
    qkv: Linear,
    out: Linear,
    q_norm: LayerNorm,
    k_norm: LayerNorm,
    feed0: Linear,
    feed2: Linear,
    spatial_bias: Option<Linear>,
}

struct DetectNet {
    text0: Linear,
    text2: LayerNorm,
    pos0: Linear,
    pos2: Linear,
    pos4: LayerNorm,
    look0: Linear,
    look2: LayerNorm,
    anchor: Embedding,
    decided0: Linear,
    decided2: LayerNorm,
    fuse: Linear,
    pre: TransformerLayer,
    post: TransformerLayer,
    trans_geom0: Linear,
    trans_geom2: Linear,
    trans_fuse0: LayerNorm,
    trans_fuse1: Linear,
    trans_fuse3: Linear,
    succ_q: Linear,
    succ_k: Linear,
    succ_r0: Linear,
    succ_r2: Linear,
    pred_q: Linear,
    pred_k: Linear,
    pred_r0: Linear,
    pred_r2: Linear,
    role: Linear,
    flags: Linear,
    table_col: Linear,
    speaker: Linear,
}

struct TemplatesNet {
    row0: Linear,
    row2: LayerNorm,
    pool: Linear,
    encoder: Vec<TransformerLayer>,
    back0: Linear,
    back1: LayerNorm,
}

struct PairHeadNet {
    q: Linear,
    k: Linear,
    r0: Linear,
    r2: Linear,
    link_dim: usize,
    far: f32,
}

struct OrderNet {
    pool0: Linear,
    pool2: Linear,
    role0: Embedding,
    role2: LayerNorm,
    fuse: Linear,
    encoder: Vec<TransformerLayer>,
    goes_on: PairHeadNet,
    note_of: PairHeadNet,
    successor: PairHeadNet,
    role_head: Linear,
    start_head: Linear,
    doc_kind_head: Linear,
}

pub struct ConstructNet {
    in_proj: Option<Linear>,
    node_role: Embedding,
    node_fam: Embedding,
    node_geom: Linear,
    enclosing: Linear,
    opener_proj: Linear,
    layers: Vec<TransformerLayer>,
    parent_q: Linear,
    parent_k: Linear,
    sibling_q: Linear,
    sibling_k: Linear,
    gap: Linear,
}

/// Complete DOC neural network loaded in memory.
pub struct Net {
    detect: DetectNet,
    templates: TemplatesNet,
    order: OrderNet,
    pub headings: ConstructNet,
    pub bodies: ConstructNet,
}

fn req<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor, String> {
    tensors
        .get(name)
        .ok_or_else(|| format!("missing tensor {name}"))
}

impl Linear {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self, String> {
        let wq = req(tensors, &format!("{prefix}.w_q"))?;
        let ws = req(tensors, &format!("{prefix}.w_s"))?;
        if wq.dtype != "I8" || wq.shape.len() != 2 {
            return Err(format!("{prefix}.w_q: expected 2D I8 tensor"));
        }
        let (out, inp) = (wq.shape[0], wq.shape[1]);
        let scales = ws.f32s();
        if scales.len() != out || wq.bytes.len() != out * inp {
            return Err(format!("{prefix}: mismatched weight/scale shapes"));
        }
        let mut w = Vec::with_capacity(out * inp);
        for (r, &scale) in scales.iter().enumerate() {
            let row = &wq.bytes[r * inp..(r + 1) * inp];
            w.extend(row.iter().map(|&q| f32::from(q as i8) * scale));
        }
        let b = tensors
            .get(&format!("{prefix}.b"))
            .map(Tensor::f32s);
        Ok(Self { out, inp, w, b })
    }

    fn forward(&self, x: &[f32], n: usize, threads: usize) -> Vec<f32> {
        let mut out = vec![0f32; n * self.out];
        if n == 0 {
            return out;
        }
        linear_into(x, n, self, &mut out, threads);
        out
    }
}

impl LayerNorm {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self, String> {
        let w = req(tensors, &format!("{prefix}.w"))?.f32s();
        let b = req(tensors, &format!("{prefix}.b"))?.f32s();
        if w.len() != b.len() || w.is_empty() {
            return Err(format!("{prefix}: mismatched LayerNorm shapes"));
        }
        Ok(Self { w, b })
    }

    fn forward_into(&self, x: &[f32], out: &mut [f32]) {
        let dim = self.w.len();
        let inv_n = (dim as f32).recip();
        for (src, dst) in x.chunks_exact(dim).zip(out.chunks_exact_mut(dim)) {
            let mean = src.iter().copied().sum::<f32>() * inv_n;
            let var = src
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                * inv_n;
            let inv_std = (var + 1e-5f32).sqrt().recip();
            for j in 0..dim {
                dst[j] = (src[j] - mean) * inv_std * self.w[j] + self.b[j];
            }
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = vec![0f32; x.len()];
        self.forward_into(x, &mut out);
        out
    }

    fn forward_inplace(&self, x: &mut [f32]) {
        let dim = self.w.len();
        let inv_n = (dim as f32).recip();
        for row in x.chunks_exact_mut(dim) {
            let mean = row.iter().copied().sum::<f32>() * inv_n;
            let var = row
                .iter()
                .map(|&v| {
                    let d = v - mean;
                    d * d
                })
                .sum::<f32>()
                * inv_n;
            let inv_std = (var + 1e-5f32).sqrt().recip();
            for j in 0..dim {
                row[j] = (row[j] - mean) * inv_std * self.w[j] + self.b[j];
            }
        }
    }
}

impl Embedding {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str) -> Result<Self, String> {
        let t = req(tensors, &format!("{prefix}.w"))?;
        if t.shape.len() != 2 {
            return Err(format!("{prefix}.w: expected 2D tensor"));
        }
        Ok(Self {
            rows: t.shape[0],
            dim: t.shape[1],
            w: t.f32s(),
        })
    }

    fn row(&self, idx: i64) -> &[f32] {
        let clamped = idx.clamp(0, (self.rows.saturating_sub(1)) as i64) as usize;
        &self.w[clamped * self.dim..(clamped + 1) * self.dim]
    }
}

impl TransformerLayer {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        has_spatial_bias: bool,
    ) -> Result<Self, String> {
        let norm1 = LayerNorm::load(tensors, &format!("{prefix}.norm1"))?;
        let norm2 = LayerNorm::load(tensors, &format!("{prefix}.norm2"))?;
        let q_norm = LayerNorm::load(tensors, &format!("{prefix}.q_norm"))?;
        let k_norm = LayerNorm::load(tensors, &format!("{prefix}.k_norm"))?;
        let head_dim = q_norm.w.len();
        let d = norm1.w.len();
        if head_dim == 0 || d % head_dim != 0 {
            return Err(format!("{prefix}: invalid d={d} head_dim={head_dim}"));
        }
        let spatial_bias = has_spatial_bias
            .then(|| Linear::load(tensors, &format!("{prefix}.spatial_bias")))
            .transpose()?;
        Ok(Self {
            heads: d / head_dim,
            head_dim,
            norm1,
            norm2,
            qkv: Linear::load(tensors, &format!("{prefix}.qkv"))?,
            out: Linear::load(tensors, &format!("{prefix}.out"))?,
            q_norm,
            k_norm,
            feed0: Linear::load(tensors, &format!("{prefix}.feed.0"))?,
            feed2: Linear::load(tensors, &format!("{prefix}.feed.2"))?,
            spatial_bias,
        })
    }
}

impl PairHeadNet {
    fn load(
        tensors: &HashMap<String, Tensor>,
        prefix: &str,
        link_dim: usize,
        far: f32,
    ) -> Result<Self, String> {
        Ok(Self {
            q: Linear::load(tensors, &format!("{prefix}.q"))?,
            k: Linear::load(tensors, &format!("{prefix}.k"))?,
            r0: Linear::load(tensors, &format!("{prefix}.r.0"))?,
            r2: Linear::load(tensors, &format!("{prefix}.r.2"))?,
            link_dim,
            far,
        })
    }
}

impl ConstructNet {
    fn load(tensors: &HashMap<String, Tensor>, prefix: &str, tree_layers: usize) -> Result<Self, String> {
        let in_proj = tensors
            .contains_key(&format!("{prefix}.in_proj.w_q"))
            .then(|| Linear::load(tensors, &format!("{prefix}.in_proj")))
            .transpose()?;
        let layers = (0..tree_layers)
            .map(|i| TransformerLayer::load(tensors, &format!("{prefix}.layers.{i}"), false))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            in_proj,
            node_role: Embedding::load(tensors, &format!("{prefix}.node_role"))?,
            node_fam: Embedding::load(tensors, &format!("{prefix}.node_fam"))?,
            node_geom: Linear::load(tensors, &format!("{prefix}.node_geom"))?,
            enclosing: Linear::load(tensors, &format!("{prefix}.enclosing"))?,
            opener_proj: Linear::load(tensors, &format!("{prefix}.opener_proj"))?,
            layers,
            parent_q: Linear::load(tensors, &format!("{prefix}.parent.q"))?,
            parent_k: Linear::load(tensors, &format!("{prefix}.parent.k"))?,
            sibling_q: Linear::load(tensors, &format!("{prefix}.sibling.q"))?,
            sibling_k: Linear::load(tensors, &format!("{prefix}.sibling.k"))?,
            gap: Linear::load(tensors, &format!("{prefix}.gap"))?,
        })
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn dot4_scalar(x: &[f32], w: &[f32]) -> f32 {
    let inp = x.len();
    let mut a0 = 0.0f32;
    let mut a1 = 0.0f32;
    let mut a2 = 0.0f32;
    let mut a3 = 0.0f32;
    let mut k = 0usize;
    while k + 4 <= inp {
        a0 += x[k] * w[k];
        a1 += x[k + 1] * w[k + 1];
        a2 += x[k + 2] * w[k + 2];
        a3 += x[k + 3] * w[k + 3];
        k += 4;
    }
    let mut s = (a0 + a1) + (a2 + a3);
    while k < inp {
        s += x[k] * w[k];
        k += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline]
unsafe fn hsum256_ps(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_ps::<1>(v);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let hi64 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, hi64))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot4_avx2(x: &[f32], w: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    let inp = x.len().min(w.len());
    let xp = x.as_ptr();
    let wp = w.as_ptr();
    let mut acc = _mm256_setzero_ps();
    let mut k = 0usize;
    while k + 8 <= inp {
        let xv = _mm256_loadu_ps(xp.add(k));
        let wv = _mm256_loadu_ps(wp.add(k));
        acc = _mm256_fmadd_ps(xv, wv, acc);
        k += 8;
    }
    let mut s = hsum256_ps(acc);
    while k < inp {
        s += *xp.add(k) * *wp.add(k);
        k += 1;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn linear_chunk_avx2(x: &[f32], rows: usize, lin: &Linear, dst: &mut [f32]) {
    use std::arch::x86_64::*;
    let inp = lin.inp;
    let out_dim = lin.out;
    assert!(x.len() >= rows * inp && lin.w.len() >= out_dim * inp && dst.len() >= rows * out_dim);
    let bias = lin.b.as_deref();
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let mut r = 0usize;
    while r + 2 <= rows {
        let xp0 = x_base.add(r * inp);
        let xp1 = x_base.add((r + 1) * inp);
        let x0 = &x[r * inp..(r + 1) * inp];
        let x1 = &x[(r + 1) * inp..(r + 2) * inp];
        let (row0, rest) = dst[r * out_dim..].split_at_mut(out_dim);
        let row1 = &mut rest[..out_dim];
        let mut c = 0usize;
        while c + 4 <= out_dim {
            let wp0 = w_base.add(c * inp);
            let wp1 = w_base.add((c + 1) * inp);
            let wp2 = w_base.add((c + 2) * inp);
            let wp3 = w_base.add((c + 3) * inp);
            let mut s00 = _mm256_setzero_ps();
            let mut s01 = _mm256_setzero_ps();
            let mut s02 = _mm256_setzero_ps();
            let mut s03 = _mm256_setzero_ps();
            let mut s10 = _mm256_setzero_ps();
            let mut s11 = _mm256_setzero_ps();
            let mut s12 = _mm256_setzero_ps();
            let mut s13 = _mm256_setzero_ps();
            let mut k = 0usize;
            while k + 8 <= inp {
                let xv0 = _mm256_loadu_ps(xp0.add(k));
                let xv1 = _mm256_loadu_ps(xp1.add(k));
                let wv0 = _mm256_loadu_ps(wp0.add(k));
                let wv1 = _mm256_loadu_ps(wp1.add(k));
                let wv2 = _mm256_loadu_ps(wp2.add(k));
                let wv3 = _mm256_loadu_ps(wp3.add(k));
                s00 = _mm256_fmadd_ps(xv0, wv0, s00);
                s01 = _mm256_fmadd_ps(xv0, wv1, s01);
                s02 = _mm256_fmadd_ps(xv0, wv2, s02);
                s03 = _mm256_fmadd_ps(xv0, wv3, s03);
                s10 = _mm256_fmadd_ps(xv1, wv0, s10);
                s11 = _mm256_fmadd_ps(xv1, wv1, s11);
                s12 = _mm256_fmadd_ps(xv1, wv2, s12);
                s13 = _mm256_fmadd_ps(xv1, wv3, s13);
                k += 8;
            }
            let mut d00 = hsum256_ps(s00);
            let mut d01 = hsum256_ps(s01);
            let mut d02 = hsum256_ps(s02);
            let mut d03 = hsum256_ps(s03);
            let mut d10 = hsum256_ps(s10);
            let mut d11 = hsum256_ps(s11);
            let mut d12 = hsum256_ps(s12);
            let mut d13 = hsum256_ps(s13);
            while k < inp {
                let xv0 = *xp0.add(k);
                let xv1 = *xp1.add(k);
                let wv0 = *wp0.add(k);
                let wv1 = *wp1.add(k);
                let wv2 = *wp2.add(k);
                let wv3 = *wp3.add(k);
                d00 += xv0 * wv0;
                d01 += xv0 * wv1;
                d02 += xv0 * wv2;
                d03 += xv0 * wv3;
                d10 += xv1 * wv0;
                d11 += xv1 * wv1;
                d12 += xv1 * wv2;
                d13 += xv1 * wv3;
                k += 1;
            }
            if let Some(b) = bias {
                d00 += b[c];
                d01 += b[c + 1];
                d02 += b[c + 2];
                d03 += b[c + 3];
                d10 += b[c];
                d11 += b[c + 1];
                d12 += b[c + 2];
                d13 += b[c + 3];
            }
            row0[c] = d00;
            row0[c + 1] = d01;
            row0[c + 2] = d02;
            row0[c + 3] = d03;
            row1[c] = d10;
            row1[c + 1] = d11;
            row1[c + 2] = d12;
            row1[c + 3] = d13;
            c += 4;
        }
        while c < out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            let b = bias.map_or(0.0, |b| b[c]);
            row0[c] = dot4_avx2(x0, w) + b;
            row1[c] = dot4_avx2(x1, w) + b;
            c += 1;
        }
        r += 2;
    }
    if r < rows {
        let x0 = &x[r * inp..(r + 1) * inp];
        let row0 = &mut dst[r * out_dim..(r + 1) * out_dim];
        for c in 0..out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            row0[c] = dot4_avx2(x0, w) + bias.map_or(0.0, |b| b[c]);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn hsum128_ps(v: core::arch::wasm32::v128) -> f32 {
    use core::arch::wasm32::*;
    (f32x4_extract_lane::<0>(v) + f32x4_extract_lane::<1>(v))
        + (f32x4_extract_lane::<2>(v) + f32x4_extract_lane::<3>(v))
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
fn dot4_wasm(x: &[f32], w: &[f32]) -> f32 {
    use core::arch::wasm32::*;
    let inp = x.len().min(w.len());
    let xp = x.as_ptr();
    let wp = w.as_ptr();
    let mut acc = f32x4_splat(0.0);
    let mut k = 0usize;
    while k + 4 <= inp {
        // SAFETY: k + 4 <= inp <= min(x.len(), w.len()).
        let (xv, wv) = unsafe { (v128_load(xp.add(k) as *const v128), v128_load(wp.add(k) as *const v128)) };
        acc = f32x4_add(acc, f32x4_mul(xv, wv));
        k += 4;
    }
    let mut s = hsum128_ps(acc);
    while k < inp {
        s += x[k] * w[k];
        k += 1;
    }
    s
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn linear_chunk_wasm(x: &[f32], rows: usize, lin: &Linear, dst: &mut [f32]) {
    use core::arch::wasm32::*;
    let inp = lin.inp;
    let out_dim = lin.out;
    assert!(x.len() >= rows * inp && lin.w.len() >= out_dim * inp && dst.len() >= rows * out_dim);
    let bias = lin.b.as_deref();
    let x_base = x.as_ptr();
    let w_base = lin.w.as_ptr();
    let mut r = 0usize;
    while r + 2 <= rows {
        // SAFETY: rows * inp <= x.len(), out_dim * inp <= lin.w.len().
        let (xp0, xp1) = unsafe { (x_base.add(r * inp), x_base.add((r + 1) * inp)) };
        let x0 = &x[r * inp..(r + 1) * inp];
        let x1 = &x[(r + 1) * inp..(r + 2) * inp];
        let (row0, rest) = dst[r * out_dim..].split_at_mut(out_dim);
        let row1 = &mut rest[..out_dim];
        let mut c = 0usize;
        while c + 4 <= out_dim {
            let (wp0, wp1, wp2, wp3) = unsafe {
                (
                    w_base.add(c * inp),
                    w_base.add((c + 1) * inp),
                    w_base.add((c + 2) * inp),
                    w_base.add((c + 3) * inp),
                )
            };
            let mut s00 = f32x4_splat(0.0);
            let mut s01 = f32x4_splat(0.0);
            let mut s02 = f32x4_splat(0.0);
            let mut s03 = f32x4_splat(0.0);
            let mut s10 = f32x4_splat(0.0);
            let mut s11 = f32x4_splat(0.0);
            let mut s12 = f32x4_splat(0.0);
            let mut s13 = f32x4_splat(0.0);
            let mut k = 0usize;
            while k + 4 <= inp {
                let (xv0, xv1, wv0, wv1, wv2, wv3) = unsafe {
                    (
                        v128_load(xp0.add(k) as *const v128),
                        v128_load(xp1.add(k) as *const v128),
                        v128_load(wp0.add(k) as *const v128),
                        v128_load(wp1.add(k) as *const v128),
                        v128_load(wp2.add(k) as *const v128),
                        v128_load(wp3.add(k) as *const v128),
                    )
                };
                s00 = f32x4_add(s00, f32x4_mul(xv0, wv0));
                s01 = f32x4_add(s01, f32x4_mul(xv0, wv1));
                s02 = f32x4_add(s02, f32x4_mul(xv0, wv2));
                s03 = f32x4_add(s03, f32x4_mul(xv0, wv3));
                s10 = f32x4_add(s10, f32x4_mul(xv1, wv0));
                s11 = f32x4_add(s11, f32x4_mul(xv1, wv1));
                s12 = f32x4_add(s12, f32x4_mul(xv1, wv2));
                s13 = f32x4_add(s13, f32x4_mul(xv1, wv3));
                k += 4;
            }
            let mut d00 = hsum128_ps(s00);
            let mut d01 = hsum128_ps(s01);
            let mut d02 = hsum128_ps(s02);
            let mut d03 = hsum128_ps(s03);
            let mut d10 = hsum128_ps(s10);
            let mut d11 = hsum128_ps(s11);
            let mut d12 = hsum128_ps(s12);
            let mut d13 = hsum128_ps(s13);
            while k < inp {
                let (xv0, xv1) = (x[r * inp + k], x[(r + 1) * inp + k]);
                let wv0 = lin.w[c * inp + k];
                let wv1 = lin.w[(c + 1) * inp + k];
                let wv2 = lin.w[(c + 2) * inp + k];
                let wv3 = lin.w[(c + 3) * inp + k];
                d00 += xv0 * wv0;
                d01 += xv0 * wv1;
                d02 += xv0 * wv2;
                d03 += xv0 * wv3;
                d10 += xv1 * wv0;
                d11 += xv1 * wv1;
                d12 += xv1 * wv2;
                d13 += xv1 * wv3;
                k += 1;
            }
            if let Some(b) = bias {
                d00 += b[c];
                d01 += b[c + 1];
                d02 += b[c + 2];
                d03 += b[c + 3];
                d10 += b[c];
                d11 += b[c + 1];
                d12 += b[c + 2];
                d13 += b[c + 3];
            }
            row0[c] = d00;
            row0[c + 1] = d01;
            row0[c + 2] = d02;
            row0[c + 3] = d03;
            row1[c] = d10;
            row1[c + 1] = d11;
            row1[c + 2] = d12;
            row1[c + 3] = d13;
            c += 4;
        }
        while c < out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            let b = bias.map_or(0.0, |b| b[c]);
            row0[c] = dot4_wasm(x0, w) + b;
            row1[c] = dot4_wasm(x1, w) + b;
            c += 1;
        }
        r += 2;
    }
    if r < rows {
        let x0 = &x[r * inp..(r + 1) * inp];
        let row0 = &mut dst[r * out_dim..(r + 1) * out_dim];
        for c in 0..out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            row0[c] = dot4_wasm(x0, w) + bias.map_or(0.0, |b| b[c]);
        }
    }
}

#[inline(always)]
fn dot4(x: &[f32], w: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA availability checked above; dot4_avx2 guards bounds with min(len).
        return unsafe { dot4_avx2(x, w) };
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        dot4_wasm(x, w)
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        dot4_scalar(x, w)
    }
}

#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
#[inline(always)]
fn linear_chunk_scalar(x: &[f32], rows: usize, lin: &Linear, dst: &mut [f32]) {
    let inp = lin.inp;
    let out_dim = lin.out;
    let bias = lin.b.as_deref();
    let mut r = 0usize;
    while r + 2 <= rows {
        let x0 = &x[r * inp..(r + 1) * inp];
        let x1 = &x[(r + 1) * inp..(r + 2) * inp];
        let (row0, rest) = dst[r * out_dim..].split_at_mut(out_dim);
        let row1 = &mut rest[..out_dim];
        let mut c = 0usize;
        while c + 4 <= out_dim {
            let w0 = &lin.w[c * inp..(c + 1) * inp];
            let w1 = &lin.w[(c + 1) * inp..(c + 2) * inp];
            let w2 = &lin.w[(c + 2) * inp..(c + 3) * inp];
            let w3 = &lin.w[(c + 3) * inp..(c + 4) * inp];
            let mut s00 = [0.0f32; 4];
            let mut s01 = [0.0f32; 4];
            let mut s02 = [0.0f32; 4];
            let mut s03 = [0.0f32; 4];
            let mut s10 = [0.0f32; 4];
            let mut s11 = [0.0f32; 4];
            let mut s12 = [0.0f32; 4];
            let mut s13 = [0.0f32; 4];
            let mut k = 0usize;
            while k + 4 <= inp {
                for l in 0..4 {
                    let xv0 = x0[k + l];
                    let xv1 = x1[k + l];
                    let wv0 = w0[k + l];
                    let wv1 = w1[k + l];
                    let wv2 = w2[k + l];
                    let wv3 = w3[k + l];
                    s00[l] += xv0 * wv0;
                    s01[l] += xv0 * wv1;
                    s02[l] += xv0 * wv2;
                    s03[l] += xv0 * wv3;
                    s10[l] += xv1 * wv0;
                    s11[l] += xv1 * wv1;
                    s12[l] += xv1 * wv2;
                    s13[l] += xv1 * wv3;
                }
                k += 4;
            }
            let mut d00 = (s00[0] + s00[1]) + (s00[2] + s00[3]);
            let mut d01 = (s01[0] + s01[1]) + (s01[2] + s01[3]);
            let mut d02 = (s02[0] + s02[1]) + (s02[2] + s02[3]);
            let mut d03 = (s03[0] + s03[1]) + (s03[2] + s03[3]);
            let mut d10 = (s10[0] + s10[1]) + (s10[2] + s10[3]);
            let mut d11 = (s11[0] + s11[1]) + (s11[2] + s11[3]);
            let mut d12 = (s12[0] + s12[1]) + (s12[2] + s12[3]);
            let mut d13 = (s13[0] + s13[1]) + (s13[2] + s13[3]);
            while k < inp {
                let xv0 = x0[k];
                let xv1 = x1[k];
                let wv0 = w0[k];
                let wv1 = w1[k];
                let wv2 = w2[k];
                let wv3 = w3[k];
                d00 += xv0 * wv0;
                d01 += xv0 * wv1;
                d02 += xv0 * wv2;
                d03 += xv0 * wv3;
                d10 += xv1 * wv0;
                d11 += xv1 * wv1;
                d12 += xv1 * wv2;
                d13 += xv1 * wv3;
                k += 1;
            }
            if let Some(b) = bias {
                d00 += b[c];
                d01 += b[c + 1];
                d02 += b[c + 2];
                d03 += b[c + 3];
                d10 += b[c];
                d11 += b[c + 1];
                d12 += b[c + 2];
                d13 += b[c + 3];
            }
            row0[c] = d00;
            row0[c + 1] = d01;
            row0[c + 2] = d02;
            row0[c + 3] = d03;
            row1[c] = d10;
            row1[c + 1] = d11;
            row1[c + 2] = d12;
            row1[c + 3] = d13;
            c += 4;
        }
        while c < out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            let b = bias.map_or(0.0, |b| b[c]);
            row0[c] = dot4_scalar(x0, w) + b;
            row1[c] = dot4_scalar(x1, w) + b;
            c += 1;
        }
        r += 2;
    }
    if r < rows {
        let x0 = &x[r * inp..(r + 1) * inp];
        let row0 = &mut dst[r * out_dim..(r + 1) * out_dim];
        for c in 0..out_dim {
            let w = &lin.w[c * inp..(c + 1) * inp];
            row0[c] = dot4_scalar(x0, w) + bias.map_or(0.0, |b| b[c]);
        }
    }
}

#[inline(always)]
fn linear_chunk(x: &[f32], rows: usize, lin: &Linear, dst: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
        // SAFETY: AVX2+FMA availability checked above; linear_chunk_avx2 asserts all slice bounds up front.
        unsafe { linear_chunk_avx2(x, rows, lin, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        linear_chunk_wasm(x, rows, lin, dst);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        linear_chunk_scalar(x, rows, lin, dst);
    }
}

fn linear_into(x: &[f32], n: usize, lin: &Linear, dst: &mut [f32], threads: usize) {
    #[cfg(not(target_arch = "wasm32"))]
    if threads > 1 && n >= 256 && lin.out * lin.inp >= 16_384 {
        let workers = threads.min(n / 128).max(1);
        if workers > 1 {
            let chunk_rows = n.div_ceil(workers);
            std::thread::scope(|scope| {
                for (t, out_chunk) in dst.chunks_mut(chunk_rows * lin.out).enumerate() {
                    let row_start = t * chunk_rows;
                    let rows = out_chunk.len() / lin.out;
                    let x_chunk = &x[row_start * lin.inp..(row_start + rows) * lin.inp];
                    scope.spawn(move || linear_chunk(x_chunk, rows, lin, out_chunk));
                }
            });
            return;
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = threads;
    linear_chunk(x, n, lin, dst);
}

fn rotary(n: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let freqs: Vec<f32> = (0..half)
        .map(|k| 10_000.0f32.powf(-(2.0 * k as f32) / head_dim as f32))
        .collect();
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for pos in 0..n {
        let p = pos as f32;
        for &f in &freqs {
            let a = p * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    (cos, sin)
}

#[inline(always)]
fn apply_rope(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    for (k, (&c, &s)) in cos.iter().zip(sin).enumerate() {
        let even = v[2 * k];
        let odd = v[2 * k + 1];
        v[2 * k] = even * c - odd * s;
        v[2 * k + 1] = even * s + odd * c;
    }
}

fn feed_residual(layer: &TransformerLayer, x: &mut [f32], n: usize, threads: usize) {
    let normed = layer.norm2.forward(x);
    let mut hidden = layer.feed0.forward(&normed, n, threads);
    for v in &mut hidden {
        let r = v.max(0.0);
        *v = r * r;
    }
    let delta = layer.feed2.forward(&hidden, n, threads);
    for (dst, &src) in x.iter_mut().zip(&delta) {
        *dst += src;
    }
}

/// Split `qkv` `[n, 3*d]` into head-major `[heads, n, head_dim]` `(q, k, v)`, apply `q_norm`/`k_norm`, and optional RoPE.
fn prepare_qkv(
    layer: &TransformerLayer,
    x: &[f32],
    n: usize,
    rope: Option<(&[f32], &[f32])>,
    threads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let d = layer.norm1.w.len();
    let h = layer.heads;
    let s = layer.head_dim;
    let half = s / 2;
    let normed = layer.norm1.forward(x);
    let qkv = layer.qkv.forward(&normed, n, threads);
    let mut q = vec![0f32; h * n * s];
    let mut k = vec![0f32; h * n * s];
    let mut v = vec![0f32; h * n * s];
    for pos in 0..n {
        let base = pos * 3 * d;
        for head in 0..h {
            let off = (head * n + pos) * s;
            let h_off = head * s;
            let q_dst = &mut q[off..off + s];
            let k_dst = &mut k[off..off + s];
            let v_dst = &mut v[off..off + s];
            layer
                .q_norm
                .forward_into(&qkv[base + h_off..base + h_off + s], q_dst);
            layer
                .k_norm
                .forward_into(&qkv[base + d + h_off..base + d + h_off + s], k_dst);
            v_dst.copy_from_slice(&qkv[base + 2 * d + h_off..base + 2 * d + h_off + s]);
            if let Some((cos, sin)) = rope {
                let r_off = pos * half;
                apply_rope(q_dst, &cos[r_off..r_off + half], &sin[r_off..r_off + half]);
                apply_rope(k_dst, &cos[r_off..r_off + half], &sin[r_off..r_off + half]);
            }
        }
    }
    (q, k, v)
}

/// Half-window for Construct self-attention (`2 * 256 + 1 = 513` neighbor nodes), bounding long statutes to O(h * W).
const CONSTRUCT_WINDOW: usize = 256;

#[allow(clippy::too_many_arguments)]
fn full_attn_chunk(
    row_start: usize,
    rows: usize,
    n: usize,
    h: usize,
    s: usize,
    d: usize,
    scale: f32,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    window: Option<usize>,
    dst_chunk: &mut [f32],
) {
    let cap = window.map_or(n, |w| (2 * w + 1).min(n));
    let mut scores = vec![0f32; cap];
    for head in 0..h {
        let qh = &q[head * n * s..(head + 1) * n * s];
        let kh = &k[head * n * s..(head + 1) * n * s];
        let vh = &v[head * n * s..(head + 1) * n * s];
        for r in 0..rows {
            let i = row_start + r;
            let (lo, hi) = match window {
                Some(w) if n > 2 * w + 1 => (i.saturating_sub(w), (i + w + 1).min(n)),
                _ => (0, n),
            };
            let len = hi - lo;
            let qi = &qh[i * s..(i + 1) * s];
            let mut top = f32::NEG_INFINITY;
            for (idx, j) in (lo..hi).enumerate() {
                let sc = dot4(qi, &kh[j * s..(j + 1) * s]) * scale;
                scores[idx] = sc;
                if sc > top {
                    top = sc;
                }
            }
            let mut total = 0.0f32;
            for sc in &mut scores[..len] {
                let e = (*sc - top).exp();
                *sc = e;
                total += e;
            }
            let inv = total.recip();
            let dst = &mut dst_chunk[r * d + head * s..r * d + (head + 1) * s];
            for (idx, &w) in scores[..len].iter().enumerate() {
                let p = w * inv;
                let vj = &vh[(lo + idx) * s..(lo + idx + 1) * s];
                for c in 0..s {
                    dst[c] += p * vj[c];
                }
            }
        }
    }
}

/// Self-attention layer (`Templates` global when `window.is_none()`, `Construct` windowed when `window.is_some()`).
fn run_full_layer(
    layer: &TransformerLayer,
    x: &mut [f32],
    n: usize,
    rope: Option<(&[f32], &[f32])>,
    window: Option<usize>,
    threads: usize,
) {
    let d = layer.norm1.w.len();
    let h = layer.heads;
    let s = layer.head_dim;
    let scale = (s as f32).sqrt().recip();
    let (q, k, v) = prepare_qkv(layer, x, n, rope, threads);
    let mut attn = vec![0f32; n * d];
    let eff_span = window.map_or(n, |w| (2 * w + 1).min(n));
    #[cfg(not(target_arch = "wasm32"))]
    let workers = if threads > 1 && n >= 128 && n * eff_span >= 16_384 {
        threads.min(n / 64).max(1)
    } else {
        1
    };
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let chunk_rows = n.div_ceil(workers);
        std::thread::scope(|scope| {
            for (t, out_chunk) in attn.chunks_mut(chunk_rows * d).enumerate() {
                let row_start = t * chunk_rows;
                let rows = out_chunk.len() / d;
                let (q_ref, k_ref, v_ref) = (&q, &k, &v);
                scope.spawn(move || {
                    full_attn_chunk(
                        row_start, rows, n, h, s, d, scale, q_ref, k_ref, v_ref, window, out_chunk,
                    );
                });
            }
        });
    } else {
        full_attn_chunk(0, n, n, h, s, d, scale, &q, &k, &v, window, &mut attn);
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = eff_span;
        full_attn_chunk(0, n, n, h, s, d, scale, &q, &k, &v, window, &mut attn);
    }
    let projected = layer.out.forward(&attn, n, threads);
    for (dst, &src) in x.iter_mut().zip(&projected) {
        *dst += src;
    }
    feed_residual(layer, x, n, threads);
}

/// Precompute 2D relative layout features `[B * blk * 3*blk, 7]` and validity mask for `SpatialBlockLayer`.
fn spatial_window_features(
    valid: &[bool],
    boxes: &[f32],
    pages: &[i64],
    blk: usize,
) -> (Vec<f32>, Vec<bool>) {
    let n = valid.len();
    let blocks = n / blk;
    let kw = 3 * blk;
    let total = n * kw;
    let mut feats = vec![0f32; total * 7];
    let mut seen = vec![false; total];
    for b in 0..blocks {
        for r in 0..blk {
            let qi = b * blk + r;
            let row_off = qi * kw;
            let (ax0, ay0, ax1, ay1) = (
                boxes[qi * 4],
                boxes[qi * 4 + 1],
                boxes[qi * 4 + 2],
                boxes[qi * 4 + 3],
            );
            let q_page = pages[qi];
            let ay = 0.5 * (ay0 + ay1);
            let ah = (ay1 - ay0).max(1e-3);
            let aw = ax1 - ax0;
            for c in 0..kw {
                let is_self = c == blk + r;
                let k_idx = if c < blk {
                    if b == 0 {
                        None
                    } else {
                        Some((b - 1) * blk + c)
                    }
                } else if c < 2 * blk {
                    Some(b * blk + (c - blk))
                } else if b + 1 >= blocks {
                    None
                } else {
                    Some((b + 1) * blk + (c - 2 * blk))
                };
                let k_valid = k_idx.is_some_and(|ki| valid[ki]);
                let ok = k_valid || is_self;
                seen[row_off + c] = ok;
                if !ok {
                    continue;
                }
                let (bx0, by0, bx1, by1, k_page) = match k_idx {
                    Some(ki) => (
                        boxes[ki * 4],
                        boxes[ki * 4 + 1],
                        boxes[ki * 4 + 2],
                        boxes[ki * 4 + 3],
                        pages[ki],
                    ),
                    None => (0.0, 0.0, 0.0, 0.0, -1),
                };
                let by = 0.5 * (by0 + by1);
                let same_page = if q_page >= 0 && q_page == k_page { 1.0 } else { 0.0 };
                let bh = (by1 - by0).max(1e-3);
                let href = ah.max(bh);
                let same_y = same_page * if (by - ay).abs() < 0.5 * href { 1.0 } else { 0.0 };
                let same_x0 = same_page * if (bx0 - ax0).abs() < 0.015 { 1.0 } else { 0.0 };
                let x_ov = same_page
                    * (ax1.min(bx1) - ax0.max(bx0)).max(0.0)
                    / aw.max(bx1 - bx0).max(1e-3);
                let f = &mut feats[(row_off + c) * 7..(row_off + c + 1) * 7];
                f[0] = (bx0 - ax0).clamp(-1.0, 1.0);
                f[1] = (bx1 - ax1).clamp(-1.0, 1.0);
                f[2] = ((by - ay) / href).clamp(-8.0, 8.0) / 8.0;
                f[3] = same_y;
                f[4] = same_x0;
                f[5] = x_ov;
                f[6] = same_page;
            }
        }
    }
    (feats, seen)
}

#[allow(clippy::too_many_arguments)]
fn block_attn_chunk(
    b_start: usize,
    b_end: usize,
    n: usize,
    blk: usize,
    blocks: usize,
    h: usize,
    s: usize,
    d: usize,
    scale: f32,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    bias: Option<&[f32]>,
    seen: &[bool],
    dst_chunk: &mut [f32],
) {
    let kw = 3 * blk;
    let mut scores = vec![0f32; kw];
    for b in b_start..b_end {
        let local_b = b - b_start;
        let block_seen = if bias.is_none() {
            Some(&seen[b * kw..(b + 1) * kw])
        } else {
            None
        };
        for head in 0..h {
            let qh = &q[head * n * s..(head + 1) * n * s];
            let kh = &k[head * n * s..(head + 1) * n * s];
            let vh = &v[head * n * s..(head + 1) * n * s];
            for r in 0..blk {
                let qi_idx = b * blk + r;
                let local_row = local_b * blk + r;
                let qi = &qh[qi_idx * s..(qi_idx + 1) * s];
                let mut top = f32::NEG_INFINITY;
                for c in 0..kw {
                    let sc = if let Some(b_mat) = bias {
                        let ok = seen[qi_idx * kw + c];
                        let raw_b = b_mat[(qi_idx * kw + c) * h + head];
                        if !ok {
                            raw_b
                        } else if c < blk {
                            if b == 0 {
                                raw_b
                            } else {
                                let kj = (b - 1) * blk + c;
                                dot4(qi, &kh[kj * s..(kj + 1) * s]) * scale + raw_b
                            }
                        } else if c < 2 * blk {
                            let kj = b * blk + (c - blk);
                            dot4(qi, &kh[kj * s..(kj + 1) * s]) * scale + raw_b
                        } else if b + 1 >= blocks {
                            raw_b
                        } else {
                            let kj = (b + 1) * blk + (c - 2 * blk);
                            dot4(qi, &kh[kj * s..(kj + 1) * s]) * scale + raw_b
                        }
                    } else if block_seen.unwrap()[c] {
                        let kj = if c < blk {
                            (b - 1) * blk + c
                        } else if c < 2 * blk {
                            b * blk + (c - blk)
                        } else {
                            (b + 1) * blk + (c - 2 * blk)
                        };
                        dot4(qi, &kh[kj * s..(kj + 1) * s]) * scale
                    } else {
                        f32::NEG_INFINITY
                    };
                    scores[c] = sc;
                    if sc > top {
                        top = sc;
                    }
                }
                if top.is_finite() {
                    let mut total = 0.0f32;
                    for sc in &mut scores {
                        let e = (*sc - top).exp();
                        *sc = e;
                        total += e;
                    }
                    let inv = total.recip();
                    let dst = &mut dst_chunk[local_row * d + head * s..local_row * d + (head + 1) * s];
                    for (c, &w) in scores.iter().enumerate() {
                        let p = w * inv;
                        if p == 0.0 {
                            continue;
                        }
                        let kj = if c < blk {
                            if b == 0 {
                                continue;
                            }
                            (b - 1) * blk + c
                        } else if c < 2 * blk {
                            b * blk + (c - blk)
                        } else {
                            if b + 1 >= blocks {
                                continue;
                            }
                            (b + 1) * blk + (c - 2 * blk)
                        };
                        let vj = &vh[kj * s..(kj + 1) * s];
                        for m in 0..s {
                            dst[m] += p * vj[m];
                        }
                    }
                }
            }
        }
    }
}

/// Block-local transformer layer for both `Detect` (`spatial_bias` over `3*blk`) and `Order` (`valid` mask over `3*blk`).
fn run_block_layer(
    layer: &TransformerLayer,
    x: &mut [f32],
    n: usize,
    blk: usize,
    rope: (&[f32], &[f32]),
    feats: Option<&[f32]>,
    seen: &[bool],
    threads: usize,
) {
    let d = layer.norm1.w.len();
    let h = layer.heads;
    let s = layer.head_dim;
    let scale = (s as f32).sqrt().recip();
    let kw = 3 * blk;
    let blocks = n / blk;
    let (q, k, v) = prepare_qkv(layer, x, n, Some(rope), threads);

    let bias = layer.spatial_bias.as_ref().zip(feats).map(|(sb, f)| {
        let mut b = sb.forward(f, n * kw, threads);
        for (idx, &ok) in seen.iter().enumerate() {
            if !ok {
                b[idx * h..(idx + 1) * h].fill(-1e4);
            }
        }
        b
    });

    let mut attn = vec![0f32; n * d];
    let bias_ref = bias.as_deref();
    #[cfg(not(target_arch = "wasm32"))]
    let workers = if threads > 1 && blocks >= 16 {
        threads.min(blocks / 8).max(1)
    } else {
        1
    };
    #[cfg(not(target_arch = "wasm32"))]
    if workers > 1 {
        let chunk_blocks = blocks.div_ceil(workers);
        std::thread::scope(|scope| {
            for (t, out_chunk) in attn.chunks_mut(chunk_blocks * blk * d).enumerate() {
                let b_start = t * chunk_blocks;
                let b_end = b_start + out_chunk.len() / (blk * d);
                let (q_ref, k_ref, v_ref) = (&q, &k, &v);
                scope.spawn(move || {
                    block_attn_chunk(
                        b_start, b_end, n, blk, blocks, h, s, d, scale, q_ref, k_ref, v_ref,
                        bias_ref, seen, out_chunk,
                    );
                });
            }
        });
    } else {
        block_attn_chunk(
            0, blocks, n, blk, blocks, h, s, d, scale, &q, &k, &v, bias_ref, seen, &mut attn,
        );
    }
    #[cfg(target_arch = "wasm32")]
    block_attn_chunk(
        0, blocks, n, blk, blocks, h, s, d, scale, &q, &k, &v, bias_ref, seen, &mut attn,
    );
    let projected = layer.out.forward(&attn, n, threads);
    for (dst, &src) in x.iter_mut().zip(&projected) {
        *dst += src;
    }
    feed_residual(layer, x, n, threads);
}

#[inline(always)]
fn relu(v: &mut [f32]) {
    for x in v {
        if *x < 0.0 {
            *x = 0.0;
        }
    }
}

#[inline(always)]
fn silu(v: &mut [f32]) {
    for x in v {
        *x = *x / (1.0 + (-*x).exp());
    }
}

fn delta_boxes(a: [f32; 4], b: [f32; 4], out: &mut [f32]) {
    let wa = (a[2] - a[0]).max(1e-3);
    let ha = (a[3] - a[1]).max(1e-3);
    let wb = (b[2] - b[0]).max(1e-3);
    let hb = (b[3] - b[1]).max(1e-3);
    let xa = 0.5 * (a[0] + a[2]);
    let ya = 0.5 * (a[1] + a[3]);
    let xb = 0.5 * (b[0] + b[2]);
    let yb = 0.5 * (b[1] + b[3]);
    out[0] = (xa - xb) / wa;
    out[1] = (ya - yb) / ha;
    out[2] = (wa / wb).ln();
    out[3] = (ha / hb).ln();
    out[4] = (xb - xa) / wb;
    out[5] = (yb - ya) / hb;
}

fn box_measures(a: [f32; 4], b: [f32; 4], out: &mut [f32]) {
    let u = [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].max(b[2]),
        a[3].max(b[3]),
    ];
    delta_boxes(a, b, &mut out[0..6]);
    delta_boxes(a, u, &mut out[6..12]);
    delta_boxes(b, u, &mut out[12..18]);
    for v in &mut out[..18] {
        *v = v.clamp(-20.0, 20.0);
    }
}

impl Net {
    /// Load DOC neural weights from safetensors tensors according to `meta`.
    pub fn from_tensors(tensors: &HashMap<String, Tensor>, meta: &Meta) -> Result<Self, String> {
        let templates_layers = meta.template_layers_or(tensors);
        let order_layers = meta.order_layers_or(tensors);
        let tree_layers = meta.tree_layers_or(tensors);
        let detect = DetectNet {
            text0: Linear::load(tensors, "detect.text.0")?,
            text2: LayerNorm::load(tensors, "detect.text.2")?,
            pos0: Linear::load(tensors, "detect.position.0")?,
            pos2: Linear::load(tensors, "detect.position.2")?,
            pos4: LayerNorm::load(tensors, "detect.position.4")?,
            look0: Linear::load(tensors, "detect.look.0")?,
            look2: LayerNorm::load(tensors, "detect.look.2")?,
            anchor: Embedding::load(tensors, "detect.anchor")?,
            decided0: Linear::load(tensors, "detect.decided.0")?,
            decided2: LayerNorm::load(tensors, "detect.decided.2")?,
            fuse: Linear::load(tensors, "detect.fuse")?,
            pre: TransformerLayer::load(tensors, "detect.pre", true)?,
            post: TransformerLayer::load(tensors, "detect.post", true)?,
            trans_geom0: Linear::load(tensors, "detect.trans_geom.0")?,
            trans_geom2: Linear::load(tensors, "detect.trans_geom.2")?,
            trans_fuse0: LayerNorm::load(tensors, "detect.trans_fuse.0")?,
            trans_fuse1: Linear::load(tensors, "detect.trans_fuse.1")?,
            trans_fuse3: Linear::load(tensors, "detect.trans_fuse.3")?,
            succ_q: Linear::load(tensors, "detect.successor.q")?,
            succ_k: Linear::load(tensors, "detect.successor.k")?,
            succ_r0: Linear::load(tensors, "detect.successor.r.0")?,
            succ_r2: Linear::load(tensors, "detect.successor.r.2")?,
            pred_q: Linear::load(tensors, "detect.predecessor.q")?,
            pred_k: Linear::load(tensors, "detect.predecessor.k")?,
            pred_r0: Linear::load(tensors, "detect.predecessor.r.0")?,
            pred_r2: Linear::load(tensors, "detect.predecessor.r.2")?,
            role: Linear::load(tensors, "detect.role")?,
            flags: Linear::load(tensors, "detect.flags")?,
            table_col: Linear::load(tensors, "detect.table_col")?,
            speaker: Linear::load(tensors, "detect.speaker")?,
        };
        let templates = TemplatesNet {
            row0: Linear::load(tensors, "templates.row.0")?,
            row2: LayerNorm::load(tensors, "templates.row.2")?,
            pool: Linear::load(tensors, "templates.pool")?,
            encoder: (0..templates_layers)
                .map(|i| TransformerLayer::load(tensors, &format!("templates.encoder.{i}"), false))
                .collect::<Result<Vec<_>, _>>()?,
            back0: Linear::load(tensors, "templates.back.0")?,
            back1: LayerNorm::load(tensors, "templates.back.1")?,
        };
        let far_of = |name: &str| meta.order_far.get(name).copied().unwrap_or(0.0);
        let order = OrderNet {
            pool0: Linear::load(tensors, "order.pool.0")?,
            pool2: Linear::load(tensors, "order.pool.2")?,
            role0: Embedding::load(tensors, "order.role.0")?,
            role2: LayerNorm::load(tensors, "order.role.2")?,
            fuse: Linear::load(tensors, "order.fuse")?,
            encoder: (0..order_layers)
                .map(|i| TransformerLayer::load(tensors, &format!("order.encoder.{i}"), false))
                .collect::<Result<Vec<_>, _>>()?,
            goes_on: PairHeadNet::load(tensors, "order.goes_on", meta.order_link_dim, far_of("goes_on"))?,
            note_of: PairHeadNet::load(tensors, "order.note_of", meta.order_link_dim, far_of("note_of"))?,
            successor: PairHeadNet::load(tensors, "order.successor", meta.order_link_dim, far_of("successor"))?,
            role_head: Linear::load(tensors, "order.role_head")?,
            start_head: Linear::load(tensors, "order.start_head")?,
            doc_kind_head: Linear::load(tensors, "order.doc_kind_head")?,
        };
        Ok(Self {
            detect,
            templates,
            order,
            headings: ConstructNet::load(tensors, "headings", tree_layers)?,
            bodies: ConstructNet::load(tensors, "bodies", tree_layers)?,
        })
    }

    /// Run Detect (`pre -> templates -> post -> refine`) and line-head projections over `l_pad` padded lines,
    /// returning unpadded `n`-line outputs.
    #[allow(clippy::too_many_arguments)]
    pub fn run_detect(
        &self,
        features: &[f32],
        text: &[f32],
        boxes: &[f32],
        pages: &[i64],
        anchors: &[i64],
        t0: &[f32],
        valid: &[bool],
        templates: &[i64],
        rows: &[f32],
        n: usize,
        l_pad: usize,
        num_templates: usize,
        meta: &Meta,
        threads: usize,
    ) -> DetectOut {
        let d = meta.d;
        let det = &self.detect;
        let mut txt = det.text0.forward(text, l_pad, threads);
        relu(&mut txt);
        det.text2.forward_inplace(&mut txt);

        let mut pos_in = Vec::with_capacity(l_pad * 6);
        for b in boxes.chunks_exact(4) {
            pos_in.extend_from_slice(&[
                b[0],
                b[1],
                b[2],
                b[3],
                (b[2] - b[0]).max(0.0),
                (b[3] - b[1]).max(0.0),
            ]);
        }
        let mut pos = det.pos0.forward(&pos_in, l_pad, threads);
        relu(&mut pos);
        pos = det.pos2.forward(&pos, l_pad, threads);
        relu(&mut pos);
        det.pos4.forward_inplace(&mut pos);

        let mut lk = det.look0.forward(features, l_pad, threads);
        relu(&mut lk);
        det.look2.forward_inplace(&mut lk);

        let mut dec = det.decided0.forward(t0, l_pad, threads);
        relu(&mut dec);
        det.decided2.forward_inplace(&mut dec);

        let mut cat = Vec::with_capacity(l_pad * 5 * d);
        for i in 0..l_pad {
            cat.extend_from_slice(&txt[i * d..(i + 1) * d]);
            cat.extend_from_slice(&pos[i * d..(i + 1) * d]);
            cat.extend_from_slice(&lk[i * d..(i + 1) * d]);
            cat.extend_from_slice(det.anchor.row(anchors[i]));
            cat.extend_from_slice(&dec[i * d..(i + 1) * d]);
        }
        let mut u = det.fuse.forward(&cat, l_pad, threads);
        drop((txt, pos, lk, dec, cat));

        let blk = meta.line_block.max(1);
        let (cos, sin) = rotary(l_pad, det.pre.head_dim);
        let (feats, seen_2d) = spatial_window_features(valid, boxes, pages, blk);
        run_block_layer(
            &det.pre,
            &mut u,
            l_pad,
            blk,
            (&cos, &sin),
            Some(&feats),
            &seen_2d,
            threads,
        );

        // Templates: scatter-mean over valid lines, transform stylesheet nodes, scatter back.
        let mut t_sum = vec![0f32; num_templates * d];
        let mut t_cnt = vec![0f32; num_templates];
        for i in 0..l_pad {
            if !valid[i] {
                continue;
            }
            let tid = templates[i].clamp(0, num_templates.saturating_sub(1) as i64) as usize;
            t_cnt[tid] += 1.0;
            let dst = &mut t_sum[tid * d..(tid + 1) * d];
            let src = &u[i * d..(i + 1) * d];
            for c in 0..d {
                dst[c] += src[c];
            }
        }
        for tid in 0..num_templates {
            let inv = t_cnt[tid].max(1.0).recip();
            for v in &mut t_sum[tid * d..(tid + 1) * d] {
                *v *= inv;
            }
        }
        let mut t_row = self.templates.row0.forward(rows, num_templates, threads);
        relu(&mut t_row);
        self.templates.row2.forward_inplace(&mut t_row);
        let mut t_node = self.templates.pool.forward(&t_sum, num_templates, threads);
        for (dst, &src) in t_node.iter_mut().zip(&t_row) {
            *dst += src;
        }
        for layer in &self.templates.encoder {
            run_full_layer(layer, &mut t_node, num_templates, None, None, threads);
        }
        let mut t_back = self.templates.back0.forward(&t_node, num_templates, threads);
        self.templates.back1.forward_inplace(&mut t_back);
        for i in 0..l_pad {
            let tid = templates[i].clamp(0, num_templates.saturating_sub(1) as i64) as usize;
            let dst = &mut u[i * d..(i + 1) * d];
            let src = &t_back[tid * d..(tid + 1) * d];
            for c in 0..d {
                dst[c] += src[c];
            }
        }

        run_block_layer(
            &det.post,
            &mut u,
            l_pad,
            blk,
            (&cos, &sin),
            Some(&feats),
            &seen_2d,
            threads,
        );

        // Consecutive same-page transition refinement over real lines 0..n (matches `Detect.refine`).
        let mut geom = vec![0f32; n * 12];
        for i in 0..n {
            let step = |other: usize, mask: bool, forward: bool, dst: &mut [f32]| {
                if !mask {
                    return;
                }
                let (ax0, ay0, ax1, ay1) = (
                    boxes[i * 4],
                    boxes[i * 4 + 1],
                    boxes[i * 4 + 2],
                    boxes[i * 4 + 3],
                );
                let (bx0, by0, bx1, by1) = (
                    boxes[other * 4],
                    boxes[other * 4 + 1],
                    boxes[other * 4 + 2],
                    boxes[other * 4 + 3],
                );
                let wa = (ax1 - ax0).max(1e-3);
                let ha = (ay1 - ay0).max(1e-3);
                let wb = (bx1 - bx0).max(1e-3);
                let hb = (by1 - by0).max(1e-3);
                let href = ha.max(hb);
                let vgap = if forward { by0 - ay1 } else { ay0 - by1 };
                dst[0] = ((ax0 - bx0) / href).clamp(-10.0, 10.0) / 10.0;
                dst[1] = ((ax1 - bx1) / href).clamp(-10.0, 10.0) / 10.0;
                dst[2] = (vgap / href).clamp(-4.0, 8.0) / 4.0;
                dst[3] = (ha / hb).ln().clamp(-2.0, 2.0);
                dst[4] = (ax1.min(bx1) - ax0.max(bx0)).max(0.0) / wa.max(wb);
                dst[5] = 1.0;
            };
            let has_p = i > 0 && pages[i] == pages[i - 1] && pages[i] >= 0;
            let has_n = i + 1 < n && pages[i] == pages[i + 1] && pages[i] >= 0;
            step(i.saturating_sub(1), has_p, false, &mut geom[i * 12..i * 12 + 6]);
            step((i + 1).min(n - 1), has_n, true, &mut geom[i * 12 + 6..i * 12 + 12]);
        }
        let mut g = det.trans_geom0.forward(&geom, n, threads);
        silu(&mut g);
        g = det.trans_geom2.forward(&g, n, threads);

        let mut fuse_in = vec![0f32; n * 3 * d];
        for i in 0..n {
            let row = &mut fuse_in[i * 3 * d..(i + 1) * 3 * d];
            let xi = &u[i * d..(i + 1) * d];
            if i > 0 && pages[i] == pages[i - 1] && pages[i] >= 0 {
                let xp = &u[(i - 1) * d..i * d];
                for c in 0..d {
                    row[c] = xi[c] - xp[c];
                }
            }
            if i + 1 < n && pages[i] == pages[i + 1] && pages[i] >= 0 {
                let xn = &u[(i + 1) * d..(i + 2) * d];
                for c in 0..d {
                    row[d + c] = xn[c] - xi[c];
                }
            }
            row[2 * d..3 * d].copy_from_slice(&g[i * d..(i + 1) * d]);
        }
        det.trans_fuse0.forward_inplace(&mut fuse_in);
        let mut hidden = det.trans_fuse1.forward(&fuse_in, n, threads);
        silu(&mut hidden);
        let delta = det.trans_fuse3.forward(&hidden, n, threads);
        let mut states = u[..n * d].to_vec();
        for (dst, &src) in states.iter_mut().zip(&delta) {
            *dst += src;
        }

        let inv_sqrt = (meta.link_dim.max(1) as f32).sqrt().recip();
        let mut succ_q = det.succ_q.forward(&states, n, threads);
        let succ_k = det.succ_k.forward(&states, n, threads);
        let mut pred_q = det.pred_q.forward(&states, n, threads);
        let pred_k = det.pred_k.forward(&states, n, threads);
        for v in &mut succ_q {
            *v *= inv_sqrt;
        }
        for v in &mut pred_q {
            *v *= inv_sqrt;
        }
        DetectOut {
            role: det.role.forward(&states, n, threads),
            flags: det.flags.forward(&states, n, threads),
            succ_q,
            succ_k,
            pred_q,
            pred_k,
            table_col: det.table_col.forward(&states, n, threads),
            speaker: det.speaker.forward(&states, n, threads),
            states,
        }
    }

    /// Score local line-pair spatial compatibility (`successor` for `(i, j)`, `predecessor` for `(j, i)`).
    pub fn run_spatial(&self, boxes: &[f32], pair_i: &[i64], pair_j: &[i64]) -> (Vec<f32>, Vec<f32>) {
        let p = pair_i.len();
        if p == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut ms = vec![0f32; p * 18];
        let mut mp = vec![0f32; p * 18];
        let box_of = |idx: i64| {
            let k = idx as usize;
            [boxes[k * 4], boxes[k * 4 + 1], boxes[k * 4 + 2], boxes[k * 4 + 3]]
        };
        for at in 0..p {
            let ba = box_of(pair_i[at]);
            let bb = box_of(pair_j[at]);
            box_measures(ba, bb, &mut ms[at * 18..(at + 1) * 18]);
            box_measures(bb, ba, &mut mp[at * 18..(at + 1) * 18]);
        }
        let mut hs = self.detect.succ_r0.forward(&ms, p, 1);
        relu(&mut hs);
        let succ = self.detect.succ_r2.forward(&hs, p, 1);
        let mut hp = self.detect.pred_r0.forward(&mp, p, 1);
        relu(&mut hp);
        let pred = self.detect.pred_r2.forward(&hp, p, 1);
        (succ, pred)
    }

    /// Run Order encoder and pair heads over `padded` regions.
    #[allow(clippy::too_many_arguments)]
    pub fn run_order(
        &self,
        line_states: &[f32],
        line_region: &[i64],
        opener: &[f32],
        closer: &[f32],
        region_roles: &[i64],
        region_valid: &[bool],
        region_boxes: &[f32],
        region_pages: &[i64],
        goes_i: &[i64],
        goes_j: &[i64],
        note_i: &[i64],
        note_j: &[i64],
        note_match: &[bool],
        ord_i: &[i64],
        ord_j: &[i64],
        padded: usize,
        meta: &Meta,
        threads: usize,
    ) -> OrderOut {
        let d = meta.d;
        let ord = &self.order;
        let k = line_region.len();
        let mut pooled = vec![0f32; padded * d];
        if k > 0 {
            let mut h = ord.pool0.forward(line_states, k, threads);
            for v in &mut h {
                *v = v.tanh();
            }
            let alpha = ord.pool2.forward(&h, k, threads);
            let mut top = vec![-1e4f32; padded];
            for (at, &reg) in line_region.iter().enumerate() {
                let r = reg as usize;
                if alpha[at] > top[r] {
                    top[r] = alpha[at];
                }
            }
            let mut total = vec![0f32; padded];
            for (at, &reg) in line_region.iter().enumerate() {
                let r = reg as usize;
                let w = (alpha[at] - top[r]).exp();
                total[r] += w;
                let dst = &mut pooled[r * d..(r + 1) * d];
                let src = &line_states[at * d..(at + 1) * d];
                for c in 0..d {
                    dst[c] += w * src[c];
                }
            }
            for r in 0..padded {
                let inv = total[r].max(1e-9).recip();
                for v in &mut pooled[r * d..(r + 1) * d] {
                    *v *= inv;
                }
            }
        }

        let mut role_emb = vec![0f32; padded * d];
        for (i, &r) in region_roles.iter().enumerate() {
            role_emb[i * d..(i + 1) * d].copy_from_slice(ord.role0.row(r));
        }
        relu(&mut role_emb);
        ord.role2.forward_inplace(&mut role_emb);

        let mut fuse_in = Vec::with_capacity(padded * 4 * d);
        for i in 0..padded {
            fuse_in.extend_from_slice(&opener[i * d..(i + 1) * d]);
            fuse_in.extend_from_slice(&closer[i * d..(i + 1) * d]);
            fuse_in.extend_from_slice(&pooled[i * d..(i + 1) * d]);
            fuse_in.extend_from_slice(&role_emb[i * d..(i + 1) * d]);
        }
        let mut regions = ord.fuse.forward(&fuse_in, padded, threads);
        drop((pooled, role_emb, fuse_in));

        let blk = meta.order_block.max(1);
        let blocks = padded / blk;
        let kw = 3 * blk;
        let mut block_seen = vec![false; blocks * kw];
        for b in 0..blocks {
            for c in 0..kw {
                block_seen[b * kw + c] = if c < blk {
                    b > 0 && region_valid[(b - 1) * blk + c]
                } else if c < 2 * blk {
                    region_valid[b * blk + (c - blk)]
                } else {
                    b + 1 < blocks && region_valid[(b + 1) * blk + (c - 2 * blk)]
                };
            }
        }
        let (cos, sin) = rotary(padded, ord.encoder[0].head_dim);
        for layer in &ord.encoder {
            run_block_layer(
                layer,
                &mut regions,
                padded,
                blk,
                (&cos, &sin),
                None,
                &block_seen,
                threads,
            );
        }

        let run_pair = |head: &PairHeadNet, pi: &[i64], pj: &[i64], matches: Option<&[bool]>| {
            let scale = (head.link_dim.max(1) as f32).sqrt().recip();
            let mut q = head.q.forward(&regions, padded, threads);
            for v in &mut q {
                *v *= scale;
            }
            let k = head.k.forward(&regions, padded, threads);
            let p = pi.len();
            if p == 0 {
                return (q, k, Vec::new());
            }
            let mut feats = vec![0f32; p * 25];
            let mut near = vec![false; p];
            for at in 0..p {
                let i = pi[at] as usize;
                let j = pj[at] as usize;
                let matched = matches.is_some_and(|m| m[at]);
                near[at] = i.abs_diff(j) <= meta.order_window || matched;
                let row = &mut feats[at * 25..(at + 1) * 25];
                box_measures(
                    [
                        region_boxes[i * 4],
                        region_boxes[i * 4 + 1],
                        region_boxes[i * 4 + 2],
                        region_boxes[i * 4 + 3],
                    ],
                    [
                        region_boxes[j * 4],
                        region_boxes[j * 4 + 1],
                        region_boxes[j * 4 + 2],
                        region_boxes[j * 4 + 3],
                    ],
                    row,
                );
                let gap = (region_pages[j] - region_pages[i]).clamp(-1, 1);
                row[18 + (gap + 1) as usize] = 1.0;
                row[21] = (1.0 + i.abs_diff(j) as f32).ln() / 5.0;
                row[22] = if j > i {
                    1.0
                } else if j < i {
                    -1.0
                } else {
                    0.0
                };
                row[23] = if region_roles[i] == region_roles[j] {
                    1.0
                } else {
                    0.0
                };
                row[24] = if matched { 1.0 } else { 0.0 };
            }
            let mut h = head.r0.forward(&feats, p, threads);
            relu(&mut h);
            let mut r = head.r2.forward(&h, p, threads);
            for (at, val) in r.iter_mut().enumerate() {
                if !near[at] {
                    *val = head.far;
                }
            }
            (q, k, r)
        };

        let (goes_on_q, goes_on_k, goes_on_r) = run_pair(&ord.goes_on, goes_i, goes_j, None);
        let (note_of_q, note_of_k, note_of_r) =
            run_pair(&ord.note_of, note_i, note_j, Some(note_match));
        let (order_q, order_k, order_r) = run_pair(&ord.successor, ord_i, ord_j, None);

        OrderOut {
            goes_on_q,
            goes_on_k,
            goes_on_r,
            note_of_q,
            note_of_k,
            note_of_r,
            order_q,
            order_k,
            order_r,
            region_logits: ord.role_head.forward(&regions, padded, threads),
            region_start: ord.start_head.forward(&regions, padded, threads),
            region_doc_kind: ord.doc_kind_head.forward(&regions, padded, threads),
            regions,
        }
    }

    /// Run Construct (`headings` or `bodies`) over `h` tree nodes, returning unscaled parent/sibling QK + gap logits.
    #[allow(clippy::too_many_arguments)]
    pub fn run_construct(
        &self,
        tree: &ConstructNet,
        states: &[f32],
        opener_states: &[f32],
        heading_states: &[f32],
        family: &[i64],
        value: &[i64],
        prefix: &[i64],
        x0: &[f32],
        size: &[f32],
        roles: &[i64],
        h: usize,
        threads: usize,
    ) -> ConstructOut {
        let mut geom_in = Vec::with_capacity(h * 4);
        for i in 0..h {
            geom_in.extend_from_slice(&[
                x0[i],
                size[i],
                if family[i] > 0 && value[i] == 1 {
                    1.0
                } else {
                    0.0
                },
                (prefix[i].clamp(0, 4) as f32) / 4.0,
            ]);
        }
        let geom = tree.node_geom.forward(&geom_in, h, threads);
        let enc = tree.enclosing.forward(heading_states, h, threads);
        let opener = tree.opener_proj.forward(opener_states, h, threads);
        let mut x = match &tree.in_proj {
            Some(proj) => proj.forward(states, h, threads),
            None => states.to_vec(),
        };
        let td = tree.node_geom.out;
        for i in 0..h {
            let role_e = tree.node_role.row(roles[i]);
            let fam_e = tree.node_fam.row(family[i]);
            let dst = &mut x[i * td..(i + 1) * td];
            for c in 0..td {
                dst[c] += role_e[c]
                    + fam_e[c]
                    + geom[i * td + c]
                    + enc[i * td + c]
                    + opener[i * td + c];
            }
        }
        let (cos, sin) = rotary(h, tree.layers[0].head_dim);
        for layer in &tree.layers {
            run_full_layer(
                layer,
                &mut x,
                h,
                Some((&cos, &sin)),
                Some(CONSTRUCT_WINDOW),
                threads,
            );
        }
        ConstructOut {
            parent_q: tree.parent_q.forward(&x, h, threads),
            parent_k: tree.parent_k.forward(&x, h, threads),
            sibling_q: tree.sibling_q.forward(&x, h, threads),
            sibling_k: tree.sibling_k.forward(&x, h, threads),
            gap: tree.gap.forward(&x, h, threads),
        }
    }
}

impl Meta {
    fn template_layers_or(&self, tensors: &HashMap<String, Tensor>) -> usize {
        (0..)
            .take_while(|i| tensors.contains_key(&format!("templates.encoder.{i}.norm1.w")))
            .count()
    }

    fn order_layers_or(&self, tensors: &HashMap<String, Tensor>) -> usize {
        (0..)
            .take_while(|i| tensors.contains_key(&format!("order.encoder.{i}.norm1.w")))
            .count()
    }

    fn tree_layers_or(&self, tensors: &HashMap<String, Tensor>) -> usize {
        (0..)
            .take_while(|i| tensors.contains_key(&format!("headings.layers.{i}.norm1.w")))
            .count()
    }
}

/// Normalize each `width`-wide logit row into a probability distribution.
pub fn softmax_rows(values: &[f32], width: usize) -> Vec<f32> {
    values
        .chunks_exact(width)
        .flat_map(|row| {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = row.iter().map(|x| (x - top).exp()).sum();
            row.iter().map(move |x| (x - top).exp() / total)
        })
        .collect()
}
