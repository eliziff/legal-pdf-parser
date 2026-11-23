//! Pure-Rust SIMD inference for the Indra FPN line-box detector (`assets/detectors/indra/model.onnx`),
//! running on x86_64 AVX2+FMA, wasm32 SIMD128, and scalar fallback with zero ORT runtime dependency.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::indra_native::{
    hardswish_inplace, linear_packed, parse_onnx_model, OnnxTensor, PackedLinear, HARD_SWISH_SIXTH,
};

/// Reusable per-worker scratch buffers so per-page detection allocates zero heap memory after warmup.
#[derive(Default)]
pub struct DetectorScratch {
    stem_act: Vec<f32>,
    s0: Vec<f32>,
    s1: Vec<f32>,
    s2: Vec<f32>,
    a: Vec<f32>,
    b: Vec<f32>,
    exp: Vec<f32>,
    fpn_a: Vec<f32>,
    fpn_b: Vec<f32>,
    out: Vec<f32>,
}

struct DwWeights {
    /// `[K * K, C]` transposed layout for contiguous SIMD vector loads across channels.
    w_t: Vec<f32>,
    b: Vec<f32>,
    channels: usize,
    kernel: usize,
}

impl DwWeights {
    fn from_chw(raw_w: &[f32], raw_b: &[f32], channels: usize, kernel: usize) -> Result<Self, String> {
        let kk = kernel * kernel;
        if !channels.is_multiple_of(8) || raw_w.len() != channels * kk || raw_b.len() != channels {
            return Err(format!(
                "unsupported detector depthwise weights: C={channels} K={kernel} w={} b={}",
                raw_w.len(),
                raw_b.len()
            ));
        }
        let mut w_t = vec![0.0_f32; kk * channels];
        for c in 0..channels {
            for k in 0..kk {
                w_t[k * channels + c] = raw_w[c * kk + k];
            }
        }
        Ok(Self {
            w_t,
            b: raw_b.to_vec(),
            channels,
            kernel,
        })
    }
}

struct DetResBlock {
    expand: PackedLinear,
    dw: DwWeights,
    project: PackedLinear,
    expand_channels: usize,
}

struct DetStage {
    down_dw: DwWeights,
    down_pw: PackedLinear,
    out_channels: usize,
    blocks: Vec<DetResBlock>,
}

/// Loaded Indra FPN line-box detector ready for zero-allocation per-page SIMD inference.
pub struct DetectorNative {
    stem_w_t: Vec<f32>,
    stem_b: Vec<f32>,
    stem_channels: usize,
    stages: [DetStage; 4],
    lat: [PackedLinear; 4],
    fpn_channels: usize,
    smooth: DwWeights,
    stem_lat: PackedLinear,
    stem_lat_channels: usize,
    head0: PackedLinear,
    head1: DwWeights,
    head2: PackedLinear,
    softplus_scale: f32,
}

impl DetectorNative {
    /// Load an Indra detector ONNX model directly from disk (not compiled on `wasm32`).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn from_onnx_path(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|err| format!("failed to read detector ONNX model {}: {err}", path.display()))?;
        Self::from_onnx_bytes(&bytes)
    }

    /// Parse weights and graph order directly from an Indra detector ONNX protobuf byte slice.
    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self, String> {
        let (mut tensors, nodes) = parse_onnx_model(bytes)?;

        // Also capture scalar Constant node values so `Softplus * const` is read directly from the graph.
        let mut consts = std::collections::HashMap::<String, OnnxTensor>::new();
        let mut conv_nodes = Vec::new();
        let mut mul_inputs = None;
        for node in &nodes {
            match node.op_type.as_str() {
                "Constant" => {
                    if let (Some(out_name), Some(tensor)) = (node.outputs.first(), &node.attr_t) {
                        consts.insert(
                            out_name.clone(),
                            OnnxTensor {
                                name: tensor.name.clone(),
                                dims: tensor.dims.clone(),
                                data: tensor.data.clone(),
                            },
                        );
                    }
                }
                "Conv" => {
                    if node.inputs.len() < 3 {
                        return Err("detector Conv node missing weight or bias input".to_owned());
                    }
                    conv_nodes.push((node.inputs[1].clone(), node.inputs[2].clone()));
                }
                "Mul" => {
                    mul_inputs = Some(node.inputs.clone());
                }
                _ => {}
            }
        }

        if conv_nodes.len() != 39 {
            return Err(format!(
                "unexpected detector Conv count {} (expected 39)",
                conv_nodes.len()
            ));
        }

        let mut next_conv = 0usize;
        let mut take_conv = |tensors: &mut std::collections::HashMap<String, OnnxTensor>| -> Result<(OnnxTensor, OnnxTensor), String> {
            let (ref w_name, ref b_name) = conv_nodes[next_conv];
            next_conv += 1;
            let w = tensors
                .remove(w_name)
                .ok_or_else(|| format!("detector ONNX missing initializer '{w_name}'"))?;
            let b = tensors
                .remove(b_name)
                .ok_or_else(|| format!("detector ONNX missing initializer '{b_name}'"))?;
            Ok((w, b))
        };

        // 0: stem Conv [C_stem, 1, 3, 3]
        let (stem_w_t_raw, stem_b_raw) = take_conv(&mut tensors)?;
        if stem_w_t_raw.dims.len() != 4 || stem_w_t_raw.dims[1] != 1 || stem_w_t_raw.dims[2] != 3 || stem_w_t_raw.dims[3] != 3 {
            return Err(format!("unexpected detector stem shape {:?}", stem_w_t_raw.dims));
        }
        let stem_channels = stem_w_t_raw.dims[0];
        if stem_channels == 0 || !stem_channels.is_multiple_of(8) || stem_b_raw.data.len() != stem_channels {
            return Err(format!("unexpected detector stem channel count {stem_channels}"));
        }
        let mut stem_w_t = vec![0.0_f32; 9 * stem_channels];
        for c in 0..stem_channels {
            for k in 0..9 {
                stem_w_t[k * stem_channels + c] = stem_w_t_raw.data[c * 9 + k];
            }
        }

        // Stages 0..3 (block counts: [1, 2, 2, 2]).
        let mut load_stage = |in_c: usize, res_count: usize| -> Result<DetStage, String> {
            let (dw_w, dw_b) = take_conv(&mut tensors)?;
            if dw_w.dims.len() != 4 || dw_w.dims[0] != in_c || dw_w.dims[1] != 1 || dw_w.dims[2] != dw_w.dims[3] {
                return Err(format!("unexpected detector stage down_dw shape {:?}", dw_w.dims));
            }
            let kernel = dw_w.dims[2];
            if kernel != 3 && kernel != 5 {
                return Err(format!("unsupported detector DW kernel size {kernel}"));
            }
            let down_dw = DwWeights::from_chw(&dw_w.data, &dw_b.data, in_c, kernel)?;

            let (pw_w, pw_b) = take_conv(&mut tensors)?;
            if pw_w.dims.len() != 4 || pw_w.dims[1] != in_c || pw_w.dims[2] != 1 || pw_w.dims[3] != 1 {
                return Err(format!("unexpected detector stage down_pw shape {:?}", pw_w.dims));
            }
            let out_channels = pw_w.dims[0];
            if out_channels == 0 || !out_channels.is_multiple_of(8) || pw_b.data.len() != out_channels {
                return Err(format!("unexpected detector stage out_channels {out_channels}"));
            }
            let down_pw = PackedLinear::pack(&pw_w.data, &pw_b.data, out_channels, in_c);

            let mut blocks = Vec::with_capacity(res_count);
            for _ in 0..res_count {
                let (exp_w, exp_b) = take_conv(&mut tensors)?;
                if exp_w.dims.len() != 4 || exp_w.dims[1] != out_channels || exp_w.dims[2] != 1 || exp_w.dims[3] != 1 {
                    return Err(format!("unexpected detector expand shape {:?}", exp_w.dims));
                }
                let expand_channels = exp_w.dims[0];
                if expand_channels == 0 || !expand_channels.is_multiple_of(8) || exp_b.data.len() != expand_channels {
                    return Err(format!("unexpected detector expand_channels {expand_channels}"));
                }
                let expand = PackedLinear::pack(&exp_w.data, &exp_b.data, expand_channels, out_channels);

                let (bdw_w, bdw_b) = take_conv(&mut tensors)?;
                if bdw_w.dims.len() != 4
                    || bdw_w.dims[0] != expand_channels
                    || bdw_w.dims[1] != 1
                    || bdw_w.dims[2] != kernel
                    || bdw_w.dims[3] != kernel
                {
                    return Err(format!("unexpected detector block DW shape {:?}", bdw_w.dims));
                }
                let dw = DwWeights::from_chw(&bdw_w.data, &bdw_b.data, expand_channels, kernel)?;

                let (proj_w, proj_b) = take_conv(&mut tensors)?;
                if proj_w.dims.len() != 4
                    || proj_w.dims[0] != out_channels
                    || proj_w.dims[1] != expand_channels
                    || proj_w.dims[2] != 1
                    || proj_w.dims[3] != 1
                    || proj_b.data.len() != out_channels
                {
                    return Err(format!("unexpected detector project shape {:?}", proj_w.dims));
                }
                let project = PackedLinear::pack(&proj_w.data, &proj_b.data, out_channels, expand_channels);

                blocks.push(DetResBlock {
                    expand,
                    dw,
                    project,
                    expand_channels,
                });
            }
            Ok(DetStage {
                down_dw,
                down_pw,
                out_channels,
                blocks,
            })
        };

        let s0 = load_stage(stem_channels, 1)?;
        let s1 = load_stage(s0.out_channels, 2)?;
        let s2 = load_stage(s1.out_channels, 2)?;
        let s3 = load_stage(s2.out_channels, 2)?;
        let stage_cs = [s0.out_channels, s1.out_channels, s2.out_channels, s3.out_channels];
        let stages = [s0, s1, s2, s3];

        // Laterals lat.3, lat.2, lat.1, lat.0 -> ordered [lat0, lat1, lat2, lat3].
        fn pack_pw(w: &OnnxTensor, b: &OnnxTensor, expected_in: usize, name: &str) -> Result<(PackedLinear, usize), String> {
            if w.dims.len() != 4 || w.dims[1] != expected_in || w.dims[2] != 1 || w.dims[3] != 1 || b.data.len() != w.dims[0] {
                return Err(format!("unexpected detector {name} shape {:?}", w.dims));
            }
            let out_c = w.dims[0];
            Ok((PackedLinear::pack(&w.data, &b.data, out_c, expected_in), out_c))
        }

        let (w, b) = take_conv(&mut tensors)?;
        let (lat3, fpn_channels) = pack_pw(&w, &b, stage_cs[3], "lat.3")?;
        if fpn_channels == 0 || !fpn_channels.is_multiple_of(8) {
            return Err(format!("unexpected detector FPN channels {fpn_channels}"));
        }
        let (w, b) = take_conv(&mut tensors)?;
        let (lat2, c_lat2) = pack_pw(&w, &b, stage_cs[2], "lat.2")?;
        let (w, b) = take_conv(&mut tensors)?;
        let (lat1, c_lat1) = pack_pw(&w, &b, stage_cs[1], "lat.1")?;
        let (w, b) = take_conv(&mut tensors)?;
        let (lat0, c_lat0) = pack_pw(&w, &b, stage_cs[0], "lat.0")?;
        if c_lat2 != fpn_channels || c_lat1 != fpn_channels || c_lat0 != fpn_channels {
            return Err("mismatched detector FPN lateral channel counts".to_owned());
        }
        let lat = [lat0, lat1, lat2, lat3];

        let (smooth_w, smooth_b) = take_conv(&mut tensors)?;
        if smooth_w.dims != [fpn_channels, 1, 3, 3] {
            return Err(format!("unexpected detector smooth shape {:?}", smooth_w.dims));
        }
        let smooth = DwWeights::from_chw(&smooth_w.data, &smooth_b.data, fpn_channels, 3)?;

        let (w, b) = take_conv(&mut tensors)?;
        let (stem_lat, stem_lat_channels) = pack_pw(&w, &b, stem_channels, "stem_lat")?;
        let head0_in = fpn_channels + stem_lat_channels * 4;
        let (w, b) = take_conv(&mut tensors)?;
        let (head0, head0_c) = pack_pw(&w, &b, head0_in, "head.0")?;
        if head0_c != fpn_channels {
            return Err(format!("unexpected detector head.0 out_channels {head0_c}"));
        }

        let (head1_w, head1_b) = take_conv(&mut tensors)?;
        if head1_w.dims != [fpn_channels, 1, 3, 3] {
            return Err(format!("unexpected detector head.1 shape {:?}", head1_w.dims));
        }
        let head1 = DwWeights::from_chw(&head1_w.data, &head1_b.data, fpn_channels, 3)?;

        let (w, b) = take_conv(&mut tensors)?;
        let (head2, head2_c) = pack_pw(&w, &b, fpn_channels, "head.2")?;
        if head2_c != 12 {
            return Err(format!("unexpected detector head.2 out_channels {head2_c} (expected 12)"));
        }

        let softplus_scale = mul_inputs
            .as_deref()
            .and_then(|ins| ins.iter().find_map(|name| consts.get(name).or_else(|| tensors.get(name))))
            .and_then(|t| t.data.first().copied())
            .filter(|v| v.is_finite())
            .ok_or_else(|| "detector ONNX missing valid Softplus scale multiplier".to_owned())?;

        Ok(Self {
            stem_w_t,
            stem_b: stem_b_raw.data,
            stem_channels,
            stages,
            lat,
            fpn_channels,
            smooth,
            stem_lat,
            stem_lat_channels,
            head0,
            head1,
            head2,
            softplus_scale,
        })
    }

    /// Run detector forward pass on NCHW `[1, 1, height, width]` normalized pixels, returning NCHW
    /// `[1, 3, height / 2, width / 2]` stored inside `scratch.out`.
    pub fn run<'s>(
        &self,
        input_nchw: &[f32],
        height: usize,
        width: usize,
        scratch: &'s mut DetectorScratch,
    ) -> &'s [f32] {
        assert!(
            height >= 32 && width >= 32 && height.is_multiple_of(32) && width.is_multiple_of(32),
            "detector input {height}x{width} must be non-zero multiples of 32"
        );
        assert!(input_nchw.len() >= height * width);

        let h2 = height / 2;
        let w2 = width / 2;
        let h4 = h2 / 2;
        let w4 = w2 / 2;
        let h8 = h4 / 2;
        let w8 = w4 / 2;
        let h16 = h8 / 2;
        let w16 = w8 / 2;
        let h32 = h16 / 2;
        let w32 = w16 / 2;

        // Stage outputs held in separate buffers for FPN laterals.
        scratch.stem_act.resize(h2 * w2 * self.stem_channels, 0.0);
        stem_conv3x3_s2_hardswish(
            input_nchw,
            height,
            width,
            &self.stem_w_t,
            &self.stem_b,
            self.stem_channels,
            &mut scratch.stem_act,
        );

        let run_stage = |stage: &DetStage,
                         src: &[f32],
                         in_h: usize,
                         in_w: usize,
                         out_h: usize,
                         out_w: usize,
                         dst: &mut Vec<f32>,
                         tmp_a: &mut Vec<f32>,
                         tmp_exp: &mut Vec<f32>| {
            let in_c = stage.down_dw.channels;
            let out_c = stage.out_channels;
            let n = out_h * out_w;
            tmp_a.resize(n * in_c, 0.0);
            depthwise_conv2d_hardswish(&stage.down_dw, 2, src, in_h, in_w, tmp_a);
            dst.resize(n * out_c, 0.0);
            linear_packed::<false>(&tmp_a[..n * in_c], n, &stage.down_pw, dst);
            hardswish_inplace(&mut dst[..n * out_c]);

            for block in &stage.blocks {
                let ec = block.expand_channels;
                tmp_a.resize(n * ec, 0.0);
                linear_packed::<false>(&dst[..n * out_c], n, &block.expand, tmp_a);
                hardswish_inplace(&mut tmp_a[..n * ec]);
                tmp_exp.resize(n * ec, 0.0);
                depthwise_conv2d_hardswish(&block.dw, 1, tmp_a, out_h, out_w, tmp_exp);
                linear_packed::<true>(&tmp_exp[..n * ec], n, &block.project, dst);
            }
        };

        scratch.s0.resize(h4 * w4 * self.stages[0].out_channels, 0.0);
        run_stage(
            &self.stages[0],
            &scratch.stem_act,
            h2,
            w2,
            h4,
            w4,
            &mut scratch.s0,
            &mut scratch.a,
            &mut scratch.exp,
        );

        scratch.s1.resize(h8 * w8 * self.stages[1].out_channels, 0.0);
        run_stage(
            &self.stages[1],
            &scratch.s0,
            h4,
            w4,
            h8,
            w8,
            &mut scratch.s1,
            &mut scratch.a,
            &mut scratch.exp,
        );

        scratch.s2.resize(h16 * w16 * self.stages[2].out_channels, 0.0);
        run_stage(
            &self.stages[2],
            &scratch.s1,
            h8,
            w8,
            h16,
            w16,
            &mut scratch.s2,
            &mut scratch.a,
            &mut scratch.exp,
        );

        // Stage 3 goes directly into scratch.b, leaving s0/s1/s2 intact for FPN.
        run_stage(
            &self.stages[3],
            &scratch.s2,
            h16,
            w16,
            h32,
            w32,
            &mut scratch.b,
            &mut scratch.a,
            &mut scratch.exp,
        );

        let fpn_c = self.fpn_channels;
        // FPN top-down: lat3 (H/32) -> +lat2 (H/16) -> +lat1 (H/8) -> +lat0 (H/4) -> smooth.
        scratch.fpn_a.resize(h32 * w32 * fpn_c, 0.0);
        linear_packed::<false>(&scratch.b[..h32 * w32 * self.stages[3].out_channels], h32 * w32, &self.lat[3], &mut scratch.fpn_a);
        hardswish_inplace(&mut scratch.fpn_a[..h32 * w32 * fpn_c]);

        scratch.fpn_b.resize(h16 * w16 * fpn_c, 0.0);
        linear_packed::<false>(&scratch.s2[..h16 * w16 * self.stages[2].out_channels], h16 * w16, &self.lat[2], &mut scratch.fpn_b);
        hardswish_inplace(&mut scratch.fpn_b[..h16 * w16 * fpn_c]);
        upsample_nearest2x_add(&scratch.fpn_a, h32, w32, fpn_c, &mut scratch.fpn_b);

        scratch.fpn_a.resize(h8 * w8 * fpn_c, 0.0);
        linear_packed::<false>(&scratch.s1[..h8 * w8 * self.stages[1].out_channels], h8 * w8, &self.lat[1], &mut scratch.fpn_a);
        hardswish_inplace(&mut scratch.fpn_a[..h8 * w8 * fpn_c]);
        upsample_nearest2x_add(&scratch.fpn_b, h16, w16, fpn_c, &mut scratch.fpn_a);

        scratch.fpn_b.resize(h4 * w4 * fpn_c, 0.0);
        linear_packed::<false>(&scratch.s0[..h4 * w4 * self.stages[0].out_channels], h4 * w4, &self.lat[0], &mut scratch.fpn_b);
        hardswish_inplace(&mut scratch.fpn_b[..h4 * w4 * fpn_c]);
        upsample_nearest2x_add(&scratch.fpn_a, h8, w8, fpn_c, &mut scratch.fpn_b);

        scratch.fpn_a.resize(h4 * w4 * fpn_c, 0.0);
        depthwise_conv2d_hardswish(&self.smooth, 1, &scratch.fpn_b, h4, w4, &mut scratch.fpn_a);

        // stem_lat: [H2 * W2, stem_channels] -> [H2 * W2, out_pad(=8)] + HardSwish.
        let stem_lat_stride = self.stem_lat.out_pad;
        scratch.b.resize(h2 * w2 * stem_lat_stride, 0.0);
        linear_packed::<false>(&scratch.stem_act[..h2 * w2 * self.stem_channels], h2 * w2, &self.stem_lat, &mut scratch.b);
        hardswish_inplace(&mut scratch.b[..h2 * w2 * stem_lat_stride]);

        // Concat [smooth (fpn_c), PixelUnshuffle(2)(stem_lat) (stem_lat_c * 4)] into scratch.fpn_b.
        let concat_c = self.head0.inp;
        let sl_c = self.stem_lat_channels;
        scratch.fpn_b.resize(h4 * w4 * concat_c, 0.0);
        for y in 0..h4 {
            for x in 0..w4 {
                let p = y * w4 + x;
                let dst_px = &mut scratch.fpn_b[p * concat_c..(p + 1) * concat_c];
                dst_px[..fpn_c].copy_from_slice(&scratch.fpn_a[p * fpn_c..(p + 1) * fpn_c]);
                let tail = &mut dst_px[fpn_c..];
                let r0 = (2 * y) * w2 + 2 * x;
                let r1 = (2 * y + 1) * w2 + 2 * x;
                for c in 0..sl_c {
                    tail[c * 4] = scratch.b[r0 * stem_lat_stride + c];
                    tail[c * 4 + 1] = scratch.b[(r0 + 1) * stem_lat_stride + c];
                    tail[c * 4 + 2] = scratch.b[r1 * stem_lat_stride + c];
                    tail[c * 4 + 3] = scratch.b[(r1 + 1) * stem_lat_stride + c];
                }
            }
        }

        // Head: head0 (concat_c -> fpn_c + HS) -> head1 (DW 3x3 + HS) -> head2 (fpn_c -> 12).
        scratch.a.resize(h4 * w4 * fpn_c, 0.0);
        linear_packed::<false>(&scratch.fpn_b[..h4 * w4 * concat_c], h4 * w4, &self.head0, &mut scratch.a);
        hardswish_inplace(&mut scratch.a[..h4 * w4 * fpn_c]);

        depthwise_conv2d_hardswish(&self.head1, 1, &scratch.a, h4, w4, &mut scratch.fpn_a);

        let head2_stride = self.head2.out_pad;
        scratch.b.resize(h4 * w4 * head2_stride, 0.0);
        linear_packed::<false>(&scratch.fpn_a[..h4 * w4 * fpn_c], h4 * w4, &self.head2, &mut scratch.b);

        // DepthToSpace(blocksize=2, mode='CRD') into NCHW [3, H2, W2] with Softplus * scale on ch 1..2.
        let plane = h2 * w2;
        scratch.out.resize(3 * plane, 0.0);
        let scale = self.softplus_scale;
        let (out0, rest) = scratch.out.split_at_mut(plane);
        let (out1, out2) = rest.split_at_mut(plane);
        for y in 0..h4 {
            let row0 = (2 * y) * w2;
            let row1 = (2 * y + 1) * w2;
            for x in 0..w4 {
                let px = &scratch.b[(y * w4 + x) * head2_stride..];
                let col = 2 * x;
                out0[row0 + col] = px[0];
                out0[row0 + col + 1] = px[1];
                out0[row1 + col] = px[2];
                out0[row1 + col + 1] = px[3];

                out1[row0 + col] = softplus(px[4]) * scale;
                out1[row0 + col + 1] = softplus(px[5]) * scale;
                out1[row1 + col] = softplus(px[6]) * scale;
                out1[row1 + col + 1] = softplus(px[7]) * scale;

                out2[row0 + col] = softplus(px[8]) * scale;
                out2[row0 + col + 1] = softplus(px[9]) * scale;
                out2[row1 + col] = softplus(px[10]) * scale;
                out2[row1 + col + 1] = softplus(px[11]) * scale;
            }
        }

        &scratch.out
    }
}

#[inline(always)]
fn softplus(v: f32) -> f32 {
    if v > 20.0 {
        v
    } else {
        v.exp().ln_1p()
    }
}

// ============================================================================
// Stem Conv3x3 (stride 2, pad 1, C_in=1) + fused HardSwish
// ============================================================================

fn stem_conv3x3_s2_hardswish(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    w_t: &[f32],
    b: &[f32],
    channels: usize,
    dst: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        unsafe { stem_conv3x3_s2_hardswish_avx2(src, in_h, in_w, w_t, b, channels, dst) };
        return;
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        stem_conv3x3_s2_hardswish_wasm(src, in_h, in_w, w_t, b, channels, dst);
        return;
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let out_h = in_h / 2;
        let out_w = in_w / 2;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let d_px = &mut dst[(oy * out_w + ox) * channels..(oy * out_w + ox + 1) * channels];
                d_px.copy_from_slice(&b[..channels]);
                for ky in 0..3isize {
                    let iy = oy as isize * 2 - 1 + ky;
                    if iy < 0 || iy >= in_h as isize {
                        continue;
                    }
                    let row = &src[iy as usize * in_w..];
                    for kx in 0..3isize {
                        let ix = ox as isize * 2 - 1 + kx;
                        if ix < 0 || ix >= in_w as isize {
                            continue;
                        }
                        let xv = row[ix as usize];
                        let wk = &w_t[(ky as usize * 3 + kx as usize) * channels..];
                        for c in 0..channels {
                            d_px[c] += xv * wk[c];
                        }
                    }
                }
                for v in d_px.iter_mut() {
                    *v = *v * ((*v + 3.0).clamp(0.0, 6.0) * HARD_SWISH_SIXTH);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn stem_conv3x3_s2_hardswish_avx2(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    w_t: &[f32],
    b: &[f32],
    channels: usize,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let out_h = in_h / 2;
    let out_w = in_w / 2;
    let src_p = src.as_ptr();
    let w_p = w_t.as_ptr();
    let b_p = b.as_ptr();
    let dst_p = dst.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let three = _mm256_set1_ps(3.0);
    let six = _mm256_set1_ps(6.0);
    let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);

    for oy in 0..out_h {
        for ox in 0..out_w {
            let out_base = dst_p.add((oy * out_w + ox) * channels);
            let mut c = 0usize;
            let interior = oy >= 1 && ox >= 1 && (oy * 2 + 1) < in_h && (ox * 2 + 1) < in_w;
            if interior {
                let s0 = src_p.add((oy * 2 - 1) * in_w + ox * 2 - 1);
                let s1 = s0.add(in_w);
                let s2 = s1.add(in_w);
                let x00 = _mm256_broadcast_ss(&*s0);
                let x01 = _mm256_broadcast_ss(&*s0.add(1));
                let x02 = _mm256_broadcast_ss(&*s0.add(2));
                let x10 = _mm256_broadcast_ss(&*s1);
                let x11 = _mm256_broadcast_ss(&*s1.add(1));
                let x12 = _mm256_broadcast_ss(&*s1.add(2));
                let x20 = _mm256_broadcast_ss(&*s2);
                let x21 = _mm256_broadcast_ss(&*s2.add(1));
                let x22 = _mm256_broadcast_ss(&*s2.add(2));
                while c + 8 <= channels {
                    let mut acc = _mm256_loadu_ps(b_p.add(c));
                    acc = _mm256_fmadd_ps(x00, _mm256_loadu_ps(w_p.add(c)), acc);
                    acc = _mm256_fmadd_ps(x01, _mm256_loadu_ps(w_p.add(channels + c)), acc);
                    acc = _mm256_fmadd_ps(x02, _mm256_loadu_ps(w_p.add(2 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x10, _mm256_loadu_ps(w_p.add(3 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x11, _mm256_loadu_ps(w_p.add(4 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x12, _mm256_loadu_ps(w_p.add(5 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x20, _mm256_loadu_ps(w_p.add(6 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x21, _mm256_loadu_ps(w_p.add(7 * channels + c)), acc);
                    acc = _mm256_fmadd_ps(x22, _mm256_loadu_ps(w_p.add(8 * channels + c)), acc);
                    let gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(acc, three), zero), six);
                    _mm256_storeu_ps(out_base.add(c), _mm256_mul_ps(acc, _mm256_mul_ps(gate, sixth)));
                    c += 8;
                }
            } else {
                while c + 8 <= channels {
                    let mut acc = _mm256_loadu_ps(b_p.add(c));
                    for ky in 0..3isize {
                        let iy = oy as isize * 2 - 1 + ky;
                        if iy < 0 || iy >= in_h as isize {
                            continue;
                        }
                        for kx in 0..3isize {
                            let ix = ox as isize * 2 - 1 + kx;
                            if ix < 0 || ix >= in_w as isize {
                                continue;
                            }
                            let xv = _mm256_broadcast_ss(&*src_p.add(iy as usize * in_w + ix as usize));
                            let wv = _mm256_loadu_ps(w_p.add((ky as usize * 3 + kx as usize) * channels + c));
                            acc = _mm256_fmadd_ps(xv, wv, acc);
                        }
                    }
                    let gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(acc, three), zero), six);
                    _mm256_storeu_ps(out_base.add(c), _mm256_mul_ps(acc, _mm256_mul_ps(gate, sixth)));
                    c += 8;
                }
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn stem_conv3x3_s2_hardswish_wasm(
    src: &[f32],
    in_h: usize,
    in_w: usize,
    w_t: &[f32],
    b: &[f32],
    channels: usize,
    dst: &mut [f32],
) {
    use core::arch::wasm32::*;
    let out_h = in_h / 2;
    let out_w = in_w / 2;
    let src_p = src.as_ptr();
    let w_p = w_t.as_ptr();
    let b_p = b.as_ptr();
    let dst_p = dst.as_mut_ptr();
    let zero = f32x4_splat(0.0);
    let three = f32x4_splat(3.0);
    let six = f32x4_splat(6.0);
    let sixth = f32x4_splat(HARD_SWISH_SIXTH);

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut c = 0usize;
            while c + 4 <= channels {
                unsafe {
                    let mut acc = v128_load(b_p.add(c) as *const v128);
                    for ky in 0..3isize {
                        let iy = oy as isize * 2 - 1 + ky;
                        if iy < 0 || iy >= in_h as isize {
                            continue;
                        }
                        for kx in 0..3isize {
                            let ix = ox as isize * 2 - 1 + kx;
                            if ix < 0 || ix >= in_w as isize {
                                continue;
                            }
                            let xv = f32x4_splat(*src_p.add(iy as usize * in_w + ix as usize));
                            let wv = v128_load(w_p.add((ky as usize * 3 + kx as usize) * channels + c) as *const v128);
                            acc = f32x4_add(acc, f32x4_mul(xv, wv));
                        }
                    }
                    let gate = f32x4_pmin(f32x4_pmax(f32x4_add(acc, three), zero), six);
                    v128_store(
                        dst_p.add((oy * out_w + ox) * channels + c) as *mut v128,
                        f32x4_mul(acc, f32x4_mul(gate, sixth)),
                    );
                }
                c += 4;
            }
        }
    }
}

// ============================================================================
// Depthwise Conv2D (K in {3, 5}, stride in {1, 2}, pad = K/2) + fused HardSwish
// ============================================================================

fn depthwise_conv2d_hardswish(
    dw: &DwWeights,
    stride: usize,
    src: &[f32],
    in_h: usize,
    in_w: usize,
    dst: &mut [f32],
) {
    let out_h = in_h / stride;
    let out_w = in_w / stride;
    let c = dw.channels;
    assert!(src.len() >= in_h * in_w * c && dst.len() >= out_h * out_w * c);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        match (dw.kernel, stride) {
            (3, 1) => unsafe { dw_avx2::<3, 1>(dw, src, in_h, in_w, out_h, out_w, dst) },
            (3, 2) => unsafe { dw_avx2::<3, 2>(dw, src, in_h, in_w, out_h, out_w, dst) },
            (5, 1) => unsafe { dw_avx2::<5, 1>(dw, src, in_h, in_w, out_h, out_w, dst) },
            (5, 2) => unsafe { dw_avx2::<5, 2>(dw, src, in_h, in_w, out_h, out_w, dst) },
            _ => unreachable!(),
        }
        return;
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        match (dw.kernel, stride) {
            (3, 1) => dw_wasm::<3, 1>(dw, src, in_h, in_w, out_h, out_w, dst),
            (3, 2) => dw_wasm::<3, 2>(dw, src, in_h, in_w, out_h, out_w, dst),
            (5, 1) => dw_wasm::<5, 1>(dw, src, in_h, in_w, out_h, out_w, dst),
            (5, 2) => dw_wasm::<5, 2>(dw, src, in_h, in_w, out_h, out_w, dst),
            _ => unreachable!(),
        }
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    {
        let pad = (dw.kernel / 2) as isize;
        let channels = dw.channels;
        let k = dw.kernel;
        for oy in 0..out_h {
            for ox in 0..out_w {
                let d_px = &mut dst[(oy * out_w + ox) * channels..(oy * out_w + ox + 1) * channels];
                d_px.copy_from_slice(&dw.b);
                for ky in 0..k {
                    let iy = oy as isize * stride as isize - pad + ky as isize;
                    if iy < 0 || iy >= in_h as isize {
                        continue;
                    }
                    for kx in 0..k {
                        let ix = ox as isize * stride as isize - pad + kx as isize;
                        if ix < 0 || ix >= in_w as isize {
                            continue;
                        }
                        let s_px = &src[(iy as usize * in_w + ix as usize) * channels..];
                        let w_px = &dw.w_t[(ky * k + kx) * channels..];
                        for ch in 0..channels {
                            d_px[ch] += s_px[ch] * w_px[ch];
                        }
                    }
                }
                for v in d_px.iter_mut() {
                    *v = *v * ((*v + 3.0).clamp(0.0, 6.0) * HARD_SWISH_SIXTH);
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dw_avx2<const K: usize, const STRIDE: usize>(
    dw: &DwWeights,
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let channels = dw.channels;
    let pad = (K / 2) as isize;
    let s_p = src.as_ptr();
    let w_p = dw.w_t.as_ptr();
    let b_p = dw.b.as_ptr();
    let d_p = dst.as_mut_ptr();
    let zero = _mm256_setzero_ps();
    let three = _mm256_set1_ps(3.0);
    let six = _mm256_set1_ps(6.0);
    let sixth = _mm256_set1_ps(HARD_SWISH_SIXTH);

    for oy in 0..out_h {
        let iy_base = oy as isize * STRIDE as isize - pad;
        let y_in = iy_base >= 0 && iy_base + K as isize <= in_h as isize;
        for ox in 0..out_w {
            let ix_base = ox as isize * STRIDE as isize - pad;
            let interior = y_in && ix_base >= 0 && ix_base + K as isize <= in_w as isize;
            let out_base = d_p.add((oy * out_w + ox) * channels);
            let mut ch = 0usize;
            if interior {
                let uy = iy_base as usize;
                let ux = ix_base as usize;
                while ch + 8 <= channels {
                    let mut acc = _mm256_loadu_ps(b_p.add(ch));
                    for ky in 0..K {
                        let row_p = s_p.add(((uy + ky) * in_w + ux) * channels + ch);
                        for kx in 0..K {
                            let sv = _mm256_loadu_ps(row_p.add(kx * channels));
                            let wv = _mm256_loadu_ps(w_p.add((ky * K + kx) * channels + ch));
                            acc = _mm256_fmadd_ps(sv, wv, acc);
                        }
                    }
                    let gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(acc, three), zero), six);
                    _mm256_storeu_ps(out_base.add(ch), _mm256_mul_ps(acc, _mm256_mul_ps(gate, sixth)));
                    ch += 8;
                }
            } else {
                while ch + 8 <= channels {
                    let mut acc = _mm256_loadu_ps(b_p.add(ch));
                    for ky in 0..K {
                        let iy = iy_base + ky as isize;
                        if iy < 0 || iy >= in_h as isize {
                            continue;
                        }
                        for kx in 0..K {
                            let ix = ix_base + kx as isize;
                            if ix < 0 || ix >= in_w as isize {
                                continue;
                            }
                            let sv = _mm256_loadu_ps(s_p.add((iy as usize * in_w + ix as usize) * channels + ch));
                            let wv = _mm256_loadu_ps(w_p.add((ky * K + kx) * channels + ch));
                            acc = _mm256_fmadd_ps(sv, wv, acc);
                        }
                    }
                    let gate = _mm256_min_ps(_mm256_max_ps(_mm256_add_ps(acc, three), zero), six);
                    _mm256_storeu_ps(out_base.add(ch), _mm256_mul_ps(acc, _mm256_mul_ps(gate, sixth)));
                    ch += 8;
                }
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn dw_wasm<const K: usize, const STRIDE: usize>(
    dw: &DwWeights,
    src: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dst: &mut [f32],
) {
    use core::arch::wasm32::*;
    let channels = dw.channels;
    let pad = (K / 2) as isize;
    let s_p = src.as_ptr();
    let w_p = dw.w_t.as_ptr();
    let b_p = dw.b.as_ptr();
    let d_p = dst.as_mut_ptr();
    let zero = f32x4_splat(0.0);
    let three = f32x4_splat(3.0);
    let six = f32x4_splat(6.0);
    let sixth = f32x4_splat(HARD_SWISH_SIXTH);

    for oy in 0..out_h {
        let iy_base = oy as isize * STRIDE as isize - pad;
        let y_in = iy_base >= 0 && iy_base + K as isize <= in_h as isize;
        for ox in 0..out_w {
            let ix_base = ox as isize * STRIDE as isize - pad;
            let interior = y_in && ix_base >= 0 && ix_base + K as isize <= in_w as isize;
            let mut ch = 0usize;
            if interior {
                let uy = iy_base as usize;
                let ux = ix_base as usize;
                while ch + 4 <= channels {
                    unsafe {
                        let mut acc = v128_load(b_p.add(ch) as *const v128);
                        for ky in 0..K {
                            let row_p = s_p.add(((uy + ky) * in_w + ux) * channels + ch);
                            for kx in 0..K {
                                let sv = v128_load(row_p.add(kx * channels) as *const v128);
                                let wv = v128_load(w_p.add((ky * K + kx) * channels + ch) as *const v128);
                                acc = f32x4_add(acc, f32x4_mul(sv, wv));
                            }
                        }
                        let gate = f32x4_pmin(f32x4_pmax(f32x4_add(acc, three), zero), six);
                        v128_store(
                            d_p.add((oy * out_w + ox) * channels + ch) as *mut v128,
                            f32x4_mul(acc, f32x4_mul(gate, sixth)),
                        );
                    }
                    ch += 4;
                }
            } else {
                while ch + 4 <= channels {
                    unsafe {
                        let mut acc = v128_load(b_p.add(ch) as *const v128);
                        for ky in 0..K {
                            let iy = iy_base + ky as isize;
                            if iy < 0 || iy >= in_h as isize {
                                continue;
                            }
                            for kx in 0..K {
                                let ix = ix_base + kx as isize;
                                if ix < 0 || ix >= in_w as isize {
                                    continue;
                                }
                                let sv = v128_load(s_p.add((iy as usize * in_w + ix as usize) * channels + ch) as *const v128);
                                let wv = v128_load(w_p.add((ky * K + kx) * channels + ch) as *const v128);
                                acc = f32x4_add(acc, f32x4_mul(sv, wv));
                            }
                        }
                        let gate = f32x4_pmin(f32x4_pmax(f32x4_add(acc, three), zero), six);
                        v128_store(
                            d_p.add((oy * out_w + ox) * channels + ch) as *mut v128,
                            f32x4_mul(acc, f32x4_mul(gate, sixth)),
                        );
                    }
                    ch += 4;
                }
            }
        }
    }
}

// ============================================================================
// Nearest 2x upsample add in NHWC
// ============================================================================

fn upsample_nearest2x_add(src: &[f32], src_h: usize, src_w: usize, channels: usize, dst: &mut [f32]) {
    let dst_w = src_w * 2;
    assert!(src.len() >= src_h * src_w * channels && dst.len() >= src_h * 2 * dst_w * channels);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        unsafe { upsample_nearest2x_add_avx2(src, src_h, src_w, dst_w, channels, dst) };
        return;
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        upsample_nearest2x_add_wasm(src, src_h, src_w, dst_w, channels, dst);
        return;
    }

    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for ly in 0..src_h {
        let row0 = (2 * ly) * dst_w;
        let row1 = (2 * ly + 1) * dst_w;
        for lx in 0..src_w {
            let s_px = &src[(ly * src_w + lx) * channels..(ly * src_w + lx + 1) * channels];
            for idx in [row0 + 2 * lx, row0 + 2 * lx + 1, row1 + 2 * lx, row1 + 2 * lx + 1] {
                let d_px = &mut dst[idx * channels..(idx + 1) * channels];
                for c in 0..channels {
                    d_px[c] += s_px[c];
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn upsample_nearest2x_add_avx2(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_w: usize,
    channels: usize,
    dst: &mut [f32],
) {
    use std::arch::x86_64::*;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    for ly in 0..src_h {
        let row0 = (2 * ly) * dst_w;
        let row1 = (2 * ly + 1) * dst_w;
        for lx in 0..src_w {
            let s_px = sp.add((ly * src_w + lx) * channels);
            let d00 = dp.add((row0 + 2 * lx) * channels);
            let d01 = d00.add(channels);
            let d10 = dp.add((row1 + 2 * lx) * channels);
            let d11 = d10.add(channels);
            let mut c = 0usize;
            while c + 8 <= channels {
                let sv = _mm256_loadu_ps(s_px.add(c));
                _mm256_storeu_ps(d00.add(c), _mm256_add_ps(_mm256_loadu_ps(d00.add(c)), sv));
                _mm256_storeu_ps(d01.add(c), _mm256_add_ps(_mm256_loadu_ps(d01.add(c)), sv));
                _mm256_storeu_ps(d10.add(c), _mm256_add_ps(_mm256_loadu_ps(d10.add(c)), sv));
                _mm256_storeu_ps(d11.add(c), _mm256_add_ps(_mm256_loadu_ps(d11.add(c)), sv));
                c += 8;
            }
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
fn upsample_nearest2x_add_wasm(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_w: usize,
    channels: usize,
    dst: &mut [f32],
) {
    use core::arch::wasm32::*;
    let sp = src.as_ptr();
    let dp = dst.as_mut_ptr();
    for ly in 0..src_h {
        let row0 = (2 * ly) * dst_w;
        let row1 = (2 * ly + 1) * dst_w;
        for lx in 0..src_w {
            unsafe {
                let s_px = sp.add((ly * src_w + lx) * channels);
                let d00 = dp.add((row0 + 2 * lx) * channels);
                let d01 = d00.add(channels);
                let d10 = dp.add((row1 + 2 * lx) * channels);
                let d11 = d10.add(channels);
                let mut c = 0usize;
                while c + 4 <= channels {
                    let sv = v128_load(s_px.add(c) as *const v128);
                    v128_store(d00.add(c) as *mut v128, f32x4_add(v128_load(d00.add(c) as *const v128), sv));
                    v128_store(d01.add(c) as *mut v128, f32x4_add(v128_load(d01.add(c) as *const v128), sv));
                    v128_store(d10.add(c) as *mut v128, f32x4_add(v128_load(d10.add(c) as *const v128), sv));
                    v128_store(d11.add(c) as *mut v128, f32x4_add(v128_load(d11.add(c) as *const v128), sv));
                    c += 4;
                }
            }
        }
    }
}

#[cfg(target_arch = "wasm32")]
pub mod wasm_abi {
    use super::{DetectorNative, DetectorScratch};

    struct WasmDetector {
        model: DetectorNative,
        scratch: DetectorScratch,
    }

    #[no_mangle]
    pub extern "C" fn detector_load(ptr: *const u8, len: usize) -> usize {
        if ptr.is_null() || len == 0 {
            return 0;
        }
        // SAFETY: caller provides `len` readable bytes allocated in Wasm linear memory.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        match DetectorNative::from_onnx_bytes(bytes) {
            Ok(model) => Box::into_raw(Box::new(WasmDetector {
                model,
                scratch: DetectorScratch::default(),
            })) as usize,
            Err(_) => 0,
        }
    }

    #[no_mangle]
    pub extern "C" fn detector_free(handle: usize) {
        if handle != 0 {
            // SAFETY: handle was created by Box::into_raw in detector_load.
            unsafe { drop(Box::from_raw(handle as *mut WasmDetector)) };
        }
    }

    #[no_mangle]
    pub extern "C" fn detector_run(
        handle: usize,
        input_ptr: *const f32,
        height: usize,
        width: usize,
        out_ptr: *mut f32,
    ) -> usize {
        if handle == 0
            || input_ptr.is_null()
            || out_ptr.is_null()
            || height < 32
            || width < 32
            || !height.is_multiple_of(32)
            || !width.is_multiple_of(32)
        {
            return 0;
        }
        // SAFETY: caller passes `height * width` input f32s and `3 * (height/2) * (width/2)` output slots.
        let state = unsafe { &mut *(handle as *mut WasmDetector) };
        let input = unsafe { std::slice::from_raw_parts(input_ptr, height * width) };
        let output = state.model.run(input, height, width, &mut state.scratch);
        let dst = unsafe { std::slice::from_raw_parts_mut(out_ptr, output.len()) };
        dst.copy_from_slice(output);
        output.len()
    }
}
