//! Tier 1 and tier 2's local scores in the engine: the document encoder `tier.py` fits and exports
//! (`encoder.onnx` beside `encoder.json`, LEGALPDF_TRANSITION_ENCODER), run over a whole document at
//! once. It reads each line's features as the parser reads them (standardized by the fit's means and
//! spreads, clipped at 8), its text as the tagger's runner does (each character min(code point, 0x2FFF)
//! % 2048 + 1, each lowercase word's FNV-1a % 16383 + 1) and its look among the document's, and gives
//! each line every region-and-opening class's probability. The parser and its ranker are the same over
//! any tier's scores.

use super::*;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::TensorRef;
use std::sync::Mutex;

#[derive(serde::Deserialize)]
struct Meta {
    width: usize,
    chars: usize,
    words: usize,
    classes: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
}

pub(crate) struct Encoder {
    session: Mutex<Session>,
    meta: Meta,
}

fn fnv1a(token: &str) -> u32 {
    token.bytes().fold(0x811C_9DC5u32, |h, byte| (h ^ u32::from(byte)).wrapping_mul(0x0100_0193))
}

impl Encoder {
    /// The encoder under `dir`, its ONNX Runtime the one LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME names.
    pub(crate) fn load(dir: &Path) -> std::result::Result<Self, String> {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("encoder.json")).map_err(|e| format!("encoder.json: {e}"))?)
            .map_err(|e| format!("encoder.json: {e}"))?;
        if meta.width != ALL || meta.classes != CLASSES || meta.mean.len() != ALL || meta.std.len() != ALL {
            return Err(format!("encoder.json reads {} features and {} classes; the parser has {ALL} and {CLASSES}", meta.width, meta.classes));
        }
        let runtime = ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"].iter().find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()))
            .ok_or("the encoder needs LEGALPDF_ONNX_RUNTIME")?;
        legal_pdf_core::init_ort_runtime(Path::new(&runtime), None).map_err(|e| e.to_string())?;
        let session = Session::builder()
            .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
            .and_then(|b| b.with_intra_threads(1))
            .and_then(|b| b.with_inter_threads(1))
            .and_then(|b| b.commit_from_file(dir.join("encoder.onnx")))
            .map_err(|e| format!("encoder.onnx: {e}"))?;
        Ok(Self { session: Mutex::new(session), meta })
    }

    /// Each line's class probabilities, lines in reading order, `CLASSES` a line.
    pub(crate) fn probabilities(&self, table: &LineTable, texts: &[&str]) -> std::result::Result<Vec<f64>, String> {
        let n = table.order.len();
        let m = &self.meta;
        let features: Vec<f32> = table.features.iter().enumerate()
            .map(|(at, &value)| ((value - m.mean[at % ALL]) / m.std[at % ALL]).clamp(-8.0, 8.0)).collect();
        let mut chars = vec![0i64; n * m.chars];
        let mut words = vec![0i64; n * m.words];
        for (k, text) in texts.iter().enumerate() {
            for (c, ch) in text.chars().take(m.chars).enumerate() {
                chars[k * m.chars + c] = i64::from((u32::from(ch)).min(0x2FFF) % 2048) + 1;
            }
            let lower = text.to_lowercase();
            let tokens = lower.split(|c: char| !(c.is_ascii_alphanumeric() || ('\u{00C0}'..='\u{024F}').contains(&c))).filter(|t| !t.is_empty());
            for (w, token) in tokens.take(m.words).enumerate() {
                words[k * m.words + w] = i64::from(fnv1a(token) % 16383) + 1;
            }
        }
        // Each line's look among the document's looks, in their sorted order.
        let mut distinct = table.looks.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let looks: Vec<i64> = table.looks.iter().map(|look| distinct.binary_search(look).unwrap_or(0) as i64).collect();
        let f = TensorRef::from_array_view(([n, ALL], &features[..])).map_err(|e| e.to_string())?;
        let c = TensorRef::from_array_view(([n, m.chars], &chars[..])).map_err(|e| e.to_string())?;
        let w = TensorRef::from_array_view(([n, m.words], &words[..])).map_err(|e| e.to_string())?;
        let l = TensorRef::from_array_view(([n], &looks[..])).map_err(|e| e.to_string())?;
        let mut session = self.session.lock().map_err(|_| "encoder session poisoned")?;
        let outputs = session.run(ort::inputs!["features" => f, "chars" => c, "words" => w, "looks" => l]).map_err(|e| format!("encoder run: {e}"))?;
        let (_, logits) = outputs["logits"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok(logits.chunks_exact(CLASSES).flat_map(|row| {
            let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let total: f32 = row.iter().map(|x| (x - top).exp()).sum();
            row.iter().map(move |x| f64::from((x - top).exp() / total))
        }).collect())
    }
}
