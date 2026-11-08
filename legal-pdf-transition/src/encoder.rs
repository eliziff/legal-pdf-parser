//! Tier 1 and tier 2's local scores in the engine: the document encoder `tier.py` fits and exports
//! (`encoder.onnx` beside `encoder.json`, LEGALPDF_TRANSITION_ENCODER), run over a whole document at
//! once. It reads each line's features as the parser reads them (standardized by the fit's means and
//! spreads, clipped at 8), its text as the tagger's runner does (each character min(code point, 0x2FFF)
//! % 2048 + 1, each lowercase word's FNV-1a % 16383 + 1), its look and its template among the document's
//! (`stylesheet::template_ids`), and the document's openers (`stylesheet::opener`), whose states attend
//! to one another across the whole document, each line reading the last opener at or before it; and
//! gives each line every region-and-opening class's probability and, were it a heading's opening, each
//! level's. The parser and its ranker are the same over any tier's scores.

use super::*;
use ort::session::{builder::GraphOptimizationLevel, Session};
use std::sync::Mutex;

/// A graph's input of `shape` over `data`, its error as text.
macro_rules! tensor { ($shape:expr, $data:expr) => { ort::value::TensorRef::from_array_view(($shape, $data)).map_err(|e| e.to_string())? }; }
/// A graph's output `name`, its values, its error as text.
macro_rules! output { ($outputs:expr, $name:expr) => { $outputs[$name].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1 }; }

#[derive(serde::Deserialize)]
struct Meta {
    width: usize,
    chars: usize,
    words: usize,
    classes: usize,
    mean: Vec<f32>,
    std: Vec<f32>,
}

pub struct Encoder {
    session: Mutex<Session>,
    meta: Meta,
}

fn fnv1a(token: &str) -> u32 {
    token.bytes().fold(0x811C_9DC5u32, |h, byte| (h ^ u32::from(byte)).wrapping_mul(0x0100_0193))
}

/// The ONNX Runtime LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME names, initialized; `missing` the error where neither does.
pub(crate) fn init_runtime(missing: &str) -> std::result::Result<(), String> {
    let runtime = ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"].iter().find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()))
        .ok_or(missing)?;
    legal_pdf_core::init_ort_runtime(Path::new(&runtime), None).map_err(|e| e.to_string())?;
    Ok(())
}

/// The graph at `path` in a session of one thread, its error named `name`.
pub(crate) fn session(path: &Path, name: impl std::fmt::Display) -> std::result::Result<Session, String> {
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(1))
        .and_then(|b| b.with_inter_threads(1))
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| format!("{name}: {e}"))
}

/// Each row of a `width`-wide matrix made a probability distribution.
pub(crate) fn softmax_rows(values: &[f32], width: usize) -> Vec<f32> {
    values.chunks_exact(width).flat_map(|row| {
        let top = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total: f32 = row.iter().map(|x| (x - top).exp()).sum();
        row.iter().map(move |x| (x - top).exp() / total)
    }).collect()
}

impl Encoder {
    /// The encoder under `dir`, its ONNX Runtime the one LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME names.
    pub fn load(dir: &Path) -> std::result::Result<Self, String> {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("encoder.json")).map_err(|e| format!("encoder.json: {e}"))?)
            .map_err(|e| format!("encoder.json: {e}"))?;
        if meta.width != ALL || meta.classes != CLASSES || meta.mean.len() != ALL || meta.std.len() != ALL {
            return Err(format!("encoder.json reads {} features and {} classes; the parser has {ALL} and {CLASSES}", meta.width, meta.classes));
        }
        init_runtime("the encoder needs LEGALPDF_ONNX_RUNTIME")?;
        Ok(Self { session: Mutex::new(session(&dir.join("encoder.onnx"), "encoder.onnx")?), meta })
    }

    /// Each line's class probabilities (`CLASSES` a line) and heading-level probabilities
    /// (`LEVEL_CLASSES` a line), lines in reading order.
    pub fn probabilities(&self, table: &LineTable, measures: &Measures, texts: &[&str], template: &[u32]) -> std::result::Result<(Vec<f32>, Vec<f32>), String> {
        let (n, m) = (table.readings.len(), &self.meta);
        let features: Vec<f32> = measures.features.iter().enumerate()
            .map(|(at, &value)| ((value - m.mean[at % ALL]) / m.std[at % ALL]).clamp(-8.0, 8.0)).collect();
        let mut chars = vec![0i64; n * m.chars];
        let mut words = vec![0i64; n * m.words];
        for (k, text) in texts.iter().enumerate() {
            text.chars().take(m.chars).enumerate().for_each(|(c, ch)| chars[k * m.chars + c] = i64::from((u32::from(ch)).min(0x2FFF) % 2048) + 1);
            let lower = text.to_lowercase();
            let tokens = lower.split(|c: char| !(c.is_ascii_alphanumeric() || ('\u{00C0}'..='\u{024F}').contains(&c))).filter(|t| !t.is_empty());
            tokens.take(m.words).enumerate().for_each(|(w, token)| words[k * m.words + w] = i64::from(fnv1a(token) % 16383) + 1);
        }
        // Each line's look among the document's looks, in their sorted order.
        let mut distinct = table.looks.clone();
        distinct.sort_unstable();
        distinct.dedup();
        let looks: Vec<i64> = table.looks.iter().map(|look| distinct.binary_search(look).unwrap_or(0) as i64).collect();
        let templates: Vec<i64> = template.iter().map(|&t| i64::from(t)).collect();
        // The openers' places (the first line where none is one), and each line's last opener at or
        // before it, counted from 1 (0 for none).
        let mut openers: Vec<i64> = (0..n).filter(|&k| opener(table, measures, k)).map(|k| k as i64).collect();
        if openers.is_empty() { openers.push(0); }
        let mut seen = 0usize;
        let last_opener: Vec<i64> = (0..n).map(|k| {
            seen += openers[seen..].iter().take_while(|&&opener| opener as usize <= k).count();
            seen as i64
        }).collect();
        let inputs = ort::inputs!["features" => tensor!([n, ALL], &features[..]), "chars" => tensor!([n, m.chars], &chars[..]), "words" => tensor!([n, m.words], &words[..]),
            "looks" => tensor!([n], &looks[..]), "templates" => tensor!([n], &templates[..]), "openers" => tensor!([openers.len()], &openers[..]),
            "last_opener" => tensor!([n], &last_opener[..])];
        let mut session = self.session.lock().map_err(|_| "encoder session poisoned")?;
        let outputs = session.run(inputs).map_err(|e| format!("encoder run: {e}"))?;
        Ok((softmax_rows(output!(outputs, "logits"), CLASSES), softmax_rows(output!(outputs, "levels"), LEVEL_CLASSES)))
    }
}
