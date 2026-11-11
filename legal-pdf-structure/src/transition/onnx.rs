//! The ONNX Runtime as DOC's stages run in it: the runtime found, a graph's session, its inputs and outputs.

use std::path::Path;
use ort::session::{builder::GraphOptimizationLevel, Session};

/// A graph's input of `shape` over `data`, its error as text.
macro_rules! tensor { ($shape:expr, $data:expr) => { ort::value::TensorRef::from_array_view(($shape, $data)).map_err(|e| e.to_string())? }; }
/// A graph's output `name`, its values, its error as text.
macro_rules! output { ($outputs:expr, $name:expr) => { $outputs[$name].try_extract_tensor::<f32>().map_err(|e| e.to_string())?.1 }; }

/// The ONNX Runtime at `runtime`, else the one LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME names, initialized; `missing`
/// the error where none is named.
pub(crate) fn init_runtime(runtime: Option<&Path>, missing: &str) -> std::result::Result<(), String> {
    let named = ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"].iter().find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()));
    let runtime = runtime.map(Path::as_os_str).or(named.as_deref()).ok_or(missing)?;
    legal_pdf_core::init_ort_runtime(Path::new(runtime), None).map_err(|e| e.to_string())?;
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
