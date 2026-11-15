//! The ONNX Runtime as DOC's stages run in it: the runtime found, a graph's session, its inputs and outputs.

use std::path::Path;
use ort::execution_providers::CPUExecutionProvider;
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

/// The graph at `path` in a session of `threads` threads, its error named `name`. No CPU memory arena and no memory-pattern
/// plan: DOC's graphs take a document's sizes (lines, regions, a tree's nodes), and an arena kept by each pooled session
/// grows to the largest document it has read and never gives it back; a grade of the gold store held 53 GB so. Each
/// run's buffers are freed when it ends.
/// On the backend LEGALPDF_DOC_BACKEND names ("directml", "cuda", as the recognizer's backends are named; LEGALPDF_DOC_DEVICE
/// its device), else the CPU.
pub(crate) fn session(path: &Path, name: impl std::fmt::Display, threads: usize) -> std::result::Result<Session, String> {
    let backend: legal_pdf_core::OrtBackend = std::env::var("LEGALPDF_DOC_BACKEND").ok()
        .map(|value| serde_json::from_value(serde_json::Value::String(value)).map_err(|e| format!("LEGALPDF_DOC_BACKEND: {e}")))
        .transpose()?.unwrap_or_default();
    let device = backend.normalized_device(std::env::var("LEGALPDF_DOC_DEVICE").ok().as_deref()).map_err(|e| e.to_string())?;
    let mut providers = backend.providers(&device);
    providers.push(CPUExecutionProvider::default().with_arena_allocator(false).build());
    Session::builder()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_intra_threads(threads.max(1)))
        .and_then(|b| b.with_inter_threads(1))
        .and_then(|b| b.with_memory_pattern(false))
        .and_then(|b| b.with_parallel_execution(false))
        .and_then(|b| b.with_execution_providers(providers))
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
