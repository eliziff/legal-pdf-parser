//! The ONNX Runtime as DOC's stages run in it: the runtime found, a graph's session, its inputs and outputs.

use ort::execution_providers::CPUExecutionProvider;
use ort::session::{builder::GraphOptimizationLevel, run_options::RunOptions, Session};
use std::path::Path;

/// A graph's input of `shape` over `data`, its error as text.
macro_rules! tensor {
    ($shape:expr, $data:expr) => {
        ort::value::TensorRef::from_array_view(($shape, $data)).map_err(|e| e.to_string())?
    };
}
/// A graph's output `name`, its values, its error as text.
macro_rules! output {
    ($outputs:expr, $name:expr) => {
        $outputs[$name]
            .try_extract_tensor::<f32>()
            .map_err(|e| e.to_string())?
            .1
    };
}

/// Whether the ONNX Runtime CPU memory arena is enabled (`LEGALPDF_DOC_ARENA=0` disables it).
pub fn arena_enabled() -> bool {
    std::env::var("LEGALPDF_DOC_ARENA").map_or(true, |v| v != "0")
}

/// Run options that shrink the CPU memory arena after each run so giant documents do not hold multi-gigabyte arenas across pooled sessions.
pub fn shrink_arena_options() -> Option<RunOptions> {
    if !arena_enabled() {
        return None;
    }
    let mut opts = RunOptions::new().ok()?;
    let _ = opts.add_config_entry("memory.enable_memory_arena_shrinkage", "cpu:0");
    Some(opts)
}

/// The ONNX Runtime at `runtime`, else the one LEGALPDF_ONNX_RUNTIME or LEGALPDF_ONNXRUNTIME names, initialized; `missing`
/// the error where none is named.
pub fn init_runtime(
    runtime: Option<&Path>,
    missing: &str,
) -> std::result::Result<(), String> {
    let named = ["LEGALPDF_ONNX_RUNTIME", "LEGALPDF_ONNXRUNTIME"]
        .iter()
        .find_map(|name| std::env::var_os(name).filter(|v| !v.is_empty()));
    let runtime = runtime
        .map(Path::as_os_str)
        .or(named.as_deref())
        .ok_or(missing)?;
    legal_pdf_core::init_ort_runtime(Path::new(runtime), None).map_err(|e| e.to_string())?;
    Ok(())
}

/// The graph at `path` in a session of `threads` threads, its error named `name`. The CPU memory arena is enabled by
/// default (`LEGALPDF_DOC_ARENA=0` disables it) and shrunk after each run (`run_options`) so normal pages reuse buffers
/// without holding giant document arenas across pooled sessions.
/// On the backend LEGALPDF_DOC_BACKEND names ("directml", "cuda", as the recognizer's backends are named; LEGALPDF_DOC_DEVICE
/// its device), else the CPU.
pub fn session(
    path: &Path,
    name: impl std::fmt::Display,
    threads: usize,
) -> std::result::Result<Session, String> {
    let backend: legal_pdf_core::OrtBackend = std::env::var("LEGALPDF_DOC_BACKEND")
        .ok()
        .map(|value| {
            serde_json::from_value(serde_json::Value::String(value))
                .map_err(|e| format!("LEGALPDF_DOC_BACKEND: {e}"))
        })
        .transpose()?
        .unwrap_or_default();
    let device = backend
        .normalized_device(std::env::var("LEGALPDF_DOC_DEVICE").ok().as_deref())
        .map_err(|e| e.to_string())?;
    let mut providers = backend.providers(&device);
    let arena = arena_enabled();
    providers.push(
        CPUExecutionProvider::default()
            .with_arena_allocator(arena)
            .build(),
    );
    let cache_path = (backend == legal_pdf_core::OrtBackend::Cpu)
        .then(|| {
            let meta = std::fs::metadata(path).ok()?;
            let mtime = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            let stem = path.file_stem()?.to_string_lossy();
            let dir = std::env::temp_dir().join("legaldoc-ort-cache");
            let _ = std::fs::create_dir_all(&dir);
            Some(dir.join(format!("{stem}_{}_{}_v1.opt.onnx", meta.len(), mtime)))
        })
        .flatten();
    let use_cached = cache_path.as_ref().is_some_and(|p| p.is_file());
    let (load_path, opt_level, write_tmp) = if use_cached {
        (
            cache_path.as_deref().unwrap(),
            GraphOptimizationLevel::Disable,
            None,
        )
    } else {
        let tmp = cache_path
            .as_ref()
            .map(|p| p.with_extension(format!("tmp.{}.onnx", std::process::id())));
        (path, GraphOptimizationLevel::Level3, tmp)
    };
    let mut builder = Session::builder()
        .and_then(|b| b.with_optimization_level(opt_level))
        .and_then(|b| b.with_intra_threads(threads.max(1)))
        .and_then(|b| b.with_inter_threads(1))
        .and_then(|b| b.with_config_entry("session.set_denormal_as_zero", "1"))
        .and_then(|b| b.with_memory_pattern(arena))
        .and_then(|b| b.with_parallel_execution(false))
        .map_err(|e| format!("{name}: {e}"))?;
    if threads <= 1 {
        builder = builder
            .with_config_entry("session.intra_op.allow_spinning", "0")
            .and_then(|b| b.with_config_entry("session.inter_op.allow_spinning", "0"))
            .map_err(|e| format!("{name}: {e}"))?;
    }
    if let Some(ref tmp_path) = write_tmp {
        builder = builder
            .with_optimized_model_path(tmp_path)
            .map_err(|e| format!("{name}: {e}"))?;
    }
    let session = builder
        .with_execution_providers(providers)
        .and_then(|b| b.commit_from_file(load_path))
        .map_err(|e| format!("{name}: {e}"))?;
    if let (Some(tmp_path), Some(final_path)) = (write_tmp, cache_path) {
        if tmp_path.is_file() {
            let _ = std::fs::rename(&tmp_path, &final_path);
        }
    }
    Ok(session)
}

/// Each row of a `width`-wide matrix made a probability distribution.
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
