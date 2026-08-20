use crate::error::{Error, Result};
use ort::environment::GlobalThreadPoolOptions;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

struct Runtime {
    path: PathBuf,
    global_threads: Option<usize>,
}

pub fn init(path: &Path, global_threads: Option<usize>) -> Result<Option<usize>> {
    static RUNTIME: OnceLock<Runtime> = OnceLock::new();
    if let Some(selected) = RUNTIME.get() {
        return (selected.path == path)
            .then_some(selected.global_threads)
            .ok_or_else(|| {
                Error::Message(format!(
                    "ONNX Runtime is already initialized from {}; refusing {}",
                    selected.path.display(),
                    path.display()
                ))
            });
    }
    let environment = ort::init_from(path.to_string_lossy());
    let environment = if let Some(threads) = global_threads {
        if threads == 0 {
            return Err(Error::Message(
                "ONNX Runtime global thread count must be positive".to_owned(),
            ));
        }
        let pool = GlobalThreadPoolOptions::default()
            .with_intra_threads(threads)
            .and_then(|pool| pool.with_inter_threads(1))
            .map_err(|error| Error::Message(format!("global ORT thread pool failed: {error}")))?;
        environment.with_global_thread_pool(pool)
    } else {
        environment
    };
    environment
        .commit()
        .map_err(|error| Error::Message(format!("ONNX Runtime initialization failed: {error}")))?;
    let _ = RUNTIME.set(Runtime {
        path: path.to_path_buf(),
        global_threads,
    });
    Ok(global_threads)
}
