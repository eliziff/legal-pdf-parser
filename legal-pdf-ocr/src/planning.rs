use crate::indra::{OcrOptions, OcrProvider, PreparedOcrProvider};
use legal_pdf_core::OrtBackend;
use legal_pdf_core::{provider_asset_sha256, Error, Result};
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};

#[derive(Debug, Clone, Serialize)]
pub struct RuntimePlan {
    pub backend: OrtBackend,
    pub device: String,
    pub workers: usize,
    pub threads: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub width_bucket: usize,
}

pub(crate) fn prepare(
    options: &OcrOptions,
    root: &Path,
    runtime: &Path,
) -> Result<PreparedOcrProvider> {
    static PLANS: OnceLock<Mutex<HashMap<String, RuntimePlan>>> = OnceLock::new();
    let budget = cpu_budget(std::thread::available_parallelism().map_or(1, |n| n.get()));
    if let Some(backend) = options.backend {
        return OcrProvider::prepare_plan(
            options,
            root,
            runtime.to_path_buf(),
            plan(options, backend, options.device.as_deref(), budget)?,
        );
    }
    let mut selection_options = options.clone();
    selection_options.expected_identity = None;
    let key = format!(
        "{}:{}:{selection_options:?}:{budget}",
        provider_asset_sha256(&root.join("manifest.json"))?,
        provider_asset_sha256(runtime)?
    );
    let mut cache = PLANS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| Error::Message("Indra selection lock poisoned".into()))?;
    if let Some(plan) = cache.get(&key) {
        return OcrProvider::prepare_plan(options, root, runtime.to_path_buf(), plan.clone());
    }
    let (backend, device) = preferred_backend(options);
    let mut selected = plan(options, backend, device.as_deref(), budget)?;
    let mut prepared =
        OcrProvider::prepare_plan(options, root, runtime.to_path_buf(), selected.clone())?;
    if backend != OrtBackend::Cpu {
        // Verify the provider once and keep its sessions. Startup never runs a synthetic OCR sweep.
        match OcrProvider::open_prepared(&selection_options, &prepared) {
            Ok(engine) => {
                prepared.ready = Some(Box::new(engine));
            }
            Err(error) => {
                if options.device.is_some() {
                    return Err(error);
                }
                selected = plan(options, OrtBackend::Cpu, None, budget)?;
                prepared =
                    OcrProvider::prepare_plan(options, root, runtime.to_path_buf(), selected.clone())?;
            }
        }
    }
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.insert(key, selected);
    Ok(prepared)
}

fn cpu_budget(available: usize) -> usize {
    available.saturating_sub(1).clamp(1, 16)
}

fn plan(
    options: &OcrOptions,
    backend: OrtBackend,
    device: Option<&str>,
    budget: usize,
) -> Result<RuntimePlan> {
    let gpu = matches!(
        backend,
        OrtBackend::Cuda | OrtBackend::TensorRt | OrtBackend::DirectMl
    );
    let threads = options.threads.max(1);
    let workers = if options.workers > 0 {
        options.workers
    } else {
        let wanted = if gpu {
            if budget >= 8 {
                4
            } else if budget >= 4 {
                2
            } else {
                1
            }
        } else {
            budget.div_ceil(2)
        };
        wanted.min((budget / threads).max(1))
    };
    let layout_workers = if options.layout_workers > 0 {
        options.layout_workers
    } else if gpu {
        2.min(budget.saturating_sub(workers * threads).max(1))
    } else {
        4.min(budget.saturating_sub(workers * threads).max(1))
    };
    Ok(RuntimePlan {
        backend,
        device: backend.normalized_device(device)?,
        workers,
        threads,
        layout_workers,
        batch_size: if options.batch_size > 0 {
            options.batch_size
        } else if gpu {
            32
        } else {
            16
        },
        width_bucket: if options.width_bucket > 0 {
            options.width_bucket
        } else if gpu {
            128
        } else {
            24
        },
    })
}

fn preferred_backend(options: &OcrOptions) -> (OrtBackend, Option<String>) {
    if options.cpu_arena {
        if CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            return (OrtBackend::Cuda, options.device.clone());
        }
        if DirectMLExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            if let Some(device) = &options.device {
                return (OrtBackend::DirectMl, Some(device.clone()));
            }
            if let Some(device) = preferred_dml_adapter() {
                return (OrtBackend::DirectMl, Some(device.to_string()));
            }
        }
    }
    (OrtBackend::Cpu, options.device.clone())
}

#[cfg(target_os = "windows")]
fn preferred_dml_adapter() -> Option<i32> {
    extern "C" {
        fn legalpdf_preferred_dml_adapter() -> i32;
    }
    static DEVICE: OnceLock<Option<i32>> = OnceLock::new();
    *DEVICE.get_or_init(|| {
        // Windows ranks the adapters; the shim only checks support and maps the DirectML ID.
        let device = unsafe { legalpdf_preferred_dml_adapter() };
        (device >= 0).then_some(device)
    })
}

#[cfg(not(target_os = "windows"))]
fn preferred_dml_adapter() -> Option<i32> {
    None
}
