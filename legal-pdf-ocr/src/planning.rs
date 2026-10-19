use crate::indra::{IndraBackend, IndraOcr, IndraOptions, PreparedIndra};
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
    pub backend: IndraBackend,
    pub device: String,
    pub workers: usize,
    pub threads: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub width_bucket: usize,
}

pub(crate) fn prepare(
    options: &IndraOptions,
    root: &Path,
    runtime: &Path,
) -> Result<PreparedIndra> {
    static PLANS: OnceLock<Mutex<HashMap<String, RuntimePlan>>> = OnceLock::new();
    let budget = cpu_budget(std::thread::available_parallelism().map_or(1, |n| n.get()));
    if let Some(backend) = options.backend {
        return IndraOcr::prepare_plan(
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
        return IndraOcr::prepare_plan(options, root, runtime.to_path_buf(), plan.clone());
    }
    let (backend, device) = preferred_backend(options);
    let mut selected = plan(options, backend, device.as_deref(), budget)?;
    let mut prepared =
        IndraOcr::prepare_plan(options, root, runtime.to_path_buf(), selected.clone())?;
    if backend != IndraBackend::Cpu {
        // Verify the provider once and keep its sessions. Startup never runs a synthetic OCR sweep.
        match IndraOcr::open_prepared(&selection_options, &prepared) {
            Ok(engine) => {
                prepared.ready = Some(Box::new(engine));
            }
            Err(error) => {
                if options.device.is_some() {
                    return Err(error);
                }
                selected = plan(options, IndraBackend::Cpu, None, budget)?;
                prepared =
                    IndraOcr::prepare_plan(options, root, runtime.to_path_buf(), selected.clone())?;
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
    options: &IndraOptions,
    backend: IndraBackend,
    device: Option<&str>,
    budget: usize,
) -> Result<RuntimePlan> {
    let gpu = matches!(
        backend,
        IndraBackend::Cuda | IndraBackend::TensorRt | IndraBackend::DirectMl
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

fn preferred_backend(options: &IndraOptions) -> (IndraBackend, Option<String>) {
    if options.cpu_arena {
        if CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            return (IndraBackend::Cuda, options.device.clone());
        }
        if DirectMLExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            if let Some(device) = &options.device {
                return (IndraBackend::DirectMl, Some(device.clone()));
            }
            if let Some(device) = preferred_dml_adapter() {
                return (IndraBackend::DirectMl, Some(device.to_string()));
            }
        }
    }
    (IndraBackend::Cpu, options.device.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn automatic_plans_respect_cpu_budget_and_explicit_overrides() {
        for available in [1, 2, 4, 8, 12, 64] {
            let budget = cpu_budget(available);
            for backend in [IndraBackend::Cpu, IndraBackend::DirectMl] {
                let p = plan(&IndraOptions::default(), backend, None, budget).unwrap();
                assert!(p.workers >= 1 && p.layout_workers >= 1);
                assert!(p.workers * p.threads <= budget);
                assert!(p.workers * p.threads + p.layout_workers <= budget.max(2));
            }
        }
        let options = IndraOptions {
            workers: 3,
            threads: 2,
            layout_workers: 2,
            batch_size: 7,
            width_bucket: 48,
            ..Default::default()
        };
        let p = plan(&options, IndraBackend::DirectMl, Some("2"), 8).unwrap();
        assert_eq!(
            (
                p.workers,
                p.threads,
                p.layout_workers,
                p.batch_size,
                p.width_bucket
            ),
            (3, 2, 2, 7, 48)
        );
        assert_eq!(p.device, "2");
    }
}
