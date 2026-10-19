use crate::indra::{IndraBackend, IndraOcr, IndraOptions};
use crate::model_pack::ModelPack;
use image::{GrayImage, Luma};
use legal_pdf_core::{provider_asset_sha256, Error, Result};
use ort::execution_providers::{
    CUDAExecutionProvider, DirectMLExecutionProvider, ExecutionProvider,
};
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Serialize)]
pub struct RuntimePlan {
    pub backend: IndraBackend,
    pub workers: usize,
    pub threads: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub width_bucket: usize,
}

pub(crate) fn select(options: &IndraOptions, root: &Path, runtime: &Path) -> Result<RuntimePlan> {
    static PLANS: OnceLock<Mutex<HashMap<String, RuntimePlan>>> = OnceLock::new();
    let budget = cpu_budget(std::thread::available_parallelism().map_or(1, |n| n.get()));
    if let Some(backend) = options.backend {
        if [
            options.workers,
            options.threads,
            options.layout_workers,
            options.batch_size,
            options.width_bucket,
        ]
        .iter()
        .all(|&value| value > 0)
        {
            return Ok(candidates(options, backend, budget).remove(0));
        }
    }
    let _ = ModelPack::open(
        root,
        options.tier,
        options.backend.unwrap_or(IndraBackend::Cpu),
    )?;
    let mut calibration_options = options.clone();
    calibration_options.expected_identity = None;
    let key = format!(
        "{}:{}:{calibration_options:?}:{budget}",
        provider_asset_sha256(&root.join("manifest.json"))?,
        provider_asset_sha256(runtime)?
    );
    let mut cache = PLANS
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| Error::Message("Indra calibration lock poisoned".into()))?;
    if let Some(plan) = cache.get(&key) {
        return Ok(plan.clone());
    }
    let mut backends = vec![IndraBackend::Cpu];
    if let Some(backend) = options.backend {
        backends = vec![backend];
    } else if options.cpu_arena {
        if CUDAExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            backends.push(IndraBackend::Cuda);
        }
        if DirectMLExecutionProvider::default()
            .is_available()
            .unwrap_or(false)
        {
            backends.push(IndraBackend::DirectMl);
        }
    }
    let pages = calibration_pages();
    let mut best: Option<(f64, RuntimePlan)> = None;
    let mut failure = None;
    for backend in backends {
        let started = Instant::now();
        let mut measured = false;
        for plan in candidates(options, backend, budget) {
            // Bound the search; a driver's initial shader compilation can itself exceed this window.
            if measured && started.elapsed() > Duration::from_secs(5) {
                break;
            }
            let result = (|| -> Result<f64> {
                let prepared =
                    IndraOcr::prepare_plan(options, root, runtime.to_path_buf(), plan.clone())?;
                let mut probe_options = options.clone();
                probe_options.expected_identity = None;
                let mut engine = IndraOcr::from_prepared(&probe_options, prepared)?;
                let _ = engine.recognize_gray_images_profile(&pages)?;
                let begin = Instant::now();
                let _ = engine.recognize_gray_images_profile(&pages)?;
                Ok(begin.elapsed().as_secs_f64())
            })();
            match result {
                Ok(seconds) if seconds.is_finite() && seconds > 0.0 => {
                    measured = true;
                    // Keep the smaller plan when the gain is within measurement noise.
                    if best.as_ref().is_none_or(|(old, _)| seconds < old * 0.95) {
                        best = Some((seconds, plan));
                    }
                }
                Ok(_) => {}
                Err(error) => {
                    failure = Some(error);
                    break;
                }
            }
        }
    }
    let plan = best.map(|(_, plan)| plan).ok_or_else(|| {
        failure.unwrap_or_else(|| Error::Message("no usable Indra execution provider".into()))
    })?;
    if cache.len() >= 32 {
        cache.clear();
    }
    cache.insert(key, plan.clone());
    Ok(plan)
}

fn cpu_budget(available: usize) -> usize {
    available.saturating_sub(1).clamp(1, 16)
}

fn candidates(options: &IndraOptions, backend: IndraBackend, budget: usize) -> Vec<RuntimePlan> {
    let gpu = backend != IndraBackend::Cpu;
    let mut plans = Vec::new();
    for (workers, layouts, batch) in if gpu {
        vec![(1, 1, 32), (2, 1, 32), (2, 2, 64), (4, 2, 32)]
    } else {
        vec![
            (budget.div_ceil(2), (budget / 2).max(1), 16),
            (1, 1, 16),
            ((budget / 3).max(1), (budget / 3).max(1), 32),
        ]
    } {
        let threads = options.threads.max(1);
        let workers = if options.workers > 0 {
            options.workers
        } else {
            workers.min((budget / threads).max(1))
        };
        let layout_workers = if options.layout_workers > 0 {
            options.layout_workers
        } else {
            layouts.min(4).min(if gpu {
                budget
            } else {
                budget.saturating_sub(workers * threads).max(1)
            })
        };
        let plan = RuntimePlan {
            backend,
            workers,
            threads,
            layout_workers,
            batch_size: if options.batch_size > 0 {
                options.batch_size
            } else {
                batch
            },
            width_bucket: if options.width_bucket > 0 {
                options.width_bucket
            } else if gpu {
                128
            } else {
                24
            },
        };
        if !plans.iter().any(|old: &RuntimePlan| {
            old.workers == plan.workers
                && old.threads == plan.threads
                && old.layout_workers == plan.layout_workers
                && old.batch_size == plan.batch_size
        }) {
            plans.push(plan);
        }
    }
    plans
}

// Invented glyph-like strokes at mixed widths exercise the complete detection/crop/inference path.
fn calibration_pages() -> Vec<GrayImage> {
    (0..2)
        .map(|page| {
            let mut image = GrayImage::from_pixel(960, 1280, Luma([255]));
            for line in 0..42 {
                for glyph in 0..(if line % 9 == 0 {
                    5 + (line + page) % 12
                } else {
                    24 + (line * 13 + page * 7) % 17
                }) {
                    let x = 45 + glyph * 10;
                    let y = 32 + line * 29;
                    for dy in 0..14 {
                        for dx in 0..7 {
                            if dx == 0 || dy == 0 || dy == 7 || (dx == 6 && (glyph + line) % 3 == 0)
                            {
                                image.put_pixel(x + dx, y + dy, Luma([30]));
                            }
                        }
                    }
                }
            }
            image
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn automatic_plans_respect_cpu_budget_and_explicit_overrides() {
        for available in [1, 2, 4, 8, 12, 64] {
            let budget = cpu_budget(available);
            for backend in [IndraBackend::Cpu, IndraBackend::DirectMl] {
                for plan in candidates(&IndraOptions::default(), backend, budget) {
                    assert!(plan.workers >= 1 && plan.layout_workers >= 1);
                    assert!(plan.workers * plan.threads <= budget);
                    if backend == IndraBackend::Cpu {
                        assert!(plan.workers + plan.layout_workers <= budget.max(2));
                    }
                }
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
        for plan in candidates(&options, IndraBackend::Cpu, 8) {
            assert_eq!(
                (
                    plan.workers,
                    plan.threads,
                    plan.layout_workers,
                    plan.batch_size,
                    plan.width_bucket
                ),
                (3, 2, 2, 7, 48)
            );
        }
    }
}
