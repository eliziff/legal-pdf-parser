//! PageIndex Flash's page and document layout statistics.
//!
//! Ported from VectifyAI/PageIndex `pageindex/flash/stats/aggregates.py` (`weighted_percentile`,
//! `compute_page_stats`, `compute_doc_stats`) at 6d23caf416858f2ca136840305d1f479a86f6ef7.
//! Copyright (c) 2025 Vectify AI; MIT License, reproduced in `mod.rs`.

use super::model::FlashLine;
use std::collections::HashMap;

/// Weighted percentile over `(value, weight)` samples: NaN when empty; a sample landing exactly
/// on the target averages with its predecessor, an overshoot returns the current value.
pub(super) fn weighted_percentile(mut samples: Vec<(f64, f64)>, percentile: f64) -> f64 {
    if samples.is_empty() || !(0.0..=100.0).contains(&percentile) {
        return f64::NAN;
    }
    samples.sort_by(|left, right| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    let total: f64 = samples.iter().map(|sample| sample.1).sum();
    let target = total * percentile / 100.0;
    let mut accumulated = 0.0;
    let mut previous: Option<f64> = None;
    for (value, weight) in samples {
        if accumulated == target {
            return previous.map_or(value, |previous| (previous + value) / 2.0);
        }
        previous = Some(value);
        accumulated += weight;
        if accumulated > target {
            return value;
        }
    }
    previous.unwrap_or(f64::NAN)
}

#[derive(Debug, Clone)]
pub(super) struct PageStats {
    pub(super) line_count: usize,
    pub(super) total_weight: f64,
    /// Median vertical step between a line and the line above it.
    pub(super) line_gap: f64,
    pub(super) line_width: f64,
    pub(super) fill: f64,
    pub(super) font_size: f64,
    pub(super) dominant_font: String,
    pub(super) dominant_style: String,
}

fn ordered_add(
    order: &mut Vec<String>,
    weights: &mut HashMap<String, f64>,
    key: String,
    weight: f64,
) {
    match weights.get_mut(&key) {
        Some(total) => *total += weight,
        None => {
            weights.insert(key.clone(), weight);
            order.push(key);
        }
    }
}

fn dominant(order: &[String], weights: &HashMap<String, f64>) -> String {
    let mut best = String::new();
    let mut best_weight = 0.0;
    for key in order {
        if weights[key] > best_weight {
            best_weight = weights[key];
            best.clone_from(key);
        }
    }
    best
}

pub(super) fn page_stats(lines: &[FlashLine], page_width: f64) -> PageStats {
    let mut gaps = Vec::new();
    let mut widths = Vec::new();
    let mut fills = Vec::new();
    let mut sizes = Vec::new();
    let (mut font_order, mut fonts) = (Vec::new(), HashMap::new());
    let (mut style_order, mut styles) = (Vec::new(), HashMap::new());
    let mut total = 0.0;
    let mut valid = 0;
    let bucket = page_width / 20.0;
    let mut buckets: Vec<Option<usize>> = vec![None; 21];
    for (index, line) in lines.iter().enumerate() {
        if line.skew > 1.0 {
            continue;
        }
        valid += 1;
        for span in &line.spans {
            let weight = span.stats.info_weight();
            let weight = weight * weight * span.rect.height();
            ordered_add(&mut font_order, &mut fonts, span.font_name.clone(), weight);
            ordered_add(&mut style_order, &mut styles, span.style_key(), weight);
        }
        let line_weight = line.stats.info_weight();
        total += line_weight;
        let sample = line_weight * line.size;
        sizes.push((line.size, sample));
        widths.push((line.rect.width(), sample));
        fills.push((line.fill, line.rect.area()));
        let (left, right) = (line.rect.left, line.rect.right);
        if !(left < f64::INFINITY && right > f64::NEG_INFINITY) || bucket <= 0.0 {
            continue;
        }
        let first = if left == f64::NEG_INFINITY {
            0
        } else {
            ((left / bucket) as i64).max(0) as usize
        };
        let last = if right == f64::INFINITY {
            20
        } else {
            ((right / bucket).ceil() as i64).min(20).max(0) as usize
        };
        let mut best_gap = f64::INFINITY;
        let mut best_prior = None;
        for slot in first..last {
            let prior = buckets[slot].replace(index);
            let Some(prior) = prior else {
                continue;
            };
            let gap = lines[prior].rect.bottom.max(line.rect.top) - line.rect.bottom;
            if gap < best_gap {
                best_gap = gap;
                best_prior = Some(prior);
            }
        }
        if let Some(prior) = best_prior.filter(|_| best_gap < f64::INFINITY) {
            gaps.push((best_gap, lines[prior].stats.info_weight() * line_weight));
        }
    }
    PageStats {
        line_count: valid,
        total_weight: total,
        line_gap: weighted_percentile(gaps, 50.0),
        line_width: weighted_percentile(widths, 50.0),
        fill: weighted_percentile(fills, 50.0),
        font_size: weighted_percentile(sizes, 50.0),
        dominant_font: dominant(&font_order, &fonts),
        dominant_style: dominant(&style_order, &styles),
    }
}

#[derive(Debug, Clone)]
pub(super) struct DocStats {
    pub(super) line_width: f64,
    /// 80th percentile of per-page median span coverage.
    pub(super) upper_fill: f64,
    pub(super) body_font_size: f64,
}

pub(super) fn doc_stats(pages: &[PageStats]) -> DocStats {
    let mut widths = Vec::new();
    let mut fills = Vec::new();
    let mut sizes = Vec::new();
    for stats in pages {
        if stats.line_count == 0 || stats.total_weight <= 0.0 {
            continue;
        }
        let per_page = 100f64.min(stats.total_weight / stats.line_count as f64);
        widths.push((stats.line_width, per_page));
        fills.push((stats.fill, per_page));
        sizes.push((stats.font_size, per_page));
    }
    DocStats {
        line_width: weighted_percentile(widths, 50.0),
        upper_fill: weighted_percentile(fills, 80.0),
        body_font_size: weighted_percentile(sizes, 50.0),
    }
}
