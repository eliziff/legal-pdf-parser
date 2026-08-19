use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

const OTSU_MIN_THRESHOLD: usize = 64;
const OTSU_MAX_THRESHOLD: usize = 200;
const DARK_PAGE_SHARE: f64 = 0.5;
const MIN_RULE_WIDTH_RATIO: f64 = 0.08;
const MAX_RULE_THICKNESS_RATIO: f64 = 0.01;
const MIN_VERTICAL_RULE_LENGTH_RATIO: f64 = 0.05;
const MAX_ROW_TRANSITIONS: usize = 40;
const INK_GUARD_RATIO: f64 = 0.002;
const INK_WINDOW_RATIO: f64 = 0.006;
const BLOCK_INK_WINDOW_RATIO: f64 = 0.18;
const MAX_RECORDED_RULES: usize = 40;
const MAX_RECORDED_VERTICAL_RULES: usize = 20;

const SEPARATOR_MIN_Y_RATIO: f64 = 0.30;
const SEPARATOR_MAX_Y_RATIO: f64 = 0.97;
const SEPARATOR_MIN_X0_RATIO: f64 = 0.015;
const SEPARATOR_MAX_X0_RATIO: f64 = 0.55;
const SEPARATOR_MAX_X1_RATIO: f64 = 0.985;
const SEPARATOR_MIN_WIDTH_RATIO: f64 = 0.08;
const SEPARATOR_MAX_WIDTH_RATIO: f64 = 0.95;
const SEPARATOR_MIN_DARKNESS: f64 = 0.55;
const SEPARATOR_MAX_THICKNESS_RATIO: f64 = 0.006;
const SEPARATOR_MAX_ABOVE_INK: f64 = 0.05;
const SEPARATOR_MAX_BELOW_INK: f64 = 0.15;
const SEPARATOR_MIN_BLOCK_INK: f64 = 0.015;
const FULL_RULE_MIN_WIDTH_RATIO: f64 = 0.6;
const STACK_NEIGHBOR_Y_RATIO: f64 = 0.035;
const STACK_OVERLAP_SHARE: f64 = 0.5;
const TWO_COLUMN_Y_DELTA_RATIO: f64 = 0.012;
const VERTICAL_CROSS_X_PAD_RATIO: f64 = 0.01;

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct RuleRecord {
    pub y0_ratio: f64,
    pub y1_ratio: f64,
    pub y_center_ratio: f64,
    pub x0_ratio: f64,
    pub x1_ratio: f64,
    pub width_ratio: f64,
    pub thickness_px: usize,
    pub thickness_ratio: f64,
    pub darkness: f64,
    pub above_ink: f64,
    pub below_ink: f64,
    pub above_block_ink: f64,
    pub below_block_ink: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct VerticalRuleRecord {
    pub x_center_ratio: f64,
    pub y0_ratio: f64,
    pub y1_ratio: f64,
    pub length_ratio: f64,
    pub thickness_px: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub(crate) struct ScanRecord {
    pub status: &'static str,
    pub page_size: Option<[usize; 2]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dark_share: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rule_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertical_rule_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rules: Option<Vec<RuleRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vertical_rules: Option<Vec<VerticalRuleRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub separators: Option<Vec<RuleRecord>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub separator_status: Option<&'static str>,
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round_ties_even() / 10_000.0
}

fn otsu_threshold(gray: &[u8]) -> usize {
    let mut histogram = [0_f64; 256];
    for &value in gray {
        histogram[usize::from(value)] += 1.0;
    }
    let total = histogram.iter().sum::<f64>();
    if total <= 0.0 {
        return 128;
    }
    let mut omega = 0.0;
    let mut mu = 0.0;
    let total_mu = histogram
        .iter()
        .enumerate()
        .map(|(value, count)| value as f64 * count)
        .sum::<f64>();
    let mut best_index = 0;
    let mut best_sigma = 0.0;
    for (value, count) in histogram.iter().enumerate() {
        omega += count;
        mu += value as f64 * count;
        let denominator = omega * (total - omega);
        let sigma = if denominator > 0.0 {
            (total_mu * omega - mu * total).powi(2) / denominator
        } else {
            0.0
        };
        if sigma > best_sigma {
            best_sigma = sigma;
            best_index = value;
        }
    }
    best_index
}

fn index(width: usize, x: usize, y: usize) -> usize {
    y * width + x
}

fn horizontal_dilation(dark: &[bool], width: usize, height: usize) -> Vec<bool> {
    let mut result = dark.to_vec();
    for y in 0..height {
        for x in 0..width {
            result[index(width, x, y)] = dark[index(width, x, y)]
                || (y > 0 && dark[index(width, x, y - 1)])
                || (y + 1 < height && dark[index(width, x, y + 1)]);
        }
    }
    result
}

fn candidate_rows(mask: &[bool], width: usize, height: usize, minimum: usize) -> Vec<usize> {
    (0..height)
        .filter(|&y| {
            let mut ink = 0;
            let mut transitions = 0;
            for x in 0..width {
                let value = mask[index(width, x, y)];
                ink += usize::from(value);
                if x > 0 && value != mask[index(width, x - 1, y)] {
                    transitions += 1;
                }
            }
            ink >= minimum && transitions <= MAX_ROW_TRANSITIONS
        })
        .collect()
}

fn longest_row_run(mask: &[bool], width: usize, y: usize) -> (usize, usize, usize) {
    let mut best_length = 0;
    let mut best_start = 0;
    let mut run_length = 0;
    let mut run_start = 0;
    for x in 0..width {
        if mask[index(width, x, y)] {
            if run_length == 0 {
                run_start = x;
            }
            run_length += 1;
            if run_length > best_length {
                best_length = run_length;
                best_start = run_start;
            }
        } else {
            run_length = 0;
        }
    }
    (best_length, best_start, best_start + best_length)
}

fn groups(indices: impl IntoIterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut result: Vec<(usize, usize)> = Vec::new();
    for value in indices {
        if result
            .last()
            .is_some_and(|(_, end)| value == end.saturating_add(1))
        {
            result.last_mut().expect("group").1 = value;
        } else {
            result.push((value, value));
        }
    }
    result
}

fn median_usize(values: impl IntoIterator<Item = usize>) -> usize {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_unstable();
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2
    } else {
        values[middle]
    }
}

#[allow(clippy::too_many_arguments)]
fn ink_share(
    dark: &[bool],
    width: usize,
    height: usize,
    y0: isize,
    y1: isize,
    x0: usize,
    x1: usize,
    empty: f64,
) -> f64 {
    let y0 = y0.max(0) as usize;
    let y1 = y1.min(height as isize).max(0) as usize;
    if y1 <= y0 || x1 <= x0 {
        return empty;
    }
    let mut count = 0;
    let mut total = 0;
    for y in y0..y1 {
        for x in x0..x1 {
            count += usize::from(dark[index(width, x, y)]);
            total += 1;
        }
    }
    count as f64 / total as f64
}

fn horizontal_rule_records(dark: &[bool], width: usize, height: usize) -> Vec<RuleRecord> {
    let dilated = horizontal_dilation(dark, width, height);
    let minimum = 8.max((width as f64 * MIN_RULE_WIDTH_RATIO) as usize);
    let mut metrics = BTreeMap::new();
    for row in candidate_rows(&dilated, width, height, minimum) {
        let run = longest_row_run(&dilated, width, row);
        if run.0 >= minimum {
            metrics.insert(row, run);
        }
    }
    let maximum_thickness = 4.max((height as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    let guard = 2.max((height as f64 * INK_GUARD_RATIO) as usize);
    let ink_window = 4.max((height as f64 * INK_WINDOW_RATIO) as usize);
    let block_window = (ink_window * 2).max((height as f64 * BLOCK_INK_WINDOW_RATIO) as usize);
    let mut records = Vec::new();
    for (band_y0, band_y1) in groups(metrics.keys().copied()) {
        let thickness = band_y1 - band_y0 + 1;
        if thickness > maximum_thickness + 2 {
            continue;
        }
        let x0 = median_usize((band_y0..=band_y1).map(|row| metrics[&row].1));
        let x1 = median_usize((band_y0..=band_y1).map(|row| metrics[&row].2));
        if x1 <= x0 {
            continue;
        }
        let darkness = (band_y0..=band_y1)
            .map(|y| {
                (x0..x1).filter(|&x| dark[index(width, x, y)]).count() as f64 / (x1 - x0) as f64
            })
            .fold(0.0, f64::max);
        let above_end = band_y0 as isize - guard as isize;
        let below_start = band_y1 + 1 + guard;
        let real_thickness = thickness.saturating_sub(2).max(1);
        records.push(RuleRecord {
            y0_ratio: round4(band_y0 as f64 / height as f64),
            y1_ratio: round4((band_y1 + 1) as f64 / height as f64),
            y_center_ratio: round4((band_y0 + band_y1 + 1) as f64 / 2.0 / height as f64),
            x0_ratio: round4(x0 as f64 / width as f64),
            x1_ratio: round4(x1 as f64 / width as f64),
            width_ratio: round4((x1 - x0) as f64 / width as f64),
            thickness_px: real_thickness,
            thickness_ratio: round4(real_thickness as f64 / height as f64),
            darkness: round4(darkness),
            above_ink: round4(ink_share(
                dark,
                width,
                height,
                above_end - ink_window as isize,
                above_end,
                x0,
                x1,
                1.0,
            )),
            below_ink: round4(ink_share(
                dark,
                width,
                height,
                below_start as isize,
                (below_start + ink_window) as isize,
                x0,
                x1,
                1.0,
            )),
            above_block_ink: round4(ink_share(
                dark,
                width,
                height,
                above_end - block_window as isize,
                above_end,
                x0,
                x1,
                0.0,
            )),
            below_block_ink: round4(ink_share(
                dark,
                width,
                height,
                below_start as isize,
                (below_start + block_window) as isize,
                x0,
                x1,
                0.0,
            )),
            kind: None,
        });
    }
    records.sort_by(|left, right| right.width_ratio.total_cmp(&left.width_ratio));
    records
}

fn vertical_rule_records(dark: &[bool], width: usize, height: usize) -> Vec<VerticalRuleRecord> {
    let minimum = 12.max((height as f64 * MIN_VERTICAL_RULE_LENGTH_RATIO) as usize);
    let mut metrics = BTreeMap::new();
    for x in 0..width {
        let mut ink = 0;
        let mut transitions = 0;
        let mut best_length = 0;
        let mut best_start = 0;
        let mut run_length = 0;
        let mut run_start = 0;
        let mut prior = false;
        for y in 0..height {
            let value = dark[index(width, x, y)]
                || (x > 0 && dark[index(width, x - 1, y)])
                || (x + 1 < width && dark[index(width, x + 1, y)]);
            ink += usize::from(value);
            if y > 0 && value != prior {
                transitions += 1;
            }
            if value {
                if run_length == 0 {
                    run_start = y;
                }
                run_length += 1;
                if run_length > best_length {
                    best_length = run_length;
                    best_start = run_start;
                }
            } else {
                run_length = 0;
            }
            prior = value;
        }
        if ink >= minimum && transitions <= MAX_ROW_TRANSITIONS && best_length >= minimum {
            metrics.insert(x, (best_length, best_start, best_start + best_length));
        }
    }
    let maximum_thickness = 4.max((width as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    let mut records = Vec::new();
    for (band_x0, band_x1) in groups(metrics.keys().copied()) {
        let thickness = band_x1 - band_x0 + 1;
        if thickness > maximum_thickness + 2 {
            continue;
        }
        let y0 = median_usize((band_x0..=band_x1).map(|column| metrics[&column].1));
        let y1 = median_usize((band_x0..=band_x1).map(|column| metrics[&column].2));
        if y1 <= y0 {
            continue;
        }
        records.push(VerticalRuleRecord {
            x_center_ratio: round4((band_x0 + band_x1 + 1) as f64 / 2.0 / width as f64),
            y0_ratio: round4(y0 as f64 / height as f64),
            y1_ratio: round4(y1 as f64 / height as f64),
            length_ratio: round4((y1 - y0) as f64 / height as f64),
            thickness_px: thickness.saturating_sub(2).max(1),
        });
    }
    records.sort_by(|left, right| right.length_ratio.total_cmp(&left.length_ratio));
    records
}

fn overlap_share(left: &RuleRecord, right: &RuleRecord) -> f64 {
    let overlap = left.x1_ratio.min(right.x1_ratio) - left.x0_ratio.max(right.x0_ratio);
    if overlap <= 0.0 {
        return 0.0;
    }
    let narrower = left.width_ratio.min(right.width_ratio);
    if narrower > 0.0 {
        overlap / narrower
    } else {
        0.0
    }
}

pub(crate) fn classify_separator(
    rules: &[RuleRecord],
    verticals: &[VerticalRuleRecord],
    minimum_y: f64,
) -> (Vec<RuleRecord>, &'static str) {
    let mut candidates = Vec::new();
    for (index, rule) in rules.iter().enumerate() {
        if !(minimum_y..=SEPARATOR_MAX_Y_RATIO).contains(&rule.y_center_ratio)
            || !(SEPARATOR_MIN_X0_RATIO..=SEPARATOR_MAX_X0_RATIO).contains(&rule.x0_ratio)
            || rule.x1_ratio > SEPARATOR_MAX_X1_RATIO
            || !(SEPARATOR_MIN_WIDTH_RATIO..=SEPARATOR_MAX_WIDTH_RATIO).contains(&rule.width_ratio)
            || rule.darkness < SEPARATOR_MIN_DARKNESS
            || rule.thickness_ratio > SEPARATOR_MAX_THICKNESS_RATIO
            || rule.above_ink > SEPARATOR_MAX_ABOVE_INK
            || rule.below_ink > SEPARATOR_MAX_BELOW_INK
            || rule.above_block_ink < SEPARATOR_MIN_BLOCK_INK
            || rule.below_block_ink < SEPARATOR_MIN_BLOCK_INK
        {
            continue;
        }
        if rules.iter().enumerate().any(|(other_index, other)| {
            other_index != index
                && (other.y_center_ratio - rule.y_center_ratio).abs() <= STACK_NEIGHBOR_Y_RATIO
                && overlap_share(rule, other) >= STACK_OVERLAP_SHARE
        }) {
            continue;
        }
        if verticals.iter().any(|vertical| {
            (rule.x0_ratio - VERTICAL_CROSS_X_PAD_RATIO
                ..=rule.x1_ratio + VERTICAL_CROSS_X_PAD_RATIO)
                .contains(&vertical.x_center_ratio)
                && (vertical.y0_ratio..=vertical.y1_ratio).contains(&rule.y_center_ratio)
        }) {
            continue;
        }
        let mut candidate = rule.clone();
        candidate.kind = Some(
            if candidate.width_ratio >= FULL_RULE_MIN_WIDTH_RATIO {
                "full_rule"
            } else {
                "short_rule"
            }
            .to_owned(),
        );
        candidates.push(candidate);
    }
    match candidates.len() {
        0 => (Vec::new(), "none"),
        1 => (candidates, "found"),
        2 => {
            candidates.sort_by(|left, right| left.x0_ratio.total_cmp(&right.x0_ratio));
            let left = &candidates[0];
            let right = &candidates[1];
            if (left.y_center_ratio - right.y_center_ratio).abs() <= TWO_COLUMN_Y_DELTA_RATIO
                && left.x1_ratio <= right.x0_ratio
            {
                (candidates, "found_two_column")
            } else {
                (Vec::new(), "ambiguous")
            }
        }
        _ => (Vec::new(), "ambiguous"),
    }
}

pub(crate) fn scan_gray_page(gray: &[u8], width: usize, height: usize) -> ScanRecord {
    if width < 64 || height < 64 || gray.len() != width.saturating_mul(height) {
        return ScanRecord {
            status: "unusable_image",
            page_size: (gray.len() == width.saturating_mul(height)).then_some([width, height]),
            threshold: None,
            dark_share: None,
            rule_count: None,
            vertical_rule_count: None,
            rules: None,
            vertical_rules: None,
            separators: None,
            separator_status: None,
        };
    }
    let threshold = otsu_threshold(gray).clamp(OTSU_MIN_THRESHOLD, OTSU_MAX_THRESHOLD);
    let dark = gray
        .iter()
        .map(|value| usize::from(*value) < threshold)
        .collect::<Vec<_>>();
    let dark_share = dark.iter().filter(|value| **value).count() as f64 / dark.len() as f64;
    if dark_share > DARK_PAGE_SHARE {
        return ScanRecord {
            status: "dark_page",
            page_size: Some([width, height]),
            threshold: Some(threshold),
            dark_share: Some(round4(dark_share)),
            rule_count: Some(0),
            vertical_rule_count: Some(0),
            rules: Some(Vec::new()),
            vertical_rules: Some(Vec::new()),
            separators: Some(Vec::new()),
            separator_status: Some("none"),
        };
    }
    let rules = horizontal_rule_records(&dark, width, height);
    let verticals = vertical_rule_records(&dark, width, height);
    let (separators, status) = classify_separator(&rules, &verticals, SEPARATOR_MIN_Y_RATIO);
    ScanRecord {
        status: "ok",
        page_size: Some([width, height]),
        threshold: Some(threshold),
        dark_share: Some(round4(dark_share)),
        rule_count: Some(rules.len()),
        vertical_rule_count: Some(verticals.len()),
        rules: Some(rules.into_iter().take(MAX_RECORDED_RULES).collect()),
        vertical_rules: Some(
            verticals
                .into_iter()
                .take(MAX_RECORDED_VERTICAL_RULES)
                .collect(),
        ),
        separators: Some(separators),
        separator_status: Some(status),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(with_rule: bool) -> Vec<u8> {
        let (width, height) = (800, 1000);
        let mut page = vec![255; width * height];
        for y in (100..620).step_by(4) {
            for row in y..y + 2 {
                for x in (80..720).step_by(3) {
                    page[index(width, x, row)] = 0;
                }
            }
        }
        if with_rule {
            for y in 700..702 {
                for x in 64..320 {
                    page[index(width, x, y)] = 0;
                }
            }
        }
        for y in (720..950).step_by(4) {
            for row in y..y + 2 {
                for x in (80..720).step_by(3) {
                    page[index(width, x, row)] = 0;
                }
            }
        }
        page
    }

    #[test]
    fn synthetic_separator_matches_the_frozen_scanner() {
        let found = scan_gray_page(&synthetic(true), 800, 1000);
        assert_eq!(found.separator_status, Some("found"));
        assert_eq!(found.separators.unwrap()[0].y_center_ratio, 0.701);
        let absent = scan_gray_page(&synthetic(false), 800, 1000);
        assert_eq!(absent.separator_status, Some("none"));
    }
}
