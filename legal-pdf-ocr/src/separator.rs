//! A page render's horizontal rules, and of them the rule printed above the page's notes.
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
const STACK_NEIGHBOR_Y_RATIO: f64 = 0.035;
const STACK_OVERLAP_SHARE: f64 = 0.5;
const TWO_COLUMN_Y_DELTA_RATIO: f64 = 0.012;
const VERTICAL_CROSS_X_PAD_RATIO: f64 = 0.01;

/// Horizontal rules, as `(x0, x1, top, bottom)` pixels, all inclusive: runs of dark pixels (gaps of
/// two bridged) at least `min_length` long, a run in the row just under one already read joining it
/// as the same rule, thicker; at most `max_rows` rows thick, with light rows three pixels above its
/// top and below its bottom across most of it.
pub(crate) fn horizontal_rules(
    dark: impl Fn(usize, usize) -> bool,
    width: usize,
    height: usize,
    min_length: usize,
    max_rows: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let light_share = |y: usize, a: usize, b: usize| (a..b).filter(|&x| !dark(x, y)).count() * 10 >= (b - a) * 7;
    let mut rules: Vec<(usize, usize, usize, usize)> = Vec::new();
    // The rules reaching the row above, which a run in this row may carry on.
    let mut open: Vec<usize> = Vec::new();
    for y in 0..height {
        let mut reaching = Vec::new();
        let mut x = 0;
        while x < width {
            if !dark(x, y) {
                x += 1;
                continue;
            }
            let start = x;
            let mut gap = 0;
            let mut end = x;
            while x < width && gap <= 2 {
                if dark(x, y) {
                    end = x;
                    gap = 0
                } else {
                    gap += 1
                }
                x += 1;
            }
            if end + 1 - start >= min_length {
                let carried = open.iter().copied().find(|&at| rules[at].0 < end && start < rules[at].1);
                let at = carried.unwrap_or_else(|| {
                    rules.push((start, end, y, y));
                    rules.len() - 1
                });
                let rule = &mut rules[at];
                (rule.0, rule.1, rule.3) = (rule.0.min(start), rule.1.max(end), y);
                if !reaching.contains(&at) {
                    reaching.push(at);
                }
            }
        }
        open = reaching;
    }
    rules.retain(|rule| {
        rule.3 - rule.2 < max_rows
            && rule.2 >= 3
            && rule.3 + 3 < height
            && light_share(rule.2 - 3, rule.0, rule.1 + 1)
            && light_share(rule.3 + 3, rule.0, rule.1 + 1)
    });
    rules
}

/// Where the rule above a page's notes runs, in the units of `page_height`, read from its grey
/// render: one rule, or two side by side over two columns of notes (the left one's).
pub(crate) fn separator_y(gray: &[u8], width: usize, height: usize, page_height: f64) -> Option<f64> {
    if width < 64 || height < 64 || gray.len() != width * height {
        return None;
    }
    let threshold = otsu_threshold(gray).clamp(OTSU_MIN_THRESHOLD, OTSU_MAX_THRESHOLD);
    let dark = gray.iter().map(|value| usize::from(*value) < threshold).collect::<Vec<_>>();
    if dark.iter().filter(|value| **value).count() as f64 / dark.len() as f64 > DARK_PAGE_SHARE {
        return None;
    }
    let rules = rule_records(&dark, width, height);
    let verticals = vertical_rules(&dark, width, height);
    separator(&rules, &verticals).map(|ratio| ratio * page_height)
}

/// A horizontal rule, its place and size as shares of the page's, its darkness and the ink just
/// above and below it and in the blocks of text beyond.
struct Rule {
    y: f64,
    x0: f64,
    x1: f64,
    thickness: f64,
    darkness: f64,
    above_ink: f64,
    below_ink: f64,
    above_block_ink: f64,
    below_block_ink: f64,
}

impl Rule {
    fn width(&self) -> f64 {
        self.x1 - self.x0
    }
}

/// A vertical rule's middle and its top and bottom, as shares of the page's width and height.
struct VerticalRule {
    x: f64,
    y0: f64,
    y1: f64,
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
    let total_mu = histogram.iter().enumerate().map(|(value, count)| value as f64 * count).sum::<f64>();
    let mut best_index = 0;
    let mut best_sigma = 0.0;
    for (value, count) in histogram.iter().enumerate() {
        omega += count;
        mu += value as f64 * count;
        let denominator = omega * (total - omega);
        let sigma = if denominator > 0.0 { (total_mu * omega - mu * total).powi(2) / denominator } else { 0.0 };
        if sigma > best_sigma {
            best_sigma = sigma;
            best_index = value;
        }
    }
    best_index
}

fn groups(indices: impl IntoIterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut result: Vec<(usize, usize)> = Vec::new();
    for value in indices {
        match result.last_mut() {
            Some(last) if value == last.1 + 1 => last.1 = value,
            _ => result.push((value, value)),
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

/// The dark share of rows `y0..y1` (clipped to the page) between `x0` and `x1`; `empty` where no
/// row is left.
fn ink_share(dark: &[bool], width: usize, height: usize, (y0, y1): (isize, isize), (x0, x1): (usize, usize), empty: f64) -> f64 {
    let y0 = y0.max(0) as usize;
    let y1 = y1.min(height as isize).max(0) as usize;
    if y1 <= y0 || x1 <= x0 {
        return empty;
    }
    let count = (y0..y1).map(|y| dark[y * width + x0..y * width + x1].iter().filter(|value| **value).count()).sum::<usize>();
    count as f64 / ((y1 - y0) * (x1 - x0)) as f64
}

fn rule_records(dark: &[bool], width: usize, height: usize) -> Vec<Rule> {
    let minimum = 8.max((width as f64 * MIN_RULE_WIDTH_RATIO) as usize);
    let maximum_thickness = 4.max((height as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    let guard = 2.max((height as f64 * INK_GUARD_RATIO) as usize) as isize;
    let ink_window = 4.max((height as f64 * INK_WINDOW_RATIO) as usize) as isize;
    let block_window = (ink_window * 2).max((height as f64 * BLOCK_INK_WINDOW_RATIO) as isize);
    let (w, h) = (width as f64, height as f64);
    horizontal_rules(|x, y| dark[y * width + x], width, height, minimum, maximum_thickness)
        .into_iter()
        .map(|(x0, end, top, bottom)| {
            let x1 = end + 1;
            let darkness = (top..=bottom)
                .map(|y| dark[y * width + x0..y * width + x1].iter().filter(|value| **value).count() as f64 / (x1 - x0) as f64)
                .fold(0.0, f64::max);
            let above = top as isize - guard;
            let below = (bottom + 1) as isize + guard;
            let share = |rows, empty| ink_share(dark, width, height, rows, (x0, x1), empty);
            Rule {
                y: (top + bottom + 1) as f64 / 2.0 / h,
                x0: x0 as f64 / w,
                x1: x1 as f64 / w,
                thickness: (bottom + 1 - top) as f64 / h,
                darkness,
                above_ink: share((above - ink_window, above), 1.0),
                below_ink: share((below, below + ink_window), 1.0),
                above_block_ink: share((above - block_window, above), 0.0),
                below_block_ink: share((below, below + block_window), 0.0),
            }
        })
        .collect()
}

fn vertical_rules(dark: &[bool], width: usize, height: usize) -> Vec<VerticalRule> {
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
            let value = dark[y * width + x] || (x > 0 && dark[y * width + x - 1]) || (x + 1 < width && dark[y * width + x + 1]);
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
            metrics.insert(x, (best_start, best_start + best_length));
        }
    }
    let maximum_thickness = 4.max((width as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    groups(metrics.keys().copied())
        .into_iter()
        .filter(|(x0, x1)| x1 - x0 < maximum_thickness + 2)
        .filter_map(|(band_x0, band_x1)| {
            let y0 = median_usize((band_x0..=band_x1).map(|column| metrics[&column].0));
            let y1 = median_usize((band_x0..=band_x1).map(|column| metrics[&column].1));
            (y1 > y0).then(|| VerticalRule {
                x: (band_x0 + band_x1 + 1) as f64 / 2.0 / width as f64,
                y0: y0 as f64 / height as f64,
                y1: y1 as f64 / height as f64,
            })
        })
        .collect()
}

/// How much of the narrower of two rules the other spans.
fn overlap_share(left: &Rule, right: &Rule) -> f64 {
    let overlap = left.x1.min(right.x1) - left.x0.max(right.x0);
    let narrower = left.width().min(right.width());
    if overlap > 0.0 && narrower > 0.0 {
        overlap / narrower
    } else {
        0.0
    }
}

/// The note separator's height as a share of the page's: a rule in the page's lower part, from
/// near its left margin, dark, thin, clear of ink just above and below, between blocks of text,
/// with no rule stacked on it and none crossing it; one such rule, or two at one height over two
/// columns (the left one's).
fn separator(rules: &[Rule], verticals: &[VerticalRule]) -> Option<f64> {
    let candidates = rules
        .iter()
        .enumerate()
        .filter(|(index, rule)| {
            (SEPARATOR_MIN_Y_RATIO..=SEPARATOR_MAX_Y_RATIO).contains(&rule.y)
                && (SEPARATOR_MIN_X0_RATIO..=SEPARATOR_MAX_X0_RATIO).contains(&rule.x0)
                && rule.x1 <= SEPARATOR_MAX_X1_RATIO
                && (SEPARATOR_MIN_WIDTH_RATIO..=SEPARATOR_MAX_WIDTH_RATIO).contains(&rule.width())
                && rule.darkness >= SEPARATOR_MIN_DARKNESS
                && rule.thickness <= SEPARATOR_MAX_THICKNESS_RATIO
                && rule.above_ink <= SEPARATOR_MAX_ABOVE_INK
                && rule.below_ink <= SEPARATOR_MAX_BELOW_INK
                && rule.above_block_ink >= SEPARATOR_MIN_BLOCK_INK
                && rule.below_block_ink >= SEPARATOR_MIN_BLOCK_INK
                && !rules.iter().enumerate().any(|(other_index, other)| {
                    other_index != *index && (other.y - rule.y).abs() <= STACK_NEIGHBOR_Y_RATIO && overlap_share(rule, other) >= STACK_OVERLAP_SHARE
                })
                && !verticals.iter().any(|vertical| {
                    (rule.x0 - VERTICAL_CROSS_X_PAD_RATIO..=rule.x1 + VERTICAL_CROSS_X_PAD_RATIO).contains(&vertical.x)
                        && (vertical.y0..=vertical.y1).contains(&rule.y)
                })
        })
        .map(|(_, rule)| rule)
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [rule] => Some(rule.y),
        [first, second] => {
            let (left, right) = if first.x0 <= second.x0 { (first, second) } else { (second, first) };
            ((left.y - right.y).abs() <= TWO_COLUMN_Y_DELTA_RATIO && left.x1 <= right.x0).then_some(left.y)
        }
        _ => None,
    }
}
