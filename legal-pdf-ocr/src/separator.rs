//! A page render's horizontal rules, and of them the rule printed above the page's notes.

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

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_next_dark_avx2(row: &[u8], mut x: usize, thresh: u8) -> usize {
    use std::arch::x86_64::*;
    let len = row.len();
    let ptr = row.as_ptr();
    let sign = _mm256_set1_epi8(-128);
    let t_biased = _mm256_set1_epi8((thresh ^ 0x80) as i8);
    while x + 32 <= len {
        let px = _mm256_loadu_si256(ptr.add(x) as *const __m256i);
        let dark = _mm256_cmpgt_epi8(t_biased, _mm256_xor_si256(px, sign));
        let mask = _mm256_movemask_epi8(dark) as u32;
        if mask != 0 {
            return x + mask.trailing_zeros() as usize;
        }
        x += 32;
    }
    while x < len && *ptr.add(x) >= thresh {
        x += 1;
    }
    x
}

#[inline]
fn find_next_dark(row: &[u8], mut x: usize, thresh: u8) -> usize {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: x <= row.len(), bounds checked inside helper.
        return unsafe { find_next_dark_avx2(row, x, thresh) };
    }
    while x < row.len() && row[x] >= thresh {
        x += 1;
    }
    x
}

/// Horizontal rules, as `(x0, x1, top, bottom)` pixels, all inclusive: runs of dark pixels (gaps of
/// two bridged) at least `min_length` long, a run in the row just under one already read joining it
/// as the same rule, thicker; at most `max_rows` rows thick, with light rows three pixels above its
/// top and below its bottom across most of it.
fn horizontal_rules_gray(
    gray: &[u8],
    thresh: u8,
    width: usize,
    height: usize,
    min_length: usize,
    max_rows: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let light_share = |y: usize, a: usize, b: usize| {
        let row = &gray[y * width + a..y * width + b];
        row.iter().filter(|&&v| v >= thresh).count() * 10 >= (b - a) * 7
    };
    let mut rules: Vec<(usize, usize, usize, usize)> = Vec::new();
    // The rules reaching the row above, which a run in this row may carry on.
    let mut open: Vec<usize> = Vec::new();
    let mut reaching: Vec<usize> = Vec::new();
    for y in 0..height {
        reaching.clear();
        let row = &gray[y * width..(y + 1) * width];
        let mut x = 0;
        while x < width {
            x = find_next_dark(row, x, thresh);
            if x >= width {
                break;
            }
            let start = x;
            let mut gap = 0;
            let mut end = x;
            while x < width && gap <= 2 {
                if row[x] < thresh {
                    end = x;
                    gap = 0;
                } else {
                    gap += 1;
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
        std::mem::swap(&mut open, &mut reaching);
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
    let (threshold, dark_count) = otsu_threshold_and_dark_count(gray);
    if dark_count as f64 / gray.len() as f64 > DARK_PAGE_SHARE {
        return None;
    }
    let thresh = threshold as u8;
    let rules = rule_records(gray, thresh, width, height);
    separator(gray, thresh, width, height, &rules).map(|ratio| ratio * page_height)
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

fn otsu_threshold_and_dark_count(gray: &[u8]) -> (usize, usize) {
    let (mut h0, mut h1, mut h2, mut h3) = ([0_u32; 256], [0_u32; 256], [0_u32; 256], [0_u32; 256]);
    let mut chunks = gray.chunks_exact(4);
    for c in &mut chunks {
        h0[usize::from(c[0])] += 1;
        h1[usize::from(c[1])] += 1;
        h2[usize::from(c[2])] += 1;
        h3[usize::from(c[3])] += 1;
    }
    for &v in chunks.remainder() {
        h0[usize::from(v)] += 1;
    }
    let mut histogram = [0_u32; 256];
    let mut total_mu = 0.0_f64;
    for value in 0..256 {
        let c = h0[value] + h1[value] + h2[value] + h3[value];
        histogram[value] = c;
        total_mu += value as f64 * f64::from(c);
    }
    let total = gray.len() as f64;
    let mut omega = 0.0;
    let mut mu = 0.0;
    let mut best_index = 0;
    let mut best_sigma = 0.0;
    for (value, &cnt) in histogram.iter().enumerate() {
        let count = f64::from(cnt);
        omega += count;
        mu += value as f64 * count;
        let denominator = omega * (total - omega);
        let sigma = if denominator > 0.0 { (total_mu * omega - mu * total).powi(2) / denominator } else { 0.0 };
        if sigma > best_sigma {
            best_sigma = sigma;
            best_index = value;
        }
    }
    let clamped = best_index.clamp(OTSU_MIN_THRESHOLD, OTSU_MAX_THRESHOLD);
    let dark_count = histogram[..clamped].iter().map(|&c| c as usize).sum();
    (clamped, dark_count)
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
fn ink_share(gray: &[u8], thresh: u8, width: usize, height: usize, (y0, y1): (isize, isize), (x0, x1): (usize, usize), empty: f64) -> f64 {
    let y0 = y0.max(0) as usize;
    let y1 = y1.min(height as isize).max(0) as usize;
    if y1 <= y0 || x1 <= x0 {
        return empty;
    }
    let count = (y0..y1)
        .map(|y| gray[y * width + x0..y * width + x1].iter().filter(|&&value| value < thresh).count())
        .sum::<usize>();
    count as f64 / ((y1 - y0) * (x1 - x0)) as f64
}

fn rule_records(gray: &[u8], thresh: u8, width: usize, height: usize) -> Vec<Rule> {
    let minimum = 8.max((width as f64 * MIN_RULE_WIDTH_RATIO) as usize);
    let maximum_thickness = 4.max((height as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    let guard = 2.max((height as f64 * INK_GUARD_RATIO) as usize) as isize;
    let ink_window = 4.max((height as f64 * INK_WINDOW_RATIO) as usize) as isize;
    let block_window = (ink_window * 2).max((height as f64 * BLOCK_INK_WINDOW_RATIO) as isize);
    let (w, h) = (width as f64, height as f64);
    horizontal_rules_gray(gray, thresh, width, height, minimum, maximum_thickness)
        .into_iter()
        .map(|(x0, end, top, bottom)| {
            let x1 = end + 1;
            let darkness = (top..=bottom)
                .map(|y| gray[y * width + x0..y * width + x1].iter().filter(|&&value| value < thresh).count() as f64 / (x1 - x0) as f64)
                .fold(0.0, f64::max);
            let above = top as isize - guard;
            let below = (bottom + 1) as isize + guard;
            let share = |rows, empty| ink_share(gray, thresh, width, height, rows, (x0, x1), empty);
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

/// Row-major computation of vertical rule columns, invoked only when candidate horizontal rules exist.
fn vertical_rules(gray: &[u8], thresh: u8, width: usize, height: usize) -> Vec<VerticalRule> {
    #[derive(Clone, Copy, Default)]
    struct ColState {
        ink: u32,
        transitions: u32,
        best_length: u32,
        best_start: u32,
        run_length: u32,
        run_start: u32,
        prior: bool,
    }
    let minimum = 12.max((height as f64 * MIN_VERTICAL_RULE_LENGTH_RATIO) as usize) as u32;
    let mut cols = vec![ColState::default(); width];
    for y in 0..height {
        let row = &gray[y * width..(y + 1) * width];
        let yu = y as u32;
        for (x, state) in cols.iter_mut().enumerate() {
            let value = row[x] < thresh || (x > 0 && row[x - 1] < thresh) || (x + 1 < width && row[x + 1] < thresh);
            state.ink += u32::from(value);
            if y > 0 && value != state.prior {
                state.transitions += 1;
            }
            if value {
                if state.run_length == 0 {
                    state.run_start = yu;
                }
                state.run_length += 1;
                if state.run_length > state.best_length {
                    state.best_length = state.run_length;
                    state.best_start = state.run_start;
                }
            } else {
                state.run_length = 0;
            }
            state.prior = value;
        }
    }
    let metrics: Vec<Option<(usize, usize)>> = cols
        .into_iter()
        .map(|s| {
            (s.ink >= minimum && s.transitions <= MAX_ROW_TRANSITIONS as u32 && s.best_length >= minimum)
                .then_some((s.best_start as usize, (s.best_start + s.best_length) as usize))
        })
        .collect();
    let maximum_thickness = 4.max((width as f64 * MAX_RULE_THICKNESS_RATIO) as usize);
    let mut result = Vec::new();
    let mut x = 0;
    while x < width {
        if metrics[x].is_none() {
            x += 1;
            continue;
        }
        let band_x0 = x;
        while x + 1 < width && metrics[x + 1].is_some() {
            x += 1;
        }
        let band_x1 = x;
        x += 1;
        if band_x1 - band_x0 < maximum_thickness + 2 {
            let y0 = median_usize((band_x0..=band_x1).map(|c| metrics[c].unwrap().0));
            let y1 = median_usize((band_x0..=band_x1).map(|c| metrics[c].unwrap().1));
            if y1 > y0 {
                result.push(VerticalRule {
                    x: (band_x0 + band_x1 + 1) as f64 / 2.0 / width as f64,
                    y0: y0 as f64 / height as f64,
                    y1: y1 as f64 / height as f64,
                });
            }
        }
    }
    result
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
fn separator(gray: &[u8], thresh: u8, width: usize, height: usize, rules: &[Rule]) -> Option<f64> {
    let pre_candidates = rules
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
        })
        .map(|(_, rule)| rule)
        .collect::<Vec<_>>();
    if pre_candidates.is_empty() {
        return None;
    }
    let verticals = vertical_rules(gray, thresh, width, height);
    let candidates = pre_candidates
        .into_iter()
        .filter(|rule| {
            !verticals.iter().any(|vertical| {
                (rule.x0 - VERTICAL_CROSS_X_PAD_RATIO..=rule.x1 + VERTICAL_CROSS_X_PAD_RATIO).contains(&vertical.x)
                    && (vertical.y0..=vertical.y1).contains(&rule.y)
            })
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separator_y_detects_footnote_rule_and_rejects_crossed_box() {
        let (w, h) = (400usize, 600usize);
        let mut page = vec![255u8; w * h];
        // Text block above and below separator.
        for y in (120..360).step_by(24) {
            for dy in 0..8 {
                for x in 40..340 {
                    if (x / 4 + dy) % 3 != 0 {
                        page[(y + dy) * w + x] = 25;
                    }
                }
            }
        }
        for y in (450..560).step_by(24) {
            for dy in 0..8 {
                for x in 40..340 {
                    if (x / 4 + dy) % 3 != 0 {
                        page[(y + dy) * w + x] = 25;
                    }
                }
            }
        }
        // Horizontal separator at y = 410..412, x = 40..180.
        for y in 410..=411 {
            for x in 40..180 {
                page[y * w + x] = 10;
            }
        }
        let expected_ratio = (410 + 411 + 1) as f64 / 2.0 / h as f64;
        let found = separator_y(&page, w, h, 792.0).expect("expected footnote separator");
        assert!((found - expected_ratio * 792.0).abs() < 1e-6);

        // Crossing vertical rule across the separator disqualifies it as a box border.
        for y in 360..460 {
            page[y * w + 100] = 10;
        }
        assert_eq!(separator_y(&page, w, h, 792.0), None);
    }
}
