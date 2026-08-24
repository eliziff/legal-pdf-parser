use crate::structure::{is_note_symbol, label_prefix, scalar_suffix, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, union_bbox, Diagnostic, Line, Page, Region};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

fn p50(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OrderRepair {
    None,
    Column,
    Geometry,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ColumnModel {
    pub(super) kind: &'static str,
    pub(super) split_x: f64,
    left_count: usize,
    right_count: usize,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct OrderDecision {
    pub(super) repair: OrderRepair,
    source_switches: usize,
    strategy: &'static str,
    pub(super) reason: &'static str,
}

impl OrderDecision {
    const fn keep(reason: &'static str) -> Self {
        Self {
            repair: OrderRepair::None,
            source_switches: 0,
            strategy: "kraken-native",
            reason,
        }
    }
}

pub(super) fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

pub(super) fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        && line.bbox[3] > line.bbox[1]
}

pub(super) fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

#[derive(Default)]
pub(super) struct TableEvidence {
    pub(super) lines: HashSet<usize>,
    pub(super) contents: bool,
    rows: usize,
    columns: usize,
    numeric_cells: usize,
    cells: usize,
    top: f64,
    bottom: f64,
    left: f64,
    right: f64,
    line_height: f64,
}

impl TableEvidence {
    pub(super) fn strong(&self) -> bool {
        self.rows >= 6 && self.columns >= 3 && self.numeric_cells * 5 >= self.cells
    }

    pub(super) fn continuation(&self) -> bool {
        self.rows >= 6 && self.columns >= 2 && self.numeric_cells * 10 >= self.cells
    }

    pub(super) fn reaches_page_bottom(&self, page_height: f64) -> bool {
        page_height > 0.0 && self.bottom >= page_height * 0.70
    }

    pub(super) fn continuation_on_page(&self, page_height: f64) -> bool {
        page_height > 0.0 && self.top <= page_height * 0.30 && self.continuation()
    }

    pub(super) fn expanded_lines(
        &self,
        lines: &[Line],
        page_height: f64,
        continuation: bool,
        separator: Option<f64>,
    ) -> HashSet<usize> {
        if self.lines.is_empty() {
            return HashSet::new();
        }
        let top = if continuation {
            lines
                .iter()
                .filter(|line| {
                    has_valid_bbox(line)
                        && !line.exclude_from_body
                        && line.bbox[1] >= page_height * 0.08
                        && line.bbox[1] <= self.top
                })
                .map(|line| line.bbox[1])
                .min_by(f64::total_cmp)
                .unwrap_or(self.top)
        } else {
            self.top
        };
        let table_size = p50(self
            .lines
            .iter()
            .map(|index| line_font_size(&lines[*index]))
            .filter(|size| *size > 0.0)
            .collect());
        let bottom = separator
            .filter(|cut| {
                *cut > top
                    && lines
                        .iter()
                        .filter(|line| has_valid_bbox(line) && line.bbox[1] >= *cut)
                        .min_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]))
                        .is_some_and(|next| {
                            let gap = next.bbox[1] - cut;
                            gap >= self.line_height * 1.5
                                && (gap >= self.line_height * 4.0
                                    || (table_size > 0.0
                                        && line_font_size(next) <= table_size * 0.90))
                        })
            })
            .unwrap_or(self.bottom);
        lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                (has_valid_bbox(line)
                    && !line.exclude_from_body
                    && line.bbox[3] >= top - self.line_height
                    && line.bbox[1] <= bottom
                    && line.bbox[2] >= self.left
                    && line.bbox[0] <= self.right)
                    .then_some(index)
            })
            .collect()
    }

    pub(super) fn table_note_lines(
        &self,
        lines: &[Line],
        cells: &HashSet<usize>,
    ) -> HashSet<usize> {
        let anchors: Vec<_> = lines
            .iter()
            .enumerate()
            .filter_map(|(index, line)| {
                let prefix = label_prefix(&line.text)?;
                let symbolic = prefix
                    .label
                    .chars()
                    .all(|character| !character.is_ascii_digit());
                let tail = scalar_suffix(&line.text, prefix.end);
                (symbolic
                    && tail
                        .chars()
                        .filter(|character| character.is_alphabetic())
                        .count()
                        >= 4
                    && ((!self.lines.contains(&index) && cells.contains(&index))
                        || (line.bbox[1] > self.bottom
                            && line.bbox[1] <= self.bottom + self.line_height * 3.0)))
                    .then_some(index)
            })
            .collect();
        let mut notes: HashSet<_> = anchors.iter().copied().collect();
        for anchor in anchors {
            let mut bottom = lines[anchor].bbox[3];
            let mut following: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(index, line)| *index != anchor && line.bbox[1] >= lines[anchor].bbox[1])
                .collect();
            following.sort_by(|(_, left), (_, right)| left.bbox[1].total_cmp(&right.bbox[1]));
            for (index, line) in following {
                if line.bbox[1] - bottom > self.line_height * 1.6
                    || cells.contains(&index)
                    || has_table_caption(std::slice::from_ref(line))
                {
                    break;
                }
                notes.insert(index);
                bottom = bottom.max(line.bbox[3]);
            }
        }
        notes
    }
}

pub(super) fn strong_table_evidence(evidence: &TableEvidence, lines: &[Line]) -> bool {
    let mut prior = None;
    let mut run = 0;
    let mut longest = 0;
    for line in lines {
        let Some(prefix) = label_prefix(&line.text) else {
            continue;
        };
        let Ok(label) = prefix.label.parse::<u32>() else {
            continue;
        };
        if !scalar_suffix(&line.text, prefix.end)
            .chars()
            .any(char::is_alphabetic)
        {
            continue;
        }
        run = if prior.is_some_and(|value| (1..=3).contains(&label.saturating_sub(value))) {
            run + 1
        } else {
            1
        };
        longest = longest.max(run);
        prior = Some(label);
    }
    evidence.strong()
        && longest < 3
        && lines
            .iter()
            .enumerate()
            .filter(|(index, line)| {
                standalone_note_label(line) && aligned_note_body_index(lines, *index).is_some()
            })
            .take(3)
            .count()
            < 3
}

pub(super) fn has_table_caption(lines: &[Line]) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    let regex =
        RE.get_or_init(|| Regex::new(r"(?i)^(?:table|tableau)\s+(?:\d+|[ivxlcdm]+)\b").unwrap());
    lines.iter().any(|line| regex.is_match(line.text.trim()))
}

pub(super) fn contents_grid(lines: &[Line], page_width: f64) -> bool {
    if page_width <= 0.0 {
        return false;
    }
    let mut locators: Vec<_> = lines
        .iter()
        .filter_map(|line| {
            let value = line.text.trim().parse::<u32>().ok()?;
            (has_valid_bbox(line) && line.bbox[0] >= page_width * 0.72).then_some((line, value))
        })
        .filter(|(locator, _)| {
            lines.iter().any(|peer| {
                let overlap = locator.bbox[3].min(peer.bbox[3]) - locator.bbox[1].max(peer.bbox[1]);
                let height = (locator.bbox[3] - locator.bbox[1]).min(peer.bbox[3] - peer.bbox[1]);
                peer.bbox[0] < page_width * 0.68
                    && peer.text.chars().any(char::is_alphabetic)
                    && height > 0.0
                    && overlap / height >= 0.5
            })
        })
        .collect();
    if locators.len() < 6 {
        return false;
    }
    locators.sort_by(|(left, _), (right, _)| band_geometry_order(left, right));
    let monotone = locators
        .windows(2)
        .filter(|pair| pair[0].1 <= pair[1].1)
        .count();
    monotone * 5 >= (locators.len() - 1) * 4
}

pub(super) fn table_evidence(lines: &[Line], page_width: f64) -> TableEvidence {
    let height = p50(lines
        .iter()
        .filter(|line| has_valid_bbox(line))
        .map(|line| line.bbox[3] - line.bbox[1])
        .filter(|value| *value > 0.0)
        .collect());
    if height <= 0.0 {
        return TableEvidence::default();
    }
    let caption = lines
        .iter()
        .filter(|line| has_table_caption(std::slice::from_ref(line)))
        .min_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]));
    let caption_bottom = caption.map(|line| line.bbox[3]);
    let mut rows: HashMap<i64, Vec<usize>> = HashMap::new();
    for (index, line) in lines.iter().enumerate().filter(|(_, line)| {
        has_valid_bbox(line)
            && !line.exclude_from_body
            && !matches!(line.region_type.as_str(), "header" | "footer")
    }) {
        rows.entry((line_center_y(line) / (height * 0.75)).round() as i64)
            .or_default()
            .push(index);
    }
    let mut dense_rows: Vec<_> = rows
        .into_values()
        .filter(|row| {
            row.len() >= 2
                && p50(row
                    .iter()
                    .map(|index| lines[*index].text.trim().chars().count() as f64)
                    .collect())
                    <= 24.0
        })
        .collect();
    dense_rows.sort_by(|left, right| {
        let center = |row: &[usize]| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .unwrap_or(0.0)
        };
        center(left).total_cmp(&center(right))
    });
    if let Some(bottom) = caption_bottom {
        dense_rows.retain(|row| {
            row.iter()
                .map(|index| line_center_y(&lines[*index]))
                .min_by(f64::total_cmp)
                .is_some_and(|center| center >= bottom - height)
        });
    }
    let contents = contents_grid(lines, page_width);
    if !contents {
        let mut prior = None;
        dense_rows = dense_rows
            .into_iter()
            .take_while(|row| {
                let center = row
                    .iter()
                    .map(|index| line_center_y(&lines[*index]))
                    .min_by(f64::total_cmp)
                    .unwrap_or(0.0);
                let connected = prior.is_none_or(|value| center - value <= height * 6.0);
                prior = Some(center);
                connected
            })
            .collect();
    }
    if dense_rows.len() < 3 {
        return TableEvidence {
            contents,
            ..TableEvidence::default()
        };
    }
    let mut columns: HashMap<i64, usize> = HashMap::new();
    for row in &dense_rows {
        let mut seen = HashSet::new();
        for index in row {
            seen.insert((lines[*index].bbox[0] / (height * 2.0)).round() as i64);
        }
        for column in seen {
            *columns.entry(column).or_default() += 1;
        }
    }
    let row_count = dense_rows.len();
    let column_count = columns.values().filter(|count| **count >= 3).count();
    let compact: Vec<_> = dense_rows
        .iter()
        .flatten()
        .map(|index| lines[*index].text.trim())
        .collect();
    let numeric_cells = compact
        .iter()
        .filter(|text| {
            text.chars().any(|character| character.is_ascii_digit())
                && text.chars().all(|character| {
                    character.is_ascii_digit()
                        || character.is_whitespace()
                        || matches!(
                            character,
                            '.' | ',' | '%' | '(' | ')' | '/' | '$' | '-' | '\u{2013}' | '\u{2014}'
                        )
                })
        })
        .count();
    let dense_lines: HashSet<_> = dense_rows.into_iter().flatten().collect();
    TableEvidence {
        contents,
        top: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[1])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        bottom: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[3])
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        left: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[0])
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        right: dense_lines
            .iter()
            .map(|index| lines[*index].bbox[2])
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        line_height: height,
        lines: dense_lines,
        rows: row_count,
        columns: column_count,
        numeric_cells,
        cells: compact.len(),
    }
}

pub(super) fn column_model(lines: &[Line], page_width: f64) -> ColumnModel {
    column_model_with_furniture(lines, page_width, false)
}

pub(super) fn margin_note_model(
    lines: &[Line],
    labels: &[usize],
    page_width: f64,
    body_size: f64,
    minimum_labels: usize,
) -> Option<ColumnModel> {
    if labels.is_empty() || page_width <= 0.0 || body_size <= 0.0 {
        return None;
    }
    [false, true]
        .into_iter()
        .filter_map(|right| {
            let lane: Vec<_> = labels
                .iter()
                .copied()
                .filter(|index| (line_center_x(&lines[*index]) >= page_width / 2.0) == right)
                .collect();
            if lane.len() < minimum_labels {
                return None;
            }
            let label_set: HashSet<_> = lane.iter().copied().collect();
            let mut note_left = page_width;
            let mut note_right: f64 = 0.0;
            let mut note_top = f64::INFINITY;
            let mut note_bottom: f64 = 0.0;
            for index in &lane {
                let label = &lines[*index];
                note_left = note_left.min(label.bbox[0]);
                note_right = note_right.max(label.bbox[2]);
                note_top = note_top.min(label.bbox[1]);
                note_bottom = note_bottom.max(label.bbox[3]);
                if standalone_note_label(label) {
                    if let Some(body) = aligned_note_body_index(lines, *index) {
                        note_left = note_left.min(lines[body].bbox[0]);
                        note_right = note_right.max(lines[body].bbox[2]);
                        note_top = note_top.min(lines[body].bbox[1]);
                        note_bottom = note_bottom.max(lines[body].bbox[3]);
                    }
                }
            }
            let prose: Vec<_> = lines
                .iter()
                .enumerate()
                .filter(|(index, line)| {
                    !label_set.contains(index)
                        && has_valid_bbox(line)
                        && !line.exclude_from_body
                        && !matches!(line.region_type.as_str(), "header" | "footer")
                        && line_font_size(line) >= body_size * 0.90
                        && line_width(line) >= page_width * 0.25
                        && line.bbox[3] >= note_top
                        && line.bbox[1] <= note_bottom
                })
                .map(|(_, line)| line)
                .collect();
            if prose.len() < 3 {
                return None;
            }
            let body_left = prose
                .iter()
                .map(|line| line.bbox[0])
                .min_by(f64::total_cmp)?;
            let body_right = prose
                .iter()
                .map(|line| line.bbox[2])
                .max_by(f64::total_cmp)?;
            let gap = (body_size * 1.5).max(page_width * 0.02);
            let split_x = if note_right + gap <= body_left {
                (note_right + body_left) / 2.0
            } else if body_right + gap <= note_left {
                (body_right + note_left) / 2.0
            } else {
                return None;
            };
            Some((
                lane.len(),
                ColumnModel {
                    kind: "margin_column",
                    split_x,
                    left_count: if right { prose.len() } else { lane.len() },
                    right_count: if right { lane.len() } else { prose.len() },
                },
            ))
        })
        .max_by_key(|(count, _)| *count)
        .map(|(_, model)| model)
}

pub(super) fn column_model_with_furniture(
    lines: &[Line],
    page_width: f64,
    ignore_centered_furniture: bool,
) -> ColumnModel {
    let single = ColumnModel {
        kind: "single",
        split_x: 0.0,
        left_count: 0,
        right_count: 0,
    };
    if page_width <= 0.0 || lines.is_empty() {
        return single;
    }
    let boxed: Vec<&Line> = lines
        .iter()
        .filter(|line| {
            has_valid_bbox(line)
                && !line.exclude_from_body
                && !matches!(line.region_type.as_str(), "header" | "footer")
        })
        .collect();
    let centered_band = |line: &Line| {
        let width_ratio = line_width(line) / page_width;
        ignore_centered_furniture
            && width_ratio <= 0.30
            && (line_center_x(line) / page_width - 0.5).abs() <= 0.12
    };
    let inference_lines: Vec<&Line> = boxed
        .iter()
        .copied()
        .filter(|line| !centered_band(line))
        .collect();
    let candidates: Vec<&Line> = inference_lines
        .iter()
        .copied()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .collect();
    if candidates.len() < 6
        || (!inference_lines.is_empty()
            && 1.0 - candidates.len() as f64 / inference_lines.len() as f64 > 0.40)
    {
        return single;
    }
    let mut centers: Vec<f64> = candidates.iter().map(|line| line_center_x(line)).collect();
    centers.sort_by(f64::total_cmp);
    centers
        .windows(2)
        .filter_map(|pair| {
            let left = pair[0];
            let right = pair[1];
            let center_gap = pair[1] - pair[0];
            let initial_split = (left + right) / 2.0;
            if center_gap / page_width < 0.12
                || !(0.25..=0.75).contains(&(initial_split / page_width))
            {
                return None;
            }
            let (left_lines, right_lines): (Vec<_>, Vec<_>) = candidates
                .iter()
                .copied()
                .partition(|line| line_center_x(line) < initial_split);
            if left_lines.len() < 3 || right_lines.len() < 3 {
                return None;
            }
            let (split_x, gap, imbalance) = if ignore_centered_furniture {
                let left_edge = left_lines
                    .iter()
                    .map(|line| line.bbox[2])
                    .max_by(f64::total_cmp)
                    .unwrap_or(initial_split);
                let right_edge = right_lines
                    .iter()
                    .map(|line| line.bbox[0])
                    .min_by(f64::total_cmp)
                    .unwrap_or(initial_split);
                if right_edge <= left_edge {
                    return None;
                }
                (
                    (left_edge + right_edge) / 2.0,
                    right_edge - left_edge,
                    left_lines.len().abs_diff(right_lines.len()),
                )
            } else {
                (initial_split, center_gap, 0)
            };
            let split_ratio = (split_x / page_width * 10_000.0).round_ties_even() / 10_000.0;
            let vertical_extent = |values: &[&Line]| {
                (
                    values
                        .iter()
                        .map(|line| line.bbox[1])
                        .min_by(f64::total_cmp)
                        .unwrap_or(0.0),
                    values
                        .iter()
                        .map(|line| line.bbox[3])
                        .max_by(f64::total_cmp)
                        .unwrap_or(0.0),
                )
            };
            let left_y = vertical_extent(&left_lines);
            let right_y = vertical_extent(&right_lines);
            let span = left_y.1.max(right_y.1) - left_y.0.min(right_y.0);
            let overlap = left_y.1.min(right_y.1) - left_y.0.max(right_y.0);
            if span <= 0.0 || overlap.max(0.0) / span < 0.30 {
                return None;
            }
            let crossings = candidates
                .iter()
                .filter(|line| line.bbox[0] < split_x && line.bbox[2] > split_x)
                .count();
            let left_width = p50(left_lines
                .iter()
                .map(|line| line_width(line) / page_width)
                .collect());
            let right_width = p50(right_lines
                .iter()
                .map(|line| line_width(line) / page_width)
                .collect());
            let width_ratio = left_width.min(right_width) / left_width.max(right_width);
            Some((
                crossings,
                imbalance,
                gap,
                ColumnModel {
                    kind: if (crossings == 0 && (0.40..=0.60).contains(&split_ratio))
                        || width_ratio >= 0.60
                    {
                        "two_column"
                    } else {
                        "margin_column"
                    },
                    split_x,
                    left_count: left_lines.len(),
                    right_count: right_lines.len(),
                },
            ))
        })
        .min_by(|left, right| {
            left.0
                .cmp(&right.0)
                .then(left.1.cmp(&right.1))
                .then_with(|| right.2.total_cmp(&left.2))
        })
        .map(|(_, _, _, model)| model)
        .unwrap_or(single)
}

fn note_column_model(lines: &[Line], page_width: f64) -> ColumnModel {
    let mut model = column_model(lines, page_width);
    if model.kind != "two_column" {
        return model;
    }
    let crossing_prose = lines
        .iter()
        .filter(|line| {
            has_valid_bbox(line)
                && line
                    .text
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .count()
                    >= 8
                && line.bbox[0] < model.split_x
                && line.bbox[2] > model.split_x
        })
        .count();
    if crossing_prose > 0 {
        model.kind = "margin_column";
    }
    model
}

fn geometry_order(lines: &mut [Line]) {
    lines.sort_by(|left, right| {
        line_center_y(left)
            .total_cmp(&line_center_y(right))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

pub(super) fn column_order(lines: &mut [Line], split_x: f64) {
    let spans = |line: &Line| {
        let width = line_width(line);
        width > 0.0
            && line.bbox[0] <= split_x - width * 0.20
            && line.bbox[2] >= split_x + width * 0.20
    };
    let bounds = |right: bool| {
        let mut values = lines
            .iter()
            .filter(|line| {
                has_valid_bbox(line) && !spans(line) && (line_center_x(line) >= split_x) == right
            })
            .map(line_center_y);
        let first = values.next()?;
        Some(values.fold((first, first), |(top, bottom), value| {
            (top.min(value), bottom.max(value))
        }))
    };
    let Some((left, right)) = bounds(false).zip(bounds(true)) else {
        geometry_order(lines);
        return;
    };
    let overlap = (left.0.max(right.0), left.1.min(right.1));
    let mut anchors: Vec<_> = lines
        .iter()
        .filter(|line| spans(line) && (overlap.0..=overlap.1).contains(&line_center_y(line)))
        .map(line_center_y)
        .collect();
    anchors.sort_by(f64::total_cmp);
    lines.sort_by(|left, right| {
        let left_y = line_center_y(left);
        let right_y = line_center_y(right);
        let band = |y: f64| usize::from(y >= overlap.0) + usize::from(y > overlap.1);
        let left_band = band(left_y);
        let right_band = band(right_y);
        let band_order = left_band.cmp(&right_band);
        if band_order != std::cmp::Ordering::Equal || left_band != 1 {
            return band_order
                .then(left_y.total_cmp(&right_y))
                .then(left.bbox[0].total_cmp(&right.bbox[0]))
                .then(left.id.cmp(&right.id));
        }
        let left_anchor = spans(left);
        let right_anchor = spans(right);
        let left_segment = anchors.partition_point(|anchor| *anchor < left_y);
        let right_segment = anchors.partition_point(|anchor| *anchor < right_y);
        let left_column = usize::from(line_center_x(left) >= split_x);
        let right_column = usize::from(line_center_x(right) >= split_x);
        left_segment
            .cmp(&right_segment)
            .then(left_anchor.cmp(&right_anchor))
            .then_with(|| {
                if left_anchor {
                    std::cmp::Ordering::Equal
                } else {
                    left_column.cmp(&right_column)
                }
            })
            .then(left_y.total_cmp(&right_y))
            .then(line_center_x(left).total_cmp(&line_center_x(right)))
            .then(left.bbox[0].total_cmp(&right.bbox[0]))
            .then(left.id.cmp(&right.id))
    });
}

fn column_switches(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut switches = 0;
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_some_and(|prior| prior != column) {
            switches += 1;
        }
        previous = Some(column);
    }
    switches
}

fn median_column_run(lines: &[Line], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut current = 0;
    let mut runs = Vec::new();
    for line in lines {
        if !has_valid_bbox(line) || line_width(line) / page_width > 0.55 {
            continue;
        }
        let column = line_center_x(line) >= model.split_x;
        if previous.is_none_or(|prior| prior == column) {
            current += 1;
        } else {
            runs.push(current);
            current = 1;
        }
        previous = Some(column);
    }
    if current > 0 {
        runs.push(current);
    }
    runs.sort_unstable();
    runs.get(runs.len() / 2).copied().unwrap_or(0)
}

fn hyphen_join_score(lines: &[Line]) -> (usize, usize) {
    let mut candidates = 0;
    let mut satisfied = 0;
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim_end();
        let Some(last) = text
            .chars()
            .last()
            .filter(|character| matches!(character, '-' | '\u{00ad}' | '\u{00ac}'))
        else {
            continue;
        };
        let prefix = &text[..text.len() - last.len_utf8()];
        if prefix
            .chars()
            .rev()
            .take_while(|character| character.is_ascii_alphabetic())
            .count()
            < 2
        {
            continue;
        }
        candidates += 1;
        if lines.get(index + 1).is_some_and(|next| {
            next.text
                .trim_start()
                .chars()
                .take_while(|character| character.is_ascii_alphabetic())
                .count()
                >= 2
        }) {
            satisfied += 1;
        }
    }
    (candidates, satisfied)
}

fn y_regressions(lines: &[Line]) -> usize {
    let boxed: Vec<&Line> = lines.iter().filter(|line| has_valid_bbox(line)).collect();
    let tolerance = p50(boxed
        .iter()
        .map(|line| (line.bbox[3] - line.bbox[1]).max(0.0))
        .collect());
    boxed
        .windows(2)
        .filter(|pair| line_center_y(pair[1]) < line_center_y(pair[0]) - tolerance)
        .count()
}

pub(super) fn arbitrate_body_order(
    lines: &mut [Line],
    page_width: f64,
    page_height: f64,
) -> OrderDecision {
    let boxed = lines.iter().filter(|line| has_valid_bbox(line)).count();
    if lines.len() < 8 || boxed * 5 < lines.len() * 4 || page_width <= 0.0 || page_height <= 0.0 {
        return OrderDecision::keep("insufficient_geometry");
    }
    if has_table_caption(lines) {
        return OrderDecision::keep("table_grid");
    }
    let mut model = column_model(lines, page_width);
    if model.kind != "two_column" {
        let alternative = column_model_with_furniture(lines, page_width, true);
        if alternative.kind == "two_column" {
            model = alternative;
        }
    }
    if model.kind == "two_column" {
        let source_switches = column_switches(lines, model, page_width);
        let source_run = median_column_run(lines, model, page_width);
        let source_hyphens = hyphen_join_score(lines);
        let mut challenger = lines.to_vec();
        column_order(&mut challenger, model.split_x);
        let challenger_switches = column_switches(&challenger, model, page_width);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let minimum_switches = if source_hyphens.0 > 0 { 3 } else { 5 }
            .max(((model.left_count + model.right_count) as f64 * 0.10) as usize);
        if source_switches >= minimum_switches
            && source_run <= 6
            && challenger_switches <= 2
            && source_switches.saturating_sub(challenger_switches) >= 2
            && challenger_hyphens.1 >= source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                <= source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Column,
                source_switches,
                strategy: "column-geometry",
                reason: "column_interleave_repair",
            };
        }
        return OrderDecision {
            repair: OrderRepair::None,
            source_switches,
            strategy: "kraken-native",
            reason: "two_column_kraken_coherent",
        };
    }
    if model.kind == "single" {
        let source_regressions = y_regressions(lines);
        let source_hyphens = hyphen_join_score(lines);
        let mut challenger = lines.to_vec();
        geometry_order(&mut challenger);
        let challenger_regressions = y_regressions(&challenger);
        let challenger_hyphens = hyphen_join_score(&challenger);
        let threshold = 3.max((boxed as f64 * 0.08) as usize);
        if source_regressions >= threshold
            && challenger_regressions <= source_regressions / 3
            && challenger_hyphens.1 > source_hyphens.1
            && challenger_hyphens.0.saturating_sub(challenger_hyphens.1)
                < source_hyphens.0.saturating_sub(source_hyphens.1)
        {
            lines.clone_from_slice(&challenger);
            return OrderDecision {
                repair: OrderRepair::Geometry,
                source_switches: 0,
                strategy: "geometry",
                reason: "kraken_order_scrambled",
            };
        }
    }
    OrderDecision::keep(if model.kind == "single" {
        "single_column_kraken_coherent"
    } else {
        "non_two_column"
    })
}

pub(super) fn repair_drop_caps(lines: &mut Vec<Line>) {
    let body_size = p50(lines
        .iter()
        .map(line_font_size)
        .filter(|size| (4.0..=24.0).contains(size))
        .collect());
    if body_size <= 0.0 {
        return;
    }
    let mut moves = Vec::new();
    for (index, line) in lines.iter().enumerate() {
        let text = line.text.trim();
        if text.chars().count() != 1
            || !text.chars().all(char::is_alphabetic)
            || line_font_size(line) < body_size * 1.8
        {
            continue;
        }
        let target = lines
            .iter()
            .enumerate()
            .filter(|(other, candidate)| {
                *other != index
                    && line_font_size(candidate) <= body_size * 1.4
                    && candidate.bbox[0] >= line.bbox[2] - body_size
                    && candidate.bbox[0] - line.bbox[2] <= body_size * 3.0
                    && candidate.bbox[1] < line.bbox[3]
                    && candidate.bbox[3] > line.bbox[1]
            })
            .min_by(|(_, left), (_, right)| {
                left.bbox[1]
                    .total_cmp(&right.bbox[1])
                    .then(left.bbox[0].total_cmp(&right.bbox[0]))
            })
            .map(|(target, _)| target);
        if let Some(target) = target.filter(|target| *target < index) {
            moves.push((index, target));
        }
    }
    for (index, target) in moves.into_iter().rev() {
        let line = lines.remove(index);
        lines.insert(target, line);
    }
}

pub(super) fn band_geometry_top(line: &Line) -> f64 {
    line.words
        .iter()
        .map(|word| word.bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(line.bbox[1])
}

fn band_geometry_order(left: &Line, right: &Line) -> std::cmp::Ordering {
    band_geometry_top(left)
        .total_cmp(&band_geometry_top(right))
        .then(left.bbox[0].total_cmp(&right.bbox[0]))
}

pub(super) fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

pub(super) fn aligned_note_body(label: &Line, body: &Line) -> bool {
    if standalone_note_label(body) || body.bbox[0] < label.bbox[0] {
        return false;
    }
    let overlap = label.bbox[3].min(body.bbox[3]) - label.bbox[1].max(body.bbox[1]);
    let minimum_height = (label.bbox[3] - label.bbox[1]).min(body.bbox[3] - body.bbox[1]);
    overlap > 0.0 && minimum_height > 0.0 && overlap / minimum_height >= 0.5
}

pub(super) fn aligned_note_body_index(lines: &[Line], label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    lines
        .iter()
        .enumerate()
        .filter(|(_, body)| aligned_note_body(label, body))
        .min_by(|(_, left), (_, right)| {
            (left.bbox[0] - label.bbox[2])
                .max(0.0)
                .total_cmp(&(right.bbox[0] - label.bbox[2]).max(0.0))
                .then(
                    (left.bbox[1] - label.bbox[1])
                        .abs()
                        .total_cmp(&(right.bbox[1] - label.bbox[1]).abs()),
                )
        })
        .map(|(index, _)| index)
}

pub(super) fn order_note_lines(lines: &mut Vec<Line>, page_width: f64) {
    let mut tops: Vec<f64> = lines.iter().map(band_geometry_top).collect();
    for (label_index, label) in lines.iter().enumerate() {
        if !standalone_note_label(label) {
            continue;
        }
        if let Some(body_index) = aligned_note_body_index(lines, label_index) {
            tops[label_index] = tops[body_index];
        }
    }
    let columns = note_column_model(lines, page_width);
    let mut keyed: Vec<_> = std::mem::take(lines).into_iter().zip(tops).collect();
    keyed.sort_by(|(left, left_top), (right, right_top)| {
        let left_column =
            usize::from(columns.kind == "two_column" && line_center_x(left) >= columns.split_x);
        let right_column =
            usize::from(columns.kind == "two_column" && line_center_x(right) >= columns.split_x);
        left_column.cmp(&right_column).then(
            left_top
                .total_cmp(right_top)
                .then(left.bbox[0].total_cmp(&right.bbox[0])),
        )
    });
    lines.extend(keyed.into_iter().map(|(line, _)| line));
}

pub(super) fn weave_note_columns(
    mut body: Vec<Line>,
    note: Vec<Line>,
    page_width: f64,
) -> Vec<Line> {
    let note_columns = note_column_model(&note, page_width);
    let body_model = column_model(&body, page_width);
    let split_x = if note_columns.kind == "two_column" {
        note_columns.split_x
    } else if body_model.kind == "two_column" {
        body_model.split_x
    } else {
        body.extend(note);
        return body;
    };
    let note_sides = note
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if note_sides.contains(&0) {
        body.extend(note);
        return body;
    }
    let (left_notes, right_notes): (Vec<_>, Vec<_>) = note
        .into_iter()
        .partition(|line| line_center_x(line) < split_x);
    let body_columns = body
        .iter()
        .filter(|line| line_width(line) / page_width <= 0.55)
        .fold([0_usize; 2], |mut counts, line| {
            counts[usize::from(line_center_x(line) >= split_x)] += 1;
            counts
        });
    if left_notes.is_empty()
        || right_notes.is_empty()
        || body_columns.iter().any(|count| *count < 3)
    {
        body.extend(left_notes);
        body.extend(right_notes);
        return body;
    }
    let insert_at = body
        .iter()
        .rposition(|line| line_width(line) / page_width <= 0.55 && line_center_x(line) < split_x)
        .map_or(0, |index| index + 1);
    let right_body = body.split_off(insert_at);
    body.extend(left_notes);
    body.extend(right_body);
    body.extend(right_notes);
    body
}

pub(super) fn order_page(
    page: &mut Page,
    table_page: bool,
    table_notes: &HashSet<usize>,
) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut header = Vec::new();
    let mut body = Vec::new();
    let mut note = Vec::new();
    let mut footer = Vec::new();
    for (index, line) in std::mem::take(&mut page.lines).into_iter().enumerate() {
        match line.region_type.as_str() {
            "header" => header.push(line),
            "footnote" if table_notes.contains(&index) => body.push(line),
            "footnote" => note.push(line),
            "footer" => footer.push(line),
            _ => body.push(line),
        }
    }
    header.sort_by(band_geometry_order);
    repair_drop_caps(&mut body);
    let contents = table_page && !has_table_caption(&body) && contents_grid(&body, page.width);
    let decision = if contents && y_regressions(&body) > 0 {
        geometry_order(&mut body);
        OrderDecision {
            repair: OrderRepair::Geometry,
            source_switches: 0,
            strategy: "table-row-geometry",
            reason: "table_source_order_scrambled",
        }
    } else if table_page {
        OrderDecision::keep("table_grid")
    } else {
        arbitrate_body_order(&mut body, page.width, page.height)
    };
    if decision.repair != OrderRepair::None {
        let mut diagnostic = Diagnostic::info(
            "COLUMN_ORDER_REPAIRED",
            format!(
                "Extraction order replaced by {}: {}.",
                decision.strategy, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    } else if decision.source_switches > 2 {
        let mut diagnostic = Diagnostic::warning(
            "COLUMN_ORDER_UNCERTAIN",
            format!(
                "Two-column page keeps an order that crosses columns {} times ({}).",
                decision.source_switches, decision.reason
            ),
            Some(page.index),
        );
        diagnostic.line_ids = body.iter().take(20).map(|line| line.id.clone()).collect();
        diagnostics.push(diagnostic);
    }
    order_note_lines(&mut note, page.width);
    footer.sort_by(band_geometry_order);
    let mut ordered = header;
    ordered.extend(weave_note_columns(body, note, page.width));
    ordered.extend(footer);
    for (index, line) in ordered.iter_mut().enumerate() {
        line.reading_order = index + 1;
    }
    page.lines = ordered;
    diagnostics
}

#[cfg(test)]
pub(super) fn order_pages(pages: &mut [Page]) -> Vec<Diagnostic> {
    pages
        .iter_mut()
        .flat_map(|page| {
            let evidence = table_evidence(&page.lines, page.width);
            let table_page =
                has_table_caption(&page.lines) || strong_table_evidence(&evidence, &page.lines);
            order_page(page, table_page, &HashSet::new())
        })
        .collect()
}

pub(super) fn build_regions(pages: &mut [Page]) {
    for page in pages {
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for index in 0..page.lines.len() {
            if groups.last().is_some_and(|group| {
                let prior = &page.lines[*group.last().expect("non-empty group")];
                prior.region_type == page.lines[index].region_type
                    && prior.block_index == page.lines[index].block_index
            }) {
                groups.last_mut().expect("group exists").push(index);
            } else {
                groups.push(vec![index]);
            }
        }
        page.regions.clear();
        for (region_index, indexes) in groups.into_iter().enumerate() {
            let id = format!("p{:04}-r{:04}", page.number, region_index + 1);
            let kind = page.lines[indexes[0]].region_type.clone();
            let line_ids = indexes
                .iter()
                .map(|&index| page.lines[index].id.clone())
                .collect();
            let bbox = union_bbox(indexes.iter().map(|&index| page.lines[index].bbox));
            let reading_order = indexes
                .iter()
                .map(|&index| page.lines[index].reading_order)
                .min()
                .unwrap_or(0);
            for &index in &indexes {
                page.lines[index].region_id.clone_from(&id);
            }
            page.regions.push(Region {
                id,
                page_index: page.index,
                kind,
                line_ids,
                bbox,
                reading_order,
            });
        }
    }
}
