//! A page's columns, as pdf-inspector reads them, and the margin column a page sets its notes in.

use super::*;

/// The text items of the page's lines `keep` holds, taken from the page's items (`page_items`),
/// each carrying its line's slot on the page (as its `mcid`).
pub(crate) fn kept_items(items: &legal_pdf_tables::PageItems, keep: impl Fn(usize) -> bool) -> Vec<pdf_inspector::TextItem> {
    items
        .items
        .iter()
        .zip(&items.origin)
        .filter(|(_, (line, _))| keep(*line))
        .map(|(item, (line, _))| pdf_inspector::TextItem { mcid: Some(*line as i64), ..item.clone() })
        .collect()
}

/// The page's columns as pdf-inspector finds them in the words of the lines `members` names (slots
/// of `lines`), read on their text items on page `number` (`detect_columns`, a projection of the
/// words across the page whose empty valleys are the gutters): two side by side, set apart at the
/// gutter between them, a pair whose narrower is under half the wider's width being a body and its
/// margin column (pdf-inspector's sidebar); more than two, a multi-column page.
pub(crate) fn columns_of(
    lines: &[Line],
    members: impl Iterator<Item = usize>,
    items: &[pdf_inspector::TextItem],
    number: u32,
) -> ColumnModel {
    let regions = pdf_inspector::extractor::detect_columns(items, number, false);
    let [left, right, ..] = regions.as_slice() else {
        return ColumnModel { kind: "single", split_x: 0.0, left_count: 0, right_count: 0 };
    };
    let split_x = f64::from(left.x_max + right.x_min) / 2.0;
    let (mut left_count, mut right_count) = (0, 0);
    for line in members.map(|slot| &lines[slot]).filter(|line| has_valid_bbox(line) && !line.exclude_from_body) {
        if line_center_x(line) < split_x {
            left_count += 1;
        } else {
            right_count += 1;
        }
    }
    let (left_width, right_width) = (left.x_max - left.x_min, right.x_max - right.x_min);
    let kind = if regions.len() > 2 {
        "multi_column"
    } else if left_width.min(right_width) >= 0.5 * left_width.max(right_width) {
        "two_column"
    } else {
        "margin_column"
    };
    ColumnModel { kind, split_x, left_count, right_count }
}

/// The lines `members` names (slots of the page's `count` lines), in the order pdf-inspector reads
/// their words' items (`group_into_lines_preserving_all_text`: column by column, banded where the
/// columns change down the page, a newspaper's columns read through); a line it was not offered
/// stays after the line it followed.
pub(crate) fn inspected_order(members: &[usize], items: Vec<pdf_inspector::TextItem>, count: usize) -> Vec<usize> {
    let mut place = vec![usize::MAX; count];
    for (at, &slot) in members.iter().enumerate() {
        place[slot] = at;
    }
    let mut placed = vec![false; members.len()];
    let mut order = Vec::with_capacity(members.len());
    for line in pdf_inspector::extractor::group_into_lines_preserving_all_text(items) {
        for at in line.items.iter().filter_map(|item| item.mcid).filter_map(|slot| place.get(slot as usize).copied()) {
            if at < members.len() && !std::mem::replace(&mut placed[at], true) {
                order.push(at);
            }
        }
    }
    for at in (0..members.len()).filter(|&at| !placed[at]) {
        let after = (0..at).rev().find_map(|prior| order.iter().position(|&placed| placed == prior)).map_or(0, |found| found + 1);
        order.insert(after, at);
    }
    order.into_iter().map(|at| members[at]).collect()
}

/// The margin column a page's note labels (`labels`, slots of `lines`) stand in beside its prose:
/// at least `minimum_labels` of them on one side, apart from the body lines of `members` on their
/// rows that `set_aside` leaves (running heads and feet, lines held out of the body).
pub(crate) fn margin_note_model(
    lines: &[Line],
    members: &[usize],
    set_aside: impl Fn(usize) -> bool,
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
                    if let Some(body) = aligned_note_body_among(lines, members.iter().copied(), *index) {
                        note_left = note_left.min(lines[body].bbox[0]);
                        note_right = note_right.max(lines[body].bbox[2]);
                        note_top = note_top.min(lines[body].bbox[1]);
                        note_bottom = note_bottom.max(lines[body].bbox[3]);
                    }
                }
            }
            let prose: Vec<_> = members
                .iter()
                .map(|&index| (index, &lines[index]))
                .filter(|(index, line)| {
                    !label_set.contains(index)
                        && has_valid_bbox(line)
                        && !set_aside(*index)
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

/// The columns of a page's note band (`members`, slots of `lines`), read on the page's `items`
/// (on page `number`): two columns a line of prose runs across are a body and its margin.
pub(crate) fn note_column_model(lines: &[Line], members: &[usize], items: &legal_pdf_tables::PageItems, number: u32) -> ColumnModel {
    let mut kept = vec![false; lines.len()];
    for &slot in members {
        kept[slot] = true;
    }
    let mut model = columns_of(lines, members.iter().copied(), &kept_items(items, |slot| kept[slot]), number);
    if model.kind != "two_column" {
        return model;
    }
    let crossing_prose = members
        .iter()
        .map(|&slot| &lines[slot])
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

pub(crate) fn column_switches(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut switches = 0;
    for line in members.iter().map(|&slot| &lines[slot]) {
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

pub(crate) fn median_column_run(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> usize {
    let mut previous = None;
    let mut current = 0;
    let mut runs = Vec::new();
    for line in members.iter().map(|&slot| &lines[slot]) {
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
