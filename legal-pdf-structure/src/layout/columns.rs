//! A page's columns, as pdf-inspector reads them, and the margin column a page sets its notes in.

use super::*;

/// The text items of the page's lines `keep` holds, taken from the page's items (`page_items`),
/// each carrying its line's slot on the page (as its `mcid`).
pub fn kept_items(items: &crate::tables::PageItems, keep: impl Fn(usize) -> bool) -> Vec<pdf_inspector::TextItem> {
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
/// margin column (pdf-inspector's sidebar); more than two, a multi-column page. A lane whose lines are all bare
/// numbers is no column: a report's paragraph numbers printed in its margin.
pub fn columns_of(lines: &[Line], members: impl Iterator<Item = usize>, items: &[pdf_inspector::TextItem], number: u32) -> ColumnModel {
    let members: Vec<&Line> = members.map(|slot| &lines[slot]).filter(|line| has_valid_bbox(line) && !line.exclude_from_body).collect();
    let regions: Vec<_> = pdf_inspector::extractor::detect_columns(items, number, false)
        .into_iter()
        .filter(|region| {
            let inside = |line: &&&Line| (f64::from(region.x_min)..=f64::from(region.x_max)).contains(&line_center_x(line));
            !members.iter().filter(inside).all(|line| bare_number(line))
        })
        .collect();
    let [left, right, ..] = regions.as_slice() else {
        return ColumnModel { kind: ColumnKind::Single, split_x: 0.0, left_count: 0, right_count: 0 };
    };
    let split_x = f64::from(left.x_max + right.x_min) / 2.0;
    let left_count = members.iter().filter(|line| line_center_x(line) < split_x).count();
    let (left_width, right_width) = (left.x_max - left.x_min, right.x_max - right.x_min);
    let kind = if regions.len() > 2 {
        ColumnKind::Multi
    } else if left_width.min(right_width) >= 0.5 * left_width.max(right_width) {
        ColumnKind::Two
    } else {
        ColumnKind::Margin
    };
    ColumnModel { kind, split_x, left_count, right_count: members.len() - left_count }
}

/// A number standing alone on its line ("24", "[24]"): a report's paragraph number when it stands in the margin.
pub fn bare_number(line: &Line) -> bool {
    let text = line.text.trim();
    let text = text.strip_prefix('[').and_then(|rest| rest.strip_suffix(']')).unwrap_or(text);
    !text.is_empty() && text.len() <= 4 && text.bytes().all(|byte| byte.is_ascii_digit())
}

/// The body lines (`body`, slots of `lines`) that are bare numbers clear of the body's prose on either side: a law
/// report's margin numbers, a lane of their own that is no column of the page's text.
pub fn margin_lane(lines: &[Line], body: &[usize], page_width: f64) -> Vec<usize> {
    let prose: Vec<&Line> = body
        .iter()
        .map(|&slot| &lines[slot])
        .filter(|line| has_valid_bbox(line) && line_width(line) > page_width * 0.20 && line.text.chars().any(char::is_alphabetic))
        .collect();
    let (Some(left), Some(right)) = (prose.iter().map(|line| line.bbox[0]).min_by(f64::total_cmp), prose.iter().map(|line| line.bbox[2]).max_by(f64::total_cmp))
    else {
        return Vec::new();
    };
    let lane = |line: &Line| has_valid_bbox(line) && bare_number(line) && (line.bbox[2] < left || line.bbox[0] > right);
    body.iter().copied().filter(|&slot| lane(&lines[slot])).collect()
}

/// The lines `members` names (slots of the page's `count` lines), in the order pdf-inspector reads
/// their words' items (`group_into_lines_preserving_all_text`: column by column, banded where the
/// columns change down the page, a newspaper's columns read through); a line it was not offered
/// stays after the line it followed.
pub fn inspected_order(members: &[usize], items: Vec<pdf_inspector::TextItem>, count: usize) -> Vec<usize> {
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

/// The box covering `bbox` and the box `area` (left, top, right, bottom).
fn grow(area: [f64; 4], bbox: [f64; 4]) -> [f64; 4] {
    [area[0].min(bbox[0]), area[1].min(bbox[1]), area[2].max(bbox[2]), area[3].max(bbox[3])]
}

/// The margin column a page's note labels (`labels`, slots of `lines`) stand in beside its prose:
/// at least `minimum_labels` of them on one side, apart from the body lines of `members` on their
/// rows that `set_aside` leaves (running heads and feet, lines held out of the body).
pub fn margin_note_model(
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
            let lane: Vec<_> = labels.iter().copied().filter(|index| (line_center_x(&lines[*index]) >= page_width / 2.0) == right).collect();
            if lane.len() < minimum_labels {
                return None;
            }
            let label_set: HashSet<_> = lane.iter().copied().collect();
            // The notes' box: each label's, with the note body set level with a label standing alone.
            let [note_left, note_top, note_right, note_bottom] = lane.iter().fold([page_width, f64::INFINITY, 0.0, 0.0], |area, &index| {
                let area = grow(area, lines[index].bbox);
                let body = standalone_note_label(&lines[index]).then(|| aligned_note_body_among(lines, members.iter().copied(), index)).flatten();
                body.map_or(area, |body| grow(area, lines[body].bbox))
            });
            let prose: Vec<_> = members
                .iter()
                .filter(|&&index| {
                    let line = &lines[index];
                    !label_set.contains(&index)
                        && has_valid_bbox(line)
                        && !set_aside(index)
                        && line_font_size(line) >= body_size * 0.90
                        && line_width(line) >= page_width * 0.25
                        && line.bbox[3] >= note_top
                        && line.bbox[1] <= note_bottom
                })
                .map(|&index| &lines[index])
                .collect();
            if prose.len() < 3 {
                return None;
            }
            let body_left = prose.iter().map(|line| line.bbox[0]).min_by(f64::total_cmp)?;
            let body_right = prose.iter().map(|line| line.bbox[2]).max_by(f64::total_cmp)?;
            let gap = (body_size * 1.5).max(page_width * 0.02);
            let split_x = if note_right + gap <= body_left {
                (note_right + body_left) / 2.0
            } else if body_right + gap <= note_left {
                (body_right + note_left) / 2.0
            } else {
                return None;
            };
            let (left_count, right_count) = if right { (prose.len(), lane.len()) } else { (lane.len(), prose.len()) };
            Some((lane.len(), ColumnModel { kind: ColumnKind::Margin, split_x, left_count, right_count }))
        })
        .max_by_key(|(count, _)| *count)
        .map(|(_, model)| model)
}

/// The columns of a page's note band (`members`, slots of `lines`), read on the page's `items`
/// (on page `number`): two columns a line of prose runs across are a body and its margin.
pub fn note_column_model(lines: &[Line], members: &[usize], items: &crate::tables::PageItems, number: u32) -> ColumnModel {
    let mut kept = vec![false; lines.len()];
    for &slot in members {
        kept[slot] = true;
    }
    let mut model = columns_of(lines, members.iter().copied(), &kept_items(items, |slot| kept[slot]), number);
    let crossing = |line: &Line| {
        has_valid_bbox(line) && line.text.chars().filter(|character| !character.is_whitespace()).count() >= 8 && line.bbox[0] < model.split_x && line.bbox[2] > model.split_x
    };
    if model.kind == ColumnKind::Two && members.iter().any(|&slot| crossing(&lines[slot])) {
        model.kind = ColumnKind::Margin;
    }
    model
}

/// The side of the column split each of the lines (`members`, slots of `lines`) stands on, the
/// lines without a box or running across the page aside.
fn column_sides(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> Vec<bool> {
    members
        .iter()
        .map(|&slot| &lines[slot])
        .filter(|line| has_valid_bbox(line) && !(line_width(line) / page_width > 0.55))
        .map(|line| line_center_x(line) >= model.split_x)
        .collect()
}

pub fn column_switches(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> usize {
    column_sides(lines, members, model, page_width).windows(2).filter(|pair| pair[0] != pair[1]).count()
}

pub fn median_column_run(lines: &[Line], members: &[usize], model: ColumnModel, page_width: f64) -> usize {
    let sides = column_sides(lines, members, model, page_width);
    let mut runs: Vec<usize> = sides.chunk_by(|left, right| left == right).map(<[bool]>::len).collect();
    runs.sort_unstable();
    runs.get(runs.len() / 2).copied().unwrap_or(0)
}
