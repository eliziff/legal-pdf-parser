//! A note band's lines: where a band begins, a label set apart from its note's text, and the order
//! of a band set in columns.

use super::*;

pub(crate) fn band_geometry_top(line: &Line) -> f64 {
    line.words
        .iter()
        .map(|word| word.bbox[1])
        .min_by(f64::total_cmp)
        .unwrap_or(line.bbox[1])
}

pub(crate) fn band_geometry_order(left: &Line, right: &Line) -> std::cmp::Ordering {
    band_geometry_top(left)
        .total_cmp(&band_geometry_top(right))
        .then(left.bbox[0].total_cmp(&right.bbox[0]))
}

pub(crate) fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

pub(crate) fn aligned_note_body(label: &Line, body: &Line) -> bool {
    if standalone_note_label(body) || body.bbox[0] < label.bbox[0] {
        return false;
    }
    let overlap = label.bbox[3].min(body.bbox[3]) - label.bbox[1].max(body.bbox[1]);
    let minimum_height = (label.bbox[3] - label.bbox[1]).min(body.bbox[3] - body.bbox[1]);
    overlap > 0.0 && minimum_height > 0.0 && overlap / minimum_height >= 0.5
}

pub(crate) fn aligned_note_body_index(lines: &[Line], label_index: usize) -> Option<usize> {
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

pub(crate) fn order_note_lines(lines: &mut Vec<Line>) {
    let mut tops: Vec<f64> = lines.iter().map(band_geometry_top).collect();
    for (label_index, label) in lines.iter().enumerate() {
        if !standalone_note_label(label) {
            continue;
        }
        if let Some(body_index) = aligned_note_body_index(lines, label_index) {
            tops[label_index] = tops[body_index];
        }
    }
    let columns = note_column_model(lines);
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

pub(crate) fn weave_note_columns(
    mut body: Vec<Line>,
    note: Vec<Line>,
    page_width: f64,
) -> Vec<Line> {
    let note_columns = note_column_model(&note);
    let body_model = column_model(&body);
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
