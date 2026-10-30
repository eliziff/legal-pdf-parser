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

/// The line among the lines `among` names (slots of `lines`) that a label printed apart from its
/// note's text stands beside: on the label's row, at or right of it, nearest it.
pub(crate) fn aligned_note_body_among(lines: &[Line], among: impl IntoIterator<Item = usize>, label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    among
        .into_iter()
        .map(|index| (index, &lines[index]))
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

/// A page's note band (`note`, slots of `lines`) in reading order: column by column, top to
/// bottom, a label set apart at its note's first line. `items` are the page's (on page `number`).
pub(crate) fn order_note_lines(lines: &[Line], note: &mut Vec<usize>, items: &legal_pdf_tables::PageItems, number: u32) {
    let mut tops: Vec<f64> = note.iter().map(|&slot| band_geometry_top(&lines[slot])).collect();
    for (at, &label) in note.iter().enumerate() {
        if !standalone_note_label(&lines[label]) {
            continue;
        }
        if let Some(body) = aligned_note_body_among(lines, note.iter().copied(), label) {
            tops[at] = tops[note.iter().position(|&slot| slot == body).unwrap_or(at)];
        }
    }
    let columns = note_column_model(lines, note, items, number);
    let mut keyed: Vec<(usize, f64)> = note.iter().copied().zip(tops).collect();
    keyed.sort_by(|&(left, left_top), &(right, right_top)| {
        let (left, right) = (&lines[left], &lines[right]);
        let left_column =
            usize::from(columns.kind == "two_column" && line_center_x(left) >= columns.split_x);
        let right_column =
            usize::from(columns.kind == "two_column" && line_center_x(right) >= columns.split_x);
        left_column.cmp(&right_column).then(
            left_top
                .total_cmp(&right_top)
                .then(left.bbox[0].total_cmp(&right.bbox[0])),
        )
    });
    *note = keyed.into_iter().map(|(slot, _)| slot).collect();
}

/// A page's body and note band (slots of `lines`) read together: on a page of two columns, each
/// column's notes after that column's body. `items` are the page's (on page `number`).
pub(crate) fn weave_note_columns(
    lines: &[Line],
    mut body: Vec<usize>,
    note: Vec<usize>,
    items: &legal_pdf_tables::PageItems,
    page_width: f64,
    number: u32,
) -> Vec<usize> {
    if note.is_empty() {
        return body;
    }
    let note_columns = note_column_model(lines, &note, items, number);
    let body_model = {
        let mut kept = vec![false; lines.len()];
        for &slot in &body {
            kept[slot] = true;
        }
        columns_of(lines, body.iter().copied(), &kept_items(items, |slot| kept[slot]), number)
    };
    let split_x = if note_columns.kind == "two_column" {
        note_columns.split_x
    } else if body_model.kind == "two_column" {
        body_model.split_x
    } else {
        body.extend(note);
        return body;
    };
    let narrow = |slot: &&usize| line_width(&lines[**slot]) / page_width <= 0.55;
    let side = |slot: usize| usize::from(line_center_x(&lines[slot]) >= split_x);
    let note_sides = note.iter().filter(narrow).fold([0_usize; 2], |mut counts, &slot| {
        counts[side(slot)] += 1;
        counts
    });
    if note_sides.contains(&0) {
        body.extend(note);
        return body;
    }
    let (left_notes, right_notes): (Vec<_>, Vec<_>) = note.into_iter().partition(|&slot| side(slot) == 0);
    let body_columns = body.iter().filter(narrow).fold([0_usize; 2], |mut counts, &slot| {
        counts[side(slot)] += 1;
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
        .rposition(|&slot| line_width(&lines[slot]) / page_width <= 0.55 && side(slot) == 0)
        .map_or(0, |index| index + 1);
    let right_body = body.split_off(insert_at);
    body.extend(left_notes);
    body.extend(right_body);
    body.extend(right_notes);
    body
}
