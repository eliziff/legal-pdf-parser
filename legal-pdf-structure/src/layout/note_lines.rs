//! A note band's lines: where a band begins, a label set apart from its note's text, and the order
//! of a band set in columns.

use std::collections::HashMap;
use legal_pdf_core::Line;
use legal_structure_model::text::{is_note_symbol, MAX_SYMBOL_LABEL_LEN};
use super::{line_center_x, line_width, ColumnKind};
use super::columns::{columns_of, kept_items, note_column_model};

pub(crate) fn band_geometry_top(line: &Line) -> f64 {
    line.words.iter().map(|word| word.bbox[1]).min_by(f64::total_cmp).unwrap_or(line.bbox[1])
}

/// Lines (`members`, slots of `lines`) in rows, top to bottom by `top`, each row left to right: a row is the lines of
/// one of the page's rows (`rows`, each line's, `u32::MAX` for none, a row of its own).
pub(crate) fn read_rows(lines: &[Line], rows: &[u32], members: &mut Vec<usize>, top: impl Fn(usize) -> f64) {
    let mut keyed: Vec<(usize, f64)> = members.iter().map(|&slot| (slot, top(slot))).collect();
    keyed.sort_by(|left, right| left.1.total_cmp(&right.1).then(lines[left.0].bbox[0].total_cmp(&lines[right.0].bbox[0])));
    let mut read: Vec<Vec<usize>> = Vec::new();
    for (slot, _) in keyed {
        match read.iter_mut().find(|row| rows[slot] != u32::MAX && rows[row[0]] == rows[slot]) {
            Some(row) => row.push(slot),
            None => read.push(vec![slot]),
        }
    }
    for row in &mut read {
        row.sort_by(|&left, &right| lines[left].bbox[0].total_cmp(&lines[right].bbox[0]));
    }
    *members = read.into_iter().flatten().collect();
}

pub(crate) fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

/// Whether the line at `body` carries the text of the label at `label` printed apart from it: on the label's row (`rows`,
/// the page's, by line), at or right of it, and no label itself.
pub(crate) fn aligned_note_body(lines: &[Line], rows: &[u32], label: usize, body: usize) -> bool {
    !standalone_note_label(&lines[body]) && lines[body].bbox[0] >= lines[label].bbox[0] && rows[label] != u32::MAX && rows[label] == rows[body]
}

/// The line among the lines `among` names (slots of `lines`) that a label printed apart from its
/// note's text stands beside: on the label's row, at or right of it, nearest it.
pub(crate) fn aligned_note_body_among(lines: &[Line], rows: &[u32], among: impl IntoIterator<Item = usize>, label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    let (gap, drop) = (|body: &Line| (body.bbox[0] - label.bbox[2]).max(0.0), |body: &Line| (body.bbox[1] - label.bbox[1]).abs());
    among
        .into_iter()
        .map(|index| (index, &lines[index]))
        .filter(|&(index, _)| aligned_note_body(lines, rows, label_index, index))
        .min_by(|(_, left), (_, right)| gap(left).total_cmp(&gap(right)).then(drop(left).total_cmp(&drop(right))))
        .map(|(index, _)| index)
}

/// A page's note band (`note`, slots of `lines`) in reading order: column by column, row by row
/// (`read_rows`), a label set apart on its note's first line. `items` are the page's (on page `number`).
pub(crate) fn order_note_lines(lines: &[Line], rows: &[u32], note: &mut Vec<usize>, items: &crate::tables::PageItems, number: u32) {
    let mut tops: HashMap<usize, f64> = note.iter().map(|&slot| (slot, band_geometry_top(&lines[slot]))).collect();
    for &label in note.iter() {
        if !standalone_note_label(&lines[label]) {
            continue;
        }
        if let Some(body) = aligned_note_body_among(lines, rows, note.iter().copied(), label) {
            tops.insert(label, tops[&body]);
        }
    }
    let columns = note_column_model(lines, note, items, number);
    let right = |slot: &usize| columns.kind == ColumnKind::Two && line_center_x(&lines[*slot]) >= columns.split_x;
    let (mut left_column, mut right_column): (Vec<usize>, Vec<usize>) = note.iter().copied().partition(|slot| !right(slot));
    read_rows(lines, rows, &mut left_column, |slot| tops[&slot]);
    read_rows(lines, rows, &mut right_column, |slot| tops[&slot]);
    left_column.extend(right_column);
    *note = left_column;
}

/// A page's body and note band (slots of `lines`) read together: on a page of two columns, each
/// column's notes after that column's body. `items` are the page's (on page `number`).
pub(crate) fn weave_note_columns(lines: &[Line], mut body: Vec<usize>, note: Vec<usize>, items: &crate::tables::PageItems, page_width: f64, number: u32) -> Vec<usize> {
    if note.is_empty() {
        return body;
    }
    let note_columns = note_column_model(lines, &note, items, number);
    let mut kept = vec![false; lines.len()];
    for &slot in &body {
        kept[slot] = true;
    }
    let body_model = columns_of(lines, body.iter().copied(), &kept_items(items, |slot| kept[slot]), number);
    let split_x = if note_columns.kind == ColumnKind::Two {
        note_columns.split_x
    } else if body_model.kind == ColumnKind::Two {
        body_model.split_x
    } else {
        body.extend(note);
        return body;
    };
    let narrow = |slot: &&usize| line_width(&lines[**slot]) / page_width <= 0.55;
    let side = |slot: usize| usize::from(line_center_x(&lines[slot]) >= split_x);
    // How many of the narrow lines among `slots` stand on each side.
    let sides = |slots: &[usize]| {
        slots.iter().filter(narrow).fold([0_usize; 2], |mut counts, &slot| {
            counts[side(slot)] += 1;
            counts
        })
    };
    if sides(&note).contains(&0) {
        body.extend(note);
        return body;
    }
    let (left_notes, right_notes): (Vec<_>, Vec<_>) = note.into_iter().partition(|&slot| side(slot) == 0);
    if left_notes.is_empty() || right_notes.is_empty() || sides(&body).iter().any(|count| *count < 3) {
        body.extend(left_notes);
        body.extend(right_notes);
        return body;
    }
    let insert_at = body.iter().rposition(|&slot| line_width(&lines[slot]) / page_width <= 0.55 && side(slot) == 0).map_or(0, |index| index + 1);
    let right_body = body.split_off(insert_at);
    body.extend(left_notes);
    body.extend(right_body);
    body.extend(right_notes);
    body
}
