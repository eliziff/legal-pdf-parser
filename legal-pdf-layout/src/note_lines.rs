//! A note band's lines: where a band begins, a label set apart from its note's text, and the order
//! of a band set in columns.

use super::*;

pub fn band_geometry_top(line: &Line) -> f64 {
    line.words.iter().map(|word| word.bbox[1]).min_by(f64::total_cmp).unwrap_or(line.bbox[1])
}

/// Lines (`members`, slots of `lines`) in rows, top to bottom, each row left to right: taken by
/// `top`, a line joins the row whose middle its own sits within half the lines' median height of,
/// so a skewed scan's row (each word a line, the tops rising across the page) reads left to right.
pub fn read_rows(lines: &[Line], members: &mut Vec<usize>, top: impl Fn(usize) -> f64) {
    let height = |slot: usize| (lines[slot].bbox[3] - lines[slot].bbox[1]).max(0.0);
    let half = median(members.iter().map(|&slot| height(slot)).collect()) / 2.0;
    let mut keyed: Vec<(usize, f64)> = members.iter().map(|&slot| (slot, top(slot))).collect();
    keyed.sort_by(|left, right| left.1.total_cmp(&right.1).then(lines[left.0].bbox[0].total_cmp(&lines[right.0].bbox[0])));
    let mut rows: Vec<(f64, Vec<usize>)> = Vec::new();
    for (slot, top) in keyed {
        let middle = top + height(slot) / 2.0;
        match rows.last_mut() {
            Some((sum, row)) if (middle - *sum / row.len() as f64).abs() <= half => {
                *sum += middle;
                row.push(slot);
            }
            _ => rows.push((middle, vec![slot])),
        }
    }
    let row = |(_, mut row): (f64, Vec<usize>)| {
        row.sort_by(|&left, &right| lines[left].bbox[0].total_cmp(&lines[right].bbox[0]));
        row
    };
    *members = rows.into_iter().flat_map(row).collect();
}

pub fn standalone_note_label(line: &Line) -> bool {
    let text = line.text.trim();
    let length = text.chars().count();
    ((1..=MAX_SYMBOL_LABEL_LEN).contains(&length) && text.chars().all(is_note_symbol))
        || ((1..=4).contains(&length) && text.chars().all(|character| character.is_ascii_digit()))
}

pub fn aligned_note_body(label: &Line, body: &Line) -> bool {
    if standalone_note_label(body) || body.bbox[0] < label.bbox[0] {
        return false;
    }
    let overlap = label.bbox[3].min(body.bbox[3]) - label.bbox[1].max(body.bbox[1]);
    let minimum_height = (label.bbox[3] - label.bbox[1]).min(body.bbox[3] - body.bbox[1]);
    overlap > 0.0 && minimum_height > 0.0 && overlap / minimum_height >= 0.5
}

/// The line among the lines `among` names (slots of `lines`) that a label printed apart from its
/// note's text stands beside: on the label's row, at or right of it, nearest it.
pub fn aligned_note_body_among(lines: &[Line], among: impl IntoIterator<Item = usize>, label_index: usize) -> Option<usize> {
    let label = &lines[label_index];
    let (gap, drop) = (|body: &Line| (body.bbox[0] - label.bbox[2]).max(0.0), |body: &Line| (body.bbox[1] - label.bbox[1]).abs());
    among
        .into_iter()
        .map(|index| (index, &lines[index]))
        .filter(|(_, body)| aligned_note_body(label, body))
        .min_by(|(_, left), (_, right)| gap(left).total_cmp(&gap(right)).then(drop(left).total_cmp(&drop(right))))
        .map(|(index, _)| index)
}

/// A page's note band (`note`, slots of `lines`) in reading order: column by column, row by row
/// (`read_rows`), a label set apart on its note's first line. `items` are the page's (on page `number`).
pub fn order_note_lines(lines: &[Line], note: &mut Vec<usize>, items: &legal_pdf_tables::PageItems, number: u32) {
    let mut tops: HashMap<usize, f64> = note.iter().map(|&slot| (slot, band_geometry_top(&lines[slot]))).collect();
    for &label in note.iter() {
        if !standalone_note_label(&lines[label]) {
            continue;
        }
        if let Some(body) = aligned_note_body_among(lines, note.iter().copied(), label) {
            tops.insert(label, tops[&body]);
        }
    }
    let columns = note_column_model(lines, note, items, number);
    let right = |slot: &usize| columns.kind == ColumnKind::Two && line_center_x(&lines[*slot]) >= columns.split_x;
    let (mut left_column, mut right_column): (Vec<usize>, Vec<usize>) = note.iter().copied().partition(|slot| !right(slot));
    read_rows(lines, &mut left_column, |slot| tops[&slot]);
    read_rows(lines, &mut right_column, |slot| tops[&slot]);
    left_column.extend(right_column);
    *note = left_column;
}

/// A page's body and note band (slots of `lines`) read together: on a page of two columns, each
/// column's notes after that column's body. `items` are the page's (on page `number`).
pub fn weave_note_columns(lines: &[Line], mut body: Vec<usize>, note: Vec<usize>, items: &legal_pdf_tables::PageItems, page_width: f64, number: u32) -> Vec<usize> {
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
