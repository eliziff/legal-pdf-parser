//! The document's tables and each line's cell, and the cells given back to the lines no fence took.
use crate::model::fences::Fences;
use crate::model::{Evidence, Role, Rule, Slot, Slots, WORD_SPEAKER};
use crate::model::PageItems;
use legal_pdf_core::model::Page;
use std::collections::HashSet;

/// The document's tables, the lines any of them holds, and the lines a contents list read as one holds.
/// The document's tables, the lines any of them holds, and the lines a contents list read as one holds.
pub fn find_tables(fences: &Fences, pages: &[Page],
    slots: &Slots,
    items: &[PageItems],
) -> (Vec<crate::model::tables::PdfTable>, Vec<bool>, HashSet<String>) {
    let (tables, listed) = legal_pdf_core::profile::measure("classify.tables", || {
        crate::tables::document_tables(pages, items, &|id| {
            slots
                .of_id(id)
                .is_some_and(|slot| fences.at(slot.index()).translation)
        })
    });
    let mut in_table = vec![false; slots.len()];
    for slot in tables
        .iter()
        .flat_map(|table| table.line_ids())
        .filter_map(|id| slots.of_id(id))
    {
        in_table[slot.index()] = true;
    }
    (tables, in_table, listed)
}

/// The tables a contents list does not hold for the most part, and their cells (`cell`); a cell
/// no fence took is the table's.
/// The tables a contents list does not hold for the most part, and their cells (`cell`); a cell
/// no fence took is the table's.
pub fn read_tables(fences: &mut Fences, slots: &Slots,
    tables: Vec<crate::model::tables::PdfTable>,
    listed: &[bool],
) {
    fences.tables = tables
        .into_iter()
        .filter(|table| {
            let ids = table.line_ids();
            let held = ids
                .iter()
                .filter(|id| slots.of_id(id).is_some_and(|slot| listed[slot.index()]))
                .count();
            held * 2 <= ids.len()
        })
        .collect();
    crate::find::number_cells(fences, slots);
    for slot in 0..slots.len() {
        if fences.cell[slot] != 0 {
            crate::find::set(fences, Slot(slot as u32), Role::Table, Rule::Grid);
        }
    }
}

/// A transcript's testimony beside its number column, read as a grid (a speaker's turn in
/// one of its cells), is no table's; a table the transcript sets among its numbered lines
/// stays one. Whether any cell was freed.
/// A transcript's testimony beside its number column, read as a grid (a speaker's turn in
/// one of its cells), is no table's; a table the transcript sets among its numbered lines
/// stays one. Whether any cell was freed.
pub fn free_spoken_cells(fences: &mut Fences, slots: &Slots, evidence: &Evidence) -> bool {
    let count = slots.len();
    let mut spoken = vec![false; count];
    for table in &fences.tables {
        let lines: Vec<Slot> = table
            .line_ids()
            .into_iter()
            .filter_map(|id| slots.of_id(id))
            .collect();
        if lines.iter().any(|slot| {
            fences.numbered_row[slot.index()] && evidence.words[slot.index()] & WORD_SPEAKER != 0
        }) {
            for slot in lines {
                spoken[slot.index()] = true;
            }
        }
    }
    let freed: Vec<usize> = (0..count)
        .filter(|&slot| {
            fences.numbered_row[slot] && fences.role[slot] == Role::Table && spoken[slot]
        })
        .collect();
    for &slot in &freed {
        crate::find::free_cell(fences, slot);
    }
    !freed.is_empty()
}

/// Gives a table's cell back to the lines no fence took.
/// Gives a table's cell back to the lines no fence took.
pub fn free_cell(fences: &mut Fences, slot: usize) {
    fences.role[slot] = Role::Open;
    fences.rule[slot] = Rule::None;
    fences.cell[slot] = 0;
}

/// The tables without the lines freed of their cells: a cell's parts on those lines dropped, and
/// a row or table left with no part dropped; each line's cell numbered again.
/// The tables without the lines freed of their cells: a cell's parts on those lines dropped, and
/// a row or table left with no part dropped; each line's cell numbered again.
pub fn prune_tables(fences: &mut Fences, slots: &Slots) {
    let cell = &fences.cell;
    let kept = |part: &crate::model::tables::CellPart| {
        slots
            .of_id(&part.line_id)
            .is_some_and(|slot| cell[slot.index()] != 0)
    };
    for table in &mut fences.tables {
        for row in &mut table.rows {
            for value in &mut row.cells {
                value.parts.retain(|part| kept(part));
            }
        }
        table
            .rows
            .retain(|row| row.cells.iter().any(|value| !value.parts.is_empty()));
    }
    fences.tables.retain(|table| !table.rows.is_empty());
    crate::find::number_cells(fences, slots);
}

/// Each line's cell (`cell`), counted over the tables' rows and their cells, from 1; a line
/// parted across cells stands in the first.
/// Each line's cell (`cell`), counted over the tables' rows and their cells, from 1; a line
/// parted across cells stands in the first.
pub fn number_cells(fences: &mut Fences, slots: &Slots) {
    fences.cell = vec![0; slots.len()];
    let cells = fences
        .tables
        .iter()
        .flat_map(|table| &table.rows)
        .flat_map(|row| &row.cells);
    for (cell, value) in cells.enumerate() {
        for slot in value
            .parts
            .iter()
            .filter_map(|part| slots.of_id(&part.line_id))
        {
            if fences.cell[slot.index()] == 0 {
                fences.cell[slot.index()] = cell as u32 + 1;
            }
        }
    }
}

