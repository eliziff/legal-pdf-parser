//! The document's tables and each line's cell, and the cells given back to the lines no fence took.
use super::*;

impl Fences {
    /// The document's tables, the lines any of them holds, and the lines a contents list read as one holds.
    pub fn find_tables(&self, pages: &[Page], slots: &Slots, items: &[PageItems]) -> (Vec<legal_pdf_tables::PdfTable>, Vec<bool>, HashSet<String>) {
        let (tables, listed) = legal_pdf_core::profile::measure("classify.tables", || {
            legal_pdf_tables::document_tables(
                pages,
                items,
                &|id| slots.of_id(id).is_some_and(|slot| self.at(slot.index()).translation),
                &TableReadings,
            )
        });
        let mut in_table = vec![false; slots.len()];
        for slot in tables.iter().flat_map(|table| table.line_ids()).filter_map(|id| slots.of_id(id)) {
            in_table[slot.index()] = true;
        }
        (tables, in_table, listed)
    }

    /// The tables a contents list does not hold for the most part, and their cells (`cell`); a cell
    /// no fence took is the table's.
    pub fn read_tables(&mut self, slots: &Slots, tables: Vec<legal_pdf_tables::PdfTable>, listed: &[bool]) {
        self.tables = tables
            .into_iter()
            .filter(|table| {
                let ids = table.line_ids();
                let held = ids.iter().filter(|id| slots.of_id(id).is_some_and(|slot| listed[slot.index()])).count();
                held * 2 <= ids.len()
            })
            .collect();
        self.number_cells(slots);
        for slot in 0..slots.len() {
            if self.cell[slot] != 0 {
                self.set(Slot(slot as u32), Role::Table, Rule::Grid);
            }
        }
    }

    /// A transcript's testimony beside its number column, read as a grid (a speaker's turn in
    /// one of its cells), is no table's; a table the transcript sets among its numbered lines
    /// stays one. Whether any cell was freed.
    pub fn free_spoken_cells(&mut self, slots: &Slots, evidence: &Evidence) -> bool {
        let count = slots.len();
        let mut spoken = vec![false; count];
        for table in &self.tables {
            let lines: Vec<Slot> = table.line_ids().into_iter().filter_map(|id| slots.of_id(id)).collect();
            if lines.iter().any(|slot| self.numbered_row[slot.index()] && evidence.words[slot.index()] & WORD_SPEAKER != 0) {
                for slot in lines {
                    spoken[slot.index()] = true;
                }
            }
        }
        let freed: Vec<usize> =
            (0..count).filter(|&slot| self.numbered_row[slot] && self.role[slot] == Role::Table && spoken[slot]).collect();
        for &slot in &freed {
            self.free_cell(slot);
        }
        !freed.is_empty()
    }

    /// Gives a table's cell back to the lines no fence took.
    pub fn free_cell(&mut self, slot: usize) {
        self.role[slot] = Role::Open;
        self.rule[slot] = Rule::None;
        self.cell[slot] = 0;
    }

    /// The tables without the lines freed of their cells: a cell's parts on those lines dropped, and
    /// a row or table left with no part dropped; each line's cell numbered again.
    pub fn prune_tables(&mut self, slots: &Slots) {
        let cell = &self.cell;
        let kept = |part: &legal_pdf_tables::CellPart| slots.of_id(&part.line_id).is_some_and(|slot| cell[slot.index()] != 0);
        for table in &mut self.tables {
            for row in &mut table.rows {
                for value in &mut row.cells {
                    value.parts.retain(|part| kept(part));
                }
            }
            table.rows.retain(|row| row.cells.iter().any(|value| !value.parts.is_empty()));
        }
        self.tables.retain(|table| !table.rows.is_empty());
        self.number_cells(slots);
    }

    /// Each line's cell (`cell`), counted over the tables' rows and their cells, from 1; a line
    /// parted across cells stands in the first.
    fn number_cells(&mut self, slots: &Slots) {
        self.cell = vec![0; slots.len()];
        let cells = self.tables.iter().flat_map(|table| &table.rows).flat_map(|row| &row.cells);
        for (cell, value) in cells.enumerate() {
            for slot in value.parts.iter().filter_map(|part| slots.of_id(&part.line_id)) {
                if self.cell[slot.index()] == 0 {
                    self.cell[slot.index()] = cell as u32 + 1;
                }
            }
        }
    }
}
