//! The note band: under each page's separator, clear of its tables, and what shows a rule is no separator; the body's type.
use super::*;
use crate::notes::has_prior_reference;
use legal_structure_model::text::{is_note_symbol, label_prefix};

impl Fences {
    /// A table cannot claim a referenced footer note: a symbol after the table, or the words
    /// under a physical separator that opens on a referenced label. Free these before deciding
    /// whether the table crosses the separator, otherwise its mistaken claim hides the band.
    pub fn free_referenced_note_cells(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) -> bool {
        let mut freed = Vec::new();
        for (page_slot, page) in pages.iter().enumerate() {
            let referenced = slots.on(page_slot).map(Slot).filter_map(|slot| {
                let line = slots.line(pages, slot);
                if line.bbox[1] < page.height * 0.48 || !evidence.label[slot.index()].is_some_and(|label| label.suppress == 0) {
                    return None;
                }
                let prefix = label_prefix(&line.text)?;
                has_prior_reference(&page.lines, &prefix.label, line.bbox[1])
                    .then(|| (slot, prefix.label.chars().all(is_note_symbol)))
            }).collect::<Vec<_>>();
            let tolerance = separator_tolerance(page);
            let cut = evidence.pages[page_slot].separator.filter(|cut| referenced.iter()
                .any(|(slot, _)| slots.line(pages, *slot).bbox[1] >= cut - tolerance));
            freed.extend(slots.on(page_slot).map(Slot).filter(|slot| self.role[slot.index()] == Role::Table
                && (cut.is_some_and(|cut| slots.line(pages, *slot).bbox[1] >= cut - tolerance)
                    || referenced.iter().any(|(note, symbol)| *symbol && note == slot)))
                .map(Slot::index));
        }
        for &slot in &freed { self.free_cell(slot); }
        !freed.is_empty()
    }

    /// The note band: under the page's separator, clear of its tables.
    pub fn read_note_band(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        for (page_slot, page) in pages.iter().enumerate() {
            let range = slots.on(page_slot);
            let cells = range.clone().filter(|&slot| self.cell[slot as usize] != 0).map(|slot| slots.line(pages, Slot(slot)));
            let separator = separator_beside_tables(page, evidence.pages[page_slot].separator, cells);
            self.separators.push(separator);
            if let Some(cut) = separator {
                let tolerance = separator_tolerance(page);
                for slot in range.map(Slot) {
                    let line = slots.line(pages, slot);
                    self.band[slot.index()] = line.bbox[1] >= cut - tolerance
                        && !matches!(self.role[slot.index()], Role::LineLabel | Role::Furniture | Role::PageLabel);
                }
            }
        }
    }

    /// The body's type, of the lines no fence took, the note band aside.
    pub fn read_body_type(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        let mut mass: HashMap<(u16, u16), usize> = HashMap::new();
        for slot in (0..slots.len()).filter(|&slot| self.role[slot] == Role::Open && !self.band[slot] && evidence.face[slot] != u16::MAX) {
            *mass.entry((evidence.face[slot], evidence.half_points[slot])).or_default() +=
                slots.line(pages, Slot(slot as u32)).text.trim().chars().count();
        }
        self.body = mass
            .into_iter()
            .max_by(|a, b| {
                a.1.cmp(&b.1).then_with(|| {
                    (evidence.faces[usize::from(a.0 .0)].as_str(), a.0 .1).cmp(&(evidence.faces[usize::from(b.0 .0)].as_str(), b.0 .1))
                })
            })
            .map(|(style, _)| style);
    }

    /// A rule drawn across a page's running text is no note separator ("Cette disposition est r\u{e9}dig\u{e9}e ainsi:" under it,
    /// an enactment's small type below): a page's notes open on a label, or hold no words in the body's type. (A margin's
    /// figure opens nothing.)
    pub fn clear_false_bands(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        for page_slot in 0..pages.len() {
            let worded = |slot: &usize| slots.line(pages, Slot(*slot as u32)).text.chars().any(char::is_alphabetic);
            let band: Vec<usize> = slots.on(page_slot).map(|slot| slot as usize).filter(|&slot| self.band[slot]).collect();
            let top = |slot: usize| slots.line(pages, Slot(slot as u32)).bbox[1];
            let opens_on_label = band.iter().copied().filter(|slot| worded(slot)).min_by(|&a, &b| top(a).total_cmp(&top(b))).is_some_and(|first| evidence.label[first].is_some_and(|label| label.suppress == 0));
            // (Nor is the rule under an underlined line: "Issues", a heading's, over a list opening on its label.)
            let underlined = self.separators[page_slot].is_some_and(|cut| slots.on(page_slot).any(|slot| {
                let line = slots.line(pages, Slot(slot));
                !self.band[slot as usize] && (cut - line.bbox[3]).abs() <= 1.0 && !line.words.is_empty() && line.words.iter().all(|word| word.underlined)
            }));
            if underlined || !opens_on_label && band.iter().filter(|slot| worded(slot)).any(|&slot| self.body.is_some_and(|(_, size)| evidence.half_points[slot] == size)) {
                band.iter().for_each(|&slot| self.band[slot] = false);
                self.separators[page_slot] = None;
            }
        }
    }

    /// A band holding a note label is the page's notes, its table cells among them. Whether any cell was freed.
    pub fn free_noted_cells(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) -> bool {
        self.noted = (0..pages.len())
            .map(|page_slot| {
                slots.on(page_slot).map(|slot| slot as usize).any(|slot| {
                    self.band[slot]
                        && matches!(self.role[slot], Role::Open | Role::Table)
                        && evidence.label[slot].is_some_and(|label| label.suppress == 0)
                })
            })
            .collect();
        let freed: Vec<usize> = (0..slots.len())
            .filter(|&slot| self.band[slot] && self.role[slot] == Role::Table && self.noted[slots.page(Slot(slot as u32))])
            .collect();
        for &slot in &freed {
            self.free_cell(slot);
        }
        !freed.is_empty()
    }
}
