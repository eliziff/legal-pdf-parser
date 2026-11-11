//! Each page's columns, read on the words the fences before them left, and the parallel translation's lines.
use std::collections::HashMap;
use legal_pdf_core::model::Page;
use crate::layout::{columns_of, translation_lines};
use crate::model::{Evidence, PageView, Role, Slot, Slots};
use crate::tables::PageItems;
use super::Fences;

impl Fences {
    /// Each page's words outside its line numbers and furniture, and its columns read on them.
    pub(crate) fn read_columns(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence) -> Vec<PageItems> {
        let starts: Vec<usize> = (0..pages.len()).map(|page_slot| slots.on(page_slot).start as usize).collect();
        let items: Vec<PageItems> = evidence
            .items
            .iter()
            .zip(&starts)
            .map(|(items, &start)| items_of(items, |line| self.role[start + line] == Role::Open))
            .collect();
        self.columns = pages
            .iter()
            .zip(&items)
            .zip(&starts)
            .map(|((page, items), &start)| {
                let lines = (0..page.lines.len()).filter(|&line| self.role[start + line] == Role::Open);
                columns_of(&page.lines, lines, &items.items, page.index as u32 + 1)
            })
            .collect();
        items
    }

    /// The parallel translation, on the pages' columns.
    pub(crate) fn read_translation(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, views: &[PageView]) {
        for slot in translation_lines(views, &self.columns, |slot| self.role[slot.index()] != Role::Open) {
            self.recontext(slot, |context| context.translation = true);
        }
        // A justified line the PDF sets a word at a time crosses the page's middle, in one language: a row of three or more
        // pieces of a word or two is one line, its language its first piece's.
        for page_slot in 0..pages.len() {
            let mut rows = HashMap::<u32, Vec<Slot>>::new();
            for slot in slots.on(page_slot).map(Slot) {
                rows.entry(evidence.row(slots, slot)).or_default().push(slot);
            }
            for (row, mut pieces) in rows {
                if row != u32::MAX && pieces.len() >= 3 && pieces.iter().all(|&slot| slots.line(pages, slot).text.split_whitespace().count() <= 2) {
                    pieces.sort_by(|&a, &b| slots.line(pages, a).bbox[0].total_cmp(&slots.line(pages, b).bbox[0]));
                    // (Across the gutter a row is two: "I. Introduction" level in both columns.)
                    let word_apart = pieces.windows(2).all(|pair| {
                        let (left, right) = (slots.line(pages, pair[0]).bbox, slots.line(pages, pair[1]).bbox);
                        right[0] - left[2] <= (left[3] - left[1]) * 3.0
                    });
                    if !word_apart {
                        continue;
                    }
                    let language = self.at(pieces[0].index()).translation;
                    for slot in pieces {
                        self.recontext(slot, |context| context.translation = language);
                    }
                }
            }
        }
    }
}

/// The page's items (`page_items`) of the lines `keep` holds (slots of the page's lines).
fn items_of(items: &PageItems, keep: impl Fn(usize) -> bool) -> PageItems {
    let (items, origin) = items
        .items
        .iter()
        .zip(&items.origin)
        .filter(|(_, (line, _))| keep(*line))
        .map(|(item, origin)| (item.clone(), *origin))
        .unzip();
    PageItems { items, origin }
}
