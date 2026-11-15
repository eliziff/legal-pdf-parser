//! Page numbers, running heads and feet: the document's and, once its constituent documents are known, each one's own.
use legal_pdf_core::model::Page;
use crate::model::{Edge, Evidence, PageView, Role, Rows, Rule, Slot, Slots};
use super::Fences;


impl Fences {
    /// Each page's number and the line printing it, as the page map reads them (`legal_pdf_pages::page_map`) over the
    /// pages' lines, a transcript's line numbers left out, and the labels the PDF declares (`page_labels`, by page
    /// index): the line its page's label.
    pub(crate) fn read_page_numbers(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, page_labels: &[Option<String>]) {
        let line_label = |page: usize, at: usize| evidence.line_number[slots.at(page, at).index()] || pages[page].lines[at].exclude_from_body;
        let mut read = legal_pdf_core::profile::tally("pages.evidence", || crate::furniture::page_evidence(pages, page_labels, line_label));
        let mut rowed = legal_pdf_pages::PageRows::of(&read);
        let map = legal_pdf_core::profile::tally("pages.map", || legal_pdf_pages::page_map_rowed(&read, &rowed, &[], &|_, _| false));
        self.folios = pages
            .iter()
            .enumerate()
            .map(|(page_slot, page)| {
                let binding = map.get(page.index)?;
                let line = binding.line.map(|at| self.folio(pages, slots, page_slot, at));
                Some((binding.label.clone()?, line))
            })
            .collect();
        // A page printing a second numbering (a record's page over a transcript's own, a book's over an article's): the
        // same fit over the pages with the first numbering's lines left blank. Its lines that print the number alone (a
        // folio's few characters, not a running head holding one) in the page's head or foot are page labels too; the
        // page keeps the first's number.
        // Only a page whose line was blanked has rows of its own to read again.
        let mut blanked = Vec::new();
        for (at, (page, binding)) in read.iter_mut().zip(&map).enumerate() {
            if let Some(line) = binding.line.and_then(|line| page.lines.get_mut(line)) {
                line.text.clear();
                blanked.push(at);
            }
        }
        for at in blanked {
            rowed.reread(&read, at);
        }
        let second = legal_pdf_core::profile::tally("pages.second", || legal_pdf_pages::page_map_rowed(&read, &rowed, &[], &|_, _| false));
        // A numbering counts the pages: most of its lines print the PDF page's number at one shift, on three pages at
        // least; its lines at that shift are its own (a note's number in the foot is none).
        let shift = |page: &Page| {
            let binding = second.get(page.index)?;
            binding.line?;
            Some(binding.label.as_deref()?.parse::<i64>().ok()? - page.index as i64)
        };
        let mut shifts: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
        for shift in pages.iter().filter_map(shift) {
            *shifts.entry(shift).or_default() += 1;
        }
        let total: usize = shifts.values().sum();
        let counted = shifts.into_iter().max_by_key(|&(shift, count)| (count, -shift.abs()))
            .filter(|&(_, count)| count >= 3 && count * 2 > total).map(|(shift, _)| shift);
        for (page_slot, page) in pages.iter().enumerate().filter(|(_, page)| counted.is_some() && shift(page) == counted) {
            // In the page's head or foot (a margin's paragraph numbers count too, in the body).
            let band = legal_pdf_pages::EDGE_BAND * page.height;
            let alone = |at: &usize| {
                let line = &page.lines[*at];
                line.text.trim().chars().count() <= 12 && (line.bbox[3] < band || line.bbox[1] > page.height - band)
            };
            if let Some(at) = second.get(page.index).and_then(|binding| binding.line).filter(alone) {
                self.folio(pages, slots, page_slot, at);
            }
        }
    }

    /// Line `at` of page slot `page_slot` set apart as its page's label, at the edge it stands on.
    fn folio(&mut self, pages: &[Page], slots: &Slots, page_slot: usize, at: usize) -> Slot {
        let slot = slots.at(page_slot, at);
        let (bbox, height) = (pages[page_slot].lines[at].bbox, pages[page_slot].height);
        self.set(slot, Role::PageLabel, Rule::Folio);
        self.edge[slot.index()] = Some(if bbox[1] + bbox[3] < height { Edge::Top } else { Edge::Bottom });
        slot
    }

    /// Running heads and feet, the folios' rows among them (`furniture::read`); and the edge lines naming their page's
    /// sections.
    pub(crate) fn read_furniture(&mut self, evidence: &Evidence, views: &[PageView], rows: &[&Rows]) {
        let found = legal_pdf_core::profile::measure("prepare.furniture", || {
            crate::furniture::read(views, rows, |slot| evidence.line_number[slot.index()], |slot| self.role[slot.index()] == Role::PageLabel)
        });
        for (slot, edge, rule) in found {
            if self.set(slot, Role::Furniture, rule) {
                self.edge[slot.index()] = Some(edge);
            }
        }
        self.section_heads = crate::furniture::section_heads(views, rows)
            .into_iter()
            .filter(|(slot, _, _)| self.role[slot.index()] == Role::Open)
            .collect();
    }

    /// Each constituent document's own furniture, read over its own lines as a document of its own
    /// (`furniture::read`): the running heads and feet it repeats on its own pages, a short
    /// document's edges. A line no fence took is its furniture.
    pub(crate) fn read_document_furniture(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, rows: &[&Rows]) {
        let documents = self.documents();
        let mut lines: Vec<Vec<Slot>> = vec![Vec::new(); documents];
        for slot in 0..slots.len() {
            lines[usize::from(self.at(slot).document)].push(Slot(slot as u32));
        }
        for lines in lines.iter().filter(|lines| !lines.is_empty()) {
            let views = PageView::of_slots(pages, slots, lines);
            let rows: Vec<&Rows> = views.iter().map(|view| rows[slots.page(view.lines[0])]).collect();
            let found = legal_pdf_core::profile::measure("prepare.furniture", || {
                crate::furniture::read(&views, &rows, |slot| evidence.line_number[slot.index()], |slot| self.role[slot.index()] == Role::PageLabel)
            });
            for (slot, edge, rule) in found {
                if self.set(slot, Role::Furniture, rule) {
                    self.edge[slot.index()] = Some(edge);
                    self.band[slot.index()] = false;
                }
            }
        }
    }
}
