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
        let read = crate::furniture::page_evidence(pages, page_labels, line_label);
        let map = legal_pdf_pages::page_map(&read, &[], &|_, _| false);
        self.folios = pages
            .iter()
            .enumerate()
            .map(|(page_slot, page)| {
                let binding = map.get(page.index)?;
                let line = binding.line.map(|at| {
                    let slot = slots.at(page_slot, at);
                    let bbox = page.lines[at].bbox;
                    self.set(slot, Role::PageLabel, Rule::Folio);
                    self.edge[slot.index()] = Some(if bbox[1] + bbox[3] < page.height { Edge::Top } else { Edge::Bottom });
                    slot
                });
                Some((binding.label.clone()?, line))
            })
            .collect();
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
