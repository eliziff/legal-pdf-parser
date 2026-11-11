//! Folios, running heads and feet: the document's and, once its constituent documents are known, each one's own.
use super::*;


impl Fences {
    /// Folios, running heads and feet and the lines on their rows (`furniture::read`); the folios'
    /// offset, tallied over every label each page's furniture prints; each page's printed label
    /// from its furniture (`page_label`), its line the page's label; and the edge lines naming their
    /// page's sections.
    pub fn read_furniture(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, views: &[PageView], rows: &[&Rows]) {
        let found = legal_pdf_core::profile::measure("prepare.furniture", || {
            crate::furniture::read(views, rows, |slot| evidence.line_number[slot.index()])
        });
        for (slot, edge, rule) in found {
            if self.set(slot, Role::Furniture, rule) {
                self.edge[slot.index()] = Some(edge);
            }
        }
        let labels: Vec<Vec<(String, usize)>> =
            pages.iter().enumerate().map(|(page_slot, page)| self.printed_labels(page, slots, page_slot)).collect();
        // Each page counts each offset its labels print once: a record that prints its own page
        // beside the page of the transcript it files counts both.
        let mut offsets: HashMap<i64, usize> = HashMap::new();
        for (page, labels) in pages.iter().zip(&labels) {
            let mut printed: Vec<i64> =
                labels.iter().filter_map(|(label, _)| label.parse::<i64>().ok()).map(|number| number - i64::from(page.number)).collect();
            printed.sort_unstable();
            printed.dedup();
            for offset in printed {
                *offsets.entry(offset).or_default() += 1;
            }
        }
        let folios: usize = offsets.values().sum();
        self.folio_offset = offsets
            .into_iter()
            .max_by_key(|&(offset, pages)| (pages, std::cmp::Reverse(offset.abs())))
            .filter(|&(_, pages)| pages >= 2 && pages * 2 >= folios)
            .map(|(offset, _)| offset);
        self.folios = pages
            .iter()
            .zip(&labels)
            .enumerate()
            .map(|(page_slot, (page, labels))| self.page_label(page, slots, page_slot, labels))
            .collect();
        self.section_heads = crate::furniture::section_heads(views, rows)
            .into_iter()
            .filter(|(slot, _, _)| self.role[slot.index()] == Role::Open)
            .collect();
    }

    /// The labels the page's furniture prints (`page_labels_among`), by the line printing each.
    fn printed_labels(&self, page: &Page, slots: &Slots, page_slot: usize) -> Vec<(String, usize)> {
        let range = slots.on(page_slot);
        let furniture = range
            .clone()
            .map(Slot)
            .filter(|slot| self.role[slot.index()] == Role::Furniture)
            .filter_map(|slot| Some(((slot.0 - range.start) as usize, self.edge[slot.index()]?)));
        page_labels_among(page, furniture)
    }

    /// The page's printed label among those its furniture prints (`labels`), the line printing it
    /// the page's label: the one label, or where they differ the one on the folios' offset; none
    /// where its furniture prints none, or prints labels that differ and none on the offset.
    fn page_label(&mut self, page: &Page, slots: &Slots, page_slot: usize, labels: &[(String, usize)]) -> Option<(Slot, String)> {
        let first = labels.first()?;
        let agree = labels.iter().all(|(label, _)| label.eq_ignore_ascii_case(&first.0));
        let on_offset = |(label, _): &&(String, usize)| {
            self.folio_offset.is_some_and(|offset| label.parse::<i64>().is_ok_and(|number| number - i64::from(page.number) == offset))
        };
        let Some((label, line)) = (if agree { Some(first) } else { labels.iter().find(on_offset) }) else {
            return None;
        };
        let slot = slots.at(page_slot, *line);
        self.role[slot.index()] = Role::PageLabel;
        self.rule[slot.index()] = Rule::Folio;
        Some((slot, label.clone()))
    }

    /// Each constituent document's own furniture, read over its own lines as a document of its own
    /// (`furniture::read`): the running heads and feet it repeats on its own pages, a short
    /// document's edges. A line no fence took is its furniture; a page that gains furniture and had
    /// no printed label takes its label from it.
    pub fn read_document_furniture(&mut self, pages: &[Page], slots: &Slots, evidence: &Evidence, rows: &[&Rows]) {
        let documents = self.documents();
        let mut lines: Vec<Vec<Slot>> = vec![Vec::new(); documents];
        for slot in 0..slots.len() {
            lines[usize::from(self.at(slot).document)].push(Slot(slot as u32));
        }
        let mut gained = vec![false; pages.len()];
        for lines in lines.iter().filter(|lines| !lines.is_empty()) {
            let views = PageView::of_slots(pages, slots, lines);
            let rows: Vec<&Rows> = views.iter().map(|view| rows[slots.page(view.lines[0])]).collect();
            let found = legal_pdf_core::profile::measure("prepare.furniture", || {
                crate::furniture::read(&views, &rows, |slot| evidence.line_number[slot.index()])
            });
            for (slot, edge, rule) in found {
                if self.set(slot, Role::Furniture, rule) {
                    self.edge[slot.index()] = Some(edge);
                    self.band[slot.index()] = false;
                    gained[slots.page(slot)] = true;
                }
            }
        }
        for (page_slot, page) in pages.iter().enumerate().filter(|(page_slot, _)| gained[*page_slot]) {
            if self.folios[page_slot].is_none() {
                let labels = self.printed_labels(page, slots, page_slot);
                self.folios[page_slot] = self.page_label(page, slots, page_slot, &labels);
            }
        }
    }
}
