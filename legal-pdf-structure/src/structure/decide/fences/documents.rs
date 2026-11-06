//! Each line's constituent document, and which documents are transcripts.
use super::*;

impl Fences {
    /// Each line's constituent document: the openings the document's bookmarks name, and those its
    /// own lines show, read on the lines no fence took and the note band left.
    pub(super) fn read_documents(&mut self, analysis: &dyn StructureAnalysis, views: &[PageView], outline: &[PdfOutlineEntry]) {
        let (role, band) = (&self.role, &self.band);
        let set_aside = |slot: Slot| role[slot.index()] != Role::Open || band[slot.index()];
        let mut starts = constituents::bookmarked_starts(analysis, views, outline, set_aside);
        constituents::reconcile_starts(views, &mut starts, set_aside);
        let documents = constituents::documents(views, &starts, |slot| {
            matches!(role[slot.index()], Role::LineLabel | Role::Furniture | Role::PageLabel)
        });
        for (slot, document) in documents {
            self.document[slot.index()] = document.min(usize::from(u16::MAX)) as u16;
        }
        self.starts = starts;
    }

    /// Which constituent documents are transcripts (`transcripts`) and which of them are testimony
    /// throughout (`testimony`), and each transcript's lines beside its number column
    /// (`numbered_row`).
    pub(super) fn read_transcripts(&mut self, slots: &Slots, evidence: &Evidence) {
        let count = slots.len();
        let documents = usize::from(self.document.iter().copied().max().unwrap_or(0)) + 1;
        let (mut numbered, mut speakers, mut lines) = (vec![0usize; documents], vec![0usize; documents], vec![0usize; documents]);
        // Each document's pages, and those printing its line numbers: (count, last page met).
        let (mut pages, mut numbered_pages) = (vec![(0usize, usize::MAX); documents], vec![(0usize, usize::MAX); documents]);
        let meet = |met: &mut (usize, usize), page: usize| {
            if met.1 != page {
                *met = (met.0 + 1, page);
            }
        };
        for slot in 0..count {
            let document = usize::from(self.document[slot]);
            let page = slots.page(Slot(slot as u32));
            meet(&mut pages[document], page);
            if evidence.line_number[slot] {
                numbered[document] += 1;
                meet(&mut numbered_pages[document], page);
            } else {
                lines[document] += 1;
                speakers[document] += usize::from(evidence.words[slot] & WORD_SPEAKER != 0);
            }
        }
        let spoken = |document: usize| speakers[document] >= 20 && speakers[document] * 10 >= lines[document];
        self.transcripts = (0..documents).map(|document| numbered[document] > 0 || spoken(document)).collect();
        // A contents page's locators or a table's row numbers read as line numbers make a document a
        // transcript, not its testimony.
        self.testimony = (0..documents)
            .map(|document| spoken(document) || numbered_pages[document].0 * 2 >= pages[document].0.max(1))
            .collect();
        self.numbered_row = vec![false; count];
        for page_slot in 0..evidence.pages.len() {
            let range = slots.on(page_slot);
            let rows = &evidence.pages[page_slot].rows;
            let numbered: Vec<u32> = range
                .clone()
                .filter(|&slot| self.role[slot as usize] == Role::LineLabel)
                .map(|slot| rows.of[(slot - range.start) as usize])
                .filter(|row| *row != u32::MAX)
                .collect();
            if numbered.is_empty() {
                continue;
            }
            for slot in range.clone() {
                let row = rows.of[(slot - range.start) as usize];
                self.numbered_row[slot as usize] = row != u32::MAX && numbered.contains(&row) && self.transcript(Slot(slot));
            }
        }
    }
}
