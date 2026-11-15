//! The quotations of another writing's numbered paragraphs, read off the judgment's own count.
use std::collections::{HashMap, HashSet};
use legal_pdf_core::model::Page;
use crate::model::{Role, Slots};
use super::Fences;

impl Fences {
    /// The quotations of another writing's numbered paragraphs, read off the judgment's count (`read_paragraphs`): in
    /// each document and language, a line no fence took and the band left that opens on a printed paragraph number the
    /// count takes at another line (a number nothing else prints is the count's own, however it broke the count: a
    /// misread digit), with the lines its block goes on into (`continues`) and the editorial lines under it ("[Emphasis
    /// added]"), each a quotation of its own. Whether it found any.
    pub(crate) fn read_quotations(&mut self, pages: &[Page], slots: &Slots) -> bool {
        let mut own = HashMap::<(u16, bool), HashSet<u32>>::new();
        for at in 0..self.paragraphs.count.len() {
            if let Some(number) = self.paragraphs.count[at] {
                own.entry((self.at(at).document, self.at(at).translation)).or_default().insert(number);
            }
        }
        let mut quotations = OpenQuotations::default();
        for place in 0..self.order.len() {
            let slot = self.order[place];
            let at = slot.index();
            if self.role[at] != Role::Open || self.band[at] {
                continue;
            }
            let stream = (self.at(at).document, self.at(at).translation);
            let text = slots.line(pages, slot).text.as_str();
            if let Some(quotation) = quotations.open_in(stream) {
                if self.continues[at].is_some() || legal_structure::numbered_paragraphs::editorial_line(text) {
                    self.recontext(slot, |context| context.quotation = quotation);
                    continue;
                }
                quotations.close(stream);
            }
            let quoted = self.paragraphs.count[at].is_none()
                && legal_structure::numbered_paragraphs::paragraph_number(text).is_some_and(|number| own.get(&stream).is_some_and(|numbers| numbers.contains(&(number as u32))));
            if quoted {
                let quotation = quotations.open(stream);
                self.recontext(slot, |context| context.quotation = quotation);
            }
        }
        quotations.last > 0
    }
}

/// A document's language: its lines read in one stream.
type QuotedStream = (u16, bool);

/// The walk over the lines' quotations: the last opened, and the one each stream stands in.
#[derive(Default)]
struct OpenQuotations {
    last: u16,
    open: HashMap<QuotedStream, u16>,
}

impl OpenQuotations {
    fn open_in(&self, stream: QuotedStream) -> Option<u16> {
        self.open.get(&stream).copied()
    }

    fn close(&mut self, stream: QuotedStream) {
        self.open.remove(&stream);
    }

    /// A new quotation, standing open in the stream.
    fn open(&mut self, stream: QuotedStream) -> u16 {
        self.last += 1;
        self.open.insert(stream, self.last);
        self.last
    }
}
