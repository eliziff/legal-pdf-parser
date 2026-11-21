//! The quotations of another writing's numbered paragraphs, read off the judgment's own count.
use crate::model::fences::Fences;
use crate::model::{Role, Slots};
use legal_pdf_core::model::Page;
use std::collections::{HashMap, HashSet};

/// The quotations of another writing's numbered paragraphs, read off the judgment's count (`read_paragraphs`): in
/// each document and language, a line no fence took and the band left that opens on a printed paragraph number the
/// count takes at another line (a number nothing else prints is the count's own, however it broke the count: a
/// misread digit), with the lines its block goes on into (`continues`) and the editorial lines under it ("[Emphasis
/// added]"), each a quotation of its own. Whether it found any.
/// The quotations of another writing's numbered paragraphs, read off the judgment's count (`read_paragraphs`): in
/// each document and language, a line no fence took and the band left that opens on a printed paragraph number the
/// count takes at another line (a number nothing else prints is the count's own, however it broke the count: a
/// misread digit), with the lines its block goes on into (`continues`) and the editorial lines under it ("[Emphasis
/// added]"), each a quotation of its own. Whether it found any.
pub fn read_quotations(fences: &mut Fences, pages: &[Page], slots: &Slots) -> bool {
    let mut own = HashMap::<(u16, bool), HashSet<u32>>::new();
    for at in 0..fences.paragraphs.count.len() {
        if let Some(number) = fences.paragraphs.count[at] {
            own.entry((fences.at(at).document, fences.at(at).translation))
                .or_default()
                .insert(number);
        }
    }
    let mut quotations = OpenQuotations::default();
    for place in 0..fences.order.len() {
        let slot = fences.order[place];
        let at = slot.index();
        if fences.role[at] != Role::Open || fences.band[at] {
            continue;
        }
        let stream = (fences.at(at).document, fences.at(at).translation);
        let text = slots.line(pages, slot).text.as_str();
        if let Some(quotation) = quotations.open_in(stream) {
            if fences.continues[at].is_some()
                || legal_structure_lines::numbered_paragraphs::editorial_line(text)
            {
                crate::find::recontext(fences, slot, |context| context.quotation = quotation);
                continue;
            }
            quotations.close(stream);
        }
        let quoted = fences.paragraphs.count[at].is_none()
            && legal_structure_lines::numbered_paragraphs::paragraph_number(text).is_some_and(
                |number| {
                    own.get(&stream)
                        .is_some_and(|numbers| numbers.contains(&(number as u32)))
                },
            );
        if quoted {
            let quotation = quotations.open(stream);
            crate::find::recontext(fences, slot, |context| context.quotation = quotation);
        }
    }
    quotations.last > 0
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
