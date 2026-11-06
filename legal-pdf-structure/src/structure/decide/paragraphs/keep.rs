//! The lines a stream keeps as its words, in the order it reads them.

use super::*;

/// The most letters and figures a recognizer's reading of a folio holds.
const FOLIO_MARKS: usize = 4;

impl Reading<'_> {
    /// The lines of one count of one stream that are its words.
    pub(super) fn kept(&self, document: u16, stream: Stream) -> Vec<Slot> {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let mut kept = Vec::<Slot>::new();
        let mut notes = NoteWalk::default();
        for (reading_place, &slot) in fences.order.iter().enumerate() {
            let at = slot.index();
                // The notes a fence took keep the count, and may go on onto the next page: a note left among the words goes on from them.
                if fences.role[at] == Role::Note && self.stream_of[at] == stream {
                    notes.fenced(evidence.label[at].map(|label| (label.value, label.raised)), &slots.line(pages, slot).text, evidence.size[at], slots.page(slot));
                }
                if self.counts.count_of[at][stream.index()] != document || self.stream_of[at] != stream || fences.set_apart(at) || self.attached.contains(&slot) {
                    continue;
                }
            if self.numbers_nothing(slot) || self.page_mark(slot, reading_place) {
                continue;
            }
            if !self.in_note(slot, reading_place, stream, &kept, &mut notes) && self.text_words(slot, &kept) {
                kept.push(slot);
            }
        }
        self.in_reading_order(&mut kept);
        kept
    }

    /// A figure in the margin beside no words, and out of the count's turn, numbers nothing: a heading's call on a row of its own.
    fn numbers_nothing(&self, slot: Slot) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let at = slot.index();
        let out_of_turn = self.numbers.place_in.get(&slot).is_some_and(|&place| {
            let numbers = &self.numbers.sequence[self.stream_of[at].index()];
            let here = numbers[place].1;
            let last = place.checked_sub(1).and_then(|before| numbers[before].1);
            let next = numbers.get(place + 1).and_then(|(_, number)| *number);
            here != last.map(|number| number + 1) && next != here.map(|number| number + 1)
        });
        out_of_turn && evidence.beside[at].is_none() && fences.margin_number[at].is_none() && fences.margin_twin[at].is_none()
            && evidence.marker[at].is_some_and(|reading| reading.only) && slots.line(pages, slot).bbox[2] < self.measures.column(slots.page(slot), fences.translation[at])
    }

    /// A line the page prints for itself, no one's words.
    fn page_mark(&self, slot: Slot, reading_place: usize) -> bool {
        let (fences, pages, slots) = (self.fences, self.pages, self.slots);
        let line = &slots.line(pages, slot).text;
        // A folio that lost its figures ("-  -") is no one's words; nor is a recognizer's reading of one, a few marks
        // at the very top or foot of its page ("4-", "=f>~").
        if !line.chars().any(char::is_alphanumeric) && line.chars().all(|c| c.is_whitespace() || matches!(c, '-' | '\u{2013}' | '\u{2014}')) {
            return true;
        }
        let (bbox, (top, foot)) = (slots.line(pages, slot).bbox, self.measures.page_ends[slots.page(slot)]);
        // (Set apart from the page's lines: a sentence's last word may stand alone at its foot, "thew", "692,".)
        let neighbour = |step: isize| fences.order.get(reading_place.wrapping_add_signed(step)).filter(|other| slots.page(**other) == slots.page(slot)).map(|other| slots.line(pages, *other).bbox);
        let height = bbox[3] - bbox[1];
        let apart = if bbox[3] >= foot { neighbour(-1).is_none_or(|above| bbox[1] - above[3] > height) } else { neighbour(1).is_none_or(|below| below[1] - bbox[3] > height) };
        if slots.line(pages, slot).recognized() && (bbox[1] <= top || bbox[3] >= foot) && apart && line.chars().filter(|c| c.is_alphanumeric()).count() <= FOLIO_MARKS {
            return true;
        }
        // A folio in words ("Page: 3"), and what the page's furniture prints elsewhere, are the page's.
        let folio = line.trim().strip_prefix("Page").or_else(|| line.trim().strip_prefix("PAGE")).is_some_and(|rest| {
            let rest = rest.trim_start_matches([':', ' ']);
            !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
        });
        self.edge_line(slot) && (folio || self.measures.running.contains(line.trim()))
    }

    fn edge_line(&self, slot: Slot) -> bool {
        let (bbox, (top, foot)) = (self.slots.line(self.pages, slot).bbox, self.measures.page_ends[self.slots.page(slot)]);
        bbox[1] <= top + (bbox[3] - bbox[1]) * 3.0 || bbox[3] >= foot - (bbox[3] - bbox[1]) * 3.0
    }

    /// (Words far smaller than the body and in no note are a page's aside, "OUR VISION...": but a size no one
    /// measured, an editor's "[Emphasis added]", and on a scan a line among the page's lines, whose layer boxes
    /// lines by their letters, are the text's.)
    /// (A letterhead, far larger than the body, is no one's: "Federal Court" / "of Appeal" at twice its type.)
    fn text_words(&self, slot: Slot, kept: &[Slot]) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let (at, line, bbox) = (slot.index(), &slots.line(pages, slot).text, slots.line(pages, slot).bbox);
        let text = evidence.size[at] * 2.0 >= fences.label_size() || evidence.size[at] == 0.0 || self.analysis.editorial_line(line)
    || slots.line(pages, slot).recognized() && (!self.edge_line(slot) || kept.last().is_some_and(|last| !sentence_ended(&slots.line(pages, *last).text)));
        let wordmark = fences.body.is_some_and(|(_, size)| evidence.size[at] * 2.0 >= f64::from(size) * 1.6) && line.trim().chars().count() > 1 && line.split_whitespace().count() <= 6;
        // (A layer may print one line twice, the second a suffix of the first in its box.)
        let repeated = kept.last().is_some_and(|last| {
            let above = slots.line(pages, *last);
            slots.page(*last) == slots.page(slot) && (above.bbox[1] - bbox[1]).abs() <= 2.0 && line.trim().chars().count() >= 12 && above.text.trim_end().ends_with(line.trim())
        });
        text && !wordmark && !repeated
    }

    fn in_reading_order(&self, kept: &mut [Slot]) {
        let (pages, slots, fences) = (self.pages, self.slots, self.fences);
        // A table's reading may take prose set about it out of its order ("[4] This appeal ..." under its own next lines):
        // what a table holds reads down its page.
        let mut from = 0;
        while from < kept.len() {
            let page = slots.page(kept[from]);
            let to = from + kept[from..].iter().take_while(|slot| fences.role[slot.index()] == Role::Table && slots.page(**slot) == page).count();
            kept[from..to].sort_by(|a, b| {
                let (a, b) = (slots.line(pages, *a).bbox, slots.line(pages, *b).bbox);
                (a[1] / 2.0).round().total_cmp(&(b[1] / 2.0).round()).then(a[0].total_cmp(&b[0]))
            });
            from = to.max(from + 1);
        }
        // A layer may split a line and set its tail first: the tail's top stands a little above its line's.
        for at in 1..kept.len() {
            let (tail, opening) = (slots.line(pages, kept[at - 1]).bbox, slots.line(pages, kept[at]).bbox);
            let shared = tail[3].min(opening[3]) - tail[1].max(opening[1]);
            // (A scan's pieces may overlap by a few points.)
            let overlap = if slots.line(pages, kept[at]).recognized() { opening[3] - opening[1] } else { 0.0 };
            if slots.page(kept[at - 1]) == slots.page(kept[at]) && opening[2] <= tail[0] + overlap && shared * 2.0 > (tail[3] - tail[1]).min(opening[3] - opening[1]) {
                kept.swap(at - 1, at);
            }
        }
    }
}
