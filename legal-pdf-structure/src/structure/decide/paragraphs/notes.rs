//! The notes among a stream's lines: a note's label counts on from the last, and a stream's walk knows where it stands.

use super::*;

/// How far a note's label may pass the last, its lost labels between.
const NOTE_LOST: u32 = 3;

/// Where a stream's walk over its lines stands among the page's notes.
#[derive(Clone, Copy, Default, PartialEq)]
enum NoteState {
    /// In the text.
    #[default]
    Text,
    /// In a note on this page. A note goes on over a page only as carried.
    Inside(usize),
}

/// The last note line the walk passed: a note carried over is set as it was, and goes on only from an unfinished line.
#[derive(Clone, Copy)]
struct LastNote {
    size: f64,
    page: usize,
    finished: bool,
}

/// The walk over a stream's lines, noting which are notes.
#[derive(Default)]
pub(super) struct NoteWalk {
    state: NoteState,
    last: Option<LastNote>,
    /// The note the labels have counted to.
    count: u32,
    /// The page a quoted list a colon opened runs on: its next numbers are no notes.
    quoted_list: Option<usize>,
}

impl NoteWalk {
    pub(super) fn open(&self) -> bool {
        self.last.is_some_and(|last| !last.finished)
    }

    /// The note a labelled line leaves the count at: a figure opening a line of a note going on is its words ("2012 ABQB 375").
    pub(super) fn counted(&self, value: u32) -> u32 {
        if !self.open() || value == 1 || (self.count + 1..=self.count + NOTE_LOST).contains(&value) { value } else { self.count }
    }

    pub(super) fn counts_on(&self, value: u32) -> bool {
        value == 1 || (self.count + 1..=self.count + NOTE_LOST).contains(&value)
    }

    pub(super) fn inside(&self) -> bool {
        matches!(self.state, NoteState::Inside(_))
    }

    pub(super) fn turn_to(&mut self, page: usize) {
        if !matches!(self.state, NoteState::Inside(at) if at == page) {
            self.state = NoteState::Text;
        }
    }

    /// A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation stands
    /// smaller than the body too, "Immunity waived".
    pub(super) fn carries_on(&self, size: f64, page: usize) -> bool {
        self.last.is_some_and(|last| !last.finished && (size - last.size).abs() <= 0.5 && last.page + 1 == page)
    }

    pub(super) fn opens_quoted_list(&mut self, page: usize) {
        self.quoted_list = Some(page);
    }

    pub(super) fn in_quoted_list(&self, page: usize) -> bool {
        self.quoted_list == Some(page)
    }

    /// A note a fence took: it keeps the count, and may go on onto the next page.
    pub(super) fn fenced(&mut self, label: Option<u32>, text: &str, size: f64, page: usize) {
        self.count = label.map_or(self.count, |value| self.counted(value));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) });
    }

    /// A line the walk read as a note's.
    pub(super) fn took(&mut self, label: Option<u32>, text: &str, size: f64, page: usize) {
        self.count = label.map_or(self.count, |value| self.counted(value));
        // (A note's line stopping on "s." goes on.)
        let stopped = [" s.", " ss.", " para.", " paras.", " p.", " pp."].iter().any(|shortened| text.trim_end().ends_with(shortened));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) && !stopped });
        self.state = NoteState::Inside(page);
    }

    /// A line the walk kept as the text's.
    pub(super) fn left(&mut self) {
        self.state = NoteState::Text;
    }
}

impl Reading<'_> {
    // A note is known by its smaller type: small under the page's note band; or, wherever the band stands,
    // set smaller than the body and opening on a note's label below the page's last line in the body's type,
    // or carrying such a note on, its next line or the lines below the body after it (a label standing alone
    // on its line, its words on the next). A quotation is set small too, among the body's lines, and is the
    // paragraph's.
    pub(super) fn in_note(&self, slot: Slot, reading_place: usize, stream: Stream, kept: &[Slot], notes: &mut NoteWalk) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let (at, line) = (slot.index(), &slots.line(pages, slot).text);
        let (top, language) = (slots.line(pages, slot).bbox[1], stream.index());
            let band = self.measures.band_top[slots.page(slot)][language];
            let below = top >= self.measures.body_end[slots.page(slot)][language] - 1.0 && (band == f64::MAX || top >= band);
            notes.turn_to(slots.page(slot));
            // Notes count on from 1, each page's or the document's: a number that goes on with no count is a small
            // quotation's at the page's foot ("75 Along the same lines", "2 In this Act,").
            // A figure alone, far smaller than the body, is a label a scan's layer set on its own line.
            let lone_figure = || line.trim().parse::<u32>().ok().filter(|_| evidence.size[at] * 2.0 < fences.label_size());
            let label = evidence.label[at].filter(|label| label.suppress == 0).map(|label| label.value).or_else(lone_figure);
            // (Set in a note's type, or raised: a quotation's own numbers run on at the foot of a page in its larger one.)
            // (Or a tab after its label: "1\tHighway 5 runs ...", some reports' notes set little under the body.)
            let tabbed = self.smaller(at) && line.trim_start().trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('\t');
            let set_as_note = fences.below_body(evidence, at, 0.85) || tabbed || evidence.label[at].is_some_and(|label| label.raised);
            // (A colon over it sets a quotation out: "... as follows:" / "1. The party ...".)
            // (A note may stand under one all the same: its label a lone small figure, or raised.)
            let lead_in = kept.last().is_some_and(|last| slots.line(pages, *last).text.trim_end().ends_with(':')) && lone_figure().is_none() && !evidence.label[at].is_some_and(|label| label.raised);
            if lead_in && label.is_some() {
                notes.opens_quoted_list(slots.page(slot));
            }
            let lead_in = lead_in || notes.in_quoted_list(slots.page(slot));
            // (A table's cell is no note's label.)
            let lead_in = lead_in || fences.role[at] == Role::Table;
            let labelled = below && set_as_note && !lead_in && label.is_some_and(|value| notes.counts_on(value));
            // A scan's layer may read a label standing alone after its note's first line.
            let label_after = fences.order.get(reading_place + 1).is_some_and(|next| {
                evidence.size[next.index()] * 2.0 < fences.label_size() && evidence.label[next.index()].is_some() && slots.line(pages, *next).text.trim().chars().count() <= 4
            });
            let labelled = labelled || below && label_after && set_as_note && !lead_in;
            // Under the page's rule a label makes a note in any type: some set their notes in the body's.
            // A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation
            // stands smaller than the body too, "Immunity waived".
            let carried = self.smaller(at) && below && notes.carries_on(evidence.size[at], slots.page(slot));
            let in_note = fences.band[at] && (self.small(at) || label.is_some() || notes.inside())
                || self.smaller(at) && (labelled || notes.inside() && (fences.continues[at].is_some() || below)) || carried;
        if in_note {
            notes.took(label, line, evidence.size[at], slots.page(slot));
        } else {
            notes.left();
        }
        in_note
    }
}
