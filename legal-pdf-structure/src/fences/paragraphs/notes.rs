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
    /// In a note on this page, whose labels open at `top`. A note goes on over a page only as carried.
    Inside { page: usize, top: f64 },
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
pub(crate) struct NoteWalk {
    state: NoteState,
    last: Option<LastNote>,
    /// The note the labels have counted to.
    count: u32,
    /// The page a quoted list a colon opened runs on, and its type: its next numbers are no notes, and a note's are set in another.
    quoted_list: Option<(usize, f64)>,
}

impl NoteWalk {
    /// The note a labelled line leaves the count at: a figure opening a line of a note is its words unless it counts on, whatever
    /// stopped the line above ("[2008]" / "1 S.C.R. 372", "532 A.R." / "312 (arrest then search)"). What is set as a label passes the count.
    pub(crate) fn counted(&self, value: u32, set_as_label: bool, page: usize) -> u32 {
        let first = value == 1 && !self.last.is_some_and(|last| !last.finished && last.page == page);
        if set_as_label || first || (self.count + 1..=self.count + NOTE_LOST).contains(&value) { value } else { self.count }
    }

    pub(crate) fn counts_on(&self, value: u32) -> bool {
        value == 1 || (self.count + 1..=self.count + NOTE_LOST).contains(&value)
    }

    pub(crate) fn inside(&self) -> bool {
        matches!(self.state, NoteState::Inside { .. })
    }

    /// A line ending above the labels the walk has passed on its page: a layer may read a page's labels before its text.
    pub(crate) fn above_labels(&self, bottom: f64) -> bool {
        matches!(self.state, NoteState::Inside { top, .. } if bottom < top)
    }

    pub(crate) fn turn_to(&mut self, page: usize) {
        if !matches!(self.state, NoteState::Inside { page: at, .. } if at == page) {
            self.state = NoteState::Text;
        }
    }

    /// A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation stands
    /// smaller than the body too, "Immunity waived".
    pub(crate) fn carries_on(&self, size: f64, page: usize) -> bool {
        self.last.is_some_and(|last| !last.finished && (size - last.size).abs() <= 0.5 && last.page + 1 == page)
    }

    /// The page before held a note.
    pub(crate) fn noted_before(&self, page: usize) -> bool {
        self.last.is_some_and(|last| last.page + 1 == page)
    }

    pub(crate) fn opens_quoted_list(&mut self, page: usize, size: f64) {
        self.quoted_list = Some((page, size));
    }

    pub(crate) fn in_quoted_list(&self, page: usize, size: f64) -> bool {
        self.quoted_list.is_some_and(|(at, list)| at == page && (size - list).abs() <= 0.5)
    }

    /// A note a fence took: it keeps the count, and may go on onto the next page.
    pub(crate) fn fenced(&mut self, label: Option<(u32, bool)>, text: &str, size: f64, page: usize) {
        self.count = label.map_or(self.count, |(value, set_as_label)| self.counted(value, set_as_label, page));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) });
    }

    /// A line the walk read as a note's.
    pub(crate) fn took(&mut self, label: Option<(u32, bool)>, text: &str, top: f64, size: f64, page: usize) {
        self.count = label.map_or(self.count, |(value, set_as_label)| self.counted(value, set_as_label, page));
        // (A note's line stopping on "s." goes on.)
        let stopped = [" s.", " ss.", " para.", " paras.", " p.", " pp."].iter().any(|shortened| text.trim_end().ends_with(shortened));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) && !stopped });
        self.state = match self.state {
            NoteState::Inside { page: at, top: first } if at == page => NoteState::Inside { page, top: first.min(top) },
            _ => NoteState::Inside { page, top },
        };
    }

    /// A line the walk kept as the text's.
    pub(crate) fn left(&mut self) {
        self.state = NoteState::Text;
    }
}

impl Reading<'_> {
    // A note is known by its smaller type: small under the page's note band; or, wherever the band stands,
    // set smaller than the body and opening on a note's label below the page's last line in the body's type,
    // or carrying such a note on, its next line or the lines below the body after it (a label standing alone
    // on its line, its words on the next). A quotation is set small too, among the body's lines, and is the
    // paragraph's.
    pub(crate) fn in_note(&self, slot: Slot, reading_place: usize, stream: Stream, kept: &[Slot], notes: &mut NoteWalk) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let (at, line) = (slot.index(), &slots.line(pages, slot).text);
        let (top, language) = (slots.line(pages, slot).bbox[1], stream.index());
            let band = self.measures.band_top[slots.page(slot)][language];
            let below = top >= self.measures.body_end[slots.page(slot)][language] - 1.0 && (band == f64::MAX || top >= band);
            notes.turn_to(slots.page(slot));
            // Notes count on from 1, each page's or the document's: a number that goes on with no count is a small
            // quotation's at the page's foot ("75 Along the same lines", "2 In this Act,").
            // A figure alone, far smaller than the body, is a label a scan's layer set on its own line.
            let lone_figure = || line.trim().parse::<u32>().ok().filter(|_| evidence.size[at] * 2.0 < fences.label_size() && self.first_on_row(reading_place));
            let label = evidence.label[at].filter(|label| label.suppress == 0).map(|label| label.value).or_else(lone_figure);
            // (Set in a note's type, or raised: a quotation's own numbers run on at the foot of a page in its larger one.)
            // (Or a tab after its label: "1\tHighway 5 runs ...", some reports' notes set little under the body.)
            let tabbed = self.smaller(at) && line.trim_start().trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('\t');
            let set_as_label = lone_figure().is_some() || tabbed || evidence.label[at].is_some_and(|label| label.raised);
            let set_as_note = fences.below_body(evidence, at, 0.85) || set_as_label;
            // (A colon over it sets a quotation out: "... as follows:" / "1. The party ...".)
            // (A note may stand under one all the same: its label a lone small figure, raised, or before a tab.)
            let lead_in = kept.last().is_some_and(|last| slots.line(pages, *last).text.trim_end().ends_with(':')) && !set_as_label;
            if lead_in && label.is_some() {
                notes.opens_quoted_list(slots.page(slot), evidence.size[at]);
            }
            let lead_in = lead_in || notes.in_quoted_list(slots.page(slot), evidence.size[at]);
            // (A table's cell is no note's label, but under a table holding the page's running text: a word processor's grid of numbered
            // paragraphs, its notes at the foot.)
            let lead_in = lead_in || fences.role[at] == Role::Table && self.measures.cells_end[slots.page(slot)][language].is_none_or(|end| top < end - 1.0);
            let labelled = below && set_as_note && !lead_in && label.is_some_and(|value| notes.counts_on(value));
            // A scan's layer may read a label standing alone after its note's first line.
            let label_after = fences.order.get(reading_place + 1).is_some_and(|next| {
                evidence.size[next.index()] * 2.0 < fences.label_size() && evidence.label[next.index()].is_some() && slots.line(pages, *next).text.trim().chars().count() <= 4
                    && self.first_on_row(reading_place + 1)
            });
            let labelled = labelled || below && label_after && set_as_note && !lead_in;
            // Under the page's rule a label makes a note in any type: some set their notes in the body's.
            // A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation
            // stands smaller than the body too, "Immunity waived".
            let carried = self.smaller(at) && below && (notes.carries_on(evidence.size[at], slots.page(slot)) || notes.noted_before(slots.page(slot)) && self.over_label(reading_place, notes));
            // (The page's text above labels a layer read first goes on being the text, and the words under them notes: "8", "9" / "[...] ce ne
            // sont pas" / "L'Oratoire, supra".)
            let above_labels = notes.above_labels(slots.line(pages, slot).bbox[3]);
            let in_note = fences.band[at] && (self.small(at) || label.is_some() || notes.inside())
                || self.smaller(at) && (labelled || notes.inside() && !above_labels && (fences.continues[at].is_some() || below)) || carried;
        if in_note {
            notes.took(label.map(|value| (value, set_as_label)), line, top, evidence.size[at], slots.page(slot));
        } else if !above_labels {
            notes.left();
        }
        in_note
    }

    /// A note the last page held may go on from a sentence or a quotation of its own, on no unfinished line: it stands right over the
    /// page's first label, in the type of its note or smaller ("This provision contemplates ..." / "69 This is ...").
    fn over_label(&self, reading_place: usize, notes: &NoteWalk) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let slot = fences.order[reading_place];
        let same_column = |next: &&Slot| slots.page(**next) == slots.page(slot) && fences.at(next.index()).translation == fences.at(slot.index()).translation;
        let under: Vec<Slot> = fences.order[reading_place + 1..].iter().take_while(same_column).copied().collect();
        let mut above = slots.line(pages, slot).bbox;
        for (step, next) in under.iter().enumerate() {
            let line = slots.line(pages, *next);
            if line.bbox[1] - above[3] > above[3] - above[1] || !self.smaller(next.index()) {
                return false;
            }
            if evidence.label[next.index()].is_some_and(|label| label.suppress == 0 && notes.counts_on(label.value)) {
                // (A label alone on its line has its words on the next.)
                let alone = line.text.trim().chars().all(|c| c.is_ascii_digit());
                let words = if alone { under.get(step + 1).unwrap_or(next) } else { next };
                return evidence.size[slot.index()] <= evidence.size[words.index()] + 0.5;
            }
            above = line.bbox;
        }
        false
    }

    /// A label stands first on its row: a figure after words on it is theirs ("(1981)," / "45").
    fn first_on_row(&self, reading_place: usize) -> bool {
        let (order, pages, slots) = (&self.fences.order, self.pages, self.slots);
        let (slot, bbox) = (order[reading_place], slots.line(pages, order[reading_place]).bbox);
        reading_place.checked_sub(1).is_none_or(|before| {
            let words = slots.line(pages, order[before]).bbox;
            slots.page(order[before]) != slots.page(slot) || words[2] > bbox[0] || words[3] <= bbox[1] || words[1] >= bbox[3]
        })
    }
}
