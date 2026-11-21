//! Each judgment's numbered paragraphs, read from the fenced lines: the streams of its counts and languages, the lines each
//! keeps, and what the reader of its text holds of them.

use crate::model::fences::Fences;
use crate::model::{Evidence, Held, Paragraphs, Role, Slot, Slots};
use legal_pdf_core::model::{sentence_ended, Page};
use std::collections::{BTreeMap, HashMap, HashSet};

pub mod ends;
pub mod keep;
pub mod measures;
pub mod notes;
pub mod write;

use measures::{Counts, Numbers, PageMeasures};
use write::Written;

/// The stream a line is read in, apart from the column it stands in (`Fences::translation`): a judgment in one language
/// reads a statute quoted in two columns in its one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Stream {
    Text,
    Translation,
}

impl Stream {
    const BOTH: [Stream; 2] = [Stream::Text, Stream::Translation];

    fn index(self) -> usize {
        usize::from(self == Stream::Translation)
    }

    fn is_translation(self) -> bool {
        self == Stream::Translation
    }
}

/// How far further in than the judgment's own numbers a quotation's number stands: more than a few characters.
const QUOTED_INDENT: f64 = 9.0;

/// The lines a judgment's paragraphs are read from, and what was measured of them: read once, for all its streams.
struct Reading<'a> {
    fences: &'a Fences,
    pages: &'a [Page],
    slots: &'a Slots,
    evidence: &'a Evidence,
    stream_of: Vec<Stream>,
    marker_of: HashMap<(Slot, Stream), (Slot, bool)>,
    attached: HashSet<Slot>,
    measures: PageMeasures<'a>,
    numbers: Numbers,
    counts: Counts,
    judgment_paragraphs: crate::JudgmentParagraphs<'a>,
}

impl<'a> Reading<'a> {
    fn new(
        fences: &'a Fences,
        pages: &'a [Page],
        slots: &'a Slots,
        evidence: &'a Evidence,
        judgment_paragraphs: crate::JudgmentParagraphs<'a>,
    ) -> Self {
        // (Each line's printed number, read once.)
        let printed: Vec<Option<usize>> = (0..slots.len())
            .map(|at| {
                legal_structure_lines::numbered_paragraphs::paragraph_number(
                    &slots.line(pages, Slot(at as u32)).text,
                )
            })
            .collect();
        let stream_of = crate::find::streams(fences, &printed);
        let marker_of = crate::find::marker_rows(fences, pages, evidence, slots, &stream_of);
        let attached = marker_of
            .values()
            .filter(|(_, added)| !added)
            .map(|(marker, _)| *marker)
            .collect();
        let measures = PageMeasures::of(fences, pages, slots);
        let numbers = Numbers::of(
            fences,
            pages,
            slots,
            evidence,
            &printed,
            &stream_of,
            &measures.edge,
        );
        let counts = Counts::of(fences, pages, slots, &stream_of, &numbers);
        Self {
            fences,
            pages,
            slots,
            evidence,
            stream_of,
            marker_of,
            attached,
            measures,
            numbers,
            counts,
            judgment_paragraphs,
        }
    }

    fn small(&self, at: usize) -> bool {
        crate::find::below_body(self.fences, self.slots.line(self.pages, Slot(at as u32)).size, at, 0.9)
    }

    fn smaller(&self, at: usize) -> bool {
        crate::find::below_body(self.fences, self.slots.line(self.pages, Slot(at as u32)).size, at, 0.95)
    }

    fn read(self) -> Paragraphs {
        let slots = self.slots;
        let mut paragraphs = Paragraphs {
            count: vec![None; slots.len()],
            held: vec![Vec::new(); slots.len()],
        };
        let mut led = Vec::<(Slot, Slot)>::new();
        for (document, stream) in (0..self.counts.streams)
            .flat_map(|count| Stream::BOTH.map(|stream| (count, stream)))
            .filter(|stream| !self.counts.note_lists.contains(stream))
        {
            let kept = self.kept(document, stream);
            let mut written = self.written(&kept, stream);
            led.append(&mut written.led);
            let text = String::from_utf16_lossy(&written.text);
            if text.trim().is_empty() {
                continue;
            }
            self.hold(&text, &written, &kept, stream, &mut paragraphs);
        }
        self.number_rows(led, &mut paragraphs);
        self.quoted_in_both(&mut paragraphs);
        paragraphs
    }

    /// What the shared reading of the stream's text holds of its lines.
    fn hold(
        &self,
        text: &str,
        written: &Written,
        kept: &[Slot],
        stream: Stream,
        paragraphs: &mut Paragraphs,
    ) {
        let (pages, slots, place) = (self.pages, self.slots, &written.place);
        let kept_at: HashMap<Slot, usize> = kept
            .iter()
            .enumerate()
            .map(|(at, slot)| (*slot, at))
            .collect();
        let read = (self.judgment_paragraphs)(text);
        // (The lines the numbers open on: a marker opening any other line is a quotation's or a list's.)
        let mut opens: HashSet<Slot> = read
            .iter()
            .filter_map(|(_, ranges)| ranges.first())
            .filter_map(|range| {
                place
                    .get(range.start..range.end.min(place.len()))
                    .unwrap_or_default()
                    .iter()
                    .flatten()
                    .next()
                    .map(|(slot, _)| *slot)
            })
            .collect();
        // (A number alone on its line opens the row of words it stands by.)
        let rows: Vec<Slot> = kept
            .iter()
            .copied()
            .filter(|&row| {
                self.marker_of
                    .get(&(row, stream))
                    .is_some_and(|(marker, _)| opens.contains(marker))
            })
            .collect();
        opens.extend(rows);
        for (number, ranges) in read {
            let number = number as u32;
            let mut spans = BTreeMap::<Slot, (u32, u32)>::new();
            let mut first = None;
            // A number alone on its line, before the words its range opens at, is the paragraph's.
            let opening = ranges.first().and_then(|range| {
                place[..range.start.min(place.len())]
                    .iter()
                    .rev()
                    .flatten()
                    .next()
                    .copied()
            });
            for range in ranges {
                for &(slot, offset) in place
                    .get(range.start..range.end.min(place.len()))
                    .unwrap_or_default()
                    .iter()
                    .flatten()
                {
                    first.get_or_insert(slot);
                    let span = spans.entry(slot).or_insert((offset, offset + 1));
                    *span = (span.0.min(offset), span.1.max(offset + 1));
                }
            }
            if let Some((slot, _)) = opening {
                let words = slots.line(pages, slot).text.trim();
                let digits = words.trim_matches(['[', ']', '(', ')', '.', ' ']);
                // (Unless the paragraph before holds it: "101." ending a citation just above the number's row.)
                if digits.parse::<u32>().ok() == Some(number)
                    && paragraphs.held[slot.index()].is_empty()
                {
                    spans
                        .entry(slot)
                        .or_insert((0, words.encode_utf16().count() as u32));
                }
            }
            self.settle(&mut spans, kept, &opens, &kept_at);
            if let Some(first) = first {
                paragraphs.count[first.index()].get_or_insert(number);
            }
            for (slot, (start, end)) in spans {
                paragraphs.held[slot.index()].push(Held {
                    translation: stream.is_translation(),
                    number,
                    start,
                    end,
                });
            }
        }
    }

    fn number_rows(&self, led: Vec<(Slot, Slot)>, paragraphs: &mut Paragraphs) {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        // A number standing alone beside the row it opens, or over it, is that paragraph's, in every language the row prints it.
        let rows = (0..slots.len()).flat_map(|at| {
            [
                fences.margin_twin[at],
                fences.margin_number[at],
                evidence.beside[at],
            ]
            .into_iter()
            .flatten()
            .map(move |row| (Slot(at as u32), row))
        });
        for (number_line, row) in rows.chain(led) {
            let at = number_line.index();
            let opened: Vec<(bool, u32)> = paragraphs.held[row.index()]
                .iter()
                .filter(|held| paragraphs.count[row.index()] == Some(held.number))
                .map(|held| (held.translation, held.number))
                .collect();
            let length = slots.line(pages, number_line).text.encode_utf16().count() as u32;
            // (A heading's enumerator on the row, "i.", is no number of the paragraph's.)
            if !slots
                .line(pages, number_line)
                .text
                .chars()
                .any(|c| c.is_ascii_digit())
            {
                continue;
            }
            for (translation, number) in opened {
                if !paragraphs.held[at]
                    .iter()
                    .any(|held| held.translation == translation && held.number == number)
                {
                    paragraphs.held[at].push(Held {
                        translation,
                        number,
                        start: 0,
                        end: length,
                    });
                }
            }
        }
    }

    /// An enactment quoted in both languages inside reasons in one: on a page none of whose lines in a language any paragraph
    /// holds, that language's lines beside a paragraph are that paragraph's.
    fn quoted_in_both(&self, paragraphs: &mut Paragraphs) {
        let (fences, pages, slots) = (self.fences, self.pages, self.slots);
        for page_slot in 0..pages.len() {
            let on: Vec<usize> = slots
                .on(page_slot)
                .map(|slot| slot as usize)
                .filter(|&at| fences.role[at] == Role::Open && !fences.band[at])
                .collect();
            for language in [false, true] {
                let mut own: Vec<usize> = on
                    .iter()
                    .copied()
                    .filter(|&at| fences.at(at).translation == language)
                    .collect();
                if own.iter().any(|&at| !paragraphs.held[at].is_empty()) {
                    continue;
                }
                let mut spans = BTreeMap::<(bool, u32), (f64, f64)>::new();
                for &at in on
                    .iter()
                    .filter(|&&at| fences.at(at).translation != language)
                {
                    let bbox = slots.line(pages, Slot(at as u32)).bbox;
                    for held in &paragraphs.held[at] {
                        let span = spans
                            .entry((held.translation, held.number))
                            .or_insert((bbox[1], bbox[3]));
                        *span = (span.0.min(bbox[1]), span.1.max(bbox[3]));
                    }
                }
                // (A column may run a line longer than the other: a sentence left open goes on past the span, "... de la" / "Cour d'appel.".)
                own.sort_by(|&a, &b| {
                    slots.line(pages, Slot(a as u32)).bbox[1]
                        .total_cmp(&slots.line(pages, Slot(b as u32)).bbox[1])
                });
                let (mut open, mut foot) = (None::<(bool, u32)>, f64::MIN);
                for at in own {
                    let line = slots.line(pages, Slot(at as u32));
                    let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
                    let beside = spans
                        .iter()
                        .find(|(_, span)| span.0 <= middle && middle <= span.1)
                        .map(|(key, _)| *key);
                    // (The next line down, no gap over it.)
                    let goes_on =
                        open.filter(|_| line.bbox[1] - foot <= line.bbox[3] - line.bbox[1]);
                    open = beside.or(goes_on);
                    if let Some((translation, number)) = open {
                        paragraphs.held[at].push(Held {
                            translation,
                            number,
                            start: 0,
                            end: line.text.encode_utf16().count() as u32,
                        });
                    }
                    foot = line.bbox[3];
                    if sentence_ended(&line.text) {
                        open = None;
                    }
                }
            }
        }
    }
}

/// A figure in far smaller type than the body is a note's label, however it lines up: a page's labels count on as
/// margin numbers do, and a scan's layer boxes its lines alike. Beside a line in the body's own type it numbers that
/// line: a tribunal's word processor sets its paragraph numbers small.
/// A figure in far smaller type than the body is a note's label, however it lines up: a page's labels count on as
/// margin numbers do, and a scan's layer boxes its lines alike. Beside a line in the body's own type it numbers that
/// line: a tribunal's word processor sets its paragraph numbers small.
pub fn note_label(fences: &Fences, evidence: &Evidence, size: f64, row: Option<Slot>) -> bool {
    size * 2.0 < crate::find::label_size(fences)
        && !row.is_some_and(|row| {
            fences.body
                .is_some_and(|(_, size)| evidence.half_points[row.index()] == size)
        })
}

pub fn label_size(fences: &Fences) -> f64 {
    fences.body.map_or(0.0, |(_, size)| f64::from(size) * 0.75)
}

/// The document's numbered paragraphs, read once, before notes, lists or headings claim a line. Each language's
/// body is written as a provider writes a decision, a block to a line: a block the lines `read_continuations`
/// chains, a marker standing alone on its line set before the words of its row (`Evidence::beside`; a margin
/// number before its row in both languages); the page's furniture, labels, contents and notes (smaller type under
/// the band, or opening on a note's label) are no part of it. The shared reading of a
/// decision's text (`judgment_paragraphs`) reads it, and its paragraphs come back to the lines as `paragraphs`.
/// Each judgment's numbered paragraphs: which lines each number holds, and the line each opens on. Reading and
/// storing are apart: only the margin's small figures are cleared before it, and the answer is stored once after.
/// The document's numbered paragraphs, read once, before notes, lists or headings claim a line. Each language's
/// body is written as a provider writes a decision, a block to a line: a block the lines `read_continuations`
/// chains, a marker standing alone on its line set before the words of its row (`Evidence::beside`; a margin
/// number before its row in both languages); the page's furniture, labels, contents and notes (smaller type under
/// the band, or opening on a note's label) are no part of it. The shared reading of a
/// decision's text (`judgment_paragraphs`) reads it, and its paragraphs come back to the lines as `paragraphs`.
/// Each judgment's numbered paragraphs: which lines each number holds, and the line each opens on. Reading and
/// storing are apart: only the margin's small figures are cleared before it, and the answer is stored once after.
pub fn read_paragraphs(fences: &mut Fences, pages: &[Page], slots: &Slots, evidence: &Evidence, judgment_paragraphs: crate::JudgmentParagraphs) {
    for at in 0..slots.len() {
        if crate::find::note_label(fences, 
            evidence,
            slots.line(pages, Slot(at as u32)).size,
            fences.margin_number[at],
        ) {
            fences.margin_number[at] = None;
            fences.margin_twin[at] = None;
        }
    }
    fences.paragraphs = Reading::new(fences, pages, slots, evidence, judgment_paragraphs).read();
}

/// A line no paragraph holds: a fence took it.
/// A line no paragraph holds: a fence took it.
pub fn set_apart(fences: &Fences, at: usize) -> bool {
    matches!(
        fences.role[at],
        Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents | Role::Note
    )
}

/// The stream each line reads in.
/// The stream each line reads in.
pub fn streams(fences: &Fences, printed: &[Option<usize>]) -> Vec<Stream> {
    // A judgment in one language may quote a statute in two, its columns side by side: the other language numbers
    // nothing, and its lines are the paragraph's quoting them. A parallel report prints its paragraphs in both.
    let numbered_apart = |at: usize| {
        fences.margin_twin[at].is_some()
            || fences.at(at).translation
                && (fences.margin_number[at].is_some() || printed[at].is_some())
    };
    let translated = (0..printed.len())
        .filter(|&at| fences.at(at).translation && fences.role[at] == Role::Open)
        .count();
    let open = (0..printed.len())
        .filter(|&at| fences.role[at] == Role::Open)
        .count();
    // (A quoted section's own number numbers no count: what counts is how few the other language prints.)
    let printed_number = |at: usize| fences.margin_number[at].is_some() || printed[at].is_some();
    let (apart, own) = (
        (0..printed.len()).filter(|&at| numbered_apart(at)).count(),
        (0..printed.len())
            .filter(|&at| !fences.at(at).translation && printed_number(at))
            .count(),
    );
    let one_text = translated * 4 < open && apart * 4 < own;
    (0..fences.ctx.len())
        .map(|at| {
            if fences.at(at).translation && !one_text {
                Stream::Translation
            } else {
                Stream::Text
            }
        })
        .collect()
}

/// Each row's marker, standing alone on its line: the margin's numbers by their rows, other markers by theirs.
/// Each row's marker, standing alone on its line: the margin's numbers by their rows, other markers by theirs.
pub fn marker_rows(fences: &Fences, pages: &[Page],
    evidence: &Evidence,
    slots: &Slots,
    stream_of: &[Stream],
) -> HashMap<(Slot, Stream), (Slot, bool)> {
    let mut marker_of = HashMap::<(Slot, Stream), (Slot, bool)>::new();
    for at in 0..slots.len() {
        let marker = Slot(at as u32);
        // (A marker takes one row: the margin's, where it has one.)
        let rows = [
            fences.margin_number[at],
            fences.margin_twin[at],
            evidence.beside[at].filter(|&row| {
                !crate::find::note_label(fences, evidence, slots.line(pages, marker).size, Some(row))
                    && fences.margin_number[at].is_none()
            }),
        ];
        for (k, row) in rows
            .into_iter()
            .enumerate()
            .filter_map(|(k, row)| Some((k, row?)))
        {
            // A marker's row is the line beside it in its own language (the nearest line on a parallel report's
            // row may be the other column's); only a margin number serves the other language's row.
            if crate::find::set_apart(fences, row.index())
                || k != 1 && fences.at(row.index()).translation != fences.at(at).translation
            {
                continue;
            }
            // The twin row's language takes the margin's number as text the stream adds, not as its line. (A number of the
            // margin's count takes its row from a stray figure beside it, a recognizer's "1" for a mark at the row's head.)
            let held = marker_of
                .entry((row, stream_of[row.index()]))
                .or_insert((marker, k == 1));
            if k < 2 && fences.margin_number[held.0.index()].is_none() {
                *held = (marker, k == 1);
            }
        }
    }
    marker_of
}

/// Set smaller than the body by the share: a margin's number stands in its own type.
/// Set smaller than the body by the share: a margin's number stands in its own type.
pub fn below_body(fences: &Fences, size: f64, at: usize, share: f64) -> bool {
    size > 0.0
        && fences.margin_number[at].is_none()
        && fences
            .body
            .is_some_and(|(_, body)| size * 2.0 < f64::from(body) * share)
}

/// A line of the running text.
/// A line of the running text.
pub fn body_line(fences: &Fences, at: usize) -> bool {
    fences.role[at] == Role::Open && !fences.band[at] && !crate::find::set_apart(fences, at)
}

