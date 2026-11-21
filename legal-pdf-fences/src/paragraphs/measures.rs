//! What a judgment's paragraphs are read against: where each page's text stands, which printed numbers are the
//! judgment's own, and which of its counts each line stands in.

use super::{Stream, QUOTED_INDENT};
use crate::model::fences::Fences;
use crate::model::{Evidence, Role, Slot, Slots};
use legal_pdf_core::model::Page;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Where each page's text stands.
pub struct PageMeasures<'a> {
    /// Where the page's last running line of the body's type ends (a folio or a running citation, a few words, is none):
    /// a page's notes stand below it.
    pub body_end: Vec<[f64; 2]>,
    pub band_top: Vec<[f64; 2]>,
    /// Where the page's last line of a table in the body's type ends, a table holding the running text.
    pub cells_end: Vec<[Option<f64>; 2]>,
    pub edge: HashMap<(usize, bool), f64>,
    columns: HashMap<(usize, bool), f64>,
    right_edges: HashMap<(usize, bool), f64>,
    pub page_ends: Vec<(f64, f64)>,
    pub running: HashSet<&'a str>,
}

impl<'a> PageMeasures<'a> {
    pub fn of(fences: &Fences, pages: &'a [Page], slots: &Slots) -> Self {
        // (Where a page's note band opens: small type above it is a quotation, whatever it numbers.)
        let (mut body_end, mut band_top) = (
            vec![[f64::MIN; 2]; pages.len()],
            vec![[f64::MAX; 2]; pages.len()],
        );
        let mut cells_end = vec![[None::<f64>; 2]; pages.len()];
        for &slot in &fences.order {
            let (page, language) = (
                slots.page(slot),
                usize::from(fences.at(slot.index()).translation),
            );
            if !crate::find::below_body(fences, slots.line(pages, slot).size, slot.index(), 0.95)
                && slots.line(pages, slot).text.split_whitespace().count() > 3
            {
                let end = slots.line(pages, slot).bbox[3];
                match fences.role[slot.index()] {
                    Role::Open => body_end[page][language] = body_end[page][language].max(end),
                    Role::Table => {
                        cells_end[page][language] =
                            Some(cells_end[page][language].map_or(end, |last: f64| last.max(end)))
                    }
                    _ => {}
                }
            }
            if fences.band[slot.index()] {
                band_top[page][language] =
                    band_top[page][language].min(slots.line(pages, slot).bbox[1]);
            }
        }
        // Where the judgment's own numbers print, in each language: the place, from the left edge of its page's body,
        // where most of its lines that open on a printed paragraph number open. A number printed further in is a
        // quotation's: its line runs on in the paragraph quoting it rather than opening one.
        let mut edge = HashMap::<(usize, bool), f64>::new();
        for &slot in fences
            .order
            .iter()
            .filter(|slot| crate::find::body_line(fences, slot.index()))
        {
            let left = edge
                .entry((slots.page(slot), fences.at(slot.index()).translation))
                .or_insert(f64::MAX);
            *left = left.min(slots.line(pages, slot).bbox[0]);
        }
        // Where the document's running text stands on a page: most of its lines of words open there, odd pages and even apart.
        let mut lefts = HashMap::<(usize, bool), BTreeMap<i64, usize>>::new();
        let mut right_edges = HashMap::<(usize, bool), f64>::new();
        for &slot in fences
            .order
            .iter()
            .filter(|slot| crate::find::body_line(fences, slot.index()))
        {
            let line = slots.line(pages, slot);
            if line.text.split_whitespace().count() >= 5 {
                *lefts
                    .entry((slots.page(slot) % 2, fences.at(slot.index()).translation))
                    .or_default()
                    .entry((line.bbox[0] / 3.0).round() as i64)
                    .or_default() += 1;
                let right = right_edges
                    .entry((slots.page(slot), fences.at(slot.index()).translation))
                    .or_insert(f64::MIN);
                *right = right.max(line.bbox[2]);
            }
        }
        // (Read once for each page's parity and language.)
        let columns: HashMap<(usize, bool), f64> = lefts
            .iter()
            .filter_map(|(key, lefts)| {
                lefts
                    .iter()
                    .max_by_key(|(_, count)| **count)
                    .map(|(left, _)| (*key, *left as f64 * 3.0))
            })
            .collect();
        // Where each page's text starts and ends.
        let mut page_ends = vec![(f64::MAX, f64::MIN); pages.len()];
        for &slot in &fences.order {
            let (ends, bbox) = (
                &mut page_ends[slots.page(slot)],
                slots.line(pages, slot).bbox,
            );
            *ends = (ends.0.min(bbox[1]), ends.1.max(bbox[3]));
        }
        let mut running: HashSet<&str> = fences
            .order
            .iter()
            .filter(|slot| fences.role[slot.index()] == Role::Furniture)
            .map(|&slot| slots.line(pages, slot).text.trim())
            .collect();
        // A short decision may print its running head on too few pages for the page reader: the same words on the top row of two
        // pages or more, smaller than the body, a gap under them ("Rocktree Solar Project Time Extension and Approvals Transfer").
        let top_row = |slot: Slot| {
            let bbox = slots.line(pages, slot).bbox;
            bbox[1] < page_ends[slots.page(slot)].0 + bbox[3] - bbox[1]
        };
        let mut under = vec![f64::MAX; pages.len()];
        for &slot in fences.order.iter().filter(|&&slot| !top_row(slot)) {
            under[slots.page(slot)] = under[slots.page(slot)].min(slots.line(pages, slot).bbox[1]);
        }
        let mut heads = HashMap::<&str, HashSet<usize>>::new();
        for &slot in fences.order.iter().filter(|&&slot| {
            top_row(slot) && crate::find::below_body(fences, slots.line(pages, slot).size, slot.index(), 0.95)
        }) {
            let (line, page) = (slots.line(pages, slot), slots.page(slot));
            if under[page] - line.bbox[3] > line.bbox[3] - line.bbox[1] {
                heads.entry(line.text.trim()).or_default().insert(page);
            }
        }
        running.extend(
            heads
                .into_iter()
                .filter(|(_, pages)| pages.len() >= 2)
                .map(|(head, _)| head),
        );
        Self {
            body_end,
            band_top,
            cells_end,
            edge,
            columns,
            right_edges,
            page_ends,
            running,
        }
    }

    pub fn column(&self, page: usize, language: bool) -> f64 {
        self.columns
            .get(&(page % 2, language))
            .copied()
            .unwrap_or(f64::MAX)
    }

    /// Where the page's running lines end.
    pub fn right_edge(&self, page: usize, language: bool) -> f64 {
        self.right_edges
            .get(&(page, language))
            .copied()
            .unwrap_or(f64::MIN)
    }
}

/// The printed numbers, in reading order.
pub struct Numbers {
    pub sequence: [Vec<(Slot, Option<usize>)>; 2],
    pub place_in: HashMap<Slot, usize>,
    /// The numbered lines whose words stand further in than the judgment's own.
    pub further: HashSet<Slot>,
    /// Those whose numbers the judgment's own place prints too.
    pub indented: HashSet<Slot>,
}

impl Numbers {
    pub fn of(
        fences: &Fences,
        pages: &[Page],
        slots: &Slots,
        evidence: &Evidence,
        printed: &[Option<usize>],
        stream_of: &[Stream],
        edge: &HashMap<(usize, bool), f64>,
    ) -> Self {
        // Where a numbered line's words stand, from that edge: a quotation may hang its number in the judgment's own margin
        // (its words alone set in); a number alone on its line has them on its row.
        let indent = |slot: Slot| {
            let line = slots.line(pages, slot);
            let row = fences.margin_number[slot.index()].or(evidence.beside[slot.index()]);
            let words = match row {
                Some(row) if evidence.marker[slot.index()].is_some_and(|reading| reading.only) => {
                    slots.line(pages, row).bbox[0]
                }
                _ => line.words.get(1).map_or(line.bbox[0], |word| word.bbox[0]),
            };
            words
                - edge
                    .get(&(slots.page(slot), fences.at(slot.index()).translation))
                    .copied()
                    .unwrap_or(0.0)
        };
        let numbered: Vec<Slot> = fences
            .order
            .iter()
            .copied()
            .filter(|slot| crate::find::body_line(fences, slot.index()) && printed[slot.index()].is_some())
            .collect();
        let mut places = BTreeMap::<(bool, i64), usize>::new();
        for &slot in &numbered {
            *places
                .entry((
                    fences.at(slot.index()).translation,
                    (indent(slot) / 3.0).round() as i64,
                ))
                .or_default() += 1;
        }
        let own_place = [false, true].map(|translation| {
            places
                .iter()
                .filter(|((own, _), _)| *own == translation)
                .max_by_key(|(_, count)| **count)
                .map(|((_, at), _)| *at as f64 * 3.0)
        });
        // Words set further in than the judgment's are a quotation's. Words closer to their number say nothing: the judgment's
        // own paragraphs may print so among quotations that print as it does (`quoted_runs` reads those by their counts).
        let further = |slot: Slot| {
            own_place[usize::from(fences.at(slot.index()).translation)]
                .is_some_and(|own| indent(slot) - own > QUOTED_INDENT)
        };
        let number_of = |slot: Slot| (fences.at(slot.index()).translation, printed[slot.index()]);
        // A number no line at the judgment's own place prints is the judgment's, however it stands.
        let standing: HashSet<(bool, Option<usize>)> = numbered
            .iter()
            .copied()
            .filter(|&slot| !further(slot))
            .map(number_of)
            .collect();
        let indented: HashSet<Slot> = numbered
            .iter()
            .copied()
            .filter(|&slot| further(slot) && standing.contains(&number_of(slot)))
            .collect();
        let sequence: [Vec<(Slot, Option<usize>)>; 2] = Stream::BOTH.map(|stream| {
            numbered
                .iter()
                .copied()
                .filter(|slot| stream_of[slot.index()] == stream)
                .map(|slot| (slot, printed[slot.index()]))
                .collect()
        });
        let place_in: HashMap<Slot, usize> = sequence
            .iter()
            .flat_map(|numbers| {
                numbers
                    .iter()
                    .enumerate()
                    .map(|(place, (slot, _))| (*slot, place))
            })
            .collect();
        let further = numbered
            .iter()
            .copied()
            .filter(|&slot| further(slot))
            .collect();
        Self {
            sequence,
            place_in,
            further,
            indented,
        }
    }
}

/// A new count opens at a "1" in the body's type and at the judgment's own place, five or more numbers on, that goes
/// on 2, 3, 4, 5: another decision in the file. A note list's "[1]" is small, a quotation's stands elsewhere, a
/// list's stops short.
pub struct Counts {
    /// The count each line stands in, in each stream.
    pub count_of: Vec<[u16; 2]>,
    pub note_lists: HashSet<(u16, Stream)>,
    pub streams: u16,
}

impl Counts {
    pub fn of(
        fences: &Fences,
        pages: &[Page],
        slots: &Slots,
        stream_of: &[Stream],
        numbers: &Numbers,
    ) -> Self {
        let (mut count_of, mut printed_numbers, mut counts) =
            (vec![[0u16; 2]; slots.len()], [0usize; 2], [0u16; 2]);
        // Notes listed after the reasons in the body's own type open no decision: the reasons' lines call their numbers in
        // turn ("... of violence.[1]"), and the list is no one's paragraphs.
        let called = |upto: usize, stream: Stream| {
            let mut calls = [1, 2, 3]
                .map(|number| format!("[{number}]"))
                .into_iter()
                .peekable();
            for slot in fences.order[..upto]
                .iter()
                .filter(|slot| stream_of[slot.index()] == stream)
            {
                let line = slots.line(pages, *slot).text.trim();
                while calls
                    .peek()
                    .is_some_and(|call| line.find(call.as_str()).is_some_and(|at| at > 0))
                {
                    calls.next();
                }
            }
            calls.peek().is_none()
        };
        let mut note_lists = HashSet::<(u16, Stream)>::new();
        for (reading_place, &slot) in fences.order.iter().enumerate() {
            let (at, language) = (slot.index(), stream_of[slot.index()].index());
            if let Some(&place) = numbers.place_in.get(&slot) {
                let sequence = &numbers.sequence[language];
                let goes_on = (1..5).all(|step| {
                    sequence
                        .get(place + step)
                        .is_some_and(|(_, number)| *number == Some(step + 1))
                });
                if sequence[place].1 == Some(1)
                    && goes_on
                    && !crate::find::below_body(fences, slots.line(pages, slot).size, at, 0.95)
                    && !numbers.further.contains(&slot)
                    && printed_numbers[language] >= 5
                {
                    counts[language] += 1;
                    printed_numbers[language] = 0;
                    if called(reading_place, stream_of[at]) {
                        note_lists.insert((counts[language], stream_of[at]));
                    }
                }
                printed_numbers[language] += 1;
            }
            count_of[at][language] = counts[language];
        }
        let streams = counts.iter().copied().max().unwrap_or(0) + 1;
        Self {
            count_of,
            note_lists,
            streams,
        }
    }
}
