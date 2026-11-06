//! What a judgment's paragraphs are read against: where each page's text stands, which printed numbers are the
//! judgment's own, and which of its counts each line stands in.

use super::*;

/// Where each page's text stands.
pub(super) struct PageMeasures<'a> {
    /// Where the page's last running line of the body's type ends (a folio or a running citation, a few words, is none):
    /// a page's notes stand below it.
    pub(super) body_end: Vec<[f64; 2]>,
    pub(super) band_top: Vec<[f64; 2]>,
    /// Where the page's last line of a table in the body's type ends, a table holding the running text.
    pub(super) cells_end: Vec<[Option<f64>; 2]>,
    pub(super) edge: HashMap<(usize, bool), f64>,
    columns: HashMap<(usize, bool), f64>,
    right_edges: HashMap<(usize, bool), f64>,
    pub(super) page_ends: Vec<(f64, f64)>,
    pub(super) running: HashSet<&'a str>,
}

impl<'a> PageMeasures<'a> {
    pub(super) fn of(fences: &Fences, pages: &'a [Page], slots: &Slots, evidence: &Evidence) -> Self {
        // (Where a page's note band opens: small type above it is a quotation, whatever it numbers.)
        let (mut body_end, mut band_top) = (vec![[f64::MIN; 2]; pages.len()], vec![[f64::MAX; 2]; pages.len()]);
        let mut cells_end = vec![[None::<f64>; 2]; pages.len()];
        for &slot in &fences.order {
            let (page, language) = (slots.page(slot), usize::from(fences.translation[slot.index()]));
            if !fences.below_body(evidence, slot.index(), 0.95) && slots.line(pages, slot).text.split_whitespace().count() > 3 {
                let end = slots.line(pages, slot).bbox[3];
                match fences.role[slot.index()] {
                    Role::Open => body_end[page][language] = body_end[page][language].max(end),
                    Role::Table => cells_end[page][language] = Some(cells_end[page][language].map_or(end, |last: f64| last.max(end))),
                    _ => {}
                }
            }
            if fences.band[slot.index()] {
                band_top[page][language] = band_top[page][language].min(slots.line(pages, slot).bbox[1]);
            }
        }
        // Where the judgment's own numbers print, in each language: the place, from the left edge of its page's body,
        // where most of its lines that open on a printed paragraph number open. A number printed further in is a
        // quotation's: its line runs on in the paragraph quoting it rather than opening one.
        let mut edge = HashMap::<(usize, bool), f64>::new();
        for &slot in fences.order.iter().filter(|slot| fences.body_line(slot.index())) {
            let left = edge.entry((slots.page(slot), fences.translation[slot.index()])).or_insert(f64::MAX);
            *left = left.min(slots.line(pages, slot).bbox[0]);
        }
        // Where the document's running text stands on a page: most of its lines of words open there, odd pages and even apart.
        let mut lefts = HashMap::<(usize, bool), BTreeMap<i64, usize>>::new();
        let mut right_edges = HashMap::<(usize, bool), f64>::new();
        for &slot in fences.order.iter().filter(|slot| fences.body_line(slot.index())) {
            let line = slots.line(pages, slot);
            if line.text.split_whitespace().count() >= 5 {
                *lefts.entry((slots.page(slot) % 2, fences.translation[slot.index()])).or_default().entry((line.bbox[0] / 3.0).round() as i64).or_default() += 1;
                let right = right_edges.entry((slots.page(slot), fences.translation[slot.index()])).or_insert(f64::MIN);
                *right = right.max(line.bbox[2]);
            }
        }
        // (Read once for each page's parity and language.)
        let columns: HashMap<(usize, bool), f64> = lefts.iter().filter_map(|(key, lefts)| lefts.iter().max_by_key(|(_, count)| **count).map(|(left, _)| (*key, *left as f64 * 3.0))).collect();
        // Where each page's text starts and ends.
        let mut page_ends = vec![(f64::MAX, f64::MIN); pages.len()];
        for &slot in &fences.order {
            let (ends, bbox) = (&mut page_ends[slots.page(slot)], slots.line(pages, slot).bbox);
            *ends = (ends.0.min(bbox[1]), ends.1.max(bbox[3]));
        }
        let running: HashSet<&str> = fences.order.iter().filter(|slot| fences.role[slot.index()] == Role::Furniture).map(|&slot| slots.line(pages, slot).text.trim()).collect();
        Self { body_end, band_top, cells_end, edge, columns, right_edges, page_ends, running }
    }

    pub(super) fn column(&self, page: usize, language: bool) -> f64 {
        self.columns.get(&(page % 2, language)).copied().unwrap_or(f64::MAX)
    }

    /// Where the page's running lines end.
    pub(super) fn right_edge(&self, page: usize, language: bool) -> f64 {
        self.right_edges.get(&(page, language)).copied().unwrap_or(f64::MIN)
    }
}

/// The printed numbers, in reading order.
pub(super) struct Numbers {
    pub(super) sequence: [Vec<(Slot, Option<usize>)>; 2],
    pub(super) place_in: HashMap<Slot, usize>,
    /// The numbered lines whose words stand further in than the judgment's own.
    pub(super) further: HashSet<Slot>,
    /// Those whose numbers the judgment's own place prints too.
    pub(super) indented: HashSet<Slot>,
}

impl Numbers {
    pub(super) fn of(fences: &Fences, pages: &[Page], slots: &Slots, evidence: &Evidence, printed: &[Option<usize>], stream_of: &[Stream], edge: &HashMap<(usize, bool), f64>) -> Self {
        // Where a numbered line's words stand, from that edge: a quotation may hang its number in the judgment's own margin
        // (its words alone set in); a number alone on its line has them on its row.
        let indent = |slot: Slot| {
            let line = slots.line(pages, slot);
            let row = fences.margin_number[slot.index()].or(evidence.beside[slot.index()]);
            let words = match row {
                Some(row) if evidence.marker[slot.index()].is_some_and(|reading| reading.only) => slots.line(pages, row).bbox[0],
                _ => line.words.get(1).map_or(line.bbox[0], |word| word.bbox[0]),
            };
            words - edge.get(&(slots.page(slot), fences.translation[slot.index()])).copied().unwrap_or(0.0)
        };
        let numbered: Vec<Slot> = fences.order.iter().copied()
            .filter(|slot| fences.body_line(slot.index()) && printed[slot.index()].is_some())
            .collect();
        let mut places = BTreeMap::<(bool, i64), usize>::new();
        for &slot in &numbered {
            *places.entry((fences.translation[slot.index()], (indent(slot) / 3.0).round() as i64)).or_default() += 1;
        }
        let own_place = [false, true].map(|translation| places.iter().filter(|((own, _), _)| *own == translation).max_by_key(|(_, count)| **count).map(|((_, at), _)| *at as f64 * 3.0));
        // Words set further in than the judgment's are a quotation's. Words closer to their number say nothing: the judgment's
        // own paragraphs may print so among quotations that print as it does (`quoted_runs` reads those by their counts).
        let further = |slot: Slot| own_place[usize::from(fences.translation[slot.index()])].is_some_and(|own| indent(slot) - own > QUOTED_INDENT);
        let number_of = |slot: Slot| (fences.translation[slot.index()], printed[slot.index()]);
        // A number no line at the judgment's own place prints is the judgment's, however it stands.
        let standing: HashSet<(bool, Option<usize>)> = numbered.iter().copied().filter(|&slot| !further(slot)).map(number_of).collect();
        let indented: HashSet<Slot> = numbered.iter().copied().filter(|&slot| further(slot) && standing.contains(&number_of(slot))).collect();
        let sequence: [Vec<(Slot, Option<usize>)>; 2] = Stream::BOTH.map(|stream| {
            numbered.iter().copied().filter(|slot| stream_of[slot.index()] == stream)
                .map(|slot| (slot, printed[slot.index()])).collect()
        });
        let place_in: HashMap<Slot, usize> = sequence.iter().flat_map(|numbers| numbers.iter().enumerate().map(|(place, (slot, _))| (*slot, place))).collect();
        let further = numbered.iter().copied().filter(|&slot| further(slot)).collect();
        Self { sequence, place_in, further, indented }
    }
}

/// A new count opens at a "1" in the body's type and at the judgment's own place, five or more numbers on, that goes
/// on 2, 3, 4, 5: another decision in the file. A note list's "[1]" is small, a quotation's stands elsewhere, a
/// list's stops short.
pub(super) struct Counts {
    /// The count each line stands in, in each stream.
    pub(super) count_of: Vec<[u16; 2]>,
    pub(super) note_lists: HashSet<(u16, Stream)>,
    pub(super) streams: u16,
}

impl Counts {
    pub(super) fn of(fences: &Fences, pages: &[Page], slots: &Slots, evidence: &Evidence, stream_of: &[Stream], numbers: &Numbers) -> Self {
        let (mut count_of, mut printed_numbers, mut counts) = (vec![[0u16; 2]; slots.len()], [0usize; 2], [0u16; 2]);
        // Notes listed after the reasons in the body's own type open no decision: the reasons' lines call their numbers in
        // turn ("... of violence.[1]"), and the list is no one's paragraphs.
        let called = |upto: usize, stream: Stream| {
            let mut calls = [1, 2, 3].map(|number| format!("[{number}]")).into_iter().peekable();
            for slot in fences.order[..upto].iter().filter(|slot| stream_of[slot.index()] == stream) {
                let line = slots.line(pages, *slot).text.trim();
                while calls.peek().is_some_and(|call| line.find(call.as_str()).is_some_and(|at| at > 0)) {
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
                let goes_on = (1..5).all(|step| sequence.get(place + step).is_some_and(|(_, number)| *number == Some(step + 1)));
                if sequence[place].1 == Some(1) && goes_on && !fences.below_body(evidence, at, 0.95) && !numbers.further.contains(&slot) && printed_numbers[language] >= 5 {
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
        Self { count_of, note_lists, streams }
    }
}
