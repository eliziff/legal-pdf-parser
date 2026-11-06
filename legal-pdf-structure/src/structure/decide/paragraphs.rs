//! Each judgment's numbered paragraphs, read from the fenced lines: the streams of its counts and languages, the lines each
//! keeps, and what the reader of its text holds of them.

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
struct NoteWalk {
    state: NoteState,
    last: Option<LastNote>,
    /// The note the labels have counted to.
    count: u32,
    /// The page a quoted list a colon opened runs on: its next numbers are no notes.
    quoted_list: Option<usize>,
}

impl NoteWalk {
    fn open(&self) -> bool {
        self.last.is_some_and(|last| !last.finished)
    }

    /// The note a labelled line leaves the count at: a figure opening a line of a note going on is its words ("2012 ABQB 375").
    fn counted(&self, value: u32) -> u32 {
        if !self.open() || value == 1 || (self.count + 1..=self.count + NOTE_LOST).contains(&value) { value } else { self.count }
    }

    fn counts_on(&self, value: u32) -> bool {
        value == 1 || (self.count + 1..=self.count + NOTE_LOST).contains(&value)
    }

    fn inside(&self) -> bool {
        matches!(self.state, NoteState::Inside(_))
    }

    fn turn_to(&mut self, page: usize) {
        if !matches!(self.state, NoteState::Inside(at) if at == page) {
            self.state = NoteState::Text;
        }
    }

    /// A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation stands
    /// smaller than the body too, "Immunity waived".
    fn carries_on(&self, size: f64, page: usize) -> bool {
        self.last.is_some_and(|last| !last.finished && (size - last.size).abs() <= 0.5 && last.page + 1 == page)
    }

    fn opens_quoted_list(&mut self, page: usize) {
        self.quoted_list = Some(page);
    }

    fn in_quoted_list(&self, page: usize) -> bool {
        self.quoted_list == Some(page)
    }

    /// A note a fence took: it keeps the count, and may go on onto the next page.
    fn fenced(&mut self, label: Option<u32>, text: &str, size: f64, page: usize) {
        self.count = label.map_or(self.count, |value| self.counted(value));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) });
    }

    /// A line the walk read as a note's.
    fn took(&mut self, label: Option<u32>, text: &str, size: f64, page: usize) {
        self.count = label.map_or(self.count, |value| self.counted(value));
        // (A note's line stopping on "s." goes on.)
        let stopped = [" s.", " ss.", " para.", " paras.", " p.", " pp."].iter().any(|shortened| text.trim_end().ends_with(shortened));
        self.last = Some(LastNote { size, page, finished: sentence_ended(text) && !stopped });
        self.state = NoteState::Inside(page);
    }

    /// A line the walk kept as the text's.
    fn left(&mut self) {
        self.state = NoteState::Text;
    }
}

/// The stream a line is read in, apart from the column it stands in (`Fences::translation`): a judgment in one language
/// reads a statute quoted in two columns in its one stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Stream {
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

/// A bold line under an elision is a quotation's emphasis.
fn elided(text: &str) -> bool {
    matches!(text.trim(), "..." | "\u{2026}" | "[...]" | "[\u{2026}]" | ". . .")
}

/// Where a paragraph ends, against what its text alone says: the facts of the page the text cannot see.
struct ParagraphEnd<'a> {
    pages: &'a [Page],
    slots: &'a Slots,
    translation: &'a [bool],
    edge: &'a HashMap<(usize, bool), f64>,
    column: &'a dyn Fn(usize, bool) -> f64,
    smaller: &'a dyn Fn(usize) -> bool,
}

impl ParagraphEnd<'_> {
    // A bold or underlined line at the margin, under a sentence's end, heads what follows, its stop or not ("Impact on
    // the Community."): the text alone cannot say so, and the paragraph ends there. A heading set in a quotation
    // stands further in. (A quoted paragraph's bold number, alone on its line, heads nothing: a heading has words.)
    fn marked(&self, slot: Slot, share: f64) -> bool {
        let (pages, slots) = (self.pages, self.slots);
        let words = &slots.line(pages, slot).words;
        legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > share) || !words.is_empty() && words.iter().all(|word| word.underlined)
    }

    /// The held line a bold heading follows, the paragraph's last.
    fn heading_above(&self, held_lines: &[Slot]) -> Option<usize> {
        let (pages, slots, edge, column, smaller) = (self.pages, self.slots, self.edge, self.column, self.smaller);
        let marked = |slot: Slot, share: f64| self.marked(slot, share);
        held_lines.windows(2).enumerate().position(|(at, pair)| {
            let (before, line) = (slots.line(pages, pair[0]), slots.line(pages, pair[1]));
            // (A page all quotation has its edge there: the margin is no further in than the document's running text.)
            let (page, language) = (slots.page(pair[1]), self.translation[pair[1].index()]);
            let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(column(page, language));
            // (What a heading in a quotation heads stands further in, or smaller: an enactment's "Prohibited grounds". A
            // table's head stands over figures: "Township" / "2012".)
            // (An enactment's marginal note stands over its section's number, its own lines bold or not: "Burden of proof" /
            // "(3) The burden ...", "Publication, etc., of an" / "intimate image without" / "consent" / "162.1 (1) ...".)
            let under_heading = held_lines[(at + 2).min(held_lines.len())..].iter().find(|&&under| !marked(under, 0.8)).copied();
            let section = under_heading.is_some_and(|under| {
                let words = slots.line(pages, under).text.trim_start().trim_start_matches('(');
                words.starts_with(|c: char| c.is_ascii_digit()) && words.split_whitespace().next().is_some_and(|figures| figures.chars().filter(char::is_ascii_digit).count() >= figures.chars().filter(|c| c.is_alphabetic()).count())
            });
            let body_under = held_lines.get(at + 2).is_none_or(|&under| !smaller(under.index()) && slots.line(pages, under).bbox[0] - line.bbox[0] <= QUOTED_INDENT
                && slots.line(pages, under).text.chars().any(char::is_alphabetic)
                // (A run-in label's sentence goes on under it: "Video #3 – Published October 13, 2021 – This 12" / "second video ...".)
                && (marked(under, 0.5) || !slots.line(pages, under).text.trim_start().starts_with(char::is_lowercase)));
            // (A bracket left open goes on: "Les séances du conseil doivent être publiques (art. 322".)
            marked(pair[1], 0.8) && !marked(pair[0], 0.2) && line.text.split_whitespace().count() <= 12 && line.text.chars().any(char::is_alphabetic)
                && sentence_ended(&before.text) && !elided(&before.text) && line.bbox[0] - margin <= QUOTED_INDENT && body_under && !section
                && line.text.matches('(').count() <= line.text.matches(')').count()
        })
    }

    // All in bold from a line at the margin or on an enumerator to the paragraph's end, over a line that is not, is a
    // heading whatever its length or the stop above it. (An italic last line under a sentence's end too.)
    fn marked_tail(&self, held_lines: &[Slot]) -> Option<usize> {
        let (pages, slots, edge, column) = (self.pages, self.slots, self.edge, self.column);
        let marked = |slot: Slot, share: f64| self.marked(slot, share);
        (1..held_lines.len()).find(|&from| {
            let (run, line) = (&held_lines[from..], slots.line(pages, held_lines[from]));
            let (page, language) = (slots.page(held_lines[from]), self.translation[held_lines[from].index()]);
            let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(column(page, language));
            let enumerator = line.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
            let above = slots.line(pages, held_lines[from - 1]);
            // (A heading in italics ends on no stop: "Basi, supra." is the paragraph's source.)
            let italic = run.len() == 1 && !line.text.trim_end().ends_with('.') && legal_pdf_core::flag_share(line, legal_pdf_core::ITALIC).is_some_and(|share| share > 0.8) && sentence_ended(&above.text);
            let bold = |slot: Slot| legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > 0.8);
            // (Emphasis running over a line is the sentence's: "... (b) a deprivation or risk of" / "deprivation.".)
            let wrapped = !sentence_ended(&above.text) && (line.text.trim_start().starts_with(char::is_lowercase) || above.words.last().is_some_and(|word| word.underlined)
                || above.spans.iter().rev().find(|span| !span.text.trim().is_empty()).is_some_and(|span| span.flags & legal_pdf_core::BOLD != 0));
            // (Nor is what a colon sets out a heading, or what follows an elision.)
            let set_out = above.text.trim_end().ends_with(':') || elided(&above.text);
            run.len() <= 24 && run.iter().any(|&held| slots.line(pages, held).text.chars().any(char::is_alphabetic)) && !marked(held_lines[from - 1], 0.2) && (line.bbox[0] - margin <= QUOTED_INDENT || enumerator)
                && !wrapped && !set_out && (run.iter().all(|&held| bold(held)) || italic || run.iter().all(|&held| marked(held, 0.8)) && sentence_ended(&above.text))
        })
    }

    // A quotation's last lines, in its indent, are the paragraph's: the text alone cannot see an indent ("Kelty", a letter's "Per:").
    fn quotation_tail(&self, spans: &mut BTreeMap<Slot, (u32, u32)>, kept: &[Slot], to_number: &[usize], kept_at: &HashMap<Slot, usize>) {
        let (pages, slots) = (self.pages, self.slots);
        if let Some(&last) = spans.keys().next_back() {
            let (bbox, page) = (slots.line(pages, last).bbox, slots.page(last));
            // (Inside a quotation: the line above stands at its indent too, as a first line's does not.)
            let held: Vec<f64> = spans.keys().map(|&slot| slots.line(pages, slot).bbox[0]).collect();
            let left = held.iter().copied().fold(f64::MAX, f64::min);
            if bbox[0] - left > QUOTED_INDENT && held.len() >= 2 && (held[held.len() - 2] - bbox[0]).abs() <= 3.0 {
                let from = kept_at.get(&last).copied().unwrap_or(kept.len());
                let mut row = bbox;
                for at in from + 1..kept.len() {
                    let under = slots.line(pages, kept[at]);
                    let enumerator = under.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
                    let same_row = (under.bbox[1] - row[1]).abs() <= 2.0;
                    if to_number[at] == 0 || slots.page(kept[at]) != page || (under.bbox[0] - bbox[0]).abs() > 3.0 && !same_row || enumerator {
                        break;
                    }
                    row = under.bbox;
                    spans.entry(kept[at]).or_insert((0, under.text.encode_utf16().count() as u32));
                }
            }
        }
    }

    /// The paragraph's held lines, its tail cut or taken back.
    fn settle(&self, spans: &mut BTreeMap<Slot, (u32, u32)>, kept: &[Slot], to_number: &[usize], kept_at: &HashMap<Slot, usize>) {
        let held_lines: Vec<Slot> = spans.keys().copied().collect();
        let heading = self.heading_above(&held_lines);
        if let Some(heading) = heading {
            for line in &held_lines[heading + 1..] {
                spans.remove(line);
            }
        }
        let tail = self.marked_tail(&held_lines);
        if let Some(from) = tail {
            for line in &held_lines[from..] {
                spans.remove(line);
            }
        }
        if heading.is_none() && tail.is_none() {
            self.quotation_tail(spans, kept, to_number, kept_at);
        }
    }
}

impl Fences {
    /// The document's numbered paragraphs, read once, before notes, lists or headings claim a line. Each language's
    /// body is written as a provider writes a decision, a block to a line: a block the lines `read_continuations`
    /// chains, a marker standing alone on its line set before the words of its row (`Evidence::beside`; a margin
    /// number before its row in both languages); the page's furniture, labels, contents and notes (smaller type under
    /// the band, or opening on a note's label) are no part of it. The shared reading of a
    /// decision's text (`judgment_paragraphs`) reads it, and its paragraphs come back to the lines as `count` and `held`.
    /// A figure in far smaller type than the body is a note's label, however it lines up: a page's labels count on as
    /// margin numbers do, and a scan's layer boxes its lines alike. Beside a line in the body's own type it numbers that
    /// line: a tribunal's word processor sets its paragraph numbers small.
    fn note_label(&self, evidence: &Evidence, at: usize, row: Option<Slot>) -> bool {
        evidence.size[at] * 2.0 < self.label_size() && !row.is_some_and(|row| self.body.is_some_and(|(_, size)| evidence.half_points[row.index()] == size))
    }

    fn label_size(&self) -> f64 {
        self.body.map_or(0.0, |(_, size)| f64::from(size) * 0.75)
    }

    /// Each judgment's numbered paragraphs: which lines each number holds, and the line each opens on. Reading and
    /// storing are apart: only the margin's small figures are cleared before it, and the answer is stored once after.
    pub(super) fn read_paragraphs(&mut self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, evidence: &Evidence) {
        for at in 0..slots.len() {
            if self.note_label(evidence, at, self.margin_number[at]) {
                self.margin_number[at] = None;
                self.margin_twin[at] = None;
            }
        }
        (self.count, self.held) = self.paragraph_holds(analysis, pages, slots, evidence);
    }

    /// A line no paragraph holds: a fence took it.
    fn set_apart(&self, at: usize) -> bool {
        matches!(self.role[at], Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents | Role::Note)
    }

    /// The stream each line reads in.
    fn streams(&self, printed: &[Option<usize>]) -> Vec<Stream> {
        // A judgment in one language may quote a statute in two, its columns side by side: the other language numbers
        // nothing, and its lines are the paragraph's quoting them. A parallel report prints its paragraphs in both.
        let numbered_apart = |at: usize| self.margin_twin[at].is_some() || self.translation[at] && (self.margin_number[at].is_some()
            || printed[at].is_some());
        let translated = (0..printed.len()).filter(|&at| self.translation[at] && self.role[at] == Role::Open).count();
        let open = (0..printed.len()).filter(|&at| self.role[at] == Role::Open).count();
        // (A quoted section's own number numbers no count: what counts is how few the other language prints.)
        let printed_number = |at: usize| self.margin_number[at].is_some() || printed[at].is_some();
        let (apart, own) = ((0..printed.len()).filter(|&at| numbered_apart(at)).count(), (0..printed.len()).filter(|&at| !self.translation[at] && printed_number(at)).count());
        let one_text = translated * 4 < open && apart * 4 < own;
        self.translation.iter().map(|&translation| if translation && !one_text { Stream::Translation } else { Stream::Text }).collect()
    }

    /// Each row's marker, standing alone on its line: the margin's numbers by their rows, other markers by theirs.
    fn marker_rows(&self, evidence: &Evidence, slots: &Slots, stream_of: &[Stream]) -> HashMap<(Slot, Stream), (Slot, bool)> {
        let mut marker_of = HashMap::<(Slot, Stream), (Slot, bool)>::new();
        for at in 0..slots.len() {
            let marker = Slot(at as u32);
            // (A marker takes one row: the margin's, where it has one.)
            let rows = [self.margin_number[at], self.margin_twin[at], evidence.beside[at].filter(|&row| !self.note_label(evidence, at, Some(row)) && self.margin_number[at].is_none())];
            for (k, row) in rows.into_iter().enumerate().filter_map(|(k, row)| Some((k, row?))) {
                // A marker's row is the line beside it in its own language (the nearest line on a parallel report's
                // row may be the other column's); only a margin number serves the other language's row.
                if self.set_apart(row.index()) || k != 1 && self.translation[row.index()] != self.translation[at] {
                    continue;
                }
                // The twin row's language takes the margin's number as text the stream adds, not as its line.
                marker_of.entry((row, stream_of[row.index()])).or_insert((marker, k == 1));
            }
        }
        marker_of
    }

    fn paragraph_holds(&self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, evidence: &Evidence) -> (Vec<Option<u32>>, Vec<Vec<(bool, u32, u32, u32)>>) {
        let mut opens_on: Vec<Option<u32>> = vec![None; slots.len()];
        let mut holds: Vec<Vec<(bool, u32, u32, u32)>> = vec![Vec::new(); slots.len()];
        let label_size = self.label_size();
        let layout = analysis;
        // (Each line's printed number, read once.)
        let printed: Vec<Option<usize>> = (0..slots.len()).map(|at| layout.printed_paragraph_number(&slots.line(pages, Slot(at as u32)).text)).collect();
        let stream_of = self.streams(&printed);
        // The most letters and figures a recognizer's reading of a folio holds.
        const FOLIO_MARKS: usize = 4;
        // The most lines a heading over a numbered paragraph runs to.
        const CLOSING_LINES: usize = 4;
        let set_apart = |at: usize| self.set_apart(at);
        let marker_of = self.marker_rows(evidence, slots, &stream_of);
        let attached: HashSet<Slot> = marker_of.values().filter(|(_, added)| !added).map(|(marker, _)| *marker).collect();
        // The body's type, and where on each page its last running line of that type ends (a folio or a running
        // citation, a few words, is none): a page's notes stand below it.
        let below_body = |at: usize, share: f64| evidence.size[at] > 0.0 && self.margin_number[at].is_none() && self.body.is_some_and(|(_, size)| evidence.size[at] * 2.0 < f64::from(size) * share);
        let small = |at: usize| below_body(at, 0.9);
        let smaller = |at: usize| below_body(at, 0.95);
        // (Where a page's note band opens: small type above it is a quotation, whatever it numbers.)
        let (mut body_end, mut band_top) = (vec![[f64::MIN; 2]; pages.len()], vec![[f64::MAX; 2]; pages.len()]);
        for &slot in &self.order {
            let (page, language) = (slots.page(slot), usize::from(self.translation[slot.index()]));
            if self.role[slot.index()] == Role::Open && !smaller(slot.index()) && slots.line(pages, slot).text.split_whitespace().count() > 3 {
                body_end[page][language] = body_end[page][language].max(slots.line(pages, slot).bbox[3]);
            }
            if self.band[slot.index()] {
                band_top[page][language] = band_top[page][language].min(slots.line(pages, slot).bbox[1]);
            }
        }
        // Development only: each line's fence and type, beside the streams.
        if let Some(path) = std::env::var_os("LEGALPDF_JUDGMENT_STREAM") {
            let lines = self.order.iter().map(|&slot| {
                let at = slot.index();
                format!("{:?}\t{}\t{}\t{:.1}\t{:?}\t{}\n", self.role[at], self.band[at], self.translation[at], evidence.size[at], self.body, slots.line(pages, slot).text)
            }).collect::<String>();
            let _ = std::fs::write(std::path::PathBuf::from(path).with_extension("lines.txt"), lines);
        }
        // Where the judgment's own numbers print, in each language: the place, from the left edge of its page's body,
        // where most of its lines that open on a printed paragraph number open. A number printed further in is a
        // quotation's: its line runs on in the paragraph quoting it rather than opening one.
        let body = |at: usize| self.role[at] == Role::Open && !self.band[at] && !set_apart(at);
        let mut edge = HashMap::<(usize, bool), f64>::new();
        for &slot in self.order.iter().filter(|slot| body(slot.index())) {
            let left = edge.entry((slots.page(slot), self.translation[slot.index()])).or_insert(f64::MAX);
            *left = left.min(slots.line(pages, slot).bbox[0]);
        }
        // Where the document's running text stands on a page: most of its lines of words open there, odd pages and even apart.
        let mut lefts = HashMap::<(usize, bool), BTreeMap<i64, usize>>::new();
        for &slot in self.order.iter().filter(|slot| body(slot.index())) {
            let line = slots.line(pages, slot);
            if line.text.split_whitespace().count() >= 5 {
                *lefts.entry((slots.page(slot) % 2, self.translation[slot.index()])).or_default().entry((line.bbox[0] / 3.0).round() as i64).or_default() += 1;
            }
        }
        // (Read once for each page's parity and language.)
        let columns: HashMap<(usize, bool), f64> = lefts.iter().filter_map(|(key, lefts)| lefts.iter().max_by_key(|(_, count)| **count).map(|(left, _)| (*key, *left as f64 * 3.0))).collect();
        let column = |page: usize, language: bool| columns.get(&(page % 2, language)).copied().unwrap_or(f64::MAX);
        // Where a numbered line's words stand, from that edge: a quotation may hang its number in the judgment's own margin
        // (its words alone set in); a number alone on its line has them on its row.
        let indent = |slot: Slot| {
            let line = slots.line(pages, slot);
            let row = self.margin_number[slot.index()].or(evidence.beside[slot.index()]);
            let words = match row {
                Some(row) if evidence.marker[slot.index()].is_some_and(|reading| reading.only) => slots.line(pages, row).bbox[0],
                _ => line.words.get(1).map_or(line.bbox[0], |word| word.bbox[0]),
            };
            words - edge.get(&(slots.page(slot), self.translation[slot.index()])).copied().unwrap_or(0.0)
        };
        let numbered: Vec<Slot> = self.order.iter().copied()
            .filter(|slot| body(slot.index()) && printed[slot.index()].is_some())
            .collect();
        let mut places = BTreeMap::<(bool, i64), usize>::new();
        for &slot in &numbered {
            *places.entry((self.translation[slot.index()], (indent(slot) / 3.0).round() as i64)).or_default() += 1;
        }
        let own_place = [false, true].map(|translation| places.iter().filter(|((own, _), _)| *own == translation).max_by_key(|(_, count)| **count).map(|((_, at), _)| *at as f64 * 3.0));
        let place = |translation: bool| own_place[usize::from(translation)];
        // Words set further in than the judgment's are a quotation's. Words closer to their number say nothing: the judgment's
        // own paragraphs may print so among quotations that print as it does (`quoted_runs` reads those by their counts).
        let further = |slot: Slot| place(self.translation[slot.index()]).is_some_and(|own| indent(slot) - own > QUOTED_INDENT);
        let number_of = |slot: Slot| (self.translation[slot.index()], printed[slot.index()]);
        // A number no line at the judgment's own place prints is the judgment's, however it stands.
        let standing: HashSet<(bool, Option<usize>)> = numbered.iter().copied().filter(|&slot| !further(slot)).map(number_of).collect();
        let indented: HashSet<Slot> = numbered.iter().copied().filter(|&slot| further(slot) && standing.contains(&number_of(slot))).collect();
        // Where each page's text starts and ends.
        let mut page_ends = vec![(f64::MAX, f64::MIN); pages.len()];
        for &slot in &self.order {
            let (ends, bbox) = (&mut page_ends[slots.page(slot)], slots.line(pages, slot).bbox);
            *ends = (ends.0.min(bbox[1]), ends.1.max(bbox[3]));
        }
        let running: HashSet<&str> = self.order.iter().filter(|slot| self.role[slot.index()] == Role::Furniture).map(|&slot| slots.line(pages, slot).text.trim()).collect();
        // Each number alone on its line and the line of words it opens.
        let mut led = Vec::<(Slot, Slot)>::new();
        // A new count opens at a "1" in the body's type and at the judgment's own place, five or more numbers on, that goes
        // on 2, 3, 4, 5: another decision in the file. A note list's "[1]" is small, a quotation's stands elsewhere, a
        // list's stops short.
        let (mut count_of, mut printed_numbers, mut counts) = (vec![[0u16; 2]; slots.len()], [0usize; 2], [0u16; 2]);
        let sequence: [Vec<(Slot, Option<usize>)>; 2] = Stream::BOTH.map(|stream| {
            numbered.iter().copied().filter(|slot| stream_of[slot.index()] == stream)
                .map(|slot| (slot, printed[slot.index()])).collect()
        });
        let place_in: HashMap<Slot, usize> = sequence.iter().flat_map(|numbers| numbers.iter().enumerate().map(|(place, (slot, _))| (*slot, place))).collect();
        // Notes listed after the reasons in the body's own type open no decision: the reasons' lines call their numbers in
        // turn ("... of violence.[1]"), and the list is no one's paragraphs.
        let called = |upto: usize, stream: Stream| {
            let mut calls = [1, 2, 3].map(|number| format!("[{number}]")).into_iter().peekable();
            for slot in self.order[..upto].iter().filter(|slot| stream_of[slot.index()] == stream) {
                let line = slots.line(pages, *slot).text.trim();
                while calls.peek().is_some_and(|call| line.find(call.as_str()).is_some_and(|at| at > 0)) {
                    calls.next();
                }
            }
            calls.peek().is_none()
        };
        let mut note_lists = HashSet::<(u16, Stream)>::new();
        for (reading_place, &slot) in self.order.iter().enumerate() {
            let (at, language) = (slot.index(), stream_of[slot.index()].index());
            if let Some(&place) = place_in.get(&slot) {
                let numbers = &sequence[language];
                let goes_on = (1..5).all(|step| numbers.get(place + step).is_some_and(|(_, number)| *number == Some(step + 1)));
                if numbers[place].1 == Some(1) && goes_on && !smaller(at) && !further(slot) && printed_numbers[language] >= 5 {
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
        for (document, stream) in (0..streams).flat_map(|count| Stream::BOTH.map(|stream| (count, stream))).filter(|stream| !note_lists.contains(stream)) {
            // The stream, and each of its UTF-16 units' line and place in that line's text (none for what the
            // stream adds: breaks, a margin number set before the other language's row).
            let (mut text, mut place) = (Vec::<u16>::new(), Vec::<Option<(Slot, u32)>>::new());
            let write = |text: &mut Vec<u16>, place: &mut Vec<Option<(Slot, u32)>>, words: &str, line: Option<Slot>| {
                for (offset, unit) in words.encode_utf16().enumerate() {
                    text.push(if unit == 10 { 32 } else { unit });
                    place.push(line.map(|line| (line, offset as u32)));
                }
            };
            let mut kept = Vec::<Slot>::new();
            let mut notes = NoteWalk::default();
            for (reading_place, &slot) in self.order.iter().enumerate() {
                let at = slot.index();
                // The notes a fence took keep the count, and may go on onto the next page: a note left among the words goes on from them.
                if self.role[at] == Role::Note && stream_of[at] == stream {
                    notes.fenced(evidence.label[at].map(|label| label.value), &slots.line(pages, slot).text, evidence.size[at], slots.page(slot));
                }
                if count_of[at][stream.index()] != document || stream_of[at] != stream || set_apart(at) || attached.contains(&slot) {
                    continue;
                }
                let line = &slots.line(pages, slot).text;
                // A figure in the margin beside no words, and out of the count's turn, numbers nothing: a heading's call on a row of its own.
                let out_of_turn = place_in.get(&slot).is_some_and(|&place| {
                    let numbers = &sequence[stream_of[at].index()];
                    let here = numbers[place].1;
                    let last = place.checked_sub(1).and_then(|before| numbers[before].1);
                    let next = numbers.get(place + 1).and_then(|(_, number)| *number);
                    here != last.map(|number| number + 1) && next != here.map(|number| number + 1)
                });
                if out_of_turn && evidence.beside[at].is_none() && self.margin_number[at].is_none() && self.margin_twin[at].is_none()
                    && evidence.marker[at].is_some_and(|reading| reading.only) && slots.line(pages, slot).bbox[2] < column(slots.page(slot), self.translation[at]) {
                    continue;
                }
                // A folio that lost its figures ("-  -") is no one's words; nor is a recognizer's reading of one, a few marks
                // at the very top or foot of its page ("4-", "=f>~").
                if !line.chars().any(char::is_alphanumeric) && line.chars().all(|c| c.is_whitespace() || matches!(c, '-' | '\u{2013}' | '\u{2014}')) {
                    continue;
                }
                let (bbox, (top, foot)) = (slots.line(pages, slot).bbox, page_ends[slots.page(slot)]);
                // (Set apart from the page's lines: a sentence's last word may stand alone at its foot, "thew", "692,".)
                let neighbour = |step: isize| self.order.get(reading_place.wrapping_add_signed(step)).filter(|other| slots.page(**other) == slots.page(slot)).map(|other| slots.line(pages, *other).bbox);
                let height = bbox[3] - bbox[1];
                let apart = if bbox[3] >= foot { neighbour(-1).is_none_or(|above| bbox[1] - above[3] > height) } else { neighbour(1).is_none_or(|below| below[1] - bbox[3] > height) };
                if slots.line(pages, slot).recognized() && (bbox[1] <= top || bbox[3] >= foot) && apart && line.chars().filter(|c| c.is_alphanumeric()).count() <= FOLIO_MARKS {
                    continue;
                }
                // A folio in words ("Page: 3"), and what the page's furniture prints elsewhere, are the page's.
                let edge_line = bbox[1] <= top + (bbox[3] - bbox[1]) * 3.0 || bbox[3] >= foot - (bbox[3] - bbox[1]) * 3.0;
                let folio = line.trim().strip_prefix("Page").or_else(|| line.trim().strip_prefix("PAGE")).is_some_and(|rest| {
                    let rest = rest.trim_start_matches([':', ' ']);
                    !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit())
                });
                if edge_line && (folio || running.contains(line.trim())) {
                    continue;
                }
                // A note is known by its smaller type: small under the page's note band; or, wherever the band stands,
                // set smaller than the body and opening on a note's label below the page's last line in the body's type,
                // or carrying such a note on, its next line or the lines below the body after it (a label standing alone
                // on its line, its words on the next). A quotation is set small too, among the body's lines, and is the
                // paragraph's.
                let (top, language) = (slots.line(pages, slot).bbox[1], stream.index());
                let band = band_top[slots.page(slot)][language];
                let below = top >= body_end[slots.page(slot)][language] - 1.0 && (band == f64::MAX || top >= band);
                notes.turn_to(slots.page(slot));
                // Notes count on from 1, each page's or the document's: a number that goes on with no count is a small
                // quotation's at the page's foot ("75 Along the same lines", "2 In this Act,").
                // A figure alone, far smaller than the body, is a label a scan's layer set on its own line.
                let lone_figure = || line.trim().parse::<u32>().ok().filter(|_| evidence.size[at] * 2.0 < label_size);
                let label = evidence.label[at].filter(|label| label.suppress == 0).map(|label| label.value).or_else(lone_figure);
                // (Set in a note's type, or raised: a quotation's own numbers run on at the foot of a page in its larger one.)
                // (Or a tab after its label: "1\tHighway 5 runs ...", some reports' notes set little under the body.)
                let tabbed = smaller(at) && line.trim_start().trim_start_matches(|c: char| c.is_ascii_digit()).starts_with('\t');
                let set_as_note = below_body(at, 0.85) || tabbed || evidence.label[at].is_some_and(|label| label.raised);
                // (A colon over it sets a quotation out: "... as follows:" / "1. The party ...".)
                // (A note may stand under one all the same: its label a lone small figure, or raised.)
                let lead_in = kept.last().is_some_and(|last| slots.line(pages, *last).text.trim_end().ends_with(':')) && lone_figure().is_none() && !evidence.label[at].is_some_and(|label| label.raised);
                if lead_in && label.is_some() {
                    notes.opens_quoted_list(slots.page(slot));
                }
                let lead_in = lead_in || notes.in_quoted_list(slots.page(slot));
                // (A table's cell is no note's label.)
                let lead_in = lead_in || self.role[at] == Role::Table;
                let labelled = below && set_as_note && !lead_in && label.is_some_and(|value| notes.counts_on(value));
                // A scan's layer may read a label standing alone after its note's first line.
                let label_after = self.order.get(reading_place + 1).is_some_and(|next| {
                    evidence.size[next.index()] * 2.0 < label_size && evidence.label[next.index()].is_some() && slots.line(pages, *next).text.trim().chars().count() <= 4
                });
                let labelled = labelled || below && label_after && set_as_note && !lead_in;
                // Under the page's rule a label makes a note in any type: some set their notes in the body's.
                // A note a page's foot left unfinished goes on at the next page's, with no label, in its own type: a quotation
                // stands smaller than the body too, "Immunity waived".
                let carried = smaller(at) && below && notes.carries_on(evidence.size[at], slots.page(slot));
                let in_note = self.band[at] && (small(at) || label.is_some() || notes.inside())
                    || smaller(at) && (labelled || notes.inside() && (self.continues[at].is_some() || below)) || carried;
                if !in_note {
                    notes.left();
                }
                if in_note {
                    notes.took(label, line, evidence.size[at], slots.page(slot));
                } else if evidence.size[at] * 2.0 >= label_size || evidence.size[at] == 0.0 || analysis.editorial_line(line)
                    || slots.line(pages, slot).recognized() && (!edge_line || kept.last().is_some_and(|last| !sentence_ended(&slots.line(pages, *last).text))) {
                    // (Words far smaller than the body and in no note are a page's aside, "OUR VISION...": but a size no one
                    // measured, an editor's "[Emphasis added]", and on a scan a line among the page's lines, whose layer boxes
                    // lines by their letters, are the text's.)
                    // (A letterhead, far larger than the body, is no one's: "Federal Court" / "of Appeal" at twice its type.)
                    let wordmark = self.body.is_some_and(|(_, size)| evidence.size[at] * 2.0 >= f64::from(size) * 1.6) && line.trim().chars().count() > 1 && line.split_whitespace().count() <= 6;
                    // (A layer may print one line twice, the second a suffix of the first in its box.)
                    let repeated = kept.last().is_some_and(|last| {
                        let above = slots.line(pages, *last);
                        slots.page(*last) == slots.page(slot) && (above.bbox[1] - bbox[1]).abs() <= 2.0 && line.trim().chars().count() >= 12 && above.text.trim_end().ends_with(line.trim())
                    });
                    if !wordmark && !repeated {
                        kept.push(slot);
                    }
                }
            }
            // A table's reading may take prose set about it out of its order ("[4] This appeal ..." under its own next lines):
            // what a table holds reads down its page.
            let mut from = 0;
            while from < kept.len() {
                let page = slots.page(kept[from]);
                let to = from + kept[from..].iter().take_while(|slot| self.role[slot.index()] == Role::Table && slots.page(**slot) == page).count();
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
            // The printed paragraph each line stands in, as the page reader reads a page (its own leading, first lines
            // set in, a short last line, a scan's blanks) and a paragraph over a page (`continues_onto`).
            let mut runs = Vec::<&[Slot]>::new();
            let mut from = 0;
            for at in 1..=kept.len() {
                if at == kept.len() || slots.page(kept[at]) != slots.page(kept[from]) {
                    runs.push(&kept[from..at]);
                    from = at;
                }
            }
            let read: Vec<Vec<&Line>> = runs.iter().map(|run| run.iter().map(|&slot| slots.line(pages, slot)).collect()).collect();
            let measure = crate::paragraphs::measure_of(&read);
            let mut printed = HashMap::<Slot, (usize, usize)>::new();
            let mut flows = HashSet::<Slot>::new();
            for (k, run) in runs.iter().enumerate() {
                let found = crate::paragraphs::paragraphs_of(&read[k], &vec![true; run.len()], measure);
                printed.extend(run.iter().zip(&found).filter_map(|(&slot, paragraph)| Some((slot, (k, (*paragraph)?)))));
                if k > 0 && crate::paragraphs::continues_onto(&read[k - 1], &read[k]) {
                    flows.insert(run[0]);
                }
            }
            // How many lines each line stands before the next that opens on a number (none after the last).
            let mut to_number = vec![usize::MAX; kept.len()];
            let mut number = None::<usize>;
            for at in (0..kept.len()).rev() {
                if evidence.marker[kept[at].index()].is_some() || marker_of.contains_key(&(kept[at], stream)) {
                    number = Some(at);
                }
                to_number[at] = number.map_or(usize::MAX, |number| number - at);
            }
            // A first paragraph's number fused to its judge's drop capital: "1I" / "ACOBUCCI J. (dissenting) -- ...".
            let drop_capital = |at: usize| {
                let words = slots.line(pages, kept[at]).text.trim();
                let mut letters = words.chars();
                letters.next() == Some('1') && letters.next().is_some_and(char::is_uppercase) && letters.next().is_none()
                    && kept.get(at + 1).is_some_and(|next| slots.line(pages, *next).text.chars().filter(|c| !c.is_whitespace()).take(3).filter(|c| c.is_uppercase()).count() == 3)
            };
            let (mut previous, mut last_advance) = (None::<Slot>, 0.0f64);
            for (place_in_kept, &slot) in kept.iter().enumerate() {
                let at = slot.index();
                let line = &slots.line(pages, slot).text;
                let marker = marker_of.get(&(slot, stream)).copied();
                if let Some(before) = previous {
                    let (above, here) = (slots.line(pages, before), slots.line(pages, slot));
                    // A marker standing alone on its line opens the line after it, whatever its row, a line of words set
                    // no further left: "2025." ending a wrapped sentence opens no heading at the margin.
                    // (Figures only: a heading's "I." alone in a gutter opens the heading, not the next line read.)
                    let figures = above.text.trim().chars().any(|c| c.is_ascii_digit());
                    // (Nor does a range's end wrapped over its line: "at paras. 11-" / "13.".)
                    let wrapped = place_in_kept >= 2 && slots.line(pages, kept[place_in_kept - 2]).text.trim_end().ends_with(['-', '\u{2013}']);
                    let alone = figures && !wrapped && evidence.marker[before.index()].is_some_and(|reading| reading.only) && evidence.marker[at].is_none()
                        && above.bbox[0] <= here.bbox[0] + (here.bbox[3] - here.bbox[1]);
                    // A line goes on with the printed paragraph it stands in, unless it opens on a marker of its own.
                    let same = evidence.marker[at].is_none() && (flows.contains(&slot) || printed.get(&slot).is_some_and(|paragraph| printed.get(&before) == Some(paragraph)));
                    // The last few lines before a number, under a line ending a sentence, stand on their own, whatever the
                    // page's geometry says: whether they head what follows is the reader's to say (`trailing_headings`).
                    // A line in the body's own type ending as a sentence or a lead-in does is the paragraph's words, and so are the
                    // lines of a sentence that wraps there ("Finally, would admission of the evidence bring the" / "administration of
                    // justice into disrepute?").
                    let in_body = |slot: &&Slot| self.body == Some((evidence.face[slot.index()], evidence.half_points[slot.index()]));
                    let stops = |slot: &Slot| slots.line(pages, *slot).text.trim_end().trim_end_matches([')', '"', '\u{201d}', '\u{bb}']).ends_with(['.', '?', '!', ':']);
                    let plain = kept[place_in_kept..].iter().take(CLOSING_LINES.min(to_number[place_in_kept])).take_while(in_body).any(stops);
                    let closes = sentence_ended(&above.text) || above.text.trim_end().ends_with([')', ']']);
                    let apart = closes && !plain && to_number[place_in_kept] <= CLOSING_LINES;
                    // After the last number a line at the same place and leading goes on with the paragraph, however short
                    // the line above it stopped: only a real break lets the last paragraph end before the signatures.
                    let advance = here.bbox[1] - above.bbox[1];
                    let level = (here.bbox[0] - above.bbox[0]).abs() <= here.bbox[3] - here.bbox[1];
                    // (Asked only where it can tell: after the last number, or under a sentence's end.)
                    let closing = (to_number[place_in_kept] == usize::MAX || closes) && (analysis.signature_line(line) || analysis.end_matter(line) || analysis.author_line(line) || analysis.date_line(line) || analysis.disposition_line(line));
                    let last_goes_on = to_number[place_in_kept] == usize::MAX && level && !closing && last_advance > 0.0 && advance > 0.0 && advance <= last_advance * 1.15;
                    let same = same || last_goes_on;
                    if level && advance > 0.0 {
                        last_advance = advance;
                    }
                    // The lines closing the last paragraph stand on their own, for the reader to end it there.
                    // (Under a sentence's end they do wherever they stand: a schedule's numbers may follow.)
                    // (An author's line stands on its own wherever it does: a column's last word may have gone over the gutter.)
                    // (A signer's name over the title under it too: "... signée en son nom par Claude Roy" / "Vice-president".)
                    let signer = line.split_whitespace().count() <= 4 && line.split_whitespace().all(|word| word.starts_with(char::is_uppercase)) && !line.trim_end().ends_with('.') && kept.get(place_in_kept + 1).is_some_and(|next| analysis.signature_line(&slots.line(pages, *next).text));
                    let apart = apart || (closing || signer) && (to_number[place_in_kept] == usize::MAX || closes) || analysis.author_line(line);
                    let fused = place_in_kept > 0 && drop_capital(place_in_kept - 1);
                    let joined = marker.is_none() && (alone || fused || !apart && (same || self.continues[at].is_some())) || indented.contains(&slot);
                    if alone && marker.is_none() {
                        led.push((before, slot));
                    }
                    // A break between blocks, which a line's own text never holds (`write` makes its breaks spaces).
                    text.push(if joined { 32 } else { 10 });
                    place.push(None);
                }
                previous = Some(slot);
                if let Some((marker, added)) = marker {
                    write(&mut text, &mut place, slots.line(pages, marker).text.trim_end(), (!added).then_some(marker));
                    write(&mut text, &mut place, " ", None);
                }
                if drop_capital(place_in_kept) {
                    let (figure, capital) = line.trim().split_at(1);
                    write(&mut text, &mut place, figure, Some(slot));
                    write(&mut text, &mut place, " ", None);
                    write(&mut text, &mut place, capital, Some(slot));
                } else {
                    write(&mut text, &mut place, line, Some(slot));
                }
            }
            let text = String::from_utf16_lossy(&text);
            if text.trim().is_empty() {
                continue;
            }
            // Development only: the stream the reader takes.
            if let Some(path) = std::env::var_os("LEGALPDF_JUDGMENT_STREAM") {
                let _ = std::fs::write(std::path::PathBuf::from(path).with_extension(if stream.is_translation() { "fr.txt" } else { "txt" }), &text);
            }
            let kept_at: HashMap<Slot, usize> = kept.iter().enumerate().map(|(at, slot)| (*slot, at)).collect();
            let ends = ParagraphEnd { pages, slots, translation: &self.translation, edge: &edge, column: &column, smaller: &smaller };
            for (number, ranges) in analysis.judgment_paragraphs(&text) {
                let number = number as u32;
                let mut spans = BTreeMap::<Slot, (u32, u32)>::new();
                let mut first = None;
                // A number alone on its line, before the words its range opens at, is the paragraph's.
                let opening = ranges.first().and_then(|range| place[..range.start.min(place.len())].iter().rev().flatten().next().copied());
                for range in ranges {
                    for &(slot, offset) in place.get(range.start..range.end.min(place.len())).unwrap_or_default().iter().flatten() {
                        first.get_or_insert(slot);
                        let span = spans.entry(slot).or_insert((offset, offset + 1));
                        *span = (span.0.min(offset), span.1.max(offset + 1));
                    }
                }
                if let Some((slot, _)) = opening {
                    let words = slots.line(pages, slot).text.trim();
                    let digits = words.trim_matches(['[', ']', '(', ')', '.', ' ']);
                    // (Unless the paragraph before holds it: "101." ending a citation just above the number's row.)
                    if digits.parse::<u32>().ok() == Some(number) && holds[slot.index()].is_empty() {
                        spans.entry(slot).or_insert((0, words.encode_utf16().count() as u32));
                    }
                }
                ends.settle(&mut spans, &kept, &to_number, &kept_at);
                if let Some(first) = first {
                    opens_on[first.index()].get_or_insert(number);
                }
                for (slot, (start, end)) in spans {
                    holds[slot.index()].push((stream.is_translation(), number, start, end));
                }
            }
        }
        // A number standing alone beside the row it opens, or over it, is that paragraph's, in every language the row prints it.
        let rows = (0..slots.len()).flat_map(|at| {
            [self.margin_twin[at], self.margin_number[at], evidence.beside[at]].into_iter().flatten().map(move |row| (Slot(at as u32), row))
        });
        for (number_line, row) in rows.chain(led).collect::<Vec<_>>() {
            let at = number_line.index();
            let opened: Vec<(bool, u32)> = holds[row.index()].iter().filter(|held| opens_on[row.index()] == Some(held.1)).map(|held| (held.0, held.1)).collect();
            let length = slots.line(pages, number_line).text.encode_utf16().count() as u32;
            // (A heading's enumerator on the row, "i.", is no number of the paragraph's.)
            if !slots.line(pages, number_line).text.chars().any(|c| c.is_ascii_digit()) {
                continue;
            }
            for (translation, number) in opened {
                if !holds[at].iter().any(|held| held.0 == translation && held.1 == number) {
                    holds[at].push((translation, number, 0, length));
                }
            }
        }
        // An enactment quoted in both languages inside reasons in one: on a page none of whose lines in a language any paragraph
        // holds, that language's lines beside a paragraph are that paragraph's.
        for page_slot in 0..pages.len() {
            let on: Vec<usize> = slots.on(page_slot).map(|slot| slot as usize).filter(|&at| self.role[at] == Role::Open && !self.band[at]).collect();
            for language in [false, true] {
                let own: Vec<usize> = on.iter().copied().filter(|&at| self.translation[at] == language).collect();
                if own.iter().any(|&at| !holds[at].is_empty()) {
                    continue;
                }
                let mut spans = BTreeMap::<(bool, u32), (f64, f64)>::new();
                for &at in on.iter().filter(|&&at| self.translation[at] != language) {
                    let bbox = slots.line(pages, Slot(at as u32)).bbox;
                    for held in &holds[at] {
                        let span = spans.entry((held.0, held.1)).or_insert((bbox[1], bbox[3]));
                        *span = (span.0.min(bbox[1]), span.1.max(bbox[3]));
                    }
                }
                for at in own {
                    let line = slots.line(pages, Slot(at as u32));
                    let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
                    if let Some((&(translation, number), _)) = spans.iter().find(|(_, span)| span.0 <= middle && middle <= span.1) {
                        holds[at].push((translation, number, 0, line.text.encode_utf16().count() as u32));
                    }
                }
            }
        }
        (opens_on, holds)
    }
}
