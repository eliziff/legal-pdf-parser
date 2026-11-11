//! The parser's moves and state. Each line in reading order makes one move: it opens a unit of a
//! region or goes on with the one open in its stream, and a line that opens may count on its marker
//! (one of its grammar readings) or on a note label; a heading opens at a level. What is open is kept
//! per reading context (the stack of numbered units, family, value and region: PART II > 12 > (2) >
//! (a); the last note opened and the reference marks the body set that no note has answered, with the
//! line that set each). Applying a move moves the state and says what the move made of the line: the reference it
//! paired.

use std::collections::HashMap;
use crate::fences::Fences;
use crate::machine::lines::LineTable;
use crate::model::{Evidence, MarkerReading, Slot};
use crate::model::regions::{LINE_LABEL, NOTE, PAGE_LABEL, QUOTE, TEXT};

/// Regions whose lines set reference marks notes answer: the body's, and headings' and titles'.
pub(crate) const REFERRING: [u8; 8] = [1, 2, 3, 4, 8, 9, 14, 16];
/// The body regions that take a depth (the tagger's ranked regions, headings aside).
pub(crate) const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];
/// A stack holds at most this many open units; headings run nine levels deep.
const DEPTH: usize = 8;

/// One line's move: its region, whether it opens a unit, what the unit counts on (one of the line's
/// grammar readings, `reading`, an index; or a note's number, `note`), and a heading's level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Move {
    pub(crate) region: u8,
    pub(crate) opens: bool,
    pub(crate) reading: Option<u8>,
    pub(crate) note: Option<u32>,
    pub(crate) level: i8,
    /// A body unit's depth in its document's body (0 for none): the move's own, not its numbering's,
    /// which is evidence for it.
    pub(crate) depth: u8,
    /// Goes on with the unit open in its stream before another region's took the stream (a paragraph
    /// picked up past a block quotation or a page's notes).
    pub(crate) resume: bool,
    /// Goes on with the unit whose first line is this place, whichever unit its stream holds open (two units
    /// read in turn: a form's labels and their values): over `resume`, which goes on with the last unit of its
    /// region in its stream.
    pub(crate) resumes: Option<u32>,
    /// The line's place in the reading the parse makes, where it reads the page in another order than the
    /// parser's lines (two columns read one after the other, a form's labels with their values).
    pub(crate) rank: Option<u32>,
    /// Opens a constituent document of the source (an exhibit, a schedule, a message an email quotes):
    /// its contexts, outline and numbering begin anew.
    pub(crate) starts: bool,
    /// A note's opening: the reference marks it answers, each its line's place and its first and past-last
    /// character, where the reading chose them, wherever they stand (none: the marks of its number the text set
    /// last; a place of `u32::MAX` alone: no mark).
    pub(crate) answers: [Option<(u32, u32, u32)>; 4],
    /// Tier 0's opening: the run member the unit opens on (a run and its member), and the further markers its line
    /// opens on, each a unit under it ("2(1) The Court has ...": the section and its subsection).
    pub(crate) member: Option<(u32, u32)>,
    pub(crate) within: Vec<(u32, u32)>,
}

impl Move {
    /// A move making nothing but its region and whether it opens.
    pub(crate) fn plain(region: u8, opens: bool) -> Self {
        Self { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, resumes: None, rank: None, starts: false, answers: [None; 4], member: None, within: Vec::new() }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Frame {
    pub(crate) family: String,
    pub(crate) value: u32,
    pub(crate) region: u8,
}

/// What is open in one reading context.
#[derive(Clone, Debug, Default)]
pub(crate) struct Open {
    pub(crate) stack: Vec<Frame>,
    pub(crate) note: Option<u32>,
    /// The reference marks the body set that no note has answered: each one's value, the line that set
    /// it, and where it stands in that line (first and past-last character).
    pub(crate) unanswered: Vec<(u32, u32, u32, u32)>,
    /// Whether the context reads a quotation (a quote opened and no paragraph of its own text since),
    /// and the quoted text's own numbering, which the quotation counts on apart from the text's.
    pub(crate) quoting: bool,
    pub(crate) quoted: Vec<Frame>,
}

impl Open {
    /// The numbering the context counts on: a quotation's own while it reads one.
    pub(crate) fn frames(&self) -> &Vec<Frame> {
        if self.quoting { &self.quoted } else { &self.stack }
    }

    fn frames_mut(&mut self) -> &mut Vec<Frame> {
        if self.quoting { &mut self.quoted } else { &mut self.stack }
    }

    /// A unit opening in `region`: a quote opens a quotation the context reads until a paragraph of the text's own opens.
    pub(crate) fn opens(&mut self, region: u8) {
        if region == QUOTE {
            if !self.quoting {
                self.quoted.clear();
            }
            self.quoting = true;
        } else if region == TEXT {
            self.quoting = false;
        }
    }

    /// The numbering the context counts on, after `reading` counts in `region` (`counted`).
    pub(crate) fn count(&mut self, reading: &MarkerReading, region: u8) {
        let frames = counted(self.frames(), reading, region);
        *self.frames_mut() = frames;
    }
}

/// The parse so far: the constituent document read and what is open in each of its contexts.
#[derive(Clone, Debug, Default)]
pub(crate) struct State {
    pub(crate) document: u16,
    pub(crate) streams: HashMap<u16, Open>,
}

/// A line as the state reads it: its place in reading order, and what holds what is read of it, borrowed: the
/// evidence, the fences and the line table (`LineTable`).
#[derive(Clone, Copy)]
pub(crate) struct LineView<'a> {
    pub(crate) place: u32,
    pub(crate) evidence: &'a Evidence,
    pub(crate) fences: &'a Fences,
    pub(crate) table: &'a LineTable,
}

/// How a move stands to the units of its stream.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stands {
    /// Opens a unit at its line.
    Opens,
    /// Goes on with the unit open in its stream, whose first line this is.
    GoesOn(usize),
    /// Takes up another unit, whose first line this is.
    Resumes(usize),
}

/// A document's units as the moves make them, each by its first line: the one open in each context's stream, the last
/// of each region in each stream, which a line may resume, and each one's region.
#[derive(Default)]
pub(crate) struct Streams {
    open: HashMap<(u16, usize), usize>,
    last_of_region: HashMap<(u16, usize, u8), usize>,
    regions: HashMap<usize, u8>,
}

impl Streams {
    /// How `chosen` stands in the stream `key`. A transcript's line number and a page's label are a line each.
    pub(crate) fn stands(&self, key: (u16, usize), chosen: &Move) -> Stands {
        let single = matches!(chosen.region, LINE_LABEL | PAGE_LABEL);
        match (chosen.resumes, self.open.get(&key)) {
            (Some(first), _) => Stands::Resumes(first as usize),
            (_, Some(&first)) if !single && !chosen.opens && self.regions.get(&first) == Some(&chosen.region) => Stands::GoesOn(first),
            _ if !single && chosen.resume => self.last_of_region.get(&(key.0, key.1, chosen.region)).map_or(Stands::Opens, |&first| Stands::Resumes(first)),
            _ => Stands::Opens,
        }
    }

    /// The stream after the move at `line`.
    pub(crate) fn step(&mut self, key: (u16, usize), region: u8, stands: Stands, line: usize) {
        match stands {
            Stands::Opens => {
                self.open.insert(key, line);
                self.last_of_region.insert((key.0, key.1, region), line);
                self.regions.insert(line, region);
            }
            Stands::Resumes(first) => {
                self.open.insert(key, first);
            }
            Stands::GoesOn(_) => {}
        }
    }
}

/// A move's `reading` that counts on the number the margin prints beside the line's row (`LineView::margin`),
/// not on one of the line's own.
pub(crate) const MARGIN: u8 = u8::MAX;
/// A move's `reading` that opens on the line's marker where no reading makes it out: its opening token, counting on
/// nothing.
pub(crate) const OPENING: u8 = u8::MAX - 1;

impl<'a> LineView<'a> {
    /// Every line of the document, in reading order.
    pub(crate) fn all(evidence: &'a Evidence, fences: &'a Fences, table: &'a LineTable) -> Vec<Self> {
        (0..fences.order.len()).map(|k| Self { place: k as u32, evidence, fences, table }).collect()
    }

    fn k(&self) -> usize {
        self.place as usize
    }

    /// The line's slot.
    pub(crate) fn slot(&self) -> Slot {
        self.fences.order[self.k()]
    }

    /// Whether the line reads in a parallel translation's column.
    pub(crate) fn translation(&self) -> bool {
        self.fences.at(self.slot().index()).translation
    }

    /// The line's marker's readings (`LineTable::readings`).
    pub(crate) fn readings(&self) -> &'a [MarkerReading] {
        &self.table.readings[self.k()]
    }

    /// The value of the note label the line opens with.
    pub(crate) fn label(&self) -> Option<u32> {
        self.evidence.label[self.slot().index()].map(|label| label.value)
    }

    /// The reference marks the line sets (`LineTable::references`).
    pub(crate) fn references(&self) -> &'a [(u32, u32, u32)] {
        &self.table.references[self.k()]
    }


    /// The paragraph number the margin prints beside the line's row, numbering the paragraph the line opens in its
    /// column (`Fences::read_margin_numbers`: in both columns of a parallel report): the margin line's place and its
    /// number's reading. A move counts on it as `MARGIN`.
    pub(crate) fn margin(&self) -> Option<(u32, &'a MarkerReading)> {
        let margin = *self.table.margins.get(&self.k())?;
        Some((margin as u32, self.table.readings[margin].first()?))
    }

    /// The paragraph number the decide stage's judgment reader reads the line opening (`Fences::paragraphs`): t0's
    /// reading of the document's numbered paragraphs, which DOC's decoding works from.
    pub(crate) fn count(&self) -> Option<u32> {
        self.fences.paragraphs.count[self.slot().index()]
    }

    /// The reading a move's `reading` names: one of the line's own, the margin's number (`MARGIN`), or none (`OPENING`).
    pub(crate) fn reading(&self, reading: Option<u8>) -> Option<&'a MarkerReading> {
        match reading? {
            OPENING => None,
            MARGIN => self.margin().map(|(_, reading)| reading),
            r => self.readings().get(usize::from(r)),
        }
    }

    /// The reading a unit opening on the line's marker counts on: the margin's number where the margin numbers it, in
    /// either column; else the number t0 counts it opening on; else `next`, the reader's own pick among the readings (the
    /// run member's for tier 0, the open stack's next number for DOC); else the line's first; a marker no reading makes
    /// out (a recognizer's "(2]") its opening token.
    pub(crate) fn opening(&self, next: impl FnOnce() -> Option<usize>) -> u8 {
        let readings = self.readings();
        let counted = self.count().and_then(|count| readings.iter().position(|reading| reading.value == count));
        match (self.margin(), counted.or_else(next), readings.is_empty()) {
            (Some(_), ..) => MARGIN,
            (_, Some(at), _) => at as u8,
            (_, _, false) => 0,
            _ => OPENING,
        }
    }
}

/// What a move made of its line: the earlier line whose reference mark its note answered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Made {
    pub(crate) paired: Option<(u32, u32, u32)>,
    /// The other marks of the note's number it answered too (a note several places in the text cite).
    pub(crate) also: [Option<(u32, u32, u32)>; 3],
}

/// The stack after a reading counts: popped to the open frame of its family and replaced, else pushed.
fn counted(stack: &[Frame], reading: &MarkerReading, region: u8) -> Vec<Frame> {
    let frame = Frame { family: reading.family.clone(), value: reading.value, region };
    let mut after = stack.to_vec();
    match after.iter().rposition(|open| open.family == reading.family) {
        Some(depth) => {
            after.truncate(depth);
            after.push(frame);
        }
        None => {
            after.push(frame);
            if after.len() > DEPTH {
                after.remove(0);
            }
        }
    }
    after
}

impl State {
    /// The context a line reads in: its document and whether it is a translation's (a quotation in it is
    /// read as the parser reads it, not as the fences part it); after `starts`, the next document's.
    pub(crate) fn ctx(&self, line: &LineView, starts: bool) -> u16 {
        (self.document + u16::from(starts)).saturating_mul(2) + u16::from(line.translation())
    }

    /// What is open in the context `line` reads in (after `starts`, the next document's), where anything is.
    pub(crate) fn open(&self, line: &LineView, starts: bool) -> Option<&Open> {
        self.streams.get(&self.ctx(line, starts))
    }

    /// The depth a body opening's numbering puts it at: the body units open on its context's stack once
    /// its reading counts (the stack as it stands for an unnumbered one), one at least.
    pub(crate) fn stack_depth(&self, line: &LineView, candidate: &Move) -> usize {
        let empty = Open::default();
        let stream = self.open(line, candidate.starts).unwrap_or(&empty);
        let body = |frames: &[Frame]| frames.iter().filter(|f| RANKED_BODY.contains(&f.region)).count();
        match line.reading(candidate.reading) {
            Some(reading) => body(&counted(stream.frames(), reading, candidate.region)),
            None => body(stream.frames()),
        }
        .max(1)
    }

    /// The state after `line` makes `chosen`, and what the move made of the line.
    pub(crate) fn apply(&mut self, line: &LineView, chosen: &Move) -> Made {
        let mut made = Made::default();
        if chosen.starts {
            self.document += 1;
        }
        let ctx = self.ctx(line, false);
        let stream = self.streams.entry(ctx).or_default();
        // The line making the move reads in the context it leaves the state in (a quote opening a quotation).
        if chosen.opens {
            stream.opens(chosen.region);
        }
        if let Some(reading) = line.reading(chosen.reading) {
            stream.count(reading, chosen.region);
        }
        if let (true, NOTE, Some(value)) = (chosen.opens, chosen.region, chosen.note) {
            stream.note = Some(value);
            if chosen.answers.iter().any(Option::is_some) {
                // The marks the reading chose, exactly.
                let mut found = Vec::new();
                for &(place, start, end) in chosen.answers.iter().flatten().filter(|mark| mark.0 != u32::MAX) {
                    if let Some(at) = stream.unanswered.iter().position(|&(r, line, s, _)| r == value && line == place && s == start) {
                        stream.unanswered.remove(at);
                    }
                    found.push((place, start, end));
                }
                let mut found = found.into_iter();
                made.paired = found.next();
                for slot in &mut made.also {
                    *slot = found.next();
                }
            } else if let Some(at) = stream.unanswered.iter().rposition(|&(r, ..)| r == value) {
                let (_, place, start, end) = stream.unanswered.remove(at);
                made.paired = Some((place, start, end));
                // The text's other marks of the number the note answers too, nearest first.
                for slot in &mut made.also {
                    let Some(at) = stream.unanswered.iter().rposition(|&(r, ..)| r == value) else { break };
                    let (_, place, start, end) = stream.unanswered.remove(at);
                    *slot = Some((place, start, end));
                }
            }
        }
        if REFERRING.contains(&chosen.region) && !line.references().is_empty() {
            stream.unanswered.retain(|(r, ..)| !line.references().iter().any(|(v, ..)| v == r));
            stream.unanswered.extend(line.references().iter().map(|&(v, start, end)| (v, line.place, start, end)));
            let excess = stream.unanswered.len().saturating_sub(30);
            stream.unanswered.drain(..excess);
        }
        made
    }
}
