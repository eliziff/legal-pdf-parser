//! The parser's moves and state. Each line in reading order makes one move: it opens a unit of a
//! region or goes on with the one open in its stream, and a line that opens may count on its marker
//! (one of its grammar readings) or on a note label; a heading opens at a level. What is open is kept
//! per reading context (the stack of numbered units, family, value and region: PART II > 12 > (2) >
//! (a); the last note opened and the reference marks the body set that no note has answered, with the
//! line that set each): that state, and the walk that applies the moves, are tier 0's (legal-pdf-decide's `grammar`,
//! `tier0`).

use crate::decide::Opening;
use crate::fences::Fences;
use crate::machine::lines::LineTable;
use crate::regions::{QUOTE, TEXT};
use crate::{Evidence, MarkerReading, Role, Rule, Slot};

/// Regions whose lines set reference marks notes answer: the body's, and headings' and titles'.
pub const REFERRING: [u8; 8] = [1, 2, 3, 4, 8, 9, 14, 16];
/// The body regions that take a depth (headings aside).
pub const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];

/// One line's move, the parse's one record of the line: its role and the rule whose reading won it, whether it goes on
/// with the line above it in one heading, caption or unit, and the unit its marker opens (`decide::Decision`); its
/// region, whether it opens a unit, what the unit counts on (one of the line's grammar readings, `reading`, an index; or
/// a note's number, `note`), and a heading's level.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Move {
    pub role: Role,
    pub rule: Rule,
    pub joined: bool,
    pub unit: Option<Opening>,
    pub region: u8,
    pub opens: bool,
    pub reading: Option<u8>,
    pub note: Option<u32>,
    pub level: u8,
    /// A body unit's depth in its document's body (0 for none): the move's own, not its numbering's,
    /// which is evidence for it.
    pub depth: u8,
    /// The unit the line is in, by its first line's place, where the line does not open it: the unit open above it, one
    /// picked up past a block quotation or a page's notes, or one of two read in turn (a form's labels and their values).
    pub resumes: Option<u32>,

    /// Opens a constituent document of the source (an exhibit, a schedule, a message an email quotes):
    /// its contexts, outline and numbering begin anew.
    pub starts: bool,
    /// A note's opening: the reference marks it answers, each its line's place and its first and past-last
    /// character, where the reading chose them, wherever they stand and however many (none chosen: the marks of its
    /// number the text set last, tier 0's; chosen and empty: no mark). A note whose marks are chosen is written from
    /// its move (`graph::write`), not paired again.
    pub answers: Option<Vec<(u32, u32, u32)>>,
    /// A unit's opening: its printed markers where the reading names them, each its line's place and its first and
    /// past-last UTF-16 code unit, in the order they print ("10." then "(1)"; a number in the margin's line); none
    /// named: the marker of the line's own reading (`reading`), tier 0's.
    pub markers: Option<Vec<(u32, u32, u32)>>,
    /// Where the reading reads the line among its page's lines, where it reads them in another order than the parser
    /// found them in: the lines are written by it.
    pub rank: Option<u32>,
    /// A table row's line: its cells where the reading names them, each its column and its first and past-last UTF-16
    /// code unit of the line; none named: the table reader's cells on the line, tier 0's.
    pub cells: Option<Vec<(u32, u32, u32)>>,
    /// The labels on the line that open a speaker's turn ("Q.", "MR. SMITH:"), where the reading names them: each one's
    /// first and past-last UTF-16 code unit.
    pub speakers: Option<Vec<(u32, u32)>>,
    /// What kind of document opens, where the reading names it: on a line that `starts` one, that document's; on the
    /// source's first line, the source's own.
    pub document: Option<String>,
    /// Tier 0's opening: the run member the unit opens on (a run and its member), and the further markers its line
    /// opens on, each a unit under it ("2(1) The Court has ...": the section and its subsection).
    pub member: Option<(u32, u32)>,
    pub within: Vec<(u32, u32)>,
}

impl Move {
    /// A move making nothing but its region and whether it opens.
    pub fn plain(region: u8, opens: bool) -> Self {
        Self {
            role: Role::Open,
            rule: Rule::None,
            joined: false,
            unit: None,
            region,
            opens,
            reading: None,
            note: None,
            level: 0,
            depth: 0,
            resumes: None,
            starts: false,
            answers: None,
            markers: None,
            rank: None,
            cells: None,
            speakers: None,
            document: None,
            member: None,
            within: Vec::new(),
        }
    }
}

/// A line as the state reads it: its place in reading order, and what holds what is read of it, borrowed: the
/// evidence, the fences and the line table (`LineTable`).
#[derive(Clone, Copy)]
pub struct LineView<'a> {
    pub place: u32,
    pub evidence: &'a Evidence,
    pub fences: &'a Fences,
    pub table: &'a LineTable,
}

/// A move's `reading` that counts on the number the margin prints beside the line's row (`LineView::margin`),
/// not on one of the line's own.
pub const MARGIN: u8 = u8::MAX;
/// A move's `reading` that opens on the line's marker where no reading makes it out: its opening token, counting on
/// nothing.
pub const OPENING: u8 = u8::MAX - 1;

impl<'a> LineView<'a> {
    /// Every line of the document, in reading order.
    pub fn all(
        evidence: &'a Evidence,
        fences: &'a Fences,
        table: &'a LineTable,
    ) -> Vec<Self> {
        (0..fences.order.len())
            .map(|k| Self {
                place: k as u32,
                evidence,
                fences,
                table,
            })
            .collect()
    }

    fn k(&self) -> usize {
        self.place as usize
    }

    /// The line's slot.
    pub fn slot(&self) -> Slot {
        self.fences.order[self.k()]
    }

    /// Whether the line reads in a parallel translation's column.
    pub fn translation(&self) -> bool {
        self.fences.at(self.slot().index()).translation
    }

    /// The line's marker's readings (`LineTable::readings`).
    pub fn readings(&self) -> &'a [MarkerReading] {
        &self.table.readings[self.k()]
    }

    /// The value of the note label the line opens with.
    pub fn label(&self) -> Option<u32> {
        self.evidence.label[self.slot().index()].map(|label| label.value)
    }

    /// The reference marks the line sets (`LineTable::references`).
    pub fn references(&self) -> &'a [(u32, u32, u32)] {
        &self.table.references[self.k()]
    }

    /// The paragraph number the margin prints beside the line's row, numbering the paragraph the line opens in its
    /// column (`Fences::read_margin_numbers`: in both columns of a parallel report): the margin line's place and its
    /// number's reading. A move counts on it as `MARGIN`.
    pub fn margin(&self) -> Option<(u32, &'a MarkerReading)> {
        let margin = *self.table.margins.get(&self.k())?;
        Some((margin as u32, self.table.readings[margin].first()?))
    }

    /// The paragraph number the decide stage's judgment reader reads the line opening (`Fences::paragraphs`): t0's
    /// reading of the document's numbered paragraphs, which DOC's decoding works from.
    pub fn count(&self) -> Option<u32> {
        self.fences.paragraphs.count[self.slot().index()]
    }

    /// The reading a move's `reading` names: one of the line's own, the margin's number (`MARGIN`), or none (`OPENING`).
    pub fn reading(&self, reading: Option<u8>) -> Option<&'a MarkerReading> {
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
    pub fn opening(&self, next: impl FnOnce() -> Option<usize>) -> u8 {
        let readings = self.readings();
        let counted = self
            .count()
            .and_then(|count| readings.iter().position(|reading| reading.value == count));
        match (self.margin(), counted.or_else(next), readings.is_empty()) {
            (Some(_), ..) => MARGIN,
            (_, Some(at), _) => at as u8,
            (_, _, false) => 0,
            _ => OPENING,
        }
    }
}

/// A stack holds at most this many open units; headings run nine levels deep.
pub const DEPTH: usize = 8;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    pub family: String,
    pub value: u32,
    pub region: u8,
}

/// What is open in one reading context.
#[derive(Clone, Debug, Default)]
pub struct Open {
    pub stack: Vec<Frame>,
    pub note: Option<u32>,
    /// The reference marks the body set that no note has answered: each one's value, the line that set
    /// it, and where it stands in that line (first and past-last character).
    pub unanswered: Vec<(u32, u32, u32, u32)>,
    /// Whether the context reads a quotation (a quote opened and no paragraph of its own text since),
    /// and the quoted text's own numbering, which the quotation counts on apart from the text's.
    pub quoting: bool,
    pub quoted: Vec<Frame>,
}

impl Open {
    /// The numbering the context counts on: a quotation's own while it reads one.
    pub fn frames(&self) -> &Vec<Frame> {
        if self.quoting {
            &self.quoted
        } else {
            &self.stack
        }
    }

    pub fn frames_mut(&mut self) -> &mut Vec<Frame> {
        if self.quoting {
            &mut self.quoted
        } else {
            &mut self.stack
        }
    }

    /// A unit opening in `region`: a quote opens a quotation the context reads until a paragraph of the text's own opens.
    pub fn opens(&mut self, region: u8) {
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
    pub fn count(&mut self, reading: &MarkerReading, region: u8) {
        let frames = counted(self.frames(), reading, region);
        *self.frames_mut() = frames;
    }
}

/// The stack after a reading counts: popped to the open frame of its family and replaced, else pushed.
pub fn counted(stack: &[Frame], reading: &MarkerReading, region: u8) -> Vec<Frame> {
    let frame = Frame {
        family: reading.family.clone(),
        value: reading.value,
        region,
    };
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
