//! The parser's moves and state. Each line in reading order makes one move: it opens a unit of a
//! region or goes on with the one open in its stream, and a line that opens may count on its marker
//! (one of its grammar readings) or on a note label; a heading opens at a level. What is open is kept
//! per reading context (the stack of numbered units, family, value and region: PART II > 12 > (2) >
//! (a); the last note opened and the reference marks the body set that no note has answered, with the
//! line that set each; the last transcript line number) and per document in reading order (the open
//! heading's level, the level each look of heading last took, the last body unit's depth). A move is
//! scored on the line's evidence and on how it fits this state; applying it moves the state and says
//! what the move made of the line: the reference it paired, its heading's level step, its body unit's
//! depth step.

use super::*;
use crate::lines::LineTable;

pub use legal_pdf_structure_model::regions::*;
/// Regions whose units count on a marker.
const NUMBERED: [u8; 9] = [1, 2, 3, 4, 7, 8, 10, 14, 16];
/// Regions whose lines set reference marks notes answer: the body's, and headings' and titles'.
pub const REFERRING: [u8; 8] = [1, 2, 3, 4, 8, 9, 14, 16];
/// The body regions that take a depth (the tagger's ranked regions, headings aside).
pub const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];
/// A stack holds at most this many open units; headings run nine levels deep.
const DEPTH: usize = 8;
const LEVELS: i8 = 9;

/// One line's move: its region, whether it opens a unit, what the unit counts on (one of the line's
/// grammar readings, `reading`, an index; or a note's number, `note`), and a heading's level.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Move {
    pub region: u8,
    pub opens: bool,
    pub reading: Option<u8>,
    pub note: Option<u32>,
    pub level: i8,
    /// A body unit's depth in its document's body (0 for none): the move's own, not its numbering's,
    /// which is evidence for it.
    pub depth: u8,
    /// Goes on with the unit open in its stream before another region's took the stream (a paragraph
    /// picked up past a block quotation or a page's notes).
    pub resume: bool,
    /// Goes on with the unit whose first line is this place, whichever unit its stream holds open (two units
    /// read in turn: a form's labels and their values): over `resume`, which goes on with the last unit of its
    /// region in its stream.
    pub resumes: Option<u32>,
    /// The line's place in the reading the parse makes, where it reads the page in another order than the
    /// parser's lines (two columns read one after the other, a form's labels with their values).
    pub rank: Option<u32>,
    /// Opens a constituent document of the source (an exhibit, a schedule, a message an email quotes):
    /// its contexts, outline and numbering begin anew.
    pub starts: bool,
    /// A note's opening: the reference marks it answers, each its line's place and its first and past-last
    /// character, where the reading chose them, wherever they stand (none: the marks of its number the text set
    /// last; a place of `u32::MAX` alone: no mark).
    pub answers: [Option<(u32, u32, u32)>; 4],
    /// Tier 0's opening: the run member the unit opens on (a run and its member), and the further markers its line
    /// opens on, each a unit under it ("2(1) The Court has ...": the section and its subsection).
    pub member: Option<(u32, u32)>,
    pub within: Vec<(u32, u32)>,
}

impl Move {
    /// A move making nothing but its region and whether it opens.
    pub fn plain(region: u8, opens: bool) -> Self {
        Self { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, resumes: None, rank: None, starts: false, answers: [None; 4], member: None, within: Vec::new() }
    }

    /// The move's class among the region-and-opening moves the local scores rank (`region * 2 +
    /// opens`).
    pub fn class(&self) -> usize {
        usize::from(self.region) * 2 + usize::from(self.opens)
    }
}

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
    pub line_number: Option<u32>,
    /// Whether the context reads a quotation (a quote opened and no paragraph of its own text since),
    /// and the quoted text's own numbering, which the quotation counts on apart from the text's.
    pub quoting: bool,
    pub quoted: Vec<Frame>,
}

impl Open {
    /// The numbering the context counts on: a quotation's own while it reads one.
    pub fn frames(&self) -> &Vec<Frame> {
        if self.quoting { &self.quoted } else { &self.stack }
    }

    fn frames_mut(&mut self) -> &mut Vec<Frame> {
        if self.quoting { &mut self.quoted } else { &mut self.stack }
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

/// What a document's outline holds so far, in reading order.
#[derive(Clone, Debug, Default)]
pub struct Outline {
    /// The open heading's level (0 before the first).
    pub heading: i8,
    /// Each look a heading took and the level it last took.
    pub look_levels: Vec<(u32, i8)>,
    /// The last body unit's depth (0 before the first).
    pub body_depth: usize,
}

/// A contents list's entry as the parser read it: its title (normalized, `bookmarks::normalize_title`), its
/// level (its indent's tier among the document's entries), and whether a heading has answered it.
#[derive(Clone, Debug)]
pub struct Entry {
    pub key: String,
    pub level: u8,
    pub answered: bool,
    /// The page the entry is printed on: the heading it names stands on a later one.
    pub page: u32,
}

/// The parse so far: what is open in each context and each document's outline, the move before, how
/// many lines ago each region last opened a unit, the constituent document read and its lines, and each
/// document's contents entries with the indents they stand at.
#[derive(Clone, Debug)]
pub struct State {
    pub document: u16,
    pub document_lines: u32,
    pub entries: HashMap<u16, (Vec<Entry>, Vec<f32>)>,
    /// The unanswered entry the line last asked about answers it (its place among the entries, its level).
    answering: std::cell::RefCell<Option<(u32, Option<(usize, u8)>)>>,
    pub streams: HashMap<u16, Open>,
    pub outlines: HashMap<u16, Outline>,
    pub previous: Option<Move>,
    pub since: [u32; 17],
    /// The region of the unit open in each context's each stream (`stream_of`).
    pub open_regions: HashMap<(u16, usize), u8>,
    /// The region of the unit each stream held before its open one's region took it.
    pub interrupted: HashMap<(u16, usize), u8>,
}

impl Default for State {
    fn default() -> Self {
        // No region has opened a unit yet: as long ago as `features` counts.
        Self { document: 0, document_lines: 0, entries: HashMap::new(), answering: Default::default(), streams: HashMap::new(), outlines: HashMap::new(), previous: None, since: [500; 17], open_regions: HashMap::new(), interrupted: HashMap::new() }
    }
}

/// What the walk reads of the document's lines beyond tier 0's records, by place in reading order: the grammar
/// tables (`grammar::Grammar`); the stylesheet's reading of each line, borrowed (`Stylesheet`: its template among the
/// templates, the level the document's outline names it at, its document's regime, its contents list's title); how
/// far each line's first word is a heading's by the instruments' tables and the others'; each look's view before any
/// move (the mean probability its lines open a heading by the local scores, and how many of its lines they hold
/// likelier a heading's opening than not, log); and each line's text normalized as a title is matched
/// (`bookmarks::normalize_title`, empty for a long line).
pub struct Walked<'a> {
    pub grammar: Option<&'a Grammar>,
    pub template: &'a [u32],
    pub templates: &'a [TemplateView],
    pub anchor: &'a [u16],
    pub regime: &'a [[f32; 4]],
    pub contents_titles: &'a [Option<String>],
    pub heading_words: Vec<[f32; 2]>,
    pub looks: HashMap<u32, [f32; 2]>,
    pub title_keys: Vec<String>,
}

/// A line as the state reads it: its place in reading order, and what holds what is read of it, borrowed: its pages
/// and slots, the evidence, the fences, the line table (`LineTable`), and, where the walk reads the document, what the
/// walk reads (`Walked`).
#[derive(Clone, Copy)]
pub struct LineView<'a> {
    pub place: u32,
    pub pages: &'a [Page],
    pub slots: &'a Slots,
    pub evidence: &'a Evidence,
    pub fences: &'a Fences,
    pub table: &'a LineTable,
    pub walked: Option<&'a Walked<'a>>,
}

/// How a move stands to the units of its stream.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stands {
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
pub struct Streams {
    open: HashMap<(u16, usize), usize>,
    last_of_region: HashMap<(u16, usize, u8), usize>,
    regions: HashMap<usize, u8>,
}

impl Streams {
    /// How `chosen` stands in the stream `key`. A transcript's line number and a page's label are a line each.
    pub fn stands(&self, key: (u16, usize), chosen: &Move) -> Stands {
        let single = matches!(chosen.region, LINE_LABEL | PAGE_LABEL);
        match (chosen.resumes, self.open.get(&key)) {
            (Some(first), _) => Stands::Resumes(first as usize),
            (_, Some(&first)) if !single && !chosen.opens && self.regions.get(&first) == Some(&chosen.region) => Stands::GoesOn(first),
            _ if !single && chosen.resume => self.last_of_region.get(&(key.0, key.1, chosen.region)).map_or(Stands::Opens, |&first| Stands::Resumes(first)),
            _ => Stands::Opens,
        }
    }

    /// The stream after the move at `line`.
    pub fn step(&mut self, key: (u16, usize), region: u8, stands: Stands, line: usize) {
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
pub const MARGIN: u8 = u8::MAX;
/// A move's `reading` that opens on the line's marker where no reading makes it out: its opening token, counting on
/// nothing.
pub const OPENING: u8 = u8::MAX - 1;

impl<'a> LineView<'a> {
    /// Every line of the document as tier 0 reads it, in reading order: nothing the walk reads.
    pub fn all(pages: &'a [Page], slots: &'a Slots, evidence: &'a Evidence, fences: &'a Fences, table: &'a LineTable, walked: Option<&'a Walked<'a>>) -> Vec<Self> {
        (0..fences.order.len()).map(|k| Self { place: k as u32, pages, slots, evidence, fences, table, walked }).collect()
    }

    fn k(&self) -> usize {
        self.place as usize
    }

    /// The line's slot.
    pub fn slot(&self) -> Slot {
        self.fences.order[self.k()]
    }

    /// The line.
    pub fn line(&self) -> &'a Line {
        self.slots.line(self.pages, self.slot())
    }

    /// Whether the line reads in a parallel translation's column.
    pub fn translation(&self) -> bool {
        self.fences.translation[self.slot().index()]
    }

    /// The line's marker's readings (`LineTable::readings`).
    pub fn readings(&self) -> &'a [MarkerReading] {
        &self.table.readings[self.k()]
    }

    /// The value of the note label the line opens with.
    pub fn label(&self) -> Option<u32> {
        self.evidence.label[self.slot().index()].map(|label| label.value)
    }

    /// The transcript line number the line is.
    pub fn line_number(&self) -> Option<u32> {
        self.evidence.line_number[self.slot().index()].then(|| self.line().text.trim().parse().ok()).flatten()
    }

    /// The line's look (`LineTable::looks`).
    pub fn look(&self) -> u32 {
        self.table.looks[self.k()]
    }

    /// The reference marks the line sets (`LineTable::references`).
    pub fn references(&self) -> &'a [(u32, u32, u32)] {
        &self.table.references[self.k()]
    }

    /// The page slot of the line.
    pub fn page(&self) -> u32 {
        self.slots.page(self.slot()) as u32
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

    /// Whether the line may open a constituent document, where the walk reads it: the source's own document opens on
    /// its first line, and any later line's opening may open another (a letter mid-page, an email chain's next message).
    pub fn may_start(&self) -> bool {
        self.walked.is_some() && self.place > 0
    }

    /// The kind the constituents reader names the line's opening (`constituents::KINDS`, 0 for none), where the walk
    /// reads it.
    pub fn start_kind(&self) -> u8 {
        let Some(_) = self.walked else { return 0 };
        let id = self.line().id.as_str();
        self.fences.starts.iter().find(|start| start.line_id == id).map_or(0, |start| start.kind_code())
    }

    /// The line's place among its page's lines, where the walk reads it.
    pub fn page_place(&self) -> u32 {
        self.walked.map_or(0, |_| self.place - self.slots.on(self.page() as usize).start)
    }

    /// The grammar tables, where the walk reads them.
    pub fn grammar(&self) -> Option<&'a Grammar> {
        self.walked.and_then(|walked| walked.grammar)
    }

    /// How far the line's first word is a heading's by the instruments' tables and the others'.
    pub fn heading_words(&self) -> [f32; 2] {
        self.walked.map_or([0.0; 2], |walked| walked.heading_words[self.k()])
    }

    /// The line's look before any move: its lines' mean probability of opening a heading, and how many they hold
    /// likelier a heading's opening than not (log).
    pub fn look_heading(&self) -> [f32; 2] {
        self.walked.and_then(|walked| walked.looks.get(&self.look()).copied()).unwrap_or([0.0; 2])
    }

    /// The line's template in the document's stylesheet.
    pub fn template(&self) -> TemplateView {
        self.walked.map_or_else(TemplateView::default, |walked| walked.templates[walked.template[self.k()] as usize])
    }

    /// The level the document's outline names the line at (0 for none).
    pub fn anchor(&self) -> u16 {
        self.walked.map_or(0, |walked| walked.anchor[self.k()])
    }

    /// The line's document's regime (`Stylesheet::regime`).
    pub fn regime(&self) -> [f32; 4] {
        self.walked.map_or([0.0; 4], |walked| walked.regime[self.k()])
    }

    /// The line's text normalized as a title is matched.
    pub fn title_key(&self) -> &'a str {
        self.walked.map_or("", |walked| walked.title_keys[self.k()].as_str())
    }

    /// The title of the contents list's row the line opens, normalized as a title is matched, where it prints a locator.
    pub fn contents_title(&self) -> Option<&'a str> {
        self.walked.and_then(|walked| walked.contents_titles[self.k()].as_deref())
    }

    /// Where the line starts on its page, a share of the page's width (`LineTable::row`'s x0), where the walk reads it.
    pub fn left(&self) -> f32 {
        self.walked.map_or(0.0, |_| self.table.row(self.k())[0])
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
        let counted = self.count().and_then(|count| readings.iter().position(|reading| reading.value == count));
        match (self.margin(), counted.or_else(next), readings.is_empty()) {
            (Some(_), ..) => MARGIN,
            (_, Some(at), _) => at as u8,
            (_, _, false) => 0,
            _ => OPENING,
        }
    }
}

/// What a move made of its line: the earlier line whose reference mark its note answered, a heading's
/// level step from the heading before it in its document (10 the same, as the gold steps), a body
/// unit's depth step from the body unit before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Made {
    pub paired: Option<(u32, u32, u32)>,
    /// The other marks of the note's number it answered too (a note several places in the text cite).
    pub also: [Option<(u32, u32, u32)>; 3],
    pub level_step: i64,
    pub depth_step: i64,
}

/// The names of a candidate move's features, read off the move and the state: the move itself, the
/// move before, how its marker or note number fits what is open, and a heading's level against the
/// open heading's, its look's last level and its place on the stack.
pub const CANDIDATE: &[&str] = &[
    "region", "opens", "previous_region", "previous_opens", "same_region", "since_region",
    "counts", "count_class", "found_open", "levels_up", "step", "pushes", "push_from_one", "depth_after", "under_region",
    "note_counts", "note_next", "note_first", "note_answers", "unanswered", "note_plain",
    "line_number_next", "line_number_first",
    "heading_step", "heading_level", "look_levelled", "look_level_agrees", "stack_level_agrees",
    "resumes", "interrupted_region",
    "grammar_relation", "grammar_child", "heading_word",
    "look_heading", "look_heading_lines",
    "other_grammar_relation", "other_grammar_child", "other_heading_word", "operative",
    "starts", "start_kind", "page_place", "document_lines", "documents",
    "template_lines", "template_pages", "template_heading", "template_monotone", "template_anchored",
    "template_anchor_level", "template_level", "template_edge", "template_read_heading", "template_read_level",
    "level_by_template", "level_by_anchor", "level_by_reading", "anchored", "anchor_level", "listed",
    "regime_bracketed", "regime_sectioned", "regime_line_numbers", "regime_speakers",
    "quoting", "template_beside",
    "answers_entry", "entry_level", "level_by_entry", "contents_row", "entries", "entries_unanswered",
    "depth", "depth_step", "depth_by_stack",
];
pub const CANDIDATES: usize = 75;
/// A body unit's depth is tried this far below the last body unit's, and at the top.
const DEPTH_SPAN: usize = 2;

/// The class of a family's count (`lines::COUNTS` order), 9 for none.
pub fn count_class(family: &str) -> f32 {
    let count = family.rsplit(':').next().unwrap_or("");
    ["1", "1.1", "1.1.1", "1.1.1.1", "a", "A", "i", "I", "bullet"].iter().position(|c| *c == count).map_or(9.0, |c| c as f32)
}

/// A move's step from the frame it pops to, its value against the frame's: 1 the next, 0 the same.
fn step(value: u32, frame: &Frame) -> f32 {
    (i64::from(value) - i64::from(frame.value)).clamp(-5, 5) as f32
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
    pub fn ctx(&self, line: &LineView, starts: bool) -> u16 {
        (self.document + u16::from(starts)).saturating_mul(2) + u16::from(line.translation())
    }

    fn open(&self, ctx: u16) -> Option<&Open> {
        self.streams.get(&ctx)
    }

    fn outline(&self, document: u16) -> Option<&Outline> {
        self.outlines.get(&document)
    }

    /// The outline a line's headings and body units go in: its document's, a quotation's headings and
    /// units stepping in it as the gold steps them (the quotation's numbering is its own); after
    /// `starts`, the next document's.
    pub fn outline_key(&self, starts: bool) -> u16 {
        self.document + u16::from(starts)
    }

    /// The depth a body opening's numbering puts it at: the body units open on its context's stack once
    /// its reading counts (the stack as it stands for an unnumbered one), one at least.
    pub fn stack_depth(&self, line: &LineView, candidate: &Move) -> usize {
        let empty = Open::default();
        let stream = self.open(self.ctx(line, candidate.starts)).unwrap_or(&empty);
        let body = |frames: &[Frame]| frames.iter().filter(|f| RANKED_BODY.contains(&f.region)).count();
        match line.reading(candidate.reading) {
            Some(reading) => body(&counted(stream.frames(), reading, candidate.region)),
            None => body(stream.frames()),
        }
        .max(1)
    }

    /// The region-and-opening classes the state itself proposes for a line, whatever the local scores
    /// say: going on with the unit open in each of the line's context's streams, or opening the next
    /// unit of its region; a heading where the line's look is one its document's headings took; a note opening on the next note number or on a number the body set; a
    /// transcript line number going on from the last; an opening of the region of an open frame its
    /// marker counts on from.
    pub fn proposed(&self, line: &LineView) -> Vec<usize> {
        let mut out = Vec::new();
        let opening = |region: u8| usize::from(region) * 2 + 1;
        let ctx = self.ctx(line, false);
        let mut open: Vec<u8> = self.open_regions.iter().filter(|((c, _), _)| *c == ctx).map(|(_, &region)| region).collect();
        open.sort_unstable();
        for region in open {
            out.push(usize::from(region) * 2);
            out.push(opening(region));
            // A note open in the stream may quote: its quotation's lines.
            if region == NOTE {
                out.push(usize::from(NOTE_QUOTE) * 2);
                out.push(opening(NOTE_QUOTE));
            }
        }
        // A document's first lines may be its title's.
        if self.document_lines < 40 {
            out.push(usize::from(TITLE) * 2);
            out.push(opening(TITLE));
        }
        // A heading where the line is set as a heading of its document was set before.
        if self.outline(self.outline_key(false)).is_some_and(|outline| outline.look_levels.iter().any(|(look, _)| *look == line.look())) {
            out.push(opening(HEADING));
        }
        if let Some(open) = self.open(ctx) {
            let values: Vec<u32> = line.label().into_iter().chain(line.readings().iter().map(|r| r.value)).collect();
            if values.iter().any(|&v| open.note.is_some_and(|last| v == last + 1) || open.unanswered.iter().any(|&(r, ..)| r == v)) {
                out.push(opening(NOTE));
            }
            if line.line_number().is_some_and(|v| open.line_number.is_some_and(|last| v == last + 1)) {
                out.push(opening(LINE_LABEL));
            }
            for reading in line.readings() {
                if let Some(frame) = open.frames().iter().rev().find(|f| f.family == reading.family && reading.value == f.value + 1) {
                    out.push(opening(frame.region));
                }
            }
        }
        out
    }

    /// Every move a line may make among `classes`: each region opening or going on; an opening of a
    /// numbered region counting on each of the line's readings or on none; a heading at every level from
    /// one to two below the open heading's; a note opening on its label's value or a reading's or on
    /// none; on a line that may open a constituent document, each of those opening it too.
    pub fn candidates(&self, line: &LineView, classes: &[usize]) -> Vec<Move> {
        let outline = self.outline(self.outline_key(false));
        let heading = outline.map_or(0, |outline| outline.heading);
        // The depths a body opening is tried at: the top, and from two above the last body unit's to two
        // below it; the numbering's depth among them (`stack_depth`).
        let last = outline.map_or(0, |outline| outline.body_depth).max(1);
        let mut depths: Vec<u8> = std::iter::once(1).chain(last.saturating_sub(DEPTH_SPAN).max(1)..=(last + DEPTH_SPAN).min(DEPTH + 1)).map(|d| d as u8).collect();
        depths.dedup();
        let mut out = Vec::new();
        for &class in classes {
            let (region, opens) = ((class / 2) as u8, class % 2 == 1);
            let plain = Move::plain(region, opens);
            let mut bases = Vec::new();
            if opens && region == NOTE {
                // A note counts on a number: its label's, or an arabic reading's (a lettered note answers no
                // figure in the text).
                let mut values: Vec<u32> = line.label().into_iter().collect();
                values.extend(line.readings().iter().filter(|r| count_class(&r.family) == 0.0).map(|r| r.value).filter(|v| Some(*v) != line.label()));
                bases.extend(values.into_iter().take(3).map(|v| Move { note: Some(v), ..plain.clone() }));
            } else if opens && NUMBERED.contains(&region) {
                bases.extend((0..line.readings().len().min(3)).map(|r| Move { reading: Some(r as u8), ..plain.clone() }));
            }
            bases.push(plain.clone());
            let key = (self.ctx(line, false), stream_of(region));
            if !opens && self.interrupted.get(&key) == Some(&region) && self.open_regions.get(&key) != Some(&region) {
                bases.push(Move { resume: true, ..plain });
            }
            if opens && region == HEADING {
                // A heading may open as much as three levels under the open one (a schedule's own parts
                // under a chapter's heading).
                let deepest = (heading.max(1) + 3).min(LEVELS);
                for base in bases {
                    out.extend((1..=deepest).map(|level| Move { level, ..base.clone() }));
                }
            } else if opens && RANKED_BODY.contains(&region) {
                for base in bases {
                    let mut tried = depths.clone();
                    let by_stack = self.stack_depth(line, &base).min(DEPTH + 1) as u8;
                    if !tried.contains(&by_stack) {
                        tried.push(by_stack);
                    }
                    out.extend(tried.into_iter().map(|depth| Move { depth, ..base.clone() }));
                }
            } else {
                out.extend(bases);
            }
        }
        // A constituent document opens on a unit's opening, its first heading or body unit at the top.
        if line.may_start() {
            let starting: Vec<Move> = out.iter().filter(|m| m.opens && !m.resume && m.depth <= 1 && m.level <= 2).map(|m| Move { starts: true, ..m.clone() }).collect();
            out.extend(starting);
        }
        out
    }

    /// The unanswered contents entry of the line's document the line's text answers, its place and
    /// level: the same title, or a title the line opens (a heading wrapped over lines) or closes (a
    /// numbering label only the entry or the heading prints). Asked once a line.
    fn answering(&self, line: &LineView) -> Option<(usize, u8)> {
        if let Some((place, found)) = *self.answering.borrow() {
            if place == line.place {
                return found;
            }
        }
        // A contents list's own row answers none: it is an entry, not the heading one names.
        let found = (line.contents_title().is_none() && line.title_key().chars().count() >= 3).then(|| self.entries.get(&self.document)).flatten().and_then(|(entries, _)| {
            let key = line.title_key();
            // A part of a title matches where it holds half of the other or more.
            let (length, half) = (key.chars().count(), |of: &str| of.chars().count().div_ceil(2));
            entries.iter().position(|entry| !entry.answered && entry.page < line.page() && (entry.key == key
                || (length >= 8 && length >= half(&entry.key) && (entry.key.starts_with(key) || entry.key.ends_with(key)))
                || (entry.key.chars().count() >= 8 && entry.key.chars().count() >= half(key) && key.ends_with(entry.key.as_str()))))
                .map(|at| (at, entries[at].level))
        });
        *self.answering.borrow_mut() = Some((line.place, found));
        found
    }

    /// The candidate's features in `CANDIDATE` order.
    pub fn features(&self, line: &LineView, candidate: &Move) -> [f32; CANDIDATES] {
        // A move opening a document reads against that document's state: nothing open yet.
        let empty = Open::default();
        let stream = self.open(self.ctx(line, candidate.starts)).unwrap_or(&empty);
        let blank = Outline::default();
        let outline = self.outline(self.outline_key(candidate.starts)).unwrap_or(&blank);
        let frames = stream.frames();
        let flag = |b: bool| f32::from(u8::from(b));
        let mut f = [0f32; CANDIDATES];
        f[0] = f32::from(candidate.region);
        f[1] = flag(candidate.opens);
        f[2] = self.previous.as_ref().map_or(-1.0, |m| f32::from(m.region));
        f[3] = flag(self.previous.as_ref().is_some_and(|m| m.opens));
        f[4] = flag(self.previous.as_ref().is_some_and(|m| m.region == candidate.region));
        f[5] = (self.since[usize::from(candidate.region)] as f32).min(500.0);
        f[7] = 9.0;
        f[14] = -1.0;
        let reading = line.reading(candidate.reading);
        if let Some(reading) = reading {
            f[6] = 1.0;
            f[7] = count_class(&reading.family);
            match frames.iter().rposition(|frame| frame.family == reading.family) {
                Some(depth) => {
                    f[8] = 1.0;
                    f[9] = (frames.len() - 1 - depth) as f32;
                    f[10] = step(reading.value, &frames[depth]);
                    f[13] = (depth + 1) as f32;
                    f[14] = depth.checked_sub(1).map_or(-1.0, |d| f32::from(frames[d].region));
                }
                None => {
                    f[11] = 1.0;
                    f[12] = flag(reading.value == 1);
                    f[13] = (frames.len() + 1).min(DEPTH) as f32;
                    f[14] = frames.last().map_or(-1.0, |frame| f32::from(frame.region));
                }
            }
        }
        if candidate.opens && candidate.region == NOTE {
            match candidate.note {
                Some(value) => {
                    f[15] = 1.0;
                    f[16] = flag(stream.note.is_some_and(|last| value == last + 1));
                    f[17] = flag(value == 1);
                    f[18] = flag(stream.unanswered.iter().any(|&(r, ..)| r == value));
                }
                None => f[20] = 1.0,
            }
        }
        f[19] = stream.unanswered.len().min(30) as f32;
        if candidate.region == LINE_LABEL {
            if let Some(value) = line.line_number() {
                f[21] = flag(stream.line_number.is_some_and(|last| value == last + 1));
                f[22] = flag(value == 1);
            }
        }
        if candidate.opens && candidate.region == HEADING {
            let level = candidate.level;
            f[23] = f32::from(level - outline.heading.max(1));
            f[24] = f32::from(level);
            let looked = outline.look_levels.iter().find(|(look, _)| *look == line.look()).map(|&(_, l)| l);
            f[25] = flag(looked.is_some());
            f[26] = looked.map_or(-9.0, |l| f32::from(level - l));
            // A numbered heading's level against the headings it would sit under on the stack.
            f[27] = reading.map_or(-9.0, |reading| {
                let under = counted(frames, reading, HEADING).iter().filter(|frame| frame.region == HEADING).count() as i8;
                f32::from(level - under)
            });
        }
        // How likely the move's numbering is in real legal text, by the instruments' tables and by the
        // others' (1 where the tables say nothing of it), the document's operative words beside them.
        let top = frames.last().map_or("", |frame| frame.family.as_str());
        let tables = line.grammar().map(|g| [&g.instruments, &g.others]);
        for (i, at) in [30, 35].into_iter().enumerate() {
            f[at] = 1.0;
            f[at + 1] = 1.0;
            f[at + 2] = 1.0;
            let Some(tables) = tables.map(|t| t[i]) else { continue };
            if let Some(reading) = reading {
                let relation = match frames.iter().rposition(|frame| frame.family == reading.family) {
                    Some(depth) if depth + 1 == frames.len() => if reading.value == frames[depth].value + 1 { grammar::NEXT } else { grammar::SAME_OTHER },
                    Some(depth) => if reading.value == frames[depth].value + 1 { grammar::POP_NEXT } else { grammar::POP_OTHER },
                    None => if reading.value == 1 { grammar::PUSH_1 } else { grammar::PUSH_OTHER },
                };
                f[at] = tables.relation(top, relation).unwrap_or(1.0);
                if !frames.iter().any(|frame| frame.family == reading.family) {
                    f[at + 1] = tables.child(top, &reading.family).unwrap_or(1.0);
                }
            } else if candidate.opens && NUMBERED.contains(&candidate.region) && !line.readings().is_empty() {
                f[at] = tables.relation(top, grammar::NONE).unwrap_or(1.0);
            }
            if candidate.opens && candidate.region == HEADING {
                f[at + 2] = line.heading_words()[i];
            }
        }
        f[33] = line.look_heading()[0];
        f[34] = line.look_heading()[1];
        f[38] = line.evidence.operative.min(100.0);
        f[28] = flag(candidate.resume);
        f[29] = self.interrupted.get(&(self.ctx(line, candidate.starts), stream_of(candidate.region))).map_or(-1.0, |&region| f32::from(region));
        f[39] = flag(candidate.starts);
        f[40] = f32::from(line.start_kind());
        f[41] = line.page_place().min(50) as f32;
        f[42] = (1.0 + self.document_lines as f32).ln();
        f[43] = f32::from(self.document.min(50));
        // The stylesheet: the line's template, the outline's word on the line, the document's regime;
        // and a heading's level against the level its template takes, the outline names, and its
        // template's lines took on the first reading.
        let t = line.template();
        f[44..54].copy_from_slice(&[t.lines, t.pages, t.heading, t.monotone, t.anchored, t.anchor_level, t.level, t.edge, t.read_heading, t.read_level]);
        let against = |level: f32| if candidate.opens && candidate.region == HEADING && level > 0.0 { f32::from(candidate.level) - level } else { -9.0 };
        f[54] = against(t.level);
        f[55] = against(if line.anchor() > 0 { f32::from(line.anchor()) } else { t.anchor_level });
        f[56] = against(t.read_level);
        f[57] = flag(line.anchor() > 0);
        f[58] = f32::from(line.anchor());
        f[59] = flag(line.contents_title().is_some());
        f[60..64].copy_from_slice(&line.regime());
        f[64] = flag(stream.quoting && !candidate.starts);
        f[65] = t.beside;
        // The document's contents entries: whether the line answers one a heading has not, at what level,
        // the heading's level against it; whether the line reads as a contents list's row.
        let answered = self.answering(line);
        f[66] = flag(answered.is_some());
        f[67] = answered.map_or(0.0, |(_, level)| f32::from(level));
        f[68] = match answered {
            Some((_, level)) if candidate.opens && candidate.region == HEADING => f32::from(candidate.level) - f32::from(level),
            _ => -9.0,
        };
        f[69] = flag(line.contents_title().is_some());
        let (entries, unanswered) = self.entries.get(&self.document).map_or((0, 0), |(e, _)| (e.len(), e.iter().filter(|e| !e.answered).count()));
        f[70] = (1.0 + entries as f32).ln();
        f[71] = (1.0 + unanswered as f32).ln();
        // A body opening's depth, its step from the last body unit's, and against its numbering's.
        if candidate.depth > 0 {
            f[72] = f32::from(candidate.depth);
            f[73] = f32::from(candidate.depth) - outline.body_depth.max(1) as f32;
            f[74] = f32::from(candidate.depth) - self.stack_depth(line, candidate) as f32;
        } else {
            f[72..75].fill(-9.0);
        }
        f
    }

    /// The state after `line` makes `chosen`, and what the move made of the line.
    pub fn apply(&mut self, line: &LineView, chosen: &Move) -> Made {
        let mut made = Made::default();
        if chosen.starts {
            self.document += 1;
            self.document_lines = 0;
        }
        self.document_lines += 1;
        let ctx = self.ctx(line, false);
        // The line making the move reads in the context it leaves the state in (a quote opening a quotation).
        if chosen.opens {
            self.streams.entry(ctx).or_default().opens(chosen.region);
        }
        // A contents list's row the parser reads in the list waits for the heading that answers it; a heading
        // answers the first entry its title matches.
        if chosen.region == CONTENTS {
            if let Some(title) = line.contents_title() {
                let (entries, lefts) = self.entries.entry(self.document).or_default();
                let tier = match lefts.iter().position(|&left| (left - line.left()).abs() < 0.012) {
                    Some(at) => at,
                    None => {
                        lefts.push(line.left());
                        lefts.sort_by(f32::total_cmp);
                        lefts.iter().position(|&left| (left - line.left()).abs() < 0.012).unwrap_or(0)
                    }
                };
                entries.push(Entry { key: title.to_owned(), level: (tier + 1).min(9) as u8, answered: false, page: line.page() });
            }
        } else if chosen.opens && chosen.region == HEADING {
            if let Some((at, _)) = self.answering(line) {
                if let Some((entries, _)) = self.entries.get_mut(&self.document) {
                    entries[at].answered = true;
                }
            }
        }
        *self.answering.borrow_mut() = None;
        let outline_key = self.outline_key(false);
        let stream = self.streams.entry(ctx).or_default();
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
        if chosen.region == LINE_LABEL {
            if let Some(value) = line.line_number() {
                stream.line_number = Some(value);
            }
        }
        let outline = self.outlines.entry(outline_key).or_default();
        if chosen.opens && chosen.region == HEADING {
            made.level_step = 10 + i64::from(chosen.level - outline.heading.max(1));
            outline.heading = chosen.level;
            match outline.look_levels.iter_mut().find(|(look, _)| *look == line.look()) {
                Some(entry) => entry.1 = chosen.level,
                None => outline.look_levels.push((line.look(), chosen.level)),
            }
        } else if chosen.opens && RANKED_BODY.contains(&chosen.region) {
            let depth = usize::from(chosen.depth.max(1));
            made.depth_step = (10 + depth as i64 - outline.body_depth.max(1) as i64).clamp(2, 18);
            outline.body_depth = depth;
        }
        for since in &mut self.since {
            *since = since.saturating_add(1);
        }
        if chosen.opens {
            self.since[usize::from(chosen.region)] = 0;
        }
        if chosen.region != 0 {
            let key = (ctx, stream_of(chosen.region));
            if chosen.resume {
                self.interrupted.remove(&key);
            } else if let Some(&open) = self.open_regions.get(&key).filter(|&&open| open != chosen.region) {
                self.interrupted.insert(key, open);
            }
            self.open_regions.insert(key, chosen.region);
        }
        self.previous = Some(chosen.clone());
        made
    }
}

/// What a line reads of its template and of the document.
#[derive(Clone, Copy, Debug, Default)]
pub struct TemplateView {
    pub lines: f32,
    pub pages: f32,
    pub heading: f32,
    pub monotone: f32,
    pub anchored: f32,
    pub anchor_level: f32,
    pub level: f32,
    pub edge: f32,
    /// The share of its lines set beside another line on their row (a marginal note's lane).
    pub beside: f32,
    /// After a first reading: the share of its lines made headings, and their commonest level.
    pub read_heading: f32,
    pub read_level: f32,
}
