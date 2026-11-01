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

/// The regions a line may be in, the gold's (`labels::REGIONS`).
pub(crate) const REGIONS: [&str; 17] = [
    "none", "text", "list_item", "heading", "quote", "note", "table", "contents", "caption", "headnote", "marginal_note",
    "furniture", "page_label", "line_label", "title", "note_quote", "provision",
];
pub(crate) const TEXT: u8 = 1;
pub(crate) const HEADING: u8 = 3;
pub(crate) const QUOTE: u8 = 4;
pub(crate) const CONTENTS: u8 = 7;
pub(crate) const NOTE: u8 = 5;
pub(crate) const TABLE: u8 = 6;
pub(crate) const PAGE_LABEL: u8 = 12;
pub(crate) const LINE_LABEL: u8 = 13;
pub(crate) const TITLE: u8 = 14;
pub(crate) const NOTE_QUOTE: u8 = 15;
/// Regions whose units count on a marker.
const NUMBERED: [u8; 9] = [1, 2, 3, 4, 7, 8, 10, 14, 16];
/// Regions whose lines set reference marks notes answer: the body's, and headings' and titles'.
const REFERRING: [u8; 8] = [1, 2, 3, 4, 8, 9, 14, 16];
/// The body regions that take a depth (the tagger's ranked regions, headings aside).
pub(crate) const RANKED_BODY: [u8; 7] = [1, 16, 2, 4, 6, 8, 9];
/// A stack holds at most this many open units; headings run nine levels deep.
const DEPTH: usize = 8;
const LEVELS: i8 = 9;

/// Regions whose units run on in one stream, as the gold's do (`labels::stream`): the body's, notes',
/// marginal notes', furniture's, page and line labels'.
pub(crate) fn stream_of(region: u8) -> usize {
    match REGIONS[usize::from(region)] {
        "note" | "note_quote" => 1,
        "marginal_note" => 2,
        "furniture" => 3,
        "page_label" => 4,
        "line_label" => 5,
        _ => 0,
    }
}
/// One line's move: its region, whether it opens a unit, what the unit counts on (one of the line's
/// grammar readings, `reading`, an index; or a note's number, `note`), and a heading's level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    /// A note's opening: the reference marks it answers, each its line's place and its first character,
    /// where the reading chose them (none: the marks of its number the text set last).
    pub(crate) answers: [Option<(u32, u32)>; 4],
}

impl Move {
    /// The move's class among the region-and-opening moves the local scores rank (`region * 2 +
    /// opens`).
    pub(crate) fn class(&self) -> usize {
        usize::from(self.region) * 2 + usize::from(self.opens)
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
    pub(crate) line_number: Option<u32>,
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
}

/// What a document's outline holds so far, in reading order.
#[derive(Clone, Debug, Default)]
pub(crate) struct Outline {
    /// The open heading's level (0 before the first).
    pub(crate) heading: i8,
    /// Each look a heading took and the level it last took.
    pub(crate) look_levels: Vec<(u32, i8)>,
    /// The last body unit's depth (0 before the first).
    pub(crate) body_depth: usize,
}

/// A contents list's entry as the parser read it: its title (normalized, `bookmarks::normalize_title`), its
/// level (its indent's tier among the document's entries), and whether a heading has answered it.
#[derive(Clone, Debug)]
pub(crate) struct Entry {
    pub(crate) key: String,
    pub(crate) level: u8,
    pub(crate) answered: bool,
    /// The page the entry is printed on: the heading it names stands on a later one.
    pub(crate) page: u32,
}

/// The parse so far: what is open in each context and each document's outline, the move before, how
/// many lines ago each region last opened a unit, the constituent document read and its lines, and each
/// document's contents entries with the indents they stand at.
#[derive(Clone, Debug)]
pub(crate) struct State {
    pub(crate) document: u16,
    pub(crate) document_lines: u32,
    pub(crate) entries: HashMap<u16, (Vec<Entry>, Vec<f32>)>,
    /// The unanswered entry the line last asked about answers it (its place among the entries, its level).
    answering: std::cell::RefCell<Option<(u32, Option<(usize, u8)>)>>,
    pub(crate) streams: HashMap<u16, Open>,
    pub(crate) outlines: HashMap<u16, Outline>,
    pub(crate) previous: Option<Move>,
    pub(crate) since: [u32; 17],
    /// The region of the unit open in each context's each stream (`stream_of`).
    pub(crate) open_regions: HashMap<(u16, usize), u8>,
    /// The region of the unit each stream held before its open one's region took it.
    pub(crate) interrupted: HashMap<(u16, usize), u8>,
}

impl Default for State {
    fn default() -> Self {
        // No region has opened a unit yet: as long ago as `features` counts.
        Self { document: 0, document_lines: 0, entries: HashMap::new(), answering: Default::default(), streams: HashMap::new(), outlines: HashMap::new(), previous: None, since: [500; 17], open_regions: HashMap::new(), interrupted: HashMap::new() }
    }
}

/// A line as the state reads it: whether it reads in a parallel translation's column, what speaks to its
/// opening a constituent document, its marker's readings, its note label's value, its transcript line
/// number, its look, the reference marks it sets, and its place in reading order.
pub(crate) struct LineView<'a> {
    pub(crate) translation: bool,
    /// Whether the line may open a constituent document (near its page's top, or one the constituents
    /// reader names), the kind the reader names it (`constituents::KINDS`, 0 for none), and its place
    /// among its page's lines.
    pub(crate) may_start: bool,
    pub(crate) start_kind: u8,
    pub(crate) page_place: u32,
    pub(crate) readings: &'a [MarkerReading],
    pub(crate) label: Option<u32>,
    pub(crate) line_number: Option<u32>,
    pub(crate) look: u32,
    pub(crate) references: &'a [(u32, u32, u32)],
    pub(crate) place: u32,
    /// The grammar tables (`grammar::Grammar`), and how far the line's first word is a heading's by the
    /// instruments' and the others'; and the document's operative words in a thousand of its words, which
    /// of the two a move should go by.
    pub(crate) grammar: Option<&'a Grammar>,
    pub(crate) heading_words: [f32; 2],
    pub(crate) operative: f32,
    /// The document's view of the line's look before any move: the mean probability its lines open a
    /// heading by the local scores, and how many lines of it they hold likelier a heading's opening than
    /// not (log).
    pub(crate) look_heading: f32,
    pub(crate) look_heading_lines: f32,
    /// The document's stylesheet: the line's template, the level the document's outline names it at
    /// (0 for none), whether a contents list holds it, and the document's regime (`Stylesheet`).
    pub(crate) template: TemplateView,
    pub(crate) anchor: u16,
    pub(crate) listed: bool,
    pub(crate) regime: [f32; 4],
    /// The line's text normalized as a title is matched (`bookmarks::normalize_title`, empty for a long
    /// line); read as a contents list's row, its title so normalized where it prints a locator, and where
    /// it starts on its page (a share of the page's width).
    pub(crate) title_key: String,
    pub(crate) contents_title: Option<String>,
    pub(crate) left: f32,
    pub(crate) page: u32,
}

/// What a move made of its line: the earlier line whose reference mark its note answered, a heading's
/// level step from the heading before it in its document (10 the same, as the gold steps), a body
/// unit's depth step from the body unit before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Made {
    pub(crate) paired: Option<(u32, u32, u32)>,
    /// The other marks of the note's number it answered too (a note several places in the text cite).
    pub(crate) also: [Option<(u32, u32, u32)>; 3],
    pub(crate) level_step: i64,
    pub(crate) depth_step: i64,
}

/// The names of a candidate move's features, read off the move and the state: the move itself, the
/// move before, how its marker or note number fits what is open, and a heading's level against the
/// open heading's, its look's last level and its place on the stack.
pub(crate) const CANDIDATE: &[&str] = &[
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
pub(crate) const CANDIDATES: usize = 75;
/// A body unit's depth is tried this far below the last body unit's, and at the top.
const DEPTH_SPAN: usize = 2;

/// The class of a family's count (`lines::COUNTS` order), 9 for none.
pub(crate) fn count_class(family: &str) -> f32 {
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
    pub(crate) fn ctx(&self, line: &LineView, starts: bool) -> u16 {
        (self.document + u16::from(starts)).saturating_mul(2) + u16::from(line.translation)
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
    pub(crate) fn outline_key(&self, _line: &LineView, starts: bool) -> u16 {
        self.document + u16::from(starts)
    }

    /// The depth a body opening's numbering puts it at: the body units open on its context's stack once
    /// its reading counts (the stack as it stands for an unnumbered one), one at least.
    pub(crate) fn stack_depth(&self, line: &LineView, candidate: &Move) -> usize {
        let empty = Open::default();
        let stream = self.open(self.ctx(line, candidate.starts)).unwrap_or(&empty);
        let body = |frames: &[Frame]| frames.iter().filter(|f| RANKED_BODY.contains(&f.region)).count();
        match candidate.reading.and_then(|r| line.readings.get(usize::from(r))) {
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
    pub(crate) fn proposed(&self, line: &LineView) -> Vec<usize> {
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
        if self.outline(self.outline_key(line, false)).is_some_and(|outline| outline.look_levels.iter().any(|(look, _)| *look == line.look)) {
            out.push(opening(HEADING));
        }
        if let Some(open) = self.open(ctx) {
            let values: Vec<u32> = line.label.into_iter().chain(line.readings.iter().map(|r| r.value)).collect();
            if values.iter().any(|&v| open.note.is_some_and(|last| v == last + 1) || open.unanswered.iter().any(|&(r, ..)| r == v)) {
                out.push(opening(NOTE));
            }
            if line.line_number.is_some_and(|v| open.line_number.is_some_and(|last| v == last + 1)) {
                out.push(opening(LINE_LABEL));
            }
            for reading in line.readings {
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
    pub(crate) fn candidates(&self, line: &LineView, classes: &[usize]) -> Vec<Move> {
        let outline = self.outline(self.outline_key(line, false));
        let heading = outline.map_or(0, |outline| outline.heading);
        // The depths a body opening is tried at: the top, and from two above the last body unit's to two
        // below it; the numbering's depth among them (`stack_depth`).
        let last = outline.map_or(0, |outline| outline.body_depth).max(1);
        let mut depths: Vec<u8> = std::iter::once(1).chain(last.saturating_sub(DEPTH_SPAN).max(1)..=(last + DEPTH_SPAN).min(DEPTH + 1)).map(|d| d as u8).collect();
        depths.dedup();
        let mut out = Vec::new();
        for &class in classes {
            let (region, opens) = ((class / 2) as u8, class % 2 == 1);
            let plain = Move { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, resumes: None, rank: None, starts: false, answers: [None; 4] };
            let mut bases = Vec::new();
            if opens && region == NOTE {
                // A note counts on a number: its label's, or an arabic reading's (a lettered note answers no
                // figure in the text).
                let mut values: Vec<u32> = line.label.into_iter().collect();
                values.extend(line.readings.iter().filter(|r| count_class(&r.family) == 0.0).map(|r| r.value).filter(|v| Some(*v) != line.label));
                bases.extend(values.into_iter().take(3).map(|v| Move { note: Some(v), ..plain }));
            } else if opens && NUMBERED.contains(&region) {
                bases.extend((0..line.readings.len().min(3)).map(|r| Move { reading: Some(r as u8), ..plain }));
            }
            bases.push(plain);
            let key = (self.ctx(line, false), stream_of(region));
            if !opens && self.interrupted.get(&key) == Some(&region) && self.open_regions.get(&key) != Some(&region) {
                bases.push(Move { resume: true, ..plain });
            }
            if opens && region == HEADING {
                // A heading may open as much as three levels under the open one (a schedule's own parts
                // under a chapter's heading).
                let deepest = (heading.max(1) + 3).min(LEVELS);
                for base in bases {
                    out.extend((1..=deepest).map(|level| Move { level, ..base }));
                }
            } else if opens && RANKED_BODY.contains(&region) {
                for base in bases {
                    let mut tried = depths.clone();
                    let by_stack = self.stack_depth(line, &base).min(DEPTH + 1) as u8;
                    if !tried.contains(&by_stack) {
                        tried.push(by_stack);
                    }
                    out.extend(tried.into_iter().map(|depth| Move { depth, ..base }));
                }
            } else {
                out.extend(bases);
            }
        }
        // A constituent document opens on a unit's opening, its first heading or body unit at the top.
        if line.may_start {
            let starting: Vec<Move> = out.iter().filter(|m| m.opens && !m.resume && m.depth <= 1 && m.level <= 2).map(|m| Move { starts: true, ..*m }).collect();
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
        let found = (line.contents_title.is_none() && line.title_key.chars().count() >= 3).then(|| self.entries.get(&self.document)).flatten().and_then(|(entries, _)| {
            let key = line.title_key.as_str();
            // A part of a title matches where it holds half of the other or more.
            let (length, half) = (key.chars().count(), |of: &str| of.chars().count().div_ceil(2));
            entries.iter().position(|entry| !entry.answered && entry.page < line.page && (entry.key == key
                || (length >= 8 && length >= half(&entry.key) && (entry.key.starts_with(key) || entry.key.ends_with(key)))
                || (entry.key.chars().count() >= 8 && entry.key.chars().count() >= half(key) && key.ends_with(entry.key.as_str()))))
                .map(|at| (at, entries[at].level))
        });
        *self.answering.borrow_mut() = Some((line.place, found));
        found
    }

    /// The candidate's features in `CANDIDATE` order.
    pub(crate) fn features(&self, line: &LineView, candidate: &Move) -> [f32; CANDIDATES] {
        // A move opening a document reads against that document's state: nothing open yet.
        let empty = Open::default();
        let stream = self.open(self.ctx(line, candidate.starts)).unwrap_or(&empty);
        let blank = Outline::default();
        let outline = self.outline(self.outline_key(line, candidate.starts)).unwrap_or(&blank);
        let frames = stream.frames();
        let flag = |b: bool| f32::from(u8::from(b));
        let mut f = [0f32; CANDIDATES];
        f[0] = f32::from(candidate.region);
        f[1] = flag(candidate.opens);
        f[2] = self.previous.map_or(-1.0, |m| f32::from(m.region));
        f[3] = flag(self.previous.is_some_and(|m| m.opens));
        f[4] = flag(self.previous.is_some_and(|m| m.region == candidate.region));
        f[5] = (self.since[usize::from(candidate.region)] as f32).min(500.0);
        f[7] = 9.0;
        f[14] = -1.0;
        let reading = candidate.reading.and_then(|r| line.readings.get(usize::from(r)));
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
            if let Some(value) = line.line_number {
                f[21] = flag(stream.line_number.is_some_and(|last| value == last + 1));
                f[22] = flag(value == 1);
            }
        }
        if candidate.opens && candidate.region == HEADING {
            let level = candidate.level;
            f[23] = f32::from(level - outline.heading.max(1));
            f[24] = f32::from(level);
            let looked = outline.look_levels.iter().find(|(look, _)| *look == line.look).map(|&(_, l)| l);
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
        let tables = line.grammar.map(|g| [&g.instruments, &g.others]);
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
            } else if candidate.opens && NUMBERED.contains(&candidate.region) && !line.readings.is_empty() {
                f[at] = tables.relation(top, grammar::NONE).unwrap_or(1.0);
            }
            if candidate.opens && candidate.region == HEADING {
                f[at + 2] = line.heading_words[i];
            }
        }
        f[33] = line.look_heading;
        f[34] = line.look_heading_lines;
        f[38] = line.operative.min(100.0);
        f[28] = flag(candidate.resume);
        f[29] = self.interrupted.get(&(self.ctx(line, candidate.starts), stream_of(candidate.region))).map_or(-1.0, |&region| f32::from(region));
        f[39] = flag(candidate.starts);
        f[40] = f32::from(line.start_kind);
        f[41] = line.page_place.min(50) as f32;
        f[42] = (1.0 + self.document_lines as f32).ln();
        f[43] = f32::from(self.document.min(50));
        // The stylesheet: the line's template, the outline's word on the line, the document's regime;
        // and a heading's level against the level its template takes, the outline names, and its
        // template's lines took on the first reading.
        let t = &line.template;
        f[44..54].copy_from_slice(&[t.lines, t.pages, t.heading, t.monotone, t.anchored, t.anchor_level, t.level, t.edge, t.read_heading, t.read_level]);
        let against = |level: f32| if candidate.opens && candidate.region == HEADING && level > 0.0 { f32::from(candidate.level) - level } else { -9.0 };
        f[54] = against(t.level);
        f[55] = against(if line.anchor > 0 { f32::from(line.anchor) } else { t.anchor_level });
        f[56] = against(t.read_level);
        f[57] = flag(line.anchor > 0);
        f[58] = f32::from(line.anchor);
        f[59] = flag(line.listed);
        f[60..64].copy_from_slice(&line.regime);
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
        f[69] = flag(line.contents_title.is_some());
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
    pub(crate) fn apply(&mut self, line: &LineView, chosen: &Move) -> Made {
        let mut made = Made::default();
        if chosen.starts {
            self.document += 1;
            self.document_lines = 0;
        }
        self.document_lines += 1;
        let ctx = self.ctx(line, false);
        // A quote opens a quotation the context reads until a paragraph of the text's own opens; the line
        // making the move reads in the context it leaves the state in.
        {
            let stream = self.streams.entry(ctx).or_default();
            if chosen.opens && chosen.region == QUOTE {
                if !stream.quoting {
                    stream.quoted.clear();
                }
                stream.quoting = true;
            } else if chosen.opens && chosen.region == TEXT {
                stream.quoting = false;
            }
        }
        // A contents list's row the parser reads in the list waits for the heading that answers it; a heading
        // answers the first entry its title matches.
        if chosen.region == CONTENTS {
            if let Some(title) = &line.contents_title {
                let (entries, lefts) = self.entries.entry(self.document).or_default();
                let tier = match lefts.iter().position(|&left| (left - line.left).abs() < 0.012) {
                    Some(at) => at,
                    None => {
                        lefts.push(line.left);
                        lefts.sort_by(f32::total_cmp);
                        lefts.iter().position(|&left| (left - line.left).abs() < 0.012).unwrap_or(0)
                    }
                };
                entries.push(Entry { key: title.clone(), level: (tier + 1).min(9) as u8, answered: false, page: line.page });
            }
        } else if chosen.opens && chosen.region == HEADING {
            if let Some((at, _)) = self.answering(line) {
                if let Some((entries, _)) = self.entries.get_mut(&self.document) {
                    entries[at].answered = true;
                }
            }
        }
        *self.answering.borrow_mut() = None;
        let outline_key = self.outline_key(line, false);
        let stream = self.streams.entry(ctx).or_default();
        if let Some(reading) = chosen.reading.and_then(|r| line.readings.get(usize::from(r))) {
            let frames = counted(stream.frames(), reading, chosen.region);
            *stream.frames_mut() = frames;
        }
        if let (true, NOTE, Some(value)) = (chosen.opens, chosen.region, chosen.note) {
            stream.note = Some(value);
            if chosen.answers.iter().any(Option::is_some) {
                // The marks the reading chose, exactly.
                let mut found = Vec::new();
                for &(place, start) in chosen.answers.iter().flatten() {
                    if let Some(at) = stream.unanswered.iter().position(|&(r, line, s, _)| r == value && line == place && s == start) {
                        let (_, line, s, e) = stream.unanswered.remove(at);
                        found.push((line, s, e));
                    }
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
        if REFERRING.contains(&chosen.region) && !line.references.is_empty() {
            stream.unanswered.retain(|(r, ..)| !line.references.iter().any(|(v, ..)| v == r));
            stream.unanswered.extend(line.references.iter().map(|&(v, start, end)| (v, line.place, start, end)));
            let excess = stream.unanswered.len().saturating_sub(30);
            stream.unanswered.drain(..excess);
        }
        if chosen.region == LINE_LABEL {
            if let Some(value) = line.line_number {
                stream.line_number = Some(value);
            }
        }
        let outline = self.outlines.entry(outline_key).or_default();
        if chosen.opens && chosen.region == HEADING {
            made.level_step = 10 + i64::from(chosen.level - outline.heading.max(1));
            outline.heading = chosen.level;
            match outline.look_levels.iter_mut().find(|(look, _)| *look == line.look) {
                Some(entry) => entry.1 = chosen.level,
                None => outline.look_levels.push((line.look, chosen.level)),
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
        self.previous = Some(*chosen);
        made
    }
}
