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
pub(crate) const HEADING: u8 = 3;
pub(crate) const NOTE: u8 = 5;
pub(crate) const PAGE_LABEL: u8 = 12;
pub(crate) const LINE_LABEL: u8 = 13;
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
    /// Goes on with the unit open in its stream before another region's took the stream (a paragraph
    /// picked up past a block quotation or a page's notes).
    pub(crate) resume: bool,
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

/// The parse so far: what is open in each context and each document's outline, the move before, and
/// how many lines ago each region last opened a unit.
#[derive(Clone, Debug)]
pub(crate) struct State {
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
        Self { streams: HashMap::new(), outlines: HashMap::new(), previous: None, since: [500; 17], open_regions: HashMap::new(), interrupted: HashMap::new() }
    }
}

/// A line as the state reads it: its reading context and document, its marker's readings, its note
/// label's value, its transcript line number, its look, the reference marks it sets, and its place in
/// reading order.
pub(crate) struct LineView<'a> {
    pub(crate) ctx: u16,
    pub(crate) document: u16,
    pub(crate) readings: &'a [MarkerReading],
    pub(crate) label: Option<u32>,
    pub(crate) line_number: Option<u32>,
    pub(crate) look: u32,
    pub(crate) references: &'a [(u32, u32, u32)],
    pub(crate) place: u32,
    /// The grammar tables for the line's document (`grammar::Grammar`), and how far its first word is a
    /// heading's by them.
    pub(crate) tables: Option<&'a Tables>,
    pub(crate) heading_word: f32,
    /// The document's view of the line's look before any move: the mean probability its lines open a
    /// heading by the local scores, and how many lines of it they hold likelier a heading's opening than
    /// not (log).
    pub(crate) look_heading: f32,
    pub(crate) look_heading_lines: f32,
}

/// What a move made of its line: the earlier line whose reference mark its note answered, a heading's
/// level step from the heading before it in its document (10 the same, as the gold steps), a body
/// unit's depth step from the body unit before it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Made {
    pub(crate) paired: Option<(u32, u32, u32)>,
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
];
pub(crate) const CANDIDATES: usize = 35;

/// The class of a family's count (`lines::COUNTS` order), 9 for none.
fn count_class(family: &str) -> f32 {
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
    fn open(&self, ctx: u16) -> Option<&Open> {
        self.streams.get(&ctx)
    }

    fn outline(&self, document: u16) -> Option<&Outline> {
        self.outlines.get(&document)
    }

    /// The region-and-opening classes the state itself proposes for a line, whatever the local scores
    /// say: going on with the unit open in each of the line's context's streams, or opening the next
    /// unit of its region; a heading where the line's look is one its document's headings took; a note opening on the next note number or on a number the body set; a
    /// transcript line number going on from the last; an opening of the region of an open frame its
    /// marker counts on from.
    pub(crate) fn proposed(&self, line: &LineView) -> Vec<usize> {
        let mut out = Vec::new();
        let opening = |region: u8| usize::from(region) * 2 + 1;
        let mut open: Vec<u8> = self.open_regions.iter().filter(|((ctx, _), _)| *ctx == line.ctx).map(|(_, &region)| region).collect();
        open.sort_unstable();
        for region in open {
            out.push(usize::from(region) * 2);
            out.push(opening(region));
        }
        // A heading where the line is set as a heading of its document was set before.
        if self.outline(line.document).is_some_and(|outline| outline.look_levels.iter().any(|(look, _)| *look == line.look)) {
            out.push(opening(HEADING));
        }
        if let Some(open) = self.open(line.ctx) {
            let values: Vec<u32> = line.label.into_iter().chain(line.readings.iter().map(|r| r.value)).collect();
            if values.iter().any(|&v| open.note.is_some_and(|last| v == last + 1) || open.unanswered.iter().any(|&(r, ..)| r == v)) {
                out.push(opening(NOTE));
            }
            if line.line_number.is_some_and(|v| open.line_number.is_some_and(|last| v == last + 1)) {
                out.push(opening(LINE_LABEL));
            }
            for reading in line.readings {
                if let Some(frame) = open.stack.iter().rev().find(|f| f.family == reading.family && reading.value == f.value + 1) {
                    out.push(opening(frame.region));
                }
            }
        }
        out
    }

    /// Every move a line may make among `classes`: each region opening or going on; an opening of a
    /// numbered region counting on each of the line's readings or on none; a heading at every level from
    /// one to two below the open heading's; a note opening on its label's value or a reading's or on
    /// none.
    pub(crate) fn candidates(&self, line: &LineView, classes: &[usize]) -> Vec<Move> {
        let heading = self.outline(line.document).map_or(0, |outline| outline.heading);
        let mut out = Vec::new();
        for &class in classes {
            let (region, opens) = ((class / 2) as u8, class % 2 == 1);
            let plain = Move { region, opens, reading: None, note: None, level: 0, resume: false };
            let mut bases = Vec::new();
            if opens && region == NOTE {
                let mut values: Vec<u32> = line.label.into_iter().collect();
                values.extend(line.readings.iter().map(|r| r.value).filter(|v| Some(*v) != line.label));
                bases.extend(values.into_iter().take(3).map(|v| Move { note: Some(v), ..plain }));
            } else if opens && NUMBERED.contains(&region) {
                bases.extend((0..line.readings.len().min(3)).map(|r| Move { reading: Some(r as u8), ..plain }));
            }
            bases.push(plain);
            let key = (line.ctx, stream_of(region));
            if !opens && self.interrupted.get(&key) == Some(&region) && self.open_regions.get(&key) != Some(&region) {
                bases.push(Move { resume: true, ..plain });
            }
            if opens && region == HEADING {
                let deepest = (heading.max(1) + 2).min(LEVELS);
                for base in bases {
                    out.extend((1..=deepest).map(|level| Move { level, ..base }));
                }
            } else {
                out.extend(bases);
            }
        }
        out
    }

    /// The candidate's features in `CANDIDATE` order.
    pub(crate) fn features(&self, line: &LineView, candidate: &Move) -> [f32; CANDIDATES] {
        let empty = Open::default();
        let stream = self.open(line.ctx).unwrap_or(&empty);
        let blank = Outline::default();
        let outline = self.outline(line.document).unwrap_or(&blank);
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
            match stream.stack.iter().rposition(|frame| frame.family == reading.family) {
                Some(depth) => {
                    f[8] = 1.0;
                    f[9] = (stream.stack.len() - 1 - depth) as f32;
                    f[10] = step(reading.value, &stream.stack[depth]);
                    f[13] = (depth + 1) as f32;
                    f[14] = depth.checked_sub(1).map_or(-1.0, |d| f32::from(stream.stack[d].region));
                }
                None => {
                    f[11] = 1.0;
                    f[12] = flag(reading.value == 1);
                    f[13] = (stream.stack.len() + 1).min(DEPTH) as f32;
                    f[14] = stream.stack.last().map_or(-1.0, |frame| f32::from(frame.region));
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
                let under = counted(&stream.stack, reading, HEADING).iter().filter(|frame| frame.region == HEADING).count() as i8;
                f32::from(level - under)
            });
        }
        // How likely the move's numbering is in real legal text (1 where the tables say nothing of it).
        f[30] = 1.0;
        f[31] = 1.0;
        f[32] = 1.0;
        if let Some(tables) = line.tables {
            let top = stream.stack.last().map_or("", |frame| frame.family.as_str());
            if let Some(reading) = reading {
                let relation = match stream.stack.iter().rposition(|frame| frame.family == reading.family) {
                    Some(depth) if depth + 1 == stream.stack.len() => if reading.value == stream.stack[depth].value + 1 { grammar::NEXT } else { grammar::SAME_OTHER },
                    Some(depth) => if reading.value == stream.stack[depth].value + 1 { grammar::POP_NEXT } else { grammar::POP_OTHER },
                    None => if reading.value == 1 { grammar::PUSH_1 } else { grammar::PUSH_OTHER },
                };
                f[30] = tables.relation(top, relation).unwrap_or(1.0);
                if !stream.stack.iter().any(|frame| frame.family == reading.family) {
                    f[31] = tables.child(top, &reading.family).unwrap_or(1.0);
                }
            } else if candidate.opens && NUMBERED.contains(&candidate.region) && !line.readings.is_empty() {
                f[30] = tables.relation(top, grammar::NONE).unwrap_or(1.0);
            }
            if candidate.opens && candidate.region == HEADING {
                f[32] = line.heading_word;
            }
        }
        f[33] = line.look_heading;
        f[34] = line.look_heading_lines;
        f[28] = flag(candidate.resume);
        f[29] = self.interrupted.get(&(line.ctx, stream_of(candidate.region))).map_or(-1.0, |&region| f32::from(region));
        f
    }

    /// The state after `line` makes `chosen`, and what the move made of the line.
    pub(crate) fn apply(&mut self, line: &LineView, chosen: &Move) -> Made {
        let mut made = Made::default();
        let stream = self.streams.entry(line.ctx).or_default();
        if let Some(reading) = chosen.reading.and_then(|r| line.readings.get(usize::from(r))) {
            stream.stack = counted(&stream.stack, reading, chosen.region);
        }
        if let (true, NOTE, Some(value)) = (chosen.opens, chosen.region, chosen.note) {
            stream.note = Some(value);
            if let Some(at) = stream.unanswered.iter().rposition(|&(r, ..)| r == value) {
                let (_, place, start, end) = stream.unanswered.remove(at);
                made.paired = Some((place, start, end));
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
        let body_frames = stream.stack.iter().filter(|f| RANKED_BODY.contains(&f.region)).count();
        let outline = self.outlines.entry(line.document).or_default();
        if chosen.opens && chosen.region == HEADING {
            made.level_step = 10 + i64::from(chosen.level - outline.heading.max(1));
            outline.heading = chosen.level;
            match outline.look_levels.iter_mut().find(|(look, _)| *look == line.look) {
                Some(entry) => entry.1 = chosen.level,
                None => outline.look_levels.push((line.look, chosen.level)),
            }
        } else if chosen.opens && RANKED_BODY.contains(&chosen.region) {
            let depth = body_frames.max(1);
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
            let key = (line.ctx, stream_of(chosen.region));
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
