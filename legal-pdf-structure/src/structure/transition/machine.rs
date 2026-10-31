//! The parser's moves and state. Each line in reading order makes one move: it opens a unit of a
//! region or goes on with the one open in its stream, and a line that opens may count on its marker
//! (one of its grammar readings) or on a note label. What is open is kept per reading context: the
//! stack of numbered units (family, value, region: PART II > 12 > (2) > (a)), the last note opened and
//! the reference marks the body set that no note has answered, the last transcript line number. A
//! move is scored on the line's evidence and on how it fits this state; applying it moves the state.

use super::*;

/// The regions a line may be in, the gold's (`labels::REGIONS`).
pub(crate) const REGIONS: [&str; 17] = [
    "none", "text", "list_item", "heading", "quote", "note", "table", "contents", "caption", "headnote", "marginal_note",
    "furniture", "page_label", "line_label", "title", "note_quote", "provision",
];
pub(crate) const NOTE: u8 = 5;
pub(crate) const LINE_LABEL: u8 = 13;
/// Regions whose units count on a marker.
const NUMBERED: [u8; 9] = [1, 2, 3, 4, 7, 8, 10, 14, 16];
/// Regions of the body, whose reference marks notes answer.
const BODY: [u8; 6] = [1, 2, 4, 8, 9, 16];
/// A stack holds at most this many open units.
const DEPTH: usize = 8;

/// One line's move: its region, whether it opens a unit, and what the unit counts on: one of the
/// line's grammar readings (`reading`, an index), or a note's number (`note`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Move {
    pub(crate) region: u8,
    pub(crate) opens: bool,
    pub(crate) reading: Option<u8>,
    pub(crate) note: Option<u32>,
}

impl Move {
    /// The move's class among the region-and-opening moves the local scores rank (`region * 2 +
    /// opens`).
    pub(crate) fn class(&self) -> usize {
        usize::from(self.region) * 2 + usize::from(self.opens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Frame {
    pub(crate) family: String,
    pub(crate) value: u32,
    pub(crate) region: u8,
}

/// What is open in one reading context.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct Open {
    pub(crate) stack: Vec<Frame>,
    pub(crate) note: Option<u32>,
    pub(crate) unanswered: Vec<u32>,
    pub(crate) line_number: Option<u32>,
}

/// The parse so far: what is open in each context, the move before, and how many lines ago each region last
/// opened a unit.
#[derive(Clone, Debug, Default)]
pub(crate) struct State {
    pub(crate) streams: HashMap<u16, Open>,
    pub(crate) previous: Option<Move>,
    pub(crate) since: [u32; 17],
}

/// The names of a candidate move's features, read off the move and the state: the move itself, the
/// move before, and how its marker or note number fits what is open.
pub(crate) const CANDIDATE: &[&str] = &[
    "region", "opens", "previous_region", "previous_opens", "same_region", "since_region",
    "counts", "count_class", "found_open", "levels_up", "step", "pushes", "push_from_one", "depth_after", "under_region",
    "note_counts", "note_next", "note_first", "note_answers", "unanswered", "note_plain",
    "line_number_next", "line_number_first",
];

/// The class of a family's count (`lines::COUNTS` order), 9 for none.
fn count_class(family: &str) -> f32 {
    let count = family.rsplit(':').next().unwrap_or("");
    ["1", "1.1", "1.1.1", "1.1.1.1", "a", "A", "i", "I", "bullet"].iter().position(|c| *c == count).map_or(9.0, |c| c as f32)
}

/// A move's step from the frame it pops to, its value against the frame's: 1 the next, 0 the same.
fn step(value: u32, frame: &Frame) -> f32 {
    (i64::from(value) - i64::from(frame.value)).clamp(-5, 5) as f32
}

impl State {
    /// Every move a line may make: each region opening or going on, an opening of a numbered
    /// region counting on each of the line's readings or on none, a note opening on its label's value
    /// or on a reading's or on none. `classes` are the region-and-opening classes kept (the local
    /// scores' best).
    pub(crate) fn candidates(&self, readings: &[MarkerReading], label: Option<u32>, classes: &[usize]) -> Vec<Move> {
        let mut out = Vec::new();
        for &class in classes {
            let (region, opens) = ((class / 2) as u8, class % 2 == 1);
            let plain = Move { region, opens, reading: None, note: None };
            if opens && region == NOTE {
                let mut values: Vec<u32> = label.into_iter().collect();
                values.extend(readings.iter().map(|r| r.value).filter(|v| Some(*v) != label));
                out.extend(values.into_iter().take(3).map(|v| Move { note: Some(v), ..plain }));
            } else if opens && NUMBERED.contains(&region) {
                out.extend((0..readings.len().min(3)).map(|r| Move { reading: Some(r as u8), ..plain }));
            }
            out.push(plain);
        }
        out
    }

    /// The candidate's features in `CANDIDATE` order.
    pub(crate) fn features(&self, ctx: u16, readings: &[MarkerReading], line_value: Option<u32>, candidate: &Move) -> [f32; 23] {
        let empty = Open::default();
        let stream = self.streams.get(&ctx).unwrap_or(&empty);
        let flag = |b: bool| f32::from(u8::from(b));
        let mut f = [0f32; 23];
        f[0] = f32::from(candidate.region);
        f[1] = flag(candidate.opens);
        f[2] = self.previous.map_or(-1.0, |m| f32::from(m.region));
        f[3] = flag(self.previous.is_some_and(|m| m.opens));
        f[4] = flag(self.previous.is_some_and(|m| m.region == candidate.region));
        f[5] = (self.since[usize::from(candidate.region)] as f32).min(500.0);
        f[7] = 9.0;
        f[14] = -1.0;
        if let Some(reading) = candidate.reading.and_then(|r| readings.get(usize::from(r))) {
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
                    f[18] = flag(stream.unanswered.contains(&value));
                }
                None => f[20] = 1.0,
            }
        }
        f[19] = stream.unanswered.len().min(30) as f32;
        if candidate.region == LINE_LABEL {
            if let Some(value) = line_value {
                f[21] = flag(stream.line_number.is_some_and(|last| value == last + 1));
                f[22] = flag(value == 1);
            }
        }
        f
    }

    /// The state after the line at `ctx` makes `chosen`; `references` are the reference marks the
    /// line sets.
    pub(crate) fn apply(&mut self, ctx: u16, readings: &[MarkerReading], line_value: Option<u32>, references: &[u32], chosen: &Move) {
        let stream = self.streams.entry(ctx).or_default();
        if let Some(reading) = chosen.reading.and_then(|r| readings.get(usize::from(r))) {
            let frame = Frame { family: reading.family.clone(), value: reading.value, region: chosen.region };
            match stream.stack.iter().rposition(|open| open.family == reading.family) {
                Some(depth) => {
                    stream.stack.truncate(depth);
                    stream.stack.push(frame);
                }
                None => {
                    stream.stack.push(frame);
                    if stream.stack.len() > DEPTH {
                        stream.stack.remove(0);
                    }
                }
            }
        }
        if let (true, NOTE, Some(value)) = (chosen.opens, chosen.region, chosen.note) {
            stream.note = Some(value);
            stream.unanswered.retain(|&r| r != value);
        }
        if BODY.contains(&chosen.region) && !references.is_empty() {
            stream.unanswered.retain(|r| !references.contains(r));
            stream.unanswered.extend_from_slice(references);
            let excess = stream.unanswered.len().saturating_sub(30);
            stream.unanswered.drain(..excess);
        }
        if chosen.region == LINE_LABEL {
            if let Some(value) = line_value {
                stream.line_number = Some(value);
            }
        }
        for since in &mut self.since {
            *since = since.saturating_add(1);
        }
        if chosen.opens {
            self.since[usize::from(chosen.region)] = 0;
        }
        self.previous = Some(*chosen);
    }
}
