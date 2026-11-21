//! The grammar state tier 0 walks the document's lines against: what is open in each reading context, the numbering
//! each counts on, a quotation's own count apart from its text's.

use crate::model::machine::machine::*;
use crate::model::regions::*;
use std::collections::HashMap;

/// The parse so far: the constituent document read and what is open in each of its contexts.
#[derive(Clone, Debug, Default)]
pub struct State {
    pub document: u16,
    pub streams: HashMap<u16, Open>,
}

impl State {
    /// The context a line reads in: its document and whether it is a translation's (a quotation in it is
    /// read as the parser reads it, not as the fences part it); after `starts`, the next document's.
    pub fn ctx(&self, line: &LineView, starts: bool) -> u16 {
        (self.document + u16::from(starts)).saturating_mul(2) + u16::from(line.translation())
    }

    /// What is open in the context `line` reads in (after `starts`, the next document's), where anything is.
    pub fn open(&self, line: &LineView, starts: bool) -> Option<&Open> {
        self.streams.get(&self.ctx(line, starts))
    }

    /// The depth a body opening's numbering puts it at: the body units open on its context's stack once
    /// its reading counts (the stack as it stands for an unnumbered one), one at least.
    pub fn stack_depth(&self, line: &LineView, candidate: &Move) -> usize {
        let empty = Open::default();
        let stream = self.open(line, candidate.starts).unwrap_or(&empty);
        let body = |frames: &[Frame]| {
            frames
                .iter()
                .filter(|f| RANKED_BODY.contains(&f.region))
                .count()
        };
        match line.reading(candidate.reading) {
            Some(reading) => body(&counted(stream.frames(), reading, candidate.region)),
            None => body(stream.frames()),
        }
        .max(1)
    }

    /// The state after `line` makes `chosen`.
    pub fn apply(&mut self, line: &LineView, chosen: &Move) {
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
            if let Some(marks) = &chosen.answers {
                // The marks the reading chose, exactly.
                for &(place, start, _) in marks {
                    if let Some(at) = stream
                        .unanswered
                        .iter()
                        .position(|&(r, line, s, _)| r == value && line == place && s == start)
                    {
                        stream.unanswered.remove(at);
                    }
                }
            } else {
                // The marks of its number the note answers: the nearest the text set, and up to three more.
                for _ in 0..4 {
                    let Some(at) = stream.unanswered.iter().rposition(|&(r, ..)| r == value) else {
                        break;
                    };
                    stream.unanswered.remove(at);
                }
            }
        }
        if REFERRING.contains(&chosen.region) && !line.references().is_empty() {
            stream
                .unanswered
                .retain(|(r, ..)| !line.references().iter().any(|(v, ..)| v == r));
            stream.unanswered.extend(
                line.references()
                    .iter()
                    .map(|&(v, start, end)| (v, line.place, start, end)),
            );
            let excess = stream.unanswered.len().saturating_sub(30);
            stream.unanswered.drain(..excess);
        }
    }
}
