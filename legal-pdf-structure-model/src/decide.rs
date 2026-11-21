//! Tier 0's decision as every stage after it reads it: each line's move, place and reading, the outline's anchors, the
//! sequences of numbered lines, the rules heard, the residual heading records, and the notes.

use crate::machine::machine::Move;
use crate::notes::NoteKind;
use crate::fences::Fences;
use crate::{Role, Rule, Slot};
use legal_pdf_core::model::Diagnostic;
use legal_structure_model::UnitRole;

/// The terms a residual heading record weighs, and the looks it signs (decide's `OWN` and `LOOK` are as long).
pub const OWN_TERMS: usize = 11;
pub const LOOK_TERMS: usize = 3;

/// A unit a line opens: its run (an index into the document's runs) and its kind.
pub type Opening = (u32, UnitRole);

/// Lines a confirmed or found bookmark or contents entry names: its lines, its level, and whether
/// its title is found whole.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct OutlineAnchor {
    pub lines: Vec<Slot>,
    pub level: u16,
    pub confirmed: bool,
}

/// Each reading context's lines in reading order, its line labels, folios, furniture and contents
/// left out; and each line's place among its context's lines (`u32::MAX` for a line left out).
#[derive(Debug, Default)]
pub struct Sequences {
    pub lines: Vec<Vec<Slot>>,
    pub at: Vec<u32>,
}

impl Sequences {
    /// The line `step` places after (or before, negative) the slot's line in its context.
    pub fn beside(&self, fences: &Fences, slot: Slot, step: isize) -> Option<Slot> {
        let at = self.at[slot.index()];
        if at == u32::MAX {
            return None;
        }
        let lines = &self.lines[usize::from(fences.ctx[slot.index()])];
        at.checked_add_signed(step as i32)
            .and_then(|at| lines.get(at as usize).copied())
    }
}

/// A reading a rule gives a line, for the one decision on it (`tier0`): the role, the rule, what the rule
/// weighs it at where it weighs (the residual's score), and whether the line goes on with the one above it.
#[derive(Clone, Copy, Debug)]
pub struct Heard {
    pub slot: Slot,
    pub role: Role,
    pub rule: Rule,
    pub weight: f32,
    pub joined: bool,
}

/// What each line is: its move (`moves`, in reading order, `place` each slot's), the one record of its role, the rule
/// that gave it, the unit it opens or goes on in, whether it goes on with the line above it and its heading level; and
/// what the hearing read, by slot: the reading of its opening marker that may open its unit (`openings`), the level a
/// rule gives it (0 for none: a transcript's part, an anchor's) and the anchor naming it (an index among the anchors,
/// plus one; 0 for none), each standing where the line's rule is that rule's; and, where the decision is traced, each
/// residual candidate as the residual weighed it. A trace holds its readings of the lines.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Decision {
    pub moves: Vec<Move>,
    pub place: Vec<u32>,
    #[serde(skip)]
    pub reading: Vec<Option<(u32, u32)>>,
    pub given: Vec<u8>,
    pub anchor: Vec<u32>,
    pub anchors: Vec<OutlineAnchor>,
    #[serde(skip)]
    pub residual: Vec<ResidualRecord>,
    #[serde(skip)]
    pub sequences: Sequences,
    /// While the rules are heard (`hear`): no rule's claim keeps another from a line, every line no fence took (`free`)
    /// is open to each rule, and what each says stands in `heard`. The moves hold the first reading of each line
    /// meanwhile, what a rule asks of the lines around it; `tier0::settle` puts the chosen move in its place.
    #[serde(skip)]
    pub free: Vec<bool>,
    #[serde(skip)]
    pub heard: Vec<Heard>,
}

impl Decision {
    /// The move of the line at slot index `at`.
    pub fn line(&self, at: usize) -> &Move {
        &self.moves[self.place[at] as usize]
    }
    pub fn line_mut(&mut self, at: usize) -> &mut Move {
        &mut self.moves[self.place[at] as usize]
    }
}

/// One residual candidate as the residual weighed it: its lines (slots), its reading context and look
/// (an index among the context's looks), the indicators its own log-odds sum (`OWN`, in order), the
/// measures they were read from, the vetoes that bar it (`VETO_*`), its look's signs (`LOOK`, in
/// order) and members, and the scores and outcome.
#[derive(Clone, Debug, serde::Serialize)]
pub struct ResidualRecord {
    pub lines: Vec<u32>,
    pub context: u32,
    pub look: u32,
    pub terms: [bool; OWN_TERMS],
    pub alone: bool,
    pub above: f64,
    pub below: Option<f64>,
    pub words: u32,
    pub caps: bool,
    pub body: bool,
    pub larger: bool,
    pub numbered: bool,
    pub vetoes: u16,
    pub signs: [bool; LOOK_TERMS],
    pub members: u32,
    pub own: f64,
    pub score: f64,
    pub heading: bool,
}

/// A page's note lines in one document and language, in reading order, and whether the first opens
/// with no label of its own and so goes on with the last note of the page before.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct NoteSegment {
    pub lines: Vec<Slot>,
    pub continued: bool,
}

/// What the note band and its lanes read of a document, by slot: each note line's kind; the lines
/// whose number the band read as a numbered paragraph's; the labels it read as no note's; an
/// endnote page's running heads; the notes headings; each page's note segments; and the warnings
/// of a band drawn on weak geometry. A trace holds its readings of the lines.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Notes {
    pub note: Vec<Option<NoteKind>>,
    pub paragraph_label: Vec<bool>,
    pub suppressed: Vec<bool>,
    pub running_head: Vec<bool>,
    pub heading: Vec<bool>,
    pub segments: Vec<NoteSegment>,
    pub diagnostics: Vec<Diagnostic>,
}
