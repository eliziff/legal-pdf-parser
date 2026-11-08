//! The numbering runs: each reading context's runs of markers, each marker resolved to the unit it opens (`Runs`).

use crate::Slot;
use legal_pdf_core::model::Word;
use legal_structure_model::{CandidateGrammar, CandidateObservationV2, ResolutionRuleV2, UnitRole};

/// A place in a line's text: the line's slot and the UTF-16 code units before the place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct Place {
    pub slot: Slot,
    pub offset: u32,
}

/// A run of markers one grammar reads in a reading context (`ctx`, an index into the fences'
/// contexts): its id, its grammar, whether it counts from its first value and without a gap, the
/// family its first marker counts in (`numbering_marker`; empty where that reads none), and its
/// members in text order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Run {
    pub ctx: u16,
    pub id: String,
    pub grammar: CandidateGrammar,
    pub rooted: bool,
    pub consecutive: bool,
    pub family: String,
    pub members: Vec<Member>,
}

/// A marker of a run: its candidate's id, label and the grammar's reading of it; the line it opens,
/// the line on its row carrying its text where it stands alone on its own, and the line carrying
/// its text where none follows it on its own line (that line beside it, else the next line of its
/// context that no other marker opens: "ARTICLE 31" over "PENSION PLAN"); the member it hangs from
/// (by its place in the run) and its depth; the value it counts; where its marker begins and ends,
/// where its text begins and where the grammar ends it; whether its number counts, the unit it
/// opens and the rule that says so; what its lines show; whether its text is prose, whether that
/// text is shaped as a title, and whether it has three words or more (eight letters or more).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Member {
    pub id: String,
    pub label: String,
    pub grammar_value: String,
    pub slot: Slot,
    pub beside: Option<Slot>,
    pub text: Option<Slot>,
    pub parent: Option<u32>,
    pub level: u16,
    pub value: u32,
    pub marker: [Place; 2],
    pub content: Place,
    pub end: Place,
    pub counted: bool,
    pub kind: Option<UnitRole>,
    pub rule: ResolutionRuleV2,
    pub observations: Vec<CandidateObservationV2>,
    pub carries_prose: bool,
    pub title_shaped: bool,
    pub worded: bool,
}

/// The document's numbering runs, and the lines whose markers the repair rewrote, with their text and
/// words as rewritten and their text before.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Runs {
    pub runs: Vec<Run>,
    pub edits: Vec<(Slot, String, Vec<Word>)>,
    pub before: Vec<(Slot, String)>,
}

impl Runs {
    /// Every member of every run.
    pub fn members(&self) -> impl Iterator<Item = &Member> {
        self.runs.iter().flat_map(|run| &run.members)
    }
}
