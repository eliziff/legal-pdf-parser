//! The numbering runs: each reading context's runs of markers, each marker resolved to the unit it opens (`Runs`).

use crate::model::Slot;
use legal_structure_model::{CandidateGrammar, CandidateObservationV2, ResolutionRuleV2, UnitRole};

/// A place in a line's text: the line's slot and the UTF-16 code units before the place.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub(crate) struct Place {
    pub(crate) slot: Slot,
    pub(crate) offset: u32,
}

/// A run of markers one grammar reads in a reading context (`ctx`, an index into the fences'
/// contexts): its id, its grammar, whether it counts from its first value and without a gap, the
/// family its first marker counts in (`numbering_marker`; empty where that reads none), and its
/// members in text order.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct Run {
    pub(crate) ctx: u16,
    pub(crate) id: String,
    pub(crate) grammar: CandidateGrammar,
    pub(crate) rooted: bool,
    pub(crate) consecutive: bool,
    pub(crate) family: String,
    pub(crate) members: Vec<Member>,
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
pub(crate) struct Member {
    pub(crate) id: String,
    pub(crate) label: String,
    pub(crate) grammar_value: String,
    /// A provision's kind as the grammar read its marker ("section", "part", ...); none for a clause read by its
    /// number alone.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) locator_kind: Option<String>,
    pub(crate) slot: Slot,
    pub(crate) beside: Option<Slot>,
    pub(crate) text: Option<Slot>,
    pub(crate) parent: Option<u32>,
    pub(crate) level: u16,
    pub(crate) value: u32,
    pub(crate) marker: [Place; 2],
    pub(crate) content: Place,
    pub(crate) end: Place,
    pub(crate) counted: bool,
    pub(crate) kind: Option<UnitRole>,
    pub(crate) rule: ResolutionRuleV2,
    pub(crate) observations: Vec<CandidateObservationV2>,
    pub(crate) carries_prose: bool,
    pub(crate) title_shaped: bool,
    pub(crate) worded: bool,
}

/// The document's numbering runs, and the lines whose markers the repair rewrote, with their text before.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Runs {
    pub(crate) runs: Vec<Run>,
    pub(crate) before: Vec<(Slot, String)>,
}

impl Runs {
    /// Every member of every run.
    pub(crate) fn members(&self) -> impl Iterator<Item = &Member> {
        self.runs.iter().flat_map(|run| &run.members)
    }
}
