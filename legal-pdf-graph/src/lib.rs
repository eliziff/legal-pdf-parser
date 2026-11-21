//! The structure graph from the decided pages: the text index every range points into
//! (`text_index`), a transcript's line numbers (`line_numbers`), the structure candidates (the
//! numbering runs the decision read, placed in the index, with the units it let them open,
//! `PdfResolutionInput`), the graph's nodes (`nodes`), each node's membership and context
//! (`contexts`), and the outline its units hang in (`outline`).

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules::text;
use crate as graph;

use crate::model::{Evidence, Role, Rule, Run, Slots};
use legal_pdf_core::model::Page;
use legal_structure_model::{CandidateGrammar, ScalarRange};
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

pub mod nodes;
pub use nodes::*;
pub mod contexts;
pub use contexts::*;
pub mod judgment;
pub use judgment::*;
pub mod outline;
pub mod write;
pub use outline::*;

/// A heading's printed number: "1.", "1.1", "I.", "(a)", "ii.", a bare "1", or a numbered word
/// ("PART 1", "Section 1.01").
const HEADING_MARKER: &str = r"((?i:PART|PARTIE|ARTICLE|SECTION|CHAPTER|CHAPITRE|SCHEDULE|ANNEXE|GC)\s+[0-9IVXLC]{1,6}(?:\.[0-9]{1,3})*\.?|\(?[0-9]{1,3}(?:\.[0-9]{1,3})*[.)]?|\(?[IVXLCivxlc]{1,6}[.)]|\(?[A-Za-z][.)])";

/// A heading's printed number before its title on its line.
fn heading_marker_re() -> &'static Regex {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER.get_or_init(|| {
        Regex::new(&format!(r"^\s*{HEADING_MARKER}\s+\S")).expect("heading marker regex")
    })
}

/// A heading's printed number alone on its line, its title on the lines under it.
fn heading_marker_alone_re() -> &'static Regex {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER.get_or_init(|| {
        Regex::new(&format!(r"^\s*{HEADING_MARKER}\s*$")).expect("heading marker regex")
    })
}

/// Whether a marker on a line the decision gave this role, by this rule, opens the unit its run
/// reads (`run`): a unit's text does; a heading does where it heads a division (a division's word,
/// a line the document's outline names, a heading of a rooted, consecutive count of provisions),
/// never as a ladder's rung, an enactment's history or a transcript's part; a contents line, a
/// quotation, a note, a table's cell and the page's edge never do.
pub fn opens_units(role: Role, rule: Rule, run: &Run) -> bool {
    match role {
        Role::Open | Role::Body | Role::ListItem => true,
        Role::Heading | Role::Title | Role::MarginalNote => match rule {
            Rule::DivisionHeading | Rule::Anchor => true,
            Rule::Ladder | Rule::HistoryHeading | Rule::TranscriptPart | Rule::TitleOnly => false,
            _ => run.grammar == CandidateGrammar::Hierarchy && run.rooted && run.consecutive,
        },
        Role::Contents
        | Role::Quote
        | Role::Note
        | Role::Table
        | Role::LineLabel
        | Role::PageLabel
        | Role::Furniture => false,
    }
}

/// Each line's citations, as spans of its text in UTF-16 code units, by line id.
pub fn citation_spans(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
) -> BTreeMap<String, Vec<ScalarRange>> {
    pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| {
            let units = |byte: u32| {
                line.text
                    .char_indices()
                    .take_while(|(at, _)| *at < byte as usize)
                    .map(|(_, c)| c.len_utf16())
                    .sum::<usize>()
            };
            let mut spans: Vec<ScalarRange> = slots
                .of_id(&line.id)
                .map(|slot| evidence.citations_of(slot))
                .unwrap_or_default()
                .iter()
                .map(|cited| ScalarRange {
                    start: units(cited.core.0),
                    end: units(cited.core.1),
                })
                .filter(|span| span.start < span.end)
                .collect();
            spans.sort_by_key(|span| (span.start, span.end));
            (line.id.clone(), spans)
        })
        .collect()
}
