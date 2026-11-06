//! The structure graph from the decided pages: the text index every range points into
//! (`text_index`), a transcript's line numbers (`line_numbers`), the structure candidates (the
//! numbering runs the decision read, placed in the index, with the units it let them open,
//! `PdfResolutionInput`), the graph's nodes (`nodes`), each node's membership and context
//! (`contexts`), and the outline its units hang in (`outline`).

use super::{arabic_page_number, Decided, Evidence, Role, Rule, Run, Slots};
use legal_pdf_core::model::{Line, NotePairClaim, NotePairKind, Page, PdfSourceSpan, Span, Word};
use legal_pdf_core::{Error, Result};
use legal_structure_model::{
    CandidateGrammar, Derivation, DiagnosticSeverity, NodeKind, NoteBodyV2, NoteKindV2, NotePairClaimV2, ScalarRange, ScalarText,
    StructureDiagnostic, StructureNode, TextAnchorV2,
};
use regex::Regex;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

mod text_index;
pub use text_index::*;
mod line_numbers;
pub(crate) use line_numbers::*;
mod nodes;
pub(crate) use nodes::*;
mod contexts;
pub(crate) use contexts::*;
mod judgment;
pub(crate) use judgment::*;
mod outline;
pub(crate) use outline::*;

/// A heading's printed number: "1.", "1.1", "I.", "(a)", "ii.", a bare "1", or a numbered word
/// ("PART 1", "Section 1.01").
const HEADING_MARKER: &str = r"((?i:PART|PARTIE|ARTICLE|SECTION|CHAPTER|CHAPITRE|SCHEDULE|ANNEXE|GC)\s+[0-9IVXLC]{1,6}(?:\.[0-9]{1,3})*\.?|\(?[0-9]{1,3}(?:\.[0-9]{1,3})*[.)]?|\(?[IVXLCivxlc]{1,6}[.)]|\(?[A-Za-z][.)])";

/// A heading's printed number before its title on its line.
fn heading_marker_re() -> &'static Regex {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER.get_or_init(|| Regex::new(&format!(r"^\s*{HEADING_MARKER}\s+\S")).expect("heading marker regex"))
}

/// A heading's printed number alone on its line, its title on the lines under it.
fn heading_marker_alone_re() -> &'static Regex {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER.get_or_init(|| Regex::new(&format!(r"^\s*{HEADING_MARKER}\s*$")).expect("heading marker regex"))
}

/// The order runs read in at one place: a count of numbers, then a ladder, then a list.
pub(crate) fn grammar_rank(grammar: CandidateGrammar) -> u8 {
    match grammar {
        CandidateGrammar::Numeric => 0,
        CandidateGrammar::Hierarchy => 1,
        CandidateGrammar::Enumerator => 2,
    }
}

/// Whether a marker on a line the decision gave this role, by this rule, opens the unit its run
/// reads (`run`): a unit's text does; a heading does where it heads a division (a division's word,
/// a line the document's outline names, a heading of a rooted, consecutive count of provisions),
/// never as a ladder's rung, an enactment's history or a transcript's part; a contents line, a
/// quotation, a note, a table's cell and the page's edge never do.
pub(crate) fn opens_units(role: Role, rule: Rule, run: &Run) -> bool {
    match role {
        Role::Open | Role::Body | Role::ListItem => true,
        Role::Heading | Role::Title | Role::MarginalNote => match rule {
            Rule::DivisionHeading | Rule::Anchor => true,
            Rule::Ladder | Rule::HistoryHeading | Rule::TranscriptPart | Rule::TitleOnly => false,
            _ => run.grammar == CandidateGrammar::Hierarchy && run.rooted && run.consecutive,
        },
        Role::Contents | Role::Quote | Role::Note | Role::Table | Role::LineLabel | Role::PageLabel | Role::Furniture => false,
    }
}

/// Each line's citations, as scalar spans of its text, by line id.
pub(super) fn citation_spans(pages: &[Page], slots: &Slots, evidence: &Evidence) -> BTreeMap<String, Vec<PdfSourceSpan>> {
    pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| {
            let scalars = |byte: u32| line.text.char_indices().take_while(|(at, _)| *at < byte as usize).count();
            let mut spans: Vec<PdfSourceSpan> = slots
                .of_id(&line.id)
                .map(|slot| evidence.citations_of(slot))
                .unwrap_or_default()
                .iter()
                .map(|cited| PdfSourceSpan { start: scalars(cited.core.0), end: scalars(cited.core.1) })
                .filter(|span| span.start < span.end)
                .collect();
            spans.sort_by_key(|span| (span.start, span.end));
            (line.id.clone(), spans)
        })
        .collect()
}
