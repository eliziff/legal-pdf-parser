//! The structure graph from the decided pages: the text index every range points into
//! (`text_index`), a transcript's line numbers (`line_numbers`), the structure candidates (the
//! numbering runs the decision read, placed in the index, with the units it let them open,
//! `PdfResolutionInput`), the graph's nodes (`nodes`), each node's membership and context
//! (`contexts`), and the outline its units hang in (`outline`).

use super::{
    arabic_page_number, sentence_ended, starts_note_or_list, Decided, DocumentContext, Evidence, Member, Place, Role, Rule,
    Run, Runs, Slot, Slots,
};
use legal_pdf_core::model::{Line, NotePairClaim, NotePairKind, Page, Paragraph, PdfSourceSpan, Span, Word};
use legal_pdf_core::{Error, Result};
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateRoleV2, Derivation, DiagnosticSeverity, NodeKind, NoteBodyV2, NoteKindV2,
    NotePairClaimV2, ResolutionRuleV2, ScalarRange, ScalarText, StructureCandidateRun, StructureDiagnostic,
    StructureMarkerCandidate, StructureNode, TextAnchorV2,
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
mod outline;
pub(crate) use outline::*;

type BodyMeasures = HashMap<(DocumentContext, usize), (f64, f64)>;

#[derive(Debug)]
pub(super) struct PdfResolutionInput {
    pub(super) index: PdfTextIndex,
    pub(super) runs: Vec<StructureCandidateRun>,
    pub(super) evidence: Vec<CandidateEvidenceV2>,
    pub(super) roles: Vec<CandidateRoleV2>,
    pub(super) citation_spans: BTreeMap<String, Vec<PdfSourceSpan>>,
}

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

/// A body line that can carry a numbered item's text: not a running head or folio, nor a note.
fn item_text_line(line: &Line) -> bool {
    !line.exclude_from_body
        && line.note_region_mode.is_empty()
        && !matches!(line.region_type.as_str(), "header" | "footer")
}

/// A number alone on its line whose next line is set out left of it is no item's: a folio, a
/// column of figures.
fn stands_apart(marker: &Line, next: &Line) -> bool {
    let below = next.bbox[1] >= marker.bbox[3] - 0.5 * (marker.bbox[3] - marker.bbox[1]);
    below && next.bbox[0] < marker.bbox[0] - (next.bbox[3] - next.bbox[1])
}

/// The text index's lines in its order, with their pages and lines, and each one's reading context.
struct IndexedLines<'a> {
    index: &'a PdfTextIndex,
    lines: Vec<(&'a Page, &'a Line)>,
    contexts: Vec<DocumentContext>,
}

impl<'a> IndexedLines<'a> {
    fn of(index: &'a PdfTextIndex, pages: &'a [Page], decided: &Decided) -> Self {
        let mut ordered: Vec<&Page> = pages.iter().collect();
        ordered.sort_by_key(|page| page.index);
        let mut lines = Vec::with_capacity(index.lines.len());
        for page in ordered {
            let mut on_page: Vec<&Line> = page.lines.iter().collect();
            on_page.sort_by(|left, right| left.reading_order.cmp(&right.reading_order).then_with(|| left.id.cmp(&right.id)));
            lines.extend(on_page.into_iter().map(|line| (page, line)));
        }
        let contexts = lines.iter().map(|(_, line)| decided.reading_context(&line.id)).collect();
        Self { index, lines, contexts }
    }

    /// The reading context of the line holding the scalar `at`.
    fn context_at(&self, at: usize) -> DocumentContext {
        self.index.position_at(at).map_or((0, false), |place| self.contexts[place])
    }

    /// The line holding a candidate's marker, by its place among the index's lines.
    fn marker(&self, candidate: &StructureMarkerCandidate) -> Option<usize> {
        self.index.position_at(candidate.marker_range.start)
    }
}

/// Where a subsection's text ends when its number stands apart from it (see `stands_apart`).
fn apart_end(lines: &IndexedLines, candidate: &StructureMarkerCandidate) -> Option<usize> {
    let at = lines.marker(candidate)?;
    let (marker, context) = (lines.lines[at].1, lines.contexts[at]);
    let (start, line) = lines
        .index
        .overlapping(candidate.range)
        .filter(|&other| lines.index.lines[other].range.end > candidate.content_start && lines.contexts[other] == context)
        .map(|other| (lines.index.lines[other].range.start, lines.lines[other].1))
        .find(|(_, line)| item_text_line(line) && !line.text.trim().is_empty())?;
    stands_apart(marker, line).then_some(start)
}

/// Where a numbered item's text ends by the page's layout, when that is before the next number.
/// It ends at a heading, at a printed block of its own set apart from the item's margin and
/// the column its lines wrap to (a centred title, a signature, a cover page), and after a
/// finished sentence at a block in line with them (a jurat, an unnumbered paragraph). A block
/// set in at full measure (a quotation), one opening with its own marker, an unfinished
/// sentence, the text after a quotation and a full line at the top of the next page carry the
/// item on.
fn layout_end(lines: &IndexedLines, measures: &BodyMeasures, candidate: &StructureMarkerCandidate) -> Option<usize> {
    let at = lines.marker(candidate)?;
    let (marker, context) = (lines.lines[at].1, lines.contexts[at]);
    let (left, _) = measures.get(&(context, marker.page_index))?;
    let offset = marker.bbox[0] - left;
    let mut previous: Option<&Line> = None;
    let mut quoted = false;
    // Where the item's own lines wrap to, from the page's left edge.
    let mut hang = None;
    for other in lines.index.overlapping(candidate.range) {
        let indexed = &lines.index.lines[other];
        if lines.contexts[other] != context || indexed.range.end <= candidate.content_start {
            continue;
        }
        let line = lines.lines[other].1;
        if line.region_type == "heading" && indexed.range.start > candidate.content_start {
            return Some(indexed.range.start);
        }
        if !item_text_line(line) || line.text.trim().is_empty() {
            continue;
        }
        let Some(prior) = previous.replace(line) else {
            if stands_apart(marker, line) {
                return Some(indexed.range.start);
            }
            continue;
        };
        let new_page = line.page_index != prior.page_index;
        let Some(&(left, right)) = measures.get(&(context, line.page_index)) else {
            continue;
        };
        if !new_page && line.block_index == prior.block_index {
            if line.bbox[1] >= prior.bbox[3] {
                hang.get_or_insert(line.bbox[0] - left);
            }
            continue;
        }
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        let margin = left + offset;
        let full = line.bbox[2] - line.bbox[0] >= 0.6 * (right - left);
        let text = line.text.trim();
        if starts_note_or_list(text) || (line.bbox[0] > margin + height && full) {
            quoted |= line.bbox[0] > margin + height;
            continue;
        }
        let aligned = (line.bbox[0] - margin).abs() <= height
            || hang.is_some_and(|hang| (line.bbox[0] - left - hang).abs() <= height);
        // A line stopped well short of the measure with a blank line under it runs on to
        // nothing: a placeholder, a label, a form's blank.
        let parted = !new_page
            && prior.bbox[2] < right - 0.25 * (right - left)
            && line.bbox[1] - prior.bbox[3] > height;
        let carried = aligned
            && ((!sentence_ended(prior.text.trim()) && !parted) || (full && (quoted || new_page)));
        quoted = false;
        if !carried {
            return Some(indexed.range.start);
        }
    }
    None
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
fn opens_units(role: Role, rule: Rule, run: &Run) -> bool {
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

impl PdfResolutionInput {
    /// The structure candidates of the decided pages (their lines in reading order): each numbering
    /// run the decision read (`read`, ended and parted where the numbered paragraphs resume),
    /// placed in the text index; a subsection's text ended where a heading stands, and a numbered
    /// paragraph's or list item's where the page's layout ends it; each marker's lines and what they
    /// show; the unit the decision lets it open; and each line's citations (`evidence`).
    pub(super) fn from_decision(pages: &[Page], slots: &Slots, evidence: &Evidence, read: &Runs, decided: &Decided) -> Self {
        let index = PdfTextIndex::from_pages(pages);
        let (runs, candidate_evidence, roles, citation_spans) = {
            let lines = IndexedLines::of(&index, pages, decided);
            // Each slot's place among the index's lines.
            let mut place = vec![u32::MAX; slots.len()];
            for (at, (_, line)) in lines.lines.iter().enumerate() {
                if let Some(slot) = slots.of_id(&line.id) {
                    place[slot.index()] = at as u32;
                }
            }
            let range_of = |slot: Slot| index.lines.get(place[slot.index()] as usize).map(|line| line.range);
            let scalar = |at: Place| range_of(at.slot).map_or(0, |range| (range.start + at.offset as usize).min(range.end));
            let mut placed: Vec<(StructureCandidateRun, &Run)> = read
                .runs
                .iter()
                .map(|run| {
                    let markers: Vec<StructureMarkerCandidate> = run
                        .members
                        .iter()
                        .map(|member| {
                            let start = scalar(member.marker[0]);
                            let marker_end = scalar(member.marker[1]).max(start);
                            let end = scalar(member.end).max(marker_end);
                            let mut content_start = scalar(member.content).clamp(marker_end, end);
                            // A marker read apart from the line carrying its text: its text starts there.
                            if let Some(beside) = member.beside.and_then(range_of) {
                                if marker_end <= beside.start && beside.start < end {
                                    content_start = beside.start;
                                }
                            }
                            StructureMarkerCandidate {
                                id: member.id.clone(),
                                range: ScalarRange { start, end },
                                marker_range: ScalarRange { start, end: marker_end },
                                label: member.label.clone(),
                                grammar_value: member.grammar_value.clone(),
                                parent_candidate_id: member.parent.map(|parent| run.members[parent as usize].id.clone()),
                                level: usize::from(member.level),
                                content_start,
                            }
                        })
                        .collect();
                    let range = ScalarRange {
                        start: markers.iter().map(|marker| marker.range.start).min().unwrap_or(0),
                        end: markers.iter().map(|marker| marker.range.end).max().unwrap_or(0),
                    };
                    let placed = StructureCandidateRun {
                        id: run.id.clone(),
                        grammar: run.grammar,
                        range,
                        rooted: run.rooted,
                        consecutive: run.consecutive,
                        markers,
                    };
                    (placed, run)
                })
                .collect();
            placed.sort_by_key(|(run, _)| (run.range.start, grammar_rank(run.grammar), run.range.end));
            // The unit each marker opens: its run's, where the decision left its line a unit's text
            // and read the marker by this run (a marker two runs read opens one unit).
            let other_reading = |r: usize, member: &Member| {
                decided.reading(member.slot).is_some_and(|(run, at)| {
                    run as usize != r && read.runs[run as usize].members[at as usize].marker[0] == member.marker[0]
                })
            };
            let roles: Vec<CandidateRoleV2> = read
                .runs
                .iter()
                .enumerate()
                .flat_map(|(r, run)| run.members.iter().map(move |member| (r, run, member)))
                .map(|(r, run, member)| {
                    let other = other_reading(r, member);
                    let role = member
                        .kind
                        .filter(|_| !other && opens_units(decided.role(member.slot), decided.rule(member.slot), run));
                    let rule = match (role, member.kind) {
                        (None, Some(_)) if other => ResolutionRuleV2::ConflictingRoles,
                        (None, Some(_)) => ResolutionRuleV2::DirectExclusion,
                        _ => member.rule,
                    };
                    CandidateRoleV2 { candidate_id: member.id.clone(), role, rule, counted: member.counted }
                })
                .collect();
            let mut measures = BodyMeasures::new();
            for (at, (page, line)) in lines.lines.iter().enumerate().filter(|(_, (_, line))| item_text_line(line)) {
                let bounds = measures.entry((lines.contexts[at], page.index)).or_insert((line.bbox[0], line.bbox[2]));
                bounds.0 = bounds.0.min(line.bbox[0]);
                bounds.1 = bounds.1.max(line.bbox[2]);
            }
            // The runs come ended and parted where the numbered paragraphs resume (`Runs::read`). A
            // provision's marker other than a top-level section's or division's ends at a heading
            // the decision set inside it; a section's and its subsections' extent is otherwise their
            // grammar's; a numbered paragraph's or list item's ends where the layout ends it.
            let mut ended = HashSet::new();
            for (run, _) in &mut placed {
                let hierarchy = run.grammar == CandidateGrammar::Hierarchy;
                for candidate in &mut run.markers {
                    let context = lines.context_at(candidate.marker_range.start);
                    let label = candidate.label.trim_start_matches('(').to_lowercase();
                    let provision = hierarchy
                        && candidate.parent_candidate_id.is_none()
                        && (label.starts_with(|character: char| character.is_ascii_digit())
                            || ["part", "section", "article", "division", "chapter", "schedule"]
                                .iter()
                                .any(|word| label.starts_with(word)));
                    if hierarchy && !provision {
                        if let Some(heading) = index.overlapping(candidate.range).find(|&at| {
                            index.lines[at].range.start > candidate.content_start
                                && lines.contexts[at] == context
                                && lines.lines[at].1.region_type == "heading"
                        }) {
                            candidate.range.end = index.lines[heading].range.start;
                        }
                    }
                    if hierarchy {
                        if let Some(end) = candidate.parent_candidate_id.as_ref().and_then(|_| apart_end(&lines, candidate)) {
                            candidate.range.end = end;
                            ended.insert(candidate.id.clone());
                        }
                        continue;
                    }
                    if let Some(end) = layout_end(&lines, &measures, candidate) {
                        candidate.range.end = end;
                        ended.insert(candidate.id.clone());
                    }
                }
            }
            // Each marker's lines in its reading context: a running head, folio or print margin inside
            // its range is the page's, not the item's, unless the item is set wholly in them.
            let mut candidate_evidence = Vec::new();
            for (run, source) in &placed {
                for (candidate, member) in run.markers.iter().zip(&source.members) {
                    let context = lines.context_at(candidate.marker_range.start);
                    let own: Vec<usize> = index.overlapping(candidate.range).filter(|&at| lines.contexts[at] == context).collect();
                    let running = |at: &usize| matches!(lines.lines[*at].1.region_type.as_str(), "header" | "footer");
                    let line_ids: Vec<String> = if own.iter().all(running) {
                        own.iter().map(|&at| index.lines[at].line_id.clone()).collect()
                    } else {
                        own.iter().filter(|at| !running(at)).map(|&at| index.lines[at].line_id.clone()).collect()
                    };
                    candidate_evidence.push(CandidateEvidenceV2 {
                        candidate_id: candidate.id.clone(),
                        page_indexes: index.page_indexes_for_line_ids(&line_ids),
                        line_ids,
                        observations: member.observations.clone(),
                    });
                }
            }
            // A list ends where the layout ended one of its items before its next item.
            let runs: Vec<StructureCandidateRun> = placed
                .into_iter()
                .flat_map(|(run, _)| {
                    if run.grammar != CandidateGrammar::Enumerator {
                        return vec![run];
                    }
                    let mut parts = Vec::<StructureCandidateRun>::new();
                    for candidate in run.markers {
                        let continued = parts.last().is_some_and(|part| {
                            let prior = part.markers.last().expect("a part has an item");
                            !ended.contains(&prior.id)
                                && lines.context_at(candidate.marker_range.start) == lines.context_at(prior.marker_range.start)
                        });
                        match parts.last_mut() {
                            Some(part) if continued => {
                                part.range.end = part.range.end.max(candidate.range.end);
                                part.markers.push(candidate);
                            }
                            _ => parts.push(StructureCandidateRun {
                                id: format!("{}.{}", run.id, parts.len() + 1),
                                grammar: run.grammar,
                                range: candidate.range,
                                rooted: run.rooted,
                                consecutive: run.consecutive,
                                markers: vec![candidate],
                            }),
                        }
                    }
                    if let [part] = parts.as_mut_slice() {
                        part.id = run.id;
                    }
                    parts
                })
                .collect();
            // Each line's citations, as scalar spans of its text.
            let citation_spans = lines
                .lines
                .iter()
                .map(|(_, line)| {
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
                .collect::<BTreeMap<_, _>>();
            (runs, candidate_evidence, roles, citation_spans)
        };
        Self { index, runs, evidence: candidate_evidence, roles, citation_spans }
    }
}
