//! The numbering runs of each reading context, read once the fences and the reading order stand.
//! A scan's misread markers are restored first, document by document and language by language
//! (the engine's `repair_label_dropouts`, which restores paragraph and item numbers too), and the
//! lines it rewrites are measured again. Each context's lines then read as one text, its fenced
//! lines (line labels, folios, furniture, contents) left out: the engine reads its counts of
//! numbers, letters and bullets and its provision ladder there, with the instrument grammar's
//! refinements (its contents column, regnal heads, repeated subsections). A list's item and a
//! subsection end where the
//! context's numbered paragraphs resume, and a list parts there, before anything is resolved. What
//! each marker's line shows is read from the evidence and the fences, never from a role a reader
//! wrote: a transcript's question number beside its line-number column is a marker like any other,
//! the column itself being line labels the view leaves out. The engine resolves each marker to the
//! unit it opens and says whether its number counts. Every place a run names is a line's slot and a
//! place in its text.

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;

use crate::model::fences::{Continuation, Fences};
use crate::model::Geometry;
use crate::model::regions;
use crate::model::{Evidence, Role, Slot, Slots, SourceTag};
use crate::model::{Member, Place, Run, Runs};
use legal_pdf_core::model::{Line, LineZone, NoteMode, Page};
use legal_pdf_core::{Error, Result};
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, ResolutionRuleV2,
    StructureCandidateRun, StructureMarkerCandidate, UnitRole,
};
use std::collections::{BTreeMap, HashMap, HashSet};

/// The markers recognition misread, restored from the runs they stand in, in each document and
/// language with its lines in reading order (`repair_label_dropouts`): each rewritten line's slot
/// and its text before. The repair reads a fenced running head or folio as the page's head or
/// foot, a line label as one, and a line under the note separator as a note's.
pub fn repair_markers(pages: &mut [Page], fences: &Fences) -> Vec<(Slot, String)> {
    if !pages
        .iter()
        .flat_map(|page| &page.lines)
        .any(Line::recognized)
    {
        return Vec::new();
    }
    // TEMP lever "repair-columns" (remove before landing): each line's column as layout reads its page's (`Columns`:
    // its left and right edge), by slot, read before the lines are lent out.
    let by_column =
        legal_structure_model::lever("repair-columns");
    let columns: Vec<(f64, f64)> = if by_column {
        pages
            .iter()
            .flat_map(|page| {
                let columns = legal_pdf_core::Columns::of(page);
                (0..page.lines.len())
                    .map(|at| columns.edges[columns.of[at]])
                    .collect::<Vec<_>>()
            })
            .collect()
    } else {
        Vec::new()
    };
    // (Before the pages are read in reading order, a slot is its line's place among the pages' lines.)
    let mut lines: Vec<Option<&mut Line>> = pages
        .iter_mut()
        .flat_map(|page| page.lines.iter_mut())
        .map(Some)
        .collect();
    let mut groups: BTreeMap<(u16, bool), Vec<Slot>> = BTreeMap::new();
    for &slot in &fences.order {
        groups
            .entry((
                fences.at(slot.index()).document,
                fences.at(slot.index()).translation,
            ))
            .or_default()
            .push(slot);
    }
    let mut repaired = Vec::new();
    for group in groups.values() {
        // A line in a column set beside the group's first on its page (its left edge at or past the first's right: a
        // scan of two facing pages, a two-column print): how far right of the first that column starts, by the line's
        // id. The group's count runs down one column and on into the next.
        let mut beside: HashMap<String, f64> = HashMap::new();
        if by_column {
            let mut first: HashMap<usize, (f64, f64)> = HashMap::new();
            for &slot in group {
                if let Some(line) = lines[slot.index()].as_deref() {
                    let column = columns[slot.index()];
                    first
                        .entry(line.page_index)
                        .and_modify(|held| {
                            if column.0 < held.0 {
                                *held = column;
                            }
                        })
                        .or_insert(column);
                }
            }
            for &slot in group {
                if let Some(line) = lines[slot.index()].as_deref() {
                    let (column, first) = (columns[slot.index()], first[&line.page_index]);
                    if column.0 >= first.1 {
                        beside.insert(line.id.clone(), column.0 - first.0);
                    }
                }
            }
        }
        let mut view = Vec::with_capacity(group.len());
        let mut before = Vec::with_capacity(group.len());
        for &slot in group {
            let Some(line) = lines[slot.index()].take() else {
                continue;
            };
            let at = slot.index();
            // (The repair asks only whether a line is the page's head or foot, or a note's.)
            let zone = match fences.role[at] {
                role @ (Role::LineLabel | Role::Furniture | Role::PageLabel) => {
                    regions::zone_of(role, fences.edge[at], NoteMode::Footnote)
                }
                _ if fences.band[at] => LineZone::Note(Some(NoteMode::Footnote)),
                _ => LineZone::Body,
            };
            before.push((slot, line.text.clone()));
            view.push((line, zone));
        }
        legal_structure::numbering_repair::repair_numbering(&mut view, &beside);
        for ((line, _), (slot, text)) in view.into_iter().zip(before) {
            if line.text != text {
                // (A rewritten line's type size is read again with its text.)
                line.size = legal_pdf_core::line_font_size(line);
                repaired.push((slot, text));
            }
        }
    }
    repaired.sort_unstable_by_key(|(slot, _)| *slot);
    repaired
}
/// A reading context's lines as one text, line under line, a marker standing alone on its line
/// run on with the line beside it that carries its text, and a citation wrapped from one line to the
/// next run on as one ("[1997]" over "1 S.C.R. 748": the volume opening the line is no marker): each
/// line's slot and where it begins and ends in the text, in UTF-16 code units.
struct View {
    text: String,
    lines: Vec<(Slot, usize, usize)>,
}

impl View {
    fn of(pages: &[Page], slots: &Slots, evidence: &Evidence, lines: &[Slot]) -> Self {
        let mut text = String::new();
        let mut placed = Vec::with_capacity(lines.len());
        let mut cursor = 0;
        let wrapped = |before: Slot, slot: Slot| {
            let end = slots.line(pages, before).text.trim_end().len() as u32;
            evidence
                .citations_of(before)
                .iter()
                .any(|cited| !cited.protected && cited.range.1 >= end)
                && evidence
                    .citations_of(slot)
                    .iter()
                    .any(|cited| !cited.protected && cited.range.0 == 0)
        };
        for (at, &slot) in lines.iter().enumerate() {
            if at > 0 {
                let before = lines[at - 1];
                text.push(
                    if evidence.beside[before.index()] == Some(slot) || wrapped(before, slot) {
                        ' '
                    } else {
                        '\n'
                    },
                );
                cursor += 1;
            }
            let line = &slots.line(pages, slot).text;
            text.push_str(line);
            let length = legal_structure_model::utf16_len(line);
            placed.push((slot, cursor, cursor + length));
            cursor += length;
        }
        Self {
            text,
            lines: placed,
        }
    }

    /// The place of the line holding the UTF-16 offset `at` (the line before, where `at` falls
    /// between lines), by its index among the view's lines.
    fn line_at(&self, at: usize) -> usize {
        self.lines
            .partition_point(|&(_, start, _)| start <= at)
            .saturating_sub(1)
    }

    /// Where the UTF-16 offset `at` falls, as a place in a line's text.
    fn place(&self, at: usize) -> Place {
        let (slot, start, end) = self.lines[self.line_at(at)];
        Place {
            slot,
            offset: (at.clamp(start, end) - start) as u32,
        }
    }

    /// The indexes of the view's lines a range overlaps; its first line where it is empty.
    fn overlapping(&self, start: usize, end: usize) -> std::ops::Range<usize> {
        let first = self.line_at(start);
        let last = self
            .lines
            .partition_point(|&(_, line_start, _)| line_start < end)
            .max(first + 1);
        first..last
    }
}

/// The byte in a line's text at its UTF-16 offset `offset`, the text's end past it.
fn byte_at(text: &str, offset: u32) -> usize {
    let mut units = 0;
    text.char_indices()
        .find(|&(_, c)| {
            let past = units >= offset as usize;
            units += c.len_utf16();
            past
        })
        .map_or(text.len(), |(byte, _)| byte)
}

/// The text from a place to its line's end.
fn text_from<'p>(pages: &'p [Page], slots: &Slots, place: Place) -> &'p str {
    let text = &slots.line(pages, place.slot).text;
    &text[byte_at(text, place.offset)..]
}

/// The byte range of a line's UTF-16 code units `start..end`.
fn bytes_of(text: &str, start: u32, end: u32) -> (u32, u32) {
    (byte_at(text, start) as u32, byte_at(text, end) as u32)
}

/// The family a marker's label counts in and the value it counts, read on its last word
/// (`numbering_marker`): "PART IV" counts 4.
fn label_reading(label: &str) -> Option<(String, u32)> {
    let reading =
        legal_structure_model::layout::numbering::numbering_marker(label.split_whitespace().next_back()?, None)?;
    Some((reading.family, reading.ordinal))
}

/// The value a marker's label counts: its reading's (`label_reading`), else its first number
/// ("63(1)" counts 63); 0 for none.
fn label_value(label: &str) -> u32 {
    label_reading(label)
        .map(|(_, value)| value)
        .unwrap_or_else(|| {
            let digits: String = label
                .chars()
                .skip_while(|c| !c.is_ascii_digit())
                .take_while(char::is_ascii_digit)
                .collect();
            digits.parse().unwrap_or(0)
        })
}

/// The runs as they read before any is resolved, so that the engine and the graph read one extent
/// for each marker: in the view, a list's item, and a provision's that numbers no top-level section
/// or division, ends where the numbered paragraphs resume (each marker of a rooted, consecutive
/// count of numbers outside the note band and not set smaller than the body's type, as a page's
/// notes under no rule are), and a list parts where a numbered paragraph opens between two of its
/// items.
fn part_at_paragraphs(
    runs: Vec<StructureCandidateRun>,
    view: &View,
    evidence: &Evidence,
    fences: &Fences,
) -> Vec<StructureCandidateRun> {
    let body_type = |at: usize| {
        fences.body.is_none_or(|(_, body)| {
            evidence.half_points[at] == 0 || evidence.half_points[at] >= body
        })
    };
    let mut starts: Vec<usize> = runs
        .iter()
        .filter(|run| run.grammar == CandidateGrammar::Numeric && run.rooted && run.consecutive)
        .flat_map(|run| &run.markers)
        .filter(|candidate| {
            let at = view.place(candidate.marker_range.start).slot.index();
            !fences.band[at] && body_type(at)
        })
        .map(|candidate| candidate.range.start)
        .collect();
    starts.sort_unstable();
    starts.dedup();
    if starts.is_empty() {
        return runs;
    }
    let next_start = |after: usize| {
        starts
            .get(starts.partition_point(|start| *start <= after))
            .copied()
    };
    let mut parted = Vec::with_capacity(runs.len());
    for mut run in runs {
        if run.grammar != CandidateGrammar::Numeric {
            let hierarchy = run.grammar == CandidateGrammar::Hierarchy;
            for candidate in &mut run.markers {
                let label = candidate.label.trim_start_matches('(').to_lowercase();
                let provision = hierarchy
                    && candidate.parent_candidate_id.is_none()
                    && (label.starts_with(|c: char| c.is_ascii_digit())
                        || [
                            "part", "section", "article", "division", "chapter", "schedule",
                        ]
                        .iter()
                        .any(|word| label.starts_with(word)));
                if !provision {
                    if let Some(end) =
                        next_start(candidate.range.start).filter(|end| *end < candidate.range.end)
                    {
                        candidate.range.end = end.max(candidate.content_start);
                    }
                }
            }
        }
        if run.grammar != CandidateGrammar::Enumerator {
            parted.push(run);
            continue;
        }
        let mut parts: Vec<StructureCandidateRun> = Vec::new();
        for candidate in std::mem::take(&mut run.markers) {
            match parts.last_mut() {
                Some(part)
                    if next_start(part.markers.last().expect("a part has an item").range.start)
                        .is_none_or(|start| start >= candidate.range.start) =>
                {
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
            part.id.clone_from(&run.id);
        }
        parted.extend(parts);
    }
    parted
}

/// What each line of a reading context shows of a marker on it: the context's view, the pages'
/// measures, and the lines a marker of the view opens (`marked`, by slot).
struct Lines<'a> {
    pages: &'a [Page],
    slots: &'a Slots,
    evidence: &'a Evidence,
    fences: &'a Fences,
    geometry: &'a Geometry,
    view: &'a View,
    marked: &'a HashSet<Slot>,
}

/// A marker read on a line, and what the line shows of it.
struct Reading {
    beside: Option<Slot>,
    text: Option<Slot>,
    observations: Vec<CandidateObservationV2>,
    carries_prose: bool,
    title_shaped: bool,
    worded: bool,
}

impl Lines<'_> {
    /// The line carrying a marker's text where the marker stands alone on its own: the line beside
    /// it on its row, or the line recognition read its text on, in its reading context.
    fn beside(&self, slot: Slot) -> Option<Slot> {
        self.evidence.beside[slot.index()].or_else(|| {
            let line = self.slots.line(self.pages, slot);
            let body = line
                .ocr
                .as_ref()?
                .marker
                .as_ref()?
                .body_line_id
                .as_deref()?;
            self.slots
                .of_id(body)
                .filter(|body| self.fences.ctx[body.index()] == self.fences.ctx[slot.index()])
        })
    }

    /// Whether a line fills its column's measure (85% of it or more).
    fn fills(&self, slot: Slot) -> bool {
        let page = self.slots.page(slot);
        let (_, columns) = self.geometry.page(self.pages, page);
        columns
            .place(
                self.slots.line(self.pages, slot),
                (slot.0 - self.slots.on(page).start) as usize,
            )
            .0
            >= 0.85
    }

    /// What the lines of a run's marker show of it: its own line's lexicon, typography and
    /// geometry, the text after it, and the run around it (`aligned`: a sibling of its level stands
    /// at its margin). Its text is prose where it runs on (with the line before or after it in the
    /// body's flow, the line after it going on with it, or its line filling its column), or where
    /// three words or more of it are no title's (over twelve words, ending a sentence or on a
    /// comma, semicolon or colon, or opening in lower case): "ARTICLE 8" over "TERMINATION OF
    /// FORBEARANCE PERIOD" heads.
    fn read(
        &self,
        run: &StructureCandidateRun,
        candidate: &StructureMarkerCandidate,
        aligned: bool,
    ) -> Reading {
        let (evidence, fences) = (self.evidence, self.fences);
        let start = self.view.place(candidate.marker_range.start);
        let slot = start.slot;
        let at = slot.index();
        let line = self.slots.line(self.pages, slot);
        let beside = self.beside(slot);
        // The marker's text: after it on its line, else on the line beside it, else on the next
        // where no other marker opens that; by its line's index among the view's lines.
        let index = self.view.line_at(candidate.marker_range.start);
        let content = self.view.place(candidate.content_start);
        let own = (content.slot == slot)
            .then(|| text_from(self.pages, self.slots, content))
            .filter(|tail| !tail.trim().is_empty());
        let next = (index + 1 < self.view.lines.len())
            .then_some(index + 1)
            .filter(|&next| !self.marked.contains(&self.view.lines[next].0));
        let text_index = if own.is_some() {
            Some(index)
        } else if let Some(beside) = beside {
            (index + 1 < self.view.lines.len() && self.view.lines[index + 1].0 == beside)
                .then_some(index + 1)
                .or_else(|| {
                    self.view
                        .lines
                        .iter()
                        .position(|&(line, _, _)| line == beside)
                })
                .or(next)
        } else {
            next
        };
        let text_line = text_index.map(|text| self.view.lines[text].0);
        let text = text_line.filter(|_| own.is_none());
        let tail = own
            .or_else(|| {
                text.map(|text| {
                    text_from(
                        self.pages,
                        self.slots,
                        Place {
                            slot: text,
                            offset: 0,
                        },
                    )
                })
            })
            .unwrap_or("");
        let letters = tail.chars().filter(|c| c.is_alphabetic()).count();
        // The text's line runs on: with the line before or after it in the body's flow, the line
        // after it going on with it, or its line filling its column.
        let runs_on = text_index.is_some_and(|text| {
            self.view.lines[text..]
                .iter()
                .take(2)
                .any(|&(line, _, _)| fences.continues[line.index()] == Some(Continuation::Flow))
                || self
                    .view
                    .lines
                    .get(text + 1)
                    .is_some_and(|&(after, _, _)| fences.continues[after.index()].is_some())
                || self.fills(self.view.lines[text].0)
        });
        // A marker standing alone with no text of its own reads as its own line does; a text opening
        // in lower case is a list's item or a sentence's, no title.
        let title_shaped = evidence.title_shaped[text_line.unwrap_or(slot).index()]
            && !tail.trim_start().starts_with(char::is_lowercase);
        let worded = letters >= 8 && tail.split_whitespace().take(3).count() == 3;
        let carries_prose = letters >= 8 && runs_on || worded && !title_shaped;
        // The marker is a citation's number, a sentence's word wrapped onto its line, or a bare
        // period marker inside prose ("v." of a case name). A number the margin prints is a paragraph's by its place on
        // the page, whatever the citation reading of the page's raw line order ran into it ("[2002] 4 R.C.S." over "67").
        let marker_end = self.view.place(candidate.marker_range.end);
        let (from, to) = bytes_of(
            &line.text,
            start.offset,
            if marker_end.slot == slot {
                marker_end.offset
            } else {
                u32::MAX
            },
        );
        let cited = fences.margin_number[at].is_none()
            && evidence
                .citations_of(slot)
                .iter()
                .any(|cited| cited.core.0 < to && from < cited.core.1);
        let continued = fences.continues[at].is_some();
        let opening =
            run.grammar == CandidateGrammar::Hierarchy && candidate.parent_candidate_id.is_none();
        let prose = run.grammar == CandidateGrammar::Enumerator
            && candidate.label.ends_with('.')
            && (continued
                || candidate.label == "v."
                    && legal_structure_lines::analysis::has_citation_signal(&line.text));
        // A list's item as its line reads it: a marker of a list's family (`UnitRole::of_family`:
        // never "[12]", "(2)", "3." or "A.") that is a bullet, or whose text, after it or on the
        // line carrying it, opens in lower case.
        let bullet = !candidate.label.chars().any(char::is_alphanumeric);
        let item = UnitRole::of_family(&candidate.label) == UnitRole::Item
            && (bullet || tail.trim_start().starts_with(char::is_lowercase));
        let observations = [
            (CandidateObservationV2::BodyProseFlow, carries_prose),
            (
                CandidateObservationV2::SectionHeading,
                title_shaped || matches!(evidence.tag[at], SourceTag::Heading(_)),
            ),
            (
                CandidateObservationV2::SourceListItem,
                evidence.tag[at] == SourceTag::ListItem,
            ),
            (
                CandidateObservationV2::ListItemLayout,
                aligned && letters >= 4 && tail.split_whitespace().count() >= 2,
            ),
            (
                CandidateObservationV2::CrossReference,
                cited || prose || opening && continued,
            ),
            // A number the margin prints beside the row its text opens on, counting on with the document's others, is
            // a report's paragraph number, never a section's (`Fences::read_margin_numbers`).
            (
                CandidateObservationV2::ComposedParagraph,
                fences.margin_number[at].is_some(),
            ),
            (
                CandidateObservationV2::TableOrForm,
                fences.cell[at] != 0
                    || matches!(evidence.tag[at], SourceTag::Table | SourceTag::Visual),
            ),
            // (TEMP lever "quoted-marker", remove before landing: a marker its line opens with a quotation mark, "“78.
            // Either the English or the French", is an amending provision's quoted text, whatever quotation the fences
            // read around it.)
            (
                CandidateObservationV2::Quotation,
                fences.at(slot.index()).quotation != 0
                    || legal_structure_model::lever("quoted-marker")
                        && line.text.get(..(from as usize).min(line.text.len())).is_some_and(|before| {
                            matches!(before.trim(), "“" | "\"" | "«" | "‘" | "„")
                        }),
            ),
            (CandidateObservationV2::ComposedListItem, item),
        ];
        let observations = observations
            .into_iter()
            .filter(|(_, holds)| *holds)
            .map(|(observation, _)| observation)
            .collect();
        Reading {
            beside,
            text,
            observations,
            carries_prose,
            title_shaped,
            worded,
        }
    }
}

/// Whether each list-context marker of a run has a sibling at its level, under its parent, at its
/// margin (eight thousandths of the page's width).
fn aligned_siblings(
    pages: &[Page],
    slots: &Slots,
    view: &View,
    run: &StructureCandidateRun,
) -> Vec<bool> {
    if run.grammar == CandidateGrammar::Numeric {
        return vec![false; run.markers.len()];
    }
    let lines: Vec<(&Page, &Line)> = run
        .markers
        .iter()
        .map(|candidate| {
            let slot = view.place(candidate.marker_range.start).slot;
            (&pages[slots.page(slot)], slots.line(pages, slot))
        })
        .collect();
    run.markers
        .iter()
        .enumerate()
        .map(|(at, candidate)| {
            let listed = candidate.parent_candidate_id.is_some()
                || (run.grammar == CandidateGrammar::Enumerator && run.rooted && run.consecutive);
            let (page, line) = lines[at];
            listed
                && run.markers.iter().zip(&lines).enumerate().any(
                    |(other, (sibling, (sibling_page, sibling_line)))| {
                        other != at
                            && sibling.level == candidate.level
                            && sibling.parent_candidate_id == candidate.parent_candidate_id
                            && (line.bbox[0] - sibling_line.bbox[0]).abs()
                                <= page.width.max(sibling_page.width).max(1.0) * 0.008
                    },
                )
        })
        .collect()
}

/// The runs of every reading context of the document's pages, measured (`evidence`) and fenced
/// and ordered (`fences`), each marker resolved to the unit it opens, once the markers are restored
/// (`repair_markers`: the measures read again and the paragraph numbers read with them). The pages' `geometry` is read
/// once their markers are restored.
pub fn read(
    pages: &[Page],
    slots: &Slots,
    evidence: &mut Evidence,
    fences: &Fences,
    geometry: &Geometry,
    repaired: Vec<(Slot, String)>,
) -> Result<Runs> {
    // A paragraph number in the margin stands beside the line of its row it numbers, as one printed line
    // (`Fences::read_margin_numbers`): the numbering reads it opening that line, not ending the one before.
    for (at, row) in fences.margin_number.iter().enumerate() {
        if let Some(row) = row {
            evidence.beside[at] = Some(*row);
        }
    }
    let evidence: &Evidence = evidence;
    // Each context's lines in reading order, its fenced lines left out.
    // (TEMP lever "cells-out", remove before landing.) An instrument's count reads no table's cells: a schedule's
    // rows count on as its sections do, and a cell opens no unit of the text.
    let cells_out = evidence.instrument
        && legal_structure_model::lever("cells-out");
    let inline_roots = evidence.instrument
        && legal_structure_model::lever("inline-roots");
    let mut by_context: Vec<Vec<Slot>> = vec![Vec::new(); fences.contexts.len()];
    for &slot in &fences.order {
        if !matches!(
            fences.role[slot.index()],
            Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents
        ) && !(cells_out && fences.role[slot.index()] == Role::Table)
        {
            by_context[usize::from(fences.ctx[slot.index()])].push(slot);
        }
    }
    let mut runs = Vec::new();
    for (ctx, lines) in by_context
        .iter()
        .enumerate()
        .filter(|(_, lines)| !lines.is_empty())
    {
        let view =
            legal_pdf_core::profile::tally("runs.view", || View::of(pages, slots, evidence, lines));
        let (found, by_count) = legal_pdf_core::profile::measure("candidates.detect", || {
            legal_structure::candidates::detect_counted_candidate_runs(&view.text, true, evidence.instrument)
        });
        // A provision ladder under one worded top-level marker that goes on with the sentence
        // before it (a witness's "it's at" / "Exhibit F.") is a cross-reference's, no outline: the
        // question numbers under it are the count's, not its subsections. A list's items go on
        // with their sentence too ("; and" / "b."), each its own top-level marker, and stand.
        let found: Vec<StructureCandidateRun> = found
            .into_iter()
            .filter(|run| {
                let mut roots = run
                    .markers
                    .iter()
                    .filter(|candidate| candidate.parent_candidate_id.is_none());
                let (Some(root), None) = (roots.next(), roots.next()) else {
                    return true;
                };
                // (TEMP lever "inline-roots": a count's section whose printed label ends on its own number is a margin
                // note's words joined before that number, "airniony°rder  f°r  16", no worded cross-reference.)
                let inline = inline_roots
                    && root.locator_kind.as_deref() == Some("section")
                    && !root.grammar_value.is_empty()
                    && root.grammar_value.bytes().all(|b| b.is_ascii_digit() || b == b'.')
                    && root.label.trim_end_matches(['.', ',', ' ']).ends_with(root.grammar_value.as_str())
                    // (Never a provision's word before it: "Section 11." going on with "pursuant to" is a reference.)
                    && !root
                        .label
                        .split_whitespace()
                        .any(|word| {
                            matches!(
                                word.rsplit(['\'', '\u{2019}']).next().unwrap_or(word).trim_end_matches(['.', ',']).to_lowercase().as_str(),
                                "section" | "sec" | "s" | "ss" | "sections" | "article" | "art" | "articles" | "rule" | "rules"
                                    | "paragraph" | "paragraphe" | "part" | "partie" | "schedule" | "annexe" | "form"
                                    | "formule" | "clause" | "subsection" | "sub-section" | "chapter" | "chapitre" | "c" | "ch"
                                    | "regulation" | "regulations" | "règlement" | "item" | "plan" | "no" | "n°" | "numéro" | "annex"
                                    | "appendix" | "exhibit" | "order" | "by-law"
                            )
                        });
                let worded = !inline
                    && root
                        .label
                        .chars()
                        .take_while(char::is_ascii_alphabetic)
                        .count()
                        >= 3;
                run.grammar != CandidateGrammar::Hierarchy
                    || !worded
                    || fences.continues[view.place(root.marker_range.start).slot.index()].is_none()
            })
            .collect();
        if found.is_empty() {
            continue;
        }
        let found = legal_pdf_core::profile::tally("runs.part", || {
            part_at_paragraphs(found, &view, evidence, fences)
        });
        let marked: HashSet<Slot> = found
            .iter()
            .flat_map(|run| &run.markers)
            .map(|candidate| view.place(candidate.marker_range.start).slot)
            .collect();
        let shown = Lines {
            pages,
            slots,
            evidence,
            fences,
            geometry,
            view: &view,
            marked: &marked,
        };
        let readings: Vec<Vec<Reading>> = legal_pdf_core::profile::tally("runs.readings", || {
            found
                .iter()
                .map(|run| {
                    let aligned = legal_pdf_core::profile::tally("runs.aligned", || {
                        aligned_siblings(pages, slots, &view, run)
                    });
                    run.markers
                        .iter()
                        .zip(aligned)
                        .map(|(candidate, aligned)| shown.read(run, candidate, aligned))
                        .collect()
                })
                .collect()
        });
        let candidate_evidence: Vec<CandidateEvidenceV2> = found
            .iter()
            .zip(&readings)
            .flat_map(|(run, readings)| run.markers.iter().zip(readings))
            .map(|(candidate, reading)| {
                let mut page_indexes: Vec<usize> = Vec::new();
                let mut line_id = |index: usize| {
                    let slot = view.lines[index].0;
                    let page = pages[slots.page(slot)].index;
                    if !page_indexes.contains(&page) {
                        page_indexes.push(page);
                    }
                    slots.line(pages, slot).id.clone()
                };
                let line_ids = view
                    .overlapping(candidate.range.start, candidate.range.end)
                    .map(&mut line_id)
                    .collect();
                CandidateEvidenceV2 {
                    candidate_id: candidate.id.clone(),
                    page_indexes,
                    line_ids,
                    observations: reading.observations.clone(),
                }
            })
            .collect();
        let roles = legal_pdf_core::profile::measure("runs.resolve", || {
            legal_structure::candidates::resolve_counted_candidates(
                &found,
                &candidate_evidence,
                evidence.instrument,
                by_count,
            )
        })
        .map_err(|error| Error::Message(error.to_string()))?;
        let role_of: HashMap<&str, &legal_structure_model::CandidateRoleV2> = roles
            .iter()
            .map(|role| (role.candidate_id.as_str(), role))
            .collect();
        // A document read in more than one context names each context's candidates apart.
        let prefix = if fences.contexts.len() > 1 {
            format!("context-{ctx:06}-")
        } else {
            String::new()
        };
        for (run, readings) in found.iter().zip(readings) {
            let members = run
                .markers
                .iter()
                .zip(readings)
                .enumerate()
                .map(|(at, (candidate, reading))| {
                    let role = role_of.get(candidate.id.as_str());
                    let marker = [
                        view.place(candidate.marker_range.start),
                        view.place(candidate.marker_range.end),
                    ];
                    Member {
                        id: format!("{prefix}{}", candidate.id),
                        label: candidate.label.clone(),
                        grammar_value: candidate.grammar_value.clone(),
                        locator_kind: candidate.locator_kind.clone(),
                        slot: marker[0].slot,
                        beside: reading.beside,
                        text: reading.text,
                        parent: candidate
                            .parent_candidate_id
                            .as_ref()
                            .and_then(|parent| {
                                run.markers[..at]
                                    .iter()
                                    .rposition(|other| other.id == *parent)
                            })
                            .map(|parent| parent as u32),
                        level: candidate.level.min(usize::from(u16::MAX)) as u16,
                        value: label_value(&candidate.label),
                        marker,
                        content: view.place(candidate.content_start),
                        end: view.place(candidate.range.end),
                        counted: role.is_some_and(|role| role.counted),
                        kind: role.and_then(|role| role.role),
                        rule: role.map_or(ResolutionRuleV2::InsufficientEvidence, |role| role.rule),
                        observations: reading.observations,
                        carries_prose: reading.carries_prose,
                        title_shaped: reading.title_shaped,
                        worded: reading.worded,
                    }
                })
                .collect();
            runs.push(Run {
                ctx: ctx as u16,
                id: format!("{prefix}{}", run.id),
                grammar: run.grammar,
                rooted: run.rooted,
                consecutive: run.consecutive,
                family: run
                    .markers
                    .first()
                    .and_then(|first| label_reading(&first.label))
                    .map_or_else(String::new, |(family, _)| family),
                members,
            });
        }
    }
    // The document's paragraphs are the count the reading took (`Fences::read_paragraphs`): a quoted decision's
    // numbers, a list's, the other language's issue list are none of it.
    let paragraph = Some(legal_structure_model::UnitRole::Paragraph);
    let counted = |member: &Member| member.kind == paragraph;
    let members: Vec<(usize, usize)> = runs
        .iter()
        .enumerate()
        .flat_map(|(r, run)| {
            run.members
                .iter()
                .enumerate()
                .filter(|(_, member)| counted(member))
                .map(move |(m, _)| (r, m))
        })
        .collect();
    // TEMP lever (remove before landing): "counted-sections" in LEGALPDF_LEVERS turns the rule below on, so one build
    // reads both ways.
    let sections = evidence.instrument
        && legal_structure_model::lever("counted-sections");
    for &(r, m) in &members {
        // An instrument's count of numbers is its sections' (a statute's, a by-law's or an agreement's "1.", "2."): a
        // number the reading counts opens a section there. A bracketed or parenthesized number is a paragraph's or a
        // subsection's in any text.
        let section = sections
            && runs[r]
                .family
                .split_once(':')
                .is_some_and(|(wrap, count)| matches!(wrap, "" | ".") && count.starts_with('1'));
        let member = &mut runs[r].members[m];
        let own = fences.paragraphs.count[member.slot.index()].is_some();
        (member.kind, member.rule) = if own && section {
            (
                Some(legal_structure_model::UnitRole::Section),
                ResolutionRuleV2::HierarchySection,
            )
        } else if own {
            (paragraph, ResolutionRuleV2::RootedNumericProse)
        } else {
            (
                Some(legal_structure_model::UnitRole::Item),
                ResolutionRuleV2::OffCount,
            )
        };
    }
    Ok(Runs {
        runs,
        before: repaired,
    })
}

/// A statute's spine: its rooted, consecutive provisions, counted and opening prose, each page's numbers. A running head
/// naming its page's sections is the page's furniture where the spine opens a section it names there or on the page
/// before (`confirm_section_heads`); the head itself bears out nothing. None where no running head names a section.
pub fn spine(pages: &[Page], slots: &Slots, fences: &Fences, runs: &Runs) -> Option<Vec<Vec<u32>>> {
    let runs = &runs.runs;
    if !fences.section_heads.is_empty() {
        let mut spine: Vec<Vec<u32>> = vec![Vec::new(); pages.len()];
        for member in runs
            .iter()
            .filter(|run| {
                run.grammar == CandidateGrammar::Hierarchy && run.rooted && run.consecutive
            })
            .flat_map(|run| &run.members)
            .filter(|member| {
                member.parent.is_none()
                    && member.counted
                    && member.carries_prose
                    && !fences
                        .section_heads
                        .iter()
                        .any(|(head, _, _)| *head == member.slot)
            })
        {
            spine[slots.page(member.slot)].push(member.value);
        }
        return Some(spine);
    }
    None
}
