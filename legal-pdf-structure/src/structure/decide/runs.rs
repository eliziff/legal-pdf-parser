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

use super::*;
use legal_pdf_core::model::Word;
use legal_structure_model::{
    CandidateEvidenceV2, CandidateGrammar, CandidateObservationV2, StructureCandidateRun, StructureMarkerCandidate,
    UnitRole,
};

/// A place in a line's text: the line's slot and the characters before the place.
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

/// The document's numbering runs, and the lines whose markers the repair rewrote, with their text and
/// words as rewritten.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Runs {
    pub(crate) runs: Vec<Run>,
    pub(crate) edits: Vec<(Slot, String, Vec<Word>)>,
}

/// A line recognition read, the only reading that misreads a marker: our own recognition, or a
/// scan's invisible recognition layer.
fn recognized(line: &Line) -> bool {
    line.source == "ocr" || line.spans.iter().any(|span| span.flags & legal_pdf_core::model::INVISIBLE != 0)
}

/// The markers recognition misread, restored from the runs they stand in, in each document and
/// language with its lines in reading order (`repair_label_dropouts`): each rewritten line's slot
/// and its text before. The repair reads a fenced running head or folio as the page's head or
/// foot, a line label as one, and a line under the note separator as a note's.
fn repair_markers(pages: &mut [Page], slots: &Slots, fences: &Fences) -> Vec<(Slot, String)> {
    if !pages.iter().flat_map(|page| &page.lines).any(recognized) {
        return Vec::new();
    }
    let mut lines: Vec<Option<Line>> = pages.iter_mut().flat_map(|page| std::mem::take(&mut page.lines)).map(Some).collect();
    let mut groups: BTreeMap<(u16, bool), Vec<Slot>> = BTreeMap::new();
    for &slot in &fences.order {
        groups.entry((fences.document[slot.index()], fences.translation[slot.index()])).or_default().push(slot);
    }
    let mut repaired = Vec::new();
    for group in groups.values() {
        let mut view: Vec<Page> = Vec::new();
        // Each line's slot and what the repair's reading changed on it: its region, note mode and
        // place in the reading order; and its text before.
        let mut held: Vec<(Slot, String, String, usize, String)> = Vec::with_capacity(group.len());
        for (position, &slot) in group.iter().enumerate() {
            let page = &pages[slots.page(slot)];
            if view.last().is_none_or(|last| last.index != page.index) {
                view.push(Page {
                    id: page.id.clone(),
                    index: page.index,
                    number: page.number,
                    width: page.width,
                    height: page.height,
                    lines: Vec::new(),
                    regions: Vec::new(),
                    source: page.source.clone(),
                    text_quality: page.text_quality,
                    printed_label: None,
                    printed_label_source: None,
                    printed_label_line_id: None,
                });
            }
            let Some(mut line) = lines[slot.index()].take() else {
                continue;
            };
            let at = slot.index();
            let (region, mode) = match fences.role[at] {
                Role::LineLabel => ("line_label", ""),
                Role::Furniture | Role::PageLabel if fences.edge[at] == Some(Edge::Top) => ("header", ""),
                Role::Furniture | Role::PageLabel => ("footer", ""),
                _ if fences.band[at] => ("footnote", "footnote"),
                _ => ("body", ""),
            };
            held.push((
                slot,
                std::mem::replace(&mut line.region_type, region.to_owned()),
                std::mem::replace(&mut line.note_region_mode, mode.to_owned()),
                std::mem::replace(&mut line.reading_order, position),
                line.text.clone(),
            ));
            view.last_mut().expect("a page for the line").lines.push(line);
        }
        layout_analysis().repair_label_dropouts(&mut view);
        for (mut line, (slot, region, mode, order, text)) in view.into_iter().flat_map(|page| page.lines).zip(held) {
            line.region_type = region;
            line.note_region_mode = mode;
            line.reading_order = order;
            if line.text != text {
                repaired.push((slot, text));
            }
            lines[slot.index()] = Some(line);
        }
    }
    for (page_slot, page) in pages.iter_mut().enumerate() {
        page.lines = slots.on(page_slot).filter_map(|slot| lines[slot as usize].take()).collect();
    }
    repaired.sort_unstable_by_key(|(slot, _)| *slot);
    repaired
}

/// A reading context's lines as one text, line under line, a marker standing alone on its line
/// run on with the line beside it that carries its text: each line's slot and where it begins and
/// ends in the text, in scalars.
struct View {
    text: String,
    lines: Vec<(Slot, usize, usize)>,
}

impl View {
    fn of(pages: &[Page], slots: &Slots, evidence: &Evidence, lines: &[Slot]) -> Self {
        let mut text = String::new();
        let mut placed = Vec::with_capacity(lines.len());
        let mut cursor = 0;
        for (at, &slot) in lines.iter().enumerate() {
            if at > 0 {
                text.push(if evidence.beside[lines[at - 1].index()] == Some(slot) { ' ' } else { '\n' });
                cursor += 1;
            }
            let line = &slots.line(pages, slot).text;
            text.push_str(line);
            let length = line.chars().count();
            placed.push((slot, cursor, cursor + length));
            cursor += length;
        }
        Self { text, lines: placed }
    }

    /// The place of the line holding the scalar `at` (the line before, where `at` falls between
    /// lines), by its index among the view's lines.
    fn line_at(&self, at: usize) -> usize {
        self.lines.partition_point(|&(_, start, _)| start <= at).saturating_sub(1)
    }

    /// Where the scalar `at` falls, as a place in a line's text.
    fn place(&self, at: usize) -> Place {
        let (slot, start, end) = self.lines[self.line_at(at)];
        Place { slot, offset: (at.clamp(start, end) - start) as u32 }
    }

    /// The indexes of the view's lines a range overlaps; its first line where it is empty.
    fn overlapping(&self, start: usize, end: usize) -> std::ops::Range<usize> {
        let first = self.line_at(start);
        let last = self.lines.partition_point(|&(_, line_start, _)| line_start < end).max(first + 1);
        first..last
    }
}

/// The text from a place to its line's end.
fn text_from<'p>(pages: &'p [Page], slots: &Slots, place: Place) -> &'p str {
    let text = &slots.line(pages, place.slot).text;
    let byte = text.char_indices().nth(place.offset as usize).map_or(text.len(), |(byte, _)| byte);
    &text[byte..]
}

/// The byte range of a line's characters `start..end`.
fn bytes_of(text: &str, start: u32, end: u32) -> (u32, u32) {
    let byte = |offset: u32| text.char_indices().nth(offset as usize).map_or(text.len(), |(byte, _)| byte) as u32;
    (byte(start), byte(end))
}

/// The family a marker's label counts in and the value it counts, read on its last word
/// (`numbering_marker`): "PART IV" counts 4.
fn label_reading(label: &str) -> Option<(String, u32)> {
    let reading = layout_analysis().numbering_marker(label.split_whitespace().next_back()?, None)?;
    Some((reading.family, reading.ordinal))
}

/// The value a marker's label counts: its reading's (`label_reading`), else its first number
/// ("63(1)" counts 63); 0 for none.
fn label_value(label: &str) -> u32 {
    label_reading(label).map(|(_, value)| value).unwrap_or_else(|| {
        let digits: String = label.chars().skip_while(|c| !c.is_ascii_digit()).take_while(char::is_ascii_digit).collect();
        digits.parse().unwrap_or(0)
    })
}

/// The runs as they read before any is resolved, so that the engine and the graph read one extent
/// for each marker: in the view, a list's item, and a provision's that numbers no top-level section
/// or division, ends where the numbered paragraphs resume (each marker of a rooted, consecutive
/// count of numbers outside the note band and not set smaller than the body's type, as a page's
/// notes under no rule are), and a list parts where a numbered paragraph opens between two of its
/// items.
fn part_at_paragraphs(runs: Vec<StructureCandidateRun>, view: &View, evidence: &Evidence, fences: &Fences) -> Vec<StructureCandidateRun> {
    let body_type = |at: usize| fences.body.is_none_or(|(_, body)| evidence.half_points[at] == 0 || evidence.half_points[at] >= body);
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
    let next_start = |after: usize| starts.get(starts.partition_point(|start| *start <= after)).copied();
    let mut parted = Vec::with_capacity(runs.len());
    for mut run in runs {
        if run.grammar != CandidateGrammar::Numeric {
            let hierarchy = run.grammar == CandidateGrammar::Hierarchy;
            for candidate in &mut run.markers {
                let label = candidate.label.trim_start_matches('(').to_lowercase();
                let provision = hierarchy
                    && candidate.parent_candidate_id.is_none()
                    && (label.starts_with(|c: char| c.is_ascii_digit())
                        || ["part", "section", "article", "division", "chapter", "schedule"].iter().any(|word| label.starts_with(word)));
                if !provision {
                    if let Some(end) = next_start(candidate.range.start).filter(|end| *end < candidate.range.end) {
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
    geometry: &'a Geometry<'a>,
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
            let body = line.ocr.as_ref()?.marker.as_ref()?.body_line_id.as_deref()?;
            self.slots.of_id(body).filter(|body| self.fences.ctx[body.index()] == self.fences.ctx[slot.index()])
        })
    }

    /// Whether a line fills its column's measure (85% of it or more).
    fn fills(&self, slot: Slot) -> bool {
        let page = self.slots.page(slot);
        let (_, columns) = self.geometry.page(page);
        columns.place(self.slots.line(self.pages, slot), (slot.0 - self.slots.on(page).start) as usize).0 >= 0.85
    }

    /// What the lines of a run's marker show of it: its own line's lexicon, typography and
    /// geometry, the text after it, and the run around it (`aligned`: a sibling of its level stands
    /// at its margin). Its text is prose where it runs on (with the line before or after it in the
    /// body's flow, the line after it going on with it, or its line filling its column), or where
    /// three words or more of it are no title's (over twelve words, ending a sentence or on a
    /// comma, semicolon or colon, or opening in lower case): "ARTICLE 8" over "TERMINATION OF
    /// FORBEARANCE PERIOD" heads.
    fn read(&self, run: &StructureCandidateRun, candidate: &StructureMarkerCandidate, aligned: bool) -> Reading {
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
        let own = (content.slot == slot).then(|| text_from(self.pages, self.slots, content)).filter(|tail| !tail.trim().is_empty());
        let next = (index + 1 < self.view.lines.len())
            .then_some(index + 1)
            .filter(|&next| !self.marked.contains(&self.view.lines[next].0));
        let text_index = if own.is_some() {
            Some(index)
        } else if let Some(beside) = beside {
            (index + 1 < self.view.lines.len() && self.view.lines[index + 1].0 == beside)
                .then_some(index + 1)
                .or_else(|| self.view.lines.iter().position(|&(line, _, _)| line == beside))
                .or(next)
        } else {
            next
        };
        let text_line = text_index.map(|text| self.view.lines[text].0);
        let text = text_line.filter(|_| own.is_none());
        let tail = own.or_else(|| text.map(|text| text_from(self.pages, self.slots, Place { slot: text, offset: 0 }))).unwrap_or("");
        let letters = tail.chars().filter(|c| c.is_alphabetic()).count();
        // The text's line runs on: with the line before or after it in the body's flow, the line
        // after it going on with it, or its line filling its column.
        let runs_on = text_index.is_some_and(|text| {
            self.view.lines[text..].iter().take(2).any(|&(line, _, _)| fences.continues[line.index()] == Some(Continuation::Flow))
                || self.view.lines.get(text + 1).is_some_and(|&(after, _, _)| fences.continues[after.index()].is_some())
                || self.fills(self.view.lines[text].0)
        });
        // A marker standing alone with no text of its own reads as its own line does; a text opening
        // in lower case is a list's item or a sentence's, no title.
        let title_shaped = evidence.title_shaped[text_line.unwrap_or(slot).index()] && !tail.trim_start().starts_with(char::is_lowercase);
        let worded = letters >= 8 && tail.split_whitespace().take(3).count() == 3;
        let carries_prose = letters >= 8 && runs_on || worded && !title_shaped;
        let mut observations = Vec::new();
        let mut observe = |observation: CandidateObservationV2| {
            if !observations.contains(&observation) {
                observations.push(observation);
            }
        };
        if carries_prose {
            observe(CandidateObservationV2::BodyProseFlow);
        }
        if title_shaped || matches!(evidence.tag[at], SourceTag::Heading(_)) {
            observe(CandidateObservationV2::SectionHeading);
        }
        if evidence.tag[at] == SourceTag::ListItem {
            observe(CandidateObservationV2::SourceListItem);
        }
        if aligned && letters >= 4 && tail.split_whitespace().count() >= 2 {
            observe(CandidateObservationV2::ListItemLayout);
        }
        // The marker is a citation's number, a sentence's word wrapped onto its line, or a bare
        // period marker inside prose ("v." of a case name).
        let marker_end = self.view.place(candidate.marker_range.end);
        let (from, to) = bytes_of(&line.text, start.offset, if marker_end.slot == slot { marker_end.offset } else { u32::MAX });
        let cited = evidence.citations_of(slot).iter().any(|cited| cited.core.0 < to && from < cited.core.1);
        let continued = fences.continues[at].is_some();
        let opening = run.grammar == CandidateGrammar::Hierarchy && candidate.parent_candidate_id.is_none();
        let prose = run.grammar == CandidateGrammar::Enumerator
            && candidate.label.ends_with('.')
            && (continued || candidate.label == "v." && layout_analysis().has_citation_signal(&line.text));
        if cited || prose || opening && continued {
            observe(CandidateObservationV2::CrossReference);
        }
        if fences.cell[at] != 0 || matches!(evidence.tag[at], SourceTag::Table | SourceTag::Visual) {
            observe(CandidateObservationV2::TableOrForm);
        }
        if fences.context(slot).quotation != 0 {
            observe(CandidateObservationV2::Quotation);
        }
        // A list's item as its line reads it: a marker of a list's family (`UnitRole::of_family`:
        // never "[12]", "(2)", "3." or "A.") that is a bullet, or whose text, after it or on the
        // line carrying it, opens in lower case.
        let bullet = !candidate.label.chars().any(char::is_alphanumeric);
        if UnitRole::of_family(&candidate.label) == UnitRole::Item && (bullet || tail.trim_start().starts_with(char::is_lowercase)) {
            observe(CandidateObservationV2::ComposedListItem);
        }
        Reading { beside, text, observations, carries_prose, title_shaped, worded }
    }
}

/// Whether each list-context marker of a run has a sibling at its level, under its parent, at its
/// margin (eight thousandths of the page's width).
fn aligned_siblings(pages: &[Page], slots: &Slots, view: &View, run: &StructureCandidateRun) -> Vec<bool> {
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
                && run.markers.iter().zip(&lines).enumerate().any(|(other, (sibling, (sibling_page, sibling_line)))| {
                    other != at
                        && sibling.level == candidate.level
                        && sibling.parent_candidate_id == candidate.parent_candidate_id
                        && (line.bbox[0] - sibling_line.bbox[0]).abs() <= page.width.max(sibling_page.width).max(1.0) * 0.008
                })
        })
        .collect()
}

impl Runs {
    /// The runs of every reading context of the document's pages, measured (`evidence`) and fenced
    /// and ordered (`fences`), each marker resolved to the unit it opens; and, once the statute's
    /// spine is read, its running heads naming its sections fenced (`confirm_section_heads`).
    pub(crate) fn read(pages: &mut [Page], slots: &Slots, evidence: &mut Evidence, fences: &mut Fences) -> Result<Self> {
        let repaired = legal_pdf_support::profile::measure("runs.repair", || repair_markers(pages, slots, fences));
        let pages: &[Page] = pages;
        evidence.reread(pages, slots, &repaired);
        let edits = repaired
            .iter()
            .map(|(slot, _)| {
                let line = slots.line(pages, *slot);
                (*slot, line.text.clone(), line.words.clone())
            })
            .collect();
        let evidence: &Evidence = evidence;
        let geometry = Geometry::of(pages);
        // Each context's lines in reading order, its fenced lines left out.
        let mut by_context: Vec<Vec<Slot>> = vec![Vec::new(); fences.contexts.len()];
        for &slot in &fences.order {
            if !matches!(fences.role[slot.index()], Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents) {
                by_context[usize::from(fences.ctx[slot.index()])].push(slot);
            }
        }
        let mut runs = Vec::new();
        for (ctx, lines) in by_context.iter().enumerate().filter(|(_, lines)| !lines.is_empty()) {
            let view = View::of(pages, slots, evidence, lines);
            let found = legal_pdf_support::profile::measure("candidates.detect", || {
                legal_structure_model::structure_analysis().detect_structure_candidate_runs(&view.text, true)
            });
            // A provision ladder whose every top-level marker goes on with the sentence before it
            // (a witness's "it's at" / "Exhibit F.") is a cross-reference's, no outline: the question
            // numbers under it are the count's, not its subsections.
            let found: Vec<StructureCandidateRun> = found
                .into_iter()
                .filter(|run| {
                    let roots: Vec<Slot> = run
                        .markers
                        .iter()
                        .filter(|candidate| candidate.parent_candidate_id.is_none())
                        .map(|candidate| view.place(candidate.marker_range.start).slot)
                        .collect();
                    run.grammar != CandidateGrammar::Hierarchy || roots.is_empty() || roots.iter().any(|root| fences.continues[root.index()].is_none())
                })
                .collect();
            if found.is_empty() {
                continue;
            }
            let found = part_at_paragraphs(found, &view, evidence, fences);
            let marked: HashSet<Slot> =
                found.iter().flat_map(|run| &run.markers).map(|candidate| view.place(candidate.marker_range.start).slot).collect();
            let shown = Lines { pages, slots, evidence, fences, geometry: &geometry, view: &view, marked: &marked };
            let readings: Vec<Vec<Reading>> = found
                .iter()
                .map(|run| {
                    let aligned = aligned_siblings(pages, slots, &view, run);
                    run.markers.iter().zip(aligned).map(|(candidate, aligned)| shown.read(run, candidate, aligned)).collect()
                })
                .collect();
            let candidate_evidence: Vec<CandidateEvidenceV2> = found
                .iter()
                .zip(&readings)
                .flat_map(|(run, readings)| run.markers.iter().zip(readings))
                .map(|(candidate, reading)| {
                    let mut page_indexes: Vec<usize> = Vec::new();
                    let line_ids = view
                        .overlapping(candidate.range.start, candidate.range.end)
                        .map(|index| {
                            let slot = view.lines[index].0;
                            let page = pages[slots.page(slot)].index;
                            if !page_indexes.contains(&page) {
                                page_indexes.push(page);
                            }
                            slots.line(pages, slot).id.clone()
                        })
                        .collect();
                    CandidateEvidenceV2 {
                        candidate_id: candidate.id.clone(),
                        page_indexes,
                        line_ids,
                        observations: reading.observations.clone(),
                    }
                })
                .collect();
            let roles = legal_pdf_support::profile::measure("runs.resolve", || {
                legal_structure_model::structure_analysis().resolve_candidates(&found, &candidate_evidence)
            })
            .map_err(|error| Error::Message(error.to_string()))?;
            let role_of: HashMap<&str, &legal_structure_model::CandidateRoleV2> =
                roles.iter().map(|role| (role.candidate_id.as_str(), role)).collect();
            // A document read in more than one context names each context's candidates apart.
            let prefix = if fences.contexts.len() > 1 { format!("context-{ctx:06}-") } else { String::new() };
            for (run, readings) in found.iter().zip(readings) {
                let members = run
                    .markers
                    .iter()
                    .zip(readings)
                    .enumerate()
                    .map(|(at, (candidate, reading))| {
                        let role = role_of.get(candidate.id.as_str());
                        let marker = [view.place(candidate.marker_range.start), view.place(candidate.marker_range.end)];
                        Member {
                            id: format!("{prefix}{}", candidate.id),
                            label: candidate.label.clone(),
                            grammar_value: candidate.grammar_value.clone(),
                            slot: marker[0].slot,
                            beside: reading.beside,
                            text: reading.text,
                            parent: candidate
                                .parent_candidate_id
                                .as_ref()
                                .and_then(|parent| run.markers[..at].iter().rposition(|other| other.id == *parent))
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
                    family: run.markers.first().and_then(|first| label_reading(&first.label)).map_or_else(String::new, |(family, _)| family),
                    members,
                });
            }
        }
        // A statute's running head naming its page's sections is the page's furniture where the
        // spine (its rooted, consecutive provisions, counted and opening prose) opens a section it
        // names there or on the page before; the head itself bears out nothing.
        if !fences.section_heads.is_empty() {
            let mut spine: Vec<Vec<u32>> = vec![Vec::new(); pages.len()];
            for member in runs
                .iter()
                .filter(|run| run.grammar == CandidateGrammar::Hierarchy && run.rooted && run.consecutive)
                .flat_map(|run| &run.members)
                .filter(|member| {
                    member.parent.is_none()
                        && member.counted
                        && member.carries_prose
                        && !fences.section_heads.iter().any(|(head, _, _)| *head == member.slot)
                })
            {
                spine[slots.page(member.slot)].push(member.value);
            }
            fences.confirm_section_heads(pages, slots, evidence, |page, number| spine[page].contains(&number));
        }
        Ok(Self { runs, edits })
    }

    /// Every member of every run.
    pub(crate) fn members(&self) -> impl Iterator<Item = &Member> {
        self.runs.iter().flat_map(|run| &run.members)
    }
}
