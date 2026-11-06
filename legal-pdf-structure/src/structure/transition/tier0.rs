//! Tier 0 as the machine's reader: the document's lines read by the rules (`decide`), every reading a rule gave a line
//! a candidate in one decision on that line, the scorer compiled in (no fitted file, no environment), and the moves
//! the decided lines make written by `emit`. It is chosen by `Reader::Machine`, in the request; `Reader::Stages` stays
//! the default until this reader matches it by mode.
//!
//! What it reads, in order:
//! - the notes and the rules (`Notes::read`, `Decision::decide`): each line's standing reading, and the readings of
//!   strong rules that came second on it (`Decision::refused`);
//! - one decision on each line two rules read differently (`settle`), in reading order against what is open
//!   (`machine::State`): whether a marker counts on, whether a note's number is the next or answers a mark the body set,
//!   what the rule itself is worth. Paragraph numbers stand outside it: a line the judgment reader counts or holds
//!   (`Fences::count`, `Fences::held`) keeps its reading, and `read_judgment_paragraphs` writes them as t0 does;
//! - the decided lines written (`write_lines`), their regions built, their notes paired (`note_pairing`), the pairer's
//!   pairs fixed input to the moves, as paragraph numbers are;
//! - the moves (`moves_of`): each of the document's reading regions a unit, each note a unit, each line set apart a
//!   unit of its own.

use super::emit::Settled;
use super::*;
use legal_structure_model::UnitRole;

/// What a reading is worth over the others on its line, before the state speaks: the reading standing keeps a tie.
const STANDING: f32 = 0.5;
/// A line the document's outline names, or a notes heading: the reading's own word.
const OWN_WORD: f32 = 0.75;
/// A marker going on from the number open in its family.
const COUNTS_ON: f32 = 1.0;
/// A marker opening a count at one.
const OPENS_COUNT: f32 = 0.25;
/// A marker neither.
const BREAKS_COUNT: f32 = -0.5;
/// A note's number a mark of the body's has set and no note has answered.
const ANSWERS: f32 = 1.5;
/// A note's number next after the last note's.
const NEXT_NOTE: f32 = 1.0;
/// A note's number neither.
const STRAY_NOTE: f32 = -0.5;

/// A move making nothing but its region and whether it opens.
fn plain(region: u8, opens: bool) -> Move {
    Move { region, opens, reading: None, note: None, level: 0, depth: 0, resume: false, resumes: None, rank: None, starts: false, answers: [None; 4] }
}

/// The region a role is, where the line opens the unit `kind`.
fn region_of(role: Role, kind: Option<UnitRole>) -> u8 {
    match role {
        Role::LineLabel => LINE_LABEL,
        Role::PageLabel => PAGE_LABEL,
        Role::Furniture => FURNITURE,
        Role::Contents => CONTENTS,
        Role::Table => TABLE,
        Role::Note => NOTE,
        Role::Heading => HEADING,
        Role::Title => TITLE,
        Role::MarginalNote => MARGINAL_NOTE,
        Role::ListItem => LIST_ITEM,
        Role::Quote => QUOTE,
        Role::Open | Role::Body => match kind {
            Some(UnitRole::Section) => PROVISION,
            Some(UnitRole::Item) => LIST_ITEM,
            _ => TEXT,
        },
    }
}

/// Each line as the state reads it, in reading order. No fitted reader runs here, so nothing asks for its stylesheet or
/// its words' tables.
fn line_views<'a>(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, table: &'a LineTable) -> Vec<LineView<'a>> {
    let left = LINE_FEATURES.iter().position(|name| *name == "x0").unwrap_or(0);
    (0..table.order.len())
        .map(|k| {
            let slot = table.order[k];
            let at = slot.index();
            LineView {
                translation: fences.translation[at],
                may_start: false,
                start_kind: 0,
                page_place: 0,
                readings: &table.readings[k],
                label: evidence.label[at].map(|label| label.value),
                line_number: slots.line(pages, slot).text.trim().parse::<u32>().ok().filter(|_| evidence.line_number[at]),
                look: table.looks[k],
                references: &table.references[k],
                place: k as u32,
                grammar: None,
                heading_words: [0.0; 2],
                operative: evidence.operative,
                look_heading: 0.0,
                look_heading_lines: 0.0,
                template: TemplateView::default(),
                anchor: 0,
                listed: false,
                regime: [0.0; 4],
                title_key: String::new(),
                contents_title: None,
                left: table.row(k)[left],
                page: slots.page(slot) as u32,
                margin: table.margins.get(&k).and_then(|&margin| Some((margin as u32, table.readings[margin].first()?))),
                count: fences.count[at],
            }
        })
        .collect()
}

/// The reading a unit opening on the line's marker counts on: the margin's number where the margin numbers it, in either
/// column; else the number t0 counts it opening on; else the run member's; else the line's first; a marker no reading
/// makes out (a recognizer's "(2]") its opening token.
fn reading_of(view: &LineView, runs: &Runs, reading: Option<(u32, u32)>) -> Option<u8> {
    let (run, member) = reading?;
    let value = runs.runs[run as usize].members[member as usize].value;
    let counted = view.count.and_then(|count| view.readings.iter().position(|reading| reading.value == count));
    let found = counted.or_else(|| view.readings.iter().position(|reading| reading.value == value));
    Some(match (view.margin, found, view.readings.is_empty()) {
        (Some(_), ..) => MARGIN,
        (_, Some(at), _) => at as u8,
        (_, _, false) => 0,
        _ => OPENING,
    })
}

/// A reading of the line as the move it would make: a unit's opening on its marker, a labelled note's opening on its
/// number, a heading's opening at `level`, else the line going on.
fn candidate(view: &LineView, runs: &Runs, role: Role, kind: Option<UnitRole>, reading: Option<(u32, u32)>, level: u8, joined: bool) -> Move {
    let mut chosen = plain(region_of(role, kind), kind.is_some());
    if kind.is_some() {
        chosen.reading = reading_of(view, runs, reading);
    }
    match role {
        Role::Note => {
            chosen.opens = view.label.is_some();
            chosen.note = view.label;
        }
        Role::Heading => {
            chosen.opens = !joined;
            chosen.level = level.clamp(1, 9) as i8;
        }
        _ => {}
    }
    chosen
}

/// How far a candidate fits what is open: `State::features`, the features the fitted ranker reads, weighed here.
fn fit(state: &State, view: &LineView, chosen: &Move) -> f32 {
    let features = state.features(view, chosen);
    let said = |name: &str| CANDIDATE.iter().position(|known| *known == name).is_some_and(|at| features[at] > 0.0);
    let step = CANDIDATE.iter().position(|known| *known == "step").map_or(0.0, |at| features[at]);
    let mut score = 0.0;
    if said("counts") {
        score += match (said("found_open"), said("push_from_one")) {
            (true, _) if step == 1.0 => COUNTS_ON,
            (false, true) => OPENS_COUNT,
            _ => BREAKS_COUNT,
        };
    }
    if said("note_counts") {
        score += if said("note_answers") {
            ANSWERS
        } else if said("note_next") {
            NEXT_NOTE
        } else {
            STRAY_NOTE
        };
    }
    score
}

/// One decision on each line two strong rules read differently, in reading order: the reading standing and those refused
/// weighed on the same terms, the state moved by the winner. A line whose paragraph number the judgment reader counts or
/// holds stays as it stands. Headings' levels are read again where a line changed.
fn settle(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, table: &LineTable, views: &[LineView], decision: &mut Decision) {
    if decision.refused.is_empty() || legal_structure_model::ablated("machine_fit") {
        return;
    }
    let mut refused: HashMap<usize, Vec<(Role, Rule)>> = HashMap::new();
    for &(slot, role, rule) in &decision.refused {
        refused.entry(slot.index()).or_default().push((role, rule));
    }
    let mut state = State::default();
    let mut changed = false;
    for (k, view) in views.iter().enumerate() {
        let at = table.order[k].index();
        let kind = decision.unit[at].map(|unit| unit.1);
        let standing = candidate(view, runs, decision.role[at], kind, decision.reading[at], decision.level[at], decision.joined[at]);
        let mut chosen = standing;
        if fences.count[at].is_none() && fences.held[at].is_empty() {
            if let Some(others) = refused.get(&at) {
                let opens = |rule: Rule| matches!(rule, Rule::Opener | Rule::Caption);
                let member_kind = decision.reading[at].and_then(|(run, member)| runs.runs[run as usize].members[member as usize].kind);
                let mut best = STANDING + fit(&state, view, &standing);
                for &(role, rule) in others {
                    let kind = member_kind.filter(|_| opens(rule));
                    let other = candidate(view, runs, role, kind, decision.reading[at], decision.given[at], false);
                    let own = if matches!(rule, Rule::Anchor | Rule::NotesHeading) { OWN_WORD } else { 0.0 };
                    let score = own + fit(&state, view, &other);
                    if score > best {
                        best = score;
                        chosen = other;
                        decision.role[at] = role;
                        decision.rule[at] = rule;
                        decision.unit[at] = kind.zip(decision.reading[at]).map(|(kind, (run, _))| (run, kind));
                        changed = true;
                    }
                }
            }
        }
        state.apply(view, &chosen);
    }
    if changed {
        let sequences = std::mem::take(&mut decision.sequences);
        heading_levels(pages, slots, evidence, fences, &sequences, &Geometry::of(pages), decision);
        decision.sequences = sequences;
    }
}

/// The moves of the written document. Each reading region (`reading_regions`) is a unit: its first line opens it, on its
/// marker where the decision gave the line a unit, a heading at its level; its other lines go on in it by name. A contents
/// list's lines, nearly all of a region or a page wholly a list's, are the list's. Each note the pairer made is a unit. Any
/// other line stands alone. A body unit's depth is its numbering's (`State::stack_depth`). With the moves, the run member
/// each unit opens on, by the unit's first line: a marker the decision lets open its unit (`opens_units`).
fn moves_of(slots: &Slots, fences: &Fences, runs: &Runs, table: &LineTable, views: &[LineView], decision: &Decision, regions: &[(&Page, &str, Vec<&Line>)], notes: &[Footnote]) -> (Vec<Move>, HashMap<usize, (u32, u32)>) {
    let place = |id: &str| slots.of_id(id).map(|slot| fences.place[slot.index()] as usize);
    let mut moves: Vec<Option<Move>> = vec![None; views.len()];
    let mut opened = HashMap::new();
    for (_, _, lines) in regions {
        let places: Vec<usize> = lines.iter().filter_map(|line| place(&line.id)).collect();
        let Some(&first) = places.iter().min() else { continue };
        let at = table.order[first].index();
        // A line of the note band the pairer gave back is the body's.
        let role = if decision.role[at] == Role::Note { Role::Body } else { decision.role[at] };
        let member = decision.reading[at].filter(|&(run, _)| graph::opens_units(role, decision.rule[at], &runs.runs[run as usize]));
        let kind = member.and_then(|(run, at)| runs.runs[run as usize].members[at as usize].kind);
        let mut opening = candidate(&views[first], runs, role, kind, decision.reading[at], decision.level[at], false);
        let listed = places.iter().filter(|&&k| {
            let slot = table.order[k];
            decision.role[slot.index()] == Role::Contents || fences.contents_pages[slots.page(slot)]
        }).count();
        if !matches!(opening.region, HEADING | TITLE | MARGINAL_NOTE) && listed * 5 >= places.len() * 4 {
            opening = plain(CONTENTS, true);
        } else if let Some(member) = member.filter(|_| kind.is_some()) {
            opened.insert(first, member);
        }
        opening.opens = true;
        for &k in &places {
            moves[k] = Some(if k == first { opening } else { Move { resumes: Some(first as u32), ..plain(opening.region, false) } });
        }
    }
    for note in notes {
        let places: Vec<usize> = note.body_line_ids.iter().filter_map(|id| place(id)).collect();
        let Some(&first) = places.iter().min() else { continue };
        for &k in &places {
            moves[k] = Some(if k == first {
                Move { note: Some(note.label.parse().unwrap_or(0)), ..plain(NOTE, true) }
            } else {
                Move { resumes: Some(first as u32), ..plain(NOTE, false) }
            });
        }
    }
    let starts: HashSet<usize> = fences.starts.iter().filter_map(|start| place(&start.line_id)).collect();
    let mut state = State::default();
    let moves = moves
        .into_iter()
        .enumerate()
        .map(|(k, chosen)| {
            let mut chosen = chosen.unwrap_or_else(|| plain(region_of(decision.role[table.order[k].index()], None), true));
            chosen.starts = starts.contains(&k);
            if chosen.opens && RANKED_BODY.contains(&chosen.region) {
                chosen.depth = state.stack_depth(&views[k], &chosen).min(9) as u8;
            }
            state.apply(&views[k], &chosen);
            chosen
        })
        .collect();
    (moves, opened)
}

/// Each page's lines back in their slots, which `emit` addresses them by.
fn in_slot_order(pages: &mut [Page], slots: &Slots) {
    for page in pages {
        page.lines.sort_by_cached_key(|line| slots.of_id(&line.id));
    }
}

/// The document's structure as the machine reads it, what `derive` gives where the request names `Reader::Machine`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn output(
    analysis: &dyn StructureAnalysis,
    pages: &mut [Page],
    slots: &Slots,
    evidence: &Evidence,
    mut fences: Fences,
    runs: &Runs,
    outline: &[PdfOutlineEntry],
    identity: &StructureIdentity,
    trace: Option<&Path>,
) -> Result<StructureOutput> {
    let mut notes = Notes::read(analysis, pages, slots, evidence, &fences, runs);
    let mut decision = Decision::decide(analysis, pages, slots, evidence, &fences, runs, &notes, outline, trace.is_some());
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    diagnostics.append(&mut notes.diagnostics);
    diagnostics.append(&mut decision.diagnostics);
    let table = LineTable::read(pages, slots, evidence, &fences, &runs.before, &HashMap::new());
    let views = line_views(pages, slots, evidence, &fences, &table);
    settle(pages, slots, evidence, &fences, runs, &table, &views, &mut decision);
    if let Some(folder) = trace {
        write_trace(folder, &identity.source_sha256, pages, slots, evidence, &fences, runs, &notes, &decision);
    }
    let decided = Decided::of(pages, slots, &fences, &decision);
    let heading_sources = write_lines(pages, slots, evidence, &fences, &notes, &decision);
    // The pairer reads each page's lines in reading order.
    in_reading_order(pages, slots, &fences);
    build_regions(pages);
    diagnostics.extend(text_flow_faults(pages));
    let index = PdfTextIndex::from_pages(pages);
    let citations = graph::citation_spans(pages, slots, evidence);
    let pairing = note_pairing::pair_notes(analysis, pages, &decided, &citations, &index);
    let (claims, graph_diagnostics) = map_note_pairs(&index, &pairing.pair_claims)?;
    let anchors = pairing.anchors;
    let footnotes = pairing.footnotes;
    diagnostics.extend(pairing.diagnostics);
    if give_back(pages, &footnotes) {
        build_regions(pages);
    }
    let (moves, opened) = moves_of(slots, &fences, runs, &table, &views, &decision, &reading_regions(pages, &decided), &footnotes);
    in_slot_order(pages, slots);
    let settled = Settled { index: &index, claims, anchors: &anchors, footnotes, diagnostics: graph_diagnostics, runs, opened };
    let (structure_graph, paragraphs, mut footnotes) = emit::structure_graph(pages, slots, evidence, &fences, &table, &views, &moves, identity, Some(settled))?;
    diagnostics.extend(unmatched_reference_diagnostics(pages, &footnotes, &anchors, &decided));
    attach_propositions(&mut footnotes, &paragraphs, &decided);
    attach_crossrefs(analysis, &mut footnotes, &mut diagnostics, &decided);
    in_reading_order(pages, slots, &fences);
    Ok(StructureOutput { paragraphs, footnotes, diagnostics, structure_graph, heading_sources })
}
