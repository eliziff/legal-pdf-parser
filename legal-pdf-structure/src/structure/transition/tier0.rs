//! Tier 0 as the machine's reader: every rule heard on every line, one decision on each line, the scorer compiled in (no
//! fitted file, no environment), and the moves the decided lines make written by `emit`. It is chosen by `Reader::Machine`,
//! the default; `Reader::Stages` stays to measure against.
//!
//! What it reads, in order:
//! - the notes (`Notes::read`), and the rules heard (`Decision::hear`): no rule's reading keeps another's from a line, and
//!   the residual weighs a line a marker opens as it weighs an open one;
//! - one decision on each line (`settle`), in reading order against what is open (`machine::State`): each reading worth
//!   what its rule is worth (`worth`, in the order the stages took their turns, the residual at its score), and where
//!   readings compete, how each fits the numbering and the notes open (`fit`). Paragraph numbers stand outside it: a line
//!   the judgment reader counts or holds (`Fences::paragraphs`) goes by its rules' worth alone, the residual
//!   does not weigh it, and `read_judgment_paragraphs` writes the numbers as t0 does;
//! - what follows the decision (`Decision::finish`): the title, the levels;
//! - the decided lines written (`write_lines`), their regions built, their notes paired (`note_pairing`), the pairer's
//!   pairs fixed input to the moves, as paragraph numbers are;
//! - the moves (`moves_of`): each of the document's reading regions a unit, each note a unit, each line set apart a
//!   unit of its own.

use super::emit::Settled;
use super::*;
use legal_structure_model::UnitRole;

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

/// What a rule's reading is worth before the state speaks: the order the stages take their turns, near enough that a
/// count or a note's number can tip two neighbours. A line a bookmark names stands just under a marker, and a heading the
/// residual weighs at its score over the worth of body.
fn worth(heard: &Heard, chosen: &Move) -> f32 {
    match heard.rule {
        _ if heard.role == Role::Open => 0.0,
        Rule::EndnoteHead => 12.0,
        Rule::NotesHeading => 11.0,
        Rule::NoteBand if chosen.note.is_some() => 7.75,
        Rule::NoteBand => 8.75,
        Rule::HistoryHeading => 7.7,
        Rule::TranscriptPart => 7.65,
        Rule::QuotedNumber => 7.6,
        Rule::TitleOnly => 7.5,
        Rule::Opener | Rule::Caption => 7.0,
        Rule::DivisionHeading | Rule::Ladder => 6.9,
        Rule::MarginalNote => 6.8,
        Rule::Anchor => 6.75,
        Rule::HistoryLine | Rule::TranscriptLine | Rule::FrontMatter => 4.0,
        Rule::Residual => 3.0 + heard.weight,
        _ => 0.0,
    }
}

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
                left: 0.0,
                page: slots.page(slot) as u32,
                margin: table.margins.get(&k).and_then(|&margin| Some((margin as u32, table.readings[margin].first()?))),
                count: fences.paragraphs.count[at],
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

/// One decision on each line no fence took, in reading order: its readings each worth its rule's word (`worth`), and where
/// more than one speaks and no paragraph number is at stake, how it fits what is open (`fit`); the state moved by the
/// winner. A heading the residual weighed over several lines goes on only from its own first line: where that line went
/// another way, the rest stand as they would alone. A line no rule read stays open, for `Decision::finish`.
fn settle(fences: &Fences, runs: &Runs, table: &LineTable, views: &[LineView], decision: &mut Decision) {
    let weighing = !legal_structure_model::ablated("machine_fit");
    let mut heard = std::mem::take(&mut decision.heard);
    heard.sort_by_key(|reading| reading.slot);
    let mut state = State::default();
    let mut starts = vec![0u32; decision.role.len() + 1];
    for reading in &heard {
        starts[reading.slot.index() + 1] += 1;
    }
    for at in 1..starts.len() {
        starts[at] += starts[at - 1];
    }
    // Each reading context: whether its last line is a heading the residual weighed.
    let mut weighed = vec![false; fences.contexts.len()];
    for (k, view) in views.iter().enumerate() {
        let slot = table.order[k];
        let at = slot.index();
        let chosen = if fences.role[at] == Role::Open {
            let context = usize::from(fences.ctx[at]);
            let readings: Vec<Heard> = heard[starts[at] as usize..starts[at + 1] as usize].iter().copied()
                .filter(|reading| !(reading.rule == Rule::Residual && reading.joined && !weighed[context]))
                .collect();
            let fixed = fences.paragraphs.count[at].is_some() || !fences.paragraphs.held[at].is_empty();
            let competing = readings.iter().filter(|reading| reading.role != Role::Open).count() > 1;
            let member_kind = decision.reading[at].and_then(|(run, member)| runs.runs[run as usize].members[member as usize].kind);
            let mut winner: Option<(Heard, Move, f32)> = None;
            for reading in &readings {
                let kind = member_kind.filter(|_| matches!(reading.rule, Rule::Opener | Rule::Caption));
                let chosen = candidate(view, runs, reading.role, kind, decision.reading[at], decision.given[at], reading.joined);
                let mut score = worth(reading, &chosen);
                if weighing && competing && !fixed {
                    score += fit(&state, view, &chosen);
                }
                if winner.as_ref().is_none_or(|(_, _, best)| score > *best) {
                    winner = Some((*reading, chosen, score));
                }
            }
            decision.settle_line(slot, winner.as_ref().map(|(reading, ..)| *reading), &readings);
            weighed[context] = winner.as_ref().is_some_and(|(reading, ..)| reading.rule == Rule::Residual && reading.role == Role::Heading);
            winner.map_or_else(|| plain(TEXT, false), |(_, chosen, _)| chosen)
        } else {
            candidate(view, runs, decision.role[at], None, None, 0, false)
        };
        state.apply(view, &chosen);
    }
    decision.hearing = false;
}

/// The moves of the written document. Each reading region (`reading_regions`) is a unit: its first line opens it, on its
/// marker where the decision gave the line a unit, a heading at its level, and so does each marker the decision lets open
/// a unit further down it (a page's leading may not part them); its other lines go on in the unit they follow by name, and
/// a region going on from the last (`Fences::continues`) goes on in the numbered unit that region ended in, over a page. A contents
/// list's lines, nearly all of a region or a page wholly a list's, are the list's. Each note the pairer made is a unit. Any
/// other line stands alone. A body unit's depth is its numbering's (`State::stack_depth`). With the moves, the run member
/// each unit opens on, by the unit's first line: a marker the decision lets open its unit (`opens_units`).
fn moves_of(slots: &Slots, fences: &Fences, runs: &Runs, table: &LineTable, views: &[LineView], decision: &Decision, regions: &[(&Page, &str, Vec<&Line>)], notes: &[Footnote]) -> (Vec<Move>, HashMap<usize, (u32, u32)>, HashMap<usize, Vec<(u32, u32)>>) {
    let place = |id: &str| slots.of_id(id).map(|slot| fences.place[slot.index()] as usize);
    let mut moves: Vec<Option<Move>> = vec![None; views.len()];
    let mut opened = HashMap::new();
    // A marker any run gives a unit opens it, where the decision read the line by no run of its own.
    let mut marked = HashMap::<Slot, Vec<(u32, u32)>>::new();
    for (r, run) in runs.runs.iter().enumerate() {
        for (at, member) in run.members.iter().enumerate().filter(|(_, member)| member.kind.is_some()) {
            marked.entry(member.slot).or_default().push((r as u32, at as u32));
        }
    }
    // The further markers a line opens on, a unit each: "2(1) The Court has ...".
    let mut within = HashMap::<usize, Vec<(u32, u32)>>::new();
    // The pairer's lines are its notes', a line it took from the body among them.
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
    // Each reading context's numbered unit open at its last region's end.
    let mut numbered = HashMap::<u16, (usize, u8)>::new();
    for (_, _, lines) in regions {
        let mut places: Vec<usize> = lines.iter().filter_map(|line| place(&line.id)).filter(|&k| moves[k].is_none()).collect();
        places.sort_unstable();
        let Some(&first) = places.first() else { continue };
        let listed = places.iter().filter(|&&k| {
            let slot = table.order[k];
            decision.role[slot.index()] == Role::Contents || fences.contents_pages[slots.page(slot)]
        }).count();
        let contents = listed * 5 >= places.len() * 4;
        let context = fences.ctx[table.order[first].index()];
        let body = matches!(decision.role[table.order[first].index()], Role::Open | Role::Body);
        let carried = numbered.get(&context).copied().filter(|_| body && !contents && fences.continues[table.order[first].index()].is_some());
        let mut open = carried.unwrap_or((first, 0));
        for &k in &places {
            let at = table.order[k].index();
            // A line of the note band the pairer gave back is the body's.
            let role = if decision.role[at] == Role::Note { Role::Body } else { decision.role[at] };
            let reading = decision.reading[at].or_else(|| marked.get(&table.order[k]).and_then(|members| members.first().copied()));
            let member = reading.filter(|&(run, _)| graph::opens_units(role, decision.rule[at], &runs.runs[run as usize]));
            let kind = member.and_then(|(run, at)| runs.runs[run as usize].members[at as usize].kind);
            if k == first && carried.is_none() || kind.is_some() && !contents {
                let mut opening = candidate(&views[k], runs, role, kind, reading, decision.level[at], false);
                if !matches!(opening.region, HEADING | TITLE | MARGINAL_NOTE) && contents {
                    opening = plain(CONTENTS, true);
                } else if let Some(member) = member.filter(|_| kind.is_some()) {
                    opened.insert(k, member);
                    let mark = |(run, at): (u32, u32)| runs.runs[run as usize].members[at as usize].marker[0];
                    let mut further: Vec<(u32, u32)> = marked.get(&table.order[k]).into_iter().flatten().copied()
                        .filter(|&other| other != member && mark(other).offset > mark(member).offset && graph::opens_units(role, decision.rule[at], &runs.runs[other.0 as usize])).collect();
                    further.sort_by_key(|&other| mark(other).offset);
                    further.dedup_by_key(|other| mark(*other).offset);
                    if !further.is_empty() {
                        within.insert(k, further);
                    }
                }
                opening.opens = true;
                moves[k] = Some(opening);
                open = (k, opening.region);
            } else {
                moves[k] = Some(Move { resumes: Some(open.0 as u32), ..plain(open.1, false) });
            }
        }
        if opened.contains_key(&open.0) {
            numbered.insert(context, open);
        } else {
            numbered.remove(&context);
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
    (moves, opened, within)
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
    let geometry = Geometry::of(pages);
    let mut decision = Decision::hear(analysis, pages, slots, evidence, &fences, runs, &notes, outline, &geometry, trace.is_some());
    let table = LineTable::plain(pages, slots, evidence, &fences, &runs.before);
    let views = line_views(pages, slots, evidence, &fences, &table);
    settle(&fences, runs, &table, &views, &mut decision);
    decision.finish(pages, slots, evidence, &fences, runs, &geometry);
    let mut diagnostics = std::mem::take(&mut fences.diagnostics);
    diagnostics.append(&mut notes.diagnostics);
    diagnostics.append(&mut decision.diagnostics);
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
    let (moves, opened, within) = moves_of(slots, &fences, runs, &table, &views, &decision, &reading_regions(pages, &decided), &footnotes);
    in_slot_order(pages, slots);
    let settled = Settled { index: &index, claims, anchors: &anchors, footnotes, diagnostics: graph_diagnostics, runs, opened, within };
    let (structure_graph, paragraphs, mut footnotes) = emit::structure_graph(pages, slots, evidence, &fences, &table, &views, &moves, identity, Some(settled))?;
    diagnostics.extend(unmatched_reference_diagnostics(pages, &footnotes, &anchors, &decided));
    attach_propositions(&mut footnotes, &paragraphs, &decided);
    attach_crossrefs(analysis, &mut footnotes, &mut diagnostics, &decided);
    in_reading_order(pages, slots, &fences);
    Ok(StructureOutput { paragraphs, footnotes, diagnostics, structure_graph, heading_sources })
}
