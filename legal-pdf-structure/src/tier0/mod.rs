//! Tier 0 as the machine's reader: every rule heard on every line, one decision on each line, the scorer compiled in (no
//! fitted file, no environment). Each line's move is its one record (`Decision::moves`), from the hearing to the walk that
//! writes the units.
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
//! - the decided lines written (`write_lines`), their notes paired (the engine's pairer, over `note_lines`), the pairer's
//!   pairs fixed input to the moves, as paragraph numbers are;
//! - the moves grouped in place (`moves_of`): each of the document's reading regions a unit, each note a unit, each line
//!   set apart a unit of its own.

use legal_pdf_core::model::{LineZone, Page, PdfOutlineEntry};
use crate::layout::opens_turn;
use crate::decide::{Decision, Heard, Notes};
use crate::layout::Geometry;
use crate::fences::Fences;
use crate::graph;
use crate::machine::machine::{LineView, Move, State};
use crate::model::regions::{region_of, CONTENTS, HEADING, MARGINAL_NOTE, NOTE, TEXT, TITLE};
use crate::model::{Evidence, Role, Rule, Runs, Slot, Slots};
use std::collections::{HashMap, HashSet};
use legal_structure_model::{StructureNode, UnitRole};

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

/// The reading a unit opening on the line's marker counts on (`LineView::opening`), tier 0 picking the run member's.
fn reading_of(view: &LineView, runs: &Runs, reading: Option<(u32, u32)>) -> Option<u8> {
    let (run, member) = reading?;
    let value = runs.runs[run as usize].members[member as usize].value;
    Some(view.opening(|| view.readings().iter().position(|reading| reading.value == value)))
}

/// A reading of the line as the move it would make: a unit's opening on its marker, a labelled note's opening on its
/// number, a heading's opening at `level`, else the line going on.
fn candidate(view: &LineView, runs: &Runs, role: Role, rule: Rule, kind: Option<UnitRole>, reading: Option<(u32, u32)>, level: u8, joined: bool) -> Move {
    let chosen = Move { role, rule, reading: kind.and_then(|_| reading_of(view, runs, reading)), ..Move::plain(region_of(role, kind), kind.is_some()) };
    match role {
        Role::Note => Move { opens: view.label().is_some(), note: view.label(), ..chosen },
        Role::Heading => Move { opens: !joined, level: level.clamp(1, 9), ..chosen },
        _ => chosen,
    }
}

/// How far a candidate fits what is open in its context: its marker against the open frame of its family, a note's
/// number against the marks no note has answered and the last note.
fn fit(state: &State, view: &LineView, chosen: &Move) -> f32 {
    let open = state.open(view, chosen.starts);
    let mut score = 0.0;
    if let Some(reading) = view.reading(chosen.reading) {
        let frames = open.map_or(&[][..], |open| open.frames());
        score += match frames.iter().rposition(|frame| frame.family == reading.family) {
            Some(depth) if reading.value == frames[depth].value + 1 => COUNTS_ON,
            None if reading.value == 1 => OPENS_COUNT,
            _ => BREAKS_COUNT,
        };
    }
    if let (true, NOTE, Some(value)) = (chosen.opens, chosen.region, chosen.note) {
        score += if open.is_some_and(|open| open.unanswered.iter().any(|&(r, ..)| r == value)) {
            ANSWERS
        } else if open.and_then(|open| open.note).is_some_and(|last| value == last + 1) {
            NEXT_NOTE
        } else {
            STRAY_NOTE
        };
    }
    score
}

/// One decision on each line no fence took, in reading order: its readings each worth its rule's word (`worth`), and where
/// more than one speaks and no paragraph number is at stake, how it fits what is open (`fit`); the state moved by the
/// winner, whose move takes the line's place. A heading the residual weighed over several lines
/// goes on only from its own first line: where that line went another way, the rest stand as they would alone. A line no
/// rule read stays open, for `Decision::finish`. A fenced line makes its role's move, its hearing's joining and unit kept.
fn settle(fences: &Fences, runs: &Runs, views: &[LineView], decision: &mut Decision) {
    let weighing = !legal_structure_model::ablated("machine_fit");
    let mut heard = std::mem::take(&mut decision.heard);
    heard.sort_by_key(|reading| reading.slot);
    let mut state = State::default();
    let mut starts = vec![0u32; decision.moves.len() + 1];
    for reading in &heard {
        starts[reading.slot.index() + 1] += 1;
    }
    for at in 1..starts.len() {
        starts[at] += starts[at - 1];
    }
    // Each reading context: whether its last line is a heading the residual weighed.
    let mut weighed = vec![false; fences.contexts.len()];
    for (k, view) in views.iter().enumerate() {
        let at = view.slot().index();
        let line = &decision.moves[k];
        if !decision.free[at] {
            let chosen = Move { joined: line.joined, unit: line.unit, ..candidate(view, runs, line.role, line.rule, None, None, 0, false) };
            state.apply(view, &chosen);
            decision.moves[k] = chosen;
            continue;
        }
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
            let chosen = candidate(view, runs, reading.role, reading.rule, kind, decision.reading[at], decision.given[at], reading.joined);
            let mut score = worth(reading, &chosen);
            if weighing && competing && !fixed {
                score += fit(&state, view, &chosen);
            }
            if winner.as_ref().is_none_or(|(_, _, best)| score > *best) {
                winner = Some((*reading, chosen, score));
            }
        }
        weighed[context] = winner.as_ref().is_some_and(|(reading, ..)| reading.rule == Rule::Residual && reading.role == Role::Heading);
        // The line goes on with the one above where its reading does; the unit, level and anchor stand only by its rule.
        let (joined, chosen) = winner.map_or_else(|| (false, Move::plain(TEXT, false)), |(reading, chosen, _)| (reading.joined, chosen));
        let unit = line.unit.filter(|_| matches!(chosen.rule, Rule::Opener | Rule::Caption));
        let chosen = Move { joined, unit, ..chosen };
        if !matches!(chosen.rule, Rule::Anchor | Rule::TranscriptPart) { decision.given[at] = 0; }
        if chosen.rule != Rule::Anchor { decision.anchor[at] = 0; }
        state.apply(view, &chosen);
        decision.moves[k] = chosen;
    }
    decision.hearing = false;
}

/// The moves of the written document. Each reading region (`reading_regions`) is a unit: its first line opens it, on its
/// marker where the decision gave the line a unit, a heading at its level, and so does each marker the decision lets open
/// a unit further down it (a page's leading may not part them); its other lines go on in the unit they follow by name, and
/// a region going on from the last (`Fences::continues`) goes on in the numbered unit that region ended in, over a page. A contents
/// list's lines, nearly all of a region or a page wholly a list's, are the list's. Each note the pairer made is a unit. Any
/// other line stands alone; a body unit's depth is the walk's to read (`emit`). A unit's first line's move carries the
/// run member the unit opens on, a marker the decision lets open its unit (`opens_units`), and the further markers the
/// line opens on, a unit each. Each line's move is grouped in its place, its decision kept (`Move::regroup`).
pub(crate) fn moves_of(slots: &Slots, fences: &Fences, runs: &Runs, views: &[LineView], decision: &mut Decision, regions: &[Vec<Slot>], notes: &[StructureNode]) {
    let place = |id: &str| slots.of_id(id).map(|slot| fences.place[slot.index()] as usize);
    let (moves, readings) = (&mut decision.moves, &decision.reading);
    let mut made = vec![false; moves.len()];
    // A marker any run gives a unit opens it, where the decision read the line by no run of its own.
    let mut marked = HashMap::<Slot, Vec<(u32, u32)>>::new();
    for (r, run) in runs.runs.iter().enumerate() {
        for (at, member) in run.members.iter().enumerate().filter(|(_, member)| member.kind.is_some()) {
            marked.entry(member.slot).or_default().push((r as u32, at as u32));
        }
    }
    // The pairer's lines are its notes', a line it took from the body among them.
    for note in notes {
        let places: Vec<usize> = note.line_ids.iter().filter_map(|id| place(id)).collect();
        let Some(&first) = places.iter().min() else { continue };
        for &k in &places {
            made[k] = true;
            moves[k].regroup(if k == first {
                Move { note: Some(note.label.as_deref().and_then(|label| label.parse().ok()).unwrap_or(0)), ..Move::plain(NOTE, true) }
            } else {
                Move { resumes: Some(first as u32), ..Move::plain(NOTE, false) }
            });
        }
    }
    // Each reading context's numbered unit open at its last region's end.
    let mut numbered = HashMap::<u16, (usize, u8)>::new();
    for lines in regions {
        let mut places: Vec<usize> = lines.iter().map(|slot| fences.place[slot.index()] as usize).filter(|&k| !made[k]).collect();
        places.sort_unstable();
        let Some(&first) = places.first() else { continue };
        let listed = places.iter().filter(|&&k| moves[k].role == Role::Contents || fences.contents_pages[slots.page(fences.order[k])]).count();
        let contents = listed * 5 >= places.len() * 4;
        let first_at = fences.order[first].index();
        let context = fences.ctx[first_at];
        let body = matches!(moves[first].role, Role::Open | Role::Body);
        let carried = numbered.get(&context).copied().filter(|_| body && !contents && fences.continues[first_at].is_some());
        let mut open = carried.unwrap_or((first, 0));
        for &k in &places {
            let slot = fences.order[k];
            // A line of the note band the pairer gave back is the body's.
            let (role, rule) = (if moves[k].role == Role::Note { Role::Body } else { moves[k].role }, moves[k].rule);
            let reading = readings[slot.index()].or_else(|| marked.get(&slot).and_then(|members| members.first().copied()));
            let member = reading.filter(|&(run, _)| graph::opens_units(role, rule, &runs.runs[run as usize]));
            let kind = member.and_then(|(run, at)| runs.runs[run as usize].members[at as usize].kind);
            made[k] = true;
            if k == first && carried.is_none() || kind.is_some() && !contents {
                let mut opening = candidate(&views[k], runs, role, rule, kind, reading, moves[k].level, false);
                if !matches!(opening.region, HEADING | TITLE | MARGINAL_NOTE) && contents {
                    opening = Move::plain(CONTENTS, true);
                } else if let Some(member) = member.filter(|_| kind.is_some()) {
                    let mark = |(run, at): (u32, u32)| runs.runs[run as usize].members[at as usize].marker[0];
                    let mut further: Vec<(u32, u32)> = marked.get(&slot).into_iter().flatten().copied()
                        .filter(|&other| other != member && mark(other).offset > mark(member).offset && graph::opens_units(role, rule, &runs.runs[other.0 as usize])).collect();
                    further.sort_by_key(|&other| mark(other).offset);
                    further.dedup_by_key(|other| mark(*other).offset);
                    opening.member = Some(member);
                    opening.within = further;
                }
                opening.opens = true;
                open = (k, opening.region);
                moves[k].regroup(opening);
            } else {
                moves[k].regroup(Move { resumes: Some(open.0 as u32), ..Move::plain(open.1, false) });
            }
        }
        if moves[open.0].member.is_some() {
            numbered.insert(context, open);
        } else {
            numbered.remove(&context);
        }
    }
    for (line, made) in moves.iter_mut().zip(made) {
        if !made {
            line.regroup(Move::plain(region_of(line.role, None), true));
        }
    }
    for start in fences.starts.iter().filter_map(|start| place(&start.line_id)) {
        moves[start].starts = true;
    }
}

/// The one decision on each line, over the notes read and the fences' roles (`fenced`: a fenced line keeps its own,
/// every other is open to the rules).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decide(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, geometry: &Geometry, outline: &[PdfOutlineEntry], notes: &Notes, views: &[LineView], fenced: (Vec<Role>, Vec<Rule>), traced: bool) -> Decision {
    let mut decision = Decision::hear(pages, slots, evidence, fences, runs, notes, outline, geometry, traced, fenced);
    settle(fences, runs, views, &mut decision);
    decision.finish(pages, slots, evidence, fences, runs, geometry);
    decision
}

/// The document's reading regions. On each page, a region is a run of its lines in reading order whose zone and block
/// go on (on a transcript's page, the body's by turns, a turn running from the line that opens it to the next that opens
/// one; a line label parting none). Each region of the body, a heading or title or a marginal note is read in its
/// reading contexts, a body's going on over a page in the region it continues.
pub(crate) fn reading_regions(pages: &[Page], slots: &Slots, fences: &Fences, zones: &[LineZone], blocks: &[usize]) -> Vec<Vec<Slot>> {
    let context = |slot: Slot| (fences.ctx[slot.index()], fences.at(slot.index()).translation);
    let mut regions: Vec<(bool, Vec<Slot>)> = Vec::new();
    let mut tails = HashMap::<_, usize>::new();
    for page_slot in 0..pages.len() {
        let range = slots.on(page_slot);
        let order = &fences.order[range.start as usize..range.end as usize];
        let transcript = order.iter().any(|slot| zones[slot.index()] == LineZone::LineLabel);
        let mut groups: Vec<Vec<Slot>> = Vec::new();
        for &slot in order.iter().filter(|slot| zones[slot.index()] != LineZone::LineLabel) {
            let zone = zones[slot.index()];
            match groups.last_mut() {
                Some(group) if {
                    let prior = group[group.len() - 1].index();
                    zones[prior] == zone
                        && if transcript && zone == LineZone::Body { !opens_turn(slots.line(pages, slot)) } else { blocks[prior] == blocks[slot.index()] }
                } => group.push(slot),
                _ => groups.push(vec![slot]),
            }
        }
        let mut seen = HashSet::new();
        for group in groups.iter().filter(|group| matches!(zones[group[0].index()], LineZone::Body | LineZone::Heading | LineZone::MarginalNote)) {
            let body = zones[group[0].index()] == LineZone::Body;
            for lines in group.chunk_by(|&left, &right| context(left) == context(right)) {
                let key = context(lines[0]);
                let first = seen.insert(key);
                // The other column's intervening regions do not close this context's paragraph; one ending the page
                // before goes on where the page's first line continues it (`Fences::continues`).
                if let Some(&at) = tails.get(&key) {
                    let (previous_body, previous) = &regions[at];
                    if first && body && *previous_body && slots.page(previous[previous.len() - 1]) + 1 == page_slot && fences.continues[lines[0].index()].is_some() {
                        regions[at].1.extend_from_slice(lines);
                        continue;
                    }
                }
                tails.insert(key, regions.len());
                regions.push((body, lines.to_vec()));
            }
        }
    }
    regions.into_iter().map(|(_, lines)| lines).collect()
}