//! Tier 0 as the machine's reader: every rule heard on every line, then one walk over the lines in reading order that
//! decides each line and groups it as it goes, the scorer compiled in (no fitted file, no environment). Each line's move
//! is its one record (`Decision::moves`), from the hearing to the units the moves make.
//!
//! What it reads, in order:
//! - the notes (`Notes::read`), and the rules heard (`Decision::hear`): no rule's reading keeps another's from a line, and
//!   the residual weighs a line a marker opens as it weighs an open one;
//! - the walk (`settle`), against what is open (`machine::State`): each line's readings each worth what its rule is worth
//!   (`worth`, in the order the stages took their turns, the residual at its score), and where readings compete, how
//!   each fits the numbering and the notes open (`fit`); then, the line decided, its block (`Paragraphs`), and the unit
//!   it opens or goes on in, at its depth. Paragraph numbers stand outside it: a line the judgment reader counts or holds
//!   (`Fences::paragraphs`) goes by its rules' worth alone, and `read_judgment_paragraphs` writes the numbers as t0 does;
//! - what needs the whole document after it (`Decision::finish`): the title, the headings' levels. The pairer's notes
//!   are laid over the units the walk made where they are written (`crate::output`).

use legal_pdf_core::model::{LineZone, NoteMode, Page, PdfOutlineEntry};
use crate::layout::opens_turn;
use crate::decide::{unit_role, Decision, Heard, Notes, Paragraphs};
use crate::layout::Geometry;
use crate::fences::Fences;
use crate::graph;
use crate::machine::machine::{LineView, Move, State, RANKED_BODY};
use crate::model::regions::{region_of, zone_of, CONTENTS, HEADING, MARGINAL_NOTE, NOTE, TEXT, TITLE};
use crate::model::{Evidence, MarkerReading, Role, Rule, Runs, Slot, Slots};
use std::collections::{HashMap, HashSet};
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

/// The reading a unit opening on the line's marker counts on (`LineView::opening`), tier 0 picking the run member's.
fn reading_of(view: &LineView, runs: &Runs, (run, member): (u32, u32)) -> u8 {
    let value = runs.runs[run as usize].members[member as usize].value;
    view.opening(|| view.readings().iter().position(|reading| reading.value == value))
}

/// A reading of the line as the move it would make: a unit's opening on its marker (`member`, where its rule opens it),
/// a labelled note's opening on its number, a heading's opening, else the line going on.
fn candidate(view: &LineView, runs: &Runs, role: Role, rule: Rule, member: Option<((u32, u32), UnitRole)>, joined: bool) -> Move {
    let chosen = Move { role, rule, reading: member.map(|(member, _)| reading_of(view, runs, member)), ..Move::plain(region_of(role, member.map(|(_, kind)| kind)), member.is_some()) };
    match role {
        Role::Note => Move { opens: view.label().is_some(), note: view.label(), ..chosen },
        Role::Heading => Move { opens: !joined, ..chosen },
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

/// A reading context's region read last: its open unit (its first line's place) and region, its last line's page, and
/// whether it is the body's and a contents list's.
struct Tail {
    open: usize,
    region: u8,
    page: usize,
    body: bool,
    contents: bool,
}

/// The walk: each line in reading order decided and grouped, the state moved by its move.
///
/// Decided: a fenced line makes its role's move, its hearing's joining and unit kept. Any other line's readings are each
/// worth their rule's word (`worth`), and where more than one speaks and no paragraph number is at stake, how each fits
/// what is open (`fit`). A heading the residual weighed over several lines goes on only from its own first line: where
/// that line went another way, the rest stand as they would alone. A line no rule settled that a counted marker opens
/// opens the unit its reading reads, the line carrying its text going on with it; any other is body, or a quotation's.
///
/// Grouped: the line starts a block where `Paragraphs::starts` says so. On each page, a region is a run of lines in one
/// reading context whose zone and block go on (on a transcript's page, the body's by turns, a turn running from the line
/// that opens it to the next that opens one; a line label parting none); a region of the body, a heading or a marginal
/// note is read in units. Its first line opens one, and so does each line a marker any run gives a unit opens where the
/// rule lets that marker open it (`opens_units`), carrying the member and the further markers its line opens on, a unit
/// each; its other lines go on in the unit open. A body region going on from the line before it (`Fences::continues`)
/// goes on in its context's last region where that ended on the page before, or in the numbered unit it ended in. A
/// region whose first line is a contents list's is the list's. A note opens on its label; any other line stands alone. A
/// body unit takes its numbering's depth (`State::stack_depth`).
///
/// Each line's block, by slot, is the walk's too: the note pairer reads it.
fn settle(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, views: &[LineView], decision: &mut Decision) -> Vec<usize> {
    let weighing = !legal_structure_model::ablated("machine_fit");
    let mut heard = std::mem::take(&mut decision.heard);
    heard.sort_by_key(|reading| reading.slot);
    let mut first_heard = vec![0u32; slots.len() + 1];
    for reading in &heard {
        first_heard[reading.slot.index() + 1] += 1;
    }
    for at in 1..first_heard.len() {
        first_heard[at] += first_heard[at - 1];
    }
    let paragraphs = Paragraphs::read(pages, slots, evidence, fences, |at| decision.line(at).role);
    // A marker any run gives a unit, by its line: the opening of a line the rules read by no run of their own, and the
    // further markers a unit's first line opens on.
    let mut marked = HashMap::<Slot, Vec<(u32, u32)>>::new();
    for (r, run) in runs.runs.iter().enumerate() {
        for (at, member) in run.members.iter().enumerate().filter(|(_, member)| member.kind.is_some()) {
            marked.entry(member.slot).or_default().push((r as u32, at as u32));
        }
    }
    let starts: HashSet<Slot> = fences.starts.iter().filter_map(|start| slots.of_id(&start.line_id)).collect();
    let transcript: Vec<bool> = (0..pages.len()).map(|page| slots.on(page).any(|at| decision.line(at as usize).role == Role::LineLabel)).collect();
    let member_of = |(run, at): (u32, u32)| &runs.runs[run as usize].members[at as usize];
    let mut state = State::default();
    // Each reading context: whether its last line is a heading the residual weighed.
    let mut weighed = vec![false; fences.contexts.len()];
    let mut blocks = vec![0usize; slots.len()];
    // The line carrying a marker's text, where the marker opened its unit on no rule's word: it goes on with it.
    let mut carrying = HashMap::<Slot, Role>::new();
    // Each document: whether a unit has opened in it.
    let mut opened = vec![false; fences.documents()];
    let mut tails = HashMap::<u16, Tail>::new();
    // On the page read: the line before (and whether it stood before its document's first unit), the line before that is
    // no line label (its zone, block and context), the contexts met, and the block numbered last.
    let mut page = usize::MAX;
    let mut prior: Option<(Slot, bool)> = None;
    let mut grouped: Option<(LineZone, usize, u16)> = None;
    let mut met = HashSet::new();
    let mut block = 0;
    for (k, view) in views.iter().enumerate() {
        let slot = view.slot();
        let at = slot.index();
        if slots.page(slot) != page {
            page = slots.page(slot);
            (prior, grouped) = (None, None);
            met.clear();
            block = pages[page].lines.iter().map(|line| line.block_index).max().unwrap_or(0);
        }
        let line = &decision.moves[k];
        let mut chosen = if decision.free[at] {
            let context = usize::from(fences.ctx[at]);
            let readings: Vec<Heard> = heard[first_heard[at] as usize..first_heard[at + 1] as usize].iter().copied()
                .filter(|reading| !(reading.rule == Rule::Residual && reading.joined && !weighed[context]))
                .collect();
            let fixed = fences.paragraphs.count[at].is_some() || !fences.paragraphs.held[at].is_empty();
            let competing = readings.iter().filter(|reading| reading.role != Role::Open).count() > 1;
            let member = decision.reading[at].and_then(|member| Some((member, member_of(member).kind?)));
            let mut winner: Option<(Heard, Move, f32)> = None;
            for reading in &readings {
                let chosen = candidate(view, runs, reading.role, reading.rule, member.filter(|_| matches!(reading.rule, Rule::Opener | Rule::Caption)), reading.joined);
                let mut score = worth(reading, &chosen);
                if weighing && competing && !fixed {
                    score += fit(&state, view, &chosen);
                }
                if winner.as_ref().is_none_or(|(_, _, best)| score > *best) {
                    winner = Some((*reading, chosen, score));
                }
            }
            weighed[context] = winner.as_ref().is_some_and(|(reading, ..)| reading.rule == Rule::Residual && reading.role == Role::Heading);
            // The line goes on with the one above where its reading does; its unit stands only by its rule.
            let (joined, chosen) = winner.map_or_else(|| (false, Move::plain(TEXT, false)), |(reading, chosen, _)| (reading.joined, chosen));
            let unit = line.unit.filter(|_| matches!(chosen.rule, Rule::Opener | Rule::Caption));
            Move { joined, unit, ..chosen }
        } else {
            Move { joined: line.joined, unit: line.unit, ..candidate(view, runs, line.role, line.rule, None, false) }
        };
        if chosen.role == Role::Open {
            if let Some(role) = carrying.remove(&slot) {
                (chosen.role, chosen.rule, chosen.joined) = (role, Rule::Opener, true);
            } else if let Some((member, kind)) = decision.reading[at].and_then(|member| Some((member, member_of(member).kind?))) {
                (chosen.role, chosen.rule, chosen.unit) = (unit_role(kind), Rule::Opener, Some((member.0, kind)));
                if let Some(text) = member_of(member).text {
                    carrying.insert(text, chosen.role);
                }
            } else {
                chosen.role = if fences.at(at).quotation != 0 { Role::Quote } else { Role::Body };
                if chosen.rule == Rule::None {
                    chosen.rule = Rule::Body;
                }
            }
        }
        chosen.starts = starts.contains(&slot);
        // Its block: a line label, folio or furniture keeps its own.
        let document = usize::from(fences.at(at).document);
        let front = !opened[document] && chosen.unit.is_none();
        opened[document] |= chosen.unit.is_some();
        blocks[at] = if matches!(chosen.role, Role::LineLabel | Role::PageLabel | Role::Furniture) {
            slots.line(pages, slot).block_index
        } else {
            block += usize::from(prior.is_none_or(|(before, front_before)| {
                paragraphs.starts(pages, slots, evidence, fences, (before, &decision.moves[fences.place[before.index()] as usize], front_before), (slot, &chosen, front))
            }));
            block
        };
        prior = Some((slot, front));
        // Its unit.
        let zone = zone_of(chosen.role, fences.edge[at], NoteMode::Footnote);
        let key = fences.ctx[at];
        if matches!(zone, LineZone::Body | LineZone::Heading | LineZone::MarginalNote) {
            let goes_on = grouped.is_some_and(|(before, held, context)| {
                before == zone && context == key && if transcript[page] && zone == LineZone::Body { !opens_turn(slots.line(pages, slot)) } else { held == blocks[at] }
            });
            let first_met = met.insert(key);
            let listed = chosen.role == Role::Contents || fences.contents_pages[page];
            let tail = tails.get(&key);
            let carried = !goes_on && zone == LineZone::Body && fences.continues[at].is_some() && tail.is_some_and(|tail| {
                first_met && tail.body && tail.page + 1 == page
                    || decision.moves[tail.open].member.is_some() && chosen.role == Role::Body && !listed
            });
            let contents = if goes_on || carried { tail.is_some_and(|tail| tail.contents) } else { listed };
            let member = decision.reading[at].or_else(|| marked.get(&slot).and_then(|members| members.first().copied()))
                .filter(|&(run, _)| graph::opens_units(chosen.role, chosen.rule, &runs.runs[run as usize]));
            let kind = member.and_then(|member| member_of(member).kind);
            if !(goes_on || carried) || kind.is_some() && !contents {
                chosen.opens = true;
                chosen.region = region_of(chosen.role, kind);
                chosen.reading = None;
                if contents && !matches!(chosen.region, HEADING | TITLE | MARGINAL_NOTE) {
                    chosen.region = CONTENTS;
                } else if let Some(member) = member.filter(|_| kind.is_some()) {
                    let mark = |other: (u32, u32)| member_of(other).marker[0].offset;
                    let mut further: Vec<(u32, u32)> = marked.get(&slot).into_iter().flatten().copied()
                        .filter(|&other| other != member && mark(other) > mark(member) && graph::opens_units(chosen.role, chosen.rule, &runs.runs[other.0 as usize])).collect();
                    further.sort_by_key(|&other| mark(other));
                    further.dedup_by_key(|other| mark(*other));
                    (chosen.reading, chosen.member, chosen.within) = (Some(reading_of(view, runs, member)), Some(member), further);
                }
                let opened = Tail { open: k, region: chosen.region, page, body: zone == LineZone::Body, contents };
                match tails.get_mut(&key).filter(|_| goes_on || carried) {
                    Some(tail) => (tail.open, tail.region, tail.page) = (k, chosen.region, page),
                    None => {
                        tails.insert(key, opened);
                    }
                }
            } else {
                let tail = tails.get_mut(&key).expect("a region going on has its context's tail");
                tail.page = page;
                tail.body |= carried;
                chosen = Move { role: chosen.role, rule: chosen.rule, joined: chosen.joined, unit: chosen.unit, starts: chosen.starts, resumes: Some(tail.open as u32), ..Move::plain(tail.region, false) };
            }
        } else if chosen.role != Role::Note {
            chosen.opens = true;
        }
        if zone != LineZone::LineLabel {
            grouped = Some((zone, blocks[at], key));
        }
        if chosen.opens && RANKED_BODY.contains(&chosen.region) {
            chosen.depth = state.stack_depth(view, &chosen).min(9) as u8;
        }
        state.apply(view, &chosen);
        decision.moves[k] = chosen;
    }
    blocks
}

/// The one decision on each line, over the notes read and the fences' roles (`fenced`: a fenced line keeps its own,
/// every other is open to the rules); and each line's block, by slot.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decide(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, runs: &Runs, geometry: &Geometry, outline: &[PdfOutlineEntry], notes: &Notes, views: &[LineView], fenced: (Vec<Role>, Vec<Rule>), traced: bool) -> (Decision, Vec<usize>) {
    let readings: &[Vec<MarkerReading>] = views.first().map_or(&[][..], |view| view.table.readings.as_slice());
    let mut decision = Decision::hear(pages, slots, evidence, fences, runs, notes, outline, geometry, readings, traced, fenced);
    let blocks = settle(pages, slots, evidence, fences, runs, views, &mut decision);
    decision.finish(pages, slots, evidence, fences, geometry, readings);
    (decision, blocks)
}
