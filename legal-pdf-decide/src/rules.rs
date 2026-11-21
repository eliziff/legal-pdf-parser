//! What each line is, decided once. The fences stand first; then, in this order of precedence:
//! the note band (a transcript's lines beside their number column never notes), the headings a
//! transcript's words and an enactment's history set; the runs' quotations, contents of titles
//! alone, counted markers opening their units, divisions, an instrument's captions and heading
//! ladders; a statute's marginal notes over its openers; the lines the document's bookmarks and
//! contents list name; the front matter and the lines that go on with the line before, which are
//! no headings; the residual weighing of the lines still open (`residual`); the title; and the
//! levels (`levels`). A rule fills only an open line: where two strong rules disagree on one, the
//! first keeps it. Each line keeps the rule that decided it, and each marker the unit it opens.

use crate::model::decide::Heard;
use crate::model::decide::Decision;
use crate::model::decide::Sequences;
use super::levels::{anchors, heading_levels};
use crate::model::decide::Notes;
use super::residual::weigh;
use crate::model::fences::{Continuation, Fences};
use crate::model::Geometry;
use crate::model::machine::machine::Move;
use crate::model::regions::TITLE;
use crate::model::MarkerReading;
use crate::model::{
    Evidence, Member, Role, Rule, Run, Runs, Slot, Slots, SourceTag, LOOK_BOLD, WORD_AUTHOR,
    WORD_COURT, WORD_DATE, WORD_DIVISION, WORD_EXAMINATION, WORD_FIELD, WORD_HEARING_PART,
    WORD_HISTORY, WORD_INDEX_ENTRY, WORD_INDEX_LETTER, WORD_JURAT, WORD_LABEL, WORD_PARTY,
    WORD_SIGNATURE, WORD_TRANSCRIPT_PART, WORD_WITNESS,
};
use crate::text::heading_ladder::{
    parse_heading_runs, rungs, EnumeratorInterpretation, HeadingLadderStatus,
};
use crate::text::text::all_caps;
use legal_pdf_core::model::{sentence_ended, Page, PdfOutlineEntry};
use legal_structure_model::{CandidateGrammar, CandidateObservationV2, UnitRole};







/// The one reading of each line's opening marker that may open its unit, as (run, member) indexes,
/// by slot: of the counted, unquoted markers on the line that resolve to a unit, the one standing
/// first on it; where runs read that one marker, a count of numbers's before the provision
/// ladder's and the ladder's before a list's, wherever each run starts (the engine has already
/// read an order's count inside a numbered paragraph as a list; a factum's "PART I" ladder,
/// starting above its first paragraph, takes no paragraph's number from the count).
fn openings(runs: &Runs, count: usize) -> Vec<Option<(u32, u32)>> {
    let mut chosen: Vec<Option<(u32, u32)>> = vec![None; count];
    let rank = |(r, k): (u32, u32)| {
        (
            runs.runs[r as usize].members[k as usize].marker[0].offset,
            grammar_rank(runs.runs[r as usize].grammar),
        )
    };
    for (r, run) in runs.runs.iter().enumerate() {
        for (k, member) in run
            .members
            .iter()
            .enumerate()
            .filter(|(_, member)| member.counted && member.kind.is_some() && !quoted(member))
        {
            let (at, reading) = (member.slot.index(), (r as u32, k as u32));
            if chosen[at].is_none_or(|chosen| rank(reading) < rank(chosen)) {
                chosen[at] = Some(reading);
            }
        }
    }
    chosen
}

/// Body text, a list's item or a quotation: what the residual reads as prose.
pub fn body_like(role: Role) -> bool {
    matches!(role, Role::Open | Role::Body | Role::ListItem | Role::Quote)
}

/// The share of a line's text its citations cover.
fn cited_share(evidence: &Evidence, slot: Slot, length: usize) -> f64 {
    let mut ranges: Vec<(u32, u32)> = evidence
        .citations_of(slot)
        .iter()
        .map(|cited| cited.range)
        .collect();
    ranges.sort_unstable();
    let (covered, _) = ranges
        .into_iter()
        .fold((0, 0), |(covered, reach), (start, end)| {
            let start = start.max(reach);
            if end > start {
                (covered + end - start, end)
            } else {
                (covered, reach)
            }
        });
    if length == 0 {
        0.0
    } else {
        f64::from(covered) / length as f64
    }
}

/// A line opening a provision's subsection or a clause's term: "(3) For the purposes", "4 (1)
/// Every", "(a) the".
fn subsection(text: &str) -> bool {
    legal_pdf_core::regex!(r"^\s*(?:\d{1,4}(?:\.\d{1,3})*\s*)?\((?:\d{1,3}|[a-z]{1,4})\)\s+\S")
        .is_match(text)
}

/// A colon closing a lead-in on a small joining word: "ORDERS THAT:", "PURPOSE OF:", "APPEARING on:".
fn joining_colon(text: &str) -> bool {
    text.strip_suffix(':')
        .and_then(|before| before.split_whitespace().next_back())
        .is_some_and(|word| {
            matches!(
                word.to_lowercase().as_str(),
                "that"
                    | "of"
                    | "on"
                    | "to"
                    | "for"
                    | "by"
                    | "and"
                    | "or"
                    | "the"
                    | "as"
                    | "at"
                    | "in"
                    | "with"
                    | "from"
                    | "under"
            )
        })
}

/// A rule's reading of a line no fence took, for the one decision on it; the line's first reading stands for the
/// rules that ask what is around them. Whether the line was open to it.
/// A rule's reading of a line no fence took, for the one decision on it; the line's first reading stands for the
/// rules that ask what is around them. Whether the line was open to it.
pub fn fill(decision: &mut Decision, slot: Slot, role: Role, rule: Rule) -> bool {
    if !decision.free[slot.index()] {
        return false;
    }
    decision.heard.push(Heard {
        slot,
        role,
        rule,
        weight: 0.0,
        joined: false,
    });
    let line = decision.line_mut(slot.index());
    if line.role == Role::Open {
        (line.role, line.rule) = (role, rule);
    }
    true
}

pub fn open(decision: &Decision, slot: Slot) -> bool {
    decision.free[slot.index()]
}

/// The reading just heard goes on with the line above it, or does not.
/// The reading just heard goes on with the line above it, or does not.
pub fn heard_joined(decision: &mut Decision, joined: bool) {
    if let Some(last) = decision.heard.last_mut() {
        last.joined = joined;
    }
}

/// The residual's reading of a line, at its score.
/// The residual's reading of a line, at its score.
pub fn hear_weighed(decision: &mut Decision, slot: Slot, heading: bool, weight: f32, joined: bool) {
    let role = if heading { Role::Heading } else { Role::Open };
    decision.heard.push(Heard {
        slot,
        role,
        rule: Rule::Residual,
        weight,
        joined,
    });
}

/// The rules heard, none deciding: every reading each gives a line no fence took, in `heard`, for the one decision on
/// each line (`tier0::settle`); `finish` follows it. `readings`: each line's marker readings, by place.
#[allow(clippy::too_many_arguments)]
pub fn hear(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    notes: &Notes,
    outline: &[PdfOutlineEntry],
    geometry: &Geometry,
    readings: &[Vec<MarkerReading>],
    traced: bool,
    (role, rule): (Vec<Role>, Vec<Rule>),
) -> Decision {
    let count = slots.len();
    let free = role.iter().map(|role| *role == Role::Open).collect();
    // Each line's move, in reading order, holds its fenced role from the first.
    let mut moves = vec![Move::plain(0, false); count];
    for (at, (role, rule)) in role.into_iter().zip(rule).enumerate() {
        moves[fences.place[at] as usize] = Move {
            role,
            rule,
            ..Move::plain(0, false)
        };
    }
    let mut decision = Decision {
        moves,
        place: fences.place.clone(),
        reading: openings(runs, count),
        given: vec![0; count],
        anchor: vec![0; count],
        free,
        ..Decision::default()
    };
    let text = |slot: Slot| slots.line(pages, slot).text.trim();
    let mut sequences = Sequences {
        lines: vec![Vec::new(); fences.contexts.len()],
        at: vec![u32::MAX; count],
    };
    // Each document a transcript of whose witnesses are named.
    let mut witnesses = vec![false; fences.transcripts.len()];
    let mut examinations = Vec::new();
    // Rules that only keep a line from the headings, applied once the anchors stand.
    let mut deferred: Vec<(Slot, Rule)> = Vec::new();
    for &slot in &fences.order {
        let at = slot.index();
        let words = evidence.words[at];
        let beside_numbers = fences.numbered_row[at];
        let transcript = fences.transcript(slot);
        let previous = if matches!(
            decision.line(at).role,
            Role::LineLabel | Role::PageLabel | Role::Furniture | Role::Contents
        ) {
            None
        } else {
            let lines = &mut sequences.lines[usize::from(fences.ctx[at])];
            let previous = lines.last().copied();
            sequences.at[at] = lines.len() as u32;
            lines.push(slot);
            previous
        };
        let line = text(slot);
        if transcript && words & WORD_WITNESS != 0 {
            witnesses[usize::from(fences.at(at).document)] = true;
        }
        if notes.running_head[at] {
            crate::find::fill(&mut decision, slot, Role::Furniture, Rule::EndnoteHead);
        }
        if notes.heading[at] {
            crate::find::fill(&mut decision, slot, Role::Heading, Rule::NotesHeading);
        } else if notes.note[at].is_some() {
            if !beside_numbers {
                crate::find::fill(&mut decision, slot, Role::Note, Rule::NoteBand);
            }
        }
        if !crate::find::open(&decision, slot) {
            continue;
        }
        // An enactment's history after a dash heads its provision.
        let dashed = line
            .strip_prefix(['\u{2014}', '\u{2013}', '-'])
            .map(str::trim_start);
        if dashed.is_some_and(|rest| {
            legal_structure_lines::statute::history_note(rest)
                || cited_share(evidence, slot, slots.line(pages, slot).text.len()) >= 0.8
        }) {
            crate::find::fill(&mut decision, slot, Role::Heading, Rule::HistoryHeading);
            continue;
        }
        let opens = !line.starts_with(char::is_lowercase) && fences.continues[at].is_none();
        if transcript && words & WORD_INDEX_ENTRY == 0 && opens {
            if words & (WORD_WITNESS | WORD_HEARING_PART | WORD_TRANSCRIPT_PART) != 0 {
                crate::find::fill(&mut decision, slot, Role::Heading, Rule::TranscriptPart);
                decision.given[at] = 1;
                continue;
            }
            if words & WORD_EXAMINATION != 0 {
                examinations.push(slot);
                continue;
            }
        }
        if transcript && words & WORD_INDEX_LETTER != 0 && opens {
            crate::find::fill(&mut decision, slot, Role::Heading, Rule::TranscriptPart);
            decision.given[at] = 2;
            continue;
        }
        // A transcript's line going on from the one before is no heading; a question its number
        // opens ("11 Q. Maybe jot in huge letters" after "BY MR. RUBY:") opens its unit.
        if words & WORD_HISTORY != 0 {
            deferred.push((slot, Rule::HistoryLine));
        } else if beside_numbers
            && (line.starts_with(char::is_lowercase)
                || previous.is_some_and(|previous| !sentence_ended(text(previous))))
        {
            if decision.reading[at].is_none() {
                deferred.push((slot, Rule::TranscriptLine));
            }
        } else if !previous.is_some_and(|prior| evidence.beside[prior.index()] == Some(slot))
            && front_matter(pages, slots, evidence, fences, slot)
        {
            // A word such as "DEFENDANT" beside its detached "II." belongs to that numbered
            // unit. It is not a free-standing party label; the residual reads the pair.
            deferred.push((slot, Rule::FrontMatter));
        }
    }
    for slot in examinations {
        if crate::find::fill(&mut decision, slot, Role::Heading, Rule::TranscriptPart) {
            decision.given[slot.index()] =
                if witnesses[usize::from(fences.at(slot.index()).document)] {
                    2
                } else {
                    1
                };
        }
    }
    crate::find::read_runs(&mut decision, pages, slots, evidence, fences, runs, &sequences);
    crate::find::marginal_notes(&mut decision, pages, slots, evidence, fences, runs, &sequences);
    // Heard, the outline names a line a marker opens as well as any other.
    let named: Vec<Role> = (0..count)
        .map(|at| {
            if matches!(decision.line(at).rule, Rule::Opener | Rule::Caption) {
                Role::Open
            } else {
                decision.line(at).role
            }
        })
        .collect();
    let anchoring = anchors(
        pages,
        slots,
        evidence,
        fences,
        geometry,
        &named,
        outline,
        &fences.contents_entries,
    );
    for (index, anchor) in anchoring.anchors.iter().enumerate() {
        for (k, &slot) in anchor.lines.iter().enumerate() {
            let spoke = crate::find::fill(&mut decision, slot, Role::Heading, Rule::Anchor);
            decision.anchor[slot.index()] = index as u32 + 1;
            decision.given[slot.index()] = anchor.level.min(u16::from(u8::MAX)) as u8;
            decision.line_mut(slot.index()).joined |= k > 0;
            if spoke {
                crate::find::heard_joined(&mut decision, k > 0);
            }
        }
    }
    decision.anchors = anchoring.anchors;
    for (slot, rule) in deferred {
        crate::find::fill(&mut decision, slot, Role::Body, rule);
    }
    // Bookmarks naming every heading leave none of the lines after their opening they do not name.
    let unnamed = |slot: Slot| {
        evidence.outline_complete
            && anchoring
                .opened
                .iter()
                .any(|&(document, translation, place)| {
                    document == fences.at(slot.index()).document
                        && translation == fences.at(slot.index()).translation
                        && place <= fences.place[slot.index()]
                })
    };
    weigh(
        pages,
        slots,
        evidence,
        fences,
        &sequences,
        geometry,
        readings,
        &unnamed,
        &mut decision,
        traced,
    );
    decision.sequences = sequences;
    decision
}

/// What needs the whole document once the walk has decided its lines: the title among the headings, and the
/// headings' levels (`readings`: each line's marker readings, by place).
/// What needs the whole document once the walk has decided its lines: the title among the headings, and the
/// headings' levels (`readings`: each line's marker readings, by place).
pub fn finish(decision: &mut Decision, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    geometry: &Geometry,
    readings: &[Vec<MarkerReading>],
) {
    let sequences = std::mem::take(&mut decision.sequences);
    crate::find::title(decision, pages, slots, evidence, fences, &sequences, geometry);
    heading_levels(
        pages, slots, evidence, fences, &sequences, geometry, readings, decision,
    );
    decision.sequences = sequences;
}

/// The rules each run settles: a quotation's numbers open nothing; three or more titles alone,
/// one under the next, are contents; a counted marker carrying prose opens its unit, by its
/// line's one reading (`readings`); one before a title alone heads its division where a
/// division's word opens it, and opens its
/// clause with the caption as its first line in an instrument, or over its clause's terms
/// ("(a)", "(3)"); outside an instrument, a count of titles alone between the text's paragraphs
/// is a heading ladder.
/// The rules each run settles: a quotation's numbers open nothing; three or more titles alone,
/// one under the next, are contents; a counted marker carrying prose opens its unit, by its
/// line's one reading (`readings`); one before a title alone heads its division where a
/// division's word opens it, and opens its
/// clause with the caption as its first line in an instrument, or over its clause's terms
/// ("(a)", "(3)"); outside an instrument, a count of titles alone between the text's paragraphs
/// is a heading ladder.
pub fn read_runs(decision: &mut Decision, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    sequences: &Sequences,
) {
    let place = |slot: Slot| sequences.at[slot.index()];
    // The last place of a member's own lines: its line, the line carrying its text, its text's
    // place.
    let end = |member: &Member| {
        [Some(member.slot), member.text, Some(member.content.slot)]
            .into_iter()
            .flatten()
            .map(place)
            .filter(|at| *at != u32::MAX)
            .max()
            .unwrap_or(u32::MAX)
    };
    for run in &runs.runs {
        for member in run.members.iter().filter(|member| quoted(member)) {
            crate::find::fill(decision, member.slot, Role::Quote, Rule::QuotedNumber);
        }
    }
    // TEMP lever "sections-open" (remove before landing). In an instrument a section's marker opens its unit whatever
    // its line carries (its number alone before its text, a caption's few words, "18 [Repealed]"): the count has named
    // it, and titles alone one under the next that are sections are no contents list.
    let sections_open = evidence.instrument
        && legal_structure_model::lever("sections-open");
    let section = |member: &Member| {
        sections_open && member.kind == Some(UnitRole::Section) && member.locator_kind.as_deref() == Some("section")
    };
    // (The lines a section's marker reads: another run's number on such a line, the count of numbers' "15" beside the
    // ladder's, is no title alone either.)
    let section_lines: std::collections::HashSet<usize> = runs
        .runs
        .iter()
        .flat_map(|run| &run.members)
        .filter(|member| section(member) && member.counted && !quoted(member))
        .map(|member| member.slot.index())
        .collect();
    // Titles alone, each directly over the next (a division's line between them aside), none of
    // three words or more (a statute's "118 [Repealed, 2014, c. 38, s. 17]" is its provision); a
    // transcript's line beside its number column is none.
    for run in &runs.runs {
        let lines = &sequences.lines[usize::from(run.ctx)];
        let alone = |member: &Member| {
            member.counted
                && member.title_shaped
                && !member.carries_prose
                && !member.worded
                && !section_lines.contains(&member.slot.index())
                && !quoted(member)
                && crate::find::open(decision, member.slot)
                && !fences.numbered_row[member.slot.index()]
        };
        let directly = |before: &Member, after: &Member| {
            let (from, to) = (end(before), place(after.slot));
            from != u32::MAX
                && to != u32::MAX
                && (from + 1..to)
                    .all(|at| evidence.words[lines[at as usize].index()] & WORD_DIVISION != 0)
        };
        let mut stretches: Vec<(usize, usize)> = Vec::new();
        let mut start = 0;
        for k in 0..=run.members.len() {
            let goes_on = k < run.members.len()
                && alone(&run.members[k])
                && (k == start || directly(&run.members[k - 1], &run.members[k]));
            if goes_on {
                continue;
            }
            if k - start >= 3 {
                stretches.push((start, k));
            }
            start = if k < run.members.len() && alone(&run.members[k]) {
                k
            } else {
                k + 1
            };
        }
        for (first, last) in stretches {
            for k in first..last {
                let member = &run.members[k];
                for slot in [Some(member.slot), member.text].into_iter().flatten() {
                    crate::find::fill(decision, slot, Role::Contents, Rule::TitleOnly);
                }
                if k + 1 < last {
                    for at in end(member) + 1..place(run.members[k + 1].slot) {
                        crate::find::fill(decision, lines[at as usize], Role::Contents, Rule::TitleOnly);
                    }
                }
            }
        }
    }
    for (r, run) in runs.runs.iter().enumerate() {
        let lines = &sequences.lines[usize::from(run.ctx)];
        for (k, member) in run.members.iter().enumerate() {
            let at = member.slot.index();
            if decision.reading[at] != Some((r as u32, k as u32)) {
                continue;
            }
            let Some(kind) = member.kind else { continue };
            if member.carries_prose || section(member) {
                let role = unit_role(kind);
                if crate::find::fill(decision, member.slot, role, Rule::Opener) {
                    decision.line_mut(at).unit = Some((r as u32, kind));
                    crate::find::join(decision, member.text, role, Rule::Opener);
                }
            } else if member.title_shaped && crate::find::open(decision, member.slot) {
                if evidence.words[at] & WORD_DIVISION != 0 {
                    crate::find::fill(decision, member.slot, Role::Heading, Rule::DivisionHeading);
                    crate::find::join(decision, member.text, Role::Heading, Rule::DivisionHeading);
                    continue;
                }
                // A caption opens its clause where text, not another title, follows it: in an
                // instrument, or over its clause's terms.
                let from = end(member);
                let to = run
                    .members
                    .get(k + 1)
                    .map_or(from.saturating_add(2), |next| place(next.slot));
                let followed = from != u32::MAX
                    && (from + 1..to.max(from + 1))
                        .take_while(|at| (*at as usize) < lines.len())
                        .any(|at| !evidence.title_shaped[lines[at as usize].index()]);
                let next = (from != u32::MAX)
                    .then(|| lines.get(from as usize + 1).copied())
                    .flatten();
                let over_terms = next.is_some_and(|next| {
                    decision.line(next.index()).role != Role::Heading
                        && subsection(slots.line(pages, next).text.trim())
                });
                // The clause's text after its caption is the clause's, set apart from it where the
                // page parts them ("2.2 / Landlord's Work" over "The Landlord shall perform").
                if (evidence.instrument || over_terms)
                    && followed
                    && crate::find::fill(decision, member.slot, Role::Body, Rule::Caption)
                {
                    decision.line_mut(at).unit = Some((r as u32, kind));
                    crate::find::join(decision, member.text, Role::Body, Rule::Caption);
                }
            }
        }
        if !evidence.instrument {
            crate::find::ladder(decision, pages, slots, evidence, fences, run, sequences);
        }
    }
}

/// Gives a marker's text line, where it has one of its own, the marker's role, going on with it.
/// Gives a marker's text line, where it has one of its own, the marker's role, going on with it.
pub fn join(decision: &mut Decision, text: Option<Slot>, role: Role, rule: Rule) {
    if let Some(text) = text {
        let filled = crate::find::fill(decision, text, role, rule);
        decision.line_mut(text.index()).joined |= filled;
        if filled {
            crate::find::heard_joined(decision, true);
        }
    }
}

/// A run whose counted markers are all titles alone between the text's paragraphs: none ending
/// as a list's item does, none after a lead-in, each followed by a paragraph of its own, and
/// their numbers a ladder the heading grammar parses (`parse_heading_runs`), not a count of
/// notes' numbers.
/// A run whose counted markers are all titles alone between the text's paragraphs: none ending
/// as a list's item does, none after a lead-in, each followed by a paragraph of its own, and
/// their numbers a ladder the heading grammar parses (`parse_heading_runs`), not a count of
/// notes' numbers.
pub fn ladder(decision: &mut Decision, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    run: &Run,
    sequences: &Sequences,
) {
    let counted: Vec<&Member> = run
        .members
        .iter()
        .filter(|member| member.counted && !quoted(member))
        .collect();
    if counted.len() < 2 {
        return;
    }
    let readings: Vec<Vec<EnumeratorInterpretation>> = counted
        .iter()
        .map(|member| rungs(&evidence.readings[member.slot.index()]))
        .filter(|readings| !readings.is_empty())
        .collect();
    let readings: Vec<&[EnumeratorInterpretation]> =
        readings.iter().map(Vec::as_slice).collect();
    let parsed = parse_heading_runs(&readings)
        .into_iter()
        .all(|(_, ladder)| {
            ladder.status != HeadingLadderStatus::Unparseable
                && ladder
                    .families
                    .values()
                    .all(|family| !family.footnote_suspect)
        });
    if !parsed {
        return;
    }
    let text = |slot: Slot| slots.line(pages, slot).text.trim();
    let title_line = |member: &Member| member.text.unwrap_or(member.slot);
    let listed = |line: &str| {
        line.ends_with([',', ';'])
            || line
                .rsplit(char::is_whitespace)
                .next()
                .is_some_and(|word| matches!(word, "and" | "or"))
    };
    let shaped = counted.iter().all(|member| {
        let title = title_line(member);
        member.title_shaped
            && !member.carries_prose
            && crate::find::open(decision, member.slot)
            && !listed(text(title))
    });
    let lead_in = sequences
        .beside(fences, counted[0].slot, -1)
        .is_some_and(|before| {
            body_like(decision.line(before.index()).role)
                && (!sentence_ended(text(before)) || text(before).ends_with(':'))
        });
    let parted = counted.iter().all(|member| {
        sequences
            .beside(fences, title_line(member), 1)
            .is_none_or(|after| fences.continues[after.index()].is_none())
    });
    if !shaped || lead_in || !parted {
        return;
    }
    for member in counted {
        crate::find::fill(decision, member.slot, Role::Heading, Rule::Ladder);
        let title = title_line(member);
        crate::find::join(decision, 
            (title != member.slot).then_some(title),
            Role::Heading,
            Rule::Ladder,
        );
    }
}

/// A statute's marginal notes, in each reading context holding its spine (three or more
/// counted provisions, rooted and consecutive, opening prose): one or two short lines shaped
/// as a note directly over an opener (a subsection's too; its second line may go on in lower
/// case, "Endorsements" over "on writ"), or in the page's margin column level with it, are a
/// caption. A caption level with its opener's first line beside it, or over exactly one
/// opener of its depth before the next caption and set as most such captions of its look
/// (bold, size, centring) are, is the opener's marginal note; any other is left to the lines
/// still open. A caption's twin on its row in the other language's column reads as it does.
/// A statute's marginal notes, in each reading context holding its spine (three or more
/// counted provisions, rooted and consecutive, opening prose): one or two short lines shaped
/// as a note directly over an opener (a subsection's too; its second line may go on in lower
/// case, "Endorsements" over "on writ"), or in the page's margin column level with it, are a
/// caption. A caption level with its opener's first line beside it, or over exactly one
/// opener of its depth before the next caption and set as most such captions of its look
/// (bold, size, centring) are, is the opener's marginal note; any other is left to the lines
/// still open. A caption's twin on its row in the other language's column reads as it does.
pub fn marginal_notes(decision: &mut Decision, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    runs: &Runs,
    sequences: &Sequences,
) {
    let text = |slot: Slot| slots.line(pages, slot).text.trim();
    let mut decided: Vec<Slot> = Vec::new();
    for (ctx, lines) in sequences.lines.iter().enumerate() {
        if fences.contexts[ctx].quotation != 0 {
            continue;
        }
        // The context's openers in reading order: place, depth, slot. A consolidation's ladder
        // skips its repealed sections and inserts others ("19.02"), so it need not count on by one.
        let mut openers: Vec<(u32, u16, Slot)> = runs
            .runs
            .iter()
            .filter(|run| {
                usize::from(run.ctx) == ctx
                    && run.grammar == CandidateGrammar::Hierarchy
                    && run.rooted
            })
            .flat_map(|run| &run.members)
            .filter(|member| {
                member.counted
                    && member.carries_prose
                    && member.kind.is_some()
                    && decision.line(member.slot.index()).rule == Rule::Opener
            })
            .map(|member| {
                (
                    sequences.at[member.slot.index()],
                    if member.parent.is_none() {
                        0
                    } else {
                        member.level.max(1)
                    },
                    member.slot,
                )
            })
            .collect();
        if openers.iter().filter(|(_, depth, _)| *depth == 0).count() < 3 {
            continue;
        }
        openers.sort_unstable();
        let shaped = |slot: Slot| {
            let line = text(slot);
            let letters = line.chars().filter(|c| c.is_alphabetic()).count();
            let upper = line.chars().filter(|c| c.is_uppercase()).count();
            // (No reading on it yet, heard or not: a note's count is asked of the document's lines no rule took.)
            decision.line(slot.index()).role == Role::Open
                && legal_structure_lines::statute::marginal_note_shaped(line)
                && upper * 5 < letters * 4
                && evidence.words[slot.index()] & WORD_DIVISION == 0
                && !fences.numbered_row[slot.index()]
        };
        // A note's line wrapped under its first: open, short, going on in lower case.
        let wrapped = |slot: Slot| {
            decision.line(slot.index()).role == Role::Open
                && fences.continues[slot.index()] == Some(Continuation::Flow)
                && text(slot).starts_with(char::is_lowercase)
                && (1..=16).contains(&text(slot).split_whitespace().count())
                && !fences.numbered_row[slot.index()]
        };
        // Each caption: its lines, top first, and the opener under it.
        let mut captions: Vec<(Vec<Slot>, usize)> = Vec::new();
        for (k, &(at, _, opener)) in openers.iter().enumerate() {
            let page = slots.page(opener);
            let above = |step: u32| {
                at.checked_sub(step)
                    .map(|place| lines[place as usize])
                    .filter(|&slot| slots.page(slot) == page)
            };
            let mut caption: Vec<Slot> = Vec::new();
            if let Some(first) = above(1).filter(|&slot| shaped(slot)) {
                caption.push(first);
                if let Some(upper) = above(2)
                    .filter(|&slot| shaped(slot) && fences.continues[first.index()].is_some())
                {
                    caption.insert(0, upper);
                }
            } else if let Some(lower) = above(1).filter(|&slot| wrapped(slot)) {
                if let Some(upper) = above(2).filter(|&slot| shaped(slot)) {
                    caption = vec![upper, lower];
                }
            }
            if caption
                .first()
                .is_some_and(|first| fences.continues[first.index()].is_some())
            {
                caption.clear();
            }
            if caption.is_empty() {
                caption = crate::find::margin_caption(decision, pages, slots, fences, opener, &shaped);
            }
            if !caption.is_empty()
                && captions
                    .last()
                    .is_none_or(|(last, _)| last[0] != caption[0])
            {
                captions.push((caption, k));
            }
        }
        // Whether each caption governs one opener of its depth before the next caption, and
        // its look.
        let governs: Vec<bool> = (0..captions.len())
            .map(|c| {
                let (_, k) = captions[c];
                let (at, depth, _) = openers[k];
                let next = captions.get(c + 1).map_or(u32::MAX, |(lines, _)| {
                    sequences.at[lines[0].index()].min(openers[captions[c + 1].1].0)
                });
                openers
                    .iter()
                    .filter(|(place, other, _)| {
                        *place >= at && *place < next && *other <= depth
                    })
                    .count()
                    == 1
            })
            .collect();
        let look = |slot: Slot| {
            (
                evidence.look[slot.index()] & LOOK_BOLD != 0,
                evidence.half_points[slot.index()],
                centred(pages, slots, slot),
            )
        };
        let mut groups: Vec<((bool, u16, bool), usize, usize)> = Vec::new();
        for ((caption, _), &one) in captions.iter().zip(&governs) {
            let key = look(caption[0]);
            match groups.iter_mut().find(|(seen, _, _)| *seen == key) {
                Some((_, members, ones)) => {
                    *members += 1;
                    *ones += usize::from(one);
                }
                None => groups.push((key, 1, usize::from(one))),
            }
        }
        // A caption level with its opener's first line, beside it in a margin, is its note: a
        // heading stands over its provision, not beside it.
        let beside = |caption: Slot, opener: Slot| {
            let (note, line) = (slots.line(pages, caption), slots.line(pages, opener));
            let row = evidence.row(slots, caption);
            let level = row != u32::MAX && row == evidence.row(slots, opener)
                || (note.bbox[1] - line.bbox[1]).abs()
                    <= (line.bbox[3] - line.bbox[1]).max(1.0);
            // (TEMP lever "hanging-first-line", remove before landing: right of a number standing alone, a line the text
            // runs on from under the number, the next row starting at the number's own edge in lower case, is the
            // provision's first line set beside its number, "5(1) | Every chief executive must establish / procedures
            // to manage", no note.)
            let hanging = legal_structure_model::lever("hanging-first-line")
                && note.bbox[0] >= line.bbox[2]
                && !line.text.chars().any(char::is_alphabetic)
                && slots.on(slots.page(caption)).map(Slot).any(|below| {
                    let next = slots.line(pages, below);
                    let height = (note.bbox[3] - note.bbox[1]).max(1.0);
                    next.bbox[1] > note.bbox[1] + 0.5 * height
                        && next.bbox[1] <= note.bbox[3] + 1.2 * height
                        && (next.bbox[0] - line.bbox[0]).abs() <= height
                        && next.text.trim_start().starts_with(char::is_lowercase)
                });
            level && !hanging && (note.bbox[2] <= line.bbox[0] || note.bbox[0] >= line.bbox[2])
        };
        for ((caption, opener), &one) in captions.iter().zip(&governs) {
            let key = look(caption[0]);
            let share = groups
                .iter()
                .find(|(seen, _, _)| *seen == key)
                .map_or(0.0, |(_, members, ones)| *ones as f64 / *members as f64);
            if !(beside(caption[0], openers[*opener].2) || one && share >= 0.6) {
                continue;
            }
            for (k, &slot) in caption.iter().enumerate() {
                if crate::find::fill(decision, slot, Role::MarginalNote, Rule::MarginalNote) {
                    decision.line_mut(slot.index()).joined = k > 0;
                    crate::find::heard_joined(decision, k > 0);
                    decided.push(slot);
                }
            }
        }
    }
    // A note's twin in the other language's column, on its row.
    for slot in decided {
        let page = slots.page(slot);
        let row = evidence.row(slots, slot);
        if row == u32::MAX {
            continue;
        }
        for twin in slots.on(page).map(Slot) {
            if fences.at(twin.index()).translation != fences.at(slot.index()).translation
                && evidence.row(slots, twin) == row
                && crate::find::open(decision, twin)
                && legal_structure_lines::statute::marginal_note_shaped(text(twin))
            {
                crate::find::fill(decision, twin, Role::MarginalNote, Rule::MarginalNote);
            }
        }
    }
}

/// The caption set in the page's margin column level with an opener: the open lines of its
/// context on the column's side (the side holding fewer lines), the first shaped as a note and
/// level with the opener's first line, with the lines wrapped under it (three at most).
/// The caption set in the page's margin column level with an opener: the open lines of its
/// context on the column's side (the side holding fewer lines), the first shaped as a note and
/// level with the opener's first line, with the lines wrapped under it (three at most).
pub fn margin_caption(decision: &Decision, pages: &[Page],
    slots: &Slots,
    fences: &Fences,
    opener: Slot,
    shaped: &impl Fn(Slot) -> bool,
) -> Vec<Slot> {
    let page_slot = slots.page(opener);
    let columns = fences.columns[page_slot];
    if columns.kind != crate::model::ColumnKind::Margin {
        return Vec::new();
    }
    let left = columns.left_count < columns.right_count;
    let line = slots.line(pages, opener);
    let height = (line.bbox[3] - line.bbox[1]).max(1.0);
    let in_margin = |slot: Slot| {
        let other = slots.line(pages, slot);
        let middle = (other.bbox[0] + other.bbox[2]) / 2.0;
        crate::find::open(decision, slot)
            && fences.ctx[slot.index()] == fences.ctx[opener.index()]
            && (middle < columns.split_x) == left
    };
    let mut margin: Vec<Slot> = slots
        .on(page_slot)
        .map(Slot)
        .filter(|&slot| in_margin(slot))
        .collect();
    margin.sort_by(|a, b| {
        slots.line(pages, *a).bbox[1].total_cmp(&slots.line(pages, *b).bbox[1])
    });
    let Some(first) = margin.iter().position(|&slot| {
        (slots.line(pages, slot).bbox[1] - line.bbox[1]).abs() <= height && shaped(slot)
    }) else {
        return Vec::new();
    };
    let mut caption = vec![margin[first]];
    for &slot in &margin[first + 1..] {
        let (above, below) = (
            slots.line(pages, caption[caption.len() - 1]),
            slots.line(pages, slot),
        );
        if caption.len() == 3 || below.bbox[1] - above.bbox[3] > 0.5 * height {
            break;
        }
        caption.push(slot);
    }
    caption
}

/// The document's title, in each context outside a quotation: the first heading after its
/// court's header and its parties and before its body opens (its first full line of prose), a
/// layout model's title among them where no rule but the residual's read the line, where the model labels it so or
/// it is set as titles are (centred in capitals or larger type, or larger in mixed case). An anchored heading is
/// none. The title's unit is the title's.
/// The document's title, in each context outside a quotation: the first heading after its
/// court's header and its parties and before its body opens (its first full line of prose), a
/// layout model's title among them where no rule but the residual's read the line, where the model labels it so or
/// it is set as titles are (centred in capitals or larger type, or larger in mixed case). An anchored heading is
/// none. The title's unit is the title's.
pub fn title(decision: &mut Decision, pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    sequences: &Sequences,
    geometry: &Geometry,
) {
    for (ctx, lines) in sequences.lines.iter().enumerate() {
        if fences.contexts[ctx].quotation != 0 {
            continue;
        }
        let measure = |slot: Slot| &geometry.page(pages, slots.page(slot)).0;
        let opens = lines.iter().position(|&slot| {
            let line = slots.line(pages, slot);
            let measure = measure(slot);
            body_like(decision.line(slot.index()).role)
                && line.text.trim().chars().count() >= 60
                && line.bbox[2] - line.bbox[0] >= 0.85 * (measure.right - measure.left)
        });
        let before = &lines[..opens.unwrap_or(lines.len())];
        let parties = before
            .iter()
            .rposition(|&slot| evidence.words[slot.index()] & (WORD_PARTY | WORD_FIELD) != 0)
            .map_or(0, |at| at + 1);
        for (k, &slot) in before.iter().enumerate().skip(parties) {
            let at = slot.index();
            let cue = evidence.tag[at] == SourceTag::Title;
            let unsettled = matches!(decision.line(at).rule, Rule::Body | Rule::Residual);
            if decision.line(at).role != Role::Heading && !(cue && unsettled) {
                continue;
            }
            let line = slots.line(pages, slot);
            let measure = measure(slot);
            let centred = centred(pages, slots, slot)
                && line.bbox[2] - line.bbox[0] < 0.8 * (measure.right - measure.left);
            let caps = all_caps(line.text.trim());
            let larger = measure.size > 0.0
                && f64::from(evidence.half_points[at]) / 2.0 >= measure.size * 1.1;
            let set = cue
                || evidence.title_shaped[at]
                    && (centred && (caps || larger) || larger && !caps);
            if set && decision.line(at).rule != Rule::Anchor {
                let wrapped = before[k + 1..]
                    .iter()
                    .take_while(|next| {
                        decision.line(next.index()).joined
                            && decision.line(next.index()).role == Role::Heading
                    })
                    .count();
                // A title has no heading's level.
                for &title in &before[k..=k + wrapped] {
                    let line = decision.line_mut(title.index());
                    (line.role, line.rule, line.level, line.region) =
                        (Role::Title, Rule::Title, 0, TITLE);
                }
            }
            break;
        }
    }
}


/// A line centred on its page.
fn centred(pages: &[Page], slots: &Slots, slot: Slot) -> bool {
    let (line, width) = (
        slots.line(pages, slot),
        pages[slots.page(slot)].width.max(1.0),
    );
    (((line.bbox[0] + line.bbox[2]) / 2.0 - width / 2.0) / width).abs() < 0.03
}

/// The role of the line a marker opens a unit of its kind on.
pub fn unit_role(kind: UnitRole) -> Role {
    if kind == UnitRole::Item {
        Role::ListItem
    } else {
        Role::Body
    }
}

/// A quotation's own number: the quoted text's, none of the document's.
fn quoted(member: &Member) -> bool {
    member
        .observations
        .contains(&CandidateObservationV2::Quotation)
}

/// Front matter no heading is: a label (counsel, a party's role, contact details, "WHEREAS"), a
/// jurat, a date, a signature, the line naming who gives the reasons ("MARTLAND J.—",
/// "Fitzpatrick J.:"), a field's label with its blank, its value or another line on its row, a
/// lead-in's colon after a joining word, a court's name heading its document's first two pages or
/// standing over its parties (a party's role, a field or a label within the twelve lines after it).
fn front_matter(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    slot: Slot,
) -> bool {
    let words = evidence.words[slot.index()];
    let line = slots.line(pages, slot);
    let text = line.text.trim();
    let court = || {
        let document = fences.at(slot.index()).document;
        let opening = fences
            .order
            .iter()
            .find(|first| fences.at(first.index()).document == document)
            .map_or(0, |first| slots.page(*first));
        let place = fences.place[slot.index()] as usize;
        slots.page(slot) < opening + 2
            || fences.order[place + 1..].iter().take(12).any(|next| {
                evidence.words[next.index()] & (WORD_PARTY | WORD_FIELD | WORD_LABEL) != 0
            })
    };
    if words & (WORD_LABEL | WORD_PARTY | WORD_JURAT | WORD_DATE | WORD_SIGNATURE | WORD_AUTHOR)
        != 0
        || text
            .strip_suffix(':')
            .is_some_and(|name| legal_structure_lines::judgment::signature_line(name))
        || joining_colon(text)
        || words & WORD_COURT != 0 && court()
    {
        return true;
    }
    if words & WORD_FIELD == 0 && !crate::text::paragraphs::field_label(line) {
        return false;
    }
    let row = evidence.row(slots, slot);
    text.contains("___")
        || text
            .split_once(':')
            .is_some_and(|(_, value)| !value.trim().is_empty())
        || row != u32::MAX
            && slots.on(slots.page(slot)).map(Slot).any(|other| {
                other != slot
                    && evidence.row(slots, other) == row
                    && !slots.line(pages, other).text.trim().is_empty()
            })
}

/// The order runs read in at one place: a count of numbers, then a ladder, then a list.
fn grammar_rank(grammar: CandidateGrammar) -> u8 {
    match grammar {
        CandidateGrammar::Numeric => 0,
        CandidateGrammar::Hierarchy => 1,
        CandidateGrammar::Enumerator => 2,
    }
}
