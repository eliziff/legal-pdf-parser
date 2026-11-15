//! A document's enumerated headings read as one ladder of counters ("I.", "A.", "1."), each
//! enumerator in the families it can belong to.

use std::collections::BTreeMap;

const MAX_COUNTER_VALUE: u32 = 200;
/// Legal outlines run six deep: PART I, A., 1., (a), (i), (A).
const MAX_OUTLINE_DEPTH: usize = 6;
const FOOTNOTE_SUSPECT_MIN_VALUE: u32 = 15;

/// A heading number's reading on the ladder: its family as the numbering grammar names it (`MarkerReading`) and its value.
pub(crate) type EnumeratorInterpretation = (String, u32);

/// A heading's number in every family the numbering grammar reads it in ("C." a letter or a numeral), each a counter of
/// the ladder's size.
pub(crate) fn rungs(readings: &[crate::model::MarkerReading]) -> Vec<EnumeratorInterpretation> {
    readings.iter().filter(|reading| reading.heading() && (1..=MAX_COUNTER_VALUE).contains(&reading.value)).map(|reading| (reading.family.clone(), reading.value)).collect()
}

#[derive(Clone, Copy)]
struct Frame<'a> {
    family: &'a str,
    value: u32,
}

#[derive(Debug, Default)]
pub(crate) struct HeadingFamilyStats {
    pub(crate) count: usize,
    pub(crate) max_value: u32,
    pub(crate) level_votes: BTreeMap<usize, usize>,
    pub(crate) violations: usize,
    pub(crate) gaps: usize,
    pub(crate) footnote_suspect: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeadingAction {
    Increment,
    OpenLevel,
    IllegalRestart,
    JumpForward,
    OpenMidcounter,
    Violation,
}

#[derive(Debug)]
pub(crate) struct HeadingAssignment {
    pub(crate) level: Option<usize>,
    pub(crate) action: HeadingAction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum HeadingLadderStatus {
    NoEnumerators,
    ParsedClean,
    ParsedWithViolations,
    Unparseable,
}

#[derive(Debug)]
pub(crate) struct HeadingLadder<'a> {
    pub(crate) assignments: Vec<HeadingAssignment>,
    pub(crate) families: BTreeMap<&'a str, HeadingFamilyStats>,
    pub(crate) status: HeadingLadderStatus,
}

/// [`parse_heading_ladder`] over each run the whole sequence starts over at the top (an illegal
/// restart to level 1), as the readers split them: each run's first index in `candidates`, with
/// its own ladder.
pub(crate) fn parse_heading_runs<'a>(candidates: &[&'a [EnumeratorInterpretation]]) -> Vec<(usize, HeadingLadder<'a>)> {
    let restarts = parse_heading_ladder(candidates.iter().copied()).assignments.iter().enumerate()
        .filter(|(_, assignment)| assignment.action == HeadingAction::IllegalRestart && assignment.level == Some(1))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut starts = [0].into_iter().chain(restarts).chain([candidates.len()]).collect::<Vec<_>>();
    starts.dedup();
    starts.windows(2).map(|run| (run[0], parse_heading_ladder(candidates[run[0]..run[1]].iter().copied()))).collect()
}

pub(crate) fn parse_heading_ladder<'a>(candidates: impl IntoIterator<Item = &'a [EnumeratorInterpretation]>) -> HeadingLadder<'a> {
    let mut stack: Vec<Frame<'a>> = Vec::new();
    let mut assignments = Vec::new();
    let mut families: BTreeMap<&'a str, HeadingFamilyStats> = BTreeMap::new();
    let (mut violations, mut enumerator_count) = (0_usize, 0_usize);
    for choices in candidates {
        enumerator_count += 1;
        let place = |stack: &[Frame], family: &str| stack.iter().rposition(|frame| frame.family == family);
        // The counter goes on by one in a family it holds.
        let mut chosen = choices.iter().find_map(|(family, value)| {
            let index = place(&stack, family).filter(|&index| stack[index].value + 1 == *value)?;
            Some((family.as_str(), *value, HeadingAction::Increment, index))
        });
        // A family it does not hold opens a level at 1.
        if chosen.is_none() && stack.len() < MAX_OUTLINE_DEPTH {
            chosen = choices.iter().find(|(family, value)| *value == 1 && place(&stack, family).is_none()).map(|(family, value)| {
                stack.push(Frame { family, value: 1 });
                (family.as_str(), *value, HeadingAction::OpenLevel, usize::MAX)
            });
        }
        if chosen.is_none() {
            // A skipped number is the shortest jump that reads it: "C." after "A." is the
            // third letter, not the Roman hundred.
            let mut jump: Option<(&'a str, u32, usize, u32)> = None;
            for (family, value) in choices {
                let Some(index) = place(&stack, family) else { continue };
                if *value == 1 && jump.is_none() {
                    chosen = Some((family.as_str(), *value, HeadingAction::IllegalRestart, index));
                    break;
                }
                let gap = value.saturating_sub(stack[index].value);
                if gap > 1 && jump.is_none_or(|(.., shortest)| gap < shortest) {
                    jump = Some((family, *value, index, gap));
                }
            }
            if chosen.is_none() {
                chosen = jump.map(|(family, value, index, _)| (family, value, HeadingAction::JumpForward, index));
            }
        }
        // A family it does not hold opens a level mid-count.
        if chosen.is_none() && stack.len() < MAX_OUTLINE_DEPTH {
            chosen = choices.iter().find(|(family, ..)| place(&stack, family).is_none()).map(|(family, value)| {
                stack.push(Frame { family, value: *value });
                (family.as_str(), *value, HeadingAction::OpenMidcounter, usize::MAX)
            });
        }
        let Some((family, value, action, index)) = chosen else {
            let family = choices.first().map_or("unknown", |(family, ..)| family.as_str());
            violations += 1;
            families.entry(family).or_default().violations += 1;
            assignments.push(HeadingAssignment { level: None, action: HeadingAction::Violation });
            continue;
        };
        // A counter that goes on in a family the stack holds closes the levels under it; one that opens a level
        // stands on top.
        let level = if index == usize::MAX {
            stack.len()
        } else {
            stack.truncate(index + 1);
            stack[index].value = value;
            index + 1
        };
        let stats = families.entry(family).or_default();
        if action == HeadingAction::IllegalRestart {
            violations += 1;
            stats.violations += 1;
        } else if matches!(action, HeadingAction::JumpForward | HeadingAction::OpenMidcounter) {
            stats.gaps += 1;
        }
        stats.count += 1;
        stats.max_value = stats.max_value.max(value);
        *stats.level_votes.entry(level).or_default() += 1;
        assignments.push(HeadingAssignment { level: Some(level), action });
    }
    for (family, stats) in &mut families {
        // A count in arabic figures (".:1", "):1.1"): notes count so.
        stats.footnote_suspect = family.rsplit(':').next().is_some_and(|count| count.starts_with('1'))
            && stats.max_value >= FOOTNOTE_SUSPECT_MIN_VALUE
            && stats.level_votes.len() == 1;
    }
    let status = if enumerator_count == 0 {
        HeadingLadderStatus::NoEnumerators
    } else if violations == 0 {
        HeadingLadderStatus::ParsedClean
    } else if violations <= (enumerator_count / 5).max(1) {
        HeadingLadderStatus::ParsedWithViolations
    } else {
        HeadingLadderStatus::Unparseable
    };
    HeadingLadder { assignments, families, status }
}
