//! A document's enumerated headings read as one ladder of counters ("I.", "A.", "1."), each
//! enumerator in the families it can belong to.

use legal_pdf_core::regex;
use std::collections::BTreeMap;

const MAX_COUNTER_VALUE: u32 = 200;
/// Legal outlines run six deep: PART I, A., 1., (a), (i), (A).
const MAX_OUTLINE_DEPTH: usize = 6;
const FOOTNOTE_SUSPECT_MIN_VALUE: u32 = 15;

fn roman_to_int(value: &str) -> Option<u32> {
    legal_structure_model::layout::roman_value(value).filter(|number| *number <= MAX_COUNTER_VALUE)
}

pub type EnumeratorInterpretation = (String, u32, usize);

pub fn enumerator_interpretations(value: &str, punct: &str) -> Vec<EnumeratorInterpretation> {
    let counted = |number: &u32| (1..=MAX_COUNTER_VALUE).contains(number);
    if regex!(r"^\d{1,2}(?:\.\d{1,2}){1,3}$").is_match(value) {
        let parts = value.split('.').collect::<Vec<_>>();
        let tail = parts.last().and_then(|part| part.parse::<u32>().ok()).filter(counted);
        return tail.map(|tail| (format!("legal_numeric_{punct}"), tail, parts.len())).into_iter().collect();
    }
    if value.chars().all(|character| character.is_ascii_digit()) {
        return value.parse::<u32>().ok().filter(counted).map(|number| (format!("numeric_{punct}"), number, 0)).into_iter().collect();
    }
    let upper = value.to_uppercase();
    // Small numerals count a family of their own ("(i)", "(ii)" under "(a)").
    let roman = || roman_to_int(value).map(|number| (format!("{}_{punct}", if value == upper { "roman" } else { "lower_roman" }), number, 0));
    if (value == upper || value == value.to_lowercase()) && regex!(r"^[IVXLCDM]{2,7}$").is_match(&upper) {
        return roman().into_iter().collect();
    }
    let mut result = Vec::new();
    if value.chars().count() == 1 && value.chars().all(char::is_alphabetic) {
        let family = if value.chars().all(char::is_uppercase) { "upper_alpha" } else { "lower_alpha" };
        if "IVXLCDM".contains(&upper) {
            result.extend(roman());
        }
        let number = u32::from(upper.chars().next().unwrap()).wrapping_sub(u32::from('A')) + 1;
        result.push((format!("{family}_{punct}"), number, 0));
    }
    result
}

#[derive(Clone, Copy)]
struct Frame<'a> {
    family: &'a str,
    value: u32,
}

#[derive(Debug, Default)]
pub struct HeadingFamilyStats {
    pub count: usize,
    pub max_value: u32,
    pub level_votes: BTreeMap<usize, usize>,
    pub violations: usize,
    pub gaps: usize,
    pub footnote_suspect: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadingAction {
    Increment,
    OpenLevel,
    IllegalRestart,
    JumpForward,
    OpenMidcounter,
    Violation,
}

#[derive(Debug)]
pub struct HeadingAssignment {
    pub level: Option<usize>,
    pub action: HeadingAction,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadingLadderStatus {
    NoEnumerators,
    ParsedClean,
    ParsedWithViolations,
    Unparseable,
}

#[derive(Debug)]
pub struct HeadingLadder<'a> {
    pub assignments: Vec<HeadingAssignment>,
    pub families: BTreeMap<&'a str, HeadingFamilyStats>,
    pub status: HeadingLadderStatus,
}

/// [`parse_heading_ladder`] over each run the whole sequence starts over at the top (an illegal
/// restart to level 1), as the readers split them: each run's first index in `candidates`, with
/// its own ladder.
pub fn parse_heading_runs<'a>(candidates: &[&'a [EnumeratorInterpretation]]) -> Vec<(usize, HeadingLadder<'a>)> {
    let restarts = parse_heading_ladder(candidates.iter().copied()).assignments.iter().enumerate()
        .filter(|(_, assignment)| assignment.action == HeadingAction::IllegalRestart && assignment.level == Some(1))
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let mut starts = [0].into_iter().chain(restarts).chain([candidates.len()]).collect::<Vec<_>>();
    starts.dedup();
    starts.windows(2).map(|run| (run[0], parse_heading_ladder(candidates[run[0]..run[1]].iter().copied()))).collect()
}

pub fn parse_heading_ladder<'a>(candidates: impl IntoIterator<Item = &'a [EnumeratorInterpretation]>) -> HeadingLadder<'a> {
    let mut stack: Vec<Frame<'a>> = Vec::new();
    let mut assignments = Vec::new();
    let mut families: BTreeMap<&'a str, HeadingFamilyStats> = BTreeMap::new();
    let (mut violations, mut enumerator_count) = (0_usize, 0_usize);
    for choices in candidates {
        enumerator_count += 1;
        let place = |stack: &[Frame], family: &str| stack.iter().rposition(|frame| frame.family == family);
        // The counter goes on by one in a family it holds.
        let mut chosen = choices.iter().find_map(|(family, value, _)| {
            let index = place(&stack, family).filter(|&index| stack[index].value + 1 == *value)?;
            Some((family.as_str(), *value, HeadingAction::Increment, index))
        });
        // A family it does not hold opens a level at 1.
        if chosen.is_none() && stack.len() < MAX_OUTLINE_DEPTH {
            chosen = choices.iter().find(|(family, value, _)| *value == 1 && place(&stack, family).is_none()).map(|(family, value, _)| {
                stack.push(Frame { family, value: 1 });
                (family.as_str(), *value, HeadingAction::OpenLevel, usize::MAX)
            });
        }
        if chosen.is_none() {
            // A skipped number is the shortest jump that reads it: "C." after "A." is the
            // third letter, not the Roman hundred.
            let mut jump: Option<(&'a str, u32, usize, u32)> = None;
            for (family, value, _) in choices {
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
            chosen = choices.iter().find(|(family, ..)| place(&stack, family).is_none()).map(|(family, value, _)| {
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
        stats.footnote_suspect = (family.starts_with("numeric_") || family.starts_with("legal_numeric_"))
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
