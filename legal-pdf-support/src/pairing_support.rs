use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

const MAX_COUNTER_VALUE: u32 = 200;
const MAX_OUTLINE_DEPTH: usize = 4;
const FOOTNOTE_SUSPECT_MIN_VALUE: u32 = 15;

pub use legal_citations::cues::{
    crossref_short_form, has_citation_cue as has_legal_citation_cue, has_citation_signal,
    heading_text_plausible, is_citation_continuation as is_legal_citation_continuation,
    is_citation_shaped_tail, is_counter_noun, reporter_abbreviation_regex,
};

pub fn protected_citation_spans(text: &str) -> Vec<(usize, usize)> {
    let document = legal_citations::text::ScalarText::new(text);
    legal_citations::cues::protected_spans(text)
        .into_iter()
        .map(|span| {
            (
                document.scalar_at_byte(span.start).expect("citation start"),
                document.scalar_at_byte(span.end).expect("citation end"),
            )
        })
        .collect()
}

fn roman_to_int(value: &str) -> Option<u32> {
    let mut total = 0;
    let mut prior = 0;
    for character in value.to_uppercase().chars().rev() {
        let current = match character {
            'I' => 1,
            'V' => 5,
            'X' => 10,
            'L' => 50,
            'C' => 100,
            'D' => 500,
            'M' => 1000,
            _ => return None,
        };
        if current < prior {
            total -= current;
        } else {
            total += current;
            prior = current;
        }
    }
    (total > 0 && total <= MAX_COUNTER_VALUE).then_some(total)
}

pub type EnumeratorInterpretation = (String, u32, usize);

pub fn enumerator_interpretations(value: &str, punct: &str) -> Vec<EnumeratorInterpretation> {
    static LEGAL_NUMERIC: OnceLock<Regex> = OnceLock::new();
    static ROMAN: OnceLock<Regex> = OnceLock::new();
    let mut result = Vec::new();
    if LEGAL_NUMERIC
        .get_or_init(|| Regex::new(r"^\d{1,2}(?:\.\d{1,2}){1,3}$").unwrap())
        .is_match(value)
    {
        let parts = value.split('.').collect::<Vec<_>>();
        if let Some(tail) = parts
            .last()
            .and_then(|part| part.parse::<u32>().ok())
            .filter(|number| (1..=MAX_COUNTER_VALUE).contains(number))
        {
            result.push((format!("legal_numeric_{punct}"), tail, parts.len()));
        }
        return result;
    }
    if value.chars().all(|character| character.is_ascii_digit()) {
        if let Some(number) = value
            .parse::<u32>()
            .ok()
            .filter(|number| (1..=MAX_COUNTER_VALUE).contains(number))
        {
            result.push((format!("numeric_{punct}"), number, 0));
        }
        return result;
    }
    let upper = value.to_uppercase();
    if value == upper
        && ROMAN
            .get_or_init(|| Regex::new(r"^[IVXLCDM]{2,7}$").unwrap())
            .is_match(&upper)
    {
        if let Some(number) = roman_to_int(value) {
            result.push((format!("roman_{punct}"), number, 0));
        }
        return result;
    }
    if value.chars().count() == 1 && value.chars().all(char::is_alphabetic) {
        let family = if value.chars().all(char::is_uppercase) {
            "upper_alpha"
        } else {
            "lower_alpha"
        };
        if value == upper && "IVXLCDM".contains(&upper) {
            if let Some(number) = roman_to_int(value) {
                result.push((format!("roman_{punct}"), number, 0));
            }
        }
        let character = upper.chars().next().unwrap();
        let number = u32::from(character).wrapping_sub(u32::from('A')) + 1;
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
pub struct HeadingAssignment<'a> {
    pub family: &'a str,
    pub value: Option<u32>,
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
    pub assignments: Vec<HeadingAssignment<'a>>,
    pub violations: usize,
    pub gaps: usize,
    pub families: BTreeMap<&'a str, HeadingFamilyStats>,
    pub status: HeadingLadderStatus,
}

pub fn parse_heading_ladder<'a>(
    candidates: impl IntoIterator<Item = &'a [EnumeratorInterpretation]>,
) -> HeadingLadder<'a> {
    let mut stack: Vec<Frame<'a>> = Vec::new();
    let mut assignments = Vec::new();
    let mut families: BTreeMap<&'a str, HeadingFamilyStats> = BTreeMap::new();
    let mut violations = 0_usize;
    let mut gaps = 0_usize;
    let mut enumerator_count = 0_usize;

    for choices in candidates {
        enumerator_count += 1;
        let mut chosen: Option<(&'a str, u32, HeadingAction, usize)> = None;
        for (family, value, _) in choices {
            if let Some(index) = stack.iter().rposition(|frame| frame.family == family) {
                if stack[index].value + 1 == *value {
                    stack.truncate(index + 1);
                    stack[index].value = *value;
                    chosen = Some((family, *value, HeadingAction::Increment, index + 1));
                    break;
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in choices {
                if *value == 1
                    && !stack.iter().any(|frame| frame.family == family)
                    && stack.len() < MAX_OUTLINE_DEPTH
                {
                    stack.push(Frame { family, value: 1 });
                    chosen = Some((family, *value, HeadingAction::OpenLevel, stack.len()));
                    break;
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in choices {
                if let Some(index) = stack.iter().rposition(|frame| frame.family == family) {
                    if *value == 1 {
                        stack.truncate(index + 1);
                        stack[index].value = 1;
                        chosen = Some((family, *value, HeadingAction::IllegalRestart, index + 1));
                        break;
                    }
                    if *value > stack[index].value + 1 {
                        stack.truncate(index + 1);
                        stack[index].value = *value;
                        chosen = Some((family, *value, HeadingAction::JumpForward, index + 1));
                        break;
                    }
                }
            }
        }
        if chosen.is_none() {
            for (family, value, _) in choices {
                if !stack.iter().any(|frame| frame.family == family)
                    && stack.len() < MAX_OUTLINE_DEPTH
                {
                    stack.push(Frame {
                        family,
                        value: *value,
                    });
                    chosen = Some((family, *value, HeadingAction::OpenMidcounter, stack.len()));
                    break;
                }
            }
        }
        let Some((family, value, action, level)) = chosen else {
            let (family, value) = choices
                .first()
                .map(|(family, value, _)| (family.as_str(), Some(*value)))
                .unwrap_or(("unknown", None));
            violations += 1;
            families.entry(family).or_default().violations += 1;
            assignments.push(HeadingAssignment {
                family,
                value,
                level: None,
                action: HeadingAction::Violation,
            });
            continue;
        };
        let family_stats = families.entry(family).or_default();
        if action == HeadingAction::IllegalRestart {
            violations += 1;
            family_stats.violations += 1;
        } else if matches!(
            action,
            HeadingAction::JumpForward | HeadingAction::OpenMidcounter
        ) {
            gaps += 1;
            family_stats.gaps += 1;
        }
        family_stats.count += 1;
        family_stats.max_value = family_stats.max_value.max(value);
        *family_stats.level_votes.entry(level).or_default() += 1;
        assignments.push(HeadingAssignment {
            family,
            value: Some(value),
            level: Some(level),
            action,
        });
    }

    for (family, stats) in &mut families {
        stats.footnote_suspect = (family.starts_with("numeric_")
            || family.starts_with("legal_numeric_"))
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
    HeadingLadder {
        assignments,
        violations,
        gaps,
        families,
        status,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heading_vectors_include_the_full_reporter_inventory() {
        let vectors = [
            ("Legal Principles", true),
            ("Background and Overview", true),
            ("Member's Right To Fair Treatment", true),
            ("D. Tax Cas. 1088.", false),
            (
                "Introduction 1 Ont Liquor Licence App Trib Dec 2 Analysis",
                false,
            ),
            ("Background 2026 Overview", true),
        ];
        for (value, expected) in vectors {
            assert_eq!(heading_text_plausible(value), expected, "{value}");
        }
    }

    #[test]
    fn us_statute_and_regulation_citations_are_not_outline_markers() {
        for citation in [
            "15 U.S.C. \u{00a7} 78j(b)",
            "17 C.F.R. \u{00a7} 240.10b-5(c)",
        ] {
            let text = format!("Under {citation}, liability follows.");
            let spans = protected_citation_spans(&text);
            assert!(
                spans.into_iter().any(|(start, end)| {
                    text.chars()
                        .skip(start)
                        .take(end - start)
                        .collect::<String>()
                        == citation
                }),
                "{text}"
            );
            assert!(has_legal_citation_cue(citation));
            assert!(!heading_text_plausible(citation));
        }
        assert!(protected_citation_spans("15. The defendant acted knowingly.").is_empty());
    }

    #[test]
    fn statute_citations_are_protected_from_footnote_pairing() {
        let text = "Criminal Code, RSC 1985, c C-46, s 7";
        let protected = protected_citation_spans(text)
            .into_iter()
            .map(|(start, end)| {
                text.chars()
                    .skip(start)
                    .take(end - start)
                    .collect::<String>()
            })
            .collect::<Vec<_>>();

        assert!(protected.iter().any(|span| span == "RSC 1985, c C-46"));
    }

    #[test]
    fn citation_signal_boundaries_survive_the_linear_regex_set() {
        for text in ["under s. 7", "see paras 12-14", "at p. 123", "supra note 4"] {
            assert!(has_citation_signal(text), "{text}");
        }
        for text in ["class 7", "xss. 12", "at p. 12345"] {
            assert!(!has_citation_signal(text), "{text}");
        }
    }

    #[test]
    fn mcgill_tenth_inventory_preserves_journal_punctuation() {
        let abbreviations: Vec<String> =
            serde_json::from_str(include_str!("../../data/mcgill_reporters.json")).unwrap();
        assert_eq!(abbreviations.len(), 2_110);
        for abbreviation in [
            "Alta L Rev",
            "Nat'l J Sexual Orientation L",
            "Actualités-Justice",
            "Res Communes: Vermont's J Env't",
            "J Energy, Nat'l Res & Envtl L",
        ] {
            let pattern = Regex::new(&format!(
                "^(?:{})$",
                reporter_abbreviation_regex(abbreviation)
            ))
            .unwrap();
            assert!(pattern.is_match(abbreviation), "{abbreviation}");
            assert!(
                has_citation_signal(&format!("1 {abbreviation} 2")),
                "{abbreviation}"
            );
        }
    }
}
