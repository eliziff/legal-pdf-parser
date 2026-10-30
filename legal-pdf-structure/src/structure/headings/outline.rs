//! The document's outline, decided once for the whole document: which of its lines open its
//! divisions. A document marks its divisions its own way (a size, a weight, capitals, a place on
//! the line, a numbering family), and every line it marks so is one; so the decision is taken for
//! each way a document sets its short lines, a group, from what its lines show together, and each
//! line follows its group unless its own words or place say otherwise.
//!
//! A candidate is a short line in the body, or a few such lines set as one (a title wrapping), that
//! ends no sentence and continues no prose. A group is the candidates set alike: one face, size,
//! weight, slant, case and underline, one place on the line (flush, indented, centred in its
//! column), and one numbering family. What says a line opens a division: a reader took it for a
//! heading, the document's bookmarks or contents list name it, it is set apart from the body type,
//! it numbers a division, space stands over it, and prose or a numbered paragraph follows it. What
//! says it does not: it is a label (counsel, a party, an address), a speaker's turn, a date, it ends
//! on a colon, it runs long, or short lines follow it as a block's fields do. What a group shows
//! together: its numbered lines count on (4, 5, 6 or I, II, III, each from the one before or from
//! the start), it recurs over pages, and readers took most of it.
//!
//! Each line scores its own evidence; in a group of two or more it takes half its own and half the
//! group's, with the group's own signs added. A line a reader placed by the document's outline, or
//! the transcript rules raised, stays a heading; one ending a sentence, a label, a turn or a date
//! never is one.

use super::rules::{bare_number, enumerator, label, numbered, numbered_paragraph, turn_or_event};
use super::*;
use legal_pdf_core::{dominant_face, flag_share, Columns, PageMeasure, ITALIC};

/// A heading's lines, and the levels the outline gives the ones no reader levelled.
pub(crate) struct Outline {
    /// Every line of a candidate the outline took for a heading.
    pub headings: HashSet<String>,
    /// Every line of every candidate the outline weighed: the lines it decides.
    pub weighed: HashSet<String>,
    /// A level for each heading line, by its group's rank among the document's heading groups.
    pub levels: HashMap<String, usize>,
}

/// How a group sets its lines: face, half-points, bold, italic, capitals, underlined, its place on
/// the line (0 flush, 1 indented, 2 centred) and its numbering family.
type Look = (String, i64, bool, bool, bool, bool, u8, String);

struct Candidate<'a> {
    lines: Vec<&'a Line>,
    look: Look,
    /// Its own evidence, as log-odds of opening a division.
    own: f64,
    /// Placed by the document's bookmarks or contents, or raised by the transcript rules.
    placed: bool,
    /// Never a heading: a sentence's end, a label, a turn, a date.
    vetoed: bool,
    voted: bool,
    page: usize,
    /// Its numbering, every family it reads in: (family, ordinal).
    numbers: Vec<(String, u32)>,
    /// Set apart from the body's own type: another face, size or weight, capitals, a division's word.
    set_apart: bool,
}

/// The numbering family of an enumerator's value and punctuation: arabic, roman, letter, by case.
fn family(value: &str, punctuation: &str) -> String {
    let kind = if value.chars().all(|c| c.is_ascii_digit()) {
        "1"
    } else if value.chars().all(|c| c.is_ascii_uppercase()) {
        if value.len() == 1 { "A" } else { "I" }
    } else if value.len() == 1 {
        "a"
    } else {
        "i"
    };
    format!("{kind}{punctuation}")
}

fn underlined(line: &Line) -> bool {
    !line.words.is_empty() && line.words.iter().filter(|word| word.underlined).count() * 2 > line.words.len()
}

/// The share of steps between a group's numbered lines, in reading order, that count on: the next
/// value of a family the line before reads in, or the first value of one (a new run).
fn counts_on(members: &[&Candidate]) -> Option<f64> {
    let numbered: Vec<&&Candidate> = members.iter().filter(|candidate| !candidate.numbers.is_empty()).collect();
    if numbered.len() < 2 {
        return None;
    }
    let steps = numbered.windows(2).filter(|pair| {
        pair[1].numbers.iter().any(|(family, ordinal)| {
            *ordinal == 1 || pair[0].numbers.iter().any(|(before, value)| before == family && value + 1 == *ordinal)
        })
    }).count();
    Some(steps as f64 / (numbered.len() - 1) as f64)
}

/// The outline of one document: its body lines in reading order (`sequence`), each page's
/// measures, the lines the transcript rules raised.
pub(crate) fn decide(
    document: &[Page],
    measures: &HashMap<usize, PageMeasure>,
    sequence: &[&Line],
    primitives: &PdfPrimitiveEvidence,
    transcript: bool,
    raised: &HashSet<&str>,
) -> Outline {
    let pages: HashMap<usize, &Page> = document.iter().map(|page| (page.index, page)).collect();
    let columns: HashMap<usize, Columns> = document.iter().map(|page| (page.index, Columns::of(page))).collect();
    let slots: HashMap<&str, usize> = document.iter()
        .flat_map(|page| page.lines.iter().enumerate().map(|(slot, line)| (line.id.as_str(), slot)))
        .collect();
    // The document's body type: its commonest face and size, by characters.
    let mut type_chars: HashMap<(&str, i64), usize> = HashMap::new();
    for line in sequence {
        if let Some((face, size)) = dominant_face(line) {
            *type_chars.entry((face, (size * 2.0) as i64)).or_default() += line.text.trim().chars().count();
        }
    }
    let body = type_chars.into_iter().max_by(|a, b| a.1.cmp(&b.1).then_with(|| a.0.cmp(&b.0))).map(|(style, _)| style);
    // A transcript's pages of testimony: those holding a speaker's turn.
    let testimony: HashSet<usize> = sequence.iter()
        .filter(|line| legal_structure_model::transcript::speaker_label(line.text.trim()).is_some())
        .map(|line| line.page_index)
        .collect();
    let width = |line: &Line| {
        let measure = &measures[&line.page_index];
        (line.bbox[2] - line.bbox[0]) / (measure.right - measure.left).max(1.0)
    };
    let eligible = |line: &Line| {
        let text = line.text.trim();
        matches!(line.region_type.as_str(), "body" | "text" | "heading")
            && !crate::structure::layout_labelled(line)
            && heading_source_eligible(primitives, line)
            && !primitives.quotations.scopes.contains_key(&line.id)
            && (1..=160).contains(&text.chars().count())
            && !text.chars().next().is_some_and(char::is_lowercase)
            && !continues_prose(pages[&line.page_index], line)
    };
    // Prose: a line of the body running most of its measure, or long.
    let prose = |line: &Line| matches!(line.region_type.as_str(), "body" | "text") && (width(line) >= 0.7 || line.text.trim().chars().count() >= 90);

    let mut candidates: Vec<Candidate> = Vec::new();
    let mut k = 0;
    while k < sequence.len() {
        let first = sequence[k];
        if !eligible(first) {
            k += 1;
            continue;
        }
        let style = dominant_face(first).map(|(face, size)| (face.to_owned(), (size * 2.0) as i64));
        // A title wrapping onto the lines under it: same page and block, same type, short, no
        // number of its own, the line before ending no sentence.
        let mut end = k + 1;
        // A number printed on a line of its own ("III.", "12") is one with the line it numbers, the
        // heading's title or the paragraph's first line, whatever their types.
        if bare_number(first.text.trim()) && sequence.get(end).is_some_and(|next| next.page_index == first.page_index && eligible(next)) {
            end += 1;
        }
        while end < sequence.len() && end - k < 3 {
            let next = sequence[end];
            let prior = sequence[end - 1];
            if next.page_index != first.page_index || next.block_index != first.block_index || !eligible(next)
                || dominant_face(next).map(|(face, size)| (face.to_owned(), (size * 2.0) as i64)) != style
                || width(prior) >= 0.75 || width(next) >= 0.75 || numbered(next.text.trim())
                || sentence_ended(prior.text.trim()) || prior.text.trim().ends_with([':', ';', ','])
            {
                break;
            }
            end += 1;
        }
        let lines: Vec<&Line> = sequence[k..end].to_vec();
        // A line running its column's measure on into the next line of its block opens a paragraph,
        // a provision or a list's item: its text goes on, as a title's does not.
        let last = lines[lines.len() - 1];
        let (last_width, _) = columns[&last.page_index].place(last, slots[last.id.as_str()]);
        if last_width >= 0.85 && sequence.get(end).is_some_and(|next| next.page_index == last.page_index && next.block_index == last.block_index) {
            k = end;
            continue;
        }
        let text = lines.iter().map(|line| line.text.trim()).collect::<Vec<_>>().join(" ");
        let measure = &measures[&first.page_index];
        let (column_width, column_centre) = columns[&first.page_index].place(first, slots[first.id.as_str()]);
        let placement = if column_centre.abs() < 0.04 && column_width < 0.8 {
            2
        } else if (first.bbox[0] - measure.left) / (measure.right - measure.left).max(1.0) > 0.05 {
            1
        } else {
            0
        };
        let enumerated = enumerator(&text);
        let numbers: Vec<(String, u32)> = enumerated.as_ref().map(|(value, punctuation)| {
            enumerator_interpretations(value, punctuation).into_iter().map(|(family, ordinal, _)| (family, ordinal)).collect()
        }).unwrap_or_default();
        let bold = bold_char_share(first) >= 0.5;
        let italic = flag_share(first, ITALIC).unwrap_or(0.0) > 0.5;
        let caps = all_caps(&text);
        let lined = underlined(first);
        let size = style.as_ref().map_or(0, |(_, size)| *size);
        let larger = measure.size > 0.0 && dominant_face(first).is_some_and(|(_, size)| size >= measure.size * 1.1);
        let body_type = style.as_ref().map(|(face, size)| (face.as_str(), *size)) == body && !bold && !lined && !larger && !italic;
        let look: Look = (
            style.as_ref().map_or_else(String::new, |(face, _)| face.clone()), size, bold, italic, caps, lined, placement,
            enumerated.as_ref().map_or_else(String::new, |(value, punctuation)| family(value, punctuation)),
        );
        let next = sequence.get(end).copied();
        let after = sequence.get(end + 1).copied();
        let above = k.checked_sub(1).map(|j| sequence[j]).filter(|prior| prior.page_index == first.page_index)
            .map_or(3.0, |prior| (first.bbox[1] - prior.bbox[3]) / measure.height);
        let words = text.split_whitespace().count();
        let division_word = enumerated.as_ref().is_some_and(|(_, punctuation)| punctuation.chars().any(char::is_alphabetic));
        let opens_prose = next.is_some_and(|next| prose(next) || numbered_paragraph(&next.text));
        let fields_follow = next.zip(after).is_some_and(|(next, after)| {
            !prose(next) && !prose(after) && next.page_index == last.page_index && width(next) < 0.5 && width(after) < 0.5
        });
        let voted = lines.iter().any(|line| line.region_type == "heading");
        let placed = lines.iter().any(|line| raised.contains(line.id.as_str()) || primitives.headings.navigation_parents.contains_key(&line.id));
        // What another reading of the document already holds is no heading: a numbered
        // paragraph's label (the paragraph reader's), a transcript's testimony (a page of its
        // speakers' turns: the transcript rules raise its headings).
        let vetoed = lines.iter().any(|line| line.paragraph_label_candidate)
            || (transcript && testimony.contains(&first.page_index))
            || (text.ends_with(['.', ',', ';']) && !bare_number(&text))
            || label(&text)
            || (transcript && turn_or_event(&text))
            || date_line(&text)
            || words > 20;
        let mut own = -1.5;
        own += if voted { 2.5 } else { 0.0 };
        own += if bold || larger || lined || (caps && !body_type) { 1.0 } else { 0.0 };
        // A number in the body's own type numbers the body (its paragraphs, provisions, items).
        own -= if body_type && !caps && !division_word { 1.5 } else { 0.0 };
        own += if division_word { 1.5 } else if enumerated.is_some() && !body_type { 0.75 } else { 0.0 };
        own += if opens_prose { 1.0 } else { 0.0 };
        own -= if fields_follow { 0.75 } else { 0.0 };
        own += if above >= 0.8 { 0.5 } else { 0.0 };
        own -= if text.ends_with(':') { 1.0 } else { 0.0 };
        own -= if words > 12 { 1.0 } else { 0.0 };
        own -= if lines.len() > 1 && enumerated.is_none() { 0.5 } else { 0.0 };
        candidates.push(Candidate { lines, look, own, placed, vetoed, voted, page: first.page_index, numbers, set_apart: !body_type || caps || division_word });
        k = end;
    }

    // Each group's own signs: its numbering counting on, its recurrence over pages, the readers'
    // share of it, and a group so large and so little read as headings that it is the body's.
    let mut groups: HashMap<&Look, Vec<usize>> = HashMap::new();
    for (n, candidate) in candidates.iter().enumerate() {
        groups.entry(&candidate.look).or_default().push(n);
    }
    let mut score = vec![0.0; candidates.len()];
    for members in groups.values() {
        let group: Vec<&Candidate> = members.iter().map(|&n| &candidates[n]).collect();
        if group.len() < 2 {
            score[members[0]] = group[0].own;
            continue;
        }
        let mean = group.iter().map(|candidate| candidate.own).sum::<f64>() / group.len() as f64;
        let voted = group.iter().filter(|candidate| candidate.voted).count() as f64 / group.len() as f64;
        let pages = group.iter().map(|candidate| candidate.page).collect::<HashSet<_>>().len();
        let mut signs = 0.0;
        // Numbering that counts on marks a division only where the group stands apart from the body:
        // the body's own paragraphs and items count on too.
        if let Some(steps) = counts_on(&group).filter(|_| group[0].set_apart) {
            signs += 1.5 * (steps - 0.5);
        }
        signs += if pages >= 2 { 0.5 } else { 0.0 };
        signs += if voted >= 0.5 { 0.5 } else { 0.0 };
        signs -= if group.len() > 40 && voted < 0.2 { 1.0 } else { 0.0 };
        for &n in members {
            score[n] = 0.5 * candidates[n].own + 0.5 * mean + signs;
        }
    }

    let mut outline = Outline { headings: HashSet::new(), weighed: HashSet::new(), levels: HashMap::new() };
    let mut chosen: HashSet<&Look> = HashSet::new();
    for (n, candidate) in candidates.iter().enumerate() {
        outline.weighed.extend(candidate.lines.iter().map(|line| line.id.clone()));
        if candidate.placed || !candidate.vetoed && score[n] > 0.0 {
            outline.headings.extend(candidate.lines.iter().map(|line| line.id.clone()));
            chosen.insert(&candidate.look);
        }
    }
    // The heading groups ranked: larger type first, then capitals, bold, centred, underlined, then
    // the one met first; numbered groups take their level from the ladder after.
    let first_met: HashMap<&Look, usize> = candidates.iter().enumerate().rev().map(|(n, candidate)| (&candidate.look, n)).collect();
    let mut ranked: Vec<&Look> = chosen.into_iter().collect();
    ranked.sort_by_key(|look| (std::cmp::Reverse(look.1), !look.4, !look.2, look.6 != 2, !look.5, first_met[look]));
    let rank: HashMap<&Look, usize> = ranked.iter().enumerate().map(|(r, look)| (*look, r + 1)).collect();
    for candidate in &candidates {
        if let Some(&level) = rank.get(&candidate.look) {
            for line in &candidate.lines {
                if outline.headings.contains(&line.id) {
                    outline.levels.insert(line.id.clone(), level);
                }
            }
        }
    }
    outline
}
