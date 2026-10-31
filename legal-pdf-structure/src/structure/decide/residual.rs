//! The lines no rule settled, weighed for headings the way a document sets its divisions: a size, a
//! weight, capitals, a place on the line, a numbering family. The decision is taken for each way a
//! document sets its short lines (a look), from what its lines show together, and each line follows
//! its look unless its own words or place say otherwise.
//!
//! A candidate is a short open line, or a few such lines set as one (a title wrapping), that ends no
//! sentence and goes on with no line before it. A marker alone on its line ("II.", "12", "ARTICLE
//! 31") is one candidate with the line it numbers, and is weighed with it: the pair is a paragraph's
//! where that text runs its column's measure, goes on into the next line or ends a sentence, and a
//! marker with no such line heads nothing. A look is one face, size, weight, slant, case and
//! underline, one place on the line (flush, indented, centred in its column) and one numbering
//! family. What says a line opens a division: its source tags it a heading or a layout model a
//! title, it is set apart from the body's type (bold, larger, underlined, in capitals, or italic at
//! the body's size; a scan's line with no face read measured so), it numbers a division, space
//! stands over it, and prose or an opening unit follows it. What says it does not: it is a label, a
//! speaker's turn, a date, it ends on a colon, it runs long, or short lines follow it as a block's
//! fields do. What a look shows together: it recurs over pages, sources tag most of it, and a large
//! look little set apart from the body's type is the body's.
//!
//! Each line scores its own evidence as log-odds; in a look of two or more it takes half its own and
//! half the look's, with the look's own signs added.
//!
//! A quotation's editorial line ("[Emphasis added]", an ellipsis alone) is no heading, wherever it
//! stands, and a transcript's testimony has its own words' and its outline's headings, never the
//! residual's.
//!
//! A line its source or a layout model reads as a heading carries that reading as its strongest
//! sign, so three ways such a reading fails are vetoes of their own for it: an unnumbered heading
//! wrapping over lines (a party block, an address, a title page's caption), capitals alone in the
//! body's own type (a name, a caption's connective), and an unnumbered line no larger than the
//! body's that sits nearer the line above than the text below (a signature block's names, a form's
//! field).

use super::*;
use std::cmp::Reverse;

/// How a document sets a short line: its face (an index into the evidence's faces), its size in
/// half points, bold, italic, capitals, underlined, its place on the line (0 flush, 1 indented, 2
/// centred in its column) and its numbering family (an index into `Families`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Look {
    pub(crate) face: u16,
    pub(crate) size: u16,
    pub(crate) bold: bool,
    pub(crate) italic: bool,
    pub(crate) caps: bool,
    pub(crate) underlined: bool,
    pub(crate) place: u8,
    pub(crate) family: u16,
}

impl Look {
    /// The order of heading looks by rank: larger type first, then capitals, bold, centred, underlined.
    pub(crate) fn rank(&self) -> (Reverse<u16>, bool, bool, bool, bool) {
        (Reverse(self.size), !self.caps, !self.bold, self.place != 2, !self.underlined)
    }
}

/// The numbering families looks are set in, each by its index.
#[derive(Default)]
pub(crate) struct Families(Vec<String>);

impl Families {
    fn id(&mut self, family: String) -> u16 {
        if let Some(at) = self.0.iter().position(|known| *known == family) {
            return at as u16;
        }
        self.0.push(family);
        (self.0.len() - 1).min(usize::from(u16::MAX)) as u16
    }
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

/// How a line's type stands against the body's: the body's own (its face and size, neither bold,
/// italic, underlined nor larger than the page's type); larger than the page's type; and set apart
/// from the body (bold, larger, underlined, in capitals, or italic at the body's size or larger). A
/// smaller size or another face alone sets nothing apart: a quotation's, a note's or a running
/// head's type is no heading's.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Typeset {
    pub(crate) body: bool,
    pub(crate) larger: bool,
    pub(crate) set_apart: bool,
}

/// The median of the positive values.
fn positive_median(mut values: Vec<f64>) -> f64 {
    values.retain(|value| *value > 0.0);
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// The look of a heading or candidate whose first line is `first` and whose text, its lines joined,
/// is `text`, and how its type stands against the body's (`Typeset`). A line no face was read for
/// (a scan's recognition) is set in the body's type unless it measures otherwise: larger where its
/// height reads a size 1.1 times its page's, bold where its strokes are 1.2 times as thick as its
/// page's.
pub(crate) fn look_of(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    geometry: &Geometry,
    first: Slot,
    text: &str,
    families: &mut Families,
) -> (Look, Typeset) {
    let at = first.index();
    let page = slots.page(first);
    let line = slots.line(pages, first);
    let (measure, columns) = geometry.page(page);
    let (column_width, column_centre) = columns.place(line, (first.0 - slots.on(page).start) as usize);
    let place = if column_centre.abs() < 0.04 && column_width < 0.8 {
        2
    } else if (line.bbox[0] - measure.left) / (measure.right - measure.left).max(1.0) > 0.05 {
        1
    } else {
        0
    };
    let look = evidence.look[at];
    let (mut bold, italic, underlined) = (look & LOOK_BOLD != 0, look & LOOK_ITALIC != 0, look & LOOK_UNDERLINED != 0);
    let faceless = evidence.face[at] == u16::MAX;
    let (larger, smaller) = if faceless {
        let on_page = slots.on(page);
        let size = positive_median(on_page.clone().map(|slot| evidence.size[slot as usize]).collect());
        let weight = positive_median(on_page.map(|slot| f64::from(evidence.weight[slot as usize])).collect());
        bold |= weight > 0.0 && f64::from(evidence.weight[at]) >= weight * 1.2;
        (size > 0.0 && evidence.size[at] >= size * 1.1, size > 0.0 && evidence.size[at] < size * 0.95)
    } else {
        let size = f64::from(evidence.half_points[at]) / 2.0;
        let smaller = match fences.body {
            Some((_, body)) => evidence.half_points[at] < body,
            None => measure.size > 0.0 && size < measure.size * 0.95,
        };
        (measure.size > 0.0 && size >= measure.size * 1.1, smaller)
    };
    let in_body_face = faceless || fences.body == Some((evidence.face[at], evidence.half_points[at]));
    let caps = all_caps(text);
    let typeset = Typeset {
        body: in_body_face && !bold && !underlined && !larger && !italic,
        larger,
        set_apart: bold || larger || underlined || caps || italic && !smaller,
    };
    let family = headings::rules::enumerator(text).map_or_else(String::new, |(value, punctuation)| family(&value, &punctuation));
    let look = Look {
        face: evidence.face[at],
        size: evidence.half_points[at],
        bold,
        italic,
        caps,
        underlined,
        place,
        family: families.id(family),
    };
    (look, typeset)
}

/// A marker standing alone on its line: a number and nothing else ("III.", "12", "(b)"), or a
/// division's word and its number ("ARTICLE 31", "PART IV").
pub(crate) fn marker_alone(evidence: &Evidence, slot: Slot, text: &str) -> bool {
    headings::rules::bare_number(text)
        || evidence.marker[slot.index()].is_some_and(|marker| marker.only)
        || evidence.words[slot.index()] & WORD_DIVISION != 0 && text.trim_end_matches('.').split_whitespace().count() == 2
}

/// Two lines a blank of most of a line parts, or on two pages: two blocks.
pub(crate) fn apart(pages: &[Page], slots: &Slots, geometry: &Geometry, upper: Slot, lower: Slot) -> bool {
    slots.page(upper) != slots.page(lower)
        || slots.line(pages, lower).bbox[1] - slots.line(pages, upper).bbox[3] > 0.8 * geometry.page(slots.page(lower)).0.height
}

/// The residual's weights. A candidate's own evidence as log-odds: `OWN_BASE`, and each of its
/// indicators' weight where it holds, in order: its source tags it a heading; its type is set apart;
/// it is set in the body's own type, in no capitals and with no division's word (a number in the
/// body's type numbers the body); it opens with a division's word; it opens with another enumerator,
/// not in the body's type; prose or a unit follows it; fields follow it; a blank of 0.8 of a line or
/// more stands above it; it ends on a colon; it runs over twelve words; it wraps over more lines than
/// its marker with no enumerator.
pub(crate) const OWN_BASE: f64 = -1.5;
pub(crate) const OWN: [f64; 11] = [2.5, 1.0, -1.5, 1.5, 0.75, 1.0, -0.75, 0.5, -1.0, -1.0, -0.5];
/// A look's signs, each weighed where it holds, in order: its candidates stand on two pages or more;
/// half or more are tagged headings by their source; it has over forty, under a fifth set apart.
pub(crate) const LOOK: [f64; 3] = [0.5, 0.5, -1.0];
/// A candidate's own evidence's share of its score where its look holds others, the look's mean
/// taking the rest.
pub(crate) const OWN_SHARE: f64 = 0.5;

/// A candidate: its lines (places in its context's lines), its look, its own evidence as log-odds,
/// whether a word or its place bars it, whether its source tags it a heading, whether its type is
/// set apart from the body's, and its page.
struct Candidate {
    lines: std::ops::Range<usize>,
    look: Look,
    own: f64,
    vetoed: bool,
    voted: bool,
    set_apart: bool,
    page: usize,
}

/// The headings among the open lines of each reading context (`sequences`), weighed by their own
/// evidence and their look's: a heading's lines take the heading's role, those after its first
/// going on with it. A transcript's testimony has its own words' headings (its parts, witnesses
/// and examinations), its anchors' and its contents' title, never the residual's: throughout a
/// transcript that is testimony throughout (`Fences::testimony`), and on a transcript's pages of
/// speakers' turns. `unnamed` holds the lines bookmarks naming every heading leave unnamed.
pub(crate) fn weigh(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    sequences: &Sequences,
    geometry: &Geometry,
    unnamed: &impl Fn(Slot) -> bool,
    decision: &mut Decision,
) {
    let mut families = Families::default();
    let line = |slot: Slot| slots.line(pages, slot);
    let measure = |slot: Slot| &geometry.page(slots.page(slot)).0;
    let width = |slot: Slot| {
        let (line, measure) = (line(slot), measure(slot));
        (line.bbox[2] - line.bbox[0]) / (measure.right - measure.left).max(1.0)
    };
    let style = |slot: Slot| (evidence.face[slot.index()], evidence.half_points[slot.index()]);
    let apart = |upper: Slot, lower: Slot| apart(pages, slots, geometry, upper, lower);
    let voted = |slot: Slot| matches!(evidence.tag[slot.index()], SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle);
    // Each page holding a speaker's turn: a transcript's testimony there.
    let mut spoken = vec![false; pages.len()];
    for slot in (0..slots.len()).filter(|&slot| evidence.words[slot] & WORD_SPEAKER != 0) {
        spoken[slots.page(Slot(slot as u32))] = true;
    }
    let traced = trace_folder().is_some();
    for (context, sequence) in sequences.lines.iter().enumerate() {
        let role = &decision.role;
        let eligible = |slot: Slot| {
            let at = slot.index();
            let text = line(slot).text.trim();
            role[at] == Role::Open
                && matches!(evidence.tag[at], SourceTag::None | SourceTag::Heading(_) | SourceTag::Title | SourceTag::ParagraphTitle)
                && fences.quotation[at] == 0
                && (1..=160).contains(&text.chars().count())
                && !text.starts_with(char::is_lowercase)
                && !text.contains("___")
                && fences.continues[at].is_none()
                && !unnamed(slot)
        };
        let prose = |slot: Slot| body_like(role[slot.index()]) && (width(slot) >= 0.7 || line(slot).text.trim().chars().count() >= 90);
        let mut candidates: Vec<Candidate> = Vec::new();
        let mut k = 0;
        while k < sequence.len() {
            let first = sequence[k];
            if !eligible(first) {
                k += 1;
                continue;
            }
            // A marker alone on its line is one with the line it numbers, the next on its page (a
            // heading's title, a paragraph's first line), whatever their types; with none, or with
            // a number of its own, it heads nothing.
            let alone = marker_alone(evidence, first, line(first).text.trim());
            let mut end = k + 1;
            if alone {
                let numbers = sequence.get(end).is_some_and(|&next| {
                    slots.page(next) == slots.page(first) && eligible(next) && !headings::rules::numbered(line(next).text.trim())
                });
                if !numbers {
                    k += 1;
                    continue;
                }
                end += 1;
            }
            // A title wrapping onto the lines under it: same page and block, same type, short, no
            // number of its own, the line before ending no sentence.
            let title = sequence[end - 1];
            while end < sequence.len() && end - k < 3 + usize::from(alone) {
                let (next, prior) = (sequence[end], sequence[end - 1]);
                let before = line(prior).text.trim();
                if apart(prior, next)
                    || !eligible(next)
                    || style(next) != style(title)
                    || width(prior) >= 0.75
                    || width(next) >= 0.75
                    || headings::rules::numbered(line(next).text.trim())
                    || sentence_ended(before)
                    || before.ends_with([':', ';', ','])
                {
                    break;
                }
                end += 1;
            }
            // A line running its column's measure on into the next line of its block opens a
            // paragraph, a provision or a list's item: its text goes on, as a title's does not.
            let last = sequence[end - 1];
            let page = slots.page(last);
            let (_, columns) = geometry.page(page);
            let (last_width, _) = columns.place(line(last), (last.0 - slots.on(page).start) as usize);
            if last_width >= 0.85 && sequence.get(end).is_some_and(|&next| !apart(last, next)) {
                k = end;
                continue;
            }
            let text = sequence[k..end].iter().map(|&slot| line(slot).text.trim()).collect::<Vec<_>>().join(" ");
            // A numbered title is set as its title's line is, whatever its number's type.
            let (look, typeset) = look_of(pages, slots, evidence, fences, geometry, title, &text, &mut families);
            let enumerated = headings::rules::enumerator(&text);
            let division_word = enumerated.as_ref().is_some_and(|(_, punctuation)| punctuation.chars().any(char::is_alphabetic));
            let next = sequence.get(end).copied();
            let after = sequence.get(end + 1).copied();
            let above = k
                .checked_sub(1)
                .map(|j| sequence[j])
                .filter(|&prior| slots.page(prior) == slots.page(first))
                .map_or(3.0, |prior| (line(first).bbox[1] - line(prior).bbox[3]) / measure(first).height);
            // The blank under its last line, in line heights; unknown at the foot of its page.
            let below = next
                .filter(|&next| slots.page(next) == page)
                .map(|next| ((line(next).bbox[1] - line(last).bbox[3]) / measure(last).height).clamp(-3.0, 3.0));
            let words = text.split_whitespace().count();
            let opens_prose = next.is_some_and(|next| prose(next) || decision.unit[next.index()].is_some());
            let fields_follow = next.zip(after).is_some_and(|(next, after)| {
                !prose(next) && !prose(after) && slots.page(next) == page && width(next) < 0.5 && width(after) < 0.5
            });
            let voted = sequence[k..end].iter().any(|&slot| voted(slot));
            let numbered = headings::rules::numbered(&text);
            // A marker's text that goes on into the next line, or ends a sentence, is a paragraph's.
            let runs_on = alone
                && (sentence_ended(&text) || next.is_some_and(|next| fences.continues[next.index()].is_some()));
            let vetoes = [
                (VETO_TESTIMONY, fences.testimony(first) || fences.transcript(first) && (spoken[slots.page(first)] || headings::rules::turn_or_event(&text))),
                (VETO_RUNS_ON, runs_on),
                (VETO_EDITORIAL, numbered_paragraphs::editorial_line(&text)),
                (VETO_ENDS, text.ends_with(['.', ',', ';'])),
                (VETO_LABEL, headings::rules::label(&text) || date_line(&text)),
                (VETO_WORDS, words > 20),
                (
                    VETO_TAGGED,
                    voted
                        && !numbered
                        && (end - k > 1
                            || typeset.body && look.caps
                            || !typeset.larger && below.is_some_and(|below| below > above.clamp(-3.0, 3.0) + 0.5)),
                ),
            ]
            .into_iter()
            .filter(|(_, holds)| *holds)
            .fold(0, |bits, (bit, _)| bits | bit);
            // The candidate's own evidence, each indicator weighed by `OWN`.
            let terms = [
                voted,
                typeset.set_apart,
                // A number in the body's own type numbers the body (its paragraphs, provisions, items).
                typeset.body && !look.caps && !division_word,
                division_word,
                !division_word && enumerated.is_some() && !typeset.body,
                opens_prose,
                fields_follow,
                above >= 0.8,
                text.ends_with(':'),
                words > 12,
                end - k > 1 + usize::from(alone) && enumerated.is_none(),
            ];
            let own = terms.iter().zip(OWN).fold(OWN_BASE, |own, (&holds, weight)| own + if holds { weight } else { 0.0 });
            if traced {
                decision.residual.push(ResidualRecord {
                    lines: sequence[k..end].iter().map(|slot| slot.0).collect(),
                    context: context as u32,
                    look: 0,
                    terms,
                    alone,
                    above,
                    below,
                    words: words as u32,
                    caps: look.caps,
                    body: typeset.body,
                    larger: typeset.larger,
                    numbered,
                    vetoes,
                    signs: [false; LOOK.len()],
                    members: 1,
                    own,
                    score: own,
                    heading: false,
                });
            }
            candidates.push(Candidate { lines: k..end, look, own, vetoed: vetoes != 0, voted, set_apart: typeset.set_apart, page: slots.page(first) });
            k = end;
        }
        // Each look's own signs: its recurrence over pages, its sources' share of it, and a look so
        // common and so little set apart from the body's type that it is the body's.
        let mut looks: Vec<(Look, Vec<usize>)> = Vec::new();
        for (n, candidate) in candidates.iter().enumerate() {
            match looks.iter_mut().find(|(look, _)| *look == candidate.look) {
                Some((_, members)) => members.push(n),
                None => looks.push((candidate.look, vec![n])),
            }
        }
        let mut score = vec![0.0; candidates.len()];
        // Each candidate's look (an index among the context's looks) and its look's signs, traced.
        let mut grouped = vec![(0u32, [false; LOOK.len()], 1u32); candidates.len()];
        for (index, (_, members)) in looks.iter().enumerate() {
            if members.len() < 2 {
                score[members[0]] = candidates[members[0]].own;
                grouped[members[0]].0 = index as u32;
                continue;
            }
            let size = members.len() as f64;
            let mean = members.iter().map(|&n| candidates[n].own).sum::<f64>() / size;
            let voted = members.iter().filter(|&&n| candidates[n].voted).count() as f64 / size;
            let set_apart = members.iter().filter(|&&n| candidates[n].set_apart).count() as f64 / size;
            let mut pages_met: Vec<usize> = members.iter().map(|&n| candidates[n].page).collect();
            pages_met.sort_unstable();
            pages_met.dedup();
            let held = [pages_met.len() >= 2, voted >= 0.5, members.len() > 40 && set_apart < 0.2];
            let signs = held.iter().zip(LOOK).fold(0.0, |signs, (&holds, weight)| signs + if holds { weight } else { 0.0 });
            for &n in members {
                score[n] = OWN_SHARE * candidates[n].own + (1.0 - OWN_SHARE) * mean + signs;
                grouped[n] = (index as u32, held, members.len() as u32);
            }
        }
        if traced {
            let first = decision.residual.len() - candidates.len();
            for (n, record) in decision.residual[first..].iter_mut().enumerate() {
                (record.look, record.signs, record.members) = grouped[n];
                record.score = score[n];
                record.heading = !candidates[n].vetoed && score[n] > 0.0;
            }
        }
        for (candidate, score) in candidates.iter().zip(score) {
            let heading = !candidate.vetoed && score > 0.0;
            for (n, &slot) in sequence[candidate.lines.clone()].iter().enumerate() {
                decision.rule[slot.index()] = Rule::Residual;
                if heading {
                    decision.role[slot.index()] = Role::Heading;
                    decision.joined[slot.index()] = n > 0;
                }
            }
        }
    }
}
