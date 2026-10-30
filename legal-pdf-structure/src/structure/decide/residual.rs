//! The lines no rule settled, weighed for headings the way a document sets its divisions: a size, a
//! weight, capitals, a place on the line, a numbering family. The decision is taken for each way a
//! document sets its short lines (a look), from what its lines show together, and each line follows
//! its look unless its own words or place say otherwise.
//!
//! A candidate is a short open line, or a few such lines set as one (a title wrapping), that ends no
//! sentence and goes on with no line before it. A look is one face, size, weight, slant, case and
//! underline, one place on the line (flush, indented, centred in its column) and one numbering
//! family. What says a line opens a division: its source tags it a heading or a layout model a
//! title, it is set off the body's type, it numbers a division, space stands over it, and prose or
//! an opening unit follows it. What says it does not: it is a label, a speaker's turn, a date, it
//! ends on a colon, it runs long, or short lines follow it as a block's fields do. What a look shows
//! together: it recurs over pages, and sources tag most of it.
//!
//! Each line scores its own evidence as log-odds; in a look of two or more it takes half its own and
//! half the look's, with the look's own signs added.
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

/// The look of a heading or candidate whose first line is `first` and whose text, its lines joined,
/// is `text`, and whether its type is the body's own (its face and size, neither bold, italic,
/// underlined nor larger than the page's type).
pub(crate) fn look_of(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    geometry: &Geometry,
    first: Slot,
    text: &str,
    families: &mut Families,
) -> (Look, bool) {
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
    let (bold, italic, underlined) = (look & LOOK_BOLD != 0, look & LOOK_ITALIC != 0, look & LOOK_UNDERLINED != 0);
    let larger = measure.size > 0.0 && f64::from(evidence.half_points[at]) / 2.0 >= measure.size * 1.1;
    let body_type = fences.body == Some((evidence.face[at], evidence.half_points[at])) && !bold && !underlined && !larger && !italic;
    let family = headings::rules::enumerator(text).map_or_else(String::new, |(value, punctuation)| family(&value, &punctuation));
    let look = Look {
        face: evidence.face[at],
        size: evidence.half_points[at],
        bold,
        italic,
        caps: all_caps(text),
        underlined,
        place,
        family: families.id(family),
    };
    (look, body_type)
}

/// Two lines a blank of most of a line parts, or on two pages: two blocks.
pub(crate) fn apart(pages: &[Page], slots: &Slots, geometry: &Geometry, upper: Slot, lower: Slot) -> bool {
    slots.page(upper) != slots.page(lower)
        || slots.line(pages, lower).bbox[1] - slots.line(pages, upper).bbox[3] > 0.8 * geometry.page(slots.page(lower)).0.height
}

/// A candidate: its lines (places in its context's lines), its look, its own evidence as log-odds,
/// whether a word or its place bars it, whether its source tags it a heading, and its page.
struct Candidate {
    lines: std::ops::Range<usize>,
    look: Look,
    own: f64,
    vetoed: bool,
    voted: bool,
    page: usize,
}

/// The headings among the open lines of each reading context (`sequences`), weighed by their own
/// evidence and their look's: a heading's lines take the heading's role, those after its first
/// going on with it. `testimony` holds a transcript's pages of speakers' turns; `unnamed`, the lines
/// bookmarks naming every heading leave unnamed.
pub(crate) fn weigh(
    pages: &[Page],
    slots: &Slots,
    evidence: &Evidence,
    fences: &Fences,
    sequences: &Sequences,
    geometry: &Geometry,
    testimony: &[bool],
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
    for sequence in &sequences.lines {
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
            // A title wrapping onto the lines under it: same page and block, same type, short, no
            // number of its own, the line before ending no sentence.
            let mut end = k + 1;
            while end < sequence.len() && end - k < 3 {
                let (next, prior) = (sequence[end], sequence[end - 1]);
                let before = line(prior).text.trim();
                if apart(prior, next)
                    || !eligible(next)
                    || style(next) != style(first)
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
            let (look, body_type) = look_of(pages, slots, evidence, fences, geometry, first, &text, &mut families);
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
            let transcript = fences.transcript(first);
            let numbered = headings::rules::numbered(&text);
            let larger = measure(first).size > 0.0 && f64::from(evidence.half_points[first.index()]) / 2.0 >= measure(first).size * 1.1;
            let vetoed = (transcript && testimony[slots.page(first)])
                || (text.ends_with(['.', ',', ';']) && !headings::rules::bare_number(&text))
                || headings::rules::label(&text)
                || (transcript && headings::rules::turn_or_event(&text))
                || date_line(&text)
                || words > 20
                || voted
                    && !numbered
                    && (end - k > 1
                        || body_type && look.caps
                        || !larger && below.is_some_and(|below| below > above.clamp(-3.0, 3.0) + 0.5));
            let mut own = -1.5;
            own += if voted { 2.5 } else { 0.0 };
            own += if !body_type || look.caps { 1.0 } else { 0.0 };
            // A number in the body's own type numbers the body (its paragraphs, provisions, items).
            own -= if body_type && !look.caps && !division_word { 1.5 } else { 0.0 };
            own += if division_word { 1.5 } else if enumerated.is_some() && !body_type { 0.75 } else { 0.0 };
            own += if opens_prose { 1.0 } else { 0.0 };
            own -= if fields_follow { 0.75 } else { 0.0 };
            own += if above >= 0.8 { 0.5 } else { 0.0 };
            own -= if text.ends_with(':') { 1.0 } else { 0.0 };
            own -= if words > 12 { 1.0 } else { 0.0 };
            own -= if end - k > 1 && enumerated.is_none() { 0.5 } else { 0.0 };
            candidates.push(Candidate { lines: k..end, look, own, vetoed, voted, page: slots.page(first) });
            k = end;
        }
        // Each look's own signs: its recurrence over pages, its sources' share of it, and a look so
        // common and so little tagged that it is the body's.
        let mut looks: Vec<(Look, Vec<usize>)> = Vec::new();
        for (n, candidate) in candidates.iter().enumerate() {
            match looks.iter_mut().find(|(look, _)| *look == candidate.look) {
                Some((_, members)) => members.push(n),
                None => looks.push((candidate.look, vec![n])),
            }
        }
        let mut score = vec![0.0; candidates.len()];
        for (_, members) in &looks {
            if members.len() < 2 {
                score[members[0]] = candidates[members[0]].own;
                continue;
            }
            let size = members.len() as f64;
            let mean = members.iter().map(|&n| candidates[n].own).sum::<f64>() / size;
            let voted = members.iter().filter(|&&n| candidates[n].voted).count() as f64 / size;
            let mut pages_met: Vec<usize> = members.iter().map(|&n| candidates[n].page).collect();
            pages_met.sort_unstable();
            pages_met.dedup();
            let mut signs = 0.0;
            signs += if pages_met.len() >= 2 { 0.5 } else { 0.0 };
            signs += if voted >= 0.5 { 0.5 } else { 0.0 };
            signs -= if members.len() > 40 && voted < 0.2 { 1.0 } else { 0.0 };
            for &n in members {
                score[n] = 0.5 * candidates[n].own + 0.5 * mean + signs;
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
