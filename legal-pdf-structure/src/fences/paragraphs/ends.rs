//! Where a paragraph ends, read from the page.

use std::collections::{BTreeMap, HashMap, HashSet};
use legal_pdf_core::model::sentence_ended;
use crate::model::Slot;
use super::{QUOTED_INDENT, Reading};

/// A bold line under an elision is a quotation's emphasis.
fn elided(text: &str) -> bool {
    matches!(text.trim(), "..." | "\u{2026}" | "[...]" | "[\u{2026}]" | ". . .")
}

/// Where a paragraph ends, against what its text alone says: the facts of the page the text cannot see.
impl Reading<'_> {
    // A bold or underlined line at the margin, under a sentence's end, heads what follows, its stop or not ("Impact on
    // the Community."): the text alone cannot say so, and the paragraph ends there. A heading set in a quotation
    // stands further in. (A quoted paragraph's bold number, alone on its line, heads nothing: a heading has words.)
    fn marked(&self, slot: Slot, share: f64) -> bool {
        let (pages, slots) = (self.pages, self.slots);
        let words = &slots.line(pages, slot).words;
        legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > share) || !words.is_empty() && words.iter().all(|word| word.underlined)
    }

    /// The held line a bold heading follows, the paragraph's last.
    fn heading_above(&self, held_lines: &[Slot]) -> Option<usize> {
        let (pages, slots, edge) = (self.pages, self.slots, &self.measures.edge);
        let marked = |slot: Slot, share: f64| self.marked(slot, share);
        held_lines.windows(2).enumerate().position(|(at, pair)| {
            let (before, line) = (slots.line(pages, pair[0]), slots.line(pages, pair[1]));
            // (A page all quotation has its edge there: the margin is no further in than the document's running text.)
            let (page, language) = (slots.page(pair[1]), self.fences.at(pair[1].index()).translation);
            let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(self.measures.column(page, language));
            // (What a heading in a quotation heads stands further in, or smaller: an enactment's "Prohibited grounds". A
            // table's head stands over figures: "Township" / "2012".)
            // (An enactment's marginal note stands over its section's number, its own lines bold or not: "Burden of proof" /
            // "(3) The burden ...", "Publication, etc., of an" / "intimate image without" / "consent" / "162.1 (1) ...".)
            let under_heading = held_lines[(at + 2).min(held_lines.len())..].iter().find(|&&under| !marked(under, 0.8)).copied();
            let section = under_heading.is_some_and(|under| {
                let words = slots.line(pages, under).text.trim_start().trim_start_matches('(');
                words.starts_with(|c: char| c.is_ascii_digit()) && words.split_whitespace().next().is_some_and(|figures| figures.chars().filter(char::is_ascii_digit).count() >= figures.chars().filter(|c| c.is_alphabetic()).count())
            });
            let body_under = held_lines.get(at + 2).is_none_or(|&under| !self.smaller(under.index()) && slots.line(pages, under).bbox[0] - line.bbox[0] <= QUOTED_INDENT
                && slots.line(pages, under).text.chars().any(char::is_alphabetic)
                // (A run-in label's sentence goes on under it: "Video #3 – Published October 13, 2021 – This 12" / "second video ...".)
                && (marked(under, 0.5) || !slots.line(pages, under).text.trim_start().starts_with(char::is_lowercase)));
            // (A bracket left open goes on: "Les séances du conseil doivent être publiques (art. 322".)
            marked(pair[1], 0.8) && !marked(pair[0], 0.2) && line.text.split_whitespace().count() <= 12 && line.text.chars().any(char::is_alphabetic)
                && sentence_ended(&before.text) && !elided(&before.text) && line.bbox[0] - margin <= QUOTED_INDENT && body_under && !section
                && line.text.matches('(').count() <= line.text.matches(')').count()
        })
    }

    // All in bold from a line at the margin or on an enumerator to the paragraph's end, over a line that is not, is a
    // heading whatever its length or the stop above it. (An italic last line under a sentence's end too.)
    fn marked_tail(&self, held_lines: &[Slot]) -> Option<usize> {
        let (pages, slots, edge) = (self.pages, self.slots, &self.measures.edge);
        let marked = |slot: Slot, share: f64| self.marked(slot, share);
        (1..held_lines.len()).find(|&from| {
            let (run, line) = (&held_lines[from..], slots.line(pages, held_lines[from]));
            let (page, language) = (slots.page(held_lines[from]), self.fences.at(held_lines[from].index()).translation);
            let margin = edge.get(&(page, language)).copied().unwrap_or(0.0).min(self.measures.column(page, language));
            let enumerator = line.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
            let above = slots.line(pages, held_lines[from - 1]);
            // (A heading in italics ends on no stop: "Basi, supra." is the paragraph's source.)
            let italic = run.len() == 1 && !line.text.trim_end().ends_with('.') && legal_pdf_core::flag_share(line, legal_pdf_core::ITALIC).is_some_and(|share| share > 0.8) && sentence_ended(&above.text);
            let bold = |slot: Slot| legal_pdf_core::flag_share(slots.line(pages, slot), legal_pdf_core::BOLD).is_some_and(|bold| bold > 0.8);
            // (Emphasis running over a line is the sentence's: "... (b) a deprivation or risk of" / "deprivation.".)
            let wrapped = !sentence_ended(&above.text) && (line.text.trim_start().starts_with(char::is_lowercase) || above.words.last().is_some_and(|word| word.underlined)
                || above.spans.iter().rev().find(|span| !span.text.trim().is_empty()).is_some_and(|span| span.flags & legal_pdf_core::BOLD != 0));
            // (Nor is what a colon sets out a heading, or what follows an elision.)
            let set_out = above.text.trim_end().ends_with(':') || elided(&above.text);
            run.len() <= 24 && run.iter().any(|&held| slots.line(pages, held).text.chars().any(char::is_alphabetic)) && !marked(held_lines[from - 1], 0.2) && (line.bbox[0] - margin <= QUOTED_INDENT || enumerator)
                && !wrapped && !set_out && (run.iter().all(|&held| bold(held)) || italic || run.iter().all(|&held| marked(held, 0.8)) && sentence_ended(&above.text))
        })
    }

    // A quotation's last lines, in its indent, are the paragraph's: the text alone cannot see an indent ("Kelty", a letter's "Per:").
    fn quotation_tail(&self, spans: &mut BTreeMap<Slot, (u32, u32)>, kept: &[Slot], opens: &HashSet<Slot>, kept_at: &HashMap<Slot, usize>) {
        let (pages, slots) = (self.pages, self.slots);
        if let Some(&last) = spans.keys().next_back() {
            let (bbox, page) = (slots.line(pages, last).bbox, slots.page(last));
            // (Inside a quotation: the line above stands at its indent too, as a first line's does not, or it is set smaller than the text.)
            let small = self.smaller(last.index());
            let held: Vec<f64> = spans.keys().map(|&slot| slots.line(pages, slot).bbox[0]).collect();
            // (On this page: a report's margin alternates, and a paragraph over a page is no quotation for it. The page's own running
            // text stands no further in: a paragraph may open on the next page's quotation. A number hanging in the margin says
            // nothing of the text's edge.)
            let language = self.fences.at(last.index()).translation;
            let left = spans.keys().filter(|&&slot| slots.page(slot) == page && self.evidence.marker[slot.index()].is_none()).map(|&slot| slots.line(pages, slot).bbox[0]).fold(self.measures.column(page, language), f64::min);
            let from = kept_at.get(&last).copied().unwrap_or(kept.len());
            let in_quotation = bbox[0] - left > QUOTED_INDENT && held.len() >= 2 && ((held[held.len() - 2] - bbox[0]).abs() <= 3.0 || small);
            // (Or a sentence left open sets one out: "Counsel for Wiseau Films advised that," / "As for the location of the assets ...".)
            let set_out = slots.line(pages, last).text.trim_end().ends_with([',', ':']).then(|| kept.get(from + 1)).flatten().filter(|next| {
                slots.page(**next) == page && !opens.contains(*next) && slots.line(pages, **next).bbox[0] - left > QUOTED_INDENT
            });
            if in_quotation || set_out.is_some() {
                let mut prior = set_out.filter(|_| !in_quotation).map_or(last, |next| *next);
                let bbox = slots.line(pages, prior).bbox;
                let mut row = bbox;
                // (The quotation's own leading, from its last two lines on the page.)
                let tops: Vec<f64> = spans.keys().rev().filter(|&&slot| slots.page(slot) == page).take(2).map(|&slot| slots.line(pages, slot).bbox[1]).collect();
                let leading = if let [last, before] = tops[..] { last - before } else { 0.0 };
                for at in from + 1..kept.len() {
                    let under = slots.line(pages, kept[at]);
                    let enumerator = under.text.split_whitespace().next().is_some_and(|token| token.chars().count() <= 5 && token.ends_with(['.', ')']));
                    let same_row = self.evidence.row(slots, kept[at]) != u32::MAX && self.evidence.row(slots, kept[at]) == self.evidence.row(slots, prior);
                    // (In the quotation's small type, an item of its own further in is its: "[ ... ]" / "d) Les défenderesses ...".)
                    let quoted = small && self.smaller(kept[at].index());
                    let marked = self.marker_of.contains_key(&(kept[at], self.stream_of[kept[at].index()]));
                    // (A line closing the reasons stands at a first line's indent: "Appeal allowed with costs.", "Solicitors for the appellants: ...".
                    // It stands apart: at the quotation's own leading it is the quotation's, a quoted order's "Counsel for the Applicant".)
                    let closing = under.bbox[1] - row[1] > leading * 1.15 && (legal_structure_model::structure_analysis().disposition_line(&under.text) || legal_structure_model::structure_analysis().end_matter(&under.text) || legal_structure_model::structure_analysis().author_line(&under.text));
                    if opens.contains(&kept[at]) || closing || marked && !quoted || slots.page(kept[at]) != page || (under.bbox[0] - bbox[0]).abs() > 3.0 && !same_row && !(quoted && under.bbox[0] > bbox[0]) || enumerator && !quoted {
                        break;
                    }
                    row = under.bbox;
                    prior = kept[at];
                    spans.entry(kept[at]).or_insert((0, under.text.encode_utf16().count() as u32));
                }
            }
        }
    }

    /// The paragraph's held lines, its tail cut or taken back.
    pub(crate) fn settle(&self, spans: &mut BTreeMap<Slot, (u32, u32)>, kept: &[Slot], opens: &HashSet<Slot>, kept_at: &HashMap<Slot, usize>) {
        let held_lines: Vec<Slot> = spans.keys().copied().collect();
        let heading = self.heading_above(&held_lines);
        if let Some(heading) = heading {
            for line in &held_lines[heading + 1..] {
                spans.remove(line);
            }
        }
        let tail = self.marked_tail(&held_lines);
        if let Some(from) = tail {
            for line in &held_lines[from..] {
                spans.remove(line);
            }
        }
        if heading.is_none() && tail.is_none() {
            self.quotation_tail(spans, kept, opens, kept_at);
        }
    }
}
