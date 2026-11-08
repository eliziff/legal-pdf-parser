//! A stream's text, written as a provider writes a decision: a block to a line.

use super::*;

/// The most lines a heading over a numbered paragraph runs to.
const CLOSING_LINES: usize = 4;

/// A stream's text, and each of its UTF-16 units' line and place in that line's text (none for what the stream adds:
/// breaks, a margin number set before the other language's row).
#[derive(Default)]
pub struct Written {
    pub text: Vec<u16>,
    pub place: Vec<Option<(Slot, u32)>>,
    /// Each number alone on its line and the line of words it opens.
    pub led: Vec<(Slot, Slot)>,
}

impl Written {
    fn add(&mut self, words: &str, line: Option<Slot>) {
        for (offset, unit) in words.encode_utf16().enumerate() {
            self.text.push(if unit == 10 { 32 } else { unit });
            self.place.push(line.map(|line| (line, offset as u32)));
        }
    }
}

/// What the page says of the lines a stream keeps.
struct Flow {
    printed: HashMap<Slot, (usize, usize)>,
    flows: HashSet<Slot>,
    to_number: Vec<usize>,
}

impl Reading<'_> {
    fn flow(&self, kept: &[Slot], stream: Stream) -> Flow {
        let (pages, slots, evidence) = (self.pages, self.slots, self.evidence);
        // The printed paragraph each line stands in, as the page reader reads a page (its own leading, first lines
        // set in, a short last line, a scan's blanks) and a paragraph over a page (`continues_onto`).
        let mut runs = Vec::<&[Slot]>::new();
        let mut from = 0;
        for at in 1..=kept.len() {
            if at == kept.len() || slots.page(kept[at]) != slots.page(kept[from]) {
                runs.push(&kept[from..at]);
                from = at;
            }
        }
        let read: Vec<Vec<&Line>> = runs.iter().map(|run| run.iter().map(|&slot| slots.line(pages, slot)).collect()).collect();
        let measure = legal_pdf_text::paragraphs::measure_of(&read);
        let mut printed = HashMap::<Slot, (usize, usize)>::new();
        let mut flows = HashSet::<Slot>::new();
        for (k, run) in runs.iter().enumerate() {
            let found = legal_pdf_text::paragraphs::paragraphs_of(&read[k], &vec![true; run.len()], measure);
            printed.extend(run.iter().zip(&found).filter_map(|(&slot, paragraph)| Some((slot, (k, (*paragraph)?)))));
            if k > 0 && legal_pdf_text::paragraphs::continues_onto(&read[k - 1], &read[k]) {
                flows.insert(run[0]);
            }
        }
        // How many lines each line stands before the next that opens on a number (none after the last).
        let mut to_number = vec![usize::MAX; kept.len()];
        let mut number = None::<usize>;
        for at in (0..kept.len()).rev() {
            if evidence.marker[kept[at].index()].is_some() || self.marker_of.contains_key(&(kept[at], stream)) {
                number = Some(at);
            }
            to_number[at] = number.map_or(usize::MAX, |number| number - at);
        }
        Flow { printed, flows, to_number }
    }

    /// A first paragraph's number fused to its judge's drop capital: "1I" / "ACOBUCCI J. (dissenting) -- ...".
    fn drop_capital(&self, kept: &[Slot], at: usize) -> bool {
        let (pages, slots) = (self.pages, self.slots);
        let words = slots.line(pages, kept[at]).text.trim();
        let mut letters = words.chars();
        letters.next() == Some('1') && letters.next().is_some_and(char::is_uppercase) && letters.next().is_none()
            && kept.get(at + 1).is_some_and(|next| slots.line(pages, *next).text.chars().filter(|c| !c.is_whitespace()).take(3).filter(|c| c.is_uppercase()).count() == 3)
    }

    /// Whether the line at `place` goes on from the line before it.
    fn joins(&self, kept: &[Slot], flow: &Flow, place: usize, before: Slot, marker: Option<(Slot, bool)>, last_advance: &mut f64, led: &mut Vec<(Slot, Slot)>) -> bool {
        let (fences, pages, slots, evidence) = (self.fences, self.pages, self.slots, self.evidence);
        let (slot, at) = (kept[place], kept[place].index());
        let line = &slots.line(pages, slot).text;
        let (above, here) = (slots.line(pages, before), slots.line(pages, slot));
        // A marker standing alone on its line opens the line after it, whatever its row, a line of words set
        // no further left: "2025." ending a wrapped sentence opens no heading at the margin.
        // (Figures only: a heading's "I." alone in a gutter opens the heading, not the next line read.)
        let figures = above.text.trim().chars().any(|c| c.is_ascii_digit());
        // (Nor does a range's end wrapped over its line: "at paras. 11-" / "13.".)
        let wrapped = place >= 2 && slots.line(pages, kept[place - 2]).text.trim_end().ends_with(['-', '\u{2013}']);
        let alone = figures && !wrapped && evidence.marker[before.index()].is_some_and(|reading| reading.only) && evidence.marker[at].is_none()
            && above.bbox[0] <= here.bbox[0] + (here.bbox[3] - here.bbox[1]);
        // (A figure under a line going on, a lower-case word after it, is the sentence's: "... provisions of Appendix" / "2 of the Referendum Act".)
        let mut words = line.split_whitespace();
        let figure = words.next().is_some_and(|first| first.chars().all(|c| c.is_ascii_digit())) && words.next().is_some_and(|word| word.starts_with(char::is_lowercase));
        let wrapped_figure = figure && !sentence_ended(&above.text) && !above.text.trim_end().ends_with(':');
        // A line goes on with the printed paragraph it stands in, unless it opens on a marker of its own.
        let same = (evidence.marker[at].is_none() || wrapped_figure) && (flow.flows.contains(&slot) || flow.printed.get(&slot).is_some_and(|paragraph| flow.printed.get(&before) == Some(paragraph)));
        // The last few lines before a number, under a line ending a sentence, stand on their own, whatever the
        // page's geometry says: whether they head what follows is the reader's to say (`trailing_headings`).
        // A line in the body's own type ending as a sentence or a lead-in does is the paragraph's words, and so are the
        // lines of a sentence that wraps there ("Finally, would admission of the evidence bring the" / "administration of
        // justice into disrepute?").
        let in_body = |slot: &&Slot| fences.body == Some((evidence.face[slot.index()], evidence.half_points[slot.index()]));
        let stops = |slot: &Slot| slots.line(pages, *slot).text.trim_end().trim_end_matches([')', '"', '\u{201d}', '\u{bb}']).ends_with(['.', '?', '!', ':']);
        let plain = kept[place..].iter().take(CLOSING_LINES.min(flow.to_number[place])).take_while(in_body).any(stops);
        // (A note's call may stand after the stop: "... at arm's length.11" / "Figure 4.".)
        let uncalled = above.text.trim_end().trim_end_matches(|c: char| c.is_ascii_digit());
        let closes = sentence_ended(&above.text) || above.text.trim_end().ends_with([')', ']']) || uncalled.ends_with(['.', '?', '!']) && sentence_ended(uncalled);
        // (A place's initials end no sentence: "... the possibility that B.C." / "Rugby was insured?".)
        let last_word = above.text.split_whitespace().last().unwrap_or_default();
        let initials = last_word.trim_end_matches('.').contains('.') && last_word.chars().all(|c| c.is_uppercase() || c == '.');
        // (Nor does a line go apart for ending a sentence where the next line's first word could not have stood on it, in the
        // paragraph the page prints: a last sentence with no stop of its own, "La liberté d'expression est nettement violée par cette interdiction".
        // A heading under a full line is no sentence: "Remedy", "Brossard J.A. (dissenting in part)".)
        let room = self.measures.right_edge(slots.page(before), fences.at(before.index()).translation) - above.bbox[2];
        let full = slots.page(before) == slots.page(slot) && !here.recognized() && here.words.first().is_some_and(|word| word.bbox[2] - word.bbox[0] >= room) && in_body(&&slot) && in_body(&&before)
            && line.split_whitespace().filter(|word| word.starts_with(char::is_lowercase)).count() >= 4;
        let apart = closes && !plain && !initials && !(same && full) && flow.to_number[place] <= CLOSING_LINES;
        // After the last number a line at the same place and leading goes on with the paragraph, however short
        // the line above it stopped: only a real break lets the last paragraph end before the signatures.
        let advance = here.bbox[1] - above.bbox[1];
        let level = (here.bbox[0] - above.bbox[0]).abs() <= here.bbox[3] - here.bbox[1];
        // (Asked only where it can tell: after the last number, or under a sentence's end.)
        let closing = (flow.to_number[place] == usize::MAX || closes) && (legal_structure_model::structure_analysis().signature_line(line) || legal_structure_model::structure_analysis().end_matter(line) || legal_structure_model::structure_analysis().author_line(line) || legal_structure_model::structure_analysis().date_line(line) || legal_structure_model::structure_analysis().disposition_line(line));
        let last_goes_on = flow.to_number[place] == usize::MAX && level && !closing && *last_advance > 0.0 && advance > 0.0 && advance <= *last_advance * 1.15;
        let same = same || last_goes_on;
        if level && advance > 0.0 {
            *last_advance = advance;
        }
        // The lines closing the last paragraph stand on their own, for the reader to end it there.
        // (Under a sentence's end they do wherever they stand: a schedule's numbers may follow.)
        // (An author's line stands on its own wherever it does: a column's last word may have gone over the gutter.)
        // (A signer's name over the title under it too: "... signée en son nom par Claude Roy" / "Vice-president".)
        let signer = line.split_whitespace().count() <= 4 && line.split_whitespace().all(|word| word.starts_with(char::is_uppercase)) && !line.trim_end().ends_with('.') && kept.get(place + 1).is_some_and(|next| legal_structure_model::structure_analysis().signature_line(&slots.line(pages, *next).text));
        // (A line further out than the line above, whose first word would have fit on it, goes on from no sentence: a quotation's or a list's
        // last line stopped short of the column, "... V6T 1Z1" / "The Security Guard Unit", "7. R. v. Slemko, [2017] B.C.J. No. 2667" / "Legal Parameters and Principles.".)
        let stopped_short = slots.page(before) == slots.page(slot) && above.bbox[0] - here.bbox[0] > here.bbox[3] - here.bbox[1] && here.words.first().is_some_and(|word| word.bbox[2] - word.bbox[0] < room) && !here.recognized();
        let apart = apart || (closing || signer) && (flow.to_number[place] == usize::MAX || closes) || legal_structure_model::structure_analysis().author_line(line) || stopped_short;
        let fused = place > 0 && self.drop_capital(kept, place - 1);
        let joined = marker.is_none() && (alone || fused || !apart && (same || fences.continues[at].is_some())) || self.numbers.indented.contains(&slot);
        if alone && marker.is_none() {
            led.push((before, slot));
        }
        joined
    }

    pub fn written(&self, kept: &[Slot], stream: Stream) -> Written {
        let (pages, slots) = (self.pages, self.slots);
        let flow = self.flow(kept, stream);
        let mut written = Written::default();
        let (mut previous, mut last_advance) = (None::<Slot>, 0.0f64);
        let margins = self.fences.margin_number.iter().filter(|row| row.is_some()).count();
        for (place_in_kept, &slot) in kept.iter().enumerate() {
            let line = &slots.line(pages, slot).text;
            let marker = self.marker_of.get(&(slot, stream)).copied();
            if let Some(before) = previous {
                let joined = self.joins(kept, &flow, place_in_kept, before, marker, &mut last_advance, &mut written.led);
                // A break between blocks, which a line's own text never holds (`write` makes its breaks spaces).
                written.text.push(if joined { 32 } else { 10 });
                written.place.push(None);
            }
            previous = Some(slot);
            if let Some((marker, added)) = marker {
                // A short ruling's bare numbers in the margin read as the paragraph's they are: too few for a count of their own.
                let number = slots.line(pages, marker).text.trim();
                let margin = (3..=6).contains(&margins) && (self.fences.margin_number[marker.index()].is_some() || self.fences.margin_twin[marker.index()].is_some())
                    && number.bytes().all(|byte| byte.is_ascii_digit());
                written.add(if margin { "[" } else { "" }, None);
                written.add(number, (!added).then_some(marker));
                written.add(if margin { "] " } else { " " }, None);
            }
            if self.drop_capital(kept, place_in_kept) {
                let (figure, capital) = line.trim().split_at(1);
                written.add(figure, Some(slot));
                written.add(" ", None);
                written.add(capital, Some(slot));
            } else {
                written.add(line, Some(slot));
            }
        }
        written
    }
}
