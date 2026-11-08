//! A document's lines measured before anything decides what they are. Normalizing parts a
//! transcript's line numbers from the lines the extraction read them into and clears the
//! underlines a table's rulings drew under its words; each line then has its slot. Measuring reads
//! each page's words as text items, its rows and note separator; each line's row, type,
//! the source's tag on it, its opening marker and the line beside it carrying the marker's text, its
//! note label, its reference tokens, the lexicon its words match, and its citations; and the
//! document's bookmarks, the page labels it declares and whether it is an instrument.

use legal_pdf_core::model::{Line, Page, PdfOutlineEntry};
use legal_pdf_core::{flag_share, layout_analysis, line_font_size, median};
use legal_pdf_furniture::compact_note_line;
use legal_pdf_headings::headings::native::{bold_char_share, date_line};
use legal_pdf_headings::{captions, contents_lists, headings};
use legal_pdf_notes::notes::{endnote_heading, label_is_raised, label_spans, reference_wrapped_onto, structural_reset_heading};
use legal_pdf_text::text::{all_caps, party_label, sentence_ended};
use legal_structure_model::text::{is_note_symbol, label_prefix, normalize_label};
use legal_structure_model::StructureAnalysis;
use regex::Regex;
use std::collections::HashMap;
pub use legal_pdf_structure_model::evidence::*;
pub use legal_pdf_structure_model::{NoteLabel, PageView, Rows, Slot, Slots, SUPPRESS_COMMA, SUPPRESS_CORNER, SUPPRESS_FOOT, SUPPRESS_WRAPPED};
use legal_pdf_core::{dominant_face, regex, CitationSpan, ITALIC};
use legal_pdf_tables::PageItems;

pub fn page_rows(page: &Page) -> Rows {
    let lines = &page.lines;
    let mut order: Vec<usize> = (0..lines.len())
        .filter(|&slot| {
            let line = &lines[slot];
            !line.text.trim().is_empty()
                && line.bbox.iter().all(|value| value.is_finite())
                && line.bbox[3] - line.bbox[1] <= 3.0 * (line.bbox[2] - line.bbox[0])
        })
        .collect();
    order.sort_by(|&a, &b| lines[a].bbox[1].total_cmp(&lines[b].bbox[1]).then(lines[a].bbox[0].total_cmp(&lines[b].bbox[0])));
    let bands = contents_lists::row_bands(&order, |slot| lines[slot].bbox);
    let mut of = vec![u32::MAX; lines.len()];
    let mut tops = Vec::with_capacity(bands.len());
    for (row, band) in bands.iter().enumerate() {
        for &slot in band {
            of[slot] = row as u32;
        }
        tops.push(band.iter().map(|&slot| lines[slot].bbox[1]).fold(f64::INFINITY, f64::min));
    }
    let steps: Vec<f64> = tops.windows(2).map(|pair| pair[1] - pair[0]).filter(|step| *step > page.height * 0.002).collect();
    let step = if steps.is_empty() { page.height * 0.02 } else { median(steps) };
    let apart = |gap: f64| gap >= 1.5 * step;
    let count = tops.len();
    let top = if count > 1 && apart(tops[1] - tops[0]) {
        1
    } else if count > 3 && apart(tops[2] - tops[1]) && tops[1] < page.height * 0.2 {
        2
    } else {
        0
    };
    let bottom = if count > 1 && apart(tops[count - 1] - tops[count - 2]) {
        1
    } else if count > 3 && apart(tops[count - 2] - tops[count - 3]) && tops[count - 2] > page.height * 0.8 {
        2
    } else {
        0
    };
    Rows { of, count: count as u32, frame: [top, bottom] }
}

/// What normalizing leaves to be measured: which lines are a transcript's line numbers, by slot,
/// and each page's words as text items.
pub struct Normalized {
    pub line_numbers: Vec<bool>,
    pub items: Vec<PageItems>,
}

/// The pages normalized, the only change to their text before the runs are read: a transcript's
/// line numbers parted from the lines they were read into and held out of the body
/// (`split_line_numbers`, so no page's items offer them), a table's rulings cleared from under its
/// words (`clear_table_underlines`), and every line's slot.
pub fn normalize(pages: &mut [Page]) -> (Slots, Normalized) {
    let numbers = legal_pdf_text::line_numbers::split_line_numbers(pages);
    let mut items: Vec<PageItems> = pages.iter().map(legal_pdf_tables::page_items).collect();
    for (page, items) in pages.iter_mut().zip(&mut items) {
        legal_pdf_tables::clear_table_underlines(page, items);
    }
    let slots = Slots::of(pages);
    let mut line_numbers = vec![false; slots.len()];
    for (page, numbers) in numbers.into_iter().enumerate() {
        for line in numbers {
            line_numbers[slots.at(page, line).index()] = true;
        }
    }
    (slots, Normalized { line_numbers, items })
}

/// Indices for strings read again and again (faces, marker families).
#[derive(Default)]
struct Names {
    ids: HashMap<String, u16>,
    names: Vec<String>,
}

impl Names {
    fn id(&mut self, name: &str) -> u16 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len().min(usize::from(u16::MAX - 1)) as u16;
        self.ids.insert(name.to_owned(), id);
        self.names.push(name.to_owned());
        id
    }
}

/// An affidavit's jurat: "Sworn before me", "AFFIRMED BEFORE ME at Toronto".
fn jurat(text: &str) -> bool {
    regex!(r"(?i)\b(?:sworn|affirmed|declared)\b.{0,40}\bbefore\s+me\b").is_match(text)
}

/// The bits of the flags that hold.
fn bits<T: std::ops::BitOr<Output = T> + Default>(flags: impl IntoIterator<Item = (T, bool)>) -> T {
    flags.into_iter().filter(|(_, holds)| *holds).fold(T::default(), |bits, (bit, _)| bits | bit)
}

/// A line's text shaped as a title: twelve words at most, ending no sentence or clause.
fn title_shaped(text: &str) -> bool {
    !text.is_empty() && text.split_whitespace().count() <= 12 && !sentence_ended(text) && !text.ends_with([',', ';', ':'])
}

/// A line opening with a division's word and number, as the heading ladder reads one
/// (`rules::enumerator`): a part, an article, a schedule, a division or a chapter, in English or
/// French.
fn division_word(text: &str) -> bool {
    headings::rules::enumerator(text).is_some_and(|(_, word)| {
        matches!(word.as_str(), "part" | "partie" | "article" | "schedule" | "annexe" | "division" | "chapter" | "chapitre")
    })
}

/// A court's name and nothing else, as a caption's header prints it: "IN THE COURT OF APPEAL FOR
/// SASKATCHEWAN", "THE CONSTITUTIONAL COURT OF SOUTH AFRICA", "ONTARIO SUPERIOR COURT OF JUSTICE",
/// "SUPERIOR COURT OF JUSTICE - ONTARIO", "Cour d'appel du Québec". It opens in capitals, each
/// word of the name is capitalized but for its connectives, and the words before the court's are
/// its name's ("ON APPEAL FROM THE COURT OF APPEAL", "JUDGES OF THE COURT" name none); a line going
/// on after the name ("THE COURT ORDERS:", "the Court of Appeal's reasons", "Court of Appeal,"), a
/// statute's title ("THE COURT OF KING'S BENCH ACT") and the word alone name none.
fn court_name(text: &str) -> bool {
    const NAME: &str = r"[\p{Lu}(][\p{L}'’.&,()-]*";
    const JOIN: &str = r"(?i:of|for|the|and|de|du|des|la|le|d'|d’|l'|l’)";
    const FRENCH: &str = r"(?i:sup[ée]rieure|supr[êe]me|f[ée]d[ée]rale|provinciale|municipale|d['’]appel)";
    let text = text.trim();
    let lower = text.to_lowercase();
    let before = lower.split("cour").next().unwrap_or("");
    text.starts_with(char::is_uppercase)
        && text.split_whitespace().count() >= 2
        && !text.ends_with([',', ';', ':'])
        && !before.split_whitespace().any(|word| matches!(word, "from" | "on" | "to" | "by" | "of" | "at" | "and" | "et"))
        && !lower.split_whitespace().next_back().is_some_and(|word| matches!(word, "act" | "rules" | "regulations" | "loi"))
        && regex!(&format!(
            r"^(?:(?:IN|In)\s+)?(?:(?:THE|The)\s+)?(?:{NAME}\s+){{0,5}}(?:COURT|Court)(?:\s+(?i:of|for)\s+(?:{JOIN}\s+|{NAME}\s*)+)?(?:\s*[-–—]\s*{NAME}(?:\s+{NAME})*)?\s*$|^(?:(?:LA|La)\s+)?(?:COUR|Cour)(?:\s+(?:{FRENCH}|{JOIN}|{NAME}))+\s*$"
        ))
        .is_match(text)
}

/// The lexicon a line's words match (`WORD_*`), the citation title aside.
fn lexicon(analysis: &dyn StructureAnalysis, line: &Line) -> u32 {
    let text = line.text.trim();
    bits([
        (WORD_LABEL, headings::rules::label(text)),
        (WORD_PARTY, party_label().is_match(text)),
        (WORD_JURAT, jurat(text)),
        (WORD_DATE, date_line(text)),
        (WORD_FIELD, captions::field(&line.text).is_some()),
        (WORD_CLOSING, captions::closing(&line.text)),
        (WORD_REASONS, captions::reasons(analysis, line)),
        (WORD_SIGNATURE, analysis.signature_line(text)),
        (WORD_AUTHOR, analysis.author_line(text)),
        (WORD_WITNESS, headings::rules::witness(text)),
        (WORD_EXAMINATION, headings::rules::examination(text)),
        (WORD_TRANSCRIPT_PART, headings::rules::transcript_part(text)),
        (WORD_HEARING_PART, headings::rules::hearing_part(text)),
        (WORD_TURN, headings::rules::turn_or_event(analysis, text)),
        (WORD_INDEX_ENTRY, headings::rules::index_entry(text)),
        (WORD_SPEAKER, analysis.speaker_label(text).is_some()),
        (WORD_NOTES_HEADING, endnote_heading(text)),
        (WORD_RESET_HEADING, structural_reset_heading(text)),
        (WORD_CONTENTS_TITLE, contents_lists::contents_title(text)),
        (WORD_INDEX_LETTER, contents_lists::index_letter(text)),
        (WORD_HISTORY, analysis.history_note(text)),
        (WORD_DIVISION, division_word(text)),
        (WORD_COURT, court_name(text)),
    ])
}

/// Figures run on after a word's punctuation: "licensing.59", "Act;12". A decimal ("59.5") and a
/// one-letter abbreviation's number ("p.12") are none.
fn glued_re() -> &'static Regex {
    regex!(r"\p{L}{2}[.,;:)\]\u{201d}\u{2019}](\d{1,3})(?:\s|$)")
}

/// A line's spans of figures or note symbols set as references, its type set at `size`: each span's
/// text and how it is set (`REFERENCE_SUPERSCRIPT`, or `REFERENCE_RAISED`: smaller or raised).
pub fn reference_marks(line: &Line, size: f64) -> impl Iterator<Item = (&str, u8)> {
    let height = line.bbox[3] - line.bbox[1];
    line.spans.iter().filter_map(move |span| {
        let text = span.text.trim();
        if text.is_empty() || !text.chars().all(|c| c.is_ascii_digit() || is_note_symbol(c)) {
            None
        } else if span.superscript {
            Some((text, REFERENCE_SUPERSCRIPT))
        } else if (size > 0.0 && span.size > 0.0 && span.size * 1.25 <= size)
            || (height > 0.0 && span.bbox[3] <= line.bbox[3] - 0.25 * height)
        {
            Some((text, REFERENCE_RAISED))
        } else {
            None
        }
    })
}

/// The values of a line's reference marks (`reference_marks`, and its detached references), 0 for
/// a note symbol.
pub fn reference_values(line: &Line, size: f64) -> Vec<u32> {
    let value = |text: &str| text.trim().parse().unwrap_or(0);
    reference_marks(line, size)
        .map(|(text, _)| value(text))
        .chain(line.detached_references.iter().map(|reference| value(&reference.selected_text)))
        .collect()
}

/// A line's reference marks (`reference_marks`) with where each stands in the line's text: its value
/// (0 for a note symbol) and its first and past-last character. A mark is found by its span's text,
/// spans read in order along the line; one whose text the line does not hold there is left out. A
/// detached reference stands where it says.
pub fn reference_spans(line: &Line, size: f64) -> Vec<(u32, u32, u32)> {
    let value = |text: &str| text.trim().parse().unwrap_or(0);
    let chars: Vec<char> = line.text.chars().collect();
    let marks: Vec<&str> = reference_marks(line, size).map(|(text, _)| text).collect();
    let mut found: Vec<(u32, u32, u32)> = Vec::new();
    let mut cursor = 0;
    for span in &line.spans {
        let text: Vec<char> = span.text.trim().chars().collect();
        if text.is_empty() || text.len() > chars.len() {
            continue;
        }
        let Some(at) = (cursor..=chars.len().saturating_sub(text.len())).find(|&at| chars[at..at + text.len()] == text[..]) else { continue };
        cursor = at + text.len();
        let trimmed = span.text.trim();
        if marks.contains(&trimmed) && (span.superscript || reference_marks(line, size).any(|(t, _)| t == trimmed)) {
            // A mark set over two spans ("2" then "3" of "23") is one mark.
            match found.last_mut() {
                Some(last) if last.2 == at as u32 && trimmed.chars().all(|c| c.is_ascii_digit()) && chars[last.1 as usize..last.2 as usize].iter().all(char::is_ascii_digit) => {
                    last.2 = cursor as u32;
                    last.0 = chars[last.1 as usize..last.2 as usize].iter().collect::<String>().parse().unwrap_or(0);
                }
                _ => found.push((value(trimmed), at as u32, cursor as u32)),
            }
        }
    }
    // Figures run on after a word's punctuation ("count.12 This", "Act;3"), where no span sets them apart.
    // Recognition sets a raised mark on the line's own run of text: figures run straight on from a word or its
    // abbreviation ("Ontario1", "Reid J.2") are a mark for the notes to answer, or no one's.
    let recognized = line.source == "ocr" || line.spans.iter().any(|span| span.flags & legal_pdf_core::model::INVISIBLE != 0);
    let run_on = regex!(r"[\p{L}.,;:)\]\u{201d}\u{2019}\u{bb}](\d{1,3})(?:[\s.,;:)]|$)");
    for pattern in [Some(glued_re()), recognized.then_some(run_on)].into_iter().flatten() {
        for found_at in pattern.captures_iter(&line.text).filter_map(|c| c.get(1)) {
            let start = line.text[..found_at.start()].chars().count() as u32;
            let end = start + found_at.as_str().chars().count() as u32;
            if !found.iter().any(|&(_, a, b)| a < end && start < b) {
                found.push((value(found_at.as_str()), start, end));
            }
        }
    }
    // A detached reference stands where it says, its blank edges trimmed.
    for r in &line.detached_references {
        let (mut start, mut end) = (r.start_offset.min(chars.len()), r.end_offset.min(chars.len()));
        while start < end && chars[start].is_whitespace() {
            start += 1;
        }
        while end > start && chars[end - 1].is_whitespace() {
            end -= 1;
        }
        found.push((value(&r.selected_text), start as u32, end as u32));
    }
    found.sort_by_key(|&(_, start, _)| start);
    found
}

/// A line's text with the space a recognizer leaves inside its opening bracket closed up ("[ 13] Having ...").
fn closed_up(text: &str) -> std::borrow::Cow<'_, str> {
    match text.trim_start().strip_prefix("[ ") {
        Some(rest) => format!("[{}", rest.trim_start()).into(),
        None => text.into(),
    }
}

/// Every reading of a line's opening token by the numbering grammar: read alone and, where a single
/// letter is also a roman numeral, as the letter after the letter before it ("(i)" as 1 and after
/// "(h)"); and, where `division` says the line opens on a division word, the number after the word.
pub fn marker_readings(text: &str, division: bool) -> Vec<MarkerReading> {
    let grammar = layout_analysis();
    let text = closed_up(text);
    let mut tokens = text.split_whitespace();
    let Some(first) = tokens.next().filter(|token| token.chars().count() <= 12) else { return Vec::new() };
    let read = |token: &str, word: Option<&str>| {
        let alone = grammar.numbering_marker(token, None);
        let letter = alone.as_ref().filter(|m| m.family.ends_with(":i") || m.family.ends_with(":I")).and_then(|m| {
            let core = token.trim_matches(|c: char| !c.is_ascii_alphabetic());
            let c = core.chars().next().filter(|_| core.chars().count() == 1)?;
            let family = format!("{}{}", &m.family[..m.family.len() - 1], if c.is_ascii_uppercase() { 'A' } else { 'a' });
            let ordinal = u32::from(c.to_ascii_lowercase()) - u32::from('a');
            grammar.numbering_marker(token, Some(&legal_structure_model::NumberingMarker { family, ordinal, prefix: Vec::new() }))
        });
        alone
            .into_iter()
            .chain(letter)
            .map(|m| MarkerReading {
                family: word.map_or(m.family.clone(), |word| format!("{}:{}", word.to_lowercase(), m.family)),
                value: m.ordinal,
                prefix: m.prefix,
            })
            .collect::<Vec<_>>()
    };
    let mut found = read(first, None);
    if division && first.chars().count() > 1 && first.chars().all(char::is_alphabetic) {
        if let Some(number) = tokens.next() {
            found.extend(read(number.trim_end_matches(['.', ':', '\u{2014}', '-']), Some(first)));
        }
    }
    found.dedup();
    found
}

/// A line's reference tokens (`REFERENCE_*`), its type set at `size`.
fn reference_tokens(line: &Line, size: f64) -> u8 {
    let marks = reference_marks(line, size).fold(0, |found, (_, set)| found | set);
    marks | bits([(REFERENCE_DETACHED, !line.detached_references.is_empty()), (REFERENCE_GLUED, glued_re().is_match(&line.text))])
}

/// The line's opening marker (`Marker`), its family named in `families`.
fn opening_marker(analysis: &dyn StructureAnalysis, line: &Line, families: &mut Names) -> Option<Marker> {
    let text = closed_up(&line.text);
    let text = text.trim_start();
    let token = text.split_whitespace().next()?;
    if token.chars().count() > 12 {
        return None;
    }
    let reading = analysis.numbering_marker(token, None)?;
    let end = line.text.len() - text.len() + token.len();
    Some(Marker {
        family: families.id(&reading.family),
        value: reading.ordinal,
        depth: reading.prefix.len().min(usize::from(u8::MAX)) as u8,
        end: end.min(usize::from(u16::MAX)) as u16,
        only: text.trim_end() == token || legal_pdf_text::paragraphs::marker_only_re().is_match(&line.text),
    })
}

/// The page's line at `slot` opening with a note label (`NoteLabel`), the page's lines' type sizes
/// `sizes`.
pub fn note_label(page: &Page, sizes: &[f64], slot: usize) -> Option<NoteLabel> {
    let line = &page.lines[slot];
    // A label glued to its note's first word ("27Ibid", "1\u{201c}significant") is one where its figures
    // are set as a label's: raised, or smaller than the line's type.
    let (prefix, glued) = match label_prefix(&line.text) {
        Some(prefix) => (prefix, false),
        None => {
            // Recognition reads a note's asterisk as a middle dot ("\u{b7} 12 Stat. at Large"): one set as a
            // label's is the asterisk.
            let lead = line.text.len() - line.text.trim_start().len();
            if let Some(rest) = line.text[lead..].strip_prefix('\u{b7}').filter(|rest| rest.starts_with(' ')) {
                // Read as a glued label is: kept only where it is set as a label's (below).
                let prefix = label_prefix(&format!("{}*{rest}", &line.text[..lead]))?;
                (prefix, true)
            } else {
                let figures = line.text[lead..].chars().take(3).take_while(char::is_ascii_digit).count();
                let next = line.text[lead + figures..].chars().next()?;
                if figures == 0 || !(next.is_uppercase() || "\u{201c}\u{2018}\"'".contains(next)) {
                    return None;
                }
                (label_prefix(&format!("{} {}", &line.text[..lead + figures], &line.text[lead + figures..]))?, true)
            }
        }
    };
    let size = sizes[slot];
    let raised = label_is_raised(line, &prefix, size);
    let label_size = label_spans(line, &prefix)
        .filter_map(|span| (span.size > 0.0).then_some(span.size))
        .min_by(f64::total_cmp)
        .unwrap_or(size);
    if glued && !raised && !(label_size > 0.0 && label_size <= size * 0.85) {
        return None;
    }
    let suppress = bits([
        (SUPPRESS_FOOT, line.bbox[1] >= page.height * 0.94),
        (SUPPRESS_CORNER, line.bbox[1] >= page.height * 0.91 && line.bbox[0] >= page.width * 0.50 && !compact_note_line(&line.text)),
        (SUPPRESS_COMMA, line.text.chars().nth(prefix.end) == Some(',') && !raised),
        (SUPPRESS_WRAPPED, !raised && reference_wrapped_onto(&page.lines, sizes, slot)),
    ]);
    Some(NoteLabel {
        value: normalize_label(&prefix.label).parse().unwrap_or(0),
        end: prefix.end.min(usize::from(u16::MAX)) as u16,
        size: label_size as f32,
        raised,
        suppress,
    })
}

/// An instrument's operative words.
const OPERATIVE: &str = r"(?i)\b(?:shall|hereby|herein|hereto|hereof|hereunder|hereinafter|thereof|whereas|aux présentes|doit|doivent)\b";

/// A text's words (two letters or more), and how many of them are operative (`OPERATIVE`).
fn operative_words(text: &str) -> (usize, usize) {
    (regex!(r"[A-Za-zÀ-ÿ]{2,}").find_iter(text).count(), regex!(OPERATIVE).find_iter(text).count())
}

/// The evidence read off the pages: every page, line and the document measured once (`measure`), the citations on the
/// lines worth reading for them (`read_citations`), and what a line's text says read again where the runs' repair
/// rewrote its marker (`reread`).
pub trait Measure {
    #[allow(clippy::too_many_arguments)]
    fn measure(analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, normalized: Normalized, separators: &[Option<f64>],
        outline: &[PdfOutlineEntry], page_labels: &[Option<String>], instrument: bool) -> Self;
    fn read_citations(&mut self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, line_numbers: &[bool]);
    fn reread(&mut self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, repaired: &[(Slot, String)]);
}

impl Measure for Evidence {
    /// Every page, line and the document measured once: `separators` holds each page's note
    /// separator as the extraction found it, `outline` the document's bookmarks, `page_labels` the
    /// page labels it declares (by page index), and `instrument` whether its caller knows it for an
    /// instrument.
    fn measure(
        analysis: &dyn StructureAnalysis,
        pages: &[Page],
        slots: &Slots,
        normalized: Normalized,
        separators: &[Option<f64>],
        outline: &[PdfOutlineEntry],
        page_labels: &[Option<String>],
        instrument: bool,
    ) -> Self {
        let count = slots.len();
        let Normalized { line_numbers, items } = normalized;
        let mut evidence = Self { beside: vec![None; count], page_labels: page_labels.to_vec(), ..Self::default() };
        let (mut faces, mut families) = (Names::default(), Names::default());
        let (mut words, mut operative) = (0, 0);
        for (page_slot, page) in pages.iter().enumerate() {
            let first = slots.on(page_slot).start as usize;
            let rows = page_rows(page);
            let sizes: Vec<f64> = page.lines.iter().map(line_font_size).collect();
            for (at, line) in page.lines.iter().enumerate() {
                let text = line.text.trim();
                let face = dominant_face(line);
                evidence.size.push(sizes[at]);
                evidence.face.push(face.map_or(u16::MAX, |(face, _)| faces.id(face)));
                evidence.half_points.push(face.map_or(0, |(_, size)| (size * 2.0).round().clamp(0.0, f64::from(u16::MAX)) as u16));
                let underlined = !line.words.is_empty() && line.words.iter().filter(|word| word.underlined).count() * 2 > line.words.len();
                evidence.look.push(bits([
                    (LOOK_BOLD, bold_char_share(line) >= 0.5),
                    (LOOK_ITALIC, flag_share(line, ITALIC).unwrap_or(0.0) > 0.5),
                    (LOOK_CAPS, all_caps(text)),
                    (LOOK_UNDERLINED, underlined),
                ]));
                evidence.weight.push(line.weight as f32);
                evidence.tag.push(SourceTag::of(line));
                evidence.marker.push(opening_marker(analysis, line, &mut families));
                evidence.label.push(note_label(page, &sizes, at));
                evidence.references.push(reference_tokens(line, sizes[at]));
                evidence.title_shaped.push(title_shaped(text));
                evidence.words.push(lexicon(analysis, line));
                if !line_numbers[first + at] {
                    let (read, found) = operative_words(&line.text);
                    words += read;
                    operative += found;
                }
            }
            // A marker alone on its line, its text on the line beside it on its row.
            let mut by_row: Vec<Vec<usize>> = vec![Vec::new(); rows.count as usize];
            for (at, &row) in rows.of.iter().enumerate() {
                if row != u32::MAX {
                    by_row[row as usize].push(at);
                }
            }
            for (at, marker) in page.lines.iter().enumerate() {
                let only = evidence.marker[first + at].is_some_and(|reading| reading.only);
                if !only || line_numbers[first + at] || rows.of[at] == u32::MAX {
                    continue;
                }
                evidence.beside[first + at] = by_row[rows.of[at] as usize]
                    .iter()
                    .copied()
                    .filter(|&other| {
                        let line = &page.lines[other];
                        other != at
                            && !line_numbers[first + other]
                            && !line.text.trim().is_empty()
                            && legal_pdf_text::paragraphs::same_row(marker, line)
                            && line.bbox[0] >= marker.bbox[2] - if line.recognized() { line.bbox[3] - line.bbox[1] } else { 1.0 }
                    })
                    .min_by(|&a, &b| page.lines[a].bbox[0].total_cmp(&page.lines[b].bbox[0]))
                    .map(|other| slots.at(page_slot, other));
            }
            evidence.pages.push(PageEvidence { rows, separator: separators.get(page_slot).copied().flatten() });
        }
        // A heading tag on so many of the tagged lines that it marks the body is no heading's.
        let overused = headings::overused_tags(pages, &items, |page, line| {
            let slot = slots.at(page, line).index();
            (!line_numbers[slot] && evidence.tag[slot] != SourceTag::None).then(|| evidence.tag[slot].name())
        });
        if !overused.is_empty() {
            for tag in &mut evidence.tag {
                if matches!(tag, SourceTag::Heading(_)) && overused.contains(tag.name()) {
                    *tag = SourceTag::None;
                }
            }
        }
        evidence.read_citations(analysis, pages, slots, &line_numbers);
        evidence.outline_usable = headings::bookmarks::usable(outline, pages.len());
        evidence.outline_complete = headings::bookmarks::complete(outline, pages.len());
        evidence.instrument = instrument || words >= 200 && operative * 1000 >= words * 6;
        evidence.operative = if instrument { 1000.0 } else if words >= 200 { (operative * 1000) as f32 / words as f32 } else { 0.0 };
        evidence.faces = faces.names;
        evidence.families = families.names;
        evidence.line_number = line_numbers;
        evidence.items = items;
        evidence
    }

    /// The citations on the lines worth reading for them, read in one window: each line carrying a
    /// marker, a note label, a reference token or a figure (a number the note pairer could read as a
    /// reference, which a citation holding it protects), or shaped as a title, and every line of the
    /// first two pages, each with the line either side of it; then which of the first two pages'
    /// lines name the case by its citation.
    fn read_citations(&mut self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, line_numbers: &[bool]) {
        let count = slots.len();
        let mut wanted: Vec<bool> = (0..count)
            .map(|slot| {
                !line_numbers[slot]
                    && (self.marker[slot].is_some()
                        || self.label[slot].is_some()
                        || self.references[slot] != 0
                        || self.title_shaped[slot]
                        || slots.page(Slot(slot as u32)) < 2
                        || slots.line(pages, Slot(slot as u32)).text.chars().any(char::is_numeric))
            })
            .collect();
        let seeds = wanted.clone();
        for slot in (0..count).filter(|&slot| seeds[slot]) {
            wanted[slot.saturating_sub(1)] = true;
            wanted[(slot + 1).min(count - 1)] = true;
        }
        // The window: each run of wanted lines a paragraph of its own, line under line.
        let mut window = String::new();
        // Each windowed line's slot, and where it starts in the window, in bytes and in scalars.
        let mut placed: Vec<(Slot, usize, usize)> = Vec::new();
        let mut scalars = 0;
        let mut previous = None;
        for slot in (0..count).filter(|&slot| wanted[slot]) {
            if let Some(prior) = previous {
                let separator = if prior + 1 == slot { "\n" } else { "\n\n" };
                window.push_str(separator);
                scalars += separator.len();
            }
            let text = &slots.line(pages, Slot(slot as u32)).text;
            placed.push((Slot(slot as u32), window.len(), scalars));
            window.push_str(text);
            scalars += text.chars().count();
            previous = Some(slot);
        }
        let mut found: Vec<Cited> = Vec::new();
        if !window.is_empty() {
            let placed = &placed;
            let line_text = |at: usize| &slots.line(pages, placed[at].0).text;
            // Each windowed line a byte range overlaps, where the line starts in the window, and the
            // range's part on it.
            let on_lines = |start: usize, end: usize| {
                let first = placed.partition_point(|&(_, byte, _)| byte <= start).saturating_sub(1);
                (first..placed.len())
                    .take_while(move |&at| placed[at].1 < end)
                    .filter_map(move |at| {
                        let (slot, byte, _) = placed[at];
                        let length = line_text(at).len();
                        let (from, to) = (start.max(byte), end.min(byte + length));
                        (from < to).then(|| (slot, byte as u32, (from - byte) as u32, (to - byte) as u32))
                    })
            };
            for citation in analysis.citations(&window) {
                for (slot, byte, from, to) in on_lines(citation.range.start, citation.range.end) {
                    let core_from = (citation.core_range.start as u32).saturating_sub(byte).clamp(from, to);
                    let core_to = (citation.core_range.end as u32).saturating_sub(byte).clamp(core_from, to);
                    found.push(Cited { slot, range: (from, to), core: (core_from, core_to), protected: false });
                }
            }
            // The protected spans, in scalars: each read back to bytes on its lines.
            for (start, end) in analysis.protected_citation_spans(&window) {
                let first = placed.partition_point(|&(_, _, scalar)| scalar <= start).saturating_sub(1);
                for at in (first..placed.len()).take_while(|&at| placed[at].2 < end) {
                    let (slot, _, scalar) = placed[at];
                    let text = line_text(at);
                    let length = text.chars().count();
                    let (from, to) = (start.max(scalar) - scalar, end.min(scalar + length).saturating_sub(scalar));
                    if from >= to {
                        continue;
                    }
                    let byte = |offset: usize| text.char_indices().nth(offset).map_or(text.len(), |(byte, _)| byte) as u32;
                    let range = (byte(from), byte(to));
                    found.push(Cited { slot, range, core: range, protected: true });
                }
            }
        }
        found.sort_by_key(|cited| (cited.slot, cited.range.0, cited.range.1));
        let mut cited = Vec::with_capacity(count + 1);
        let mut at = 0;
        for slot in 0..count {
            cited.push(at as u32);
            while at < found.len() && found[at].slot.index() == slot {
                at += 1;
            }
        }
        cited.push(found.len() as u32);
        self.citations = found;
        self.cited = cited;
        // A citation naming the case on the first two pages.
        let opening = if pages.is_empty() { 0 } else { slots.on(pages.len().min(2) - 1).end as usize };
        for slot in (0..opening).filter(|&slot| !line_numbers[slot]) {
            let line = slots.line(pages, Slot(slot as u32));
            let cites = self.citations_of(Slot(slot as u32));
            let titled = captions::citation_title(line, |text, offset| {
                let (start, end) = (offset as u32, (offset + text.len()) as u32);
                cites
                    .iter()
                    .filter(|cited| !cited.protected && cited.range.0 >= start && cited.range.1 <= end)
                    .map(|cited| CitationSpan {
                        range: (cited.range.0 - start) as usize..(cited.range.1 - start) as usize,
                        core_range: (cited.core.0.max(start) - start) as usize..(cited.core.1.max(start) - start) as usize,
                        text: String::new(),
                    })
                    .collect()
            });
            if titled.is_some() {
                self.words[slot] |= WORD_CITATION_TITLE;
            }
        }
    }

    /// Reads again what a line's text says of it, for the lines whose opening marker the runs'
    /// repair rewrote (`repaired`: each slot with its text before): its marker, note label, shape and
    /// lexicon, and its citations moved with the words after the marker.
    fn reread(&mut self, analysis: &dyn StructureAnalysis, pages: &[Page], slots: &Slots, repaired: &[(Slot, String)]) {
        if repaired.is_empty() {
            return;
        }
        let mut families = Names {
            ids: self.families.iter().enumerate().map(|(id, name)| (name.clone(), id as u16)).collect(),
            names: std::mem::take(&mut self.families),
        };
        for (slot, before) in repaired {
            let at = slot.index();
            let page_slot = slots.page(*slot);
            let range = slots.on(page_slot);
            let page = &pages[page_slot];
            let line = slots.line(pages, *slot);
            let text = line.text.trim();
            self.marker[at] = opening_marker(analysis, line, &mut families);
            self.label[at] = note_label(page, &self.size[range.start as usize..range.end as usize], (slot.0 - range.start) as usize);
            self.title_shaped[at] = title_shaped(text);
            self.words[at] = lexicon(analysis, line) | self.words[at] & WORD_CITATION_TITLE;
            // The words after the rewritten marker keep their citations, moved by what the marker
            // gained or lost.
            let after = before.len() - before.bytes().rev().zip(line.text.bytes().rev()).take_while(|(a, b)| a == b).count();
            let shift = line.text.len() as i64 - before.len() as i64;
            let length = line.text.len() as u32;
            let moved = |offset: u32| {
                let offset = if offset as usize >= after { (i64::from(offset) + shift).max(0) as u32 } else { offset };
                offset.min(length)
            };
            let (start, end) = (self.cited[at] as usize, self.cited[at + 1] as usize);
            for cited in &mut self.citations[start..end] {
                cited.range = (moved(cited.range.0), moved(cited.range.1));
                cited.core = (moved(cited.core.0), moved(cited.core.1));
            }
        }
        self.families = families.names;
    }
}
