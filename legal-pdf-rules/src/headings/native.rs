//! What a line's text and its place among its page's lines say of it as a heading reads it: a line
//! carrying a sentence on from the line above it, a full line whose sentence the next line
//! finishes, a word hyphenated or a sentence going on across two lines, a date alone, a page whose
//! text was recognized from its image, and the heading tags a source puts on so many lines that
//! they mark its body.

use crate::model::SourceTag;
use crate::text::text::{
    bracket_excess, closes_before_opening, dangling_end, hyphen_continuation, hyphen_fragment_tail,
    starts_note_or_list,
};
use legal_pdf_core::model::sentence_ended;
use legal_pdf_core::model::{Line, Page};
use legal_pdf_core::regex;
use legal_pdf_core::{flag_share, off_grid_sizes, BOLD};
use std::collections::{HashMap, HashSet};

/// A line set at its paragraph's leading under the line before it (`prior`), a full line that ends
/// no sentence, carries that sentence on ("... Hogg and" / "A. A. Bushell, ..."): its initial is no
/// enumerator. A title above it stops short of its column's edge. The line before is the one the
/// reading order puts there, so a scan's overlapping boxes never stand another line in for it: one
/// whose box reaches over a quarter of the line's height into it is none. The page's body lines,
/// those `body` names, give the column's measure (`prior` and `slot` are slots of the page's lines).
pub fn continues_prose_from(
    page: &Page,
    prior: usize,
    slot: usize,
    body: impl Fn(usize) -> bool,
) -> bool {
    let (prior, line) = (&page.lines[prior], &page.lines[slot]);
    let height = line.bbox[3] - line.bbox[1];
    if prior.exclude_from_body
        || prior.bbox[3] > line.bbox[1] + height * 0.25
        || prior.bbox[0] >= line.bbox[2]
        || line.bbox[0] >= prior.bbox[2]
    {
        return false;
    }
    let text = prior.text.trim_end();
    if height <= 0.0
        || line.bbox[1] - prior.bbox[3] >= height * 0.5
        || sentence_ended(text)
        || text.ends_with([':', '\u{201d}', '"'])
        || crate::evidence::enumerated(text)
    {
        return false;
    }
    let uncalled = text
        .trim_end_matches(|c: char| c.is_ascii_digit() || matches!(c, '*' | '\u{2020}' | '\u{2021}' | '\u{00a7}'));
    let ended_before_call = uncalled.len() < text.len()
        && !uncalled.ends_with(char::is_whitespace)
        && (sentence_ended(uncalled) || uncalled.ends_with([':', '\u{201d}', '"', '\u{bb}']));
    if ended_before_call {
        return false;
    }
    let (left, right) = column_bounds(page, prior, body);
    let indented_start =
        line.bbox[0] - left >= height * 0.75 && line.bbox[0] - prior.bbox[0] >= height * 0.5;
    !indented_start && prior.bbox[2] >= right - (right - left) * 0.1
}

/// The horizontal `[left, right]` span of the page's body column overlapping `line`.
fn column_bounds(page: &Page, line: &Line, body: impl Fn(usize) -> bool) -> (f64, f64) {
    page.lines
        .iter()
        .enumerate()
        .filter(|(other_slot, other)| {
            body(*other_slot) && other.bbox[0] < line.bbox[2] && line.bbox[0] < other.bbox[2]
        })
        .fold((line.bbox[0], line.bbox[2]), |(left, right), (_, other)| {
            (left.min(other.bbox[0]), right.max(other.bbox[2]))
        })
}

/// Whether a line runs its column: the page's body lines beside it, those `body` names, give the column's measure.
fn runs_the_column(page: &Page, line: &Line, body: impl Fn(usize) -> bool) -> bool {
    let (left, right) = column_bounds(page, line, body);
    line.bbox[2] >= right - (right - left) * 0.1
}

/// A long numbered line running the full column whose sentence the next line, at its leading,
/// finishes ("1. IAEA, Safety Standards Series, Geological Disposal Facilities for" /
/// "Radioactive Waste, Vienna, 2011.") opens a paragraph or an entry, not a heading: the line at
/// `slot` onto the line `next` (slots of the page's lines), the page's body lines those `body` names.
pub fn carries_a_sentence_onto(
    page: &Page,
    slot: usize,
    next: usize,
    body: impl Fn(usize) -> bool,
) -> bool {
    let line = &page.lines[slot];
    let height = line.bbox[3] - line.bbox[1];
    let followed_by_body = body(next);
    let next = &page.lines[next];
    let text = line.text.trim_end();
    let following = next.text.trim_end();
    height > 0.0
        && text.split_whitespace().count() >= 8
        && !sentence_ended(text)
        && !text.ends_with(':')
        && followed_by_body
        && next.bbox[1] > line.bbox[1]
        && next.bbox[1] - line.bbox[3] < height * 0.5
        && sentence_ended(following)
        && !following.ends_with('?')
        && runs_the_column(page, line, body)
        && !legal_structure_lines::analysis::has_citation_signal(&format!("{text} {following}"))
}

/// The share of a line's characters its face sets bold.
pub fn bold_char_share(line: &Line) -> f64 {
    flag_share(line, BOLD).unwrap_or(0.0)
}

/// A word hyphenated across the two lines, an unfinished sentence going on in lower case, or on a
/// line ending on a word no sentence ends on (`dangling_end`, "... the criminal law of" / "England
/// continued"), whatever the line goes on with, unless the line opens something of its own: an
/// item's marker, a speaker's turn, a judge's name after "delivered by", or a smaller or bolder
/// type. Or a bracket the line closes.
pub fn body_flow_edge(previous: &Line, current: &Line) -> bool {
    let previous_text = previous.text.trim();
    let current_text = current.text.trim();
    if previous_text.is_empty() || current_text.is_empty() {
        return false;
    }
    if hyphen_fragment_tail(previous_text) && hyphen_continuation(current_text) {
        return true;
    }
    // An enumeration within the sentence the line above leaves open ("two components:" / "(1) the risk …; and
    // (2) the gravity …"): the line prints its opening marker's successor too, as no list's item does.
    if !sentence_ended(previous_text) && inline_enumeration(current_text) {
        return true;
    }
    if starts_note_or_list(current_text) {
        return false;
    }
    if !sentence_ended(previous_text) && current_text.chars().next().is_some_and(char::is_lowercase)
    {
        return true;
    }
    // A lead-in's last words wrapped onto a line of their own ("… codified in s. 19 of the Criminal" / "Code:"): a
    // few words closing on a colon go on with the sentence the line above leaves open.
    if !sentence_ended(previous_text)
        && current_text.ends_with(':')
        && current_text.split_whitespace().count() <= 4
    {
        return true;
    }
    if dangling_end(previous_text) && !opens_after_dangling(previous, current) {
        return true;
    }
    bracket_excess(previous_text) > 0 && closes_before_opening(current_text)
}

/// A line opening on a bracketed marker ("(1)", "(a)", "(i)") that prints the marker after it later in the
/// same line ("(2)", "(b)", "(ii)"): an enumeration run into a sentence, not a list's item.
fn inline_enumeration(text: &str) -> bool {
    const ROMANS: [&str; 12] = [
        "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii",
    ];
    let Some((token, rest)) = text.strip_prefix('(').and_then(|rest| rest.split_once(')')) else {
        return false;
    };
    if !rest.starts_with(char::is_whitespace) {
        return false;
    }
    let next = if let Ok(number) = token.parse::<u32>() {
        (number + 1).to_string()
    } else if let Some(at) = ROMANS
        .iter()
        .position(|roman| roman.eq_ignore_ascii_case(token))
    {
        match ROMANS.get(at + 1) {
            Some(next) if token.chars().all(char::is_lowercase) => (*next).to_owned(),
            Some(next) => next.to_uppercase(),
            None => return false,
        }
    } else if let [letter] = token.as_bytes() {
        if !letter.is_ascii_alphabetic() || matches!(letter, b'z' | b'Z') {
            return false;
        }
        char::from(letter + 1).to_string()
    } else {
        return false;
    };
    rest.contains(&format!("({next}) ")) || rest.trim_end().ends_with(&format!("({next})"))
}

/// Whether the line after one ending on a word no sentence ends on opens something of its own.
fn opens_after_dangling(previous: &Line, current: &Line) -> bool {
    let named = regex!(r"(?i)(?:(?:delivered|given|rendered|written) by|rendue? par):?$");
    let judge = regex!(
        r"\b(?:J|JJ|C\.J|J\.A|JJ\.A|C\.J\.[A-Z]+|J\.C\.[A-Z])\.?(?:\s*[-\u{2014}\u{2013}:]\s*)?$|^(?:LE |LA |LES )?JUGES?\b"
    );
    let text = current.text.trim();
    let (size, before) = (current.size, previous.size);
    crate::text::paragraphs::opens_item(text)
        || crate::layout::opens_turn(current)
        || named.is_match(previous.text.trim())
        || text.split_whitespace().count() <= 8 && judge.is_match(text)
        || size > 0.0 && before > 0.0 && size < before * 0.9
        || bold_char_share(current) > bold_char_share(previous) + 0.5
}

/// A line that is only a date: "10 JUNE 2026", "DATED 11 JUNE 2026", "June 29, 2020", "1/4/2025".
pub fn date_line(text: &str) -> bool {
    const MONTH: &str = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?|janvier|f[ée]vrier|mars|avril|mai|juin|juillet|ao[uû]t|septembre|octobre|novembre|d[ée]cembre)";
    regex!(&format!(
        r"(?i)^(?:dated\s+(?:this\s+)?|le\s+)?(?:\d{{1,2}}(?:st|nd|rd|th|er)?\s+(?:day\s+of\s+)?{MONTH}\.?,?\s+\d{{4}}|{MONTH}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{4}}|\d{{1,2}}[/.-]\d{{1,2}}[/.-]\d{{2,4}})$"
    ))
    .is_match(text.trim())
}

/// A page whose text was recognized from its image: lines with recognition evidence or no spans,
/// or a text layer set in one face throughout with no bold or italic (a recognizer writes one
/// generic face and sizes its boxes; a typeset page changes face or weight somewhere).
pub fn recognized_page(page: &Page) -> bool {
    let body: Vec<&Line> = page
        .lines
        .iter()
        .filter(|line| !line.exclude_from_body)
        .collect();
    if body.is_empty() {
        return false;
    }
    // Only a scan's lines are measured on its image.
    if body
        .iter()
        .any(|line| line.ocr.is_some() || line.spans.is_empty() || line.cap_height > 0.0)
    {
        return true;
    }
    let mut fonts: HashSet<&str> = HashSet::new();
    let mut styled = false;
    for span in body.iter().flat_map(|line| &line.spans) {
        fonts.insert(span.font.as_str());
        styled |= span.flags & (16 | 2) != 0;
    }
    // A tab divider ("Tab" / "1") or a cover of two lines in one face says nothing about the layer.
    (body.len() >= 8 && fonts.len() == 1 && !styled) || off_grid_sizes(body.iter().copied()) >= 5
}

/// The heading tags a document puts on so many of its tagged lines that they mark its body (every
/// numbered paragraph tagged H2), as pdf-inspector reads them
/// (`detect_overused_struct_heading_levels`: a level on over 15% of 20 tagged lines or more): each
/// line's tag (`role`, by page slot and line slot) read on the pages' items (`page_items`); each overused level as a
/// source tag's heading level (0, H, is overused with H1; a tag past H6 reads as H6, as its name is written).
pub fn overused_tags(
    pages: &[Page],
    items: &[crate::model::PageItems],
    role: impl Fn(usize, usize) -> Option<SourceTag>,
) -> HashSet<u8> {
    use pdf_inspector::structure_tree::StructRole;
    use pdf_inspector::types::TextLine;
    let mut roles: HashMap<u32, HashMap<i64, StructRole>> = HashMap::new();
    let mut lines: Vec<TextLine> = Vec::new();
    for (page_slot, (page, items)) in pages.iter().zip(items).enumerate() {
        let number = page.index as u32 + 1;
        for (item, &(slot, _)) in items.items.iter().zip(&items.origin) {
            let Some(role) = role(page_slot, slot) else {
                continue;
            };
            let mut item = item.clone();
            item.mcid = Some(slot as i64);
            roles
                .entry(number)
                .or_default()
                .insert(slot as i64, StructRole::from_name(role.name()));
            match lines.last_mut() {
                Some(line) if line.page == number && line.items[0].mcid == item.mcid => {
                    line.items.push(item)
                }
                _ => lines.push(TextLine {
                    items: vec![item],
                    y: 0.0,
                    page: number,
                    adaptive_threshold: 0.1,
                }),
            }
        }
    }
    pdf_inspector::markdown::detect_overused_struct_heading_levels(&lines, Some(&roles))
        .into_iter()
        .filter_map(|level| u8::try_from(level).ok())
        .flat_map(|level| [Some(level), (level == 1).then_some(0)])
        .flatten()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_citations as _;
    use legal_citations_data as _;

    fn line_at(text: &str, left: f64, top: f64, right: f64, bottom: f64) -> Line {
        Line {
            text: text.to_owned(),
            bbox: [left, top, right, bottom],
            ..Default::default()
        }
    }

    fn page_with(lines: Vec<Line>) -> Page {
        Page {
            width: 612.0,
            height: 792.0,
            lines,
            ..Default::default()
        }
    }

    #[test]
    fn continues_prose_from_stops_after_trailing_footnote_call() {
        let page = page_with(vec![
            line_at("The court followed earlier cases.32", 72.0, 100.0, 540.0, 112.0),
            line_at("Later panels took a narrower view.", 72.0, 114.0, 540.0, 126.0),
        ]);
        assert!(!continues_prose_from(&page, 0, 1, |_| true));

        let page_quote_call = page_with(vec![
            line_at("Counsel relied on those conditions.”19", 72.0, 100.0, 540.0, 112.0),
            line_at("A separate argument followed.", 72.0, 114.0, 540.0, 126.0),
        ]);
        assert!(!continues_prose_from(&page_quote_call, 0, 1, |_| true));

        let page_section_ref = page_with(vec![
            line_at("Parliament set the rule out in s. 32", 72.0, 100.0, 540.0, 112.0),
            line_at("of the Act without further qualification.", 72.0, 114.0, 540.0, 126.0),
        ]);
        assert!(continues_prose_from(&page_section_ref, 0, 1, |_| true));
    }

    #[test]
    fn continues_prose_from_stops_on_first_line_indent_but_keeps_block_quotes() {
        let page_indented_para = page_with(vec![
            line_at("The tribunal reviewed the statutory scheme and", 72.0, 100.0, 540.0, 112.0),
            line_at("In the second stage of its analysis, the board", 90.0, 114.0, 540.0, 126.0),
        ]);
        assert!(!continues_prose_from(&page_indented_para, 0, 1, |_| true));

        let page_block_quote = page_with(vec![
            line_at("Body anchor setting the column bounds.", 72.0, 86.0, 540.0, 98.0),
            line_at("Where an administrative board considers", 108.0, 100.0, 504.0, 112.0),
            line_at("an application within its statutory mandate,", 108.0, 114.0, 504.0, 126.0),
        ]);
        assert!(continues_prose_from(&page_block_quote, 1, 2, |_| true));

        let page_wrapped = page_with(vec![
            line_at("The tribunal reviewed the statutory scheme and", 72.0, 100.0, 540.0, 112.0),
            line_at("rejected the preliminary objection below.", 72.0, 114.0, 540.0, 126.0),
        ]);
        assert!(continues_prose_from(&page_wrapped, 0, 1, |_| true));
    }
}
