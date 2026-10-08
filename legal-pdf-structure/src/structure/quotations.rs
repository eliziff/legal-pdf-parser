//! Display quotations: another text's words a document sets apart from its own.
//!
//! A display quotation is set in from the body's left margin, and no one feature makes a block
//! of set-in lines one: a document's own lists, provisions and tables are set in too. A block
//! is read as a quotation when it opens with a quotation mark that is not a defined term's and
//! an introduction, a contrast of type or a closing mark supports it; when it is set in a
//! smaller or italic type than the body after a line introducing it; or when it is also set in
//! on the right, with its own measure, after a line announcing words to follow ("as follows:",
//! "provides:"). A quotation runs on over a page break, and holds its own numbered paragraphs
//! and lists.

use super::PdfPrimitiveEvidence;
use legal_pdf_core::model::{Line, Page, Span};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// A row of body text: a line and any lines set beside it on the same row (a marker set apart
/// from its paragraph's first line).
struct Row<'a> {
    lines: Vec<&'a Line>,
    page_index: usize,
    bbox: [f64; 4],
}

impl Row<'_> {
    fn width(&self) -> f64 {
        self.bbox[2] - self.bbox[0]
    }

    fn text(&self) -> String {
        let mut lines = self.lines.clone();
        lines.sort_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]));
        lines
            .iter()
            .map(|line| line.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn spans(&self) -> impl Iterator<Item = &Span> {
        self.lines.iter().flat_map(|line| &line.spans)
    }
}

fn italic(span: &Span) -> bool {
    let font = span.font.to_ascii_lowercase();
    span.flags & 2 != 0 || font.contains("italic") || font.contains("oblique")
}

/// The share of the rows' visible characters set in italics, and their mean type size, from the
/// rows' spans.
fn typography<'a>(rows: impl IntoIterator<Item = &'a Row<'a>>) -> (f64, f64) {
    let (mut characters, mut italics, mut size) = (0.0, 0.0, 0.0);
    for span in rows.into_iter().flat_map(Row::spans) {
        let count = span.text.trim().chars().count() as f64;
        if span.size > 0.0 {
            characters += count;
            size += count * span.size;
            if italic(span) {
                italics += count;
            }
        }
    }
    if characters == 0.0 {
        (0.0, 0.0)
    } else {
        (italics / characters, size / characters)
    }
}

fn defined_term_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            "^.[^\u{201d}\"\u{2019}]{1,60}[\u{201d}\"\u{2019}]\\s*,?\\s*(?:means|includes|has|refers|shall|is|are)\\b",
        )
        .unwrap()
    })
}

fn introduction_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("(?::|\u{2014}|\u{2013})\\s*[-\u{2013}\u{2014}]?$").unwrap())
}

fn announcement_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            "(?i)\\b(?:follows?|provides?|provided|states?|stated|stating|reads?|held|holds|said|says|observed|noted|explained|concluded|found|wrote|written|set out|defines?|defined|as|that|thus|this)\\s*(?::|\u{2014}|\u{2013})\\s*[-\u{2013}\u{2014}]?$",
        )
        .unwrap()
    })
}

fn closing_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new("[\u{201d}\"\u{2019}]\\s*[.,;:]?\\s*(?:\\[[^\\]]*\\]|\\([^)]*\\))?\\s*[.,;:]?$")
            .unwrap()
    })
}

fn item_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("^\\s*(?:\\(?\\w{1,4}[.)]|[\u{2022}\u{25cf}-])\\s").unwrap())
}

/// The ids of the body lines set as display quotations.
pub(super) fn quoted_lines(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> HashSet<String> {
    let mut ordered_pages = pages.iter().collect::<Vec<_>>();
    ordered_pages.sort_by_key(|page| page.index);
    let mut rows = Vec::<Row>::new();
    for page in &ordered_pages {
        let mut lines = page
            .lines
            .iter()
            .filter(|line| {
                !line.exclude_from_body
                    && line.note_region_mode.is_empty()
                    && !matches!(line.region_type.as_str(), "header" | "footer" | "heading")
                    && !primitives.table_cell_line_ids.contains(&line.id)
                    && !line.text.trim().is_empty()
            })
            .collect::<Vec<_>>();
        lines.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        for line in lines {
            let height = (line.bbox[3] - line.bbox[1]).max(1.0);
            match rows.last_mut() {
                Some(row)
                    if row.page_index == page.index
                        && row.bbox[3].min(line.bbox[3]) - row.bbox[1].max(line.bbox[1])
                            > 0.5 * height =>
                {
                    row.lines.push(line);
                    row.bbox = [
                        row.bbox[0].min(line.bbox[0]),
                        row.bbox[1].min(line.bbox[1]),
                        row.bbox[2].max(line.bbox[2]),
                        row.bbox[3].max(line.bbox[3]),
                    ];
                }
                _ => rows.push(Row {
                    lines: vec![line],
                    page_index: page.index,
                    bbox: line.bbox,
                }),
            }
        }
    }
    let Some(widest) = rows.iter().map(Row::width).reduce(f64::max) else {
        return HashSet::new();
    };
    // The body's measure: where its full lines start and end, on each side of a spread.
    let long = rows
        .iter()
        .filter(|row| row.width() >= 0.6 * widest)
        .collect::<Vec<_>>();
    let measure_of = |rows: &[&Row]| -> Option<(f64, f64)> {
        let mut lefts = HashMap::<i64, usize>::new();
        for row in rows {
            *lefts.entry(row.bbox[0].round() as i64).or_default() += 1;
        }
        let left = lefts
            .into_iter()
            .max_by_key(|(left, count)| (*count, -*left))?
            .0 as f64;
        let mut rights = rows.iter().map(|row| row.bbox[2]).collect::<Vec<_>>();
        rights.sort_by(f64::total_cmp);
        Some((left, rights[(rights.len() - 1) * 9 / 10]))
    };
    let sides = [0, 1].map(|parity| {
        let side = long
            .iter()
            .filter(|row| row.page_index % 2 == parity)
            .copied()
            .collect::<Vec<_>>();
        measure_of(if side.is_empty() { &long } else { &side })
    });
    let measures = ordered_pages
        .iter()
        .filter_map(|page| {
            let (left, right) = sides[page.index % 2]?;
            let own = long
                .iter()
                .filter(|row| row.page_index == page.index)
                .copied()
                .collect::<Vec<_>>();
            // A page laid out further left than the body's margin has a measure of its own.
            let measure = if own.iter().any(|row| row.bbox[0] < left - 4.0) {
                let (left, _) = measure_of(&own)?;
                let right = own.iter().map(|row| row.bbox[2]).fold(f64::MIN, f64::max);
                (left, right)
            } else {
                (left, right)
            };
            Some((page.index, measure))
        })
        .collect::<HashMap<_, _>>();
    let measure = |row: &Row| measures[&row.page_index];
    // The body's own type, from the rows set at its margin.
    let margin_rows = rows
        .iter()
        .filter(|row| (row.bbox[0] - measure(row).0).abs() <= 6.0)
        .collect::<Vec<_>>();
    let mut sizes = HashMap::<i64, usize>::new();
    for span in margin_rows.iter().flat_map(|row| row.spans()) {
        if span.size > 0.0 {
            *sizes.entry((span.size * 2.0).round() as i64).or_default() +=
                span.text.trim().chars().count();
        }
    }
    let Some(body_size) = sizes
        .into_iter()
        .max_by_key(|(size, count)| (*count, *size))
        .map(|(size, _)| size as f64 / 2.0)
    else {
        return HashSet::new();
    };
    let (body_italic, _) = typography(margin_rows.iter().copied());
    let full_margin_rows = margin_rows
        .iter()
        .filter(|row| {
            let (left, right) = measure(row);
            row.width() >= 0.6 * (right - left)
        })
        .collect::<Vec<_>>();
    let justified = full_margin_rows.len() >= 5
        && 10
            * full_margin_rows
                .iter()
                .filter(|row| (row.bbox[2] - measure(row).1).abs() <= 0.5 * body_size)
                .count()
            >= 6 * full_margin_rows.len();

    let set_in = |row: &Row| row.bbox[0] - measure(row).0 >= 1.5 * body_size;
    // A running head or folio the page's furniture left in the body sits in its top or bottom
    // tenth; a block set in on both sides of it goes on over the page break.
    let heights = ordered_pages
        .iter()
        .map(|page| (page.index, page.height))
        .collect::<HashMap<_, _>>();
    let edge = |row: &Row| {
        let height = heights[&row.page_index];
        let middle = (row.bbox[1] + row.bbox[3]) / 2.0;
        middle < 0.1 * height || middle > 0.9 * height
    };
    let mut quoted = HashSet::new();
    let mut start = 0;
    while start < rows.len() {
        if !set_in(&rows[start]) {
            start += 1;
            continue;
        }
        let mut block = vec![&rows[start]];
        let mut end = start + 1;
        loop {
            let mut next = end;
            while next < rows.len() && !set_in(&rows[next]) && edge(&rows[next]) {
                next += 1;
            }
            if next < rows.len()
                && set_in(&rows[next])
                && (next == end || rows[next].page_index != rows[end - 1].page_index)
            {
                block.push(&rows[next]);
                end = next + 1;
            } else {
                break;
            }
        }
        let introduction = start
            .checked_sub(1)
            .map(|before| &rows[before])
            .filter(|row| row.bbox[0] <= measure(row).0 + 3.0 * body_size)
            .map(Row::text)
            .filter(|text| text.chars().any(char::is_lowercase));
        if quotation(
            &block,
            introduction.as_deref(),
            body_size,
            body_italic,
            justified,
            &measure,
        ) {
            quoted.extend(
                block
                    .iter()
                    .flat_map(|row| &row.lines)
                    .map(|line| line.id.clone()),
            );
        }
        start = end;
    }
    quoted
}

/// Whether a block of rows set in from the body's margin is a display quotation; `introduction`
/// is the text of the body row before it.
fn quotation(
    block: &[&Row],
    introduction: Option<&str>,
    body_size: f64,
    body_italic: f64,
    justified: bool,
    measure: &dyn Fn(&Row) -> (f64, f64),
) -> bool {
    // A centred block (an address, a title) or one reaching into the right half of the page (a
    // table's columns) is no quotation.
    let mut centres = block
        .iter()
        .map(|row| (row.bbox[0] + row.bbox[2]) / 2.0)
        .collect::<Vec<_>>();
    centres.sort_by(f64::total_cmp);
    let centre = centres[centres.len() / 2];
    let lefts = block.iter().map(|row| row.bbox[0]);
    let spread = lefts.clone().fold(f64::MIN, f64::max) - lefts.fold(f64::MAX, f64::min);
    let centred = block.len() >= 3
        && 3 * block
            .iter()
            .filter(|row| ((row.bbox[0] + row.bbox[2]) / 2.0 - centre).abs() <= 0.5 * body_size)
            .count()
            >= 2 * block.len()
        && spread > body_size;
    if centred
        || block.iter().any(|row| {
            let (left, right) = measure(row);
            row.bbox[0] >= (left + right) / 2.0
        })
    {
        return false;
    }
    let (italics, size) = typography(block.iter().copied());
    let contrast = (italics >= 0.6 && body_italic < 0.25)
        || (size >= 0.6 * body_size && size <= body_size - 0.75);
    // The opening mark may follow the quoted text's own first number ("1. “An order ...").
    let first = block[0].text();
    let first = item_marker_re()
        .find(&first)
        .map_or(first.as_str(), |marker| &first[marker.end()..])
        .trim_start();
    let closes = closing_re().is_match(block[block.len() - 1].text().trim_end());
    // The mark opens the block's text, not a phrase the line quotes and closes ("(i) “or, in
    // the” in the first line;"), nor a defined term.
    let (openings, closings) = match first.chars().next() {
        Some('\u{201c}' | '\u{201e}') => (
            first.matches(['\u{201c}', '\u{201e}']).count(),
            first.matches('\u{201d}').count(),
        ),
        Some('"') => (1, 1 - first.matches('"').count() % 2),
        Some('\u{2018}') => (
            first.matches('\u{2018}').count(),
            // A mark between letters is an apostrophe.
            first
                .char_indices()
                .filter(|(at, mark)| {
                    *mark == '\u{2019}'
                        && !first[at + mark.len_utf8()..]
                            .starts_with(|next: char| next.is_alphabetic())
                })
                .count(),
        ),
        Some('\u{ab}') => (
            first.matches('\u{ab}').count(),
            first.matches('\u{bb}').count(),
        ),
        _ => (0, 0),
    };
    let opens = openings > 0
        && (openings > closings || (block.len() == 1 && closes))
        && !defined_term_re().is_match(first);
    let introduced = introduction.is_some_and(|text| introduction_re().is_match(text.trim_end()));
    let announced = introduction.is_some_and(|text| announcement_re().is_match(text.trim_end()));
    if (opens && (introduced || contrast || closes)) || (contrast && introduced && block.len() >= 2)
    {
        return true;
    }
    // Set in on the right too: its lines end short of the body's, at a measure of their own. A
    // ragged body's lines end short anyway, so there the next line's first word must have fitted.
    if !(announced || contrast) || block.len() < 3 {
        return false;
    }
    let short = block
        .iter()
        .all(|row| measure(row).1 - row.bbox[2] >= 1.5 * body_size);
    let wide = block[..block.len() - 1]
        .iter()
        .filter(|row| {
            let (left, right) = measure(row);
            row.width() >= 0.5 * (right - left)
        })
        .count();
    let wraps = block
        .windows(2)
        .filter(|pair| pair[0].page_index == pair[1].page_index)
        .filter(|pair| !item_marker_re().is_match(&pair[1].text()))
        .collect::<Vec<_>>();
    let fitted = wraps
        .iter()
        .filter(|pair| {
            let next = &pair[1];
            let first_word = next
                .lines
                .iter()
                .min_by(|a, b| a.bbox[0].total_cmp(&b.bbox[0]))
                .and_then(|line| line.words.first())
                .map_or(0.0, |word| word.bbox[2] - word.bbox[0]);
            first_word > 0.0
                && pair[0].bbox[2] + first_word + 0.5 * body_size <= measure(&pair[0]).1
        })
        .count();
    short && wide >= 2 && (justified || (fitted >= 2 && 2 * fitted >= wraps.len()))
}
