//! Display quotations: another text's words a document sets apart from its own.
//!
//! A display quotation is set in from the body's left margin, and no one feature makes a block
//! of set-in lines one: a document's own lists, provisions and tables are set in too. A block
//! is read as a quotation when it opens with a quotation mark that is not a defined term's and
//! an introduction, a contrast of type or a closing mark supports it; when it is set in a
//! smaller or italic type than the body after a line introducing it; or when it is also set in
//! on the right, with its own measure, after a line announcing words to follow ("as follows:",
//! "provides:"). A headed extract may instead follow a cited passage ending in a pinpoint.
//! A quotation runs on over a page break and keeps its own headings, numbered paragraphs and lists;
//! deeper insets are read against that passage's margins.

use super::{DocumentContext, PdfPrimitiveEvidence};
use legal_pdf_core::layout_analysis;
use legal_pdf_core::model::{Line, Page, Span};
use regex::Regex;
use std::collections::{BTreeMap, HashMap};
use std::sync::OnceLock;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct Reading {
    pub(super) scopes: HashMap<String, String>,
    pub(super) introductions: HashMap<String, String>,
}

struct Introduction {
    first_line_id: String,
    text: String,
    last_line: String,
}

/// Quotations keep their own reading; physical document identities remain unchanged.
pub(super) fn reading_lines(
    pages: &[Page],
    primitives: &PdfPrimitiveEvidence,
) -> HashMap<String, usize> {
    if primitives.quotations.scopes.is_empty() {
        return primitives.document_line_ids.clone();
    }
    let mut next = primitives
        .document_line_ids
        .values()
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut quotes = HashMap::new();
    pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| {
            let document = primitives.quotations.scopes.get(&line.id).map_or_else(
                || primitives.document_context(&line.id).0,
                |quote| {
                    *quotes.entry(quote).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                },
            );
            (line.id.clone(), document)
        })
        .collect()
}

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

struct Typography {
    font: f64,
    glyph: f64,
    italic: f64,
}

impl Typography {
    fn size(&self, row: &Row) -> f64 {
        if row
            .spans()
            .any(|span| span.size > 0.0 && !span.text.trim().is_empty())
        {
            self.font
        } else {
            self.glyph
        }
    }
}

/// Native font sizes and recognized glyph heights keep separate character-weighted means.
fn typography<'a>(rows: impl IntoIterator<Item = &'a Row<'a>>) -> Typography {
    let (mut characters, mut italics, mut size) = (0.0, 0.0, 0.0);
    let (mut glyph_characters, mut height) = (0.0, 0.0);
    for line in rows.into_iter().flat_map(|row| &row.lines) {
        let mut native = false;
        for span in &line.spans {
            let count = span.text.trim().chars().count() as f64;
            if span.size > 0.0 && count > 0.0 {
                native = true;
                characters += count;
                size += count * span.size;
                if italic(span) {
                    italics += count;
                }
            }
        }
        if !native && line.bbox[3] > line.bbox[1] {
            let count = line.text.trim().chars().count() as f64;
            glyph_characters += count;
            height += count * (line.bbox[3] - line.bbox[1]);
        }
    }
    Typography {
        font: size / characters.max(1.0),
        glyph: height / glyph_characters.max(1.0),
        italic: italics / characters.max(1.0),
    }
}

fn defined_term_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            "^.[^\u{201d}\"\u{2019}\u{bb}]{1,60}[\u{201d}\"\u{2019}\u{bb}]\\s*,?\\s*(?:means|includes|has|refers|shall|is|are)\\b",
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
            "(?i)\\b(?:follows?|provides?|provided|states?|stated|stating|reads?|held|holds|said|says|observed|noted|explained|concluded|found|wrote|written|set out|defines?|defined|as|that|thus|this)\\s*(?::|\u{2014}|\u{2013})\\s*[-\u{2013}\u{2014}]?$|\\bas follows?\\.\\s*$",
        )
        .unwrap()
    })
}

fn closing_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            "[\u{201d}\"\u{2019}\u{bb}]\\s*[.,;:]?\\s*(?:\\[[^\\]]*\\]|\\([^)]*\\))?\\s*[.,;:]?$",
        )
        .unwrap()
    })
}

fn item_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("^\\s*(?:\\(?\\w{1,4}[.)]|[\u{2022}\u{25cf}-])\\s").unwrap())
}

/// Keep one identity for the detected block, including its fragments after page furniture.
pub(super) fn read(pages: &[Page], primitives: &PdfPrimitiveEvidence) -> Reading {
    let mut ordered_pages = pages.iter().collect::<Vec<_>>();
    ordered_pages.sort_by_key(|page| page.index);
    #[derive(Default)]
    struct Document<'a> {
        pages: Vec<&'a Page>,
        rows: Vec<Row<'a>>,
    }
    let mut readings = BTreeMap::<DocumentContext, Document>::new();
    for page in &ordered_pages {
        let mut lines = page
            .lines
            .iter()
            .filter(|line| {
                !line.exclude_from_body
                    && line.note_region_mode.is_empty()
                    && matches!(line.region_type.as_str(), "body" | "heading")
                    && !primitives.table_cell_line_ids.contains(&line.id)
                    && !primitives.table_note_line_ids.contains(&line.id)
                    && !primitives.contents_line_ids.contains(&line.id)
                    && !primitives.contents_pages.contains(&page.index)
                    && !line.text.trim().is_empty()
            })
            .collect::<Vec<_>>();
        lines.sort_by(|left, right| {
            left.reading_order
                .cmp(&right.reading_order)
                .then_with(|| left.id.cmp(&right.id))
        });
        for line in lines {
            let reading = readings
                .entry(primitives.document_context(&line.id))
                .or_default();
            if reading
                .pages
                .last()
                .is_none_or(|prior| prior.index != page.index)
            {
                reading.pages.push(page);
            }
            let height = (line.bbox[3] - line.bbox[1]).max(1.0);
            match reading.rows.last_mut() {
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
                _ => reading.rows.push(Row {
                    lines: vec![line],
                    page_index: page.index,
                    bbox: line.bbox,
                }),
            }
        }
    }
    let mut found = Reading::default();
    for document in readings.into_values() {
        let reading = quotation_rows(&document.rows, &document.pages);
        found.scopes.extend(reading.scopes);
        found.introductions.extend(reading.introductions);
    }
    found
}

fn quotation_rows(rows: &[Row<'_>], ordered_pages: &[&Page]) -> Reading {
    let mut quoted = Reading::default();
    let mut pending = vec![rows.iter().collect::<Vec<_>>()];
    while let Some(rows) = pending.pop() {
        let Some(widest) = rows.iter().copied().map(Row::width).reduce(f64::max) else {
            continue;
        };
        // The body's measure: where its full lines start and end, on each side of a spread.
        let long = rows
            .iter()
            .copied()
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
            .copied()
            .filter(|row| (row.bbox[0] - measure(row).0).abs() <= 6.0)
            .collect::<Vec<_>>();
        let mut sizes = HashMap::<i64, usize>::new();
        for span in margin_rows.iter().flat_map(|row| row.spans()) {
            if span.size > 0.0 {
                *sizes.entry((span.size * 2.0).round() as i64).or_default() +=
                    span.text.trim().chars().count();
            }
        }
        let mut glyph_sizes = HashMap::<i64, usize>::new();
        for line in margin_rows
            .iter()
            .flat_map(|row| &row.lines)
            .filter(|line| {
                !line
                    .spans
                    .iter()
                    .any(|span| span.size > 0.0 && !span.text.trim().is_empty())
            })
        {
            let height = line.bbox[3] - line.bbox[1];
            if height > 0.0 {
                *glyph_sizes
                    .entry((height * 2.0).round() as i64)
                    .or_default() += line.text.trim().chars().count();
            }
        }
        let mode = |sizes: HashMap<i64, usize>| {
            sizes
                .into_iter()
                .max_by_key(|(size, count)| (*count, *size))
                .map_or(0.0, |(size, _)| size as f64 / 2.0)
        };
        let body = Typography {
            font: mode(sizes),
            glyph: mode(glyph_sizes),
            italic: typography(margin_rows.iter().copied()).italic,
        };
        if body.font <= 0.0 && body.glyph <= 0.0 {
            continue;
        }
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
                    .filter(|row| (row.bbox[2] - measure(row).1).abs() <= 0.5 * body.size(row))
                    .count()
                >= 6 * full_margin_rows.len();

        let set_in = |row: &Row| {
            let size = body.size(row);
            size > 0.0 && row.bbox[0] - measure(row).0 >= 1.5 * size
        };
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
        let mut start = 0;
        while start < rows.len() {
            if !set_in(&rows[start]) {
                start += 1;
                continue;
            }
            let mut block = vec![rows[start]];
            let mut end = start + 1;
            loop {
                let mut next = end;
                while next < rows.len() && !set_in(&rows[next]) && edge(&rows[next]) {
                    next += 1;
                }
                let prior_page = rows[end - 1].page_index;
                if next < rows.len()
                    && set_in(&rows[next])
                    && (rows[next].page_index == prior_page
                        || prior_page.checked_add(1) == Some(rows[next].page_index))
                    && (next == end || rows[next].page_index != prior_page)
                {
                    block.push(rows[next]);
                    end = next + 1;
                } else {
                    break;
                }
            }
            let introduction = start.checked_sub(1).and_then(|before| {
                let last = rows[before];
                if !(last.page_index == rows[start].page_index
                    || last.page_index.checked_add(1) == Some(rows[start].page_index))
                    || last.bbox[0] > measure(last).0 + 3.0 * body.size(last)
                {
                    return None;
                }
                let mut first = before;
                while first > 0
                    && rows[first - 1].page_index == last.page_index
                    && rows[first - 1].lines[0].block_index == last.lines[0].block_index
                {
                    first -= 1;
                }
                let text = rows[first..start]
                    .iter()
                    .map(|row| row.text())
                    .collect::<Vec<_>>()
                    .join(" ");
                text.chars().any(char::is_lowercase).then(|| Introduction {
                    first_line_id: rows[first].lines[0].id.clone(),
                    text,
                    last_line: last.text(),
                })
            });
            if quotation(&block, introduction.as_ref(), &body, justified, &measure) {
                let id = format!("quotation:{}", block[0].lines[0].id);
                if let Some(introduction) = &introduction {
                    quoted
                        .introductions
                        .insert(id.clone(), introduction.first_line_id.clone());
                }
                quoted.scopes.extend(
                    block
                        .iter()
                        .flat_map(|row| &row.lines)
                        .map(|line| (line.id.clone(), id.clone())),
                );
            }
            // Each deeper inset is read against this block's own measure and introduction.
            if block.len() > 1 {
                pending.push(block);
            }
            start = end;
        }
    }
    quoted
}

fn opens_with_heading(block: &[&Row<'_>]) -> bool {
    let first = block[0];
    let lines = block
        .iter()
        .take_while(|row| {
            row.page_index == first.page_index
                && row.lines[0].block_index == first.lines[0].block_index
        })
        .flat_map(|row| &row.lines)
        .collect::<Vec<_>>();
    if lines.iter().all(|line| line.region_type == "heading") {
        return true;
    }
    let text = lines
        .iter()
        .map(|line| line.text.trim())
        .collect::<Vec<_>>()
        .join(" ");
    if text.chars().count() > super::MAX_FONT_HEADING_CHARS
        || !text.starts_with(char::is_uppercase)
        || text.ends_with(['.', ',', ';', ':', '?', '!'])
    {
        return false;
    }
    let (mut total, mut bold, mut italics) = (0, 0, 0);
    for span in lines.iter().flat_map(|line| &line.spans) {
        let count = span.text.trim().chars().count();
        total += count;
        italics += usize::from(italic(span)) * count;
        let font = span.font.to_ascii_lowercase();
        if span.flags & 16 != 0
            || ["bold", "black", "heavy"]
                .iter()
                .any(|word| font.contains(word))
        {
            bold += count;
        }
    }
    total > 0 && (bold * 20 >= total * 19 || italics * 20 >= total * 19)
}

/// Whether a block of rows set in from the body's margin is a display quotation; `introduction`
/// is the text of the body row before it.
fn quotation(
    block: &[&Row],
    introduction: Option<&Introduction>,
    body: &Typography,
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
            .filter(|row| {
                ((row.bbox[0] + row.bbox[2]) / 2.0 - centre).abs() <= 0.5 * body.size(row)
            })
            .count()
            >= 2 * block.len()
        && spread > body.size(block[0]);
    if centred
        || block.iter().any(|row| {
            let (left, right) = measure(row);
            row.bbox[0] >= (left + right) / 2.0
        })
    {
        return false;
    }
    let kind = typography(block.iter().copied());
    let contrast = (kind.italic >= 0.6 && body.italic < 0.25)
        || [(kind.font, body.font), (kind.glyph, body.glyph)]
            .into_iter()
            .any(|(size, body)| body > 0.0 && size >= 0.6 * body && size <= body - 0.75);
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
    let introduced =
        introduction.is_some_and(|text| introduction_re().is_match(text.text.trim_end()));
    let announced =
        introduction.is_some_and(|text| announcement_re().is_match(text.text.trim_end()));
    if (opens && (introduced || contrast || closes)) || (contrast && introduced && block.len() >= 2)
    {
        return true;
    }
    // A cited or headed extract can share the body's type and right margin.
    if block.len() >= 2 {
        let heading = opens_with_heading(block);
        let cited =
            introduction.is_some_and(|intro| layout_analysis().has_citation_signal(&intro.text));
        let located = introduction.is_some_and(|intro| {
            layout_analysis()
                .protected_citation_spans(&intro.last_line)
                .iter()
                .any(|(start, _)| {
                    let prefix = intro.last_line.chars().take(*start).collect::<String>();
                    prefix.trim().is_empty() || prefix.trim().eq_ignore_ascii_case("at")
                })
        });
        if heading && (introduced || announced) || announced && cited || heading && cited && located
        {
            return true;
        }
    }
    // Set in on the right too: its lines end short of the body's, at a measure of their own. A
    // ragged body's lines end short anyway, so there the next line's first word must have fitted.
    if !(announced || contrast) || block.len() < 3 {
        return false;
    }
    let short = block
        .iter()
        .all(|row| measure(row).1 - row.bbox[2] >= 1.5 * body.size(row));
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
                && pair[0].bbox[2] + first_word + 0.5 * body.size(pair[0]) <= measure(&pair[0]).1
        })
        .count();
    short && wide >= 2 && (justified || (fitted >= 2 && 2 * fitted >= wraps.len()))
}
