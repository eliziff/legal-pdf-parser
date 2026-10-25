//! Printed paragraphs from line geometry.
//!
//! The extractor's text blocks are unreliable paragraph evidence: a renderer may give
//! every line of double-spaced text its own block, put a whole page in one, or give
//! recognized pages none. Body lines are first parted into visual blocks where the
//! layout changes (a column or row change, a larger gap than the page's line pitch,
//! another size or emphasis), then into paragraphs where a line starts a list item, is
//! indented as a first line, or follows a line that ended short of the block's measure.

use legal_pdf_core::{line_font_size, Line, Page};
use regex::Regex;
use std::{collections::HashMap, sync::OnceLock};

const BULLETS: &str = "\u{2022}\u{25cf}\u{25aa}\u{25e6}\u{2023}\u{2043}\u{25a0}\u{25a1}\u{f0b7}\u{f0a7}\u{f0d8}\u{2013}\u{2014}\\-\\*\u{b7}";
const ENUMERATOR: &str = r"(\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\([ivxlcdm]{1,6}\)|[ivxlcdm]{1,6}\.|\[\d{1,3}\]|\d{1,3}(\.\d{1,3})+\.?)";

/// Bullet glyphs proper, which recognition can run into the item's first word.
const GLYPHS: &str =
    "\u{2022}\u{25cf}\u{25aa}\u{25e6}\u{2023}\u{2043}\u{25a0}\u{25a1}\u{f0b7}\u{f0a7}\u{f0d8}";

fn bullet_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"^\s*(([{BULLETS}]|o)\s+\S|[{GLYPHS}]\s*\S)")).unwrap())
}

fn enumerator_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"^\s*{ENUMERATOR}\s+\S")).unwrap())
}

/// A line that is nothing but a bullet or an enumerator.
fn marker_only_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"^\s*([{BULLETS}o]|{ENUMERATOR})\s*$")).unwrap())
}

fn terminal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("[.:;!?][\"\u{201d}\u{2019})\\]]?$").unwrap())
}

/// A sentence end with a note's number after it ("an injunction.1").
fn noted_terminal_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new("[.:;!?][\"\u{201d}\u{2019})\\]]?\\d{1,3}\\*?$").unwrap())
}

fn leader_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(\.\s*){4,}\s*\S{1,6}\s*$").unwrap())
}

fn size(line: &Line) -> f64 {
    let size = line_font_size(line);
    if size > 0.0 {
        size
    } else {
        height(line)
    }
}

fn height(line: &Line) -> f64 {
    line.bbox[3] - line.bbox[1]
}

fn width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

fn char_width(line: &Line) -> f64 {
    width(line) / line.text.trim().chars().count().max(1) as f64
}

/// Shares of the line's visible characters set in bold and italic, when its spans say.
fn font_shares(line: &Line) -> Option<(f64, f64)> {
    let (mut total, mut bold, mut italic) = (0, 0, 0);
    for span in &line.spans {
        let count = span.text.trim().chars().count();
        total += count;
        let font = span.font.to_ascii_lowercase();
        if span.flags & 16 != 0 || ["bold", "black", "heavy"].iter().any(|w| font.contains(w)) {
            bold += count;
        }
        if span.flags & 2 != 0 || font.contains("ital") || font.contains("oblique") {
            italic += count;
        }
    }
    (total > 0).then(|| (bold as f64 / total as f64, italic as f64 / total as f64))
}

fn first_word_width(line: &Line) -> f64 {
    let text = line.text.trim();
    // A bullet set or read against the word is no part of the word that would have fitted.
    let bullet = text.chars().next().is_some_and(|c| GLYPHS.contains(c));
    let glyph = |w: &str| w.trim().chars().next().is_some_and(|c| GLYPHS.contains(c));
    let word = line
        .words
        .iter()
        .find(|w| !w.text.trim().chars().all(|c| GLYPHS.contains(c)));
    if let Some(word) = word.filter(|w| !glyph(&w.text)) {
        return word.bbox[2] - word.bbox[0];
    }
    let words = if bullet {
        &text[text.chars().next().map_or(0, char::len_utf8)..]
    } else {
        text
    };
    let first = words
        .trim_start()
        .split(' ')
        .next()
        .unwrap_or_default()
        .chars()
        .count();
    width(line) * first as f64 / text.chars().count().max(1) as f64
}

fn same_row(a: &Line, b: &Line) -> bool {
    let overlap = a.bbox[3].min(b.bbox[3]) - a.bbox[1].max(b.bbox[1]);
    overlap > 0.5 * height(a).min(height(b))
}

/// Line advance, read from whichever box edge moved less: a recognized line's top
/// follows its ascenders and its bottom its descenders.
fn advance(a: &Line, b: &Line) -> f64 {
    (b.bbox[1] - a.bbox[1]).min(b.bbox[3] - a.bbox[3])
}

/// Are the two lines set in different sizes? A recognized line's box, its only size
/// evidence, grows with its ascenders and descenders.
fn resized(a: &Line, b: &Line) -> bool {
    let (sa, sb) = (size(a), size(b));
    let tolerance = if a.spans.is_empty() || b.spans.is_empty() {
        0.35
    } else {
        0.12
    };
    (sa - sb).abs() > 1.0_f64.max(tolerance * sa.max(sb))
}

fn size_key(value: f64) -> i64 {
    (value * 2.0).round() as i64
}

/// The page's typical baseline-to-baseline pitch, by text size.
struct Pitch {
    by_size: HashMap<i64, f64>,
    any: f64,
}

impl Pitch {
    fn of(lines: &[&Line]) -> Self {
        let mut samples: HashMap<i64, Vec<f64>> = HashMap::new();
        for pair in lines.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if b.bbox[0] > a.bbox[2] || b.bbox[2] < a.bbox[0] {
                continue;
            }
            let pitch = advance(a, b);
            if pitch > 0.5 * height(a) && pitch < 3.0 * height(a) {
                samples
                    .entry(size_key(size(a).min(size(b))))
                    .or_default()
                    .push(pitch);
            }
        }
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            let n = values.len();
            if n % 2 == 1 {
                values[n / 2]
            } else {
                (values[n / 2 - 1] + values[n / 2]) / 2.0
            }
        };
        let all: Vec<f64> = samples.values().flatten().copied().collect();
        Self {
            any: if all.is_empty() { 14.0 } else { median(all) },
            by_size: samples
                .into_iter()
                .filter(|(_, values)| values.len() >= 2)
                .map(|(key, values)| (key, median(values)))
                .collect(),
        }
    }

    fn typical(&self, size: f64) -> f64 {
        self.by_size
            .get(&size_key(size))
            .copied()
            .unwrap_or(self.any)
    }
}

/// Does `b` begin a new visual block after `a`?
fn starts_block(a: &Line, b: &Line, pitch: &Pitch) -> bool {
    if same_row(a, b) {
        // A marker set apart from its text on the text's row ("1." then the paragraph).
        let text = a.text.trim();
        let marker = marker_only_re().is_match(&a.text)
            || (text.chars().count() <= 4 && !text.contains(' '));
        if !marker
            && (bullet_re().is_match(&b.text)
                || enumerator_re().is_match(&b.text)
                || marker_only_re().is_match(&b.text))
        {
            return true;
        }
        // The extractor uses MuPDF's 0.8-em inter-run spacing for one printed line.
        let adjacent = b.bbox[0] - a.bbox[2] <= 0.8 * size(a).max(size(b));
        return !(b.bbox[0] >= a.bbox[2] - 1.0 && (marker || adjacent));
    }
    if b.bbox[0] > a.bbox[2] || b.bbox[2] < a.bbox[0] {
        return true;
    }
    let advance = advance(a, b);
    if advance <= 0.3 * height(a).min(height(b)).max(1.0) {
        return true;
    }
    // A typeset page keeps its leading exactly, so a quarter more marks a break; a
    // recognized line's box wanders more.
    let gap = if a.spans.is_empty() || b.spans.is_empty() {
        1.4
    } else {
        1.25
    };
    if resized(a, b) || advance > gap * pitch.typical(size(a).min(size(b))) {
        return true;
    }
    let contrast = |a: f64, b: f64| (a > 0.8 && b < 0.2) || (a < 0.2 && b > 0.8);
    matches!((font_shares(a), font_shares(b)), (Some((a, ai)), Some((b, bi)))
        if contrast(a, b) || a > 0.8 && b > 0.8 && contrast(ai, bi))
}

/// The left margin around the block's `slot`th line: the leftmost of its neighbours, so a
/// scan's skew does not set lines in.
fn margin_at(block: &[&Line], slot: usize) -> f64 {
    block[slot.saturating_sub(3)..(slot + 4).min(block.len())]
        .iter()
        .map(|line| line.bbox[0])
        .fold(f64::MAX, f64::min)
}

/// Does the block set its paragraphs' first lines in? Two lines indented from the
/// margin with the next line back at it show the style.
fn indents_first_lines(block: &[&Line]) -> bool {
    (1..block.len())
        .filter(|&slot| {
            let (line, next) = (block[slot - 1], block[slot]);
            let left = margin_at(block, slot);
            let cw = char_width(line);
            let indent = line.bbox[0] - left;
            indent > 1.5 * cw && indent <= 12.0 * cw && (next.bbox[0] - left).abs() < 1.5 * cw
        })
        .count()
        >= 2
}

/// Does `b` begin a new paragraph after `a` inside one visual block? Where first lines
/// are set in from the `margin`, every line set in begins one and a short line ends none
/// by itself.
fn starts_paragraph(
    a: &Line,
    b: &Line,
    block: &[&Line],
    after: Option<&Line>,
    margin: Option<f64>,
) -> bool {
    if same_row(a, b) {
        return false;
    }
    if bullet_re().is_match(&b.text) || marker_only_re().is_match(&b.text) {
        return true;
    }
    // A contents entry ends at its leader and page number; its earlier lines wrap short of that column.
    if leader_re().is_match(&a.text) {
        return true;
    }
    if leader_re().is_match(&b.text) {
        return false;
    }
    let cw = char_width(a);
    let terminal = terminal_re().is_match(a.text.trim());
    let centered = ((a.bbox[0] + a.bbox[2]) / 2.0 - (b.bbox[0] + b.bbox[2]) / 2.0).abs() < 2.0 * cw
        && (a.bbox[0] - b.bbox[0]).abs() > 1.5 * cw;
    let following = first_word_width(b) + 1.5 * cw;
    // Would b's first word have fit at the end of a?
    let fits = if centered {
        width(a) + following < block.iter().map(|line| width(line)).fold(0.0, f64::max)
    } else {
        block
            .iter()
            .map(|line| line.bbox[2])
            .fold(f64::MIN, f64::max)
            - a.bbox[2]
            > following
    };
    if enumerator_re().is_match(&b.text) && (terminal || fits) {
        return true;
    }
    let marker = bullet_re().is_match(&a.text)
        || enumerator_re().is_match(&a.text)
        || marker_only_re().is_match(&a.text);
    let first_line_indent = !centered
        && !marker
        && b.bbox[0] > a.bbox[0] + 1.5 * cw
        && (terminal
            || fits
            || after.is_some_and(|line| (line.bbox[0] - a.bbox[0]).abs() < 1.5 * cw));
    if let Some(left) = margin {
        return first_line_indent || (!marker && b.bbox[0] > left + 1.5 * cw);
    }
    // A wrap inside a sentence: a line that ran most of the measure and ended no sentence, with the
    // next going on in lower case or with the punctuation the wrap left it. Only text the page sets
    // says so; recognition misreads a line's first letter ("chief Judge").
    let wrapped = !a.spans.is_empty()
        && !b.spans.is_empty()
        && !terminal
        && !noted_terminal_re().is_match(a.text.trim())
        && goes_on(b)
        && width(a) >= 0.6 * block.iter().map(|line| width(line)).fold(0.0, f64::max);
    first_line_indent || (fits && !wrapped && !a.text.trim_end().ends_with('-'))
}

/// Does the line go on with the sentence a short line before it broke? It does when it opens in
/// lower case or with the punctuation that ends a word or clause, as a wrap leaves them ("in
/// Chaoulli" / ". In that case"); an ellipsis may open a paragraph.
fn goes_on(line: &Line) -> bool {
    let text = line.text.trim_start();
    text.chars().next().is_some_and(|c| {
        c.is_lowercase()
            || ",;:)]}\u{2019}\u{201d}%".contains(c)
            || (c == '.' && !text.starts_with("..") && !text.starts_with(". ."))
    })
}

/// Does the next page's first paragraph go on with this page's last one? It does when the
/// last line ran to the measure, ending no sentence or with the next page going on in
/// lower case, and the next page starts no indented first line, item or change of type.
pub(crate) fn continues_onto(previous: &[&Line], next: &[&Line]) -> bool {
    let (Some(a), Some(b)) = (previous.last(), next.first()) else {
        return false;
    };
    if a.page_index.checked_add(1) != Some(b.page_index) {
        return false;
    }
    let text = b.text.trim_start();
    if bullet_re().is_match(text)
        || enumerator_re().is_match(text)
        || marker_only_re().is_match(text)
    {
        return false;
    }
    if resized(a, b) {
        return false;
    }
    let cw = char_width(a);
    let left = previous
        .iter()
        .map(|line| line.bbox[0])
        .fold(f64::MAX, f64::min);
    let next_left = next
        .iter()
        .map(|line| line.bbox[0])
        .fold(f64::MAX, f64::min);
    // Columns at different places on the two pages, or a first line set in from the next
    // page's own margin, start something new.
    if (left - next_left).abs() > 3.0 * cw || (next.len() > 1 && b.bbox[0] > next_left + 1.5 * cw) {
        return false;
    }
    let right = previous
        .iter()
        .chain(next)
        .map(|line| line.bbox[2])
        .fold(f64::MIN, f64::max);
    let full = right - a.bbox[2] <= first_word_width(b) + 1.5 * cw;
    let lower = text.chars().next().is_some_and(char::is_lowercase);
    full && (!terminal_re().is_match(a.text.trim()) || lower)
}

fn in_body(line: &Line) -> bool {
    line.region_type == "body" && !line.exclude_from_body && line.note_region_mode.is_empty()
}

/// Read paragraph boundaries without changing source block indices. Classification can use
/// the same geometry before assigning roles that the final paragraph reader uses afterward.
pub(crate) fn paragraph_indices(
    page: &Page,
    counted: impl Fn(&Line) -> bool,
) -> Vec<Option<usize>> {
    let mut next = page
        .lines
        .iter()
        .map(|line| line.block_index)
        .max()
        .unwrap_or(0)
        + 1;
    let body: Vec<&Line> = page.lines.iter().filter(|line| counted(line)).collect();
    let pitch = Pitch::of(&body);
    let mut assigned = vec![None; page.lines.len()];
    let mut start = 0;
    while start < page.lines.len() {
        if !counted(&page.lines[start]) {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < page.lines.len() && counted(&page.lines[end]) {
            end += 1;
        }
        let run = &page.lines[start..end];
        let mut block_start = 0;
        for k in 1..=run.len() {
            if k < run.len() && !starts_block(&run[k - 1], &run[k], &pitch) {
                continue;
            }
            let block: Vec<&Line> = run[block_start..k].iter().collect();
            assigned[start + block_start] = Some(next);
            let indented = indents_first_lines(&block);
            for j in 1..block.len() {
                let after = block.get(j + 1).copied();
                let margin = indented.then(|| margin_at(&block, j));
                if starts_paragraph(block[j - 1], block[j], &block, after, margin) {
                    next += 1;
                }
                assigned[start + block_start + j] = Some(next);
            }
            next += 1;
            block_start = k;
        }
        start = end;
    }
    assigned
}

/// Give each printed paragraph of body text its own block index.
pub(crate) fn segment_paragraphs(pages: &mut [Page]) {
    for page in pages {
        let assigned = paragraph_indices(page, in_body);
        for (line, block) in page.lines.iter_mut().zip(assigned) {
            if let Some(block) = block {
                line.block_index = block;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn same_row_fragments_join_without_losing_paragraph_or_column_boundaries() {
        let cases = [
            (
                true,
                vec![
                    ("The kestrel circled", [50.0, 100.0, 180.0, 112.0]),
                    ("above the cliffs.", [188.0, 100.0, 300.0, 112.0]),
                ],
            ),
            (
                false,
                vec![
                    (
                        "The harbour lamps remained lit.",
                        [50.0, 100.0, 450.0, 110.0],
                    ),
                    (
                        "A separate indented paragraph begins here.",
                        [85.0, 114.0, 460.0, 124.0],
                    ),
                    (
                        "and its next line returns to the margin.",
                        [50.0, 128.0, 500.0, 138.0],
                    ),
                ],
            ),
            (
                false,
                vec![
                    (
                        "The crew checked the equipment.",
                        [50.0, 100.0, 450.0, 110.0],
                    ),
                    ("• Inspect the brass compass.", [50.0, 114.0, 450.0, 124.0]),
                ],
            ),
            (
                false,
                vec![
                    ("The left column ends here", [50.0, 100.0, 250.0, 110.0]),
                    ("the right column starts here", [350.0, 114.0, 550.0, 124.0]),
                ],
            ),
            (
                false,
                vec![
                    ("A separate column ends here", [50.0, 100.0, 250.0, 112.0]),
                    ("another column starts here", [350.0, 100.0, 550.0, 112.0]),
                ],
            ),
            (
                false,
                vec![
                    ("A field precedes an item", [50.0, 100.0, 200.0, 112.0]),
                    ("2. Inspect the harness.", [208.0, 100.0, 350.0, 112.0]),
                ],
            ),
        ];
        for (joined, rows) in cases {
            let lines: Vec<_> = rows
                .iter()
                .enumerate()
                .map(|(n, (text, bbox))| {
                    json!({
                        "id": format!("line{n}"), "page_index": 0, "page_number": 1,
                        "source_index": n, "reading_order": n, "block_index": 0,
                        "text": text, "bbox": bbox, "region_type": "body", "source": "ocr"
                    })
                })
                .collect();
            let mut pages: Vec<Page> = serde_json::from_value(json!([{
                "id": "page", "index": 0, "number": 1, "width": 600.0, "height": 800.0,
                "lines": lines, "regions": [], "source": "ocr"
            }]))
            .unwrap();
            segment_paragraphs(&mut pages);
            assert_eq!(
                pages[0].lines[0].block_index == pages[0].lines[1].block_index,
                joined,
                "{} / {}",
                rows[0].0,
                rows[1].0,
            );
        }
    }
}
