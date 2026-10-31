//! Printed paragraphs from line geometry.
//!
//! The extractor's text blocks are unreliable paragraph evidence: a renderer may give
//! every line of double-spaced text its own block, put a whole page in one, or give
//! recognized pages none. Body lines are first parted into visual blocks where the
//! layout changes (a column or row change, a larger gap than the page's line pitch,
//! another size or emphasis), then into paragraphs where a line starts a list item, is
//! indented as a first line, or follows a line that ended short of the block's measure.
//!
//! A scan's boxes and type sizes are a recognizer's guesses, so a scanned page is read by its
//! leading alone, as the Text-Fidelity Project's galley reads it (`paragraph_geometry.py`,
//! `text_flow_continuity.py`, `text_flow.py`), with the line height and paragraph gap measured
//! once for the document: a blank of twice the gap, or most of a line, opens a block; a sentence
//! end before an indent, on a short line or before a blank wider than the gap opens a paragraph,
//! as a short line before a wider blank does; a line going on with a sentence or a hyphened word
//! opens nothing.

use legal_pdf_core::{flag_share, line_font_size, upper_median as median, quantile, Line, BOLD, ITALIC};
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
pub(crate) fn marker_only_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"^\s*([{BULLETS}o]|{ENUMERATOR})\s*$")).unwrap())
}

/// A line opening with a bullet or an enumerator, or nothing but one: an item's first line.
pub(crate) fn opens_item(text: &str) -> bool {
    bullet_re().is_match(text) || enumerator_re().is_match(text) || marker_only_re().is_match(text)
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

/// A contents entry's leader: dots running to the line's end or to the locator that ends it (dots
/// inside a line are an ellipsis), or a rule before a locator (alone it is a form's blank).
pub(crate) fn contents_leader_re() -> &'static Regex {
    static LEADER: OnceLock<Regex> = OnceLock::new();
    LEADER.get_or_init(|| {
        Regex::new(r"(?:(?:\. ){3,}|\.{4,})\s*\S{0,6}\s*$|_{4,}\s*\d{1,4}\s*$")
            .expect("contents leader regex")
    })
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

/// A scanned line: recognized by us, or a scan's own text layer measured on its image.
fn scanned(line: &Line) -> bool {
    line.ocr.is_some() || line.spans.is_empty() || line.cap_height > 0.0
}

fn width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

fn char_width(line: &Line) -> f64 {
    width(line) / line.text.trim().chars().count().max(1) as f64
}

/// Shares of the line's visible characters set in bold and italic, when its spans say.
fn font_shares(line: &Line) -> Option<(f64, f64)> {
    Some((flag_share(line, BOLD)?, flag_share(line, ITALIC)?))
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

/// A form's or a signature block's field label ("Per:", "Name:", "Title:", "Email:"): the value
/// set beside or under it belongs with it, whatever the layout says.
pub(crate) fn field_label(line: &Line) -> bool {
    let text = line.text.trim();
    text.ends_with(':')
        && text.chars().count() <= 24
        && text.split_whitespace().count() <= 3
        && !text[..text.len() - 1].contains([':', ';'])
        && !text[..text.len() - 1].trim_end_matches('.').contains('.')
}

pub(crate) fn same_row(a: &Line, b: &Line) -> bool {
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

/// The page's typical baseline-to-baseline pitch, by text size, and its column edges: left
/// edges where three or more body lines start.
struct Pitch {
    by_size: HashMap<i64, f64>,
    any: f64,
    edges: Vec<f64>,
    /// The document's measure and leading (`measure_of`).
    measure: Measure,
    /// Most of the page's lines are scanned: it is read by its leading alone.
    scan: bool,
}

/// The document's measure, and its scans' leading.
#[derive(Clone, Copy, Default)]
pub(crate) struct Measure {
    /// The ninth decile of its counted lines' widths over every page. A two-column statute's
    /// column, not its full-width title; a brief's text measure, not the short lines of its page
    /// of counsel.
    width: f64,
    /// The median height of a scanned line.
    line: f64,
    /// The median blank between two scanned lines stacked in one column, counted up to 1.25 line
    /// heights: a paragraph's own leading, not the space between paragraphs.
    gap: f64,
    /// The median distance between the centres of two scanned lines stacked in one column, counted
    /// from a quarter to 1.75 line heights: the line pitch, which a scan's text layer can set
    /// closer than its boxes are tall (boxes overlapping the next line's).
    pitch: f64,
}

/// The measures a document's counted lines show (`pages`: each page's, in their order on it): the
/// width most of them reach, and on a scan its line height, paragraph gap and line pitch.
pub(crate) fn measure_of(pages: &[Vec<&Line>]) -> Measure {
    let width = quantile(pages.iter().flatten().map(|line| width(line)).collect(), 0.9);
    let scans = || {
        pages.iter().map(|lines| {
            lines
                .iter()
                .copied()
                .filter(|line| scanned(line) && height(line) > 0.0)
                .collect::<Vec<_>>()
        })
    };
    let line_height = median(scans().flatten().map(height).collect());
    let gap = median(
        scans()
            .flat_map(|lines| {
                lines
                    .windows(2)
                    .filter(|pair| pair[1].bbox[0] < pair[0].bbox[2] && pair[0].bbox[0] < pair[1].bbox[2])
                    .map(|pair| pair[1].bbox[1] - pair[0].bbox[3])
                    .filter(|gap| (0.0..=1.25 * line_height).contains(gap))
                    .collect::<Vec<_>>()
            })
            .collect(),
    );
    let centre = |line: &Line| (line.bbox[1] + line.bbox[3]) / 2.0;
    let pitch = median(
        scans()
            .flat_map(|lines| {
                lines
                    .windows(2)
                    .filter(|pair| pair[1].bbox[0] < pair[0].bbox[2] && pair[0].bbox[0] < pair[1].bbox[2])
                    .map(|pair| centre(pair[1]) - centre(pair[0]))
                    .filter(|advance| (0.25 * line_height..=1.75 * line_height).contains(advance))
                    .collect::<Vec<_>>()
            })
            .collect(),
    );
    Measure { width, line: line_height, gap, pitch }
}

impl Pitch {
    fn of(lines: &[&Line], measure: Measure) -> Self {
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
        let all: Vec<f64> = samples.values().flatten().copied().collect();
        let mut lefts: HashMap<i64, (usize, f64)> = HashMap::new();
        for line in lines {
            let slot = lefts.entry((line.bbox[0] / 3.0).round() as i64).or_default();
            slot.0 += 1;
            slot.1 += line.bbox[0];
        }
        Self {
            edges: lefts
                .values()
                .filter(|(count, _)| *count >= 3)
                .map(|(count, sum)| sum / *count as f64)
                .collect(),
            any: if all.is_empty() { 14.0 } else { median(all) },
            measure,
            scan: lines.iter().filter(|line| scanned(line)).count() * 2 > lines.len(),
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

    /// A line well short of the page's measure: a label, a value, an address or caption line.
    fn short(&self, line: &Line) -> bool {
        width(line) < 0.6 * self.measure.width
    }

    /// Are two lines on one row? A scan's boxes swell and overlap their neighbours, so there its
    /// lines share a row when their centres are within half the document's line pitch.
    fn same_row(&self, a: &Line, b: &Line) -> bool {
        if !self.scan || self.measure.pitch <= 0.0 {
            return same_row(a, b);
        }
        let centre = |line: &Line| (line.bbox[1] + line.bbox[3]) / 2.0;
        (centre(a) - centre(b)).abs() < 0.5 * self.measure.pitch
    }

    /// Does a column of the page start within `tolerance` of `x`?
    fn at_edge(&self, x: f64, tolerance: f64) -> bool {
        self.edges.iter().any(|edge| (edge - x).abs() <= tolerance)
    }
}

/// Does `b` begin a new visual block after `a`? The rule that says so, by name.
fn starts_block(a: &Line, b: &Line, pitch: &Pitch) -> Option<&'static str> {
    if pitch.same_row(a, b) {
        // A marker set apart from its text on the text's row ("1." then the paragraph).
        let text = a.text.trim();
        let marker = marker_only_re().is_match(&a.text)
            || (text.chars().count() <= 4 && !text.contains(' '));
        // An item opens with its marker and its text; a marker alone at the row's end ("[63]" after a
        // citation, a cell's number) and a value after a field label ("Tel.:") open nothing.
        if !marker
            && !field_label(a)
            && !marker_only_re().is_match(&b.text)
            && (bullet_re().is_match(&b.text) || enumerator_re().is_match(&b.text))
        {
            return Some("row-marker");
        }
        // The extractor uses MuPDF's 0.8-em inter-run spacing for one printed line; a wider
        // gap parts the row only where a column of the page starts. Two short lines on one row
        // (a table row's cells, a form's label and value, a caption's two sides) are one block
        // whatever the gap: the gold reads a row as a unit. A recognized line's pieces may
        // overlap by a little.
        let em = size(a).max(size(b));
        let gap = b.bbox[0] - a.bbox[2];
        let adjacent = gap <= 0.8 * em
            || (gap <= 3.0 * em && !pitch.at_edge(b.bbox[0], 0.5 * em))
            || (pitch.short(a) && pitch.short(b));
        let overlap_allowed = if goes_on(b) || (pitch.short(a) && pitch.short(b)) { 2.0 * em } else { 1.0 };
        return (!(b.bbox[0] >= a.bbox[2] - overlap_allowed && (marker || adjacent || field_label(a))))
            .then_some("row-apart");
    }
    // Two short lines, the first ending no sentence, stack as one block (an address, a caption, a
    // signature block, a label and its value): a change of column, weight or face parts them no
    // more than it parts the lines of a paragraph. A gap still does.
    let stacked = pitch.short(a) && pitch.short(b) && !terminal_re().is_match(a.text.trim());
    let goes_on = goes_on(b) || field_label(a) || stacked;
    if (b.bbox[0] > a.bbox[2] || b.bbox[2] < a.bbox[0])
        && !goes_on
        && pitch.at_edge(b.bbox[0], 0.5 * size(b))
        && (b.bbox[0] - a.bbox[0]).abs() > 3.0 * char_width(a)
    {
        return Some("column");
    }
    if pitch.scan {
        return scan_starts_block(a, b, goes_on, pitch);
    }
    let advance = advance(a, b);
    if advance <= 0.3 * height(a).min(height(b)).max(1.0) {
        return Some("overlap");
    }
    // A typeset page keeps its leading exactly, so a quarter more marks a break; a
    // recognized line's box wanders more.
    let gap = if a.spans.is_empty() || b.spans.is_empty() {
        1.4
    } else {
        1.25
    };
    if resized(a, b) && !goes_on {
        return Some("resized");
    }
    if advance > gap * pitch.typical(size(a).min(size(b))) {
        return Some("gap");
    }
    let contrast = |a: f64, b: f64| (a > 0.8 && b < 0.2) || (a < 0.2 && b > 0.8);
    (!goes_on
        && matches!((font_shares(a), font_shares(b)), (Some((a, ai)), Some((b, bi)))
            if contrast(a, b) || a > 0.8 && b > 0.8 && contrast(ai, bi)))
    .then_some("font-contrast")
}

/// Does `b` begin a new visual block after `a` on a scanned page? Only its leading says: a blank
/// of at least most of a line, or twice the document's paragraph gap, opens a block, unless the
/// line goes on with the sentence or the word a hyphen broke, or the line before left a bracket
/// open. A line set back up the page opens nothing by itself: a scan's text layer jumps about
/// inside one paragraph (on the gold's scans, 1,649 false breaks fell at such jumps), and a
/// change of column is the column rule's.
fn scan_starts_block(a: &Line, b: &Line, goes_on: bool, pitch: &Pitch) -> Option<&'static str> {
    let line = height(a).min(height(b)).max(1.0);
    let text = a.text.trim_end();
    let open = text.matches(['(', '[']).count() > text.matches([')', ']']).count();
    if goes_on || open || (text.ends_with('-') && !text.ends_with(" -")) {
        return None;
    }
    let blank = b.bbox[1] - a.bbox[3];
    (blank >= (0.85 * line).max(2.0 * pitch.measure.gap + 0.25).max(2.0)).then_some("gap")
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

/// Does `b` begin a new paragraph after `a` inside one visual block? The rule that says so,
/// by name. Where first lines are set in from the `margin`, every line set in begins one and
/// a short line ends none by itself.
fn starts_paragraph(
    a: &Line,
    b: &Line,
    block: &[&Line],
    after: Option<&Line>,
    margin: Option<f64>,
    pitch: &Pitch,
) -> Option<&'static str> {
    if pitch.same_row(a, b) {
        return None;
    }
    if bullet_re().is_match(&b.text) || marker_only_re().is_match(&b.text) {
        return Some("bullet");
    }
    // A contents entry ends at its leader and page number; its earlier lines wrap short of that column.
    if contents_leader_re().is_match(&a.text) {
        return Some("leader");
    }
    if contents_leader_re().is_match(&b.text) {
        return None;
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
        return Some("enumerator");
    }
    // A line that goes on with the sentence (lower case, or the punctuation a wrap left it) opens
    // nothing, however it is set; nor does the value under a field's label ("Beneficiary:" and the
    // address beneath it are one block, as a form or a letter of credit sets them).
    if goes_on(b) || field_label(a) {
        return None;
    }
    if pitch.scan {
        // A scan's paragraph opens where a sentence ended and the next line is set in from the
        // block's flush-left edge (its first quartile of left edges) by 1.8% of the measure; where
        // a sentence ended on a line well short of the measure, or before a blank 0.15 of a line
        // wider than the document's leading; or where a line well short of the measure, ending no
        // sentence (recognition loses some lines' ends), stands before a blank 0.4 of a line wider
        // than the leading, or before a full line opening a sentence across a third of a line.
        // The line after a short one may be short too: a one-line paragraph, an item, a salutation.
        // On the structure gold's scans, requiring a full line after the short one and a sentence
        // opening on it missed a quarter of the paragraph openings between body lines (recall 0.76
        // on kept text layers, 0.67 on our recognition); these rules find 2,498 and 1,356 more of
        // them, at 1,152 and 506 false openings.
        let mut lefts = block.iter().map(|line| line.bbox[0]).collect::<Vec<_>>();
        lefts.sort_by(f64::total_cmp);
        let flush = lefts[lefts.len() / 4];
        let ended = terminal || noted_terminal_re().is_match(a.text.trim());
        let opens = b.text.trim_start().chars().next().is_some_and(|c| c.is_uppercase() || c.is_ascii_digit());
        let short = pitch.measure.width > 0.0 && width(a) < 0.75 * pitch.measure.width;
        let line = if pitch.measure.line > 0.0 { pitch.measure.line } else { height(a).max(height(b)) };
        let blank = b.bbox[1] - a.bbox[3];
        let wider = blank - pitch.measure.gap;
        if ended && b.bbox[0] >= flush + (0.018 * pitch.measure.width).max(3.0) {
            return Some("sentence-indent");
        }
        if ended && (short || wider >= 0.15 * line) {
            return Some("sentence-end");
        }
        let broken = a.text.trim_end().ends_with('-');
        let wide_next = width(b) >= 0.6 * pitch.measure.width;
        if short && !broken && (wider >= 0.4 * line || opens && wide_next && blank > 0.3 * height(a).max(height(b)))
        {
            return Some("short-line");
        }
        return None;
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
    if first_line_indent {
        return Some("first-line-indent");
    }
    if let Some(left) = margin {
        return (!marker && b.bbox[0] > left + 1.5 * cw).then_some("margin-indent");
    }
    // A short line ends its paragraph with its sentence. Recognition drops a line's last period,
    // so a recognized line well short of the measure ends one when the next line opens a sentence.
    let ended = terminal || noted_terminal_re().is_match(a.text.trim());
    let measure = block.iter().map(|line| width(line)).fold(0.0, f64::max);
    let opens = b
        .text
        .trim_start()
        .chars()
        .next()
        .is_some_and(|c| c.is_uppercase() || c.is_ascii_digit());
    let recognized_end = a.spans.is_empty() && width(a) < 0.6 * measure && opens;
    (fits && (ended || recognized_end) && !a.text.trim_end().ends_with('-')).then_some("short-line")
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
    if opens_item(text) {
        return false;
    }
    let lower = text.chars().next().is_some_and(char::is_lowercase);
    // A scan's paragraph goes on over the page unless its last line ended a sentence and the
    // next page opens in capitals: its boxes and sizes say nothing.
    if scanned(a) && scanned(b) {
        return !terminal_re().is_match(a.text.trim()) || lower;
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
    full && (!terminal_re().is_match(a.text.trim()) || lower)
}

/// With LEGALPDF_PARAGRAPH_TRACE set, each break names its rule on stderr: the page, the line
/// that opens the new block or paragraph, its first characters.
fn trace(line: &Line, kind: &str, reason: &str) {
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var_os("LEGALPDF_PARAGRAPH_TRACE").is_some()) {
        let text: String = line.text.trim().chars().take(48).collect();
        eprintln!("ptrace\t{}\t{}\t{kind}\t{reason}\t{text}", line.page_index, line.id);
    }
}

/// Each of a page's `lines` (in reading order) `counted` holds numbered by its printed paragraph,
/// from 0 in order; none for the others, which part the runs of counted lines.
pub(crate) fn paragraphs_of(lines: &[&Line], counted: &[bool], measure: Measure) -> Vec<Option<usize>> {
    let mut next = 0;
    let body: Vec<&Line> = lines.iter().zip(counted).filter(|(_, counted)| **counted).map(|(line, _)| *line).collect();
    let pitch = Pitch::of(&body, measure);
    let mut assigned = vec![None; lines.len()];
    let mut start = 0;
    while start < lines.len() {
        if !counted[start] {
            start += 1;
            continue;
        }
        let mut end = start + 1;
        while end < lines.len() && counted[end] {
            end += 1;
        }
        let run = &lines[start..end];
        // LEGALPDF_LAYOUT_UNITS=on: where a page layout model has given every line a region, the
        // regions are the paragraphs: a block starts where the region changes and nowhere else.
        let layout_units = layout_units_mode() && run.iter().all(|line| !line.region_id.is_empty());
        let mut block_start = 0;
        for k in 1..=run.len() {
            if k < run.len() {
                if layout_units {
                    if run[k - 1].region_id == run[k].region_id {
                        continue;
                    }
                    trace(&run[k], "block", "layout-region");
                } else {
                    match starts_block(run[k - 1], run[k], &pitch) {
                        None => continue,
                        Some(reason) => trace(run[k], "block", reason),
                    }
                }
            }
            let block: Vec<&Line> = run[block_start..k].to_vec();
            assigned[start + block_start] = Some(next);
            let indented = indents_first_lines(&block);
            for j in 1..block.len() {
                let after = block.get(j + 1).copied();
                let margin = indented.then(|| margin_at(&block, j));
                if !layout_units {
                    if let Some(reason) = starts_paragraph(block[j - 1], block[j], &block, after, margin, &pitch) {
                        trace(block[j], "paragraph", reason);
                        next += 1;
                    }
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

/// Whether a page layout model's regions, when present on every line, define the paragraphs.
pub(crate) fn layout_units_mode() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("LEGALPDF_LAYOUT_UNITS").is_ok_and(|v| v == "on"))
}
