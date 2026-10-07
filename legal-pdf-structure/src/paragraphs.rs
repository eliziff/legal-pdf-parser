//! Printed paragraphs from line geometry.
//!
//! The extractor's text blocks are unreliable paragraph evidence: a renderer may give
//! every line of double-spaced text its own block, put a whole page in one, or give
//! recognized pages none. Body lines are first parted into visual blocks where the
//! layout changes (a column or row change, a larger gap than the page's line pitch,
//! another size or weight), then into paragraphs where a line starts a list item, is
//! indented as a first line, or follows a line that ended short of the block's measure.

use legal_pdf_core::{line_font_size, Line, Page};
use regex::Regex;
use std::{collections::HashMap, sync::OnceLock};

const BULLETS: &str = "\u{2022}\u{25cf}\u{25aa}\u{25e6}\u{2023}\u{2043}\u{25a0}\u{25a1}\u{f0b7}\u{f0a7}\u{f0d8}\u{2013}\u{2014}\\-\\*\u{b7}";
const ENUMERATOR: &str = r"(\(?\d{1,3}[.)]|\(?[a-zA-Z][.)]|\([ivxlcdm]{1,6}\)|[ivxlcdm]{1,6}\.|\[\d{1,3}\]|\d{1,3}(\.\d{1,3})+\.?)";

fn bullet_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(&format!(r"^\s*([{BULLETS}]|o)\s+\S")).unwrap())
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

/// Share of the line's visible characters set in a bold face, when its spans say.
fn bold_share(line: &Line) -> Option<f64> {
    let (mut total, mut bold) = (0, 0);
    for span in &line.spans {
        let count = span.text.trim().chars().count();
        total += count;
        let font = span.font.to_ascii_lowercase();
        if span.flags & 16 != 0 || ["bold", "black", "heavy"].iter().any(|w| font.contains(w)) {
            bold += count;
        }
    }
    (total > 0).then(|| bold as f64 / total as f64)
}

fn first_word_width(line: &Line) -> f64 {
    if let Some(word) = line.words.first() {
        return word.bbox[2] - word.bbox[0];
    }
    let text = line.text.trim();
    let first = text.split(' ').next().unwrap_or_default().chars().count();
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
        return !(marker && b.bbox[0] >= a.bbox[2] - 1.0);
    }
    if b.bbox[0] > a.bbox[2] || b.bbox[2] < a.bbox[0] {
        return true;
    }
    let advance = advance(a, b);
    if advance <= 0.3 * height(a).min(height(b)).max(1.0) {
        return true;
    }
    let (sa, sb) = (size(a), size(b));
    // A recognized line's box, its only size evidence, grows with its ascenders and descenders.
    let tolerance = if a.spans.is_empty() || b.spans.is_empty() {
        0.35
    } else {
        0.12
    };
    if (sa - sb).abs() > 1.0_f64.max(tolerance * sa.max(sb))
        || advance > 1.4 * pitch.typical(sa.min(sb))
    {
        return true;
    }
    matches!((bold_share(a), bold_share(b)), (Some(x), Some(y)) if (x > 0.8 && y < 0.2) || (x < 0.2 && y > 0.8))
}

/// Does `b` begin a new paragraph after `a` inside one visual block?
fn starts_paragraph(a: &Line, b: &Line, block: &[&Line], after: Option<&Line>) -> bool {
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
    first_line_indent || (fits && !a.text.trim_end().ends_with('-'))
}

fn in_body(line: &Line) -> bool {
    line.region_type == "body" && !line.exclude_from_body && line.note_region_mode.is_empty()
}

/// Give each printed paragraph of body text its own block index.
pub(crate) fn segment_paragraphs(pages: &mut [Page]) {
    for page in pages {
        let mut next = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0)
            + 1;
        let body: Vec<&Line> = page.lines.iter().filter(|line| in_body(line)).collect();
        let pitch = Pitch::of(&body);
        let mut assigned = vec![None; page.lines.len()];
        let mut start = 0;
        while start < page.lines.len() {
            if !in_body(&page.lines[start]) {
                start += 1;
                continue;
            }
            let mut end = start + 1;
            while end < page.lines.len() && in_body(&page.lines[end]) {
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
                for j in 1..block.len() {
                    if starts_paragraph(block[j - 1], block[j], &block, block.get(j + 1).copied()) {
                        next += 1;
                    }
                    assigned[start + block_start + j] = Some(next);
                }
                next += 1;
                block_start = k;
            }
            start = end;
        }
        for (line, block) in page.lines.iter_mut().zip(assigned) {
            if let Some(block) = block {
                line.block_index = block;
            }
        }
    }
}
