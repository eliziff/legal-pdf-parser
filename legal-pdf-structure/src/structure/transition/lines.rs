//! What the parser's scorer reads of each line before it reads any state: the line set against its
//! page and its document (place, gaps, type, look and how common the look is, ink, its column), the
//! shape of its text, and what the readers found on it (its marker's readings, a note label, reference
//! marks, the words the lexicon knows, a furniture edge, a table cell, the note band). Nothing here is
//! a decision: every value is measured on the pages or read by an evidence reader.

use super::*;
use legal_pdf_core::{flag_share, Columns, PageMeasure, BOLD, ITALIC, MONOSPACED, SUPERSCRIPT};

/// The names of a line's own features, in order.
pub(crate) const LINE_FEATURES: &[&str] = &[
    "x0", "y0", "x1", "y1", "height", "gap_above", "gap_below", "indent", "right_gap", "width", "centre",
    "size", "bold", "italic", "superscript", "monospace", "size_change", "font_change", "underlined",
    "length", "upper", "digits", "punctuation", "spaces", "starts_digit", "starts_upper", "starts_lower",
    "starts_open", "starts_mark", "ends_period", "ends_colon", "ends_comma", "ends_hyphen", "ends_digit",
    "ocr", "confidence", "page_quality", "page_position", "line_position", "first_on_page", "last_on_page",
    "page_offgrid", "page_measured",
    "style_share", "look_share", "look_lines", "look_pages", "look_length", "size_rank", "doc_size", "body_style",
    "ink", "ink_measured", "weight", "col_width", "col_centre",
    "label", "label_raised", "label_value", "label_size", "label_suppressed",
    "reference_superscript", "reference_detached", "reference_raised", "reference_glued",
    "title_shaped", "beside", "line_number", "edge_repeated", "cell", "below_separator", "translation",
    "marker", "marker_word", "marker_value", "marker_prefix", "marker_two", "marker_only",
    "count_1", "count_1.1", "count_1.1.1", "count_1.1.1.1", "count_a", "count_A", "count_i", "count_I", "count_bullet",
    "wrap_()", "wrap_[]", "wrap_)", "wrap_.", "wrap_",
];
/// The lexicon's word bits (`WORD_*`) and the look's (`LOOK_*`) follow the named features.
pub(crate) const WORD_BITS: usize = 32;
pub(crate) const LOOK_BITS: usize = 4;
pub(crate) const WIDTH: usize = LINE_FEATURES.len() + WORD_BITS + LOOK_BITS;
/// The features read again off the lines around a line (two before, two after): its place and type
/// (the first twenty) and its marker's.
const AROUND: std::ops::Range<usize> = 0..20;
const MARKER: std::ops::Range<usize> = 72..92;
/// Every feature a line has: its own, those of the two lines before and after it in reading order
/// (`AROUND` and `MARKER` of each), whether the next and previous are on its page, and the document's
/// shape.
pub(crate) const ALL: usize = WIDTH + 4 * (20 + 20) + 2 + 4;

/// The shape of a line's text: its length, its share of capitals, figures, punctuation and spaces,
/// how it opens and how it ends.
fn text_shape(text: &str) -> [f64; 15] {
    let chars: Vec<char> = text.trim().chars().collect();
    let n = chars.len().max(1) as f64;
    let letters = chars.iter().filter(|c| c.is_alphabetic()).count().max(1) as f64;
    let first = chars.first().copied().unwrap_or(' ');
    let last = chars.last().copied().unwrap_or(' ');
    let flag = |b: bool| f64::from(u8::from(b));
    [
        (1.0 + chars.len() as f64).ln() / 5.0,
        chars.iter().filter(|c| c.is_uppercase()).count() as f64 / letters,
        chars.iter().filter(|c| c.is_ascii_digit()).count() as f64 / n,
        chars.iter().filter(|c| c.is_ascii_punctuation()).count() as f64 / n,
        chars.iter().filter(|c| c.is_whitespace()).count() as f64 / n,
        flag(first.is_ascii_digit()),
        flag(first.is_uppercase()),
        flag(first.is_lowercase()),
        flag("([{".contains(first)),
        flag("•·▪–—-*†‡§¶".contains(first)),
        flag(last == '.'),
        flag(last == ':'),
        flag(last == ',' || last == ';'),
        flag(last == '-' || last == '\u{00AD}'),
        flag(last.is_ascii_digit()),
    ]
}

/// How common each look is across the document: a heading's look is rare and set apart, the body's
/// the commonest. A look is a style (face and half-point size) and its bold, italic and capitals.
struct Looks {
    style_chars: HashMap<(u16, u16), usize>,
    look_chars: HashMap<(u16, u16, u8), usize>,
    look_lines: HashMap<(u16, u16, u8), usize>,
    look_pages: HashMap<(u16, u16, u8), HashSet<usize>>,
    size_rank: HashMap<u16, f64>,
    body: Option<(u16, u16)>,
    chars: usize,
    lines: usize,
    pages: usize,
    median_length: f64,
}

const LOOK_STYLE: u8 = LOOK_BOLD | LOOK_ITALIC | LOOK_CAPS;

impl Looks {
    fn tally(pages: &[Page], slots: &Slots, evidence: &Evidence) -> Self {
        let mut looks = Looks {
            style_chars: HashMap::new(), look_chars: HashMap::new(), look_lines: HashMap::new(), look_pages: HashMap::new(),
            size_rank: HashMap::new(), body: None, chars: 0, lines: 0, pages: pages.len().max(1), median_length: 1.0,
        };
        let mut lengths = Vec::new();
        for at in 0..slots.len() {
            if evidence.face[at] == u16::MAX {
                continue;
            }
            let slot = Slot(at as u32);
            let count = slots.line(pages, slot).text.trim().chars().count();
            let style = (evidence.face[at], evidence.half_points[at]);
            let look = (style.0, style.1, evidence.look[at] & LOOK_STYLE);
            lengths.push(count as f64);
            looks.chars += count;
            looks.lines += 1;
            *looks.style_chars.entry(style).or_default() += count;
            *looks.look_chars.entry(look).or_default() += count;
            *looks.look_lines.entry(look).or_default() += 1;
            looks.look_pages.entry(look).or_default().insert(slots.page(slot));
        }
        looks.median_length = legal_pdf_core::upper_median(lengths).max(1.0);
        looks.body = looks.style_chars.iter().max_by_key(|(style, n)| (**n, std::cmp::Reverse(**style))).map(|(style, _)| *style);
        let mut sizes: Vec<u16> = looks.style_chars.keys().map(|style| style.1).collect();
        sizes.sort_unstable();
        sizes.dedup();
        let span = (sizes.len().max(2) - 1) as f64;
        looks.size_rank = sizes.iter().rev().enumerate().map(|(rank, &size)| (size, rank as f64 / span)).collect();
        looks
    }

    fn features(&self, evidence: &Evidence, at: usize) -> [f64; 8] {
        if evidence.face[at] == u16::MAX {
            return [0.0; 8];
        }
        let style = (evidence.face[at], evidence.half_points[at]);
        let look = (style.0, style.1, evidence.look[at] & LOOK_STYLE);
        let chars = self.chars.max(1) as f64;
        let lines = self.look_lines[&look] as f64;
        let body_size = self.body.map_or(0, |(_, size)| size);
        [
            self.style_chars[&style] as f64 / chars,
            self.look_chars[&look] as f64 / chars,
            (1.0 + lines).ln() / (1.0 + self.lines as f64).ln().max(1.0),
            self.look_pages[&look].len() as f64 / self.pages as f64,
            (self.look_chars[&look] as f64 / lines / self.median_length).min(3.0),
            self.size_rank[&style.1],
            if body_size > 0 { (f64::from(style.1) / f64::from(body_size)).min(4.0) } else { 0.0 },
            f64::from(u8::from(self.body == Some(style))),
        ]
    }
}

const COUNTS: [&str; 9] = ["1", "1.1", "1.1.1", "1.1.1.1", "a", "A", "i", "I", "bullet"];
const WRAPS: [&str; 5] = ["()", "[]", ")", ".", ""];

/// Whether each line (in `order`) stands at its page's top or bottom tenth with a text, figures aside,
/// that lines there on three pages or more print: a running head or foot, measured, not fenced.
fn repeated_at_edges(pages: &[Page], slots: &Slots, order: &[Slot]) -> Vec<bool> {
    let key = |slot: Slot| -> Option<String> {
        let page = &pages[slots.page(slot)];
        let line = slots.line(pages, slot);
        let height = page.height.max(1.0);
        let edge = line.bbox[3] <= 0.1 * height || line.bbox[1] >= 0.9 * height;
        let text: String = line.text.chars().filter(|c| c.is_alphabetic()).flat_map(char::to_lowercase).collect();
        (edge && text.chars().count() >= 3).then_some(text)
    };
    let mut pages_of: HashMap<String, HashSet<usize>> = HashMap::new();
    let keys: Vec<Option<String>> = order.iter().map(|&slot| key(slot)).collect();
    for (k, found) in keys.iter().enumerate() {
        if let Some(text) = found {
            pages_of.entry(text.clone()).or_default().insert(slots.page(order[k]));
        }
    }
    keys.iter().map(|found| found.as_ref().is_some_and(|text| pages_of[text].len() >= 3)).collect()
}

/// Each line's readings and features, in reading order (`fences.order`): `WIDTH` own features per
/// line, then `ALL` with the lines around it and the document's shape.
pub(crate) struct LineTable {
    pub(crate) order: Vec<Slot>,
    pub(crate) readings: Vec<Vec<MarkerReading>>,
    /// Each line's reference marks: value, first and past-last character (`reference_spans`).
    pub(crate) references: Vec<Vec<(u32, u32, u32)>>,
    /// Each line's look: its face, half-point size, bold, italic and capitals, as one number.
    pub(crate) looks: Vec<u32>,
    pub(crate) features: Vec<f32>,
}

impl LineTable {
    pub(crate) fn row(&self, k: usize) -> &[f32] {
        &self.features[k * ALL..(k + 1) * ALL]
    }

    pub(crate) fn read(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences) -> Self {
        let order: Vec<Slot> = fences.order.clone();
        let n = order.len();
        let measures: Vec<(PageMeasure, Columns, HashMap<usize, usize>, f64, f64)> = pages.iter().map(|page| {
            let measure = PageMeasure::of(page);
            let rank = measure.order.iter().enumerate().map(|(k, &line)| (line, k)).collect();
            let ink = legal_pdf_core::upper_median(page.lines.iter().map(|l| l.ink).filter(|ink| *ink > 0.0).collect());
            let weight = legal_pdf_core::upper_median(page.lines.iter().map(|l| l.weight).filter(|w| *w > 0.0).collect());
            (measure, Columns::of(page), rank, ink, weight)
        }).collect();
        let off_grid: Vec<usize> = pages.iter().map(|page| legal_pdf_core::off_grid_sizes(page.lines.iter().filter(|line| !line.exclude_from_body))).collect();
        let looks = Looks::tally(pages, slots, evidence);
        let repeated = repeated_at_edges(pages, slots, &order);
        let page_count = pages.len().max(1) as f64;
        let mut readings = Vec::with_capacity(n);
        let mut references = Vec::with_capacity(n);
        let mut look_ids = Vec::with_capacity(n);
        let mut own = vec![0f32; n * WIDTH];
        for (k, &slot) in order.iter().enumerate() {
            let at = slot.index();
            let p = slots.page(slot);
            let page = &pages[p];
            let line = slots.line(pages, slot);
            let n_on_page = page.lines.len();
            let index = (at - slots.on(p).start as usize).min(n_on_page.saturating_sub(1));
            let (measure, columns, rank_of, ink, weight) = &measures[p];
            let rank = rank_of.get(&index).copied().unwrap_or(0);
            let neighbour = |offset: isize| (rank as isize + offset).try_into().ok().and_then(|r: usize| measure.order.get(r)).map(|&m| &page.lines[m]);
            let (w, h) = (page.width.max(1.0), page.height.max(1.0));
            let [x0, y0, x1, y1] = line.bbox;
            let body = (measure.right - measure.left).max(1.0);
            let above = neighbour(-1).map_or(3.0, |prev| ((y0 - prev.bbox[3]) / measure.height).clamp(-3.0, 3.0));
            let below = neighbour(1).map_or(3.0, |next| ((next.bbox[1] - y1) / measure.height).clamp(-3.0, 3.0));
            let style = legal_pdf_core::dominant_face(line);
            let previous_style = neighbour(-1).and_then(legal_pdf_core::dominant_face);
            let size = style.map_or(0.0, |s| s.1);
            let underlined = if line.words.is_empty() { 0.0 } else { line.words.iter().filter(|w| w.underlined).count() as f64 / line.words.len() as f64 };
            let share = |flag| flag_share(line, flag).unwrap_or(0.0);
            let mut values: Vec<f64> = vec![
                x0 / w, y0 / h, x1 / w, y1 / h,
                ((y1 - y0) / measure.height).min(5.0), above, below,
                ((x0 - measure.left) / w).clamp(-1.0, 1.0), ((measure.right - x1) / w).clamp(-1.0, 1.0),
                ((x1 - x0) / body).min(2.0), (((x0 + x1) / 2.0 - w / 2.0) / w).clamp(-1.0, 1.0),
                if measure.size > 0.0 { (size / measure.size).min(4.0) } else { 0.0 },
                share(BOLD), share(ITALIC), share(SUPERSCRIPT), share(MONOSPACED),
                match (style, previous_style) { (Some(a), Some(b)) if b.1 > 0.0 => (a.1 / b.1).min(4.0), _ => 1.0 },
                match (style, previous_style) { (Some(a), Some(b)) => f64::from(u8::from(a.0 != b.0)), _ => 0.0 },
                underlined,
            ];
            values.extend(text_shape(&line.text));
            values.extend([
                f64::from(u8::from(line.source != "native")),
                line.ocr.as_ref().map_or(1.0, |o| o.confidence),
                page.text_quality,
                p as f64 / page_count,
                rank as f64 / n_on_page.max(1) as f64,
                f64::from(u8::from(rank == 0)),
                f64::from(u8::from(rank + 1 == n_on_page)),
                (off_grid[p] as f64 / 10.0).min(2.0),
                f64::from(u8::from(off_grid[p] >= 5)),
            ]);
            values.extend(looks.features(evidence, at));
            values.extend([
                if *ink > 0.0 && line.ink > 0.0 { (line.ink / ink).min(3.0) } else { 0.0 },
                f64::from(u8::from(line.ink > 0.0)),
                if *weight > 0.0 && line.weight > 0.0 { (line.weight / weight).min(3.0) } else { 0.0 },
            ]);
            let (col_width, col_centre) = columns.place(line, index);
            values.extend([col_width, col_centre]);
            let label = evidence.label[at];
            let flag = |b: bool| f64::from(u8::from(b));
            values.extend([
                flag(label.is_some()), flag(label.is_some_and(|l| l.raised)),
                label.map_or(0.0, |l| f64::from(l.value.min(999)) / 100.0),
                label.map_or(0.0, |l| f64::from(l.size) / evidence.size[at].max(1.0)),
                flag(label.is_some_and(|l| l.suppress != 0)),
                flag(evidence.references[at] & REFERENCE_SUPERSCRIPT != 0), flag(evidence.references[at] & REFERENCE_DETACHED != 0),
                flag(evidence.references[at] & REFERENCE_RAISED != 0), flag(evidence.references[at] & REFERENCE_GLUED != 0),
                flag(evidence.title_shaped[at]), flag(evidence.beside[at].is_some()), flag(evidence.line_number[at]),
                flag(repeated[k]), flag(fences.cell[at] > 0),
                flag(evidence.pages[p].separator.is_some_and(|y| line.bbox[1] >= y)), flag(fences.translation[at]),
            ]);
            let found = marker_readings(&line.text, evidence.words[at] & WORD_DIVISION != 0);
            let mut marker = [0.0f64; 20];
            if let Some(first) = found.first() {
                let (word, plain) = match first.family.matches(':').count() {
                    0 | 1 => ("", first.family.as_str()),
                    _ => first.family.split_once(':').unwrap_or(("", first.family.as_str())),
                };
                let (wrap, count) = plain.rsplit_once(':').unwrap_or(("", plain));
                marker[0] = 1.0;
                marker[1] = flag(!word.is_empty());
                marker[2] = (1.0 + f64::from(first.value)).ln();
                marker[3] = first.prefix.len() as f64;
                marker[4] = flag(found.len() > 1);
                marker[5] = flag(evidence.marker[at].is_some_and(|m| m.only));
                if let Some(c) = COUNTS.iter().position(|c| *c == count) {
                    marker[6 + c] = 1.0;
                }
                if let Some(c) = WRAPS.iter().position(|c| *c == wrap) {
                    marker[15 + c] = 1.0;
                }
            }
            values.extend(marker);
            debug_assert_eq!(values.len(), LINE_FEATURES.len());
            let row = &mut own[k * WIDTH..(k + 1) * WIDTH];
            for (cell, value) in row.iter_mut().zip(&values) {
                *cell = (*value * 1e4).round() as f32 / 1e4;
            }
            for bit in 0..WORD_BITS {
                row[LINE_FEATURES.len() + bit] = ((evidence.words[at] >> bit) & 1) as f32;
            }
            for bit in 0..LOOK_BITS {
                row[LINE_FEATURES.len() + WORD_BITS + bit] = ((evidence.look[at] >> bit) & 1) as f32;
            }
            readings.push(found);
            references.push(reference_spans(line, evidence.size[at]));
            look_ids.push(u32::from(evidence.face[at]) << 20 | u32::from(evidence.half_points[at].min(0xffff)) << 4 | u32::from(evidence.look[at] & LOOK_STYLE));
        }
        // The lines around each line, and the document's shape.
        let opens_on_marker = (0..n).filter(|&k| own[k * WIDTH + MARKER.start] > 0.0).count() as f32 / n.max(1) as f32;
        let labelled = (0..n).filter(|&k| evidence.label[order[k].index()].is_some()).count() as f32 / n.max(1) as f32;
        let mut features = vec![0f32; n * ALL];
        for k in 0..n {
            let row = &mut features[k * ALL..(k + 1) * ALL];
            row[..WIDTH].copy_from_slice(&own[k * WIDTH..(k + 1) * WIDTH]);
            let mut at = WIDTH;
            for offset in [-1isize, 1, -2, 2] {
                let other = k as isize + offset;
                let source = (0..n as isize).contains(&other).then(|| &own[other as usize * WIDTH..(other as usize + 1) * WIDTH]);
                for range in [AROUND, MARKER] {
                    for c in range {
                        row[at] = source.map_or(-9.0, |s| s[c]);
                        at += 1;
                    }
                }
            }
            let page = |k: usize| slots.page(order[k]);
            row[at] = f32::from(u8::from(k > 0 && page(k - 1) == page(k)));
            row[at + 1] = f32::from(u8::from(k + 1 < n && page(k + 1) == page(k)));
            row[at + 2] = opens_on_marker;
            row[at + 3] = labelled;
            row[at + 4] = f32::from(u8::from(evidence.instrument));
            row[at + 5] = (1.0 + n as f32).ln();
            debug_assert_eq!(at + 6, ALL);
        }
        Self { order, readings, references, looks: look_ids, features }
    }
}
