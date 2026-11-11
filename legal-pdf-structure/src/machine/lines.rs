//! What the parser's scorer reads of each line before it reads any state: the line set against its
//! page and its document (place, gaps, type, look and how common the look is, ink, its column), the
//! shape of its text, and what the readers found on it (its marker's readings, a note label, reference
//! marks, the words the lexicon knows, a furniture edge, a table cell, the note band). Nothing here is
//! a decision: every value is measured on the pages or read by an evidence reader.

use super::*;
use legal_pdf_core::{flag_share, text_shape, BOLD, ITALIC, MONOSPACED, SUPERSCRIPT};
use crate::layout::Geometry;

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
/// The roles `Role` names (`docparse.T0`).
const ROLES: usize = 14;
/// What t0 decided of a line: its role, one of `ROLES`, twelve flags and its heading's level (`docparse.T0`).
pub(crate) const T0: usize = ROLES + 13;

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

/// The line's text with its opening token read as recognition likely misread it (`confusions`, each
/// misread character's meant ones at a twentieth or more): each character read as its likeliest meaning
/// alone, and all of them at once; at most six variants.
fn misread_variants(text: &str, confusions: &HashMap<char, Vec<(char, f32)>>) -> Vec<String> {
    let start = text.len() - text.trim_start().len();
    let token_end = text[start..].find(char::is_whitespace).map_or(text.len(), |at| start + at);
    let token: Vec<char> = text[start..token_end].chars().collect();
    let meant: Vec<Option<char>> = token.iter().map(|c| confusions.get(c).and_then(|m| m.first()).filter(|(_, p)| *p >= 0.05).map(|(c, _)| *c)).collect();
    let rebuild = |chars: &[char]| format!("{}{}{}", &text[..start], chars.iter().collect::<String>(), &text[token_end..]);
    let mut variants = Vec::new();
    for (k, m) in meant.iter().enumerate() {
        if let Some(c) = m {
            let mut chars = token.clone();
            chars[k] = *c;
            variants.push(rebuild(&chars));
        }
    }
    if variants.len() > 1 {
        let all: Vec<char> = token.iter().zip(&meant).map(|(c, m)| m.unwrap_or(*c)).collect();
        variants.push(rebuild(&all));
    }
    variants.truncate(6);
    variants
}

/// A line's opening as the numbering grammar would read the markers the print runs together or
/// dresses: a revision mark before the marker ("*18.02", "†L #15"), an ordinal ("1st."), a section
/// glued to its subsection ("16(2)": the section, then the subsection), and a second marker after a
/// first ("5 (1)", "2.01 (a)").
fn marker_variants(text: &str) -> Vec<String> {
    static GLUED: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static ORDINAL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let glued = GLUED.get_or_init(|| regex::Regex::new(r"^(\d{1,4}(?:\.\d{1,3})*)(\([0-9A-Za-z]{1,4}\).*)$").unwrap());
    let ordinal = ORDINAL.get_or_init(|| regex::Regex::new(r"^(\d{1,3})(?:st|nd|rd|th)\.?$").unwrap());
    let trimmed = text.trim_start();
    let mut out = Vec::new();
    let unmarked = trimmed.trim_start_matches(['*', '†', '‡']).trim_start();
    if unmarked.len() < trimmed.len() && !unmarked.is_empty() {
        out.push(unmarked.to_owned());
    }
    let mut tokens = unmarked.split_whitespace();
    let Some(first) = tokens.next() else { return out };
    let rest: String = unmarked[first.len()..].to_owned();
    if let Some(found) = ordinal.captures(first) {
        out.push(format!("{}.{rest}", &found[1]));
    }
    if let Some(found) = glued.captures(first) {
        out.push(format!("{}.{rest}", &found[1]));
        out.push(format!("{}{rest}", &found[2]));
    }
    if let Some(second) = tokens.next().filter(|token| token.starts_with('(') && token.ends_with(')') && token.chars().count() <= 6) {
        if !marker_readings(first, false).is_empty() {
            out.push(unmarked[unmarked.find(second).unwrap_or(0)..].to_owned());
        }
    }
    out
}

/// Each line's readings and features, in reading order (`fences.order`): `WIDTH` own features per
/// line, then `ALL` with the lines around it and the document's shape.
pub(crate) struct LineTable {
    pub(crate) readings: Vec<Vec<MarkerReading>>,
    /// Each line's reference marks: value, first and past-last UTF-16 code unit (`reference_spans`).
    pub(crate) references: Vec<Vec<(u32, u32, u32)>>,
    /// Each line's look: its face, half-point size, bold, italic and capitals, as one number.
    pub(crate) looks: Vec<u32>,
    /// Each paragraph number the margin prints (`Fences::read_margin_numbers`), by its line's place: the place of
    /// the row's line it numbers, and of its twin's in the other language's column, where it has one.
    pub(crate) margin_rows: HashMap<usize, (usize, Option<usize>)>,
    /// Each line a margin number numbers (a row's or a twin's), and each margin number's own line, by its place: the
    /// margin line's place.
    pub(crate) margins: HashMap<usize, usize>,
}

/// What the walk measures over the line table, in reading order: each line's features (`ALL` a line) and what t0
/// decided of it (`T0` a line), the DOC reader's input.
pub(crate) struct Measures {
    pub(crate) features: Vec<f32>,
    pub(crate) t0: Vec<u8>,
}

impl Measures {
    /// The line's features.
    pub(crate) fn row(&self, k: usize) -> &[f32] {
        &self.features[k * ALL..(k + 1) * ALL]
    }
}

impl LineTable {
    /// Each line's features and t0 row, measured over the table both tiers read (`plain`), for the DOC reader.
    pub(crate) fn measure(&self, pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, geometry: &Geometry, notes: &Notes, decision: &Decision) -> Measures {
        let (order, readings, margin_rows, margins) = (&fences.order, &self.readings, &self.margin_rows, &self.margins);
        let n = order.len();
        let measures: Vec<(&legal_pdf_core::PageMeasure, &legal_pdf_core::Columns, HashMap<usize, usize>, f64, f64)> = pages.iter().enumerate().map(|(p, page)| {
            let (measure, columns) = geometry.page(pages, p);
            let rank = measure.order.iter().enumerate().map(|(k, &line)| (line, k)).collect();
            let ink = legal_pdf_core::upper_median(page.lines.iter().map(|l| l.ink).filter(|ink| *ink > 0.0).collect());
            let weight = legal_pdf_core::upper_median(page.lines.iter().map(|l| l.weight).filter(|w| *w > 0.0).collect());
            (measure, columns, rank, ink, weight)
        }).collect();
        let off_grid: Vec<usize> = pages.iter().map(|page| legal_pdf_core::off_grid_sizes(page.lines.iter().filter(|line| !line.exclude_from_body))).collect();
        let looks = Looks::tally(pages, slots, evidence);
        let repeated = repeated_at_edges(pages, slots, order);
        let page_count = pages.len().max(1) as f64;
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
                label.map_or(0.0, |l| f64::from(l.size) / slots.line(pages, Slot(at as u32)).size.max(1.0)),
                flag(label.is_some_and(|l| l.suppress != 0)),
                flag(evidence.references[at] & REFERENCE_SUPERSCRIPT != 0), flag(evidence.references[at] & REFERENCE_DETACHED != 0),
                flag(evidence.references[at] & REFERENCE_RAISED != 0), flag(evidence.references[at] & REFERENCE_GLUED != 0),
                flag(evidence.title_shaped[at]), flag(evidence.beside[at].is_some()), flag(evidence.line_number[at]),
                flag(repeated[k]), flag(fences.cell[at] > 0),
                flag(evidence.pages[p].separator.is_some_and(|y| line.bbox[1] >= y)), flag(fences.at(at).translation),
            ]);
            let found = &readings[k];
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
        // What t0 decided of each line, for the DOC reader that works from it: the role the machine settled on, the note
        // band, whether the judgment reader opens or holds a paragraph on it and whether that paragraph goes on from the
        // line before in its language, whether it is a margin's number or a line one numbers, whether it goes on with the
        // line before it, whether the notes read it a note, whether a quotation holds it, whether a constituent document
        // starts on it, whether it opens a unit or joins the line above, and its heading's level.
        let starts: HashSet<&str> = fences.starts.iter().map(|start| start.line_id.as_str()).collect();
        let mut t0 = vec![0u8; n * T0];
        let mut last_held: [&[Held]; 2] = [&[], &[]];
        for (k, &slot) in order.iter().enumerate() {
            let at = slot.index();
            let row = &mut t0[k * T0..(k + 1) * T0];
            row[(decision.role[at] as usize).min(ROLES - 1)] = 1;
            let held = &fences.paragraphs.held[at];
            let goes_on = held.iter().any(|mine| last_held[usize::from(mine.translation)].iter().any(|other| other.number == mine.number));
            let flags = [fences.band[at], fences.paragraphs.count[at].is_some(), !held.is_empty(), goes_on, margin_rows.contains_key(&k),
                margins.get(&k).is_some_and(|&margin| margin != k), fences.continues[at].is_some(), notes.note[at].is_some(),
                fences.at(at).quotation != 0, starts.contains(slots.line(pages, slot).id.as_str()), decision.unit[at].is_some(), decision.joined[at]];
            for (cell, flag) in row[ROLES..].iter_mut().zip(flags) {
                *cell = u8::from(flag);
            }
            row[T0 - 1] = decision.level[at];
            for mine in held {
                last_held[usize::from(mine.translation)] = held;
            }
        }
        Measures { features, t0 }
    }

    /// The table both tiers read, nothing measured: each line's readings (with the recognizer's likely misreadings,
    /// `confusions`), its reference marks (with the note marks recognition misread, `unmarked_notes`), its look and its
    /// margin number. `before`: the text of each line whose marker the recognition repair rewrote, as recognized; its
    /// markers' readings stand beside the repaired text's, the parser's to choose between.
    pub(crate) fn plain(pages: &[Page], slots: &Slots, evidence: &Evidence, fences: &Fences, before: &[(Slot, String)], confusions: &HashMap<char, Vec<(char, f32)>>) -> Self {
        let order = &fences.order;
        let recognized: HashMap<Slot, &str> = before.iter().map(|(slot, text)| (*slot, text.as_str())).collect();
        let readings: Vec<Vec<MarkerReading>> = order.iter().map(|&slot| {
            let word = evidence.words[slot.index()] & WORD_DIVISION != 0;
            line_readings(slots.line(pages, slot), word, recognized.get(&slot).copied(), confusions)
        }).collect();
        let mut references: Vec<Vec<(u32, u32, u32)>> = order.iter().map(|&slot| reference_spans(slots.line(pages, slot), slots.line(pages, slot).size)).collect();
        unmarked_notes(order, slots, pages, evidence, &mut references);
        let looks = order.iter().map(|slot| look_id(evidence, slot.index())).collect();
        let (margin_rows, margins) = margin_lines(order, &readings, fences);
        Self { readings, references, looks, margin_rows, margins }
    }
}

/// A line's look as one number: its face, half-point size, bold, italic and capitals.
fn look_id(evidence: &Evidence, at: usize) -> u32 {
    u32::from(evidence.face[at]) << 20 | u32::from(evidence.half_points[at].min(0xffff)) << 4 | u32::from(evidence.look[at] & LOOK_STYLE)
}

/// A line's markers as the numbering grammar reads them (`word`: it opens on a division's word): its text's, those of the
/// print's run-ons taken apart, those of its text as recognized before the repair (`recognized`), and those of each likely
/// misreading of a recognized line's characters.
/// The marks a page's notes call for that no line of its body sets, found as recognition likely misread them: a note's
/// number in figures read as letters ("3o" for 30, "Ioo" for 100, "l2" for 12) or standing loose after a word ("64"),
/// and a note symbol read as another sign (a dagger as "t" or "+", a double dagger as "f", an asterisk as "^", "!", "?"
/// or an unreadable sign), glued to a word's end, or alone on its line. The body is the page's lines read before its
/// first note; each wanted note takes the first such place in it that is no mark yet.
fn unmarked_notes(order: &[Slot], slots: &Slots, pages: &[Page], evidence: &Evidence, references: &mut [Vec<(u32, u32, u32)>]) {
    static NUMBER: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    static SYMBOL: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let number = NUMBER.get_or_init(|| regex::Regex::new(r"(?:^|[\p{L}.,;:)\]\u{201d}\u{2019}\u{bb}'\s])([0-9oOIl|]{1,3})(?:[\s.,;:)\u{2014}-]|$)").unwrap());
    let symbol = SYMBOL.get_or_init(|| regex::Regex::new(r"[\p{L}\p{N}.,;:)\]\u{201d}\u{2019}]([tf+^!?*\u{fffd}\u{2020}\u{2021}])(?:[\s.,;:]|$)").unwrap());
    // The figures a run of confusable signs stands for; a run with no figure stands for a number only when longer than
    // one sign ("Ioo"; a lone "I" is the pronoun).
    let figures = |run: &str| -> Option<u32> {
        let text: Option<String> = run.chars().map(|c| match c { '0'..='9' => Some(c), 'o' | 'O' => Some('0'), 'I' | 'l' | '|' => Some('1'), _ => None }).collect();
        if !run.chars().any(|c| c.is_ascii_digit()) && run.chars().count() < 2 { return None; }
        text?.parse().ok()
    };
    let mut by_page: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for (k, &slot) in order.iter().enumerate() {
        by_page.entry(slots.page(slot)).or_default().push(k);
    }
    for lines in by_page.values() {
        let Some(first_note) = lines.iter().position(|&k| evidence.label[order[k].index()].is_some()) else { continue };
        let (body, notes) = lines.split_at(first_note);
        for value in notes.iter().filter_map(|&k| evidence.label[order[k].index()].map(|label| label.value)) {
            if body.iter().any(|&k| references[k].iter().any(|&(v, ..)| v == value)) {
                continue;
            }
            'search: for &k in body {
                let text = &slots.line(pages, order[k]).text;
                let place = |start: usize, end: usize| (legal_structure_model::utf16_len(&text[..start]) as u32, legal_structure_model::utf16_len(&text[..end]) as u32);
                let mut candidates: Vec<(u32, u32)> = Vec::new();
                if value == 0 {
                    if text.trim() == "*" {
                        let start = text.find('*').unwrap_or(0);
                        candidates.push(place(start, start + 1));
                    }
                    candidates.extend(symbol.captures_iter(text).filter_map(|c| c.get(1)).map(|m| place(m.start(), m.end())));
                } else {
                    candidates.extend(number.captures_iter(text).filter_map(|c| c.get(1))
                        .filter(|m| figures(m.as_str()) == Some(value)).map(|m| place(m.start(), m.end())));
                }
                for (start, end) in candidates {
                    if !references[k].iter().any(|&(_, a, b)| a < end && start < b) {
                        references[k].push((value, start, end));
                        references[k].sort_by_key(|&(_, start, _)| start);
                        break 'search;
                    }
                }
            }
        }
    }
}

fn line_readings(line: &Line, word: bool, recognized: Option<&str>, confusions: &HashMap<char, Vec<(char, f32)>>) -> Vec<MarkerReading> {
    let mut found = marker_readings(&line.text, word);
    let mut add = |readings: Vec<MarkerReading>| {
        for reading in readings {
            if !found.contains(&reading) {
                found.push(reading);
            }
        }
    };
    for variant in marker_variants(&line.text) {
        add(marker_readings(&variant, false));
    }
    if let Some(text) = recognized {
        add(marker_readings(text, word));
    }
    if line.recognized() {
        for variant in misread_variants(&line.text, confusions) {
            add(marker_readings(&variant, word));
        }
    }
    found
}

/// The margin's paragraph numbers, each with the lines its number opens: a number the grammar reads, its row's and twin's
/// lines read in the parse. Each number's place with its row's and twin's; and each line a number numbers, and each
/// number's own line, with the number's place.
fn margin_lines(order: &[Slot], readings: &[Vec<MarkerReading>], fences: &Fences) -> (HashMap<usize, (usize, Option<usize>)>, HashMap<usize, usize>) {
    let place = |slot: Slot| fences.place[slot.index()] as usize;
    let mut margin_rows = HashMap::new();
    let mut margins = HashMap::new();
    for (k, &slot) in order.iter().enumerate() {
        let Some(row) = fences.margin_number[slot.index()].map(place) else { continue };
        if readings[k].is_empty() {
            continue;
        }
        let twin = fences.margin_twin[slot.index()].map(place).filter(|&twin| twin != row);
        margin_rows.insert(k, (row, twin));
        for opened in [k, row].into_iter().chain(twin) {
            margins.insert(opened, k);
        }
    }
    (margin_rows, margins)
}
