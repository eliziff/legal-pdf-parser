//! Tables, read by pdf-inspector's own detectors.
//!
//! Each page's words go to pdf-inspector as its text items and the page's painted rules as its
//! lines: its rule-grid detector reads the ruled tables, its white-space detector the rest of the
//! page where the parser's layout says white space can be read (`TableLayout`). What it finds is
//! mapped back onto the words it was read from, cell by cell. A contents list is no table, nor a
//! text set beside its translation, nor a list set in two columns (its markers counting down the
//! first). A table the page cuts off goes on at the top
//! of the next page, or under its repeated caption and head row, when its columns stand where
//! they stood.

use legal_pdf_core::line_font_size;
use legal_pdf_core::model::{Line, Page};
use pdf_inspector::tables::{detect_tables, detect_tables_from_lines, Table, TableKind};
use pdf_inspector::types::{ItemType, PdfLine, TextItem};
use std::collections::{HashMap, HashSet};

/// Layout readings the table detector takes from the parser that owns them.
pub trait TableLayout {
    /// The language a column of lines is written in: French, English, or neither clearly
    /// (`Some(true)`, `Some(false)`, `None`), on at least `minimum` telling words.
    fn column_language(&self, lines: &[&Line], minimum: usize) -> Option<bool>;
    /// Whether the page's white space can be read for tables set without rules: not a recognized
    /// page's, unless its lines stand in a plain grid.
    fn white_space(&self, page: &Page) -> bool;
}

/// Points within which rule positions coincide and rule ends meet.
const TOLERANCE: f64 = 2.0;

/// A line of a cell, or the part of it that stands in the cell when the line runs on
/// across a column into the next cell.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellPart {
    pub line_id: String,
    /// Character offsets of the part within its line's text.
    pub chars: Option<(usize, usize)>,
    text: String,
    bbox: [f64; 4],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableCell {
    pub column: usize,
    pub row_span: usize,
    pub column_span: usize,
    /// The cell's text in reading order; empty for a blank cell.
    pub parts: Vec<CellPart>,
}

impl TableCell {
    fn text(&self) -> String {
        self.parts
            .iter()
            .map(|part| part.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn line_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for part in &self.parts {
            if ids.last() != Some(&part.line_id) {
                ids.push(part.line_id.clone());
            }
        }
        ids
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TableRow {
    pub header: bool,
    /// The cells whose top-left corner is in this row, by column.
    pub cells: Vec<TableCell>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PdfTable {
    pub rows: Vec<TableRow>,
    page_index: usize,
    xs: Vec<f64>,
    top: f64,
    bottom: f64,
}

impl PdfTable {
    /// The table's lines, each once, in reading order.
    pub fn line_ids(&self) -> Vec<&String> {
        let mut seen = HashSet::new();
        self.rows
            .iter()
            .flat_map(|row| &row.cells)
            .flat_map(|cell| &cell.parts)
            .map(|part| &part.line_id)
            .filter(|id| seen.insert(*id))
            .collect()
    }

    /// The page the table ends on and where on it it ends: notes to the table are set below.
    pub fn end(&self) -> (usize, f64) {
        (self.page_index, self.bottom)
    }
}

fn bold(line: &Line) -> bool {
    let (mut total, mut heavy) = (0, 0);
    for span in &line.spans {
        let count = span.text.trim().chars().count();
        let font = span.font.to_ascii_lowercase();
        total += count;
        if span.flags & 16 != 0 || ["bold", "black", "heavy"].iter().any(|w| font.contains(w)) {
            heavy += count;
        }
    }
    total > 0 && heavy * 5 >= total * 4
}

/// A line, or the words of it that stand in one cell.
#[derive(Clone, Copy)]
struct Piece {
    line: usize,
    chars: Option<(usize, usize)>,
    bbox: [f64; 4],
}

/// A cell of pieces in reading order: by visual row, then from the left.
fn cell_of(page: &Page, column: usize, mut pieces: Vec<Piece>) -> TableCell {
    pieces.sort_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]));
    let mut band = 0;
    let mut bottom = f64::NEG_INFINITY;
    let mut keyed: Vec<(usize, Piece)> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        let height = piece.bbox[3] - piece.bbox[1];
        if piece.bbox[1] > bottom - height * 0.5 {
            band += 1;
        }
        bottom = bottom.max(piece.bbox[3]);
        keyed.push((band, piece));
    }
    keyed.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.bbox[0].total_cmp(&right.1.bbox[0]))
    });
    let text = |piece: &Piece| {
        let line = &page.lines[piece.line].text;
        match piece.chars {
            Some((start, end)) => line.chars().skip(start).take(end - start).collect(),
            None => line.clone(),
        }
    };
    TableCell {
        column,
        row_span: 1,
        column_span: 1,
        parts: keyed
            .into_iter()
            .map(|(_, piece)| CellPart {
                line_id: page.lines[piece.line].id.clone(),
                chars: piece.chars,
                text: text(&piece),
                bbox: piece.bbox,
            })
            .collect(),
    }
}

/// A word no word of the line stands for: the line read whole.
const WHOLE: usize = usize::MAX;

/// The page's words as pdf-inspector's text items (points, y up), with the (line, word) each was
/// read from. Running heads and footers, and lines the parser has set aside, are not offered.
fn text_items(page: &Page) -> (Vec<TextItem>, Vec<(usize, usize)>) {
    let (mut items, mut origin) = (Vec::new(), Vec::new());
    let item = |line: &Line, text: &str, bbox: [f64; 4], font: &str, underlined: bool| TextItem {
        text: text.to_owned(),
        x: bbox[0] as f32,
        y: (page.height - bbox[3]) as f32,
        width: (bbox[2] - bbox[0]) as f32,
        height: (bbox[3] - bbox[1]) as f32,
        rotation: 0.0,
        advance_known: true,
        font: font.to_owned(),
        font_tag: String::new(),
        legacy_symbol_rewrite: false,
        font_size: line_font_size(line).max(bbox[3] - bbox[1]) as f32,
        page: page.index as u32 + 1,
        is_bold: bold(line),
        is_italic: false,
        font_weight: None,
        bold_source: None,
        fixed_pitch: None,
        fill_color: None,
        stroke_color: None,
        render_mode: None,
        is_underline: underlined,
        is_strikeout: false,
        fidelity: None,
        item_type: ItemType::Text,
        mcid: None,
        baseline_shift: 0.0,
    };
    for (slot, line) in page.lines.iter().enumerate() {
        if line.exclude_from_body
            || matches!(line.region_type.as_str(), "header" | "footer")
            || line.text.trim().is_empty()
            || !line.bbox.iter().all(|value| value.is_finite())
        {
            continue;
        }
        let font = |at: usize| {
            line.spans
                .iter()
                .find(|span| span.start <= at && at < span.end.max(span.start + 1))
                .or(line.spans.first())
                .map_or("", |span| span.font.as_str())
        };
        let words: Vec<(usize, &legal_pdf_core::model::Word)> = line
            .words
            .iter()
            .enumerate()
            .filter(|(_, word)| !word.text.trim().is_empty())
            .collect();
        if words.is_empty() {
            items.push(item(line, line.text.trim(), line.bbox, font(0), false));
            origin.push((slot, WHOLE));
        }
        for (at, word) in words {
            items.push(item(line, &word.text, word.bbox, font(word.start), word.underlined));
            origin.push((slot, at));
        }
    }
    (items, origin)
}

/// The page's painted rules as pdf-inspector's lines (points, y up).
fn rules(page: &Page) -> Vec<PdfLine> {
    let number = page.index as u32 + 1;
    page.regions
        .iter()
        .filter(|region| region.kind == "rule")
        .map(|region| {
            let [x0, y0, x1, y1] = region.bbox;
            let flip = |y: f64| (page.height - y) as f32;
            if (y1 - y0).abs() <= (x1 - x0).abs() {
                let at = flip((y0 + y1) / 2.0);
                PdfLine { x1: x0.min(x1) as f32, y1: at, x2: x0.max(x1) as f32, y2: at, page: number }
            } else {
                let at = ((x0 + x1) / 2.0) as f32;
                PdfLine { x1: at, y1: flip(y0.max(y1)), x2: at, y2: flip(y0.min(y1)), page: number }
            }
        })
        .collect()
}

/// Which of `n` slots a position falls in: the detector gives `n + 1` boundaries, or one
/// position per slot (a centre or an edge), the nearest of which is taken.
fn slot(positions: &[f32], n: usize, at: f32) -> usize {
    let mut sorted = positions.to_vec();
    sorted.sort_by(f32::total_cmp);
    let k = if sorted.len() == n + 1 {
        sorted.partition_point(|position| *position <= at).saturating_sub(1)
    } else {
        (0..sorted.len())
            .min_by(|a, b| (sorted[*a] - at).abs().total_cmp(&(sorted[*b] - at).abs()))
            .unwrap_or(0)
    };
    k.min(n - 1)
}

/// The cell of the detector's grid an item stands in, by its position; where that cell's text
/// does not hold the item's, the nearest cell along its row, then its column, that does.
fn grid_cell(table: &Table, item: &TextItem) -> Option<(usize, usize)> {
    let rows = table.cells.len();
    let columns = table.cells.iter().map(Vec::len).max().unwrap_or(0);
    if rows == 0 || columns == 0 {
        return None;
    }
    let column = slot(&table.columns, columns, item.x + item.width / 2.0);
    // Rows run down the page; the detector's y runs up it.
    let row = rows - 1 - slot(&table.rows, rows, item.y + item.height / 2.0);
    let text = item.text.trim();
    let holds = |r: usize, c: usize| {
        table.cells.get(r).and_then(|cells| cells.get(c)).is_some_and(|cell| cell.contains(text))
    };
    let near = |n: usize, at: usize| {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|k| k.abs_diff(at));
        order
    };
    if holds(row, column) {
        return Some((row, column));
    }
    near(columns, column)
        .into_iter()
        .find(|&c| holds(row, c))
        .map(|c| (row, c))
        .or_else(|| near(rows, row).into_iter().find(|&r| holds(r, column)).map(|r| (r, column)))
        .or(Some((row, column)))
}

/// One cell's words as pieces: the words of one line that run on together make one piece, a
/// line all of whose words stand in the cell is read whole.
fn pieces(page: &Page, mut words: Vec<(usize, usize)>) -> Vec<Piece> {
    words.sort_unstable();
    let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
    for (line, word) in words {
        match runs.last_mut() {
            Some((at, run)) if *at == line && word != WHOLE && run.last().is_some_and(|last| *last != WHOLE && *last + 1 == word) => run.push(word),
            _ => runs.push((line, vec![word])),
        }
    }
    runs.into_iter()
        .map(|(slot, run)| {
            let line = &page.lines[slot];
            let all = line.words.iter().filter(|word| !word.text.trim().is_empty()).count();
            if run[0] == WHOLE || run.len() == all {
                return Piece { line: slot, chars: None, bbox: line.bbox };
            }
            let (first, last) = (&line.words[run[0]], &line.words[*run.last().expect("run")]);
            let bbox = run.iter().map(|&k| line.words[k].bbox).fold(first.bbox, |a, b| {
                [a[0].min(b[0]), a[1].min(b[1]), a[2].max(b[2]), a[3].max(b[3])]
            });
            Piece { line: slot, chars: Some((first.start, last.end)), bbox }
        })
        .collect()
}

/// A table pdf-inspector read, on the words it was read from.
fn converted(page: &Page, items: &[TextItem], origin: &[(usize, usize)], table: &Table) -> Option<PdfTable> {
    let columns = table.cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut grid: Vec<Vec<Vec<(usize, usize)>>> = vec![vec![Vec::new(); columns]; table.cells.len()];
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    for &k in &table.item_indices {
        let Some((row, column)) = items.get(k).and_then(|item| grid_cell(table, item)) else {
            continue;
        };
        let (line, word) = origin[k];
        grid[row][column].push((line, word));
        boxes.push(if word == WHOLE { page.lines[line].bbox } else { page.lines[line].words[word].bbox });
    }
    let rows: Vec<TableRow> = grid
        .into_iter()
        .filter(|row| row.iter().any(|cell| !cell.is_empty()))
        .map(|row| TableRow {
            header: false,
            cells: row
                .into_iter()
                .enumerate()
                .map(|(column, words)| cell_of(page, column, pieces(page, words)))
                .collect(),
        })
        .collect();
    if rows.len() < 2 || columns == 0 {
        return None;
    }
    let left = boxes.iter().map(|bbox| bbox[0]).fold(f64::INFINITY, f64::min);
    let right = boxes.iter().map(|bbox| bbox[2]).fold(f64::NEG_INFINITY, f64::max);
    let mut marks: Vec<f64> = table.columns.iter().map(|x| f64::from(*x)).collect();
    marks.sort_by(f64::total_cmp);
    // Column boundaries: the detector's own, or midway between its column positions.
    let xs = if marks.len() == columns + 1 {
        marks
    } else {
        std::iter::once(left)
            .chain(marks.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0))
            .chain(std::iter::once(right))
            .collect()
    };
    Some(PdfTable {
        rows,
        page_index: page.index,
        xs,
        top: boxes.iter().map(|bbox| bbox[1]).fold(f64::INFINITY, f64::min),
        bottom: boxes.iter().map(|bbox| bbox[3]).fold(f64::NEG_INFINITY, f64::max),
    })
}

/// Whether one column of a table reads in English and another in French: a text set beside
/// its translation under shared heads, not a table.
fn bilingual_columns(page: &Page, table: &PdfTable, layout: &dyn TableLayout) -> bool {
    let by_id: HashMap<&str, &Line> = page
        .lines
        .iter()
        .map(|line| (line.id.as_str(), line))
        .collect();
    let mut columns: HashMap<usize, Vec<&Line>> = HashMap::new();
    for cell in table.rows.iter().flat_map(|row| &row.cells) {
        columns.entry(cell.column).or_default().extend(
            cell.line_ids()
                .iter()
                .filter_map(|id| by_id.get(id.as_str()).copied()),
        );
    }
    let languages: Vec<Option<bool>> = columns
        .into_values()
        .map(|lines| layout.column_language(&lines, 5))
        .collect();
    languages.contains(&Some(true)) && languages.contains(&Some(false))
}

/// Where a marker cell stands in its count ("a)", "(iv)", "12.", "3.05", "•"): the count's family
/// and the marker's value in it. A marker keeps a bracket or a stop (a bare figure is a table's),
/// and OCR's "}" for ")" reads as ")".
fn list_marker(text: &str) -> Option<(u8, u32)> {
    let text = text.trim();
    if text.chars().count() == 1 && "•·▪◦●○■□➢-–—*\u{f0b7}\u{f0a7}\u{f0d8}".contains(text) {
        return Some((0, 0));
    }
    let inner = text.trim_start_matches(['(', '[']).trim_end_matches([')', ']', '}', '.']);
    if inner.is_empty() || inner.len() == text.len() || inner.chars().count() > 6 {
        return None;
    }
    let parts: Vec<&str> = inner.split('.').collect();
    if parts.iter().all(|part| !part.is_empty() && part.len() <= 3 && part.bytes().all(|b| b.is_ascii_digit())) {
        return Some((parts.len() as u8, parts.last()?.parse().ok()?));
    }
    let upper = inner.bytes().all(|b| b.is_ascii_uppercase());
    if inner.len() == 1 && inner.bytes().all(|b| b.is_ascii_alphabetic()) {
        let letter = inner.bytes().next()?.to_ascii_lowercase();
        return Some((if upper { 6 } else { 5 }, u32::from(letter - b'a') + 1));
    }
    let digit = |b: u8| match b.to_ascii_lowercase() { b'i' => Some(1), b'v' => Some(5), b'x' => Some(10), b'l' => Some(50), b'c' => Some(100), _ => None };
    let values = inner.bytes().map(digit).collect::<Option<Vec<i64>>>()?;
    if !(upper || inner.bytes().all(|b| b.is_ascii_lowercase())) {
        return None;
    }
    let value: i64 = values.iter().enumerate().map(|(k, v)| if values.get(k + 1).is_some_and(|next| next > v) { -v } else { *v }).sum();
    Some((if upper { 8 } else { 7 }, u32::try_from(value).ok().filter(|v| *v > 0)?))
}

/// Whether a table pdf-inspector read is a list set in two columns: each row that fills both holds
/// a marker in the first and its item's words in the second, and the markers count on from row to
/// row (a), b), c) ...; 1., 2., 3.), a count opening at its first value where a list starts again.
/// A row of one cell may stand between: a heading over the items, or a line an item runs on to.
fn two_column_list(table: &PdfTable) -> bool {
    if table.xs.len() != 3 {
        return false;
    }
    let mut markers = Vec::new();
    for row in &table.rows {
        let text = |column: usize| row.cells.iter().filter(|cell| cell.column == column).map(TableCell::text).collect::<Vec<_>>().join(" ");
        let (left, right) = (text(0), text(1));
        if left.trim().is_empty() || right.trim().is_empty() {
            continue;
        }
        match list_marker(&left) {
            Some(marker) => markers.push(marker),
            None => return false,
        }
    }
    let follows = markers.windows(2).filter(|pair| {
        let ((family, value), (next_family, next)) = (pair[0], pair[1]);
        family == next_family && (family == 0 || next == value + 1) || next == 1
    }).count();
    markers.len() >= 3 && follows * 3 >= (markers.len() - 1) * 2
}

/// A page's tables, and the lines of the contents lists pdf-inspector read there as tables.
fn page_tables(
    page: &Page,
    translation: &HashSet<String>,
    layout: &dyn TableLayout,
) -> (Vec<PdfTable>, Vec<String>) {
    let (items, origin) = text_items(page);
    if items.len() < 6 {
        return (Vec::new(), Vec::new());
    }
    let rules = rules(page);
    let mut found: Vec<Table> = if rules.is_empty() {
        Vec::new()
    } else {
        detect_tables_from_lines(&items, &rules, page.index as u32 + 1)
    };
    if layout.white_space(page) {
        let taken: HashSet<usize> = found.iter().flat_map(|table| table.item_indices.iter().copied()).collect();
        let rest: Vec<usize> = (0..items.len()).filter(|k| !taken.contains(k)).collect();
        let free: Vec<TextItem> = rest.iter().map(|&k| items[k].clone()).collect();
        let mut sizes: Vec<f32> = free.iter().map(|item| item.font_size).collect();
        sizes.sort_by(f32::total_cmp);
        let body = sizes.get(sizes.len() / 2).copied().unwrap_or(10.0);
        for mut table in detect_tables(&free, body, false) {
            table.item_indices = table.item_indices.iter().filter_map(|&k| rest.get(k).copied()).collect();
            found.push(table);
        }
    }
    let contents = found
        .iter()
        .filter(|table| table.kind == TableKind::Toc)
        .flat_map(|table| table.item_indices.iter().filter_map(|&k| origin.get(k)))
        .map(|&(line, _)| page.lines[line].id.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut tables: Vec<PdfTable> = found
        .iter()
        .filter(|table| table.kind == TableKind::Data)
        .filter_map(|table| converted(page, &items, &origin, table))
        .filter(|table| !table.line_ids().into_iter().any(|id| translation.contains(id)))
        .filter(|table| !bilingual_columns(page, table, layout))
        .filter(|table| !two_column_list(table))
        .collect();
    tables.sort_by(|left, right| left.top.total_cmp(&right.top));
    (tables, contents)
}

fn row_text(table: &PdfTable, row: usize) -> Vec<String> {
    table.rows[row].cells.iter().map(TableCell::text).collect()
}

/// The column rules of a table and of the table that may continue it, together, when
/// both stand between the same outer rules and every column rule of the one with fewer
/// columns stands where a rule of the other does.
fn shared_columns(prior: &[f64], next: &[f64]) -> Option<Vec<f64>> {
    const SLACK: f64 = 3.0 * TOLERANCE;
    let near = |value: f64, among: &[f64]| among.iter().any(|x| (x - value).abs() <= SLACK);
    let (fewer, more) = if prior.len() <= next.len() {
        (prior, next)
    } else {
        (next, prior)
    };
    ((prior[0] - next[0]).abs() <= SLACK
        && (prior[prior.len() - 1] - next[next.len() - 1]).abs() <= SLACK
        && fewer.iter().all(|x| near(*x, more)))
    .then(|| {
        let mut union: Vec<f64> = more.to_vec();
        union.extend(fewer.iter().filter(|x| !near(**x, more)));
        union.sort_by(f64::total_cmp);
        union
    })
}

/// Renumbers a table's columns on a grid with more column rules.
fn regrid(table: &mut PdfTable, xs: &[f64]) {
    let nearest = |value: f64| {
        (0..xs.len())
            .min_by(|left, right| {
                (xs[*left] - value)
                    .abs()
                    .total_cmp(&(xs[*right] - value).abs())
            })
            .unwrap_or(0)
    };
    let old = std::mem::replace(&mut table.xs, xs.to_vec());
    for cell in table.rows.iter_mut().flat_map(|row| &mut row.cells) {
        let left = nearest(old[cell.column]);
        let right = nearest(old[cell.column + cell.column_span]);
        cell.column = left;
        cell.column_span = right.saturating_sub(left).max(1);
    }
}

/// Every table in the document, a table cut off by a page break joined to its continuation,
/// in reading order; and the lines of the contents lists pdf-inspector read as tables.
pub fn document_tables(
    pages: &[Page],
    translation: &HashSet<String>,
    layout: &dyn TableLayout,
) -> (Vec<PdfTable>, HashSet<String>) {
    let mut tables: Vec<PdfTable> = Vec::new();
    let mut contents = HashSet::new();
    let mut previous_last: Option<usize> = None;
    for page in pages {
        let (found, listed) = page_tables(page, translation, layout);
        contents.extend(listed);
        let empty_band = |from: f64, to: f64| {
            !page.lines.iter().any(|line| {
                !line.exclude_from_body
                    && !line.text.trim().is_empty()
                    && line.bbox[1] > from
                    && line.bbox[3] < to
            })
        };
        let lines_above = |top: f64| {
            page.lines
                .iter()
                .filter(|line| {
                    !line.exclude_from_body
                        && !line.text.trim().is_empty()
                        && line.bbox[1] > page.height * 0.08
                        && line.bbox[3] < top
                })
                .count()
        };
        let mut first = true;
        for mut table in found {
            // A table goes on at the top of the next page, or under no more than its
            // repeated caption when it repeats its head row.
            let union = previous_last
                .filter(|index| {
                    let prior = &tables[*index];
                    first
                        && prior.page_index + 1 == page.index
                        && (empty_band(page.height * 0.12, table.top)
                            || (lines_above(table.top) <= 3
                                && !table.rows.is_empty()
                                && prior.rows[0].cells.len() == table.rows[0].cells.len()
                                && row_text(prior, 0) == row_text(&table, 0)))
                })
                .and_then(|index| {
                    let prior = &tables[index];
                    // Columns set by white space alone shift a little from page to page; a
                    // repeated head row with as many columns keeps them.
                    shared_columns(&prior.xs, &table.xs).or_else(|| {
                        (prior.xs.len() == table.xs.len()
                            && !table.rows.is_empty()
                            && row_text(prior, 0) == row_text(&table, 0))
                        .then(|| prior.xs.clone())
                    })
                });
            first = false;
            let Some(union) = union else {
                tables.push(table);
                continue;
            };
            let prior = &mut tables[previous_last.expect("continued table")];
            regrid(prior, &union);
            regrid(&mut table, &union);
            let heads = prior
                .rows
                .iter()
                .take_while(|row| row.header)
                .count()
                .max(1);
            let repeated = (0..heads.min(table.rows.len()))
                .take_while(|row| {
                    !prior.rows[*row].cells.is_empty()
                        && prior.rows[*row]
                            .cells
                            .iter()
                            .map(|cell| cell.column)
                            .eq(table.rows[*row].cells.iter().map(|cell| cell.column))
                        && row_text(prior, *row) == row_text(&table, *row)
                })
                .count();
            for row in 0..repeated {
                table.rows[row].header = true;
            }
            prior.rows.append(&mut table.rows);
            prior.page_index = table.page_index;
            prior.bottom = table.bottom;
        }
        previous_last = tables
            .iter()
            .rposition(|table| table.page_index == page.index)
            .filter(|index| empty_band(tables[*index].bottom, page.height * 0.88));
    }
    (tables, contents)
}

/// Puts each table's lines on the page in row order, cell by cell, where its first line
/// stood, and renumbers the page's reading order.
pub fn order_table_lines(page: &mut Page, tables: &[PdfTable]) {
    let keys: HashMap<&str, (usize, usize)> = tables
        .iter()
        .enumerate()
        .flat_map(|(table, value)| {
            value
                .line_ids()
                .into_iter()
                .enumerate()
                .map(move |(at, id)| (id.as_str(), (table, at)))
        })
        .collect();
    if !page
        .lines
        .iter()
        .any(|line| keys.contains_key(line.id.as_str()))
    {
        return;
    }
    let lines = std::mem::take(&mut page.lines);
    let mut held: HashMap<usize, Vec<(usize, Line)>> = HashMap::new();
    let mut order: Vec<Result<Line, usize>> = Vec::with_capacity(lines.len());
    for line in lines {
        match keys.get(line.id.as_str()).copied() {
            Some((table, at)) => {
                let slot = held.entry(table).or_default();
                if slot.is_empty() {
                    order.push(Err(table));
                }
                slot.push((at, line));
            }
            None => order.push(Ok(line)),
        }
    }
    for entry in order {
        match entry {
            Ok(line) => page.lines.push(line),
            Err(table) => {
                let mut cells = held.remove(&table).unwrap_or_default();
                cells.sort_by_key(|(at, _)| *at);
                page.lines.extend(cells.into_iter().map(|(_, line)| line));
            }
        }
    }
    for (index, line) in page.lines.iter_mut().enumerate() {
        line.reading_order = index + 1;
    }
}

/// Each cell's text is a block of its own: a paragraph never runs from one cell into the next,
/// though a cell may hold several.
pub fn part_cell_paragraphs(pages: &mut [Page], tables: &[PdfTable]) {
    // A line parted between cells reads with the first of them.
    let mut cells: HashMap<&str, usize> = HashMap::new();
    for (cell, value) in tables
        .iter()
        .flat_map(|table| &table.rows)
        .flat_map(|row| &row.cells)
        .enumerate()
    {
        for part in &value.parts {
            cells.entry(part.line_id.as_str()).or_insert(cell);
        }
    }
    for page in pages {
        let mut next = page
            .lines
            .iter()
            .map(|line| line.block_index)
            .max()
            .unwrap_or(0)
            + 1;
        let mut prior: Option<(usize, usize)> = None;
        for line in &mut page.lines {
            let Some(&cell) = cells.get(line.id.as_str()) else {
                prior = None;
                continue;
            };
            if prior
                .is_none_or(|(prior_cell, block)| prior_cell != cell || block != line.block_index)
            {
                next += 1;
            }
            prior = Some((cell, line.block_index));
            line.block_index = next;
            line.note_region_mode.clear();
        }
    }
}

/// Line ids of every table's cells.
pub fn table_line_ids(tables: &[PdfTable]) -> HashSet<String> {
    tables
        .iter()
        .flat_map(PdfTable::line_ids)
        .cloned()
        .collect()
}
