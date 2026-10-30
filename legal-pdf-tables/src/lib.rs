//! Tables, read by pdf-inspector's own detectors.
//!
//! Each page's words go to pdf-inspector as its text items, the page's filled rectangles as its
//! rects and its painted rules as its lines, and its detectors read them in its own order: filled
//! cell grids first, then ruled tables where no grid was found, then white space for the rest of
//! the page where the parser's layout says white space can be read (`TableLayout`). Each word
//! goes in the cell the detector placed it in. A contents list is no table, nor a
//! text set beside its translation, nor a list set in two columns (its markers counting down the
//! first). A table the page cuts off goes on at the top
//! of the next page, or under its repeated caption and head row, when its columns stand where
//! they stood.

use legal_pdf_core::{flag_share, line_font_size, BOLD};
use legal_pdf_core::model::{Line, Page, Span, ITALIC, MONOSPACED};
use pdf_inspector::tables::{
    detect_tables, detect_tables_from_lines, detect_tables_from_rects, try_build_rect_guided_table,
    Table, TableKind,
};
use pdf_inspector::types::{ItemType, PdfLine, PdfRect, TextItem};
use std::collections::{HashMap, HashSet};

/// Layout readings the table detector takes from the parser that owns them.
pub trait TableLayout {
    /// The language a column of lines is written in: French, English, or neither clearly
    /// (`Some(true)`, `Some(false)`, `None`), on at least `minimum` telling words.
    fn column_language(&self, lines: &[&Line], minimum: usize) -> Option<bool>;
    /// Whether the page's white space can be read for tables set without rules: not a recognized
    /// page's, unless the lines its items (`page_items`) offer stand in a plain grid.
    fn white_space(&self, page: &Page, items: &PageItems) -> bool;
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
    flag_share(line, BOLD).is_some_and(|share| share >= 0.8)
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
pub const WHOLE: usize = usize::MAX;

/// Lines' words as pdf-inspector's text items, with the (line, word) each was read from.
#[derive(Debug)]
pub struct PageItems {
    pub items: Vec<TextItem>,
    /// The line's slot and the word's index for each item; `WHOLE` for a line read whole.
    pub origin: Vec<(usize, usize)>,
}

/// A page's words as text items on the page's own number (its index + 1), the number its rules
/// and filled rectangles are read on: the one reading of the page its table readers share.
pub fn page_items(page: &Page) -> PageItems {
    line_items(&page.lines, page.index as u32 + 1, page.height)
}

/// Lines' words as pdf-inspector's text items (points, y up, on page `number`), with the (line,
/// word) each was read from, so its readers read our pages, recognized ones too. Lines the parser
/// holds out of the body (a transcript's line numbers) are not offered.
pub fn line_items(lines: &[Line], number: u32, height: f64) -> PageItems {
    let (mut items, mut origin) = (Vec::new(), Vec::new());
    // Each word as its span sets it: face, weight, slant and pitch; a recognized line's by its
    // strokes alone.
    let item = |line: &Line, text: &str, bbox: [f64; 4], span: Option<&Span>, underlined: bool, struck: bool| TextItem {
        text: text.to_owned(),
        x: bbox[0] as f32,
        y: (height - bbox[3]) as f32,
        width: (bbox[2] - bbox[0]) as f32,
        height: (bbox[3] - bbox[1]) as f32,
        rotation: 0.0,
        advance_known: true,
        font: span.map_or(String::new(), |span| span.font.clone()),
        font_tag: String::new(),
        legacy_symbol_rewrite: false,
        font_size: line_font_size(line).max(bbox[3] - bbox[1]) as f32,
        page: number,
        is_bold: span.map_or_else(|| bold(line), |span| span.flags & BOLD != 0),
        is_italic: span.is_some_and(|span| span.flags & ITALIC != 0),
        font_weight: None,
        bold_source: None,
        fixed_pitch: span.map(|span| span.flags & MONOSPACED != 0),
        fill_color: None,
        stroke_color: None,
        render_mode: None,
        is_underline: underlined,
        is_strikeout: struck,
        fidelity: None,
        item_type: ItemType::Text,
        mcid: None,
        baseline_shift: 0.0,
    };
    for (slot, line) in lines.iter().enumerate() {
        if line.exclude_from_body
            || line.text.trim().is_empty()
            || !line.bbox.iter().all(|value| value.is_finite())
        {
            continue;
        }
        let span = |at: usize| {
            line.spans
                .iter()
                .find(|span| span.start <= at && at < span.end.max(span.start + 1))
                .or(line.spans.first())
        };
        let words: Vec<(usize, &legal_pdf_core::model::Word)> = line
            .words
            .iter()
            .enumerate()
            .filter(|(_, word)| !word.text.trim().is_empty())
            .collect();
        if words.is_empty() {
            items.push(item(line, line.text.trim(), line.bbox, span(0), false, false));
            origin.push((slot, WHOLE));
        }
        for (at, word) in words {
            items.push(item(line, &word.text, word.bbox, span(word.start), word.underlined, word.struck));
            origin.push((slot, at));
        }
    }
    PageItems { items, origin }
}

/// Clears the underline and strike read under the words of a table drawn with rules or filled
/// cells, as pdf-inspector clears them (suppress_table_underlines): the table's rulings, not the
/// words'. The page's items (`page_items`) are cleared with its words.
pub fn clear_table_underlines(page: &mut Page, items: &mut PageItems) {
    if !items.items.iter().any(|item| item.is_underline || item.is_strikeout) {
        return;
    }
    let marked: Vec<bool> = items.items.iter().map(|item| item.is_underline || item.is_strikeout).collect();
    pdf_inspector::extractor::suppress_table_underlines(&mut items.items, &rects(page), &rules(page), page.index as u32 + 1);
    for ((item, marked), &(line, word)) in items.items.iter().zip(marked).zip(&items.origin) {
        if marked && !item.is_underline && !item.is_strikeout && word != WHOLE {
            let word = &mut page.lines[line].words[word];
            word.underlined = false;
            word.struck = false;
        }
    }
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

/// The page's filled rectangles as pdf-inspector's rects (points, y up, from the bottom-left
/// corner).
fn rects(page: &Page) -> Vec<PdfRect> {
    let number = page.index as u32 + 1;
    page.regions
        .iter()
        .filter(|region| region.kind == "rect")
        .map(|region| {
            let [x0, y0, x1, y1] = region.bbox;
            PdfRect {
                x: x0.min(x1) as f32,
                y: (page.height - y0.max(y1)) as f32,
                width: (x1 - x0).abs() as f32,
                height: (y1 - y0).abs() as f32,
                page: number,
            }
        })
        .collect()
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

/// A table pdf-inspector read, on the words it was read from: each word in the cell the detector
/// placed it in.
fn converted(page: &Page, origin: &[(usize, usize)], table: &Table) -> Option<PdfTable> {
    let columns = table.cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut grid: Vec<Vec<Vec<(usize, usize)>>> = vec![vec![Vec::new(); columns]; table.cells.len()];
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let mut placed = HashSet::new();
    for &(k, row, column) in &table.item_cells {
        let Some(&(line, word)) = origin.get(k) else {
            continue;
        };
        if row >= grid.len() || column >= columns || !placed.insert(k) {
            continue;
        }
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
    let value = legal_pdf_core::roman_value(inner).filter(|_| !inner.contains(['d', 'm', 'D', 'M']))?;
    Some((if upper { 8 } else { 7 }, value))
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
    items: &PageItems,
    translated: &dyn Fn(&str) -> bool,
    layout: &dyn TableLayout,
) -> (Vec<PdfTable>, Vec<String>) {
    let offered = items;
    let PageItems { items, origin } = items;
    if items.len() < 6 {
        return (Vec::new(), Vec::new());
    }
    let number = page.index as u32 + 1;
    let mut found: Vec<Table> = Vec::new();
    let mut claimed: HashSet<usize> = HashSet::new();
    // A table read on some of the items: its item indices taken back to the page's.
    let on_page = |mut table: Table, subset: &[usize]| {
        table.item_indices = table.item_indices.iter().filter_map(|&k| subset.get(k).copied()).collect();
        table.item_cells = table
            .item_cells
            .iter()
            .filter_map(|&(k, row, column)| subset.get(k).map(|&k| (k, row, column)))
            .collect();
        table
    };
    // Filled cell grids first, each on items no grid before it took.
    let (grids, hints) = detect_tables_from_rects(items, &rects(page), number);
    for table in grids {
        if table.item_indices.iter().any(|k| claimed.contains(k)) {
            continue;
        }
        claimed.extend(table.item_indices.iter().copied());
        found.push(table);
    }
    // Ruled tables where no grid was found.
    let rules = rules(page);
    if claimed.is_empty() && !rules.is_empty() {
        for table in detect_tables_from_lines(items, &rules, number) {
            claimed.extend(table.item_indices.iter().copied());
            found.push(table);
        }
    }
    // Neither: cell rects too sparse for a grid still set the columns of the words among them.
    if claimed.is_empty() {
        const PADDING: f32 = 15.0;
        for hint in hints.iter().filter(|hint| !hint.cluster_rects.is_empty()) {
            let inside: Vec<usize> = (0..items.len())
                .filter(|&k| {
                    let item = &items[k];
                    item.y >= hint.y_bottom - PADDING
                        && item.y <= hint.y_top + PADDING
                        && item.x >= hint.x_left - PADDING
                        && item.x <= hint.x_right + PADDING
                })
                .collect();
            let inside_items: Vec<TextItem> = inside.iter().map(|&k| items[k].clone()).collect();
            if let Some(table) = try_build_rect_guided_table(&inside_items, &hint.cluster_rects) {
                claimed.extend(inside.iter().copied());
                found.push(on_page(table, &inside[..]));
            }
        }
    }
    if layout.white_space(page, offered) {
        let rest: Vec<usize> = (0..items.len()).filter(|k| !claimed.contains(k)).collect();
        let free: Vec<TextItem> = rest.iter().map(|&k| items[k].clone()).collect();
        let mut sizes: Vec<f32> = free.iter().map(|item| item.font_size).collect();
        sizes.sort_by(f32::total_cmp);
        let body = sizes.get(sizes.len() / 2).copied().unwrap_or(10.0);
        for table in detect_tables(&free, body, false) {
            found.push(on_page(table, &rest[..]));
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
        .filter_map(|table| converted(page, origin, table))
        .filter(|table| !table.line_ids().into_iter().any(|id| translated(id)))
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
/// in reading order; and the lines of the contents lists pdf-inspector read as tables. Each page
/// is read on its items (`page_items`, one per page) and its own rules and filled rectangles. A
/// table holding a line `translated` names (by id) is a text beside its translation.
pub fn document_tables(
    pages: &[Page],
    items: &[PageItems],
    translated: &dyn Fn(&str) -> bool,
    layout: &dyn TableLayout,
) -> (Vec<PdfTable>, HashSet<String>) {
    let mut tables: Vec<PdfTable> = Vec::new();
    let mut contents = HashSet::new();
    let mut previous_last: Option<usize> = None;
    for (page, items) in pages.iter().zip(items) {
        let (found, listed) = page_tables(page, items, translated, layout);
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
