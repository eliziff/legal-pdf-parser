//! Tables, read by pdf-inspector's own detectors.
//!
//! Each page's words go to pdf-inspector as its text items, the page's filled rectangles as its
//! rects and its painted rules as its lines, and its detectors read them in its own order: filled
//! cell grids first, then ruled tables where no grid was found, then white space for the rest of
//! the page where its lines stand in a plain grid (`grid::white_space`). Each word
//! goes in the cell the detector placed it in. A contents list is no table, nor a
//! text set beside its translation, nor a list set in two columns (its markers counting down the
//! first). A table the page cuts off goes on at the top
//! of the next page, or under its repeated caption and head row, when its columns stand where
//! they stood.

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;

use crate::model::tables::{CellPart, PdfTable, TableCell, TableRow};
use crate::rules::layout::language::{column_language, language_votes};
use legal_pdf_core::model::{Line, Page};
use pdf_inspector::tables::{
    detect_tables, detect_tables_from_lines, detect_tables_from_rects, try_build_rect_guided_table,
    Table, TableKind,
};
use pdf_inspector::types::TextItem;
use std::collections::{HashMap, HashSet};

use crate::rules::tables::grid::white_space;

/// Points within which rule positions coincide and rule ends meet.
const TOLERANCE: f64 = 2.0;








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
    let (mut band, mut bottom) = (0, f64::NEG_INFINITY);
    let mut keyed: Vec<(usize, Piece)> = Vec::with_capacity(pieces.len());
    for piece in pieces {
        band += usize::from(piece.bbox[1] > bottom - (piece.bbox[3] - piece.bbox[1]) * 0.5);
        bottom = bottom.max(piece.bbox[3]);
        keyed.push((band, piece));
    }
    keyed.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.bbox[0].total_cmp(&right.1.bbox[0]))
    });
    let part = |piece: Piece| {
        let line = &page.lines[piece.line];
        let text = piece.chars.map_or_else(
            || line.text.clone(),
            |(start, end)| line.text.chars().skip(start).take(end - start).collect(),
        );
        // The words' characters (as the extraction places a word) stand in the line's UTF-16 code units, as the
        // structure places text.
        let units = |scalar: usize| {
            line.text
                .chars()
                .take(scalar)
                .map(char::len_utf16)
                .sum::<usize>()
        };
        let chars = piece.chars.map(|(start, end)| (units(start), units(end)));
        CellPart {
            line_id: line.id.clone(),
            chars,
            text,
            bbox: piece.bbox,
        }
    };
    TableCell {
        column,
        row_span: 1,
        column_span: 1,
        parts: keyed.into_iter().map(|(_, piece)| part(piece)).collect(),
    }
}

pub use crate::model::WHOLE;

use crate::model::PageItems;
pub use crate::rules::tables::items::{clear_table_underlines, line_items, page_items};
use crate::rules::tables::items::{rects, rules};

/// One cell's words as pieces: the words of one line that run on together make one piece, a
/// line all of whose words stand in the cell is read whole.
fn pieces(page: &Page, mut words: Vec<(usize, usize)>) -> Vec<Piece> {
    words.sort_unstable();
    let mut runs: Vec<(usize, Vec<usize>)> = Vec::new();
    for (line, word) in words {
        match runs.last_mut() {
            Some((at, run))
                if *at == line
                    && word != WHOLE
                    && run
                        .last()
                        .is_some_and(|last| *last != WHOLE && *last + 1 == word) =>
            {
                run.push(word)
            }
            _ => runs.push((line, vec![word])),
        }
    }
    runs.into_iter()
        .map(|(slot, run)| {
            let line = &page.lines[slot];
            let all = line
                .words
                .iter()
                .filter(|word| !word.text.trim().is_empty())
                .count();
            if run[0] == WHOLE || run.len() == all {
                return Piece {
                    line: slot,
                    chars: None,
                    bbox: line.bbox,
                };
            }
            let (first, last) = (&line.words[run[0]], &line.words[*run.last().expect("run")]);
            let bbox = run
                .iter()
                .map(|&k| line.words[k].bbox)
                .fold(first.bbox, |a, b| {
                    [
                        a[0].min(b[0]),
                        a[1].min(b[1]),
                        a[2].max(b[2]),
                        a[3].max(b[3]),
                    ]
                });
            Piece {
                line: slot,
                chars: Some((first.start, last.end)),
                bbox,
            }
        })
        .collect()
}

/// A table pdf-inspector read, on the words it was read from: each word in the cell the detector
/// placed it in.
fn converted(page: &Page, origin: &[(usize, usize)], table: &Table) -> Option<PdfTable> {
    let columns = table.cells.iter().map(Vec::len).max().unwrap_or(0);
    let mut grid: Vec<Vec<Vec<(usize, usize)>>> =
        vec![vec![Vec::new(); columns]; table.cells.len()];
    let mut boxes: Vec<[f64; 4]> = Vec::new();
    let mut placed = HashSet::new();
    for &(k, row, column) in &table.item_cells {
        let Some(&(line, word)) = origin.get(k) else {
            continue;
        };
        if row < grid.len() && column < columns && placed.insert(k) {
            grid[row][column].push((line, word));
            boxes.push(if word == WHOLE {
                page.lines[line].bbox
            } else {
                page.lines[line].words[word].bbox
            });
        }
    }
    let row = |row: Vec<Vec<(usize, usize)>>| TableRow {
        header: false,
        cells: row
            .into_iter()
            .enumerate()
            .map(|(column, words)| cell_of(page, column, pieces(page, words)))
            .collect(),
    };
    let rows: Vec<TableRow> = grid
        .into_iter()
        .filter(|row| row.iter().any(|cell| !cell.is_empty()))
        .map(row)
        .collect();
    if rows.len() < 2 || columns == 0 {
        return None;
    }
    let low = |edge: usize| {
        boxes
            .iter()
            .map(|bbox| bbox[edge])
            .fold(f64::INFINITY, f64::min)
    };
    let high = |edge: usize| {
        boxes
            .iter()
            .map(|bbox| bbox[edge])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let mut marks: Vec<f64> = table.columns.iter().map(|x| f64::from(*x)).collect();
    marks.sort_by(f64::total_cmp);
    // Column boundaries: the detector's own, or midway between its column positions.
    let xs = if marks.len() == columns + 1 {
        marks
    } else {
        [low(0)]
            .into_iter()
            .chain(marks.windows(2).map(|pair| (pair[0] + pair[1]) / 2.0))
            .chain([high(2)])
            .collect()
    };
    Some(PdfTable {
        rows,
        page_index: page.index,
        xs,
        top: low(1),
        bottom: high(3),
    })
}

/// Whether one column of a table reads in English and another in French: a text set beside
/// its translation under shared heads, not a table.
fn bilingual_columns(page: &Page, table: &PdfTable) -> bool {
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
        .map(|lines| column_language(language_votes(lines.into_iter()), 5))
        .collect();
    languages.contains(&Some(true)) && languages.contains(&Some(false))
}

/// Where a marker cell stands in its count ("a)", "(iv)", "12.", "3.05", "•"): the count's family
/// and the marker's value in it. A marker keeps a bracket or a stop (a bare figure is a table's),
/// and OCR's "}" for ")" reads as ")".
fn list_marker(text: &str) -> Option<(u8, u32)> {
    let text = text.trim();
    if text.chars().count() == 1 && "•·▪◦●○■□➢-–—*\u{f0b7}\u{f0a7}\u{f0d8}".contains(text)
    {
        return Some((0, 0));
    }
    let inner = text
        .trim_start_matches(['(', '['])
        .trim_end_matches([')', ']', '}', '.']);
    if inner.is_empty() || inner.len() == text.len() || inner.chars().count() > 6 {
        return None;
    }
    let parts: Vec<&str> = inner.split('.').collect();
    if parts
        .iter()
        .all(|part| !part.is_empty() && part.len() <= 3 && part.bytes().all(|b| b.is_ascii_digit()))
    {
        return Some((parts.len() as u8, parts.last()?.parse().ok()?));
    }
    let upper = inner.bytes().all(|b| b.is_ascii_uppercase());
    if inner.len() == 1 && inner.bytes().all(|b| b.is_ascii_alphabetic()) {
        let letter = inner.bytes().next()?.to_ascii_lowercase();
        return Some((if upper { 6 } else { 5 }, u32::from(letter - b'a') + 1));
    }
    let value =
        legal_pdf_core::roman_value(inner).filter(|_| !inner.contains(['d', 'm', 'D', 'M']))?;
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
        let text = |column: usize| {
            row.cells
                .iter()
                .filter(|cell| cell.column == column)
                .map(TableCell::text)
                .collect::<Vec<_>>()
                .join(" ")
        };
        let (left, right) = (text(0), text(1));
        if left.trim().is_empty() || right.trim().is_empty() {
            continue;
        }
        match list_marker(&left) {
            Some(marker) => markers.push(marker),
            None => return false,
        }
    }
    let follows = markers
        .windows(2)
        .filter(|pair| {
            let ((family, value), (next_family, next)) = (pair[0], pair[1]);
            family == next_family && (family == 0 || next == value + 1) || next == 1
        })
        .count();
    markers.len() >= 3 && follows * 3 >= (markers.len() - 1) * 2
}

/// A page's tables, and the lines of the contents lists pdf-inspector read there as tables.
fn page_tables(
    page: &Page,
    items: &PageItems,
    translated: &dyn Fn(&str) -> bool,
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
        table.item_indices = table
            .item_indices
            .iter()
            .filter_map(|&k| subset.get(k).copied())
            .collect();
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
            let near = |item: &TextItem| {
                item.y >= hint.y_bottom - PADDING
                    && item.y <= hint.y_top + PADDING
                    && item.x >= hint.x_left - PADDING
                    && item.x <= hint.x_right + PADDING
            };
            let inside: Vec<usize> = (0..items.len()).filter(|&k| near(&items[k])).collect();
            let inside_items: Vec<TextItem> = inside.iter().map(|&k| items[k].clone()).collect();
            if let Some(table) = try_build_rect_guided_table(&inside_items, &hint.cluster_rects) {
                claimed.extend(inside.iter().copied());
                found.push(on_page(table, &inside[..]));
            }
        }
    }
    if white_space(page, offered) {
        let rest: Vec<usize> = (0..items.len()).filter(|k| !claimed.contains(k)).collect();
        let free: Vec<TextItem> = rest.iter().map(|&k| items[k].clone()).collect();
        let mut sizes: Vec<f32> = free.iter().map(|item| item.font_size).collect();
        sizes.sort_by(f32::total_cmp);
        let body = sizes.get(sizes.len() / 2).copied().unwrap_or(10.0);
        for table in detect_tables(&free, body, false) {
            // Lines of running words set out by a wide indent are text, however they align ("[4]    This appeal concerns ...").
            let running = table
                .item_indices
                .iter()
                .filter(|&&k| free[k].text.split_whitespace().count() >= 8)
                .count();
            if table.kind != TableKind::Data || running < 2 {
                found.push(on_page(table, &rest[..]));
            }
        }
    }
    let toc = found
        .iter()
        .filter(|table| table.kind == TableKind::Toc)
        .flat_map(|table| table.item_indices.iter().filter_map(|&k| origin.get(k)));
    let contents = toc
        .map(|&(line, _)| page.lines[line].id.clone())
        .collect::<HashSet<_>>()
        .into_iter()
        .collect();
    let mut tables: Vec<PdfTable> = found
        .iter()
        .filter(|table| table.kind == TableKind::Data)
        .filter_map(|table| converted(page, origin, table))
        .filter(|table| !table.line_ids().into_iter().any(|id| translated(id)))
        .filter(|table| !bilingual_columns(page, table))
        .filter(|table| !two_column_list(table))
        .filter(|table| !runs_past_its_grid(page, table))
        .collect();
    tables.sort_by(|left, right| left.top.total_cmp(&right.top));
    (tables, contents)
}

/// A grid a page's rules make over running text: its lines run past the grid's edges (underlined words, a boxed quotation, a
/// heading's rule).
fn runs_past_its_grid(page: &Page, table: &PdfTable) -> bool {
    let (Some(&left), Some(&right)) = (table.xs.first(), table.xs.last()) else {
        return false;
    };
    let ids: HashSet<&String> = table.line_ids().into_iter().collect();
    let past = |line: &&Line| {
        let words = line
            .words
            .iter()
            .filter(|word| !word.text.trim().is_empty());
        words.clone().count() >= 8
            && words
                .into_iter()
                .any(|word| word.bbox[2] <= left || word.bbox[0] >= right)
    };
    page.lines
        .iter()
        .filter(|line| ids.contains(&line.id))
        .filter(past)
        .count()
        >= 3
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
        let right = nearest(old[(cell.column + cell.column_span).min(old.len() - 1)]);
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
) -> (Vec<PdfTable>, HashSet<String>) {
    let (mut tables, mut contents, mut previous_last) =
        (Vec::<PdfTable>::new(), HashSet::new(), None::<usize>);
    for (page, items) in pages.iter().zip(items) {
        let (found, listed) = page_tables(page, items, translated);
        contents.extend(listed);
        // The page's body lines between two heights.
        let between = |from: f64, to: f64| {
            page.lines.iter().filter(move |line| {
                !line.exclude_from_body
                    && !line.text.trim().is_empty()
                    && line.bbox[1] > from
                    && line.bbox[3] < to
            })
        };
        let empty_band = |from: f64, to: f64| between(from, to).next().is_none();
        let lines_above = |top: f64| between(page.height * 0.08, top).count();
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
            let columns = |table: &PdfTable, row: usize| {
                table.rows[row]
                    .cells
                    .iter()
                    .map(|cell| cell.column)
                    .collect::<Vec<_>>()
            };
            let repeated = (0..heads.min(table.rows.len()))
                .take_while(|&row| {
                    !prior.rows[row].cells.is_empty()
                        && columns(prior, row) == columns(&table, row)
                        && row_text(prior, row) == row_text(&table, row)
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
