//! Tables drawn with rules.
//!
//! A page's painted rules that cross one another form a lattice. Each rule position
//! bounds a grid row or column; a cell runs on across a side the rules leave open, so a
//! merged cell spans the rows or columns it covers. A grid holds a table only when its
//! cells are rectangles, every line of text inside it sits within one cell, and at least
//! two rows hold text in more than one cell: a ruled box around a passage, a stack of
//! single-column panels or a contents list is no table. A line the extractor ran across a
//! column rule is parted at its words. Rules that only head a table, a pair of equal rules
//! around its column labels, make one too: its rows are the lines set under those labels.
//! A table the page cuts off goes on at the top of the next page, or under its repeated
//! caption and head row, when its columns stand where they stood.

use legal_pdf_core::line_font_size;
use legal_pdf_core::model::{Line, Page};
use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

/// Points within which rule positions coincide and rule ends meet.
const TOLERANCE: f64 = 2.0;
/// Share of a cell side the rules must cover to close it.
const CLOSED: f64 = 0.5;

/// A line of a cell, or the part of it that stands in the cell when the line runs on
/// across a column rule into the next cell.
#[derive(Debug, Clone)]
pub(crate) struct CellPart {
    pub(crate) line_id: String,
    /// Character offsets of the part within its line's text.
    pub(crate) chars: Option<(usize, usize)>,
    text: String,
    bbox: [f64; 4],
}

#[derive(Debug, Clone)]
pub(crate) struct TableCell {
    pub(crate) column: usize,
    pub(crate) row_span: usize,
    pub(crate) column_span: usize,
    /// The cell's text in reading order; empty for a blank cell.
    pub(crate) parts: Vec<CellPart>,
}

impl TableCell {
    fn text(&self) -> String {
        self.parts
            .iter()
            .map(|part| part.text.trim())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// The cell's parts by visual row.
    fn bands(&self) -> Vec<Vec<&CellPart>> {
        let mut bands: Vec<Vec<&CellPart>> = Vec::new();
        let mut bottom = f64::NEG_INFINITY;
        for part in &self.parts {
            let height = part.bbox[3] - part.bbox[1];
            if part.bbox[1] > bottom - height * 0.5 {
                bands.push(Vec::new());
            }
            bottom = bottom.max(part.bbox[3]);
            bands.last_mut().expect("band").push(part);
        }
        bands
    }

    pub(crate) fn line_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = Vec::new();
        for part in &self.parts {
            if ids.last() != Some(&part.line_id) {
                ids.push(part.line_id.clone());
            }
        }
        ids
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TableRow {
    pub(crate) header: bool,
    /// The cells whose top-left corner is in this row, by column.
    pub(crate) cells: Vec<TableCell>,
}

#[derive(Debug, Clone)]
pub(crate) struct PdfTable {
    pub(crate) rows: Vec<TableRow>,
    page_index: usize,
    xs: Vec<f64>,
    top: f64,
    bottom: f64,
}

impl PdfTable {
    /// The table's lines, each once, in reading order.
    pub(crate) fn line_ids(&self) -> Vec<&String> {
        let mut seen = HashSet::new();
        self.rows
            .iter()
            .flat_map(|row| &row.cells)
            .flat_map(|cell| &cell.parts)
            .map(|part| &part.line_id)
            .filter(|id| seen.insert(*id))
            .collect()
    }
}

#[derive(Clone, Copy)]
struct Rule {
    at: f64,
    start: f64,
    end: f64,
}

fn join(mut rules: Vec<Rule>) -> Vec<Rule> {
    rules.sort_by(|left, right| left.at.total_cmp(&right.at));
    let mut groups: Vec<(Vec<f64>, Vec<(f64, f64)>)> = Vec::new();
    for rule in rules {
        match groups.last_mut() {
            Some((ats, spans)) if rule.at - ats[ats.len() - 1] <= TOLERANCE => {
                ats.push(rule.at);
                spans.push((rule.start, rule.end));
            }
            _ => groups.push((vec![rule.at], vec![(rule.start, rule.end)])),
        }
    }
    let mut joined = Vec::new();
    for (ats, mut spans) in groups {
        let at = ats.iter().sum::<f64>() / ats.len() as f64;
        spans.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut current: Option<(f64, f64)> = None;
        for (start, end) in spans {
            current = match current {
                Some((from, to)) if start <= to + TOLERANCE => Some((from, to.max(end))),
                Some((from, to)) => {
                    joined.push(Rule {
                        at,
                        start: from,
                        end: to,
                    });
                    Some((start, end))
                }
                None => Some((start, end)),
            };
        }
        if let Some((start, end)) = current {
            joined.push(Rule { at, start, end });
        }
    }
    joined.retain(|rule| rule.end - rule.start > 2.0 * TOLERANCE);
    joined
}

/// Rule positions that coincide, averaged.
fn positions(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(f64::total_cmp);
    let mut clusters: Vec<Vec<f64>> = Vec::new();
    for value in values {
        match clusters.last_mut() {
            Some(cluster) if value - cluster[cluster.len() - 1] <= TOLERANCE => {
                cluster.push(value);
            }
            _ => clusters.push(vec![value]),
        }
    }
    clusters
        .iter()
        .map(|cluster| cluster.iter().sum::<f64>() / cluster.len() as f64)
        .collect()
}

/// Share of `start..end` that the rules lying at `at` cover.
fn coverage(rules: &[Rule], at: f64, start: f64, end: f64) -> f64 {
    let mut spans: Vec<_> = rules
        .iter()
        .filter(|rule| {
            (rule.at - at).abs() <= TOLERANCE
                && rule.end > start - TOLERANCE
                && rule.start < end + TOLERANCE
        })
        .map(|rule| (rule.start.max(start), rule.end.min(end)))
        .collect();
    spans.sort_by(|left, right| left.0.total_cmp(&right.0));
    let (mut covered, mut cursor) = (0.0, start);
    for (from, to) in spans {
        let from = from.max(cursor);
        if to > from {
            covered += to - from;
            cursor = to;
        }
    }
    covered / (end - start).max(f64::EPSILON)
}

fn find(parents: &mut [usize], mut node: usize) -> usize {
    while parents[node] != node {
        parents[node] = parents[parents[node]];
        node = parents[node];
    }
    node
}

fn union(parents: &mut [usize], left: usize, right: usize) {
    let (left, right) = (find(parents, left), find(parents, right));
    parents[left.max(right)] = left.min(right);
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

fn contents_leader(text: &str) -> bool {
    static LEADER: OnceLock<Regex> = OnceLock::new();
    LEADER
        .get_or_init(|| Regex::new(r"(?:\. ?){4,}\s*\S{1,8}\s*$").expect("contents leader regex"))
        .is_match(text)
}

/// A line, or the words of it that stand in one cell.
#[derive(Clone, Copy)]
struct Piece {
    line: usize,
    chars: Option<(usize, usize)>,
    bbox: [f64; 4],
}

/// A ruled row whose cells each hold the same number of lines, level with one another
/// from cell to cell and set apart by more than a wrapped line's leading, is that many
/// rows the rules leave unparted.
fn part_unruled_rows(rows: Vec<TableRow>) -> Vec<TableRow> {
    let mut spanned = vec![false; rows.len()];
    for (index, row) in rows.iter().enumerate() {
        for cell in &row.cells {
            for covered in index + 1..(index + cell.row_span).min(rows.len()) {
                spanned[covered] = true;
            }
        }
    }
    let mut parted = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let bands: Vec<Vec<Vec<&CellPart>>> = row.cells.iter().map(TableCell::bands).collect();
        let filled: Vec<&Vec<Vec<&CellPart>>> =
            bands.iter().filter(|bands| !bands.is_empty()).collect();
        let count = filled.first().map_or(0, |bands| bands.len());
        let height = |band: &[&CellPart]| {
            band.iter()
                .map(|part| part.bbox[3] - part.bbox[1])
                .fold(0.0, f64::max)
        };
        let top = |band: &[&CellPart]| {
            band.iter()
                .map(|part| part.bbox[1])
                .fold(f64::INFINITY, f64::min)
        };
        let bottom = |band: &[&CellPart]| {
            band.iter()
                .map(|part| part.bbox[3])
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let level = || {
            (0..count).all(|band| {
                let first = top(&filled[0][band]);
                filled
                    .iter()
                    .all(|bands| (top(&bands[band]) - first).abs() <= 0.5 * height(&bands[band]))
            })
        };
        let apart = || {
            filled.iter().all(|bands| {
                bands
                    .windows(2)
                    .all(|pair| top(&pair[1]) - bottom(&pair[0]) >= 0.5 * height(&pair[0]))
            })
        };
        if spanned[index]
            || count < 2
            || filled.len() < 2
            || row.cells.iter().any(|cell| cell.row_span > 1)
            || !filled.iter().all(|bands| bands.len() == count)
            || !level()
            || !apart()
        {
            parted.push(row);
            continue;
        }
        for band in 0..count {
            parted.push(TableRow {
                header: false,
                cells: row
                    .cells
                    .iter()
                    .zip(&bands)
                    .map(|(cell, bands)| TableCell {
                        column: cell.column,
                        row_span: 1,
                        column_span: cell.column_span,
                        parts: bands
                            .get(band)
                            .map(|parts| parts.iter().map(|part| (*part).clone()).collect())
                            .unwrap_or_default(),
                    })
                    .collect(),
            });
        }
    }
    parted
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

/// Pieces of a line in the lattice cells their words stand in, or `None` when the line
/// crosses a rule its words do not part at.
fn pieces_of(
    line: &Line,
    index: usize,
    cell_at: &dyn Fn(f64, f64) -> Option<usize>,
    inside: &dyn Fn(usize, [f64; 4]) -> bool,
) -> Option<Vec<(usize, Piece)>> {
    let center = |bbox: [f64; 4]| ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0);
    let whole = |root: usize| {
        inside(root, line.bbox).then(|| {
            vec![(
                root,
                Piece {
                    line: index,
                    chars: None,
                    bbox: line.bbox,
                },
            )]
        })
    };
    let (x, y) = center(line.bbox);
    let root = cell_at(x, y)?;
    let words: Vec<_> = line
        .words
        .iter()
        .filter(|word| !word.text.trim().is_empty())
        .collect();
    let roots: Vec<Option<usize>> = words
        .iter()
        .map(|word| {
            let (x, y) = center(word.bbox);
            cell_at(x, y)
        })
        .collect();
    if words.is_empty() || roots.iter().all(|value| *value == Some(root)) {
        return whole(root);
    }
    let mut pieces: Vec<(usize, Piece)> = Vec::new();
    for (word, root) in words.iter().zip(roots) {
        let root = root?;
        match pieces.last_mut() {
            Some((prior, piece)) if *prior == root => {
                piece.chars = piece.chars.map(|(start, _)| (start, word.end));
                piece.bbox[2] = piece.bbox[2].max(word.bbox[2]);
                piece.bbox[1] = piece.bbox[1].min(word.bbox[1]);
                piece.bbox[3] = piece.bbox[3].max(word.bbox[3]);
            }
            _ => pieces.push((
                root,
                Piece {
                    line: index,
                    chars: Some((word.start, word.end)),
                    bbox: word.bbox,
                },
            )),
        }
    }
    let mut roots: Vec<usize> = pieces.iter().map(|(root, _)| *root).collect();
    roots.sort_unstable();
    roots.dedup();
    (roots.len() == pieces.len() && pieces.iter().all(|(root, piece)| inside(*root, piece.bbox)))
        .then_some(pieces)
}

/// The table one lattice of rules draws on a page, if it is one.
fn lattice_table(page: &Page, horizontal: &[Rule], vertical: &[Rule]) -> Option<PdfTable> {
    let xs = positions(vertical.iter().map(|rule| rule.at).collect());
    let ys = positions(horizontal.iter().map(|rule| rule.at).collect());
    if xs.len() < 3 || ys.len() < 3 {
        return None;
    }
    let (columns, rows) = (xs.len() - 1, ys.len() - 1);
    let slot = |row: usize, column: usize| row * columns + column;
    let mut parents: Vec<usize> = (0..rows * columns).collect();
    for row in 0..rows {
        for column in 0..columns {
            if column + 1 < columns
                && coverage(vertical, xs[column + 1], ys[row], ys[row + 1]) < CLOSED
            {
                union(&mut parents, slot(row, column), slot(row, column + 1));
            }
            if row + 1 < rows
                && coverage(horizontal, ys[row + 1], xs[column], xs[column + 1]) < CLOSED
            {
                union(&mut parents, slot(row, column), slot(row + 1, column));
            }
        }
    }
    // Each cell: its top-left slot, spans and the lines it holds.
    let mut extents: HashMap<usize, (usize, usize, usize, usize, usize)> = HashMap::new();
    for row in 0..rows {
        for column in 0..columns {
            let root = find(&mut parents, slot(row, column));
            let extent = extents.entry(root).or_insert((row, column, row, column, 0));
            extent.2 = extent.2.max(row);
            extent.3 = extent.3.max(column);
            extent.4 += 1;
        }
    }
    if extents
        .values()
        .any(|(top, left, bottom, right, count)| (bottom - top + 1) * (right - left + 1) != *count)
    {
        return None;
    }
    let roots: Vec<usize> = (0..rows * columns)
        .map(|slot| find(&mut parents, slot))
        .collect();
    let cell_at = |x: f64, y: f64| {
        (x > xs[0] && x < xs[columns] && y > ys[0] && y < ys[rows]).then(|| {
            let column = xs.partition_point(|value| *value < x) - 1;
            let row = ys.partition_point(|value| *value < y) - 1;
            roots[slot(row, column)]
        })
    };
    // Text that crosses a cell's rules belongs to no one cell.
    let inside = |root: usize, bbox: [f64; 4]| {
        let (top, left, bottom, right, _) = extents[&root];
        let width = bbox[2] - bbox[0];
        let height = bbox[3] - bbox[1];
        let outside_x = (xs[left] - bbox[0]).max(0.0) + (bbox[2] - xs[right + 1]).max(0.0);
        let outside_y = (ys[top] - bbox[1]).max(0.0) + (bbox[3] - ys[bottom + 1]).max(0.0);
        outside_x <= (width * 0.25).max(3.0) && outside_y <= (height * 0.5).max(3.0)
    };
    let mut members: HashMap<usize, Vec<Piece>> = HashMap::new();
    for (index, line) in page.lines.iter().enumerate() {
        if line.text.trim().is_empty() || !line.bbox.iter().all(|value| value.is_finite()) {
            continue;
        }
        let (x, y) = (
            (line.bbox[0] + line.bbox[2]) / 2.0,
            (line.bbox[1] + line.bbox[3]) / 2.0,
        );
        if cell_at(x, y).is_none() {
            continue;
        }
        for (root, piece) in pieces_of(line, index, &cell_at, &inside)? {
            members.entry(root).or_default().push(piece);
        }
    }
    let mut cells: Vec<(usize, usize, TableCell)> = extents
        .iter()
        .map(|(root, (top, left, bottom, right, _))| {
            let mut cell = cell_of(page, *left, members.remove(root).unwrap_or_default());
            cell.row_span = bottom - top + 1;
            cell.column_span = right - left + 1;
            (*top, *left, cell)
        })
        .collect();
    cells.sort_by_key(|(row, column, _)| (*row, *column));
    // A line parted between cells must run straight on from one cell into the next.
    let sequence: Vec<&CellPart> = cells.iter().flat_map(|(_, _, cell)| &cell.parts).collect();
    let mut runs: HashMap<&str, usize> = HashMap::new();
    for (position, part) in sequence.iter().enumerate() {
        if position == 0 || sequence[position - 1].line_id != part.line_id {
            *runs.entry(part.line_id.as_str()).or_default() += 1;
        }
    }
    if runs.values().any(|count| *count > 1) {
        return None;
    }
    let lines_in = |cell: &TableCell| cell.parts.len();
    let filled_rows = (0..rows)
        .filter(|row| {
            cells
                .iter()
                .filter(|(top, _, cell)| top == row && lines_in(cell) > 0)
                .count()
                >= 2
        })
        .count();
    if filled_rows < 2 {
        return None;
    }
    let by_id: HashMap<&str, &Line> = page
        .lines
        .iter()
        .map(|line| (line.id.as_str(), line))
        .collect();
    let text_rows: Vec<Vec<&Line>> = (0..rows)
        .map(|row| {
            cells
                .iter()
                .filter(|(top, _, _)| *top == row)
                .flat_map(|(_, _, cell)| cell.line_ids())
                .map(|id| by_id[id.as_str()])
                .collect()
        })
        .filter(|lines: &Vec<&Line>| !lines.is_empty())
        .collect();
    // A ruled contents list, or a box the page repeats as its running head or foot.
    let inside: Vec<Line> = text_rows
        .iter()
        .flatten()
        .map(|line| (*line).clone())
        .collect();
    if text_rows
        .iter()
        .filter(|lines| lines.iter().any(|line| contents_leader(&line.text)))
        .count()
        * 2
        >= text_rows.len()
        || crate::layout::contents_grid(&inside, page.width)
        || text_rows
            .iter()
            .flatten()
            .all(|line| line.exclude_from_body)
    {
        return None;
    }
    // Rows and columns in which no cell begins are covered by spanning cells: drop them.
    let row_starts: Vec<bool> = (0..rows)
        .map(|row| cells.iter().any(|(top, _, _)| *top == row))
        .collect();
    let column_starts: Vec<bool> = (0..columns)
        .map(|column| cells.iter().any(|(_, left, _)| *left == column))
        .collect();
    let renumber = |starts: &[bool], index: usize| starts[..index].iter().filter(|s| **s).count();
    let mut table_rows: Vec<TableRow> = (0..rows)
        .filter(|row| row_starts[*row])
        .map(|_| TableRow {
            header: false,
            cells: Vec::new(),
        })
        .collect();
    for (top, left, mut cell) in cells {
        let bottom = top + cell.row_span;
        let right = left + cell.column_span;
        cell.row_span = renumber(&row_starts, bottom) - renumber(&row_starts, top);
        cell.column_span = renumber(&column_starts, right) - renumber(&column_starts, left);
        cell.column = renumber(&column_starts, left);
        table_rows[renumber(&row_starts, top)].cells.push(cell);
    }
    let kept_xs: Vec<f64> = xs
        .iter()
        .enumerate()
        .filter(|(index, _)| *index == columns || column_starts[*index])
        .map(|(_, x)| *x)
        .collect();
    let mut table_rows = part_unruled_rows(table_rows);
    // Leading rows set wholly in bold over rows that are not head the table.
    let bold_row = |row: &TableRow| {
        let lines: Vec<&Line> = row
            .cells
            .iter()
            .flat_map(TableCell::line_ids)
            .map(|id| by_id[id.as_str()])
            .collect();
        !lines.is_empty() && lines.iter().all(|line| bold(line))
    };
    let leading = table_rows.iter().take_while(|row| bold_row(row)).count();
    if leading < table_rows.len() && table_rows[leading..].iter().any(|row| !bold_row(row)) {
        for row in &mut table_rows[..leading] {
            row.header = true;
        }
    }
    Some(PdfTable {
        rows: table_rows,
        page_index: page.index,
        xs: kept_xs,
        top: ys[0],
        bottom: ys[rows],
    })
}

fn whole_lines(page: &Page, column: usize, lines: Vec<usize>) -> TableCell {
    cell_of(
        page,
        column,
        lines
            .into_iter()
            .map(|line| Piece {
                line,
                chars: None,
                bbox: page.lines[line].bbox,
            })
            .collect(),
    )
}

/// Tables that rules only head: two rules of one width enclose a row of at least two
/// column labels, and the rows set below them in those columns, up to a rule of the same
/// width, a wider gap or a line that crosses a column, are the body. A title set between
/// rules heads no columns, and the rules a form leaves to be filled in enclose no labels.
fn open_tables(page: &Page, rules: &[Rule], taken: &HashSet<usize>) -> Vec<PdfTable> {
    let mut heights: Vec<f64> = page
        .lines
        .iter()
        .map(|line| line.bbox[3] - line.bbox[1])
        .filter(|height| *height > 0.0)
        .collect();
    if heights.is_empty() {
        return Vec::new();
    }
    heights.sort_by(f64::total_cmp);
    let pitch = heights[heights.len() / 2];
    let slack = 3.0 * TOLERANCE;
    let mut rules: Vec<Rule> = rules
        .iter()
        .copied()
        .filter(|rule| rule.end - rule.start >= 72.0)
        .collect();
    rules.sort_by(|left, right| left.at.total_cmp(&right.at));
    let same_width = |left: &Rule, right: &Rule| {
        (left.start - right.start).abs() <= slack && (left.end - right.end).abs() <= slack
    };
    let free: Vec<usize> = (0..page.lines.len())
        .filter(|index| !taken.contains(index) && !page.lines[*index].text.trim().is_empty())
        .collect();
    let mut tables = Vec::new();
    let mut after = f64::NEG_INFINITY;
    for (slot, top) in rules.iter().enumerate() {
        if top.at < after {
            continue;
        }
        let Some(bottom) = rules[slot + 1..].iter().find(|rule| {
            same_width(top, rule)
                && rule.at - top.at > 0.8 * pitch
                && rule.at - top.at <= 4.5 * pitch
        }) else {
            continue;
        };
        let within =
            |line: &Line| line.bbox[0] >= top.start - slack && line.bbox[2] <= top.end + slack;
        let mut labels: Vec<usize> = free
            .iter()
            .copied()
            .filter(|index| {
                let line = &page.lines[*index];
                let middle = (line.bbox[1] + line.bbox[3]) / 2.0;
                middle > top.at && middle < bottom.at && within(line)
            })
            .collect();
        labels.sort_by(|left, right| {
            page.lines[*left].bbox[0].total_cmp(&page.lines[*right].bbox[0])
        });
        // Column labels stacked over one another head one column.
        let mut columns: Vec<(f64, f64, Vec<usize>)> = Vec::new();
        for index in labels {
            let line = &page.lines[index];
            match columns.last_mut() {
                Some(column) if line.bbox[0] < column.1 => {
                    column.1 = column.1.max(line.bbox[2]);
                    column.2.push(index);
                }
                _ => columns.push((line.bbox[0], line.bbox[2], vec![index])),
            }
        }
        // Labels are words, set short; figures between rules are a row of data.
        let labelled = columns.iter().flat_map(|column| &column.2).all(|index| {
            let text = page.lines[*index].text.trim();
            text.chars().any(char::is_alphabetic) && text.chars().count() <= 60
        });
        if columns.len() < 2 || !labelled {
            continue;
        }
        let mut xs = vec![top.start];
        xs.extend(columns.windows(2).map(|pair| (pair[0].1 + pair[1].0) / 2.0));
        xs.push(top.end);
        // A body line stands under the one label it overlaps, inside that label's column.
        let column_of = |line: &Line| {
            let mut under = columns
                .iter()
                .enumerate()
                .filter(|(_, (left, right, _))| line.bbox[0] < *right && *left < line.bbox[2])
                .map(|(column, _)| column);
            let column = under.next()?;
            (under.next().is_none()
                && line.bbox[0] >= xs[column] - slack
                && line.bbox[2] <= xs[column + 1] + slack)
                .then_some(column)
        };
        let closing = rules[slot + 1..]
            .iter()
            .filter(|rule| rule.at > bottom.at && same_width(top, rule))
            .map(|rule| rule.at)
            .next()
            .unwrap_or(f64::INFINITY);
        let mut below: Vec<usize> = free
            .iter()
            .copied()
            .filter(|index| {
                let line = &page.lines[*index];
                line.bbox[1] > bottom.at - 1.0
                    && line.bbox[3] <= closing + 1.0
                    && line.bbox[2] > top.start
                    && line.bbox[0] < top.end
            })
            .collect();
        below.sort_by(|left, right| {
            page.lines[*left].bbox[1].total_cmp(&page.lines[*right].bbox[1])
        });
        // The body, band by band.
        let mut body: Vec<Vec<Vec<usize>>> = Vec::new();
        let mut floor = bottom.at;
        let mut position = 0;
        while position < below.len() {
            let first = &page.lines[below[position]];
            if first.bbox[1] - floor > 2.5 * pitch {
                break;
            }
            let mut band = vec![below[position]];
            let mut band_bottom = first.bbox[3];
            position += 1;
            while position < below.len()
                && page.lines[below[position]].bbox[1] < band_bottom - 0.5 * pitch
            {
                band_bottom = band_bottom.max(page.lines[below[position]].bbox[3]);
                band.push(below[position]);
                position += 1;
            }
            let mut cells = vec![Vec::new(); columns.len()];
            let mut fits = true;
            for index in &band {
                match column_of(&page.lines[*index]) {
                    Some(column) => cells[column].push(*index),
                    None => fits = false,
                }
            }
            if !fits {
                break;
            }
            // A band with nothing in the first column, close under a row, wraps that row.
            let wraps = cells[0].is_empty()
                && first.bbox[1] - floor <= 0.6 * pitch
                && body
                    .last()
                    .is_some_and(|row: &Vec<Vec<usize>>| !row[0].is_empty());
            match body.last_mut() {
                Some(row) if wraps => {
                    for (column, lines) in cells.into_iter().enumerate() {
                        row[column].extend(lines);
                    }
                }
                _ => body.push(cells),
            }
            floor = band_bottom;
        }
        let paired = body
            .iter()
            .any(|row| row.iter().filter(|cell| !cell.is_empty()).count() >= 2);
        if body.len() < 2 || (columns.len() < 3 && !paired) {
            continue;
        }
        let header = TableRow {
            header: true,
            cells: columns
                .iter()
                .enumerate()
                .map(|(column, (_, _, lines))| whole_lines(page, column, lines.clone()))
                .collect(),
        };
        let mut rows = vec![header];
        rows.extend(body.into_iter().map(|cells| {
            TableRow {
                header: false,
                cells: cells
                    .into_iter()
                    .enumerate()
                    .map(|(column, lines)| whole_lines(page, column, lines))
                    .collect(),
            }
        }));
        after = floor.max(bottom.at);
        tables.push(PdfTable {
            rows,
            page_index: page.index,
            xs,
            top: top.at,
            bottom: if closing.is_finite() { closing } else { floor },
        });
    }
    tables
}

/// Party roles a caption sets beside the names it lists.
fn party_role(text: &str) -> bool {
    static ROLE: OnceLock<Regex> = OnceLock::new();
    ROLE.get_or_init(|| {
        Regex::new(
            r"(?i)\b(?:applicants?|respondents?|plaintiffs?|defendants?|appellants?|petitioners?|claimants?|interveners?|intervenors?|interested\s+part(?:y|ies)|amic(?:us|i)\s+curiae|accused|complainants?|third\s+part(?:y|ies)|requérante?s?|intimée?s?|demandeur|demanderesse|défendeur|défenderesse|appelante?s?)\b",
        )
        .expect("party role regex")
    })
    .is_match(text)
}

/// A list marker or a line number standing alone: "3.", "(b)", "[12]", "iv)", "17".
fn bare_marker(text: &str) -> bool {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    MARKER
        .get_or_init(|| {
            Regex::new(r"^[\[(]?(?:\d{1,4}|[a-zA-Z]|[ivxlcdmIVXLCDM]{1,6})(?:\.\d{1,3})*[.)\]]?$")
                .expect("list marker regex")
        })
        .is_match(text.trim())
}

/// A figure: an amount, a count, a percentage, a year, a range of them.
fn figure(text: &str) -> bool {
    let text = text.trim();
    let digits = text.chars().filter(char::is_ascii_digit).count();
    digits > 0
        && digits * 2 >= text.chars().filter(|c| !c.is_whitespace()).count()
        && text.chars().filter(|c| c.is_alphabetic()).count() <= 1
}

/// A form's label, set before or above its value: "Lawyer:", "Date: 4 May".
fn form_label(text: &str) -> bool {
    static LABEL: OnceLock<Regex> = OnceLock::new();
    text.trim().ends_with(':')
        || LABEL
            .get_or_init(|| {
                Regex::new(r"^\p{L}[\p{L} .'/&-]{0,25}:(?:\s|$)").expect("form label regex")
            })
            .is_match(text.trim())
}

/// A letterhead's or an address for service's contact details.
fn contact_detail(text: &str) -> bool {
    static CONTACT: OnceLock<Regex> = OnceLock::new();
    CONTACT
        .get_or_init(|| {
            Regex::new(
                r"(?i)\b(?:tel|telephone|phone|fax|e-?mail|t[ée]l[ée]copieur|courriel)\b|@|www\.|\bLLP\b",
            )
            .expect("contact regex")
        })
        .is_match(text)
}

/// Tables set without rules: rows of text whose columns stand apart by gutters of white
/// space that run the table's full height, each column's cells aligned on a left, right
/// or centre edge. Columns of prose side by side are a page's or a bilingual text's
/// columns, not a table; parties beside their roles are a caption; labels ending in a
/// colon over their values are a form; numbered lines and items beside their text are a
/// transcript or a list unless a head row names the columns; a contents list is no table.
/// What stays has three or more columns, a column of figures or a head row.
fn aligned_tables(
    page: &Page,
    taken: &HashSet<usize>,
    translation: &HashSet<String>,
) -> Vec<PdfTable> {
    let mut heights: Vec<f64> = page
        .lines
        .iter()
        .filter(|line| !line.exclude_from_body && !line.text.trim().is_empty())
        .map(|line| line.bbox[3] - line.bbox[1])
        .filter(|height| *height > 0.0)
        .collect();
    // A recognized page's lines are no measure of its white space.
    if heights.len() < 6 || page.source == "ocr" {
        return Vec::new();
    }
    heights.sort_by(f64::total_cmp);
    let pitch = heights[heights.len() / 2];
    let gutter = (0.6 * pitch).max(6.0);
    // Lines, parted where their words leave a column's gap.
    let mut pieces: Vec<Piece> = Vec::new();
    for (index, line) in page.lines.iter().enumerate() {
        // A watermark's large or turned letters stand over the page's columns; a signing
        // stamp's tiny print stands beside the signature.
        if taken.contains(&index)
            || line.exclude_from_body
            || line.text.trim().is_empty()
            || !line.bbox.iter().all(|value| value.is_finite())
            || line.bbox[3] - line.bbox[1] > 2.5 * pitch
            || line.bbox[3] - line.bbox[1] < 0.5 * pitch
        {
            continue;
        }
        let size = line_font_size(line)
            .max(line.bbox[3] - line.bbox[1])
            .max(1.0);
        let words: Vec<_> = line
            .words
            .iter()
            .filter(|word| !word.text.trim().is_empty())
            .collect();
        let mut start = 0;
        for split in 1..=words.len() {
            if split < words.len() && words[split].bbox[0] - words[split - 1].bbox[2] < 1.2 * size {
                continue;
            }
            if start == 0 && split == words.len() {
                break;
            }
            let part = &words[start..split];
            pieces.push(Piece {
                line: index,
                chars: Some((part[0].start, part[part.len() - 1].end)),
                bbox: [
                    part[0].bbox[0],
                    part.iter()
                        .map(|word| word.bbox[1])
                        .fold(f64::INFINITY, f64::min),
                    part[part.len() - 1].bbox[2],
                    part.iter()
                        .map(|word| word.bbox[3])
                        .fold(f64::NEG_INFINITY, f64::max),
                ],
            });
            start = split;
        }
        if start == 0 {
            pieces.push(Piece {
                line: index,
                chars: None,
                bbox: line.bbox,
            });
        }
    }
    pieces.sort_by(|left, right| left.bbox[1].total_cmp(&right.bbox[1]));
    // Visual rows.
    let mut bands: Vec<Vec<Piece>> = Vec::new();
    let mut floor = f64::NEG_INFINITY;
    for piece in pieces {
        let height = piece.bbox[3] - piece.bbox[1];
        if piece.bbox[1] > floor - 0.5 * height {
            bands.push(Vec::new());
            floor = piece.bbox[3];
        }
        floor = floor.max(piece.bbox[3]);
        bands.last_mut().expect("band").push(piece);
    }
    let coverage = |band: &[Piece]| {
        let mut spans: Vec<(f64, f64)> = band
            .iter()
            .map(|piece| (piece.bbox[0], piece.bbox[2]))
            .collect();
        spans.sort_by(|left, right| left.0.total_cmp(&right.0));
        let mut merged: Vec<(f64, f64)> = Vec::new();
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end)),
            }
        }
        merged
    };
    let top = |band: &[Piece]| {
        band.iter()
            .map(|piece| piece.bbox[1])
            .fold(f64::INFINITY, f64::min)
    };
    let bottom = |band: &[Piece]| {
        band.iter()
            .map(|piece| piece.bbox[3])
            .fold(f64::NEG_INFINITY, f64::max)
    };
    let mut tables = Vec::new();
    let mut start = 0;
    while start < bands.len() {
        let spans = coverage(&bands[start]);
        let mut gutters: Vec<(f64, f64)> = spans
            .windows(2)
            .map(|pair| (pair[0].1, pair[1].0))
            .filter(|(left, right)| right - left >= gutter)
            .collect();
        if gutters.is_empty() {
            start += 1;
            continue;
        }
        let mut extent = (spans[0].0, spans[spans.len() - 1].1);
        let mut end = start + 1;
        while end < bands.len() && top(&bands[end]) - bottom(&bands[end - 1]) <= 2.5 * pitch {
            let spans = coverage(&bands[end]);
            let narrowed: Vec<(f64, f64)> = gutters
                .iter()
                .flat_map(|(left, right)| {
                    let mut open = vec![(*left, *right)];
                    for (start, end) in &spans {
                        open = open
                            .into_iter()
                            .flat_map(|(left, right)| {
                                if *end <= left || *start >= right {
                                    vec![(left, right)]
                                } else {
                                    vec![(left, *start), (*end, right)]
                                }
                            })
                            .collect();
                    }
                    open
                })
                .filter(|(left, right)| right - left >= gutter)
                .collect();
            if narrowed.is_empty() {
                break;
            }
            gutters = narrowed;
            extent = (
                extent.0.min(spans[0].0),
                extent.1.max(spans[spans.len() - 1].1),
            );
            end += 1;
        }
        let found =
            aligned_table(page, &bands[start..end], &gutters, extent, pitch).filter(|table| {
                !table
                    .line_ids()
                    .into_iter()
                    .any(|id| translation.contains(id))
            });
        start = if found.is_some() { end } else { start + 1 };
        tables.extend(found);
    }
    tables
}

/// The table some visual rows make between the gutters they share, if it passes.
fn aligned_table(
    page: &Page,
    bands: &[Vec<Piece>],
    gutters: &[(f64, f64)],
    extent: (f64, f64),
    pitch: f64,
) -> Option<PdfTable> {
    if bands.len() < 3 {
        return None;
    }
    let mut xs = vec![extent.0];
    xs.extend(gutters.iter().map(|(left, right)| (left + right) / 2.0));
    xs.push(extent.1);
    let columns = xs.len() - 1;
    let column_of = |piece: &Piece| {
        let middle = (piece.bbox[0] + piece.bbox[2]) / 2.0;
        xs.partition_point(|x| *x < middle).clamp(1, columns) - 1
    };
    // Rows: a band with nothing in the first column, close under a row, wraps that row.
    let mut rows: Vec<(Vec<Vec<Piece>>, f64)> = Vec::new();
    for band in bands {
        let mut cells = vec![Vec::new(); columns];
        for piece in band {
            cells[column_of(piece)].push(*piece);
        }
        let band_top = band
            .iter()
            .map(|piece| piece.bbox[1])
            .fold(f64::INFINITY, f64::min);
        let band_bottom = band
            .iter()
            .map(|piece| piece.bbox[3])
            .fold(f64::NEG_INFINITY, f64::max);
        match rows.last_mut() {
            Some((row, floor)) if cells[0].is_empty() && band_top - *floor <= 0.6 * pitch => {
                for (column, pieces) in cells.into_iter().enumerate() {
                    row[column].extend(pieces);
                }
                *floor = band_bottom;
            }
            _ => rows.push((cells, band_bottom)),
        }
    }
    let filled = |row: &Vec<Vec<Piece>>| row.iter().filter(|cell| !cell.is_empty()).count();
    while rows.first().is_some_and(|(row, _)| filled(row) < 2) {
        rows.remove(0);
    }
    while rows.last().is_some_and(|(row, _)| filled(row) < 2) {
        rows.pop();
    }
    let paired = rows.iter().filter(|(row, _)| filled(row) >= 2).count();
    if paired < 3 || paired * 2 < rows.len() {
        return None;
    }
    let text = |pieces: &[Piece]| {
        pieces
            .iter()
            .map(|piece| {
                let line = &page.lines[piece.line].text;
                match piece.chars {
                    Some((start, end)) => line.chars().skip(start).take(end - start).collect(),
                    None => line.clone(),
                }
            })
            .collect::<Vec<String>>()
            .join(" ")
    };
    let column_texts: Vec<Vec<String>> = (0..columns)
        .map(|column| {
            rows.iter()
                .filter(|(row, _)| !row[column].is_empty())
                .map(|(row, _)| text(&row[column]).trim().to_owned())
                .collect()
        })
        .collect();
    let share = |texts: &[String], test: &dyn Fn(&str) -> bool| {
        texts.iter().filter(|text| test(text)).count() as f64 / texts.len().max(1) as f64
    };
    // Each column's cells keep one edge or their centre in line.
    let aligned = (0..columns).all(|column| {
        let firsts: Vec<[f64; 4]> = rows
            .iter()
            .filter_map(|(row, _)| row[column].first().map(|piece| piece.bbox))
            .collect();
        if firsts.len() < 2 {
            return true;
        }
        let median = |mut values: Vec<f64>| {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let left = median(firsts.iter().map(|bbox| bbox[0]).collect());
        let right = median(firsts.iter().map(|bbox| bbox[2]).collect());
        let centre = median(
            firsts
                .iter()
                .map(|bbox| (bbox[0] + bbox[2]) / 2.0)
                .collect(),
        );
        let kept = firsts
            .iter()
            .filter(|bbox| {
                (bbox[0] - left).abs() <= 3.0
                    || (bbox[2] - right).abs() <= 3.0
                    || ((bbox[0] + bbox[2]) / 2.0 - centre).abs() <= 3.0
            })
            .count();
        kept * 10 >= firsts.len() * 7
    });
    let median_length = |texts: &[String]| {
        let mut lengths: Vec<usize> = texts.iter().map(|text| text.chars().count()).collect();
        lengths.sort_unstable();
        lengths.get(lengths.len() / 2).copied().unwrap_or(0)
    };
    let long = column_texts
        .iter()
        .filter(|texts| median_length(texts) > 30)
        .count();
    let lines: Vec<Line> = rows
        .iter()
        .flat_map(|(row, _)| row.iter().flatten())
        .map(|piece| piece.line)
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .map(|index| page.lines[index].clone())
        .collect();
    let leaders = lines
        .iter()
        .filter(|line| contents_leader(&line.text))
        .count();
    let caption = column_texts
        .iter()
        .any(|texts| share(texts, &party_role) >= 0.4)
        || column_texts
            .iter()
            .any(|texts| share(texts, &|text| text.chars().all(|c| c == ')' || c == ':')) >= 0.5);
    let labels = column_texts
        .iter()
        .any(|texts| share(texts, &form_label) >= 0.5);
    let cells: Vec<String> = column_texts.concat();
    // Bullets or boxes to tick down a column mark a list or a form.
    // Columns that each repeat one word down every row are a form's choices: Yes, No.
    let choices = column_texts
        .iter()
        .filter(|texts| texts.len() >= 3 && texts.iter().all(|text| *text == texts[0]))
        .count()
        >= 2;
    let ticked = column_texts.iter().any(|texts| {
        share(texts, &|text| {
            text.chars().count() == 1 && text.chars().all(|c| "\u{2022}\u{25e6}\u{25aa}\u{25ab}\u{25a0}\u{25a1}\u{25cf}\u{25cb}\u{f0b7}\u{f0a7}-\u{2013}*\u{b7}".contains(c))
        }) >= 0.7
    });
    // A column whose rows carry on one another's sentences, each line set out to the
    // column's measure, is running text set beside other running text: panels side by
    // side, or a text and its translation.
    let flowing = (0..columns).any(|column| {
        let measure = rows
            .iter()
            .flat_map(|(row, _)| &row[column])
            .map(|piece| piece.bbox[2])
            .fold(f64::NEG_INFINITY, f64::max);
        let start = rows
            .iter()
            .flat_map(|(row, _)| &row[column])
            .map(|piece| piece.bbox[0])
            .fold(f64::INFINITY, f64::min);
        let pairs: Vec<(&Vec<Piece>, &Vec<Piece>)> = rows
            .windows(2)
            .filter(|pair| !pair[0].0[column].is_empty() && !pair[1].0[column].is_empty())
            .map(|pair| (&pair[0].0[column], &pair[1].0[column]))
            .collect();
        let carried = pairs
            .iter()
            .filter(|(before, after)| {
                let full = before
                    .iter()
                    .map(|piece| piece.bbox[2])
                    .fold(f64::NEG_INFINITY, f64::max)
                    >= measure - 0.15 * (measure - start);
                let (before, after) = (text(before), text(after));
                full && !before.trim_end().ends_with(['.', ';', ':', '!', '?'])
                    && after
                        .trim_start()
                        .chars()
                        .next()
                        .is_some_and(char::is_lowercase)
            })
            .count();
        pairs.len() >= 2 && carried * 10 >= pairs.len() * 4
    });
    let languages: Vec<Option<bool>> = (0..columns)
        .map(|column| {
            let column_lines: Vec<&Line> = rows
                .iter()
                .flat_map(|(row, _)| &row[column])
                .map(|piece| &page.lines[piece.line])
                .collect();
            crate::layout::column_language(
                crate::layout::language_votes(column_lines.into_iter()),
                5,
            )
        })
        .collect();
    // Neighbouring cells that repeat each other's names and numbers are a text and its
    // translation, row by row.
    let parallel = (0..columns - 1).any(|column| {
        let tokens = |pieces: &[Piece]| -> HashSet<String> {
            text(pieces)
                .split(|c: char| !c.is_alphanumeric())
                .filter(|token| {
                    token.chars().count() >= 3 || token.chars().any(|c| c.is_ascii_digit())
                })
                .map(str::to_lowercase)
                .collect()
        };
        let pairs: Vec<(HashSet<String>, HashSet<String>)> = rows
            .iter()
            .filter(|(row, _)| !row[column].is_empty() && !row[column + 1].is_empty())
            .map(|(row, _)| (tokens(&row[column]), tokens(&row[column + 1])))
            .filter(|(left, right)| !left.is_empty() && !right.is_empty())
            .collect();
        let alike = pairs
            .iter()
            .filter(|(left, right)| {
                left.intersection(right).count() * 5 >= left.union(right).count() * 2
            })
            .count();
        pairs.len() >= 3 && alike * 2 >= pairs.len()
    });
    let bilingual =
        parallel || (languages.contains(&Some(true)) && languages.contains(&Some(false)));
    let contacts = share(&cells, &contact_detail) >= 0.2;
    let header = {
        let (first, _) = &rows[0];
        let cells: Vec<String> = first
            .iter()
            .filter(|cell| !cell.is_empty())
            .map(|cell| text(cell))
            .collect();
        let bold_head = first
            .iter()
            .flatten()
            .all(|piece| bold(&page.lines[piece.line]))
            && rows[1..]
                .iter()
                .flat_map(|(row, _)| row.iter().flatten())
                .any(|piece| !bold(&page.lines[piece.line]));
        let labelled = cells.len() >= 2
            && cells.iter().all(|cell| {
                let cell = cell.trim();
                cell.chars().count() <= 30
                    && cell.chars().any(char::is_alphabetic)
                    && !bare_marker(cell)
                    && !cell.ends_with(':')
            });
        let numbered = share(&column_texts[0], &bare_marker) >= 0.7;
        let figures = (1..columns).any(|column| share(&column_texts[column], &figure) >= 0.6);
        labelled
            && (bold_head || (first.iter().all(|cell| !cell.is_empty()) && (numbered || figures)))
    };
    let sequential = {
        let numbers: Vec<u32> = column_texts[0]
            .iter()
            .filter_map(|text| text.trim().parse::<u32>().ok())
            .collect();
        numbers.len() >= 5 && numbers.windows(2).all(|pair| pair[1] == pair[0] + 1)
    };
    let figures = (1..columns).any(|column| share(&column_texts[column], &figure) >= 0.6);
    // Page numbers or entry numbers running up beside titles are a contents list.
    let ascending = |texts: &[String]| {
        let numbers: Vec<u32> = texts
            .iter()
            .filter_map(|text| text.trim().trim_end_matches('.').parse::<u32>().ok())
            .collect();
        numbers.len() >= 3
            && numbers.len() * 10 >= texts.len() * 7
            && numbers.windows(2).filter(|pair| pair[0] <= pair[1]).count() * 5
                >= (numbers.len() - 1) * 4
    };
    let listed = !header && (ascending(&column_texts[0]) || ascending(&column_texts[columns - 1]));
    // Markers beside running text are a list or numbered paragraphs.
    let itemized = !header
        && (0..columns - 1).any(|column| {
            share(&column_texts[column], &bare_marker) >= 0.7
                && median_length(&column_texts[column + 1]) > 30
        });
    if !aligned
        || listed
        || itemized
        || long > 1
        || leaders >= 2
        || crate::layout::contents_grid(&lines, page.width)
        || caption
        || labels
        || contacts
        || ticked
        || choices
        || flowing
        || bilingual
        || sequential
        || !(columns >= 3 || figures || header)
    {
        return None;
    }
    let mut table_rows: Vec<TableRow> = rows
        .into_iter()
        .map(|(row, _)| TableRow {
            header: false,
            cells: row
                .into_iter()
                .enumerate()
                .map(|(column, pieces)| cell_of(page, column, pieces))
                .collect(),
        })
        .collect();
    table_rows[0].header = header;
    // A line parted between cells must run straight on from one cell into the next.
    let sequence: Vec<&CellPart> = table_rows
        .iter()
        .flat_map(|row| &row.cells)
        .flat_map(|cell| &cell.parts)
        .collect();
    let mut runs: HashMap<&str, usize> = HashMap::new();
    for (position, part) in sequence.iter().enumerate() {
        if position == 0 || sequence[position - 1].line_id != part.line_id {
            *runs.entry(part.line_id.as_str()).or_default() += 1;
        }
    }
    if runs.values().any(|count| *count > 1) {
        return None;
    }
    let top = bands[0]
        .iter()
        .map(|piece| piece.bbox[1])
        .fold(f64::INFINITY, f64::min);
    let bottom = bands[bands.len() - 1]
        .iter()
        .map(|piece| piece.bbox[3])
        .fold(f64::NEG_INFINITY, f64::max);
    Some(PdfTable {
        rows: table_rows,
        page_index: page.index,
        xs,
        top,
        bottom,
    })
}

fn drawn_tables(page: &Page) -> Vec<PdfTable> {
    let (mut horizontal, mut vertical) = (Vec::new(), Vec::new());
    for region in page.regions.iter().filter(|region| region.kind == "rule") {
        let [x0, y0, x1, y1] = region.bbox;
        if (y1 - y0).abs() <= (x1 - x0).abs() {
            horizontal.push(Rule {
                at: (y0 + y1) / 2.0,
                start: x0.min(x1),
                end: x0.max(x1),
            });
        } else {
            vertical.push(Rule {
                at: (x0 + x1) / 2.0,
                start: y0.min(y1),
                end: y0.max(y1),
            });
        }
    }
    let (horizontal, vertical) = (join(horizontal), join(vertical));
    if horizontal.len() < 2 {
        return Vec::new();
    }
    // Rules that cross or meet belong to one lattice.
    let count = horizontal.len();
    let mut parents: Vec<usize> = (0..count + vertical.len()).collect();
    for (h, across) in horizontal.iter().enumerate() {
        for (v, down) in vertical.iter().enumerate() {
            if down.start - TOLERANCE <= across.at
                && across.at <= down.end + TOLERANCE
                && across.start - TOLERANCE <= down.at
                && down.at <= across.end + TOLERANCE
            {
                union(&mut parents, h, count + v);
            }
        }
    }
    let mut lattices: HashMap<usize, (Vec<Rule>, Vec<Rule>)> = HashMap::new();
    for (index, rule) in horizontal.iter().enumerate() {
        lattices
            .entry(find(&mut parents, index))
            .or_default()
            .0
            .push(*rule);
    }
    for (index, rule) in vertical.iter().enumerate() {
        lattices
            .entry(find(&mut parents, count + index))
            .or_default()
            .1
            .push(*rule);
    }
    let mut tables: Vec<PdfTable> = lattices
        .values()
        .filter_map(|(horizontal, vertical)| lattice_table(page, horizontal, vertical))
        .collect();
    let lines: HashMap<&str, usize> = page
        .lines
        .iter()
        .enumerate()
        .map(|(index, line)| (line.id.as_str(), index))
        .collect();
    let taken: HashSet<usize> = tables
        .iter()
        .flat_map(PdfTable::line_ids)
        .map(|id| lines[id.as_str()])
        .collect();
    let free: Vec<Rule> = lattices
        .into_values()
        .filter(|(_, vertical)| vertical.is_empty())
        .flat_map(|(horizontal, _)| horizontal)
        .collect();
    tables.extend(
        open_tables(page, &free, &taken)
            .into_iter()
            .filter(|table| !bilingual_columns(page, table)),
    );
    tables
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
        .map(|lines| {
            crate::layout::column_language(crate::layout::language_votes(lines.into_iter()), 5)
        })
        .collect();
    languages.contains(&Some(true)) && languages.contains(&Some(false))
}

fn page_tables(page: &Page, translation: &HashSet<String>) -> Vec<PdfTable> {
    let mut tables = drawn_tables(page);
    let lines: HashMap<&str, usize> = page
        .lines
        .iter()
        .enumerate()
        .map(|(index, line)| (line.id.as_str(), index))
        .collect();
    let taken: HashSet<usize> = tables
        .iter()
        .flat_map(PdfTable::line_ids)
        .map(|id| lines[id.as_str()])
        .collect();
    tables.extend(aligned_tables(page, &taken, translation));
    tables.sort_by(|left, right| left.top.total_cmp(&right.top));
    tables
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

/// Every ruled table in the document, a table cut off by a page break joined to its
/// continuation, in reading order.
pub(crate) fn ruled_tables(pages: &[Page], translation: &HashSet<String>) -> Vec<PdfTable> {
    let mut tables: Vec<PdfTable> = Vec::new();
    let mut previous_last: Option<usize> = None;
    for page in pages {
        let found = page_tables(page, translation);
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
    tables
}

/// Puts each table's lines on the page in row order, cell by cell, where its first line
/// stood, and renumbers the page's reading order.
pub(crate) fn order_table_lines(page: &mut Page, tables: &[PdfTable]) {
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

/// Each cell's text is body text of its own: a paragraph never runs from one cell into
/// the next, though a cell may hold several.
pub(crate) fn part_cell_paragraphs(pages: &mut [Page], tables: &[PdfTable]) {
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
            line.region_type = "body".to_owned();
            line.note_region_mode.clear();
        }
    }
}

/// Line ids of every ruled table's cells.
pub(crate) fn table_line_ids(tables: &[PdfTable]) -> HashSet<String> {
    tables
        .iter()
        .flat_map(PdfTable::line_ids)
        .cloned()
        .collect()
}
