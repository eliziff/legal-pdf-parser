use super::{detect_tables, detect_tables_from_lines, detect_tables_from_rects, Table, TableKind};
use crate::extractor::{merge_text_items_for_layout, ItemType};
use crate::types::{PdfLine, PdfRect, TextItem};
use std::collections::HashSet;

/// A data table with page geometry and detector provenance preserved for
/// structured consumers that do not render pdf-inspector's Markdown.
#[derive(Debug, Clone)]
pub struct DetectedTable {
    pub page: u32,
    pub index: usize,
    pub bbox: [f32; 4],
    pub cells: Vec<Vec<String>>,
    pub method: &'static str,
    pub confidence: f64,
}

#[derive(Default)]
struct Shape {
    rows: usize,
    columns: usize,
    slots: usize,
    populated: usize,
    most_populated: usize,
    longest: usize,
    numeric: usize,
}

impl Shape {
    fn of(table: &Table) -> Self {
        let mut shape = Self {
            rows: table.cells.len(),
            columns: table.cells.iter().map(Vec::len).max().unwrap_or(0),
            slots: table.cells.iter().map(Vec::len).sum(),
            ..Self::default()
        };
        for row in &table.cells {
            let mut row_populated = 0;
            for cell in row.iter().filter(|cell| !cell.trim().is_empty()) {
                shape.populated += 1;
                row_populated += 1;
                let (length, numeric) = cell.chars().fold((0, false), |(length, numeric), ch| {
                    (length + 1, numeric || ch.is_ascii_digit())
                });
                shape.longest = shape.longest.max(length);
                shape.numeric += usize::from(numeric);
            }
            shape.most_populated = shape.most_populated.max(row_populated);
        }
        shape
    }
}

fn item_bbox(table: &Table, items: &[TextItem]) -> Option<[f32; 4]> {
    let mut members = table
        .item_indices
        .iter()
        .filter_map(|&index| items.get(index));
    let first = members.next()?;
    Some(members.fold(
        [
            first.x,
            first.y,
            first.x + first.width,
            first.y + first.height,
        ],
        |[x0, y0, x1, y1], item| {
            [
                x0.min(item.x),
                y0.min(item.y),
                x1.max(item.x + item.width),
                y1.max(item.y + item.height),
            ]
        },
    ))
}

fn is_caption(text: &str) -> bool {
    let lower = text.trim().to_ascii_lowercase();
    let Some(rest) = lower
        .strip_prefix("table ")
        .or_else(|| lower.strip_prefix("tableau "))
    else {
        return false;
    };
    let token = rest.split_whitespace().next().unwrap_or("");
    let label = token.trim_end_matches([':', '.', '-', '–', '—']);
    !label.is_empty()
        && label.chars().all(|ch| ch.is_ascii_alphanumeric())
        && (token.len() > label.len()
            || rest[label.len()..]
                .trim_start()
                .starts_with([':', '.', '-', '–', '—']))
}

fn has_nearby_caption(table: &Table, items: &[TextItem]) -> bool {
    let Some([left, _, right, top]) = item_bbox(table, items) else {
        return false;
    };
    items.iter().enumerate().any(|(index, item)| {
        !table.item_indices.contains(&index)
            && is_caption(&item.text)
            && item.y + item.height >= top - 5.0
            && item.y <= top + 60.0
            && item.x < right
            && item.x + item.width > left
    })
}

fn repeated_multi_cell_rows(table: &Table) -> bool {
    let count = table
        .cells
        .iter()
        .filter(|row| row.iter().filter(|cell| !cell.trim().is_empty()).count() >= 2)
        .count();
    count >= 2 && count * 3 >= table.cells.len()
}

fn page_values<T: Clone>(
    values: &[T],
    page_number: u32,
    page: impl Fn(&T) -> u32,
    sorted: bool,
    keep: impl Fn(&T) -> bool,
) -> Vec<T> {
    let values = if sorted {
        let start = values.partition_point(|value| page(value) < page_number);
        let end = start + values[start..].partition_point(|value| page(value) == page_number);
        &values[start..end]
    } else {
        values
    };
    values
        .iter()
        .filter(|value| page(value) == page_number && keep(value))
        .cloned()
        .collect()
}

fn sorted_by_page<T>(values: &[T], page: impl Fn(&T) -> u32) -> bool {
    values
        .windows(2)
        .all(|pair| page(&pair[0]) <= page(&pair[1]))
}

fn table_rule(line: &PdfLine) -> bool {
    let dx = (line.x2 - line.x1).abs();
    let dy = (line.y2 - line.y1).abs();
    let tolerance = 2.0_f32.to_radians().tan();
    dx.hypot(dy) >= 20.0
        && ((dx > 0.01 && dy / dx <= tolerance) || (dy > 0.01 && dx / dy <= tolerance))
}

fn simplified_rects(rects: &[PdfRect], page_area: f64) -> Vec<PdfRect> {
    let mut seen = HashSet::new();
    rects
        .iter()
        .filter(|rect| {
            f64::from(rect.width.abs() * rect.height.abs()) < page_area * 0.75
                && seen.insert((
                    (rect.x * 2.0).round() as i64,
                    (rect.y * 2.0).round() as i64,
                    (rect.width * 2.0).round() as i64,
                    (rect.height * 2.0).round() as i64,
                ))
        })
        .cloned()
        .collect()
}

/// Detect structured data tables for the requested `(page, width, height)`
/// triples. Text, rectangles, and lines may contain the whole document.
pub fn detect_structured_tables(
    items: &[TextItem],
    rects: &[PdfRect],
    lines: &[PdfLine],
    pages: &[(u32, f64, f64)],
) -> Vec<DetectedTable> {
    let items_sorted = sorted_by_page(items, |item| item.page);
    let rects_sorted = sorted_by_page(rects, |rect| rect.page);
    let lines_sorted = sorted_by_page(lines, |line| line.page);
    let mut output = Vec::new();

    for &(page, width, height) in pages {
        let page_items = merge_text_items_for_layout(page_values(
            items,
            page,
            |item| item.page,
            items_sorted,
            |item| matches!(item.item_type, ItemType::Text),
        ));
        if page_items.is_empty() {
            continue;
        }
        let page_rects = page_values(rects, page, |rect| rect.page, rects_sorted, |_| true);
        let page_lines = page_values(lines, page, |line| line.page, lines_sorted, table_rule);
        let text_tables = || {
            let mut sizes: Vec<_> = page_items
                .iter()
                .map(|item| item.font_size)
                .filter(|&size| size > 0.0)
                .collect();
            sizes.sort_by(f32::total_cmp);
            detect_tables(
                &page_items,
                sizes.get(sizes.len() / 2).copied().unwrap_or(10.0),
                false,
            )
        };

        let mut method = "rect";
        let mut confidence = 0.95;
        let mut tables: Vec<_> = detect_tables_from_rects(&page_items, &page_rects, page)
            .0
            .into_iter()
            .filter(|table| table.kind == TableKind::Data)
            .filter(|table| {
                if table.item_indices.len() * 10 < page_items.len() * 9 {
                    return true;
                }
                let shape = Shape::of(table);
                shape.populated * 5 >= shape.slots * 2 && shape.longest <= 300
                    || shape.columns >= 4 && shape.most_populated >= 4 && shape.longest <= 800
            })
            .collect();

        if tables.is_empty() {
            let normalized = simplified_rects(&page_rects, width * height);
            if normalized.len() < page_rects.len() {
                tables = detect_tables_from_rects(&page_items, &normalized, page)
                    .0
                    .into_iter()
                    .filter(|table| table.kind == TableKind::Data)
                    .filter(|table| {
                        has_nearby_caption(table, &page_items)
                            || item_bbox(table, &page_items).is_some_and(|bbox| {
                                let shape = Shape::of(table);
                                f64::from((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                                    >= width * height * 0.15
                                    && repeated_multi_cell_rows(table)
                                    && shape.populated * 5 > shape.slots * 2
                            })
                    })
                    .collect();
                if !tables.is_empty() {
                    (method, confidence) = ("rect-normalized", 0.93);
                }
            }
        }
        if tables.is_empty() {
            (method, confidence) = ("line", 0.90);
            tables = detect_tables_from_lines(&page_items, &page_lines, page)
                .into_iter()
                .filter(|table| table.kind == TableKind::Data)
                .collect();
        }
        if method == "line" && tables.len() == 1 {
            let vector = Shape::of(&tables[0]);
            if vector.longest > 400 {
                if let Some(replacement) = text_tables().into_iter().find(|table| {
                    let shape = Shape::of(table);
                    table.kind == TableKind::Data
                        && shape.rows + 1 == vector.rows
                        && shape.columns == vector.columns
                        && shape.longest * 2 < vector.longest
                }) {
                    (method, confidence, tables) = ("text-geometry", 0.80, vec![replacement]);
                }
            }
        }
        if tables.is_empty() {
            (method, confidence) = ("text-geometry", 0.75);
            let caption = page_items.iter().any(|item| {
                let text = item.text.trim().to_ascii_lowercase();
                text.starts_with("table ") || text.starts_with("tableau ")
            });
            tables = text_tables()
                .into_iter()
                .filter(|table| {
                    let shape = Shape::of(table);
                    table.kind == TableKind::Data
                        && shape.rows >= 4
                        && (caption
                            || shape.populated * 3 >= shape.rows * shape.columns * 2
                                && shape.numeric * 5 >= shape.populated.max(1))
                })
                .collect();
        }

        output.extend(tables.into_iter().enumerate().filter_map(|(index, table)| {
            Some(DetectedTable {
                page,
                index,
                bbox: item_bbox(&table, &page_items)?,
                cells: table.cells,
                method,
                confidence,
            })
        }));
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_rule_projection_matches_detector_contract() {
        let line = |x2, y2| PdfLine {
            x1: 0.0,
            y1: 0.0,
            x2,
            y2,
            page: 1,
        };
        assert!(table_rule(&line(20.0, 0.0)));
        assert!(table_rule(&line(0.0, 20.0)));
        assert!(!table_rule(&line(19.99, 0.0)));
        assert!(!table_rule(&line(20.0, 20.0)));
    }

    #[test]
    fn caption_grammar_requires_a_delimited_label() {
        assert!(is_caption("Table 1: Results"));
        assert!(is_caption("TABLE IV. RESULTS"));
        assert!(is_caption("Tableau 2 — Résultats"));
        assert!(!is_caption("Table 1 captures the disparities"));
        assert!(!is_caption("table manners"));
        assert!(!is_caption("Table of Contents"));
    }
}
