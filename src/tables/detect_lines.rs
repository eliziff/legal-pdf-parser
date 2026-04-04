//! Line-based table detection.
//!
//! Detects tables from PDF path operators (`m`/`l`/`S`) that draw ruled
//! gridlines.  Many IRS forms and government PDFs use these instead of
//! `re` (rectangle) operators.

use crate::tables::Table;
use crate::types::{PdfLine, TextItem};

use super::detect_rects::{assign_items_to_grid, snap_edges};

/// Detect tables from line segments on a given page.
///
/// Lines are classified as horizontal or vertical, snapped into grid edges,
/// and validated before assigning text items to the resulting grid.
pub fn detect_tables_from_lines(items: &[TextItem], lines: &[PdfLine], page: u32) -> Vec<Table> {
    // Filter lines for this page
    let page_lines: Vec<&PdfLine> = lines.iter().filter(|l| l.page == page).collect();
    if page_lines.is_empty() {
        return Vec::new();
    }

    // Classify lines as horizontal or vertical (within 2° of axis)
    let mut horizontals: Vec<(f32, f32, f32)> = Vec::new(); // (y, x_min, x_max)
    let mut verticals: Vec<(f32, f32, f32)> = Vec::new(); // (x, y_min, y_max)

    let angle_tolerance = 2.0_f32.to_radians().tan(); // ~0.035

    for line in &page_lines {
        let dx = (line.x2 - line.x1).abs();
        let dy = (line.y2 - line.y1).abs();
        let length = (dx * dx + dy * dy).sqrt();

        // Skip very short lines (decorations, tick marks)
        if length < 20.0 {
            continue;
        }

        if dx > 0.01 && dy / dx <= angle_tolerance {
            // Horizontal line
            let y = (line.y1 + line.y2) / 2.0;
            let x_min = line.x1.min(line.x2);
            let x_max = line.x1.max(line.x2);
            horizontals.push((y, x_min, x_max));
        } else if dy > 0.01 && dx / dy <= angle_tolerance {
            // Vertical line
            let x = (line.x1 + line.x2) / 2.0;
            let y_min = line.y1.min(line.y2);
            let y_max = line.y1.max(line.y2);
            verticals.push((x, y_min, y_max));
        }
        // Diagonal lines are ignored
    }

    if horizontals.len() < 3 || verticals.len() < 2 {
        return Vec::new();
    }

    log::debug!(
        "detect_lines p{}: {} horiz, {} vert lines (of {} total on page)",
        page,
        horizontals.len(),
        verticals.len(),
        page_lines.len()
    );

    // Snap Y-values of horizontal lines → row edges
    let h_ys: Vec<f32> = horizontals.iter().map(|(y, _, _)| *y).collect();
    let row_edges = snap_edges(&h_ys, 3.0);

    // Snap X-values of vertical lines → column edges
    let v_xs: Vec<f32> = verticals.iter().map(|(x, _, _)| *x).collect();
    let col_edges = snap_edges(&v_xs, 3.0);

    log::debug!(
        "detect_lines p{}: {} row edges, {} col edges after snap",
        page,
        row_edges.len(),
        col_edges.len()
    );

    // Require at least 2 columns (3 col edges) and 2 rows (3 row edges).
    // A single column of horizontal lines is just separator rules, not a table.
    if row_edges.len() < 3 || col_edges.len() < 3 {
        return Vec::new();
    }

    // Cap grid size: >20 columns is almost certainly a diagram, not a table
    if col_edges.len() > 21 || row_edges.len() > 80 {
        log::debug!(
            "detect_lines p{}: rejected — too many edges ({}x{})",
            page,
            row_edges.len(),
            col_edges.len()
        );
        return Vec::new();
    }

    let table_x_min = col_edges.first().copied().unwrap_or(0.0);
    let table_x_max = col_edges.last().copied().unwrap_or(0.0);
    let table_width = table_x_max - table_x_min;
    if table_width < 50.0 {
        return Vec::new();
    }

    let table_y_min = row_edges.first().copied().unwrap_or(0.0);
    let table_y_max = row_edges.last().copied().unwrap_or(0.0);
    let table_height = (table_y_max - table_y_min).abs();
    if table_height < 20.0 {
        return Vec::new();
    }

    // Reject page-spanning frames: if the grid covers >90% of a standard page
    // dimension in both axes, it's a border frame, not a table.
    // Standard pages are ~595×842 (A4) or ~612×792 (Letter).
    if table_width > 500.0 && table_height > 700.0 {
        log::debug!(
            "detect_lines p{}: rejected — page-spanning frame ({:.0}×{:.0})",
            page,
            table_width,
            table_height
        );
        return Vec::new();
    }

    // Validate horizontal lines: at least 3 should span a meaningful width.
    // Full-width spanning (>50%) is ideal, but tables with partial horizontal
    // rules (column-level separators) are also valid if there are enough.
    let spanning_h = horizontals
        .iter()
        .filter(|(_, x_min, x_max)| (x_max - x_min) > table_width * 0.5)
        .count();
    let partial_h = horizontals
        .iter()
        .filter(|(_, x_min, x_max)| (x_max - x_min) > table_width * 0.15)
        .count();
    if spanning_h < 3 && partial_h < 6 {
        log::debug!(
            "detect_lines p{}: rejected — {} spanning + {} partial H lines",
            page,
            spanning_h,
            partial_h
        );
        return Vec::new();
    }

    // Validate vertical lines: at least 2 should span a meaningful height.
    // Full spanning (>30%) is ideal, but accept many shorter lines (>10%)
    // for tables with partial column separators.
    let spanning_v = verticals
        .iter()
        .filter(|(_, y_min, y_max)| (y_max - y_min) > table_height * 0.3)
        .count();
    let partial_v = verticals
        .iter()
        .filter(|(_, y_min, y_max)| (y_max - y_min) > table_height * 0.10)
        .count();
    if spanning_v < 2 && partial_v < 4 {
        log::debug!(
            "detect_lines p{}: rejected — {} spanning + {} partial V lines",
            page,
            spanning_v,
            partial_v
        );
        return Vec::new();
    }

    // Row edges need to be in descending order (top of page = higher Y first)
    let mut row_edges_desc = row_edges;
    row_edges_desc.sort_by(|a, b| b.total_cmp(a));

    log::debug!(
        "detect_lines p{}: {} row_edges, {} col_edges, table=({:.0},{:.0})-({:.0},{:.0}), spanning_h={}, spanning_v={}",
        page, row_edges_desc.len(), col_edges.len(),
        table_x_min, table_y_min, table_x_max, table_y_max,
        spanning_h, spanning_v
    );

    // Assign items to grid
    let (cells, item_indices) = assign_items_to_grid(items, &col_edges, &row_edges_desc, page);

    // Require at least 2 non-empty rows
    let non_empty_rows = cells
        .iter()
        .filter(|row| row.iter().any(|cell| !cell.is_empty()))
        .count();
    if non_empty_rows < 2 {
        log::debug!(
            "detect_lines p{}: rejected — only {} non-empty rows",
            page,
            non_empty_rows
        );
        return Vec::new();
    }

    // Content density: at least 15% of cells should have content
    let num_cols_grid = cells.first().map_or(0, |r| r.len());
    let total_cells = cells.len() * num_cols_grid;
    if total_cells > 0 {
        let filled_cells = cells
            .iter()
            .flat_map(|row| row.iter())
            .filter(|cell| !cell.is_empty())
            .count();
        let density = filled_cells as f32 / total_cells as f32;
        if density < 0.15 {
            log::debug!(
                "detect_lines p{}: rejected — low density {:.2}",
                page,
                density
            );
            return Vec::new();
        }
    }

    // Require that at least 2 distinct columns have content.
    // Charts/diagrams have text concentrated on axes (1 column);
    // real tables spread data across multiple columns.
    let cols_with_content = (0..num_cols_grid)
        .filter(|&c| {
            cells
                .iter()
                .any(|row| row.get(c).is_some_and(|cell| !cell.is_empty()))
        })
        .count();
    if cols_with_content < 2 {
        return Vec::new();
    }

    // The grid must capture a meaningful portion of the page's text items.
    // Chart/graph grids on textbook pages capture scattered labels but miss
    // the bulk of the page content (explanatory text, problem statements).
    let page_item_count = items.iter().filter(|i| i.page == page).count();
    if page_item_count > 0 {
        let capture_ratio = item_indices.len() as f32 / page_item_count as f32;
        // If the grid captures less than 20% of items, it's not a real table
        if capture_ratio < 0.20 {
            return Vec::new();
        }
    }

    // Reject grids with very uniform row spacing — likely chart gridlines.
    // Real tables have variable row heights; chart Y-axes have equal spacing.
    if row_edges_desc.len() >= 5 {
        let spacings: Vec<f32> = row_edges_desc
            .windows(2)
            .map(|w| (w[0] - w[1]).abs())
            .collect();
        let mean_spacing = spacings.iter().sum::<f32>() / spacings.len() as f32;
        if mean_spacing > 0.1 {
            let variance = spacings
                .iter()
                .map(|s| (s - mean_spacing).powi(2))
                .sum::<f32>()
                / spacings.len() as f32;
            let cv = variance.sqrt() / mean_spacing;
            // CV < 0.02 means nearly identical spacing — likely chart grid.
            // Spreadsheet-exported tables often have uniform rows (CV 0.03-0.05),
            // so we use a tighter threshold to avoid false negatives.
            if cv < 0.02 {
                log::debug!(
                    "detect_lines p{}: rejected — uniform row spacing (cv={:.4}, mean={:.1})",
                    page,
                    cv,
                    mean_spacing
                );
                return Vec::new();
            }
        }
    }

    let num_cols = col_edges.len() - 1;
    let num_rows = row_edges_desc.len() - 1;

    if num_rows < 2 || num_cols < 2 {
        return Vec::new();
    }

    log::debug!(
        "detect_lines p{}: ACCEPTED {}x{} grid, {} items captured of {} on page, non_empty_rows={}, cols_with_content={}",
        page, num_rows, num_cols, item_indices.len(), page_item_count, non_empty_rows, cols_with_content
    );

    // Post-process: split rows that contain items at multiple distinct Y
    // positions.  This happens in stacked sub-tables where "Note:" footer
    // text and the next section's "Category" header land between the same
    // pair of horizontal rules.  Re-assign items to sub-rows by Y proximity.
    let cells = split_multi_y_rows(cells, items, &col_edges, &row_edges_desc, page);

    // Split the grid into separate tables at rows that lack vertical border
    // coverage (e.g. "Note:" footer text that sits between horizontal rules
    // but outside the actual table grid).  A row is "unbounded" when fewer
    // than 2 vertical lines span its Y range — it's freestanding text, not
    // a table cell.
    let mut tables = Vec::new();
    let mut current_rows: Vec<Vec<String>> = Vec::new();

    for (r, row) in cells.into_iter().enumerate() {
        // Determine Y range for this row
        let row_top = if r < row_edges_desc.len() {
            row_edges_desc[r]
        } else {
            row_edges_desc.last().copied().unwrap_or(0.0)
        };
        let row_bot = if r + 1 < row_edges_desc.len() {
            row_edges_desc[r + 1]
        } else {
            row_edges_desc.last().copied().unwrap_or(0.0) - 15.0
        };

        // Count vertical lines that span this row's Y range
        let v_covering = verticals
            .iter()
            .filter(|(_, y_min, y_max)| *y_min <= row_bot + 2.0 && *y_max >= row_top - 2.0)
            .count();

        if v_covering >= 2 {
            current_rows.push(row);
        } else {
            // Flush accumulated rows as a table
            if current_rows.len() >= 2 {
                tables.push(Table {
                    columns: col_edges.clone(),
                    rows: Vec::new(),
                    cells: std::mem::take(&mut current_rows),
                    item_indices: item_indices.clone(),
                });
            } else {
                current_rows.clear();
            }
            // The unbounded row's text becomes a standalone "table" with 1 row
            // so it gets emitted as text outside the table.
            let text = row
                .iter()
                .map(|c| c.trim())
                .filter(|c| !c.is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            if !text.is_empty() {
                // Emit as a 1-cell table which the markdown converter will
                // render as a standalone line (single-row tables are just text).
                tables.push(Table {
                    columns: vec![col_edges[0], *col_edges.last().unwrap_or(&col_edges[0])],
                    rows: Vec::new(),
                    cells: vec![vec![text]],
                    item_indices: item_indices.clone(),
                });
            }
        }
    }
    // Flush remaining rows
    if current_rows.len() >= 2 {
        tables.push(Table {
            columns: col_edges,
            rows: Vec::new(),
            cells: current_rows,
            item_indices,
        });
    }

    tables
}

/// Split table rows where items within cells span multiple Y positions.
/// Groups items by Y proximity and emits one output row per Y group.
fn split_multi_y_rows(
    cells: Vec<Vec<String>>,
    items: &[TextItem],
    col_edges: &[f32],
    row_edges: &[f32],
    page: u32,
) -> Vec<Vec<String>> {
    let num_cols = col_edges.len() - 1;
    let num_rows = cells.len();
    if num_rows == 0 {
        return cells;
    }

    // Re-collect items per cell to get Y positions
    let mut cell_items: Vec<Vec<Vec<&TextItem>>> = vec![vec![Vec::new(); num_cols]; num_rows];
    for item in items {
        if item.page != page {
            continue;
        }
        let cx = item.x + item.width / 2.0;
        let cy = item.y;
        let col = (0..num_cols).find(|&c| cx >= col_edges[c] - 2.0 && cx <= col_edges[c + 1] + 2.0);
        let row = (0..num_rows).find(|&r| cy >= row_edges[r + 1] - 2.0 && cy <= row_edges[r] + 2.0);
        if let (Some(c), Some(r)) = (col, row) {
            cell_items[r][c].push(item);
        }
    }

    let y_tol = 4.0; // items within 4pt are on the same sub-row
    let mut out: Vec<Vec<String>> = Vec::new();

    for (r, _row_cells) in cells.iter().enumerate() {
        // Collect all Y positions across all columns in this row
        let mut all_ys: Vec<f32> = cell_items[r]
            .iter()
            .flat_map(|items| items.iter().map(|i| i.y))
            .collect();
        if all_ys.is_empty() {
            out.push(vec![String::new(); num_cols]);
            continue;
        }
        all_ys.sort_by(|a, b| b.total_cmp(a)); // descending (top first)
        all_ys.dedup_by(|a, b| (*a - *b).abs() < y_tol);

        if all_ys.len() <= 1 {
            // Single Y level — keep original row
            out.push(cells[r].clone());
        } else {
            // Multiple Y levels — split into sub-rows
            for &sub_y in &all_ys {
                let mut sub_row = Vec::with_capacity(num_cols);
                for col_items in &cell_items[r] {
                    let text: String = col_items
                        .iter()
                        .filter(|i| (i.y - sub_y).abs() < y_tol)
                        .map(|i| i.text.trim())
                        .filter(|t| !t.is_empty())
                        .collect::<Vec<_>>()
                        .join(" ");
                    sub_row.push(text);
                }
                // Skip fully empty sub-rows
                if sub_row.iter().any(|s| !s.is_empty()) {
                    out.push(sub_row);
                }
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ItemType;

    fn make_item(text: &str, x: f32, y: f32, page: u32) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y,
            width: 30.0,
            height: 10.0,
            font: "F1".into(),
            font_size: 10.0,
            page,
            is_bold: false,
            is_italic: false,
            item_type: ItemType::Text,
            mcid: None,
        }
    }

    fn make_hline(y: f32, x1: f32, x2: f32, page: u32) -> PdfLine {
        PdfLine {
            x1,
            y1: y,
            x2,
            y2: y,
            page,
        }
    }

    fn make_vline(x: f32, y1: f32, y2: f32, page: u32) -> PdfLine {
        PdfLine {
            x1: x,
            y1,
            x2: x,
            y2,
            page,
        }
    }

    #[test]
    fn test_basic_grid_detection() {
        // 3x2 grid with horizontal lines at y=500, 480, 460 and vertical at x=100, 200, 300
        let lines = vec![
            make_hline(500.0, 100.0, 300.0, 1),
            make_hline(480.0, 100.0, 300.0, 1),
            make_hline(460.0, 100.0, 300.0, 1),
            make_vline(100.0, 460.0, 500.0, 1),
            make_vline(200.0, 460.0, 500.0, 1),
            make_vline(300.0, 460.0, 500.0, 1),
        ];

        let items = vec![
            make_item("Col A", 110.0, 490.0, 1),
            make_item("Col B", 210.0, 490.0, 1),
            make_item("val 1", 110.0, 470.0, 1),
            make_item("val 2", 210.0, 470.0, 1),
        ];

        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].cells.len(), 2); // 2 data rows
        assert_eq!(tables[0].cells[0].len(), 2); // 2 columns
    }

    #[test]
    fn test_short_lines_ignored() {
        // Lines shorter than 20pt should be ignored
        let lines = vec![
            make_hline(500.0, 100.0, 110.0, 1), // 10pt - too short
            make_hline(480.0, 100.0, 115.0, 1), // 15pt - too short
            make_hline(460.0, 100.0, 112.0, 1), // 12pt - too short
        ];

        let items = vec![make_item("text", 105.0, 490.0, 1)];

        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_wrong_page_ignored() {
        let lines = vec![
            make_hline(500.0, 100.0, 300.0, 2),
            make_hline(480.0, 100.0, 300.0, 2),
            make_hline(460.0, 100.0, 300.0, 2),
            make_vline(100.0, 460.0, 500.0, 2),
            make_vline(200.0, 460.0, 500.0, 2),
            make_vline(300.0, 460.0, 500.0, 2),
        ];

        let items = vec![make_item("text", 110.0, 490.0, 1)];

        // Request page 1, but lines are on page 2
        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_empty_grid_rejected() {
        // Grid with no text items inside
        let lines = vec![
            make_hline(500.0, 100.0, 300.0, 1),
            make_hline(480.0, 100.0, 300.0, 1),
            make_hline(460.0, 100.0, 300.0, 1),
            make_vline(100.0, 460.0, 500.0, 1),
            make_vline(200.0, 460.0, 500.0, 1),
            make_vline(300.0, 460.0, 500.0, 1),
        ];

        let items: Vec<TextItem> = Vec::new();

        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_horizontal_rules_not_table() {
        // Only horizontal lines with no verticals — separator rules, not a table
        let lines = vec![
            make_hline(500.0, 100.0, 500.0, 1),
            make_hline(480.0, 100.0, 500.0, 1),
            make_hline(460.0, 100.0, 500.0, 1),
            make_hline(440.0, 100.0, 500.0, 1),
        ];

        let items = vec![
            make_item("text1", 110.0, 490.0, 1),
            make_item("text2", 110.0, 470.0, 1),
        ];

        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert!(tables.is_empty());
    }

    #[test]
    fn test_single_column_rejected() {
        // Only 2 col edges (1 column) — not a table even with verticals
        let lines = vec![
            make_hline(500.0, 100.0, 200.0, 1),
            make_hline(480.0, 100.0, 200.0, 1),
            make_hline(460.0, 100.0, 200.0, 1),
            make_vline(100.0, 460.0, 500.0, 1),
            make_vline(200.0, 460.0, 500.0, 1),
        ];

        let items = vec![
            make_item("a", 110.0, 490.0, 1),
            make_item("b", 110.0, 470.0, 1),
        ];

        let tables = detect_tables_from_lines(&items, &lines, 1);
        assert!(
            tables.is_empty(),
            "Single-column grid should not be a table"
        );
    }
}
