/// What kind of structure a detected table represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TableKind {
    #[default]
    Data,
    Toc,
}

/// A table recovered from PDF geometry or aligned text.
#[derive(Debug, Clone)]
pub struct Table {
    pub columns: Vec<f32>,
    pub rows: Vec<f32>,
    pub cells: Vec<Vec<String>>,
    pub item_indices: Vec<usize>,
    pub kind: TableKind,
}

impl Table {
    pub fn new(
        columns: Vec<f32>,
        rows: Vec<f32>,
        cells: Vec<Vec<String>>,
        item_indices: Vec<usize>,
    ) -> Self {
        Self {
            columns,
            rows,
            cells,
            item_indices,
            kind: TableKind::Data,
        }
    }
}

/// Threshold policy used by aligned-text table detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TableDetectionMode {
    SmallFont,
    BodyFont,
}

use pdf_inspector_core::types::TextItem;

/// Deduplicate nearby edge values within a tolerance, returning sorted values.
pub fn snap_edges(values: &[f32], tolerance: f32) -> Vec<f32> {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut snapped: Vec<f32> = Vec::new();
    for value in sorted {
        if snapped
            .last()
            .is_some_and(|last| (value - *last).abs() <= tolerance)
        {
            continue;
        }
        snapped.push(value);
    }
    snapped
}

/// Assign page text to cells defined by column and descending row edges.
pub fn assign_items_to_grid(
    items: &[TextItem],
    col_edges: &[f32],
    row_edges: &[f32],
    page: u32,
) -> (Vec<Vec<String>>, Vec<usize>) {
    let num_cols = col_edges.len() - 1;
    let num_rows = row_edges.len() - 1;
    let mut cell_items = vec![vec![Vec::<(usize, &TextItem)>::new(); num_cols]; num_rows];
    let mut indices = Vec::new();
    for (index, item) in items
        .iter()
        .enumerate()
        .filter(|(_, item)| item.page == page)
    {
        let cx = item.x + item.width / 2.0;
        let cy = item.y;
        let col = (0..num_cols).find(|&c| cx >= col_edges[c] - 2.0 && cx <= col_edges[c + 1] + 2.0);
        let row = (0..num_rows).find(|&r| cy >= row_edges[r + 1] - 2.0 && cy <= row_edges[r] + 2.0);
        if let (Some(col), Some(row)) = (col, row) {
            cell_items[row][col].push((index, item));
            indices.push(index);
        }
    }
    let cells = cell_items
        .iter_mut()
        .map(|row| {
            row.iter_mut()
                .map(|items| {
                    items.sort_by(|a, b| {
                        b.1.y
                            .partial_cmp(&a.1.y)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                a.1.x
                                    .partial_cmp(&b.1.x)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                    });
                    let text = items
                        .iter()
                        .map(|(_, item)| item.text.trim())
                        .filter(|text| !text.is_empty())
                        .collect::<Vec<_>>()
                        .join(" ");
                    remove_inner_delimiter_spaces(&text)
                })
                .collect()
        })
        .collect();
    (cells, indices)
}

fn remove_inner_delimiter_spaces(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut result = String::with_capacity(text.len());
    for (index, character) in chars.iter().copied().enumerate() {
        if character == ' '
            && (matches!(result.chars().last(), Some('(' | '[' | '{'))
                || chars
                    .get(index + 1)
                    .is_some_and(|next| matches!(next, ')' | ']' | '}')))
        {
            continue;
        }
        result.push(character);
    }
    result
}
