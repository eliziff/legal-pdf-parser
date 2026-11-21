//! A page's tables as the fences keep them: their rows, cells and cells' parts.

use std::collections::HashSet;

/// A line of a cell, or the part of it that stands in the cell when the line runs on
/// across a column into the next cell.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CellPart {
    pub line_id: String,
    /// The part's UTF-16 offsets within its line's text.
    pub chars: Option<(usize, usize)>,
    pub text: String,
    pub bbox: [f64; 4],
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
    pub fn text(&self) -> String {
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
    pub page_index: usize,
    pub xs: Vec<f64>,
    pub top: f64,
    pub bottom: f64,
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
}
