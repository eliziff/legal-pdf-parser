//! A page's layout: its geometry, columns, languages, tables, reading order, note-band lines and
//! regions, each in its own module; the line geometry they share is here.

use legal_pdf_structure_model::{PageView, Slot};
use legal_structure_model::text::{is_note_symbol, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, median, regex, union_bbox, Columns, Diagnostic, Line, Page, PageMeasure, Region};
use std::{
    cell::OnceCell,
    collections::{HashMap, HashSet},
};

pub mod tables;
pub use tables::*;
pub mod columns;
pub use columns::*;
pub mod language;
pub use language::*;
pub mod order;
pub use order::*;
pub mod note_lines;
pub use note_lines::*;
pub mod regions;
pub use regions::*;


/// What a page's columns are: one, two of like width, a body beside a narrower margin's, or more than two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ColumnKind {
    Single,
    Two,
    Margin,
    Multi,
}

/// A page's columns: their kind, where they split, and the lines on either side.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct ColumnModel {
    pub kind: ColumnKind,
    pub split_x: f64,
    pub left_count: usize,
    pub right_count: usize,
}

pub fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

pub fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        // (A PDF's font may give its lines no height: 2001 SCC 79's every line stands on one y.)
        && line.bbox[3] >= line.bbox[1]
}

pub fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

/// Whether a page layout model labelled the line: its region is one of the model's.
pub fn layout_labelled(line: &Line) -> bool {
    line.region_id.contains("-ppdoc-r")
}

/// Each page's measures and columns, read once, when a stage first asks for them, and kept for every stage that reads
/// the pages after (the document's, made once where its stages are called, `legal_pdf_structure`): what they measure
/// (the lines' boxes, sizes and text) stands until the lines are written.
pub struct Geometry {
    read: Vec<OnceCell<(PageMeasure, Columns)>>,
}

impl Geometry {
    pub fn of(pages: &[Page]) -> Self {
        Self { read: (0..pages.len()).map(|_| OnceCell::new()).collect() }
    }

    /// The page's measures (`PageMeasure`) and columns (`Columns`), by page slot: `pages[page]`'s.
    pub fn page(&self, pages: &[Page], page: usize) -> &(PageMeasure, Columns) {
        self.read[page].get_or_init(|| (PageMeasure::of(&pages[page]), Columns::of(&pages[page])))
    }
}
