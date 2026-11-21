//! A page's geometry: its columns as layout reads them, its measures read once for every stage, and the line geometry
//! every stage shares.

use legal_pdf_core::{Columns, Line, Page, PageMeasure};
use std::cell::OnceCell;

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

pub fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

/// Each page's measures and columns, read once, when a stage first asks for them, and kept for every stage that reads
/// the pages after (the document's, made once where its stages are called, `legal_pdf_structure::legal_pdf_structure`): what they measure
/// (the lines' boxes, sizes and text) stands until the lines are written.
pub struct Geometry {
    read: Vec<OnceCell<(PageMeasure, Columns)>>,
}

impl Geometry {
    pub fn of(pages: &[Page]) -> Self {
        Self {
            read: (0..pages.len()).map(|_| OnceCell::new()).collect(),
        }
    }

    /// The page's measures (`PageMeasure`) and columns (`Columns`), by page slot: `pages[page]`'s.
    pub fn page(&self, pages: &[Page], page: usize) -> &(PageMeasure, Columns) {
        self.read[page].get_or_init(|| (PageMeasure::of(&pages[page]), Columns::of(&pages[page])))
    }
}

/// The stream of a page a line reads in: its running head, its body, its note band or its foot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Header,
    Body,
    Note,
    Footer,
}
