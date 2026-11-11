//! A page's layout: its geometry, columns, languages, reading order and note-band lines,
//! each in its own module; the line geometry they share is here.

use crate::model::{PageView, Slot};
use crate::tables::has_table_caption;
use legal_structure_model::text::{is_note_symbol, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, median, Columns, Diagnostic, Line, Page, PageMeasure};
use std::{
    cell::OnceCell,
    collections::{HashMap, HashSet},
};

pub(crate) mod columns;
pub(crate) use columns::*;
pub(crate) mod language;
pub(crate) use language::*;
pub(crate) mod order;
pub(crate) use order::*;
pub(crate) mod note_lines;
pub(crate) use note_lines::*;

/// What a page's columns are: one, two of like width, a body beside a narrower margin's, or more than two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub(crate) enum ColumnKind {
    Single,
    Two,
    Margin,
    Multi,
}

/// A page's columns: their kind, where they split, and the lines on either side.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(crate) struct ColumnModel {
    pub(crate) kind: ColumnKind,
    pub(crate) split_x: f64,
    pub(crate) left_count: usize,
    pub(crate) right_count: usize,
}

pub(crate) fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

pub(crate) fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        // (A PDF's font may give its lines no height: 2001 SCC 79's every line stands on one y.)
        && line.bbox[3] >= line.bbox[1]
}

pub(crate) fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

pub(crate) fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}

/// Whether a page layout model labelled the line: its region is one of the model's.
pub(crate) fn layout_labelled(line: &Line) -> bool {
    line.region_id.contains("-ppdoc-r")
}

/// Each page's measures and columns, read once, when a stage first asks for them, and kept for every stage that reads
/// the pages after (the document's, made once where its stages are called, `legal_pdf_structure`): what they measure
/// (the lines' boxes, sizes and text) stands until the lines are written.
pub(crate) struct Geometry {
    read: Vec<OnceCell<(PageMeasure, Columns)>>,
}

impl Geometry {
    pub(crate) fn of(pages: &[Page]) -> Self {
        Self { read: (0..pages.len()).map(|_| OnceCell::new()).collect() }
    }

    /// The page's measures (`PageMeasure`) and columns (`Columns`), by page slot: `pages[page]`'s.
    pub(crate) fn page(&self, pages: &[Page], page: usize) -> &(PageMeasure, Columns) {
        self.read[page].get_or_init(|| (PageMeasure::of(&pages[page]), Columns::of(&pages[page])))
    }
}
