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


#[derive(Debug, Clone, Copy)]
pub struct ColumnModel {
    pub kind: &'static str,
    pub split_x: f64,
    pub left_count: usize,
    pub right_count: usize,
}

/// A column model as its kind's name, its split and its two columns' line counts.
impl serde::Serialize for ColumnModel {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serde::Serialize::serialize(&(self.kind, self.split_x, self.left_count, self.right_count), serializer)
    }
}

impl<'de> serde::Deserialize<'de> for ColumnModel {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let (kind, split_x, left_count, right_count): (String, f64, usize, usize) = serde::Deserialize::deserialize(deserializer)?;
        let kind = ["single", "two_column", "margin_column", "multi_column"]
            .into_iter()
            .find(|known| *known == kind)
            .ok_or_else(|| serde::de::Error::custom(format!("unknown column model kind {kind}")))?;
        Ok(Self { kind, split_x, left_count, right_count })
    }
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
