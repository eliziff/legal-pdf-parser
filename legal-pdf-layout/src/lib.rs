//! A page's layout: its geometry, columns, languages, tables, reading order, note-band lines and
//! regions, each in its own module; the line geometry they share is here.

use legal_pdf_structure_model::{PageView, Slot};
use legal_structure_model::text::{is_note_symbol, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, median, union_bbox, Diagnostic, Line, Page, Region};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderRepair {
    None,
    Column,
    Geometry,
}

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

#[derive(Debug, Clone, Copy)]
pub struct OrderDecision {
    pub repair: OrderRepair,
    source_switches: usize,
    strategy: &'static str,
    pub reason: &'static str,
}

impl OrderDecision {
    const fn keep(reason: &'static str) -> Self {
        Self {
            repair: OrderRepair::None,
            source_switches: 0,
            strategy: "kraken-native",
            reason,
        }
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
