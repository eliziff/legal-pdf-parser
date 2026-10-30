//! A page's layout: its geometry, columns, languages, tables, reading order, note-band lines and
//! regions, each in its own module; the line geometry they share is here.

use crate::structure::{is_note_symbol, PageView, Slot, MAX_SYMBOL_LABEL_LEN};
use legal_pdf_core::{line_font_size, median, union_bbox, Diagnostic, Line, Page, Region};
use regex::Regex;
use std::{
    collections::{HashMap, HashSet},
    sync::OnceLock,
};

mod tables;
pub(crate) use tables::*;
mod columns;
pub(crate) use columns::*;
mod language;
pub(crate) use language::*;
mod order;
pub(crate) use order::*;
mod note_lines;
pub(crate) use note_lines::*;
mod regions;
pub(crate) use regions::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OrderRepair {
    None,
    Column,
    Geometry,
}

#[derive(Debug, Clone, Copy)]
pub(super) struct ColumnModel {
    pub(super) kind: &'static str,
    pub(super) split_x: f64,
    pub(super) left_count: usize,
    pub(super) right_count: usize,
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
pub(super) struct OrderDecision {
    pub(super) repair: OrderRepair,
    source_switches: usize,
    strategy: &'static str,
    pub(super) reason: &'static str,
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

pub(super) fn line_width(line: &Line) -> f64 {
    line.bbox[2] - line.bbox[0]
}

pub(super) fn has_valid_bbox(line: &Line) -> bool {
    line.bbox.iter().all(|value| value.is_finite())
        && line.bbox[2] > line.bbox[0]
        && line.bbox[3] > line.bbox[1]
}

pub(super) fn line_center_x(line: &Line) -> f64 {
    (line.bbox[0] + line.bbox[2]) / 2.0
}

fn line_center_y(line: &Line) -> f64 {
    (line.bbox[1] + line.bbox[3]) / 2.0
}
