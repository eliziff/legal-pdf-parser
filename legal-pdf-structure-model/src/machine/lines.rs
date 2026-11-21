//! Each line as the tiers read it: its readings, marks and features (`LineTable`), the features' vocabulary DOC's
//! arrays are laid out in, and a marker's family as the features count it.

use crate::MarkerReading;
use std::collections::HashMap;

/// The names of a line's own features, in order.
pub const LINE_FEATURES: &[&str] = &[
    "x0",
    "y0",
    "x1",
    "y1",
    "height",
    "gap_above",
    "gap_below",
    "indent",
    "right_gap",
    "width",
    "centre",
    "size",
    "bold",
    "italic",
    "superscript",
    "monospace",
    "size_change",
    "font_change",
    "underlined",
    "length",
    "upper",
    "digits",
    "punctuation",
    "spaces",
    "starts_digit",
    "starts_upper",
    "starts_lower",
    "starts_open",
    "starts_mark",
    "ends_period",
    "ends_colon",
    "ends_comma",
    "ends_hyphen",
    "ends_digit",
    "ocr",
    "confidence",
    "page_quality",
    "page_position",
    "line_position",
    "first_on_page",
    "last_on_page",
    "page_offgrid",
    "page_measured",
    "style_share",
    "look_share",
    "look_lines",
    "look_pages",
    "look_length",
    "size_rank",
    "doc_size",
    "body_style",
    "ink",
    "ink_measured",
    "weight",
    "col_width",
    "col_centre",
    "label",
    "label_raised",
    "label_value",
    "label_size",
    "label_suppressed",
    "reference_superscript",
    "reference_detached",
    "reference_raised",
    "reference_glued",
    "title_shaped",
    "beside",
    "line_number",
    "edge_repeated",
    "cell",
    "below_separator",
    "translation",
    "marker",
    "marker_word",
    "marker_value",
    "marker_prefix",
    "marker_two",
    "marker_only",
    "count_1",
    "count_1.1",
    "count_1.1.1",
    "count_1.1.1.1",
    "count_a",
    "count_A",
    "count_i",
    "count_I",
    "count_bullet",
    "wrap_()",
    "wrap_[]",
    "wrap_)",
    "wrap_.",
    "wrap_",
];

/// The lexicon's word bits (`WORD_*`) and the look's (`LOOK_*`) follow the named features.
pub const WORD_BITS: usize = 32;

pub const LOOK_BITS: usize = 4;

pub const WIDTH: usize = LINE_FEATURES.len() + WORD_BITS + LOOK_BITS;

/// The document's shape, four measures: the share of its lines opening on a marker, the share a note label opens, whether
/// it is an instrument, and its lines' count (log). What DOC reads of the lines around each line it reads off their own
/// features (its context graph), not a copy of them on every line.
pub const SHAPE: usize = 4;

/// What t0 decided of a line: its role, one of `ROLES`, twelve flags and its heading's level (`docparse.T0`).
pub const T0: usize = ROLES + 13;

/// The roles `Role` names (`docparse.T0`).
pub const ROLES: usize = 14;

const COUNTS: [&str; 9] = ["1", "1.1", "1.1.1", "1.1.1.1", "a", "A", "i", "I", "bullet"];

const WRAPS: [&str; 5] = ["()", "[]", ")", ".", ""];

/// A reading's family as the line features read it: whether a division's word opens it ("Section:():1"), and its
/// count's and wrap's places among `COUNTS` and `WRAPS`.
pub fn marker_family(reading: &MarkerReading) -> (bool, Option<usize>, Option<usize>) {
    let family = reading.family.as_str();
    let (word, plain) = if reading.worded() {
        family.split_once(':').unwrap_or(("", family))
    } else {
        ("", family)
    };
    let (wrap, count) = plain.rsplit_once(':').unwrap_or(("", plain));
    (
        !word.is_empty(),
        COUNTS.iter().position(|c| *c == count),
        WRAPS.iter().position(|c| *c == wrap),
    )
}

/// Each line's readings and features, in reading order (`fences.order`): `WIDTH` own features per
/// line.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct LineTable {
    pub readings: Vec<Vec<MarkerReading>>,
    /// Each line's reference marks: value, first and past-last UTF-16 code unit (`evidence::references`).
    pub references: Vec<Vec<(u32, u32, u32)>>,
    /// Each line's look: its face, half-point size, bold, italic and capitals, as one number.
    pub looks: Vec<u32>,
    /// Each paragraph number the margin prints (`Fences::read_margin_numbers`), by its line's place: the place of
    /// the row's line it numbers, and of its twin's in the other language's column, where it has one.
    pub margin_rows: HashMap<usize, (usize, Option<usize>)>,
    /// Each line a margin number numbers (a row's or a twin's), and each margin number's own line, by its place: the
    /// margin line's place.
    pub margins: HashMap<usize, usize>,
}

/// What the walk measures over the line table, in reading order: each line's own features (`WIDTH` a line), the
/// document's shape (`SHAPE`) and what t0 decided of each line (`T0` a line), the DOC reader's input.
pub struct Measures {
    pub own: Vec<f32>,
    pub shape: [f32; SHAPE],
    pub t0: Vec<u8>,
}

impl Measures {
    /// The line's features.
    pub fn row(&self, k: usize) -> &[f32] {
        &self.own[k * WIDTH..(k + 1) * WIDTH]
    }
}
