//! What each line of a document is, decided once over one table of evidence. The pages are
//! normalized and measured (`evidence`: each line's slot, its source's tag, its type, marker, note
//! label, reference tokens, words and citations; each page's items, columns, rows and separator;
//! the document's bookmarks and instrument fact), then fenced and ordered (`fences`: line labels,
//! folios and furniture, contents lists, tables, the body's type, the reading contexts, quotations,
//! the reading order and each line's continuation of the one before). Each reading context's
//! numbering runs are read and resolved (`runs`), and each document's note band is read against
//! them (`notes`). Each line's readings are then heard (`rules`), the lines no rule settled weighed
//! for headings (`residual`); the machine takes one decision on each line (`transition::tier0`); the
//! document's outline is found on its lines and every heading levelled (`levels`), and the decision
//! written to the pages once (`write`).

use legal_pdf_constituents as constituents;
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfOutlineEntry};
use legal_pdf_core::{flag_share, layout_analysis, line_font_size, median, Error, Result};
use legal_pdf_furniture::{compact_note_line, page_labels_among, Edge};
use legal_pdf_headings::headings::native::{body_flow_edge, bold_char_share, carries_a_sentence_onto, continues_prose_from, date_line};
use legal_pdf_headings::{captions, contents_lists, headings};
use legal_pdf_layout::{columns_of, layout_labelled, page_order, translation_lines, ColumnModel, Stream, TableReadings};
use legal_pdf_notes::notes::{
    endnote_heading, label_is_raised, label_spans, read_band, reference_wrapped_onto, separator_beside_tables, separator_tolerance,
    structural_reset_heading, verified_endnotes, BandLines, BandPage, NoteKind, BAND_CELL, BAND_EXCLUDED, BAND_FURNITURE, BAND_HEADED,
    BAND_LAYOUT, BAND_NOTED, BAND_PARAGRAPH,
};
use legal_pdf_text::heading_ladder::{enumerator_interpretations, parse_heading_runs, EnumeratorInterpretation};
use legal_pdf_text::text::{all_caps, enumerated, party_label, sentence_ended};
use legal_structure_model::text::{is_note_symbol, label_prefix, line_start_label_prefix, normalize_label};
use legal_structure_model::{ResolutionRuleV2, StructureAnalysis};
use pdf_inspector::markdown::has_dot_leaders as has_dot_leader;
use regex::Regex;
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::OnceLock;

pub mod evidence;
pub use evidence::*;
pub mod paragraphs;
pub use paragraphs::{Held, Paragraphs};
pub mod fences;
pub use fences::*;
pub mod runs;
pub use runs::*;
pub mod notes;
pub use notes::*;
pub mod rules;
pub use rules::*;
pub mod residual;
pub use residual::*;
pub mod levels;
pub use levels::*;
pub mod write;
pub use write::*;
pub mod trace;
pub use trace::*;
