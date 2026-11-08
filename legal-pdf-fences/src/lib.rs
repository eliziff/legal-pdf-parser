//! A PDF's lines fenced and ordered: furniture and folios, columns, contents lists, tables, the note band, the body's
//! type, the constituent documents, the reading order, quotations and reading contexts, each line's continuation
//! (`fences`), and the judgment's numbered paragraphs (`paragraphs`).

pub mod fences;
pub use fences::*;
pub mod paragraphs;

use legal_pdf_evidence::*;
pub use legal_pdf_structure_model::Context;
use legal_pdf_structure_model::{Held, PageView, Paragraphs, Role, Rows, Rule, Slot, Slots};
use legal_pdf_constituents as constituents;
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfOutlineEntry};
use legal_pdf_furniture::{page_labels_among, Edge};
use legal_pdf_headings::headings::native::{body_flow_edge, carries_a_sentence_onto, continues_prose_from};
use legal_pdf_headings::contents_lists;
use legal_pdf_layout::{columns_of, page_order, translation_lines, ColumnModel, Stream, TableReadings};
use legal_pdf_notes::notes::{
    reference_wrapped_onto, separator_beside_tables, separator_tolerance,
};
use legal_pdf_text::text::sentence_ended;
use std::collections::{BTreeMap, HashMap, HashSet};
