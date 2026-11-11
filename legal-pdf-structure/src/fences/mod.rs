//! A PDF's lines fenced and ordered: furniture and folios, columns, contents lists, tables, the note band, the body's
//! type, the constituent documents, the reading order, quotations and reading contexts, each line's continuation
//! (`fences`), and the judgment's numbered paragraphs (`paragraphs`).

pub mod fences;
pub use fences::*;
pub mod paragraphs;

use crate::evidence::*;
pub use crate::model::Context;
use crate::model::{Held, PageView, Paragraphs, Role, Rows, Rule, Slot, Slots};
use crate::constituents as constituents;
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfOutlineEntry};
use crate::furniture::{page_labels_among, Edge};
use crate::headings::native::{body_flow_edge, carries_a_sentence_onto, continues_prose_from};
use crate::headings::contents_lists;
use crate::layout::{columns_of, page_order, translation_lines, ColumnModel, Stream};
use crate::notes::{
    reference_wrapped_onto, separator_beside_tables, separator_tolerance,
};
use crate::text::text::sentence_ended;
use std::collections::{BTreeMap, HashMap, HashSet};
