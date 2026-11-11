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

use crate::model::{Member, PageView, Role, Rule, Run, Runs, Slot, Slots};
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfOutlineEntry};
use crate::headings::native::date_line;
use crate::headings;
use crate::notes::{
    endnote_heading, read_band, separator_tolerance, verified_endnotes, Document, NoteKind,
};
use crate::text::heading_ladder::{enumerator_interpretations, parse_heading_runs, EnumeratorInterpretation};
use crate::text::text::{all_caps, enumerated};
use legal_pdf_core::model::sentence_ended;
use legal_structure_model::text::{label_prefix, line_start_label_prefix, normalize_label};
use pdf_inspector::markdown::has_dot_leaders;
use legal_pdf_core::regex;
use std::collections::{BTreeMap, HashMap};

use crate::model::*;
use crate::layout::Geometry;
use crate::fences::*;
pub(crate) mod notes;
pub(crate) use notes::*;
pub(crate) mod rules;
pub(crate) use rules::*;
pub(crate) mod residual;
pub(crate) use residual::*;
pub(crate) mod levels;
pub(crate) use levels::*;
pub(crate) mod write;
pub(crate) use write::*;
pub(crate) mod trace;
pub(crate) use trace::*;
