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

use legal_pdf_structure_model::{Member, PageView, Role, Rule, Run, Runs, Slot, Slots};
use legal_pdf_core::model::{Diagnostic, Line, Page, PdfOutlineEntry};
use legal_pdf_headings::headings::native::date_line;
use legal_pdf_headings::headings;
use legal_pdf_layout::layout_labelled;
use legal_pdf_notes::notes::{
    endnote_heading, read_band, separator_tolerance, verified_endnotes, BandLines, BandPage, NoteKind, BAND_CELL, BAND_EXCLUDED, BAND_FURNITURE, BAND_HEADED,
    BAND_LAYOUT, BAND_NOTED, BAND_PARAGRAPH,
};
use legal_pdf_text::heading_ladder::{enumerator_interpretations, parse_heading_runs, EnumeratorInterpretation};
use legal_pdf_text::text::{all_caps, enumerated, sentence_ended};
use legal_structure_model::text::{label_prefix, line_start_label_prefix, normalize_label};
use pdf_inspector::markdown::has_dot_leaders as has_dot_leader;
use legal_pdf_core::regex;
use std::collections::{BTreeMap, HashMap};

use legal_pdf_evidence::*;
use legal_pdf_fences::*;
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
