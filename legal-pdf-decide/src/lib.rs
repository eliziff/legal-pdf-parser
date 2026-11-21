//! What each line of a document is, decided once over one table of evidence. The pages are
//! normalized and measured (`evidence`: each line's slot, its source's tag, its type, marker, note
//! label, reference tokens, words and citations; each page's items, columns, rows and separator;
//! the document's bookmarks and instrument fact), then fenced and ordered (`fences`: line labels,
//! folios and furniture, contents lists, tables, the body's type, the reading contexts, quotations,
//! the reading order and each line's continuation of the one before). Each reading context's
//! numbering runs are read and resolved (`runs`), and each document's note band is read against
//! them (`notes`). Each line's readings are then heard (`rules`), the lines no rule settled weighed
//! for headings (`residual`); one walk decides and groups each line (`tier0::settle`), parting blocks as
//! `write::Paragraphs` reads them; the document's outline is found on its lines and every heading
//! levelled (`levels`), and the decision written to the pages once (`write`).

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as pdf_rules;
use legal_pdf_rules::text;
use crate as decide;
use legal_pdf_graph as graph;

pub mod rules;
pub use rules::*;
pub mod contents;
pub mod grammar;
pub mod levels;
pub mod tier0;
pub mod residual;
pub mod write;
pub use write::*;
pub mod trace;
pub use trace::*;

/// The decide stage's ways of computing the contract's types, for a file that calls them.
pub mod find {
    pub use super::rules::{fill, finish, hear, hear_weighed, heard_joined, join, ladder, margin_caption, marginal_notes, open, read_runs, title};
}
