//! What each line of a document is, decided once over one table of evidence. The pages are
//! normalized and measured (`evidence`: each line's slot, its source's tag, its type, marker, note
//! label, reference tokens, words and citations; each page's items, columns, rows and separator;
//! the document's bookmarks and instrument fact), then fenced and ordered (`fences`: line labels,
//! folios and furniture, contents lists, tables, the body's type, the reading contexts, quotations,
//! the reading order and each line's continuation of the one before). Each reading context's
//! numbering runs are read and resolved (`runs`), and each document's note band is read against
//! them (`notes`). Each line's role and the unit each marker opens are then decided (`rules`), the
//! lines no rule settled weighed for headings (`residual`), the document's outline found on its
//! lines and every heading levelled (`levels`), and the decision written to the pages once
//! (`write`).

use super::*;

mod evidence;
pub(crate) use evidence::*;
mod fences;
pub(crate) use fences::*;
mod runs;
pub(crate) use runs::*;
mod notes;
pub(crate) use notes::*;
mod rules;
pub(crate) use rules::*;
mod residual;
pub(crate) use residual::*;
mod levels;
pub(crate) use levels::*;
mod write;
pub(crate) use write::*;
