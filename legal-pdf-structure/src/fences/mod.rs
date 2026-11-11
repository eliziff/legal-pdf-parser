//! A PDF's lines fenced and ordered: furniture and folios, columns, contents lists, tables, the note band, the body's
//! type, the constituent documents, the reading order, quotations and reading contexts, each line's continuation
//! (`fences`), and the judgment's numbered paragraphs (`paragraphs`).

pub(crate) mod fences;
pub(crate) use fences::*;
pub(crate) mod paragraphs;

