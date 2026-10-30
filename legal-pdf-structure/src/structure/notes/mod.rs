//! Which lines are notes, and of what kind. The note band and its lanes read a document's lines and
//! write nothing; each reads per document and language (`decide::Notes` runs them):
//!
//! 1. `band::read_band`: each page's note band, page by page, with an endnote sequence carried
//!    across pages; then the endnote runs the body never refers to given back.
//! 2. `lanes::note_promotions`: the whole document's witnesses (a printed separator's band, the
//!    note label ladder, font roles, a note running on to the next page) take more lines for notes
//!    beside the band's, each backed by two of them.
//!
//! `lines` holds what they see of a single line: a notes heading, a label's typography, a reference
//! wrapped onto the next line, a reference above a label.

use super::*;

mod band;
pub(crate) use band::*;
pub(crate) mod lanes;
mod lines;
pub(crate) use lines::*;
