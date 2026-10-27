//! Which lines are notes, and of what kind. The readers run in this order, each from the
//! structure stage where its evidence is ready:
//!
//! 1. `band` (from page classification, per document and language): each page's note band, page by
//!    page, with an endnote sequence carried across pages; then `lines::verify_endnotes` gives back
//!    endnote runs the body never refers to.
//! 2. `lanes::correct_note_regions` (after classification, per document): the whole document's
//!    witnesses (a printed separator's band, the note label ladder, font roles, a note running on
//!    to the next page) correct the band, each change backed by two of them.
//! 3. `lines::infer_note_region_modes` (once headings are read): each note line's mode.
//!
//! `lines` also holds what the readers see of a single line: a notes heading, a label's
//! typography, a reference wrapped onto the next line.

use super::*;

mod band;
pub(crate) use band::*;
pub(crate) mod lanes;
mod lines;
pub(crate) use lines::*;
