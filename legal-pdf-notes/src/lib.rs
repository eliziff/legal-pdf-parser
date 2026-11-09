//! A PDF's notes as the structure stage reads them. Which lines are notes, and of what kind: the note band and its
//! lanes read a document's lines and write nothing; each reads per document and language (`decide::Notes` runs them):
//!
//! 1. `band::read_band`: each page's note band, page by page, with an endnote sequence carried across pages; then the
//!    endnote runs the body never refers to given back.
//! 2. `lanes::note_promotions`: the whole document's witnesses (a printed separator's band, the note label ladder, font
//!    roles, a note running on to the next page) take more lines for notes beside the band's, each backed by two of
//!    them.
//!
//! `lines` holds what they see of a single line: a notes heading, a label's typography, a reference wrapped onto the
//! next line, a reference above a label. Each note's propositions and cross-references are `note_context`'s.

use legal_pdf_text::text::sentence_ended;

pub mod band;
pub use band::*;
pub mod lanes;
pub mod lines;
pub use lines::*;
pub mod note_context;
