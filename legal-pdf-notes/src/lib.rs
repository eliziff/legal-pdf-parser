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

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;
use legal_pdf_rules::text;
use legal_pdf_layout as layout;

pub mod band;
pub mod lanes;
pub mod note_context;
pub mod read;

/// The notes stage's ways of computing the contract's types, for a file that calls them.
pub mod find {
    pub use super::read::{read, segment, settle_kinds};
}
