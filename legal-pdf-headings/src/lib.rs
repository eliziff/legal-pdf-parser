//! A PDF's headings as the structure stage reads them: the predicates on a line's text and its place among its page's
//! lines (`native`: a sentence carried on, a prose line going on, a date, the heading tags a source overuses), the words
//! of headings and labels and the numbering of headings (`rules`), the document's bookmarks (`bookmarks`), its printed
//! contents list read as an outline (`contents`), its contents lists (`contents_lists`) and captions (`captions`).

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;
use crate as headings;

pub use legal_pdf_rules::headings::bookmarks;
pub mod contents_lists;
