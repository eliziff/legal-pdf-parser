//! A PDF's headings as the structure stage reads them: the predicates on a line's text and its place among its page's
//! lines (`native`: a sentence carried on, a prose line going on, a date, the heading tags a source overuses), the words
//! of headings and labels and the numbering of headings (`rules`), the document's bookmarks (`bookmarks`), its printed
//! contents list read as an outline (`contents`), its contents lists (`contents_lists`) and captions (`captions`).

pub mod bookmarks;
pub mod captions;
pub mod contents;
pub mod contents_lists;
pub mod native;
pub use native::*;
pub mod rules;
