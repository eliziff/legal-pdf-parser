//! A PDF's headings as the structure stage reads them: the predicates on a line's text and its place among its page's
//! lines (`native`: a sentence carried on, a prose line going on, a date, the heading tags a source overuses), the words
//! of headings and labels and the numbering of headings (`rules`), the document's bookmarks (`bookmarks`), its printed
//! contents list read as an outline (`contents`), its contents lists (`contents_lists`) and captions (`captions`).

pub(crate) mod bookmarks;
pub(crate) mod captions;
pub(crate) mod contents;
pub(crate) mod contents_lists;
pub(crate) mod native;
pub(crate) use native::*;
pub(crate) mod rules;
