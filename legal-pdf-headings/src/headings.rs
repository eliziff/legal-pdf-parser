//! What a document's lines say of its headings, as the decision reads them: the predicates on a
//! line's text and its place among its page's lines (`native`: a sentence carried on, a prose line
//! going on, a date, the heading tags a source overuses), the words of headings and labels and the
//! numbering of headings (`rules`), the document's bookmarks (`bookmarks`) and its printed contents
//! list read as an outline (`contents`).


pub mod bookmarks;
pub mod contents;
pub mod native;
pub use native::*;
pub mod rules;
