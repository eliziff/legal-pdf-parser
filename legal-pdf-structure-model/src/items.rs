//! A page's words as its table readers read them.

use pdf_inspector::types::TextItem;

/// Lines' words as pdf-inspector's text items, with the (line, word) each was read from.
#[derive(Debug)]
pub struct PageItems {
    pub items: Vec<TextItem>,
    /// The line's slot and the word's index for each item; `WHOLE` for a line read whole.
    pub origin: Vec<(usize, usize)>,
}

/// A word no word of the line stands for: the line read whole.
pub const WHOLE: usize = usize::MAX;
