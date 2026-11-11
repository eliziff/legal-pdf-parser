//! The text a PDF's lines hold as the structure stage reads it: a paragraph's measure (`paragraphs`), the numbering a
//! heading opens with (`heading_ladder`), the words of text and labels (`text`), lines joined into text (`joining`),
//! each line's place in the document's text (`text_index`) and a transcript's line numbers (`line_numbers`).

pub(crate) mod heading_ladder;
pub(crate) mod joining;
pub(crate) mod line_numbers;
pub(crate) mod paragraphs;
pub(crate) mod text;
pub(crate) mod text_index;
