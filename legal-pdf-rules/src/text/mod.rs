//! The text a PDF's lines hold as the structure stage reads it: a paragraph's measure (`paragraphs`), the numbering a
//! heading opens with (`heading_ladder`), the words of text and labels (`text`), lines joined into text (`joining`),
//! each line's place in the document's text (`text_index`).

pub mod heading_ladder;
pub mod joining;
pub mod paragraphs;
pub mod text;
pub mod text_index;

use legal_structure_model::utf16_len;

/// The figures a line's text opens on: what they read, and their first and past-last UTF-16 code unit.
pub fn leading_figures(text: &str) -> Option<(u32, (usize, usize))> {
    let start = text.find(|c: char| !c.is_whitespace())?;
    let length = text[start..].bytes().take_while(u8::is_ascii_digit).count();
    let at = utf16_len(&text[..start]);
    Some((text[start..start + length].parse().ok()?, (at, at + length)))
}
