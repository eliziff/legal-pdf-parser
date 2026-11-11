//! The document's numbered paragraphs, by line: the number a line opens and what each number holds of it.

/// A numbered paragraph's hold on a line: its language, number and UTF-16 span of the line's text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct Held {
    pub translation: bool,
    pub number: u32,
    pub start: u32,
    pub end: u32,
}

/// The document's numbered paragraphs, by line: the number a line opens and what each number holds of it. Read once
/// before any other reading claims a line, and fixed: nothing after overrules it.
#[derive(Debug, Default, serde::Serialize)]
pub struct Paragraphs {
    pub count: Vec<Option<u32>>,
    pub held: Vec<Vec<Held>>,
}
