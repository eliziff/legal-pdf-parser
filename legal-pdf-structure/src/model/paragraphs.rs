//! The document's numbered paragraphs, by line: the number a line opens and what each number holds of it.

/// A numbered paragraph's hold on a line: its language, number and UTF-16 span of the line's text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct Held {
    pub(crate) translation: bool,
    pub(crate) number: u32,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

/// The document's numbered paragraphs, by line: the number a line opens and what each number holds of it. Read once
/// before any other reading claims a line, and fixed: nothing after overrules it.
#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(crate) struct Paragraphs {
    pub(crate) count: Vec<Option<u32>>,
    pub(crate) held: Vec<Vec<Held>>,
}
