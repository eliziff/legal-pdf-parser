//! Where a line is read: its constituent document, its translation, its quotation.

/// A document or embedded reading and its parallel sequence; primary text is (0, false).
pub type DocumentContext = (usize, bool);

/// A reading context: a constituent document of the packet (0 for its own), its parallel
/// translation or not, and a quotation in it (0 for none).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Context {
    pub document: u16,
    pub translation: bool,
    pub quotation: u16,
}
