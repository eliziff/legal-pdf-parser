//! The machine's types every tier reads: each line as the state reads it and the moves it makes (`machine`),
//! and each line's readings, marks and features (`lines`).

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;

pub mod lines;

/// The machine stage's ways of computing the contract's types, for a file that calls them.
pub mod find {
    pub use super::lines::{measure, plain};
}
