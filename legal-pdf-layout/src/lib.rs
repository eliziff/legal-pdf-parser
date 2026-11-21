//! A page's layout: its geometry, columns, languages, reading order and note-band lines,
//! each in its own module; the line geometry they share is here.

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;

pub mod columns;
pub use columns::*;
pub mod order;
pub use order::*;
pub mod note_lines;
pub use note_lines::*;
