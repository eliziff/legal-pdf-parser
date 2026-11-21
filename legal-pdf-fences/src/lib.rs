//! A PDF's lines fenced and ordered: furniture and folios, columns, contents lists, tables, the note band, the body's
//! type, the constituent documents, the reading order, quotations and reading contexts, each line's continuation
//! (`fences`), and the judgment's numbered paragraphs (`paragraphs`).

// The crates this stage reads, by the names its paths use.
use legal_pdf_structure_model as model;
use legal_pdf_rules as rules;
use legal_pdf_furniture as furniture;
use legal_pdf_constituents as constituents;
use legal_pdf_headings as headings;
use legal_pdf_layout as layout;
use legal_pdf_tables as tables;
use legal_pdf_rules::text;

pub mod fences;
pub mod paragraphs;

/// The fences stage's ways of finding the fences, for a file that calls them.
pub mod find {
    pub use super::fences::{read, read_numbers, recontext, set};
    pub use super::fences::band::{clear_false_bands, free_noted_cells, free_referenced_note_cells, read_body_type, read_note_band};
    pub use super::fences::columns::{read_columns, read_translation};
    pub use super::fences::contents::{read_contents};
    pub use super::fences::continuations::{confirm_section_heads, read_continuations};
    pub use super::fences::documents::{read_documents, read_transcripts};
    pub use super::fences::furniture::{folio, read_document_furniture, read_furniture, read_page_numbers};
    pub use super::fences::order::{read_margin_labels, read_margin_numbers, read_order};
    pub use super::fences::quotations::{read_quotations};
    pub use super::fences::tables::{find_tables, free_cell, free_spoken_cells, number_cells, prune_tables, read_tables};
    pub use super::paragraphs::{below_body, body_line, label_size, marker_rows, note_label, read_paragraphs, set_apart, streams};
}

/// A judgment's own paragraph count over its text: each number with its paragraph's ranges (legal-structure's
/// `case_outline::judgment_paragraphs`, which the composer supplies, so the fences compile without the structure engine).
pub type JudgmentParagraphs<'a> = &'a dyn Fn(&str) -> Vec<(usize, Vec<legal_structure_model::ScalarRange>)>;
