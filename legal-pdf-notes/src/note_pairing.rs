//! A document's notes, paired within each of its constituent documents and languages.

use legal_pdf_structure_model::Decided;
use legal_pdf_text::text_index::PdfTextIndex;
use legal_pdf_core::model::{Page, ScalarRange};
use legal_structure_model::NotePairing;
use std::collections::BTreeMap;

/// The document's notes paired on the pages as the decision wrote them (their regions and note modes), each constituent
/// document and language on its own lines, in their order, each note a node placed in the document's text. A note's id
/// is the document's, and its numbering's restarts count on across the documents.
pub fn pair_notes(pages: &[Page], decided: &Decided, citations: &BTreeMap<String, Vec<ScalarRange>>, index: &PdfTextIndex) -> NotePairing {
    let place = |id: &str, start: usize, end: usize| index.global_range(id, start, end);
    let part = |line: &legal_pdf_core::model::Line| {
        let (document, translation) = decided.document_context(&line.id);
        2 * document + usize::from(translation)
    };
    legal_structure_model::structure_analysis().pair_footnotes(pages, citations, &part, &place)
}
