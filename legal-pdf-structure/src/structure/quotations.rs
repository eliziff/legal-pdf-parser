//! Quotations: another text's words a document sets apart from its own, each a reading context of
//! its own with its own headings, numbered paragraphs and lists. The scopes come from the
//! quoted-paragraph sequence reader (a run of numbered paragraphs foreign to the document's own
//! sequence); the structure tagger reads display quotations from the lines' evidence. (The
//! typographic display-quotation reader that stood here read as null under the tagger on the
//! structure gold's development split and went.)

use super::PdfPrimitiveEvidence;
use legal_pdf_core::model::Page;
use std::collections::HashMap;

#[derive(Debug, Default, serde::Serialize, serde::Deserialize)]
pub(super) struct Reading {
    pub(super) scopes: HashMap<String, String>,
    pub(super) introductions: HashMap<String, String>,
}

/// Quotations keep their own reading; physical document identities remain unchanged.
pub(super) fn reading_lines(
    pages: &[Page],
    primitives: &PdfPrimitiveEvidence,
) -> HashMap<String, usize> {
    if primitives.quotations.scopes.is_empty() {
        return primitives.contexts.document_line_ids.clone();
    }
    let mut next = primitives
        .contexts.document_line_ids
        .values()
        .copied()
        .max()
        .unwrap_or(0)
        + 1;
    let mut quotes = HashMap::new();
    pages
        .iter()
        .flat_map(|page| &page.lines)
        .map(|line| {
            let document = primitives.quotations.scopes.get(&line.id).map_or_else(
                || primitives.contexts.document_context(&line.id).0,
                |quote| {
                    *quotes.entry(quote).or_insert_with(|| {
                        let id = next;
                        next += 1;
                        id
                    })
                },
            );
            (line.id.clone(), document)
        })
        .collect()
}
