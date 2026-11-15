//! A document's page map (`legal_pdf_pages::page_map`) from what its pages print: the same fit the structure stage
//! numbers the pages by, over the same evidence, here with the first pages of the reports a caller cites.
use crate::PdfDocument;
pub use legal_pdf_pages::{EdgeLine, PageEvidence, PdfPageBinding};

/// What each page of `document` tells of its number (`legal_pdf_structure::page_evidence`): the PDF's own label, the
/// page's size and its lines, a transcript's line numbers (the lines held out of the body) left blank.
pub fn page_evidence(document: &PdfDocument) -> Vec<PageEvidence> {
    let pages = document.passage_pages();
    legal_pdf_structure::page_evidence(&pages, &document.summary().embedded_page_labels, |page, at| pages[page].lines[at].exclude_from_body)
}

/// The document's page map: `starts` the first pages of the citations a caller holds to the report it prints,
/// `is_citation` whether a line's text cites a report at the page it would read as a folio.
pub fn page_map(document: &PdfDocument, starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) -> Vec<PdfPageBinding> {
    legal_pdf_pages::page_map(&page_evidence(document), starts, is_citation)
}
