//! A document's page map (`legal_pdf_pages::page_map`) from what its pages print: each page's folio as the structure
//! stage read it, the PDF's own labels, and its opening pages' lines for a citation's first page.
use crate::PdfDocument;
pub use legal_pdf_pages::{EdgeLine, PageEvidence, PdfPageBinding};

/// What each page of `document` tells of its number: its folio as the structure read it, the PDF's own label, and its
/// lines, every one of them: a page's number may stand anywhere its layout puts it (a margin, a turned page), and the fit
/// finds it by the run its pages' numbers make.
pub fn page_evidence(document: &PdfDocument) -> Vec<PageEvidence> {
    let observed = document.page_labels();
    let embedded = &document.summary().embedded_page_labels;
    let read = document.passage_pages();
    (0..observed.len().max(embedded.len()))
        .map(|index| {
            let number = index as u32 + 1;
            let page = read.iter().find(|page| page.number == number);
            let lines = page.map(|page| {
                page.lines.iter()
                    .map(|line| EdgeLine { text: line.text.clone(), rect: line.bbox })
                    .collect()
            }).unwrap_or_default();
            PageEvidence {
                observed: observed.get(index).cloned().flatten(),
                embedded: embedded.get(index).cloned().flatten(),
                width: page.map_or(0.0, |page| page.width),
                height: page.map_or(0.0, |page| page.height),
                lines,
            }
        })
        .collect()
}

/// `page_evidence`, the pages `read_again` names read again in their heads and feet by `ocr` (`legal_pdf_pages::reread`
/// over `legal_pdf_ocr::read_bands`).
#[cfg(feature = "ocr")]
pub fn page_evidence_read(document: &PdfDocument, pdf: &[u8], ocr: &mut legal_pdf_ocr::OcrProvider) -> crate::Result<Vec<PageEvidence>> {
    legal_pdf_pages::reread(page_evidence(document), &mut |indices| legal_pdf_ocr::read_bands(pdf, indices, ocr).map_err(|e| e.to_string()))
        .map_err(crate::Error::Message)
}

/// The document's page map: `starts` the first pages of the citations a caller holds to the report it prints,
/// `is_citation` whether a line's text cites a report at the page it would read as a folio.
pub fn page_map(document: &PdfDocument, starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) -> Vec<PdfPageBinding> {
    legal_pdf_pages::page_map(&page_evidence(document), starts, is_citation)
}
