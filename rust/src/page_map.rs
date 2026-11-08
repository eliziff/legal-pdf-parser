//! A document's page map (`legal_pdf_pages::page_map`) from what its pages print: each page's folio as the structure
//! stage read it, the PDF's own labels, and its opening pages' lines for a citation's first page.
use crate::PdfDocument;
pub use legal_pdf_pages::{EdgeLine, PageEvidence, PdfPageBinding};

/// The share of a page's height at its top and bottom whose lines are its head and foot.
const EDGE_BAND: f64 = 0.15;

/// What each page of `document` tells of its number: its folio as the structure read it, the PDF's own label, and its
/// head and foot lines (each line in the page's top or bottom EDGE_BAND, and its topmost and bottommost rows).
pub fn page_evidence(document: &PdfDocument) -> Vec<PageEvidence> {
    let observed = document.page_labels();
    let embedded = &document.summary().embedded_page_labels;
    let passage = document.passage_pages();
    let read: Vec<&legal_pdf_support::PdfTextPage> = passage.iter().chain(document.recognized_pages()).collect();
    (0..observed.len().max(embedded.len()))
        .map(|index| {
            let number = index as u32 + 1;
            let page = read.iter().find(|page| page.page_number == number);
            let lines = page.map(|page| {
                let top = page.lines.iter().map(|line| line.rect[3]).fold(f64::INFINITY, f64::min);
                let bottom = page.lines.iter().map(|line| line.rect[1]).fold(f64::NEG_INFINITY, f64::max);
                page.lines.iter()
                    .filter(|line| line.rect[3] <= page.height * EDGE_BAND || line.rect[1] >= page.height * (1.0 - EDGE_BAND) || line.rect[1] <= top || line.rect[3] >= bottom)
                    .map(|line| EdgeLine {
                        text: if line.text.is_empty() { line.words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>().join(" ") } else { line.text.clone() },
                        rect: line.rect,
                    })
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

/// The scale a page's head and foot are read again at, as a margin number is (`paragraph_recovery`).
#[cfg(feature = "ocr")]
const REREAD_SCALE: f64 = 3.0;

/// `page_evidence`, each page `read_again` names read again by
/// `ocr` in its top and bottom EDGE_BAND: a scan's text layer can leave out the folio its image prints, and what the
/// bands read stands for the text layer's lines there.
#[cfg(feature = "ocr")]
pub fn page_evidence_read(document: &PdfDocument, pdf: &[u8], ocr: &mut legal_pdf_ocr::OcrProvider) -> crate::Result<Vec<PageEvidence>> {
    let mut pages = page_evidence(document);
    let again = legal_pdf_pages::read_again(&pages);
    for (index, page) in pages.iter_mut().enumerate() {
        if !again[index] {
            continue;
        }
        let Some(render) = legal_pdf_ocr::render_gray_page(pdf, index, REREAD_SCALE * 72.0) else { continue };
        let (width, height) = (render.0 as f64 / REREAD_SCALE, render.1 as f64 / REREAD_SCALE);
        let bands = [[0.0, 0.0, width, height * EDGE_BAND], [0.0, height * (1.0 - EDGE_BAND), width, height]];
        let mut lines: Vec<EdgeLine> = Vec::new();
        for band in bands {
            let read = ocr.read_rendered_region(&render, band, REREAD_SCALE)?;
            let in_band = |line: &EdgeLine| line.rect[3] > band[1] && line.rect[1] < band[3];
            if read.is_empty() {
                lines.extend(page.lines.iter().filter(|line| in_band(line)).cloned());
            } else {
                lines.extend(read.into_iter().filter(|line| !line.text.trim().is_empty()).map(|line| EdgeLine { text: line.text, rect: line.bbox }));
            }
        }
        lines.extend(page.lines.iter().filter(|line| line.rect[3] > height * EDGE_BAND && line.rect[1] < height * (1.0 - EDGE_BAND)).cloned());
        if page.height <= 0.0 {
            (page.width, page.height) = (width, height);
        }
        page.lines = lines;
    }
    Ok(pages)
}

/// The document's page map: `starts` the first pages of the citations a caller holds to the report it prints,
/// `is_citation` whether a line's text cites a report at the page it would read as a folio.
pub fn page_map(document: &PdfDocument, starts: &[u32], is_citation: &dyn Fn(&str, &str) -> bool) -> Vec<PdfPageBinding> {
    legal_pdf_pages::page_map(&page_evidence(document), starts, is_citation)
}
