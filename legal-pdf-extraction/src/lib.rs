use legal_pdf_core::{profile, PdfOcrProvider};
use legal_pdf_extraction_processor as processor;

pub use legal_pdf_extraction_processor::{recognize_pdf, Error, ExtractedPdf, Result};

pub fn extract_pdf(
    bytes: &[u8],
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    let document = profile::measure("extract.load_document", || {
        processor::load_extraction_document(bytes)
    })?;
    let page_geometries = profile::measure("extract.page_geometry", || {
        processor::page_geometries(&document)
    });
    let ((items, rects, lines, painted_rules), detection) =
        profile::measure("extract.fidelity", || {
            pdf_inspector::extract_fidelity_from_doc(&document)
        })?;
    let embedded_page_labels = processor::embedded_page_labels(&document, page_geometries.len());
    drop(document);
    drop((rects, lines));
    let mut extracted = profile::measure("extract.assemble", || {
        processor::assemble_pdf(
            bytes,
            &page_geometries,
            items,
            painted_rules,
            detection,
            ocr,
            ocr_pages,
        )
    })?;
    extracted.metadata.embedded_page_labels = embedded_page_labels;
    Ok(extracted)
}
