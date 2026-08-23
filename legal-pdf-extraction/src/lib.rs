use legal_pdf_core::{model::TableBlock, PdfOcrProvider};
use std::path::Path;

pub use legal_pdf_extraction_processor::{
    inspect_pdf, pdf_page_count, union_bbox, Error, ExtractedPdf, PdfInspection, Result,
};

pub fn extract_pdf(
    path: &Path,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    let document = legal_pdf_extraction_processor::load_extraction_document(path)?;
    let pages = legal_pdf_extraction_processor::page_dimensions(&document);
    let (items, rects, lines) = pdf_inspector::extract_fidelity_from_doc(&document)?;
    let tables: Vec<TableBlock> =
        pdf_inspector::tables::detect_structured_tables(&items, &rects, &lines, &pages)
            .into_iter()
            .filter_map(|table| {
                legal_pdf_extraction_processor::project_table(
                    &document,
                    table.page,
                    table.index,
                    table.bbox,
                    table.cells,
                    table.method,
                    table.confidence,
                )
            })
            .collect();
    legal_pdf_extraction_processor::assemble_pdf(path, document, items, tables, ocr, ocr_pages)
}
