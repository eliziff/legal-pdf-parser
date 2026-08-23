use legal_pdf_core::{model::TableBlock, profile, PdfOcrProvider};
use legal_pdf_extraction_processor as processor;
use std::path::Path;

pub use legal_pdf_extraction_processor::{
    inspect_pdf, pdf_page_count, union_bbox, Error, ExtractedPdf, PdfInspection, Result,
};

pub fn extract_pdf(
    path: &Path,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    let document = profile::measure("extract.load_document", || {
        processor::load_extraction_document(path)
    })?;
    let (pages, page_geometries) = profile::measure("extract.page_geometry", || {
        processor::page_dimensions(&document)
    });
    let ((items, rects, lines, painted_rules), detection_evidence) =
        profile::measure("extract.fidelity", || {
            pdf_inspector::extract_fidelity_from_doc(&document)
        })?;
    let tables: Vec<TableBlock> = profile::measure("extract.tables", || {
        pdf_inspector::tables::detect_structured_tables(&items, &rects, &lines, &pages)
            .into_iter()
            .filter_map(|table| {
                processor::project_table(
                    &page_geometries,
                    table.page,
                    table.index,
                    table.bbox,
                    table.cells,
                    table.method,
                    table.confidence,
                )
            })
            .collect()
    });
    profile::measure("extract.assemble", || {
        processor::assemble_pdf(
            path,
            document,
            &page_geometries,
            items,
            painted_rules,
            detection_evidence,
            tables,
            ocr,
            ocr_pages,
        )
    })
}
