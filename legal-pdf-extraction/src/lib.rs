use legal_pdf_core::{model::TableBlock, profile, PdfOcrProvider};
use legal_pdf_extraction_processor as processor;

mod tables;

pub use legal_pdf_extraction_processor::{union_bbox, Error, ExtractedPdf, Result};

pub fn extract_pdf(
    bytes: &[u8],
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    let document = profile::measure("extract.load_document", || {
        processor::load_extraction_document(bytes)
    })?;
    let (pages, page_geometries) = profile::measure("extract.page_geometry", || {
        processor::page_dimensions(&document)
    });
    let ((items, rects, lines, painted_rules), detection) =
        profile::measure("extract.fidelity", || {
            pdf_inspector::extract_fidelity_from_doc(&document)
        })?;
    let tables: Vec<TableBlock> = profile::measure("extract.tables", || {
        tables::detect_structured_tables(&items, &rects, &lines, &pages)
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
            bytes,
            document,
            &page_geometries,
            items,
            painted_rules,
            detection,
            tables,
            ocr,
            ocr_pages,
        )
    })
}
