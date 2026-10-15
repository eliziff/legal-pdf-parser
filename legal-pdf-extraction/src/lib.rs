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
    let outline = processor::embedded_outline(&document);
    let mut structure_tags = pdf_inspector::structure_tree::StructTree::from_doc(&document)
        .map(|tree| tree.mcid_to_roles(&document.get_pages()).into_iter().flat_map(|(page, roles)|
            roles.into_iter().map(move |(mcid, role)| legal_pdf_core::model::PdfStructureTagWitness {
                page_index: page.saturating_sub(1) as usize, mcid, role: role.name().to_owned(),
                ranges: Vec::new(), line_ids: Vec::new(),
            })).collect::<Vec<_>>()).unwrap_or_default();
    structure_tags.sort_by_key(|tag| (tag.page_index, tag.mcid));
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
    extracted.outline = outline;
    let roles: std::collections::HashMap<_, _> = structure_tags.iter()
        .map(|tag| ((tag.page_index, tag.mcid), tag.role.as_str())).collect();
    for page in &mut extracted.pages {
        for line in &mut page.lines {
            for span in &mut line.spans {
                span.structure_role = span.mcid.and_then(|mcid| roles.get(&(page.index, mcid)).map(|role| (*role).to_owned()));
            }
        }
    }
    extracted.metadata.structure_tags = structure_tags;
    Ok(extracted)
}
