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
    // Keep the direct inline role and the enclosing heading's role as separate evidence.
    let (roles, heading_roles) = pdf_inspector::structure_tree::StructTree::from_doc(&document)
        .map(|tree| {
            use pdf_inspector::structure_tree::StructRole;
            use std::collections::HashMap;
            let pages = document.get_pages();
            let page_numbers = pages
                .iter()
                .map(|(&number, &id)| (id, number))
                .collect::<HashMap<_, _>>();
            let roles = tree
                .mcid_to_roles(&pages)
                .into_iter()
                .flat_map(|(page, roles)| {
                    roles.into_iter().map(move |(mcid, role)| {
                        (
                            (page.saturating_sub(1) as usize, mcid),
                            role.name().to_owned(),
                        )
                    })
                })
                .collect::<HashMap<_, _>>();
            let mut headings = HashMap::<_, Option<String>>::new();
            let mut stack = Vec::<Option<String>>::new();
            for element in tree.flatten() {
                stack.truncate(element.depth);
                let heading = if legal_pdf_core::is_heading_role(element.role.name()) {
                    Some(element.role.name().to_owned())
                } else if matches!(
                    element.role,
                    StructRole::Span | StructRole::Link | StructRole::Annot | StructRole::NonStruct
                ) {
                    stack.last().cloned().flatten()
                } else {
                    None
                };
                for content in element.content_refs {
                    if let Some(page) = content.page_id.and_then(|id| page_numbers.get(&id)) {
                        headings
                            .entry((page.saturating_sub(1) as usize, content.mcid))
                            .and_modify(|prior| {
                                if prior != &heading {
                                    *prior = None;
                                }
                            })
                            .or_insert_with(|| heading.clone());
                    }
                }
                stack.push(heading);
            }
            (roles, headings)
        })
        .unwrap_or_default();
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
    for page in &mut extracted.pages {
        for line in &mut page.lines {
            for span in &mut line.spans {
                span.structure_role = span.mcid.and_then(|mcid| roles.get(&(page.index, mcid)).cloned());
            }
            if let Some(role) = legal_pdf_core::tagged_heading(line, |span| {
                span.mcid
                    .and_then(|mcid| heading_roles.get(&(page.index, mcid)))
                    .and_then(Option::as_deref)
            }) {
                line.region_type = role.to_owned();
            }
        }
    }
    Ok(extracted)
}
