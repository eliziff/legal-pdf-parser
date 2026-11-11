//! A PDF's extraction, whole: its pages' text and geometry by pdf-inspector, its page labels, outline
//! and tagged structure roles, assembled into the extracted document, which recognition reads after (`recognize_pdf`).

use crate as processor;
use crate::ExtractedPdf;
use legal_pdf_core::{profile, Result};
use std::collections::HashMap;

pub fn extract_pdf(bytes: &[u8]) -> Result<ExtractedPdf> {
    let document = profile::measure("extract.load_document", || processor::load_extraction_document(bytes))?;
    let page_geometries = profile::measure("extract.page_geometry", || processor::page_geometries(&document));
    let ((items, rects, lines, painted_rules), detection) = profile::measure("extract.fidelity", || pdf_inspector::extract_fidelity_from_doc(&document, None)).map_err(processor::pdf::unreadable)?;
    let embedded_page_labels = processor::embedded_page_labels(&document, page_geometries.len());
    let outline = processor::embedded_outline(&document);
    // Keep direct inline roles and enclosing heading/list-item roles as separate evidence.
    let (roles, block_roles) = pdf_inspector::structure_tree::StructTree::from_doc(&document)
        .map(|tree| {
            use pdf_inspector::structure_tree::StructRole;
            let pages = document.get_pages();
            let page_numbers = pages.iter().map(|(&number, &id)| (id, number)).collect::<HashMap<_, _>>();
            let roles = tree
                .mcid_to_roles(&pages)
                .into_iter()
                .flat_map(|(page, roles)| roles.into_iter().map(move |(mcid, role)| ((page.saturating_sub(1) as usize, mcid), role.name().to_owned())))
                .collect::<HashMap<_, _>>();
            let mut blocks = HashMap::<_, (Option<String>, Option<&'static str>)>::new();
            let mut stack = Vec::<(Option<String>, Option<&'static str>)>::new();
            for element in tree.flatten() {
                stack.truncate(element.depth);
                let inline = matches!(element.role, StructRole::Span | StructRole::Link | StructRole::Annot | StructRole::NonStruct);
                let heading = if legal_pdf_core::is_heading_role(element.role.name()) {
                    Some(element.role.name().to_owned())
                } else if inline {
                    stack.last().and_then(|prior| prior.0.clone())
                } else {
                    None
                };
                let list = if element.role == StructRole::LI {
                    Some("LI")
                } else if inline || matches!(element.role, StructRole::Lbl | StructRole::LBody | StructRole::P | StructRole::Div) {
                    stack.last().and_then(|prior| prior.1)
                } else {
                    None
                };
                for content in element.content_refs {
                    if let Some(page) = content.page_id.and_then(|id| page_numbers.get(&id)) {
                        blocks
                            .entry((page.saturating_sub(1) as usize, content.mcid))
                            .and_modify(|prior| {
                                if prior.0 != heading {
                                    prior.0 = None;
                                }
                                if prior.1 != list {
                                    prior.1 = None;
                                }
                            })
                            .or_insert_with(|| (heading.clone(), list));
                    }
                }
                stack.push((heading, list));
            }
            (roles, blocks)
        })
        .unwrap_or_default();
    drop(document);
    drop(lines);
    let mut extracted =
        profile::measure("extract.assemble", || processor::assemble_pdf(&page_geometries, items, painted_rules, detection))?;
    processor::add_rect_regions(&mut extracted.pages, &page_geometries, &rects);
    extracted.metadata.embedded_page_labels = embedded_page_labels;
    extracted.outline = outline;
    for page in &mut extracted.pages {
        let block = |span: &legal_pdf_core::model::Span| span.mcid.and_then(|mcid| block_roles.get(&(page.index, mcid)));
        for line in &mut page.lines {
            for span in &mut line.spans {
                span.structure_role = span.mcid.and_then(|mcid| {
                    let key = (page.index, mcid);
                    let role = roles.get(&key)?;
                    // Conflicting enclosing claims cannot reappear through the direct role.
                    let valid = if legal_pdf_core::is_heading_role(role) {
                        block_roles.get(&key).and_then(|roles| roles.0.as_deref()) == Some(role.as_str())
                    } else {
                        role != "LI" || block_roles.get(&key).and_then(|roles| roles.1) == Some("LI")
                    };
                    valid.then(|| role.clone())
                });
            }
            if let Some(role) = legal_pdf_core::tagged_heading(line, |span| block(span).and_then(|roles| roles.0.as_deref())) {
                line.source_tag = role.to_owned();
            } else if legal_pdf_core::tagged_line_role(line, |span| block(span).and_then(|roles| roles.1)).is_some() {
                line.source_tag = "LI".to_owned();
            }
        }
    }
    Ok(extracted)
}

/// Native opening-page witnesses for identifying a downloaded source. No recognition or structure inference.
#[derive(serde::Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PdfOpening {
    pub page_count: usize,
    pub pages: Vec<PdfOpeningPage>,
}

#[derive(serde::Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PdfOpeningPage {
    #[serde(flatten)]
    pub page: legal_pdf_core::model::Page,
    pub margin_text: String,
}

pub fn pdf_opening(bytes: &[u8]) -> Result<PdfOpening> {
    use pdf_inspector::types::{ItemType, TextLine};
    use std::collections::{BTreeMap, HashSet};
    let document = processor::load_extraction_document(bytes)?;
    let mut geometries = processor::page_geometries(&document);
    let page_count = geometries.len();
    let selected = HashSet::from([1, 2]);
    geometries.retain(|number, _| selected.contains(number));
    let ((items, _, _, rules), detection) = pdf_inspector::extract_fidelity_from_doc(&document, Some(&selected))
        .map_err(processor::pdf::unreadable)?;
    // The same source line owns fragments of a vertical margin stamp. Its baseline and advance,
    // rather than its words, establish that it stands vertically at the page's edge.
    let mut margins = BTreeMap::<(u32, u32, u32), TextLine>::new();
    for item in &items {
        let (Some((_, geometry)), Some(source)) = (geometries.get(&item.page), item.fidelity.as_ref()) else { continue };
        let x = f64::from(source.baseline[0]) - geometry.x0;
        if !matches!(item.item_type, ItemType::Text) || source.advance[1].abs() <= source.advance[0].abs() * 4.0
            || x < 0.0 || x > geometry.raw_width || x > geometry.raw_width * 0.12 && x < geometry.raw_width * 0.88 {
            continue;
        }
        margins.entry((item.page, source.text_object, source.source_line)).or_insert_with(|| TextLine {
            items: Vec::new(), y: item.y, page: item.page, adaptive_threshold: 0.1,
        }).items.push(item.clone());
    }
    let extracted = processor::assemble_pdf(&geometries, items, rules, detection)?;
    let pages = extracted.pages.into_iter().map(|page| {
        let margin_text = margins.iter().filter(|((number, ..), _)| *number == page.number)
            .map(|(_, line)| processor::pdf::assemble_line(line).text).collect::<Vec<_>>().join("\n");
        PdfOpeningPage { page, margin_text }
    }).collect();
    Ok(PdfOpening { page_count, pages })
}
