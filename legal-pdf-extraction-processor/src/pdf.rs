//! A PDF's pages assembled from pdf-inspector's items: geometry, painted text, the renderer's lines,
//! line assembly, rules, source lines, pictures and recognition, each in its own module; the
//! extracted document and its assembly are here.

use legal_pdf_core::{Error, FailureKind, Result};
use legal_pdf_core::model::{
    Diagnostic, Line, Page, PdfExtractionMetadata, PdfOutlineEntry, Span, Word,
};
use legal_pdf_core::{median, profile, union_bbox, OcrLine, OcrPageRequest, PdfOcrProvider};
use lopdf::{Document, Object, ObjectId};
use pdf_inspector::types::{FidelityGlyph, ItemType, PdfLine, PdfRect, TextItem, TextLine};
use pdf_inspector::PdfTypeResult;
use legal_pdf_core::regex;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

mod geometry;
pub use geometry::*;
mod painted;
pub(crate) use painted::*;
mod renderer;
pub(crate) use renderer::*;
mod lines;
pub(crate) use lines::*;
mod rules;
pub use rules::*;
mod source_lines;
pub(crate) use source_lines::*;
mod recognition;
pub use recognition::*;
mod pictures;
pub(crate) use pictures::*;

#[derive(Serialize, Deserialize)]
pub struct ExtractedPdf {
    pub pages: Vec<Page>,
    pub separators: Vec<Option<f64>>,
    pub diagnostics: Vec<Diagnostic>,
    pub metadata: PdfExtractionMetadata,
    /// The document's own bookmarks, read by `embedded_outline`.
    pub outline: Vec<PdfOutlineEntry>,
    /// By page index, the pictures on pages with text of their own that may be pictures of text
    /// the page does not otherwise hold, `[x0, y0, x1, y1]` in points from the page's top-left.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub image_regions: BTreeMap<usize, Vec<[f64; 4]>>,
}

/// The document as pdf-inspector loads it (bytes before the header dropped, bare struct names
/// mended, broken containers repaired, encryption opened), its vast null arrays pruned.
pub fn load_extraction_document(bytes: &[u8]) -> Result<Document> {
    let (mut document, _) = pdf_inspector::load_document_from_mem(bytes).map_err(unreadable)?;
    document.objects.values_mut().for_each(prune_extraction_object);
    Ok(document)
}

/// A PDF pdf-inspector cannot read: password-protected, or else corrupt.
pub(crate) fn unreadable(error: pdf_inspector::PdfError) -> Error {
    let kind = match error {
        pdf_inspector::PdfError::Encrypted => FailureKind::Encrypted,
        _ => FailureKind::Corrupt,
    };
    Error::Message(format!("PDF extraction failed: {error}")).failure(kind)
}

pub fn page_geometries(document: &Document) -> PageGeometryMap {
    document.get_pages().into_iter().map(|(page, id)| (page, (id, page_geometry(document, id)))).collect()
}

pub fn assemble_pdf(
    pdf: &[u8],
    geometries: &PageGeometryMap,
    mut items: Vec<TextItem>,
    painted_rules: Vec<PdfLine>,
    detection: PdfTypeResult,
    ocr: Option<&mut dyn PdfOcrProvider>,
    ocr_pages: Option<&[usize]>,
) -> Result<ExtractedPdf> {
    if geometries.is_empty() {
        return Err(Error::Message("PDF has no pages".to_owned()));
    }
    let mut by_page: HashMap<u32, Vec<TextLine>> = HashMap::new();
    items.retain_mut(|item| {
        let is_text = matches!(&item.item_type, ItemType::Text);
        if is_text {
            if let Some((_, geometry)) = geometries.get(&item.page) {
                transform_item(item, *geometry);
            }
            unmap_symbol_text(item);
        }
        !is_text || !item.text.is_empty()
    });
    deduplicate_painted_text(&mut items);
    assign_renderer_layout(&mut items);
    let scanned_pages = scanned_pages(geometries, &items);
    let pictures = pictures(geometries, &items);
    items.retain(|item| matches!(&item.item_type, ItemType::Text));
    // Pages whose text is all an invisible layer behind a scan: the layer is that scan's earlier
    // recognition.
    let mut shown_pages = HashSet::new();
    let mut layer_pages = HashSet::new();
    for item in items.iter().filter(|item| !item.text.trim().is_empty()) {
        if item.render_mode == Some(3) { layer_pages.insert(item.page); } else { shown_pages.insert(item.page); }
    }
    layer_pages.retain(|page| !shown_pages.contains(page));
    for line in group_source_order_lines(items) {
        by_page.entry(line.page).or_default().push(line);
    }

    let (mut pages, diagnostics, mut source_offset, mut weak_pages) = (Vec::with_capacity(geometries.len()), Vec::new(), 0, BTreeSet::<usize>::new());
    for (&number, &(_, geometry)) in geometries {
        let page_index = usize::try_from(number.saturating_sub(1)).unwrap_or(usize::MAX);
        let mut lines = Vec::new();
        for text_line in by_page.remove(&number).unwrap_or_default() {
            let local_index = lines.len() + 1;
            lines.extend(make_line(text_line, page_index, local_index, source_offset + local_index, geometry.height));
        }
        // A page pdf-inspector reads as decoding garbage holds no usable text, whatever prints.
        let reasons = detection.ocr_reasons_by_page.get(&number);
        let garbled = reasons.is_some_and(|reasons| reasons.iter().any(|reason| reason == pdf_inspector::OCR_REASON_SUSPECTED_GARBLED_TEXT));
        let quality = if garbled { 0.0 } else { text_quality(&lines) };
        // A scan's invisible text layer that reads poorly is recognized again.
        if lines.is_empty() || quality < 0.15 || layer_pages.contains(&number) && !layer_reads_well(&lines) {
            weak_pages.insert(page_index);
        }
        source_offset += lines.len();
        let (id, width, height) = (format!("p{number:04}"), round3(geometry.width), round3(geometry.height));
        pages.push(Page {
            id,
            index: page_index,
            number,
            width,
            height,
            lines,
            regions: vec![],
            source: "native".to_owned(),
            text_quality: quality,
            ..Default::default()
        });
    }
    let mut rules_by_page = HashMap::<u32, Vec<PdfLine>>::new();
    for rule in painted_rules {
        rules_by_page.entry(rule.page).or_default().push(rule);
    }
    let rules = |number: &u32| rules_by_page.get(number).map_or(&[][..], Vec::as_slice);
    let separators = profile::measure("extract.separators", || {
        geometries.iter().zip(&pages).map(|((number, &(_, geometry)), page)| separator_y(geometry, &page.lines, rules(number))).collect::<Vec<_>>()
    });
    for ((number, &(_, geometry)), page) in geometries.iter().zip(&mut pages) {
        page.regions = rule_regions(geometry, page, rules(number));
    }

    for page in &detection.pages_needing_ocr {
        if let Ok(index) = usize::try_from(page.saturating_sub(1)) {
            // A scan whose invisible text layer was read as its text needs no recognition when
            // that text reads well: the layer is the scan already recognized. One that reads
            // poorly, or a page needing recognition for any other reason, is recognized again.
            let layer_only = layer_pages.contains(page) && detection.ocr_reasons_by_page.get(page).is_some_and(|reasons|
                reasons.iter().all(|reason| reason == pdf_inspector::OCR_REASON_INVISIBLE_TEXT_LAYER));
            if layer_only && pages.get(index).is_some_and(|page| !page.lines.is_empty() && layer_reads_well(&page.lines)) {
                continue;
            }
            weak_pages.insert(index);
        }
    }

    // A scan whose own words are only a stamp over it is read from its image, whole: a stamp
    // over a picture of the page makes that picture no picture on a page of text.
    weak_pages.extend(scanned_pages.iter().copied().filter(|&index| {
        u32::try_from(index + 1).is_ok_and(|number| !layer_pages.contains(&number))
            && pages.get(index).is_some_and(|page| !page.lines.is_empty())
    }));
    crate::paragraph_recovery::picture_numbers(&mut pages, &pictures, &weak_pages);
    let image_regions = image_regions(&pages, pictures, &weak_pages);
    let metadata = PdfExtractionMetadata { embedded_page_labels: Vec::new(), pages_needing_ocr: weak_pages.into_iter().collect(), ocr_routed_pages: Vec::new(), scanned_pages };
    let mut extracted = ExtractedPdf { pages, separators, diagnostics, outline: Vec::new(), image_regions, metadata };
    recognize_pdf(pdf, &mut extracted, ocr, ocr_pages)?;
    Ok(extracted)
}
