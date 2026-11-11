//! The projected document: its pages, lines and words as the reader sees them, and its summary.

use std::sync::{Arc, OnceLock};
use legal_pdf_core::model::{Page, PARSER_VERSION, PdfExtractionMetadata, ScalarRange};
use legal_structure_model::{DocumentStructure, NodeKind, ScalarText, serialization_sha256, StructureNode};
use serde::{Deserialize, Serialize};
use super::lookup::{
    PdfLookupRequest, PdfLookupStatus, PdfStructureLookup, PrintedParagraph, rendered_slice, structure_lookup,
};

#[derive(Deserialize, Serialize)]
pub struct PdfDocument {
    pub(crate) structure: legal_structure_model::DocumentStructure,
    pub(crate) authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
    pub(crate) summary: PdfSummary,
    /// The extraction's pages as witnesses, as read before the structure stage rewrote their lines, a recognized
    /// line's words joined where recognition gave it no text of its own: the one copy passages, page maps and
    /// recognized text read.
    pub(crate) evidence: Arc<Vec<Page>>,
    #[serde(skip)]
    pub(crate) printed: OnceLock<Vec<PrintedParagraph>>,
}

#[derive(Clone, Deserialize, Serialize)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "camelCase")]
pub struct PdfSummary {
    pub embedded_page_labels: Vec<Option<String>>,
    pub sha256: String,
    pub parser_version: String,
    pub cache_key: String,
    pub page_count: usize,
    /// How far each page's text is to be trusted, from 0 to 1, by page index.
    #[serde(default)]
    pub page_text_quality: Vec<f64>,
    pub status: PdfStatus,
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
    #[serde(default)]
    pub scanned_pages: Vec<usize>,
    /// The PDF's own bookmarks, in outline order (`legal_pdf_core::model::PdfOutlineEntry`).
    #[serde(default)]
    pub outline: Vec<legal_pdf_core::model::PdfOutlineEntry>,
}

/// Whether the parse read the document whole: degraded where it diagnosed anything, a page needing recognition among
/// them (which pages, `PdfSummary::pages_needing_ocr`).
#[derive(Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[cfg_attr(feature = "binding-types", derive(ts_rs::TS))]
#[serde(rename_all = "snake_case")]
pub enum PdfStatus {
    Ready,
    Degraded,
}

/// A page node's page (`PdfDocument::page_nodes`): its index.
pub(crate) fn page_index(page: &StructureNode) -> usize {
    page.page_indexes[0]
}

/// A page node's page number, its label.
pub(crate) fn page_number(page: &StructureNode) -> u32 {
    page.label.as_deref().unwrap_or("").parse().unwrap_or(0)
}

/// How far a page's text is to be trusted, from 0 to 1, of the pages' qualities (`PdfSummary::page_text_quality`).
pub(crate) fn page_confidence(quality: &[f64], page: &StructureNode) -> f64 {
    quality.get(page_index(page)).copied().unwrap_or(0.0).clamp(0.0, 1.0)
}

/// The digest of a lookup's answer: its units and the units before and after them.
pub(crate) fn lookup_payload_sha256(lookup: &PdfStructureLookup) -> String {
    #[derive(serde::Serialize)]
    struct Payload<'a, T> { units: &'a T, before: &'a T, after: &'a T }
    serialization_sha256(&Payload { units: &lookup.units, before: &lookup.before, after: &lookup.after })
}

impl PdfDocument {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        source_sha256: &str,
        cache_key: &str,
        status: PdfStatus,
        metadata: PdfExtractionMetadata,
        outline: Vec<legal_pdf_core::model::PdfOutlineEntry>,
        page_text_quality: Vec<f64>,
        mut evidence: Vec<Page>,
        authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
        structure: DocumentStructure,
    ) -> Self {
        let summary = PdfSummary {
            embedded_page_labels: metadata.embedded_page_labels,
            sha256: source_sha256.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
            cache_key: cache_key.to_owned(),
            page_count: page_text_quality.len(),
            page_text_quality,
            status,
            pages_needing_ocr: metadata.pages_needing_ocr,
            ocr_routed_pages: metadata.ocr_routed_pages,
            scanned_pages: metadata.scanned_pages,
            outline,
        };
        for line in evidence.iter_mut().filter(|page| page.source != "native").flat_map(|page| &mut page.lines) {
            if line.text.is_empty() {
                line.text = line.words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>().join(" ");
            }
        }
        Self { evidence: Arc::new(evidence), printed: OnceLock::new(), structure, authority_text_units, summary }
    }

    /// The pages passages are read from: the extraction witnesses.
    pub fn passage_pages(&self) -> Arc<Vec<Page>> {
        Arc::clone(&self.evidence)
    }

    /// The pages recognition read, each line's words joined where it has no text of its own.
    pub fn recognized_pages(&self) -> impl Iterator<Item = &Page> {
        self.evidence.iter().filter(|page| page.source != "native")
    }

    pub fn structure(&self) -> &legal_structure_model::DocumentStructure {
        &self.structure
    }

    pub fn structure_mut(&mut self) -> &mut legal_structure_model::DocumentStructure {
        self.printed.take();
        &mut self.structure
    }

    pub fn page_count(&self) -> usize {
        self.summary.page_count
    }

    /// The document's pages, each its page node, in page order.
    pub(crate) fn page_nodes(&self) -> impl Iterator<Item = &StructureNode> {
        self.structure.nodes.iter().filter(|node| node.kind == NodeKind::Page && !node.page_indexes.is_empty())
    }

    /// Each page's printed label (its page node's first alias), by page index; none where the page prints none read.
    pub fn page_labels(&self) -> Vec<Option<String>> {
        let mut labels = vec![None; self.summary.page_count];
        for node in &self.structure.nodes {
            if node.kind != legal_structure_model::NodeKind::Page {
                continue;
            }
            if let (Some(index), Some(label)) = (node.page_indexes.first(), node.aliases.as_ref().and_then(|aliases| aliases.first())) {
                if let Some(slot) = labels.get_mut(*index) {
                    *slot = Some(label.clone());
                }
            }
        }
        labels
    }

    pub fn summary(&self) -> &PdfSummary {
        &self.summary
    }

    pub fn authority_text_units(&self) -> &[legal_pdf_core::AuthorityTextUnit] {
        &self.authority_text_units
    }

    /// The units `request` names, the document's provisions and notes named as `query` (the document's own) reads them.
    pub fn lookup(&self, query: &legal_document_query::DocumentQuery, request: &PdfLookupRequest) -> PdfStructureLookup {
        structure_lookup(self, query, request)
    }

    /// Each paragraph number this PDF prints, with its printings (`PrintedParagraph`), in the order their text opens.
    /// Empty where it numbers none: it is then placed by its pages.
    pub fn printed_paragraphs(&self) -> &[PrintedParagraph] {
        self.printed.get_or_init(|| self.read_printed_paragraphs())
    }

    /// Exact printed membership for geometry, without the text response-size limit: each paragraph's extent, its
    /// own lines and everything set within it (its list's items, its block quotations), every printing of it.
    pub fn paragraph_source_ranges(&self, locator: &str) -> (PdfLookupStatus, Vec<ScalarRange>) {
        let range = legal_locator::ordinal_range("paragraph", locator)
            .or_else(|| legal_locator::parse_ordinal("paragraph", locator).map(|n| (n, n)));
        let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
            return (PdfLookupStatus::Invalid, Vec::new());
        };
        match self.paragraph_range(from, to) {
            Ok(paragraphs) => {
                // Each printing's extent: its words, its quotations and lists, as the structure reads them.
                let mut ranges: Vec<ScalarRange> = paragraphs.iter().flat_map(|(_, printings)| printings)
                    .map(|&index| &self.structure.nodes[index])
                    .flat_map(|node| if node.extent.is_empty() { node.ranges.clone() } else { node.extent.clone() })
                    .collect();
                ScalarRange::normalize_union(&mut ranges);
                (PdfLookupStatus::Found, ranges)
            }
            Err(status) => (status, Vec::new()),
        }
    }

    pub(crate) fn paragraph_range(
        &self,
        from: usize,
        to: usize,
    ) -> Result<&[PrintedParagraph], PdfLookupStatus> {
        let paragraphs = self.printed_paragraphs();
        if paragraphs.is_empty() {
            return Err(PdfLookupStatus::Unavailable);
        }
        let at = |number: usize| paragraphs.iter().position(|(printed, _)| *printed == number).ok_or(PdfLookupStatus::NotFound);
        let (start, end) = (at(from)?, at(to)?);
        if (from..=to).any(|number| at(number).is_err()) || end < start {
            return Err(PdfLookupStatus::NotFound);
        }
        Ok(&paragraphs[start..=end])
    }

    /// Each paragraph number the structure labels, once, with every printing of it, as the structure engine reads
    /// them (`paragraph_printings`): the paragraph a pinpoint to that number addresses.
    fn read_printed_paragraphs(&self) -> Vec<PrintedParagraph> {
        let mut paragraphs = legal_structure_model::structure_analysis().paragraph_printings(&self.structure);
        // In the order the text opens them: by the envelope of their printings' text.
        paragraphs.sort_by_cached_key(|(_, printings)| {
            let ranges = printings.iter().flat_map(|&index| self.structure.query_ranges(&self.structure.nodes[index]).into_owned());
            ranges.fold(None, |span: Option<(usize, usize)>, range| Some(span.map_or((range.start, range.end), |(start, end)| (start.min(range.start), end.max(range.end)))))
                .unwrap_or((0, 0))
        });
        paragraphs
    }

    /// Every page's text, in page order, as a page lookup returns it: its page node's rendered text, empty where the page
    /// has none. One read instead of one lookup per page.
    pub fn page_texts(&self) -> Vec<String> {
        let text = ScalarText::new(self.structure.query_text());
        self.page_nodes().map(|page| rendered_slice(&text, &self.structure, page)).collect()
    }
}
