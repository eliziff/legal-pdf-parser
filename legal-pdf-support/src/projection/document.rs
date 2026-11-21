//! The projected document: its pages, lines and words as the reader sees them, and its summary.

use std::sync::{Arc, OnceLock};
use legal_pdf_core::model::{Page, PdfExtractionMetadata, ScalarRange, PARSER_VERSION};
use legal_structure_model::{serialization_sha256, DocumentStructure, NodeKind, ScalarText, StructureNode};
use serde::{Deserialize, Serialize};
use super::lookup::{
    rendered_slice, structure_lookup, PdfLookupRequest, PdfLookupStatus, PdfStructureLookup, PrintedParagraph,
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
    /// The page nodes (`page_nodes`) by their page, read once for every lookup.
    #[serde(skip)]
    page_index: OnceLock<PageNodes>,
    /// The source fragments in the order their text opens, read once for every passage placed.
    #[serde(skip)]
    fragment_index: OnceLock<FragmentIndex>,
}

/// The page nodes, in page order (their indexes among the structure's nodes), and where each page's stand among them.
struct PageNodes {
    nodes: Vec<usize>,
    by_page: std::collections::HashMap<usize, Vec<usize>>,
}

/// The source fragments (their indexes) in the order their text opens, the furthest any of them up to each reaches, and
/// each line's fragments by its id: a span's fragments are found without reading every fragment.
pub(crate) struct FragmentIndex {
    by_start: Vec<usize>,
    reach: Vec<usize>,
    pub(crate) by_id: std::collections::HashMap<String, Vec<usize>>,
}

impl FragmentIndex {
    fn of(fragments: &[legal_structure_model::SourceFragment]) -> Self {
        let mut by_start: Vec<usize> = (0..fragments.len()).collect();
        by_start.sort_by_key(|&at| fragments[at].range.start);
        let mut furthest = 0;
        let reach = by_start.iter().map(|&at| { furthest = furthest.max(fragments[at].range.end); furthest }).collect();
        let mut by_id = std::collections::HashMap::<String, Vec<usize>>::new();
        for (at, fragment) in fragments.iter().enumerate() {
            by_id.entry(fragment.id.clone()).or_default().push(at);
        }
        Self { by_start, reach, by_id }
    }

    /// The fragments whose text `range` overlaps, each once, in no order.
    pub(crate) fn overlapping<'a>(&'a self, fragments: &'a [legal_structure_model::SourceFragment], range: ScalarRange)
        -> impl Iterator<Item = usize> + 'a {
        let low = self.reach.partition_point(|&end| end <= range.start);
        let high = self.by_start.partition_point(|&at| fragments[at].range.start < range.end);
        self.by_start[low..high.max(low)].iter().copied().filter(move |&at| fragments[at].range.end > range.start)
    }
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

/// A page node's page number, from 1.
pub(crate) fn page_number(page: &StructureNode) -> u32 {
    page_index(page) as u32 + 1
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
        Self { evidence: Arc::new(evidence), printed: OnceLock::new(), page_index: OnceLock::new(), fragment_index: OnceLock::new(),
            structure, authority_text_units, summary }
    }

    fn page_index(&self) -> &PageNodes {
        self.page_index.get_or_init(|| {
            let nodes: Vec<usize> = (0..self.structure.nodes.len()).filter(|&at| {
                let node = &self.structure.nodes[at];
                node.kind == NodeKind::Page && !node.page_indexes.is_empty()
            }).collect();
            let mut by_page = std::collections::HashMap::<usize, Vec<usize>>::new();
            for (at, &node) in nodes.iter().enumerate() {
                by_page.entry(page_index(&self.structure.nodes[node])).or_default().push(at);
            }
            PageNodes { nodes, by_page }
        })
    }

    /// The page nodes of the pages (from 1) `numbers` names, in page order (`page_nodes`' order).
    pub(crate) fn page_nodes_numbered(&self, numbers: impl IntoIterator<Item = u32>) -> Vec<&StructureNode> {
        let index = self.page_index();
        let mut places: Vec<usize> = numbers.into_iter()
            .filter_map(|number| index.by_page.get(&(number as usize).checked_sub(1)?))
            .flatten().copied().collect();
        places.sort_unstable();
        places.dedup();
        places.into_iter().map(|at| &self.structure.nodes[index.nodes[at]]).collect()
    }

    /// A page's node by its index: the first of its page's nodes, or the last.
    pub(crate) fn page_node(&self, page: usize, last: bool) -> Option<&StructureNode> {
        let index = self.page_index();
        let places = index.by_page.get(&page)?;
        let at = if last { places.last() } else { places.first() }?;
        Some(&self.structure.nodes[index.nodes[*at]])
    }

    pub(crate) fn fragment_index(&self) -> &FragmentIndex {
        self.fragment_index.get_or_init(|| FragmentIndex::of(&self.structure.source_fragments))
    }

    /// The pages as the structure read them, whose lines its source fragments and line labels name.
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
        self.page_index.take();
        self.fragment_index.take();
        &mut self.structure
    }

    pub fn page_count(&self) -> usize {
        self.summary.page_count
    }

    /// The document's pages, each its page node, in page order.
    pub(crate) fn page_nodes(&self) -> impl Iterator<Item = &StructureNode> {
        self.structure.nodes.iter().filter(|node| node.kind == NodeKind::Page && !node.page_indexes.is_empty())
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
            Ok((start, end)) => {
                // Each printing's extent: its words, its quotations and lists, as the structure reads them.
                let mut ranges: Vec<ScalarRange> = self.printed_paragraphs()[start..=end].iter().flat_map(|(_, printings)| printings)
                    .map(|&index| &self.structure.nodes[index])
                    .flat_map(|node| if node.extent.is_empty() { node.ranges.clone() } else { node.extent.clone() })
                    .collect();
                ScalarRange::normalize_union(&mut ranges);
                (PdfLookupStatus::Found, ranges)
            }
            Err(status) => (status, Vec::new()),
        }
    }

    /// Where paragraphs `from` to `to` stand among the printed paragraphs: the first's place and the last's.
    pub(crate) fn paragraph_range(&self, from: usize, to: usize) -> Result<(usize, usize), PdfLookupStatus> {
        let paragraphs = self.printed_paragraphs();
        if paragraphs.is_empty() {
            return Err(PdfLookupStatus::Unavailable);
        }
        let at = |number: usize| paragraphs.iter().position(|(printed, _)| *printed == number).ok_or(PdfLookupStatus::NotFound);
        let (start, end) = (at(from)?, at(to)?);
        if (from..=to).any(|number| at(number).is_err()) || end < start {
            return Err(PdfLookupStatus::NotFound);
        }
        Ok((start, end))
    }

    /// Each paragraph number the structure labels, once, with every printing of it, as the structure engine reads
    /// them (`paragraph_printings`): the paragraph a pinpoint to that number addresses.
    fn read_printed_paragraphs(&self) -> Vec<PrintedParagraph> {
        let mut paragraphs = self.structure.paragraph_printings();
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
