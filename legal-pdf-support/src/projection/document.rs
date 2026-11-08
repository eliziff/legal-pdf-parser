//! The projected document: its pages, lines and words as the reader sees them, and its summary.

use super::*;

#[derive(Deserialize, Serialize)]
pub struct PdfDocument {
    pub(crate) structure: legal_structure_model::DocumentStructure,
    pub(crate) authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
    pub(crate) summary: PdfSummary,
    /// The extraction's pages as witnesses, as read before the structure stage rewrote their lines.
    pub(crate) evidence: Arc<Vec<Page>>,
    #[serde(skip)]
    pub(crate) printed: OnceLock<Vec<PrintedParagraph>>,
}

#[derive(Clone, Deserialize, Serialize)]
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
    pub status: String,
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
    #[serde(default)]
    pub scanned_pages: Vec<usize>,
}

/// A page node's page (`PdfDocument::page_nodes`): its index.
pub(crate) fn page_index(page: &StructureNode) -> usize {
    page.page_indexes[0]
}

/// A page node's page number, its label's ("page12").
pub(crate) fn page_number(page: &StructureNode) -> u32 {
    page.printed_label().parse().unwrap_or(0)
}

/// How far a page's text is to be trusted, from 0 to 1, of the pages' qualities (`PdfSummary::page_text_quality`).
pub(crate) fn page_confidence(quality: &[f64], page: &StructureNode) -> f64 {
    quality.get(page_index(page)).copied().unwrap_or(0.0).clamp(0.0, 1.0)
}

/// Where a page's text was read from ("unknown" where it does not say).
pub(crate) fn page_provenance(page: &StructureNode) -> String {
    page.properties.get("source").filter(|source| !source.is_empty()).cloned().unwrap_or_else(|| "unknown".to_owned())
}

pub(crate) fn lookup_payload_sha256(lookup: &PdfStructureLookup) -> String {
    let mut writer = DigestWriter::default();
    writer.0.update(b"{\"units\":");
    serde_json::to_writer(&mut writer, &lookup.units).expect("lookup serialization");
    writer.0.update(b",\"before\":");
    serde_json::to_writer(&mut writer, &lookup.before).expect("lookup serialization");
    writer.0.update(b",\"after\":");
    serde_json::to_writer(&mut writer, &lookup.after).expect("lookup serialization");
    writer.0.update(b"}");
    format!("{:x}", writer.0.finalize())
}

pub(crate) fn summary(
    sha256: &str,
    cache_key: &str,
    page_text_quality: Vec<f64>,
    status: String,
    pdf_metadata: PdfExtractionMetadata,
) -> PdfSummary {
    PdfSummary {
        embedded_page_labels: pdf_metadata.embedded_page_labels,
        sha256: sha256.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
        cache_key: cache_key.to_owned(),
        page_count: page_text_quality.len(),
        page_text_quality,
        status,
        pages_needing_ocr: pdf_metadata.pages_needing_ocr,
        ocr_routed_pages: pdf_metadata.ocr_routed_pages,
        scanned_pages: pdf_metadata.scanned_pages,
    }
}

impl PdfDocument {
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        source_sha256: &str,
        cache_key: &str,
        status: String,
        metadata: PdfExtractionMetadata,
        page_text_quality: Vec<f64>,
        evidence: Arc<Vec<Page>>,
        authority_text_units: Vec<legal_pdf_core::AuthorityTextUnit>,
        structure: DocumentStructure,
    ) -> Self {
        let summary = summary(source_sha256, cache_key, page_text_quality, status, metadata);
        Self {
            evidence,
            printed: OnceLock::new(),
            structure,
            authority_text_units,
            summary,
        }
    }

    /// Capture extraction witnesses before structural derivation rewrites lines.
    pub fn passage_evidence(pages: &[Page]) -> Arc<Vec<Page>> {
        Arc::new(pages.to_vec())
    }

    /// The pages passages are read from: the extraction witnesses, a recognized scan's lines each read in full
    /// (`recognized_line`).
    pub fn passage_pages(&self) -> Arc<Vec<Page>> {
        if self.evidence.iter().all(|page| page.source == "native") {
            return Arc::clone(&self.evidence);
        }
        Arc::new(self.evidence.iter().map(|page| if page.source == "native" { page.clone() } else {
            Page { lines: page.lines.iter().map(recognized_line).collect(), ..page.clone() }
        }).collect())
    }

    /// The pages recognition read, as it read them.
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

    pub fn fingerprint(&self) -> DocumentFingerprint {
        let mut fingerprint = document_fingerprint(&self.structure);
        let product = serialization_sha256(&(&self.summary, &self.evidence));
        fingerprint.components.insert("pdf", product.clone());
        fingerprint.result_sha256 =
            serialization_sha256(&(fingerprint.result_sha256.as_str(), product));
        fingerprint
    }

    pub fn lookup(&self, request: &PdfLookupRequest) -> PdfStructureLookup {
        structure_lookup(self, request)
    }

    /// Each paragraph number this PDF prints, with its printings (`PrintedParagraph`), in the order their text opens.
    /// Empty where it numbers none: it is then placed by its pages.
    pub fn printed_paragraphs(&self) -> &[PrintedParagraph] {
        self.printed.get_or_init(|| self.read_printed_paragraphs())
    }

    /// Exact printed membership for geometry, without the text response-size limit: each paragraph's extent, its
    /// own lines and everything set within it (its list's items, its block quotations), every printing of it.
    pub fn paragraph_source_ranges(&self, locator: &str) -> (PdfLookupStatus, Vec<ScalarRange>) {
        let range = numeric_range("paragraph", locator)
            .or_else(|| legal_structure_model::structure_analysis().parse_ordinal("paragraph", locator).map(|n| (n, n)));
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

/// A recognized line read as a witness: its text in full, its words joined where recognition gave it none of its own.
fn recognized_line(line: &Line) -> Line {
    if !line.text.is_empty() {
        return line.clone();
    }
    Line { text: line.words.iter().map(|word| word.text.as_str()).collect::<Vec<_>>().join(" "), ..line.clone() }
}
