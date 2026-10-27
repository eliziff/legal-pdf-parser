//! The projected document: its pages, lines and words as the reader sees them, and its summary.

use super::*;

#[derive(Deserialize, Serialize)]
pub struct PdfDocument {
    pub(crate) structure: legal_structure_model::DocumentStructure,
    pub(crate) pages: Vec<ProjectionPage>,
    pub(crate) footnotes: Vec<ProjectionFootnote>,
    pub(crate) authority_text_units: Vec<Value>,
    pub(crate) summary: PdfSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) recognized_pages: Option<Vec<PdfTextPage>>,
    pub(crate) passage_pages: Arc<Vec<PdfTextPage>>,
    #[serde(skip)]
    pub(crate) printed: OnceLock<PrintedParagraphMap>,
}

/// Display-space text geometry retained with the prepared document.
#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdfTextPage {
    pub page_number: u32,
    pub source: String,
    pub width: f64,
    pub height: f64,
    pub lines: Vec<PdfTextLine>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub regions: Vec<legal_pdf_core::model::Region>,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PdfTextLine {
    pub id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub text: String,
    pub rect: [f64; 4],
    pub words: Vec<PdfTextWord>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spans: Vec<legal_pdf_core::model::Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr: Option<legal_pdf_core::model::OcrEvidence>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub region_id: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub region_type: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub source: String,
}

#[derive(Clone, Deserialize, Serialize)]
pub struct PdfTextWord {
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub id: String,
    /// Line-local Unicode scalar boundaries, matching the physical Word witness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub inferred_geometry: bool,
    pub text: String,
    pub rect: [f64; 4],
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub underlined: bool,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdfSummary {
    pub embedded_page_labels: Vec<Option<String>>,
    pub sha256: String,
    pub parser_version: String,
    pub cache_key: String,
    pub page_count: usize,
    pub projection_page_count: usize,
    pub status: String,
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
    #[serde(default)]
    pub scanned_pages: Vec<usize>,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct ProjectionPage {
    pub(crate) id: String,
    pub(crate) index: usize,
    pub(crate) number: u32,
    pub(crate) evidence_text: String,
    pub(crate) source: String,
    pub(crate) text_quality: f64,
}

#[derive(Deserialize, Serialize)]
pub(crate) struct ProjectionFootnote {
    pub(crate) pair_id: String,
    pub(crate) label: String,
    pub(crate) occurrence: usize,
    pub(crate) restart_sequence: usize,
    pub(crate) reference_page: Option<u32>,
    pub(crate) body_pages: Vec<u32>,
    pub(crate) body: String,
    pub(crate) sentence_proposition: String,
    pub(crate) passage_since_prior_note: String,
    pub(crate) confidence: f64,
    pub(crate) provenance: String,
    pub(crate) warnings: Vec<String>,
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
    page_count: usize,
    status: String,
    pdf_metadata: PdfExtractionMetadata,
) -> PdfSummary {
    PdfSummary {
        embedded_page_labels: pdf_metadata.embedded_page_labels,
        sha256: sha256.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
        cache_key: cache_key.to_owned(),
        page_count,
        projection_page_count: page_count,
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
        pages: Vec<Page>,
        paragraphs: Vec<Paragraph>,
        footnotes: Vec<Footnote>,
        mut structure: DocumentStructure,
    ) -> legal_pdf_core::Result<Self> {
        let authority_text_units =
            crate::adapters::to_toa_text_units_from_parts(&paragraphs, &footnotes, &structure)?;
        let joins = attach_manuscript_edits(&paragraphs, &mut structure);
        attach_source_continuations(&paragraphs, &footnotes, &mut structure);
        let continuations = structure.continuations.as_deref().unwrap_or_default();
        structure.joins.extend(joins.into_iter().filter(|(_, required)|
            required.as_ref().is_none_or(|link| continuations.contains(link))).map(|(join, _)| join));
        attach_rendered_text(&paragraphs, &pages, &footnotes, &mut structure)?;
        let page_count = pages.len();
        let summary = summary(source_sha256, cache_key, page_count, status, metadata);
        Ok(Self::project(
            pages,
            footnotes,
            authority_text_units,
            structure,
            summary,
        ))
    }

    pub(crate) fn project(
        pages: Vec<Page>,
        footnotes: Vec<Footnote>,
        authority_text_units: Vec<Value>,
        structure: DocumentStructure,
        summary: PdfSummary,
    ) -> Self {
        let recognized_pages = pages
            .iter()
            .filter(|page| page.source != "native")
            .map(|page| PdfTextPage {
                page_number: page.number,
                source: page.source.clone(),
                width: page.width,
                height: page.height,
                regions: page.regions.clone(),
                lines: page.lines.iter().map(physical_text_line).collect(),
            })
            .collect();
        Self {
            passage_pages: Arc::new(Vec::new()),
            printed: OnceLock::new(),
            recognized_pages: Some(recognized_pages),
            structure,
            pages: pages
                .into_iter()
                .map(
                    |Page {
                         id,
                         index,
                         number,
                         lines,
                         source,
                         text_quality,
                         ..
                     }| ProjectionPage {
                        id,
                        index,
                        number,
                        evidence_text: join_page_lines(lines),
                        source,
                        text_quality,
                    },
                )
                .collect(),
            footnotes: footnotes
                .into_iter()
                .map(
                    |Footnote {
                         pair_id,
                         label,
                         occurrence,
                         restart_sequence,
                         reference_page,
                         body_pages,
                         body,
                         sentence_proposition,
                         passage_since_prior_note,
                         confidence,
                         provenance,
                         warnings,
                         ..
                     }| ProjectionFootnote {
                        pair_id,
                        label,
                        occurrence,
                        restart_sequence,
                        reference_page,
                        body_pages,
                        body,
                        sentence_proposition,
                        passage_since_prior_note,
                        confidence,
                        provenance,
                        warnings,
                    },
                )
                .collect(),
            authority_text_units,
            summary,
        }
    }

    pub fn has_recognized_geometry(&self) -> bool {
        self.recognized_pages.is_some()
    }

    /// Capture extraction witnesses before structural derivation rewrites lines.
    pub fn passage_evidence(pages: &[Page]) -> Arc<Vec<PdfTextPage>> {
        Arc::new(
            pages
                .iter()
                .map(|page| PdfTextPage {
                    page_number: page.number,
                    source: page.source.clone(),
                    width: page.width,
                    height: page.height,
                    regions: page.regions.clone(),
                    lines: page.lines.iter().map(physical_text_line).collect(),
                })
                .collect(),
        )
    }

    pub fn set_passage_evidence(&mut self, pages: Arc<Vec<PdfTextPage>>) {
        self.printed.take();
        self.recognized_pages = Some(pages.iter().filter(|page|page.source!="native").cloned().collect());
        self.passage_pages = Arc::new(pages.iter().map(|page| {
            let mut evidence=page.clone();
            if evidence.source!="native" {evidence.lines.clear();evidence.regions.clear();}
            evidence
        }).collect());
    }

    /// The pages passages are read from: the extraction witnesses, with a recognized scan's
    /// lines in place of its empty native page.
    pub fn passage_pages(&self) -> Arc<Vec<PdfTextPage>> {
        let recognized = self.recognized_pages();
        if recognized.iter().all(|page| page.lines.is_empty()) {
            return Arc::clone(&self.passage_pages);
        }
        Arc::new(
            self.passage_pages
                .iter()
                .map(|page| {
                    match recognized.iter().find(|read| {
                        read.page_number == page.page_number
                            && page.lines.is_empty()
                            && !read.lines.is_empty()
                    }) {
                        Some(read) => PdfTextPage {
                            page_number: page.page_number,
                            source: read.source.clone(),
                            regions: read.regions.clone(),
                            width: page.width,
                            height: page.height,
                            lines: read.lines.iter().map(recognized_witness_line).collect(),
                        },
                        None => page.clone(),
                    }
                })
                .collect(),
        )
    }

    pub fn recognized_pages(&self) -> &[PdfTextPage] {
        self.recognized_pages.as_deref().unwrap_or(&[])
    }

    pub fn structure(&self) -> &legal_structure_model::DocumentStructure {
        &self.structure
    }

    pub fn structure_mut(&mut self) -> &mut legal_structure_model::DocumentStructure {
        self.printed.take();
        &mut self.structure
    }

    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub fn summary(&self) -> &PdfSummary {
        &self.summary
    }

    pub fn authority_text_units(&self) -> &[Value] {
        &self.authority_text_units
    }

    pub fn fingerprint(&self) -> DocumentFingerprint {
        let mut fingerprint = document_fingerprint(&self.structure);
        let product = serialization_sha256(&(
            &self.pages,
            &self.footnotes,
            &self.summary,
            &self.passage_pages,
            &self.recognized_pages,
        ));
        fingerprint.components.insert("pdf", product.clone());
        fingerprint.result_sha256 =
            serialization_sha256(&(fingerprint.result_sha256.as_str(), product));
        fingerprint
    }

    pub fn lookup(&self, request: &PdfLookupRequest) -> PdfStructureLookup {
        structure_lookup(self, request)
    }

    /// The declared paragraph nodes of this PDF, in reading order.
    /// Empty where it numbers none: it is then placed by its pages.
    pub fn printed_paragraphs(&self) -> &[PrintedParagraph] {
        &self.printed_paragraph_map().paragraphs
    }

    /// Exact printed membership for geometry, without the text response-size limit.
    pub fn paragraph_source_ranges(&self, locator: &str) -> (PdfLookupStatus, Vec<ScalarRange>) {
        let range = numeric_range("paragraph", locator)
            .or_else(|| parse_ordinal("paragraph", locator).map(|n| (n, n)));
        let Some((from, to)) = range.filter(|(a, b)| a <= b && b - a < 100) else {
            return (PdfLookupStatus::Invalid, Vec::new());
        };
        match self.paragraph_range(from, to) {
            Ok(paragraphs) => {
                let ids = paragraphs.iter().map(|paragraph| paragraph.id.as_str()).collect::<HashSet<_>>();
                let mut ranges = self.structure.nodes.iter().filter(|node| ids.contains(node.id.as_str()))
                    .flat_map(|node| if node.ranges.is_empty() { std::slice::from_ref(&node.range) }
                        else { &node.ranges }).copied().collect();
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
        let mapping = self.printed_paragraph_map();
        if mapping.status.is_empty() {
            return Err(PdfLookupStatus::Unavailable);
        }
        for number in from..=to {
            let status = mapping
                .status
                .get(&number)
                .copied()
                .unwrap_or(PdfLookupStatus::NotFound);
            if status != PdfLookupStatus::Found {
                return Err(status);
            }
        }
        let paragraphs = &mapping.paragraphs;
        let start = paragraphs
            .iter()
            .position(|p| p.number == from)
            .ok_or(PdfLookupStatus::Unavailable)?;
        let end = paragraphs
            .iter()
            .position(|p| p.number == to)
            .filter(|end| *end >= start)
            .ok_or(PdfLookupStatus::Unavailable)?;
        Ok(&paragraphs[start..=end])
    }

    pub(crate) fn printed_paragraph_map(&self) -> &PrintedParagraphMap {
        self.printed.get_or_init(|| self.read_printed_paragraphs())
    }

    pub(crate) fn read_printed_paragraphs(&self) -> PrintedParagraphMap {
        let mut status = HashMap::new();
        let mut paragraphs = Vec::new();
        for node in self
            .structure
            .nodes
            .iter()
            .filter(|node| node.kind == NodeKind::Paragraph)
        {
            let Some(number) = node
                .label
                .as_deref()
                .and_then(|label| parse_ordinal("paragraph", label))
            else {
                continue;
            };
            if status.insert(number, PdfLookupStatus::Found).is_some() {
                status.insert(number, PdfLookupStatus::Ambiguous);
                continue;
            }
            let range = self.structure.query_range(node);
            paragraphs.push(PrintedParagraph {
                id: node.id.clone(),
                number,
                start: range.start,
                end: range.end,
                ranges: self.structure.query_ranges(node).into_owned(),
                page_numbers: node
                    .page_indexes
                    .iter()
                    .map(|page| (*page + 1) as u32)
                    .collect(),
                line_ids: node.line_ids.clone(),
            });
        }
        paragraphs
            .retain(|paragraph| status.get(&paragraph.number) == Some(&PdfLookupStatus::Found));
        paragraphs.sort_by_key(|paragraph| (paragraph.start, paragraph.end));
        PrintedParagraphMap { paragraphs, status }
    }

    /// Every page's text, in page order, as a page lookup returns it: empty where
    /// the page has no exact text. One read instead of one lookup per page.
    pub fn page_texts(&self) -> Vec<&str> {
        let text = ScalarText::new(self.structure.query_text());
        let nodes = nodes_by_id(self);
        self.pages
            .iter()
            .map(|page| {
                let exact = nodes
                    .get(page.id.as_str())
                    .is_some_and(|node| !rendered_slice(&text, &self.structure, node).is_empty());
                if exact {
                    page.evidence_text.as_str()
                } else {
                    ""
                }
            })
            .collect()
    }
}

pub(crate) fn physical_text_line(line:&Line)->PdfTextLine {
    PdfTextLine {
        id:line.id.clone(),text:line.text.clone(),rect:line.bbox,
        spans:line.spans.clone(),ocr:line.ocr.clone(),region_id:line.region_id.clone(),
        region_type:line.region_type.clone(),source:line.source.clone(),
        words:line.words.iter().map(|word|PdfTextWord {
            id:word.id.clone(),text:word.text.clone(),rect:word.bbox,
            start:Some(word.start),end:Some(word.end),inferred_geometry:false,underlined:word.underlined,
        }).collect(),
    }
}

/// A recognized line read as a witness: its text in full, and its words in order. A line
/// recognized without word boxes spaces its words across its own box.
pub(crate) fn recognized_witness_line(line: &PdfTextLine) -> PdfTextLine {
    let text = if line.text.is_empty() {
        line.words
            .iter()
            .map(|word| word.text.as_str())
            .collect::<Vec<_>>()
            .join(" ")
    } else {
        line.text.clone()
    };
    let words = if line.words.is_empty() {
        let [left, top, right, bottom] = line.rect;
        let advance = (right - left) / text.chars().count().max(1) as f64;
        let mut at = 0;
        text.split(' ')
            .filter_map(|word| {
                let start = at;
                at += word.chars().count() + 1;
                (!word.is_empty()).then(|| PdfTextWord {
                    id:String::new(),start:Some(start),end:Some(at-1),inferred_geometry:true,
                    underlined: false,
                    text: word.to_owned(),
                    rect: [
                        left + start as f64 * advance,
                        top,
                        left + (at - 1) as f64 * advance,
                        bottom,
                    ],
                })
            })
            .collect()
    } else {
        line.words.clone()
    };
    PdfTextLine {
        id: line.id.clone(),
        text,
        rect: line.rect,
        words,
        spans:line.spans.clone(),ocr:line.ocr.clone(),region_id:line.region_id.clone(),
        region_type:line.region_type.clone(),source:line.source.clone(),
    }
}
