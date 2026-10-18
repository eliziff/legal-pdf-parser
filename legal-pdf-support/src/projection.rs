pub use legal_structure_model::parse_ordinal;
use legal_pdf_core::model::{
    Footnote, Line, Page, Paragraph, PdfExtractionMetadata, ScalarRange, PARSER_VERSION,
};
use legal_structure_model::{Derivation, DocumentStructure, NodeKind, StructureNode};
use legal_structure_model::{
    document_fingerprint, normalize_compact_numbered_section_locator, utf16_len,
    DocumentFingerprint, ScalarText,
};
use regex::Regex;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use serde_json::Map;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::{self, Write};
use std::sync::{Arc, OnceLock};
use unicode_normalization::UnicodeNormalization;

const MAX_UNITS: usize = 20;
const MAX_CONTEXT: usize = 2;
const MAX_RETURN_CHARS: usize = 60_000;
const LOCATOR_KINDS: [&str; 11] = [
    "page",
    "paragraph",
    "footnote",
    "section",
    "subsection",
    "provision_paragraph",
    "subparagraph",
    "clause",
    "subclause",
    "schedule",
    "article",
];

#[derive(Deserialize, Serialize)]
pub struct PdfDocument {
    structure: legal_structure_model::DocumentStructure,
    pages: Vec<ProjectionPage>,
    footnotes: Vec<ProjectionFootnote>,
    authority_text_units: Vec<Value>,
    summary: PdfSummary,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    recognized_pages: Option<Vec<PdfTextPage>>,
    passage_pages: Arc<Vec<PdfTextPage>>,
    #[serde(skip)]
    printed: OnceLock<PrintedParagraphMap>,
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
struct ProjectionPage {
    id: String,
    index: usize,
    number: u32,
    evidence_text: String,
    source: String,
    text_quality: f64,
}

#[derive(Deserialize, Serialize)]
struct ProjectionFootnote {
    pair_id: String,
    label: String,
    occurrence: usize,
    restart_sequence: usize,
    reference_page: Option<u32>,
    body_pages: Vec<u32>,
    body: String,
    sentence_proposition: String,
    passage_since_prior_note: String,
    confidence: f64,
    provenance: String,
    warnings: Vec<String>,
}

#[derive(Default)]
struct DigestWriter(Sha256);

impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

fn serialization_sha256(value: &impl Serialize) -> String {
    let mut writer = DigestWriter::default();
    serde_json::to_writer(&mut writer, value).expect("fingerprint serialization");
    format!("{:x}", writer.0.finalize())
}

fn lookup_payload_sha256(lookup: &PdfStructureLookup) -> String {
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

fn summary(
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

    fn project(
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

    fn paragraph_range(
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

    fn printed_paragraph_map(&self) -> &PrintedParagraphMap {
        self.printed.get_or_init(|| self.read_printed_paragraphs())
    }

    fn read_printed_paragraphs(&self) -> PrintedParagraphMap {
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

fn physical_text_line(line:&Line)->PdfTextLine {
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
fn recognized_witness_line(line: &PdfTextLine) -> PdfTextLine {
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

struct PrintedParagraphMap {
    paragraphs: Vec<PrintedParagraph>,
    status: HashMap<usize, PdfLookupStatus>,
}

/// A paragraph a PDF numbers in print ("[12]", or 12 in the margin): its number, its span of the
/// query text in UTF-16 code units, the pages it is on and its native lines.
/// `ranges` is exact membership; `start` and `end` only position its envelope.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PrintedParagraph {
    #[serde(skip)]
    id: String,
    pub number: usize,
    pub start: usize,
    pub end: usize,
    pub ranges: Vec<ScalarRange>,
    pub page_numbers: Vec<u32>,
    pub line_ids: Vec<String>,
}

#[derive(Clone, Serialize)]
pub struct PdfLookupRequest {
    pub locator_kind: String,
    pub locator: String,
    pub end_locator: Option<String>,
    pub context_blocks: usize,
    pub page: Option<u32>,
    pub occurrence: Option<usize>,
}

impl PdfLookupRequest {
    pub fn new(locator_kind: impl Into<String>, locator: impl Into<String>) -> Self {
        Self {
            locator_kind: locator_kind.into(),
            locator: locator.into(),
            end_locator: None,
            context_blocks: 0,
            page: None,
            occurrence: None,
        }
    }

    fn valid(&self) -> bool {
        LOCATOR_KINDS.contains(&self.locator_kind.as_str())
            && !self.locator.trim().is_empty()
            && utf16_len(&self.locator) <= 200
            && self
                .end_locator
                .as_deref()
                .is_none_or(|end| end.is_empty() || utf16_len(end) <= 200)
            && self.context_blocks <= MAX_CONTEXT
            && self.page.is_none_or(|page| page > 0)
            && self.occurrence.is_none_or(|occurrence| occurrence > 0)
    }
}

#[derive(Serialize)]
pub struct PdfLookupProposition {
    pub sentence: String,
    pub passage_since_prior_note: String,
}

#[derive(Serialize)]
pub struct PdfLookupNote {
    pub label: String,
    pub occurrence: usize,
    pub restart_sequence: usize,
    pub reference_page: Option<u32>,
    pub body_pages: Vec<u32>,
    pub warnings: Vec<String>,
}

#[derive(Serialize)]
pub struct PdfLookupUnit {
    pub id: String,
    pub kind: &'static str,
    pub locator: String,
    pub text: String,
    pub page_numbers: Vec<u32>,
    pub confidence: Option<f64>,
    pub confidence_basis: &'static str,
    pub provenance: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposition: Option<PdfLookupProposition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<PdfLookupNote>,
}

#[derive(Serialize)]
pub struct PdfLookupPage {
    pub page_number: u32,
    pub text: String,
}

#[derive(Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PdfLookupStatus {
    Found,
    NotFound,
    Ambiguous,
    Invalid,
    Unavailable,
}

#[derive(Serialize)]
pub struct PdfStructureLookup {
    pub schema_version: &'static str,
    pub requested: PdfLookupRequest,
    pub units: Vec<PdfLookupUnit>,
    pub before: Vec<PdfLookupUnit>,
    pub after: Vec<PdfLookupUnit>,
    pub matches: Vec<String>,
    pub pages: Vec<PdfLookupPage>,
    pub status: PdfLookupStatus,
    pub exact: bool,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub payload_sha256: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub page_text_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

fn result(
    input: &PdfLookupRequest,
    status: PdfLookupStatus,
    error: Option<String>,
) -> PdfStructureLookup {
    let mut requested = input.clone();
    if requested.end_locator.as_deref() == Some("") {
        requested.end_locator = None;
    }
    PdfStructureLookup {
        schema_version: "legalpdf.structure-lookup.v1",
        requested,
        units: Vec::new(),
        before: Vec::new(),
        after: Vec::new(),
        matches: Vec::new(),
        pages: Vec::new(),
        status,
        exact: status == PdfLookupStatus::Found,
        payload_sha256: String::new(),
        page_text_sha256: String::new(),
        error,
    }
}

fn nfkc_text(value: &str) -> Cow<'_, str> {
    if value.is_ascii() {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(value.nfkc().collect())
    }
}

fn join_page_lines(lines: Vec<Line>) -> String {
    let mut output = String::with_capacity(lines.iter().map(|line| line.text.len() + 1).sum());
    for line in lines {
        let text = line.text.trim();
        if text.is_empty() {
            continue;
        }
        if output.ends_with('-') && text.chars().next().is_some_and(char::is_lowercase) {
            output.pop();
        } else if !output.is_empty() {
            output.push(' ');
        }
        output.push_str(text);
    }
    output
}

fn range_text(text: &ScalarText<'_>, ranges: &[ScalarRange]) -> String {
    ranges.iter().filter_map(|range| text.slice_utf16(range.start..range.end))
        .filter(|part| !part.is_empty()).collect::<Vec<_>>().join("\n")
}

fn rendered_slice(text: &ScalarText<'_>, document: &DocumentStructure, node: &StructureNode) -> String {
    range_text(text, &document.query_ranges(node))
}

fn section_nodes(document: &PdfDocument) -> Vec<&StructureNode> {
    document
        .structure
        .nodes
        .iter()
        // A heading's or a constituent document's extent is no provision: it has no identifier
        // to look up.
        .filter(|node| {
            node.kind == NodeKind::Section
                && node
                    .locator_kind
                    .as_deref()
                    .is_some_and(|kind| !kind.is_empty())
        })
        .collect()
}

fn nodes_by_id(document: &PdfDocument) -> HashMap<&str, &StructureNode> {
    document
        .structure
        .nodes
        .iter()
        .map(|node| (node.id.as_str(), node))
        .collect()
}

/// A declared paragraph as a lookup unit, retaining its graph identity.
fn printed_unit(
    text: &ScalarText<'_>,
    document: &PdfDocument,
    paragraph: &PrintedParagraph,
) -> PdfLookupUnit {
    let page = paragraph
        .page_numbers
        .first()
        .and_then(|number| document.pages.iter().find(|page| page.number == *number));
    PdfLookupUnit {
        id: paragraph.id.clone(),
        kind: "paragraph",
        locator: format!("paragraph {}", paragraph.number),
        text: range_text(text, &paragraph.ranges),
        page_numbers: paragraph.page_numbers.clone(),
        confidence: page.map(|item| item.text_quality.clamp(0.0, 1.0)),
        confidence_basis: if page.is_some() {
            "page_text_quality"
        } else {
            "unavailable"
        },
        provenance: "legalpdf:printed".to_owned(),
        proposition: None,
        note: None,
    }
}

fn page_unit(
    document: &DocumentStructure,
    text: &ScalarText<'_>,
    nodes: &HashMap<&str, &StructureNode>,
    page: &ProjectionPage,
) -> PdfLookupUnit {
    let text = nodes
        .get(page.id.as_str())
        .copied()
        .map(|node| rendered_slice(text, document, node))
        .unwrap_or_default();
    PdfLookupUnit {
        id: if page.id.is_empty() {
            format!("page-{}", page.number)
        } else {
            page.id.clone()
        },
        kind: "page",
        locator: format!("[page {}]", page.number),
        text,
        page_numbers: vec![page.number],
        confidence: Some(page.text_quality.clamp(0.0, 1.0)),
        confidence_basis: "page_text_quality",
        provenance: if page.source.is_empty() {
            "unknown".to_owned()
        } else {
            page.source.clone()
        },
        proposition: None,
        note: None,
    }
}

fn footnote_unit(note: &ProjectionFootnote) -> PdfLookupUnit {
    let mut page_numbers = Vec::new();
    if let Some(page) = note.reference_page {
        page_numbers.push(page);
    }
    for page in &note.body_pages {
        if !page_numbers.contains(page) {
            page_numbers.push(*page);
        }
    }
    PdfLookupUnit {
        id: note.pair_id.clone(),
        kind: "footnote",
        locator: format!("footnote {}", note.label),
        text: note.body.trim().to_owned(),
        page_numbers,
        confidence: Some(note.confidence.clamp(0.0, 1.0)),
        confidence_basis: "footnote_pairing",
        provenance: if note.provenance.is_empty() {
            "unknown".to_owned()
        } else {
            note.provenance.clone()
        },
        proposition: Some(PdfLookupProposition {
            sentence: note.sentence_proposition.trim().to_owned(),
            passage_since_prior_note: note.passage_since_prior_note.trim().to_owned(),
        }),
        note: Some(PdfLookupNote {
            label: note.label.clone(),
            occurrence: note.occurrence,
            restart_sequence: note.restart_sequence,
            reference_page: note.reference_page,
            body_pages: note.body_pages.clone(),
            warnings: note.warnings.clone(),
        }),
    }
}

fn section_unit(
    document: &DocumentStructure,
    text: &ScalarText<'_>,
    pages: &HashMap<usize, &ProjectionPage>,
    section: &StructureNode,
    index: usize,
) -> PdfLookupUnit {
    let mut page_numbers = Vec::new();
    let mut confidence: Option<f64> = None;
    for page_index in &section.page_indexes {
        if let Some(page) = pages.get(page_index) {
            if !page_numbers.contains(&page.number) {
                page_numbers.push(page.number);
            }
            let quality = page.text_quality.clamp(0.0, 1.0);
            confidence = Some(confidence.map_or(quality, |value| value.min(quality)));
        }
    }
    PdfLookupUnit {
        id: if section.id.is_empty() {
            format!("section-{}", index + 1)
        } else {
            section.id.clone()
        },
        kind: "section",
        locator: section
            .label
            .as_deref()
            .unwrap_or(&section.id)
            .trim()
            .to_owned(),
        text: rendered_slice(text, document, section).trim().to_owned(),
        page_numbers,
        confidence,
        confidence_basis: if confidence.is_some() {
            "minimum_page_text_quality"
        } else {
            "unavailable"
        },
        provenance: match section.source {
            Derivation::Native => "native",
            Derivation::Heuristic => "legal-structure",
            Derivation::Model => "model",
        }
        .to_owned(),
        proposition: None,
        note: None,
    }
}


pub fn numeric_range(kind: &str, raw: &str) -> Option<(usize, usize)> {
    static RANGE: OnceLock<Regex> = OnceLock::new();
    static PAGE_PREFIX: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH_PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    let value = value
        .trim()
        .trim_start_matches('#')
        .trim()
        .trim_start_matches('[')
        .trim_end_matches(']')
        .trim();
    let prefix = if kind == "page" {
        PAGE_PREFIX.get_or_init(|| Regex::new(r"(?i)^(?:pages?|pp?\.?)[\s:_=-]*").unwrap())
    } else {
        PARAGRAPH_PREFIX
            .get_or_init(|| Regex::new(r"(?i)^(?:paragraphs?|paras?|pars?)\.?[\s:_=-]*").unwrap())
    };
    let stripped = prefix.replace(value, "");
    let captures = RANGE
        .get_or_init(|| Regex::new(r"(?i)^(\d{1,6})\s*(?:-|–|—|\.\.|to)\s*(\d{1,6})$").unwrap())
        .captures(&stripped)?;
    Some((captures[1].parse().ok()?, captures[2].parse().ok()?))
}

fn normalize_footnote(raw: &str) -> String {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:footnotes?|notes?|fn)\s*[#.:_-]?\s*").unwrap())
        .replace(value.trim(), "")
        .to_lowercase()
}

fn normalized_section(raw: &str) -> Option<String> {
    static PREFIX: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    let compact: String = PREFIX
        .get_or_init(|| Regex::new(r"(?i)^(?:ss?\.?|sections?)\s*").unwrap())
        .replace(value.trim(), "")
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect();
    let normalized = normalize_compact_numbered_section_locator(&compact);
    (!normalized.is_empty()).then_some(normalized)
}

fn section_alias(raw: &str) -> String {
    normalized_section(raw).unwrap_or_else(|| {
        let value = nfkc_text(raw);
        value
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
            .to_lowercase()
    })
}

fn section_matches(section: &StructureNode, requested_kind: &str, requested: &str) -> bool {
    if requested_kind != "section" && section.locator_kind.as_deref() != Some(requested_kind) {
        return false;
    }
    if requested_kind == "section"
        && !section.id.is_empty()
        && section_alias(&format!("section:{}", section.id)) == requested
    {
        return true;
    }
    section
        .label
        .iter()
        .map(String::as_str)
        .chain(section.aliases.iter().flatten().map(String::as_str))
        .any(|value| section_alias(value) == requested)
}

fn exact_footnotes(document: &PdfDocument, locator: &str, input: &PdfLookupRequest) -> Vec<usize> {
    let query = normalize_footnote(locator);
    document
        .footnotes
        .iter()
        .enumerate()
        .filter(|(_, note)| {
            input
                .occurrence
                .is_none_or(|number| note.occurrence == number)
                && input.page.is_none_or(|number| {
                    note.reference_page == Some(number) || note.body_pages.contains(&number)
                })
                && (normalize_footnote(&note.pair_id) == query
                    || normalize_footnote(&note.label) == query)
        })
        .map(|(index, _)| index)
        .collect()
}

fn finish<F>(
    document: &PdfDocument,
    input: &PdfLookupRequest,
    ordered_len: usize,
    selected_start: usize,
    selected_end: usize,
    mut unit_at: F,
) -> PdfStructureLookup
where
    F: FnMut(usize) -> PdfLookupUnit,
{
    if selected_end - selected_start + 1 > MAX_UNITS {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!("Exact ranges are limited to {MAX_UNITS} units")),
        );
    }
    let before: Vec<_> = (selected_start.saturating_sub(input.context_blocks)..selected_start)
        .map(&mut unit_at)
        .collect();
    let selected: Vec<_> = (selected_start..=selected_end).map(&mut unit_at).collect();
    let after: Vec<_> = (selected_end + 1
        ..usize::min(ordered_len, selected_end + 1 + input.context_blocks))
        .map(unit_at)
        .collect();
    if selected.iter().any(|unit| unit.text.is_empty()) {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some("The requested structural unit has no exact text".to_owned()),
        );
    }
    if before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .map(|unit| utf16_len(&unit.text))
        .sum::<usize>()
        > MAX_RETURN_CHARS
    {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some(format!(
                "Exact result exceeds {MAX_RETURN_CHARS} characters; request a narrower range"
            )),
        );
    }
    let page_numbers: HashSet<_> = before
        .iter()
        .chain(selected.iter())
        .chain(after.iter())
        .flat_map(|unit| unit.page_numbers.iter().copied())
        .collect();
    let pages = document
        .pages
        .iter()
        .filter(|page| page_numbers.contains(&page.number) && !page.evidence_text.is_empty())
        .map(|page| PdfLookupPage {
            page_number: page.number,
            text: page.evidence_text.clone(),
        })
        .collect();
    let matches = selected.iter().map(|unit| unit.id.clone()).collect();
    let mut lookup = result(input, PdfLookupStatus::Found, None);
    lookup.units = selected;
    lookup.before = before;
    lookup.after = after;
    lookup.matches = matches;
    lookup.pages = pages;
    let payload_sha256 = lookup_payload_sha256(&lookup);
    let page_text_sha256 = serialization_sha256(&lookup.pages);
    lookup.payload_sha256 = payload_sha256;
    lookup.page_text_sha256 = page_text_sha256;
    lookup
}

fn structure_lookup(document: &PdfDocument, input: &PdfLookupRequest) -> PdfStructureLookup {
    if !input.valid() {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some("Invalid or unbounded PDF locator".to_owned()),
        );
    }
    let kind = match input.locator_kind.as_str() {
        "page" | "paragraph" | "footnote" => input.locator_kind.as_str(),
        _ => "section",
    };
    let end_locator = input
        .end_locator
        .as_deref()
        .filter(|value| !value.is_empty());
    if kind == "page" || kind == "paragraph" {
        let inline = numeric_range(kind, &input.locator);
        let start_number = inline
            .map(|item| item.0)
            .or_else(|| parse_ordinal(kind, &input.locator));
        let end_number = end_locator
            .map(|end| parse_ordinal(kind, end))
            .unwrap_or_else(|| inline.map(|item| item.1).or(start_number));
        let (Some(start_number), Some(end_number)) = (start_number, end_number) else {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        };
        if start_number > end_number {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Invalid exact range".to_owned()),
            );
        }
        if kind == "page" {
            let position = |number| {
                u32::try_from(number)
                    .ok()
                    .and_then(|number| document.pages.iter().position(|page| page.number == number))
            };
            let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
                return result(input, PdfLookupStatus::NotFound, None);
            };
            if end < start {
                return result(input, PdfLookupStatus::NotFound, None);
            }
            let text = ScalarText::new(document.structure.query_text());
            let nodes = nodes_by_id(document);
            return finish(document, input, document.pages.len(), start, end, |index| {
                page_unit(&document.structure, &text, &nodes, &document.pages[index])
            });
        }
        if end_number - start_number >= MAX_UNITS {
            return result(
                input,
                PdfLookupStatus::Invalid,
                Some("Paragraph range exceeds the lookup limit".to_owned()),
            );
        }
        let selected = match document.paragraph_range(start_number, end_number) {
            Ok(selected) => selected,
            Err(status) => return result(input, status, None),
        };
        let printed = document.printed_paragraphs();
        let start = printed
            .iter()
            .position(|p| std::ptr::eq(p, &selected[0]))
            .unwrap();
        let end = start + selected.len() - 1;
        let text = ScalarText::new(document.structure.query_text());
        return finish(document, input, printed.len(), start, end, |index| {
            printed_unit(&text, document, &printed[index])
        });
    }
    if kind == "footnote" {
        let start = exact_footnotes(document, &input.locator, input);
        let end = end_locator.map(|locator| exact_footnotes(document, locator, input));
        let end_matches = end.as_deref().unwrap_or(&start);
        if start.len() > 1 || end_matches.len() > 1 {
            let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
            for pair_id in start
                .iter()
                .chain(end.iter().flatten())
                .map(|index| document.footnotes[*index].pair_id.as_str())
            {
                if !lookup.matches.iter().any(|item| item == pair_id) {
                    lookup.matches.push(pair_id.to_owned());
                }
            }
            return lookup;
        }
        let (Some(start), Some(end)) = (start.first().copied(), end_matches.first().copied())
        else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        if end < start {
            return result(input, PdfLookupStatus::NotFound, None);
        }
        return finish(
            document,
            input,
            document.footnotes.len(),
            start,
            end,
            |index| footnote_unit(&document.footnotes[index]),
        );
    }
    if end_locator.is_some() {
        return result(
            input,
            PdfLookupStatus::Invalid,
            Some("Section ranges are not supported by this document contract".to_owned()),
        );
    }
    let sections = section_nodes(document);
    if input.locator_kind != "section"
        && !sections
            .iter()
            .any(|section| section.locator_kind.as_deref() == Some(input.locator_kind.as_str()))
    {
        return result(
            input,
            PdfLookupStatus::Unavailable,
            Some(format!(
                "No exact {} identifiers exist in this source PDF",
                input.locator_kind
            )),
        );
    }
    let requested = section_alias(&input.locator);
    let candidates: Vec<_> = sections
        .iter()
        .enumerate()
        .filter(|(_, section)| section_matches(section, &input.locator_kind, &requested))
        .map(|(index, _)| index)
        .collect();
    if candidates.len() > 1 {
        let mut lookup = result(input, PdfLookupStatus::Ambiguous, None);
        lookup.matches = candidates
            .into_iter()
            .map(|index| sections[index].id.clone())
            .collect();
        return lookup;
    }
    let Some(index) = candidates.first().copied() else {
        return result(input, PdfLookupStatus::NotFound, None);
    };
    let text = ScalarText::new(document.structure.query_text());
    let pages: HashMap<_, _> = document
        .pages
        .iter()
        .map(|page| (page.index, page))
        .collect();
    finish(document, input, sections.len(), index, index, |index| {
        section_unit(&document.structure, &text, &pages, sections[index], index)
    })
}

fn append(text: &mut String, position: &mut usize, value: &str) {
    text.push_str(value);
    *position += utf16_len(value);
}

fn extend_range(range: Option<ScalarRange>, next: ScalarRange) -> Option<ScalarRange> {
    Some(range.map_or(next, |range| ScalarRange {
        start: range.start.min(next.start),
        end: range.end.max(next.end),
    }))
}

/// Preserve a native producer's witnessed dehyphenation as a manuscript-only edit.
fn attach_manuscript_edits(paragraphs: &[Paragraph], structure: &mut DocumentStructure)
    -> Vec<(legal_structure_model::SourceJoin, Option<legal_structure_model::SourceContinuation>)> {
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let mut edits = legal_structure_model::ManuscriptEdits::new();
    let mut joins = Vec::new();
    let mut diagnostics = Vec::new();
    let mut node_at = HashMap::<&str, usize>::new();
    for (index, node) in structure.nodes.iter().enumerate() {
        node_at.entry(node.id.as_str()).or_insert(index);
    }
    for paragraph in paragraphs {
        let Some(node) = node_at.get(paragraph.id.as_str()).map(|&index| &structure.nodes[index]) else { continue; };
        let own = if matches!(node.kind, NodeKind::Heading)
            || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
            node.heading_range.as_ref().map(std::slice::from_ref)
        } else { None }.unwrap_or_else(|| {
            if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges }
        });
        let output = ScalarText::new(&paragraph.text);
        for pair in paragraph.source_spans.windows(2) {
            let left = &pair[0];
            let right = &pair[1];
            let discover = || {
                if left.line_id == right.line_id || output.slice_utf16(left.end..right.start) != Some("") {
                    return None;
                }
                let a = *fragments.get(left.line_id.as_str())?;
                let b = *fragments.get(right.line_id.as_str())?;
                for (span, fragment) in [(left, a), (right, b)] {
                    let start = fragment.range.start.checked_add(span.source_start)?;
                    let end = fragment.range.start.checked_add(span.source_end)?;
                    if span.start >= span.end || span.source_start >= span.source_end
                        || end > fragment.range.end || span.end - span.start != end - start
                        || !own.iter().any(|range| range.start <= fragment.range.start && fragment.range.end <= range.end)
                        || output.slice_utf16(span.start..span.end).is_none()
                        || output.slice_utf16(span.start..span.end) != source.slice_utf16(start..end) {
                        return None;
                    }
                }
                let original = source.slice_utf16(a.range.start..a.range.end)?;
                let coordinates = ScalarText::new(original);
                let omitted_end = left.source_end.checked_add(1)?;
                let omitted = coordinates.slice_utf16(left.source_end..omitted_end)?;
                if !matches!(omitted, "-" | "\u{00ad}" | "\u{00ac}")
                    || omitted_end != a.range.end - a.range.start
                    || right.source_start != 0
                    || !output.slice_utf16(right.start..right.end)?.chars().next().is_some_and(char::is_lowercase) {
                    return None;
                }
                let start = coordinates.scalar_at_utf16(left.source_end)?;
                let end = coordinates.scalar_at_utf16(omitted_end)?;
                let edit = (start, end, String::new());
                if structure.manuscript_edits.get(&left.line_id).is_some_and(|existing| !existing.contains(&edit)) {
                    return None;
                }
                {
                    // No competing leaf ownership or intervening member may change these seed starts.
                    if a.range.start >= b.range.start || structure.source_fragments.iter().any(|fragment|
                        a.range.start < fragment.range.start && fragment.range.start < b.range.start
                        && own.iter().any(|range| range.start < fragment.range.end && fragment.range.start < range.end))
                        || structure.nodes.iter().any(|other| other.id != node.id
                            && (other.role.is_some() || matches!(other.kind,
                                NodeKind::Prose | NodeKind::Paragraph | NodeKind::Section | NodeKind::ListItem
                                | NodeKind::Heading | NodeKind::Footnote | NodeKind::Endnote))
                            && (if other.ranges.is_empty() { std::slice::from_ref(&other.range) } else { &other.ranges })
                                .iter().any(|range| [a.range, b.range].iter().any(|part|
                                    range.start < part.end && part.start < range.end))) {
                        return None;
                    }
                    if structure.joins.iter().any(|join| join.from == a.range.start
                        && (join.to != b.range.start || !join.separator.is_empty())) { return None; }
                }
                let continuation = if a.page_index != b.page_index {
                    let first = |page| structure.source_fragments.iter().filter(|fragment| fragment.page_index == page)
                        .flat_map(|fragment| own.iter().filter_map(move |range| {
                            let start = range.start.max(fragment.range.start);
                            (start < range.end.min(fragment.range.end)).then_some(start)
                        })).min();
                    Some(legal_structure_model::SourceContinuation {
                        kind: if node.kind == NodeKind::Heading || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
                            legal_structure_model::ContinuationKind::Heading
                        } else { legal_structure_model::ContinuationKind::Paragraph },
                        from: first(a.page_index)?, to: first(b.page_index)?, separator: Some(String::new()),
                    })
                } else { None };
                Some((a, b, edit, ScalarRange {
                    start: a.range.start + left.source_end, end: a.range.start + omitted_end,
                }, continuation))
            };
            let Some((a, b, edit, range, continuation)) = discover() else { continue; };
            if !structure.manuscript_edits.contains_key(&left.line_id) {
                let line = edits.entry(left.line_id.clone()).or_default();
                if !line.contains(&edit) { line.push(edit); }
                diagnostics.push(legal_structure_model::StructureDiagnostic {
                    code: "native_manuscript_dehyphenation".to_owned(),
                    severity: legal_structure_model::DiagnosticSeverity::Info,
                    ranges: vec![range], node_ids: vec![node.id.clone()],
                });
            }
            if !structure.joins.iter().any(|join| join.from == a.range.start)
                && !joins.iter().any(|(join, _): &(legal_structure_model::SourceJoin, _)| join.from == a.range.start) {
                joins.push((legal_structure_model::SourceJoin { from: a.range.start,
                    to: b.range.start, separator: String::new() }, continuation));
            }
        }
    }
    for line in edits.values_mut() { line.sort_by_key(|edit| (edit.0, edit.1)); }
    structure.manuscript_edits.extend(edits);
    structure.diagnostics.extend(diagnostics);
    joins
}

/// Retain only fully mapped page seams from a single native logical producer.
/// Source endpoints name fragment starts; producer run ends only delimit the separator.
fn attach_source_continuations(
    paragraphs: &[Paragraph],
    footnotes: &[Footnote],
    structure: &mut DocumentStructure,
) {
    if structure.continuations.is_some() {
        return;
    }
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let nodes: HashMap<_, _> = structure.nodes.iter()
        .map(|node| (node.id.as_str(), node)).collect();
    // The source lines by where they start, each with the furthest end so far, so a node finds
    // the lines it overlaps without reading every line.
    let mut by_start = (0..structure.source_fragments.len()).collect::<Vec<_>>();
    by_start.sort_by_key(|&slot| structure.source_fragments[slot].range.start);
    let reach = by_start.iter().scan(0, |end, &slot| {
        *end = (*end).max(structure.source_fragments[slot].range.end);
        Some(*end)
    }).collect::<Vec<_>>();
    let links = |node: &StructureNode, produced: &str,
                 spans: &[legal_pdf_core::model::SourceTextSpan], kind| {
        let heading = node.kind == NodeKind::Heading
            || node.role == Some(legal_structure_model::PrimaryRole::Heading);
        let own = heading.then_some(node.heading_range.as_ref()).flatten()
            .map(std::slice::from_ref).unwrap_or_else(|| {
                if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges }
            });
        let output = ScalarText::new(produced);
        let mut starts = HashMap::<usize, usize>::new();
        let mut ends = HashMap::<usize, usize>::new();
        let mut member_lines = HashSet::new();
        for range in own {
            let upto = by_start.partition_point(|&slot| structure.source_fragments[slot].range.start < range.end);
            let from = reach[..upto].partition_point(|end| *end <= range.start);
            for fragment in by_start[from..upto].iter().map(|&slot| &structure.source_fragments[slot]) {
                let start = range.start.max(fragment.range.start);
                let end = range.end.min(fragment.range.end);
                if start < end {
                    starts.entry(fragment.page_index).and_modify(|at| *at = (*at).min(start)).or_insert(start);
                    ends.entry(fragment.page_index).and_modify(|at| *at = (*at).max(end)).or_insert(end);
                    member_lines.insert(fragment.id.as_str());
                }
            }
        }
        let mut pages = Vec::<(usize, usize, usize)>::new();
        let mut mapped_lines = HashSet::new();
        let mut prior_end = 0;
        for (index, span) in spans.iter().enumerate() {
            let fragment = *fragments.get(span.line_id.as_str())?;
            let start = fragment.range.start.checked_add(span.source_start)?;
            let end = fragment.range.start.checked_add(span.source_end)?;
            if span.start < prior_end || span.start >= span.end || span.source_start >= span.source_end
                || end > fragment.range.end || span.end - span.start != end - start
                || !own.iter().any(|range| range.start <= start && end <= range.end)
                || output.slice_utf16(span.start..span.end).is_none()
                || output.slice_utf16(span.start..span.end) != source.slice_utf16(start..end) {
                return None;
            }
            prior_end = span.end;
            mapped_lines.insert(span.line_id.as_str());
            if let Some(last) = pages.last_mut().filter(|last| last.0 == fragment.page_index) {
                last.2 = index;
            } else {
                if pages.iter().any(|page| page.0 == fragment.page_index) { return None; }
                pages.push((fragment.page_index, index, index));
            }
        }
        // A partial producer map must not split the native object into partial facts.
        if mapped_lines != member_lines || pages.len() != starts.len() { return None; }
        let mut result = Vec::new();
        for pair in pages.windows(2) {
            let from = *starts.get(&pair[0].0)?;
            let to = *starts.get(&pair[1].0)?;
            let left = &spans[pair[0].2];
            let right = &spans[pair[1].1];
            let separator = output.slice_utf16(left.end..right.start)?;
            let left_end = fragments.get(left.line_id.as_str())?.range.start + left.source_end;
            let right_start = fragments.get(right.line_id.as_str())?.range.start + right.source_start;
            let suffix = source.slice_utf16(left_end..*ends.get(&pair[0].0)?)?;
            let prefix = source.slice_utf16(to..right_start)?;
            let left_fragment = fragments.get(left.line_id.as_str())?;
            let left_text = ScalarText::new(source.slice_utf16(left_fragment.range.start..left_fragment.range.end)?);
            let scalar = left_text.scalar_at_utf16(left.source_end)?;
            let removed_hyphen = suffix.chars().next().is_some_and(|glyph| matches!(glyph, '-' | '\u{00ad}' | '\u{00ac}'))
                && suffix.chars().count() == 1 && prefix.is_empty()
                && structure.manuscript_edits.get(&left.line_id).is_some_and(|edits|
                    edits.iter().any(|(start, end, replacement)| *start == scalar && *end == scalar + 1 && replacement.is_empty()));
            // Only an explicit matching manuscript edit may remove the printed seam glyph.
            if from >= to || !separator.chars().all(char::is_whitespace)
                || !(suffix.chars().all(char::is_whitespace) || removed_hyphen)
                || !prefix.chars().all(char::is_whitespace) {
                return None;
            }
            result.push(legal_structure_model::SourceContinuation {
                kind, from, to, separator: Some(separator.to_owned()),
            });
        }
        Some(result)
    };
    let mut continuations = Vec::new();
    for paragraph in paragraphs {
        if let Some(node) = nodes.get(paragraph.id.as_str()) {
            let kind = if node.kind == NodeKind::Heading
                || node.role == Some(legal_structure_model::PrimaryRole::Heading) {
                legal_structure_model::ContinuationKind::Heading
            } else { legal_structure_model::ContinuationKind::Paragraph };
            if let Some(found) = links(node, &paragraph.text, &paragraph.source_spans, kind) {
                continuations.extend(found);
            }
        }
    }
    let mut notes = HashMap::<&str, &legal_structure_model::Note>::new();
    for entry in &structure.notes {
        notes.entry(entry.id.as_str()).or_insert(entry);
    }
    for note in footnotes {
        let node = notes.get(note.pair_id.as_str())
            .and_then(|entry| entry.node_id.as_deref()).and_then(|id| nodes.get(id));
        if let Some(node) = node {
            if let Some(found) = links(node, &note.body, &note.source_spans,
                legal_structure_model::ContinuationKind::Note) {
                continuations.extend(found);
            }
        }
    }
    structure.continuations = Some(continuations);
}

struct RenderedSourceLine<'a> {
    source: ScalarRange,
    target: ScalarRange,
    text: &'a str,
    edits: &'a [(usize, usize, String)],
    trimmed_start: usize,
    trimmed_end: usize,
}

impl RenderedSourceLine<'_> {
    fn boundary(&self, source_offset: usize, end: bool) -> legal_pdf_core::Result<usize> {
        let original = ScalarText::new(self.text);
        let local = source_offset - self.source.start;
        let scalar = original.scalar_at_utf16(local).ok_or_else(||
            legal_pdf_core::Error::Message("Manuscript membership splits a source character".into()))?;
        let mut removed = 0;
        let mut inserted = 0;
        for (start, stop, replacement) in self.edits {
            if *start < scalar && scalar < *stop {
                return Err(legal_pdf_core::Error::Message(
                    "Manuscript membership splits a normalization edit".into()));
            }
            // Boundary insertions belong to the following member, except at the line end.
            if *stop < scalar || *stop == scalar && (*start < *stop || end && local == utf16_len(self.text)) {
                removed += original.utf16_at_scalar(*stop).unwrap() - original.utf16_at_scalar(*start).unwrap();
                inserted += utf16_len(replacement);
            }
        }
        let position = local - removed + inserted;
        Ok(self.target.start + position.clamp(self.trimmed_start, self.trimmed_end) - self.trimmed_start)
    }
}

fn manuscript_line(text: &str, edits: &[(usize, usize, String)]) -> legal_pdf_core::Result<String> {
    let source = ScalarText::new(text);
    let mut output = String::new();
    let mut cursor = 0;
    let mut prior_start = None;
    for (start, end, replacement) in edits {
        if start > end || *start < cursor || prior_start.is_some_and(|prior| *start <= prior) {
            return Err(legal_pdf_core::Error::Message("Invalid manuscript edit order".into()));
        }
        let unchanged = source.slice(cursor..*start).ok_or_else(||
            legal_pdf_core::Error::Message("Manuscript edit starts outside its source line".into()))?;
        if source.utf16_at_scalar(*end).is_none() {
            return Err(legal_pdf_core::Error::Message("Manuscript edit ends outside its source line".into()));
        }
        output.push_str(unchanged);
        output.push_str(replacement);
        cursor = *end;
        prior_start = Some(*start);
    }
    output.push_str(source.slice_utf16(source.utf16_at_scalar(cursor).unwrap()..source.utf16_len()).unwrap());
    Ok(output)
}

fn attach_rendered_text(
    paragraphs: &[Paragraph],
    pages: &[Page],
    footnotes: &[Footnote],
    structure: &mut DocumentStructure,
) -> legal_pdf_core::Result<()> {
    let source = ScalarText::new(&structure.text);
    let fragments: HashMap<_, _> = structure.source_fragments.iter()
        .map(|fragment| (fragment.id.as_str(), fragment)).collect();
    let mut text = String::with_capacity(structure.text.len());
    let mut mappings = Vec::new();
    let mut owners = HashMap::new();
    for paragraph in paragraphs {
        for line in &paragraph.line_ids {
            owners.insert(line.as_str(), paragraph.id.as_str());
        }
    }
    let mut note_lines = HashSet::new();
    for note in footnotes {
        for line in &note.body_line_ids {
            owners.insert(line.as_str(), note.pair_id.as_str());
            note_lines.insert(line.as_str());
        }
    }
    let mut position = 0;
    let mut previous_source_start = None;
    for page in pages {
        let mut lines: Vec<_> = page.lines.iter().collect();
        lines.sort_by_key(|line| (line.reading_order, line.source_index));
        let mut previous_owner = None;
        for notes in [false, true] {
            if notes { previous_owner = None; }
            for line in lines.iter().filter(|line| {
                (note_lines.contains(line.id.as_str())
                    || matches!(line.region_type.as_str(), "footnote" | "endnote" | "note")) == notes
            }) {
                if line.text.trim().is_empty() && !structure.manuscript_edits.contains_key(&line.id) { continue; }
                let fragment = fragments.get(line.id.as_str()).ok_or_else(||
                    legal_pdf_core::Error::Message(format!("Missing PDF source fragment {}", line.id)))?;
                if fragment.page_index != page.index
                    || source.slice_utf16(fragment.range.start..fragment.range.end) != Some(line.text.as_str()) {
                    return Err(legal_pdf_core::Error::Message(format!("PDF source fragment {} no longer matches its extraction", line.id)));
                }
                let edits = structure.manuscript_edits.get(&line.id).map(Vec::as_slice).unwrap_or_default();
                let normalized = manuscript_line(&line.text, edits)?;
                let trimmed = normalized.trim();
                // An empty physical member owns no insertion, matching manuscript span projection.
                if line.text.is_empty() || trimmed.is_empty() { continue; }
                let owner = owners.get(line.id.as_str()).copied().unwrap_or(line.id.as_str());
                if !text.is_empty() {
                    let separator = structure.joins.iter().find(|join|
                        Some(join.from) == previous_source_start && join.to == fragment.range.start)
                        .map(|join| join.separator.as_str())
                        .unwrap_or(if previous_owner == Some(owner) { " " } else { "\n\n" });
                    append(&mut text, &mut position, separator);
                }
                let start = position;
                let leading = utf16_len(&normalized[..normalized.len() - normalized.trim_start().len()]);
                append(&mut text, &mut position, trimmed);
                mappings.push(RenderedSourceLine { source: fragment.range,
                    target: ScalarRange { start, end: position }, text: &line.text, edits,
                    trimmed_start: leading, trimmed_end: leading + utf16_len(trimmed) });
                previous_owner = Some(owner);
                previous_source_start = Some(fragment.range.start);
            }
        }
    }
    mappings.sort_by_key(|mapping| mapping.source.start);
    let rendered = ScalarText::new(&text);
    let project = |membership: &[ScalarRange]| -> legal_pdf_core::Result<Vec<ScalarRange>> {
        let mut ranges = Vec::<ScalarRange>::new();
        for range in membership {
            let first = mappings.partition_point(|mapping| mapping.source.end <= range.start);
            for mapping in mappings[first..].iter().take_while(|mapping| mapping.source.start < range.end) {
                let start = range.start.max(mapping.source.start);
                let end = range.end.min(mapping.source.end);
                if start < end {
                    let part = ScalarRange { start: mapping.boundary(start, false)?,
                        end: mapping.boundary(end, true)? };
                    if part.start < part.end { ranges.push(part); }
                }
            }
        }
        ranges.sort_by_key(|range| (range.start, range.end));
        let mut exact = Vec::<ScalarRange>::new();
        for range in ranges {
            if let Some(prior) = exact.last_mut().filter(|prior| range.start <= prior.end
                || rendered.slice_utf16(prior.end..range.start).is_some_and(|gap| gap.chars().all(char::is_whitespace))) {
                prior.end = prior.end.max(range.end);
            } else {
                exact.push(range);
            }
        }
        Ok(exact)
    };
    for node in &mut structure.nodes {
        let own = if node.ranges.is_empty() { std::slice::from_ref(&node.range) } else { &node.ranges };
        node.rendered_membership = project(own)?;
        let membership = if (matches!(node.kind, NodeKind::Heading | NodeKind::Section)
            || node.role == Some(legal_structure_model::PrimaryRole::Heading)) && !node.extent.is_empty() {
            node.extent.as_slice()
        } else { own };
        let exact = project(membership)?;
        // An empty mapping stays empty; a first-page or source-envelope fallback would invent membership.
        node.rendered_range = Some(exact.iter().copied().fold(None, extend_range)
            .unwrap_or(ScalarRange { start: 0, end: 0 }));
        node.rendered_ranges = exact;
    }
    structure.rendered_text = (text != structure.text).then_some(text);
    structure.refresh_revision()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LegalDocument;
    use legal_pdf_core::model::{ScalarRange, SCHEMA_VERSION};
    use serde_json::json;

    fn structure_graph(nodes: Vec<StructureNode>) -> DocumentStructure {
        serde_json::from_value(json!({
            "schema_version": "legalpdf.document-structure.v1",
            "document_id": "doc",
            "offset_unit": "utf16",
            "provider": "local-pdf",
            "revision": "00",
            "text": "",
            "text_sha256": "00",
            "source_sha256": "00",
            "scope": {"kind": "complete"},
            "origins": [],
            "nodes": nodes,
            "diagnostics": []
        }))
        .unwrap()
    }

    fn node(id: &str, kind: NodeKind) -> StructureNode {
        StructureNode::new(
            id.to_owned(),
            kind,
            ScalarRange { start: 0, end: 0 },
            "test",
            Derivation::Native,
            None,
        )
    }

    fn source_line(id: &str, text: &str) -> Line {
        serde_json::from_value(json!({
            "id": id, "page_index": 0, "page_number": 1,
            "source_index": 1, "reading_order": 1, "block_index": 1,
            "text": text, "bbox": [60.0, 100.0, 300.0, 112.0],
            "region_type": "body", "source": "native"
        }))
        .unwrap()
    }

    fn pdf_summary() -> PdfSummary {
        PdfSummary {
            embedded_page_labels: Vec::new(),
            sha256: "00".repeat(32),
            parser_version: PARSER_VERSION.to_owned(),
            cache_key: "cache".to_owned(),
            page_count: 1,
            projection_page_count: 1,
            status: "ready".to_owned(),
            pages_needing_ocr: vec![],
            ocr_routed_pages: vec![],
            scanned_pages: vec![],
        }
    }

    #[test]
    fn projection_preserves_source_text_and_moves_only_notes_to_page_end() {
        let mut body = source_line("body", "44. Repeated wording\u{1f600}1");
        body.reading_order = 0;
        let mut note = source_line("note", "1 Repeated wording\u{1f600}");
        note.reading_order = 1;
        note.region_type = "footnote".into();
        let mut unclassified = source_line("unclassified", "Unclassified source text");
        unclassified.reading_order = 2;
        unclassified.region_type = "unknown".into();
        unclassified.exclude_from_body = true;
        let pages = vec![Page {
            id: "page-1".into(),
            index: 0,
            number: 1,
            width: 612.0,
            height: 792.0,
            lines: vec![note, unclassified, body],
            regions: vec![],
            source: "native".into(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];
        let mut nodes = vec![];
        for id in ["body", "note", "unclassified"] {
            let mut item = node(id, NodeKind::Prose);
            item.line_ids = vec![id.into()];
            nodes.push(item);
        }
        let mut graph = structure_graph(nodes);
        attach_rendered_text(&[], &pages, &[], &mut graph);
        let text = graph.rendered_text.as_ref().unwrap();
        assert_eq!(text, "44. Repeated wording\u{1f600}1\n\nUnclassified source text\n\n1 Repeated wording\u{1f600}");
        let encoded: Vec<u16> = text.encode_utf16().collect();
        for item in &graph.nodes {
            let range = item.rendered_range.unwrap();
            let restored = String::from_utf16(&encoded[range.start..range.end]).unwrap();
            let original = pages[0]
                .lines
                .iter()
                .find(|line| line.id == item.id)
                .unwrap();
            assert_eq!(restored, original.text);
        }

        let native =
            PdfDocument::project(pages.clone(), vec![], vec![], graph.clone(), pdf_summary());
        assert!(native.recognized_pages().is_empty());
        assert!(native.has_recognized_geometry());
        let mut old_cache = serde_json::to_value(&native).unwrap();
        old_cache
            .as_object_mut()
            .unwrap()
            .remove("recognized_pages");
        let old: PdfDocument = serde_json::from_value(old_cache).unwrap();
        assert!(!old.has_recognized_geometry());
        assert_eq!(
            old.fingerprint().result_sha256,
            native.fingerprint().result_sha256
        );
        let mut scanned = pages;
        scanned[0].source = "ocr".into();
        scanned[0].lines[0].words.push(legal_pdf_core::model::Word {
            id: "w1".into(),
            text: "Recognized".into(),
            bbox: [60.0, 100.0, 120.0, 112.0],
            start: 0,
            end: 10,
            underlined: false,
        });
        let prepared = PdfDocument::project(scanned, vec![], vec![], graph, pdf_summary());
        let restored: PdfDocument =
            serde_json::from_slice(&serde_json::to_vec(&prepared).unwrap()).unwrap();
        let page = &restored.recognized_pages()[0];
        assert_eq!(
            (page.page_number, page.width, page.height),
            (1, 612.0, 792.0)
        );
        assert_eq!(page.lines[0].words[0].text, "Recognized");
        assert!(page.lines[0].text.is_empty());
        assert_eq!(page.lines[1].text, "Unclassified source text");
        assert!(page.lines[1].words.is_empty());
        assert_eq!(page.lines[0].words[0].rect, [60.0, 100.0, 120.0, 112.0]);
    }

    #[test]
    fn a_recognized_scan_s_lines_are_its_passage_witness() {
        let mut worded = source_line("worded", "");
        worded.words.push(legal_pdf_core::model::Word {
            id: "w1".into(),
            text: "Tow".into(),
            bbox: [60.0, 100.0, 90.0, 112.0],
            start: 0,
            end: 3,
            underlined: false,
        });
        let mut plain = source_line("plain", "[7] The tug may berth");
        plain.bbox = [60.0, 120.0, 270.0, 132.0];
        let pages = vec![Page {
            id: "page-1".into(),
            index: 0,
            number: 1,
            width: 612.0,
            height: 792.0,
            lines: vec![worded, plain],
            regions: vec![],
            source: "ocr".into(),
            text_quality: 1.0,
            printed_label: None,
            printed_label_source: None,
            printed_label_line_id: None,
        }];
        let evidence = PdfDocument::passage_evidence(&pages);
        let mut document = PdfDocument::project(
            pages,
            vec![],
            vec![],
            structure_graph(vec![]),
            pdf_summary(),
        );
        document.set_passage_evidence(evidence);
        let witness = document.passage_pages();
        let lines = &witness[0].lines;
        assert_eq!(witness[0].source, "ocr");
        assert_eq!(
            lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["Tow", "[7] The tug may berth"]
        );
        assert_eq!(lines[0].words[0].rect, [60.0, 100.0, 90.0, 112.0]);
        // Without word boxes, each word takes its share of the line's box.
        let words = lines[1]
            .words
            .iter()
            .map(|word| (word.text.as_str(), word.rect[0], word.rect[2]))
            .collect::<Vec<_>>();
        assert_eq!(
            words,
            [
                ("[7]", 60.0, 90.0),
                ("The", 100.0, 130.0),
                ("tug", 140.0, 170.0),
                ("may", 180.0, 210.0),
                ("berth", 220.0, 270.0),
            ]
        );
    }

    #[test]
    fn invalid_lookup_is_bounded_without_guessing() {
        let mut document = LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![Page {
                id: "page-1".to_owned(),
                index: 0,
                number: 1,
                width: 612.0,
                height: 792.0,
                lines: vec![source_line("l1", "text")],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![
                Paragraph {
                    id: "marker".to_owned(),
                    page_index: 0,
                    region_type: "body".to_owned(),
                    text: "⟦FN:one⟧".to_owned(),
                    line_ids: vec![],
                    anchors: vec![],
                },
                Paragraph {
                    id: "p".to_owned(),
                    page_index: 0,
                    region_type: "body".to_owned(),
                    text: "text".to_owned(),
                    line_ids: vec!["l1".to_owned()],
                    anchors: vec![],
                },
            ],
            footnotes: vec![],
            structure_graph: structure_graph(vec![
                node("page-1", NodeKind::Page),
                node("p", NodeKind::Prose),
            ]),
            diagnostics: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        };
        attach_rendered_text(
            &document.paragraphs,
            &document.pages,
            &document.footnotes,
            &mut document.structure_graph,
        );
        assert_eq!(document.structure_graph.query_text(), "text");
        assert_eq!(
            document.structure_graph.revision,
            format!("{:x}", Sha256::digest(b"text"))
        );
        assert_eq!(
            document.structure_graph.nodes[1].rendered_range,
            Some(ScalarRange { start: 0, end: 4 })
        );
        let document = PdfDocument::project(
            document.pages,
            document.footnotes,
            vec![],
            document.structure_graph,
            pdf_summary(),
        );
        assert!(matches!(
            document.lookup(&PdfLookupRequest::new("page", "")).status,
            PdfLookupStatus::Invalid
        ));
        // Nothing here is numbered in print: a paragraph is not its block's position.
        let lookup = document.lookup(&PdfLookupRequest::new("paragraph", "par1"));
        assert!(matches!(lookup.status, PdfLookupStatus::Unavailable));
        let page = document.lookup(&PdfLookupRequest::new("page", "1"));
        assert!(matches!(page.status, PdfLookupStatus::Found));
        assert_eq!(page.units[0].text, "text");
    }

    #[test]
    fn section_ids_are_exact_lookup_locators() {
        let mut section = node("section-000001", NodeKind::Section);
        section.range.end = 18;
        section.source = Derivation::Heuristic;
        section.label = Some("A heading too long to use as a trusted locator".to_owned());
        section.content_start = Some(0);
        section.grammar = Some("hierarchy".to_owned());
        assert!(section_matches(
            &section,
            "section",
            &section_alias("section:section-000001"),
        ));
    }

    #[test]
    fn projection_does_not_invent_section_nodes() {
        let text = "😀".repeat(MAX_RETURN_CHARS / 2 + 1);
        let mut document = LegalDocument {
            document_id: "doc".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![Page {
                id: "page-1".to_owned(),
                index: 0,
                number: 1,
                width: 612.0,
                height: 792.0,
                lines: vec![source_line("l1", &text)],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![Paragraph {
                id: "heading".to_owned(),
                page_index: 0,
                region_type: "heading".to_owned(),
                text: text.clone(),
                line_ids: vec!["l1".to_owned()],
                anchors: vec![],
            }],
            footnotes: vec![],
            structure_graph: structure_graph(vec![node("heading", NodeKind::Heading)]),
            diagnostics: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        };

        attach_rendered_text(
            &document.paragraphs,
            &document.pages,
            &document.footnotes,
            &mut document.structure_graph,
        );
        assert_eq!(document.structure_graph.query_text(), text);
        assert_eq!(document.structure_graph.nodes[0].kind, NodeKind::Heading);
        assert_eq!(
            document.structure_graph.nodes[0].rendered_range,
            Some(ScalarRange {
                start: 0,
                end: utf16_len(&text),
            })
        );
        let document = PdfDocument::project(
            document.pages,
            document.footnotes,
            vec![],
            document.structure_graph,
            pdf_summary(),
        );
        let lookup = document.lookup(&PdfLookupRequest::new("section", "section:section-000001"));
        assert!(matches!(lookup.status, PdfLookupStatus::NotFound));
    }

    #[test]
    fn cache_round_trip_keeps_authority_text_units() {
        let expected = json!({
            "key": "body:0", "kind": "body", "ordinal": 0,
            "page_numbers": [1],
            "footnote_id": null, "text": "😀", "footnote_refs": [[1, 2]],
        });
        let marker = "⟦FN:one⟧";
        let document = PdfDocument::from_parts(
            &"00".repeat(32),
            "cache",
            "ready".to_owned(),
            PdfExtractionMetadata {
                embedded_page_labels: Vec::new(),
                pages_needing_ocr: vec![],
                ocr_routed_pages: vec![],
                scanned_pages: vec![],
            },
            vec![],
            vec![Paragraph {
                id: "p".to_owned(),
                page_index: 0,
                region_type: "body".to_owned(),
                text: format!("😀{marker}"),
                line_ids: vec![],
                anchors: vec![legal_pdf_core::model::ParagraphAnchor {
                    pair_id: "one".to_owned(),
                    label: "1".to_owned(),
                    offset: 1,
                }],
            }],
            vec![Footnote {
                pair_id: "one".to_owned(),
                label: "1".to_owned(),
                occurrence: 1,
                restart_sequence: 1,
                reference_page: Some(1),
                body_pages: vec![1],
                reference_line_id: Some("p".to_owned()),
                body_line_ids: vec![],
                body: "Note.".to_owned(),
                sentence_proposition: String::new(),
                passage_since_prior_note: String::new(),
                confidence: 1.0,
                provenance: "deterministic".to_owned(),
                warnings: vec![],
                crossrefs: vec![],
            }],
            structure_graph(vec![node("p", NodeKind::Prose)]),
        )
        .unwrap();
        assert_eq!(document.authority_text_units()[0], expected);

        let restored: PdfDocument =
            serde_json::from_value(serde_json::to_value(document).unwrap()).unwrap();
        assert_eq!(restored.authority_text_units()[0], expected);
    }
}
