use legal_pdf_core::model::{
    Derivation, DocumentStructure, Footnote, Line, NodeKind, Page, Paragraph,
    PdfExtractionMetadata, ScalarRange, StructureNode, PARSER_VERSION,
};
use legal_structure::{
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
use std::sync::OnceLock;
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
    structure: legal_structure::DocumentStructure,
    pages: Vec<ProjectionPage>,
    footnotes: Vec<ProjectionFootnote>,
    authority_text_units: Vec<Value>,
    summary: PdfSummary,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PdfSummary {
    pub sha256: String,
    pub parser_version: String,
    pub cache_key: String,
    pub page_count: usize,
    pub projection_page_count: usize,
    pub status: String,
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
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
        sha256: sha256.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
        cache_key: cache_key.to_owned(),
        page_count,
        projection_page_count: page_count,
        status,
        pages_needing_ocr: pdf_metadata.pages_needing_ocr,
        ocr_routed_pages: pdf_metadata.ocr_routed_pages,
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
            crate::adapters::to_toa_text_units_from_parts(&paragraphs, &footnotes)?;
        attach_rendered_text(&paragraphs, &pages, &footnotes, &mut structure);
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
        Self {
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

    pub fn structure(&self) -> &legal_structure::DocumentStructure {
        &self.structure
    }

    pub fn structure_mut(&mut self) -> &mut legal_structure::DocumentStructure {
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
        let product = serialization_sha256(&(&self.pages, &self.footnotes, &self.summary));
        fingerprint.components.insert("pdf", product.clone());
        fingerprint.result_sha256 =
            serialization_sha256(&(fingerprint.result_sha256.as_str(), product));
        fingerprint
    }

    pub fn lookup(&self, request: &PdfLookupRequest) -> PdfStructureLookup {
        structure_lookup(self, request)
    }
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

fn clean_text(value: &str) -> Cow<'_, str> {
    static MARKER: OnceLock<Regex> = OnceLock::new();
    let value = value.trim();
    if !value.contains("⟦FN:") {
        Cow::Borrowed(value)
    } else {
        Cow::Owned(
            MARKER
                .get_or_init(|| Regex::new(r"⟦FN:[^⟧]+⟧").unwrap())
                .replace_all(value, "")
                .trim()
                .to_owned(),
        )
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

fn rendered_slice<'a>(text: &ScalarText<'a>, node: &StructureNode) -> &'a str {
    node.rendered_range
        .and_then(|range| text.slice_utf16(range.start..range.end))
        .unwrap_or_default()
}

fn section_nodes(document: &PdfDocument) -> Vec<&StructureNode> {
    document
        .structure
        .nodes
        .iter()
        .filter(|node| node.kind == NodeKind::Section)
        .collect()
}

fn paragraph_nodes<'a>(document: &'a PdfDocument, text: &ScalarText<'_>) -> Vec<&'a StructureNode> {
    document
        .structure
        .nodes
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::Prose | NodeKind::Heading))
        .filter(|node| !rendered_slice(text, node).is_empty())
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

fn paragraph_unit(
    text: &ScalarText<'_>,
    pages: &HashMap<usize, &ProjectionPage>,
    paragraph: &StructureNode,
    number: usize,
) -> PdfLookupUnit {
    let page = paragraph
        .page_indexes
        .first()
        .and_then(|page| pages.get(page));
    PdfLookupUnit {
        id: if paragraph.id.is_empty() {
            format!("paragraph-{number}")
        } else {
            paragraph.id.clone()
        },
        kind: "paragraph",
        locator: format!("paragraph {number}"),
        text: rendered_slice(text, paragraph).to_owned(),
        page_numbers: page.map_or_else(Vec::new, |item| vec![item.number]),
        confidence: page.map(|item| item.text_quality.clamp(0.0, 1.0)),
        confidence_basis: if page.is_some() {
            "page_text_quality"
        } else {
            "unavailable"
        },
        provenance: format!(
            "legalpdf:{}",
            match paragraph.kind {
                NodeKind::Prose => "body",
                NodeKind::Heading => "heading",
                _ => "unknown",
            }
        ),
        proposition: None,
        note: None,
    }
}

fn page_unit(
    text: &ScalarText<'_>,
    nodes: &HashMap<&str, &StructureNode>,
    page: &ProjectionPage,
) -> PdfLookupUnit {
    let text = nodes
        .get(page.id.as_str())
        .copied()
        .map(|node| rendered_slice(text, node).to_owned())
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
        text: rendered_slice(text, section).trim().to_owned(),
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

pub fn parse_ordinal(kind: &str, raw: &str) -> Option<usize> {
    static PAGE: OnceLock<Regex> = OnceLock::new();
    static PARAGRAPH: OnceLock<Regex> = OnceLock::new();
    let value = nfkc_text(raw);
    let captures = if kind == "page" {
        PAGE.get_or_init(|| {
            Regex::new(r"(?i)^#?\s*\[?\s*(?:(?:pages?|pp?\.?)[\s:_=-]*)?0*(\d{1,6})\s*\]?$")
                .unwrap()
        })
        .captures(value.trim())
    } else {
        PARAGRAPH.get_or_init(|| Regex::new(r"(?i)^#?\s*(?:(?:paragraphs?|paras?|pars?|¶)\.?\s*)?(?:(?:paragraph|para|par)[\s:_=-]*)?0*(\d{1,6})$" ).unwrap()).captures(value.trim())
    }?;
    captures.get(1)?.as_str().parse().ok()
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
                page_unit(&text, &nodes, &document.pages[index])
            });
        }
        let text = ScalarText::new(document.structure.query_text());
        let paragraphs = paragraph_nodes(document, &text);
        let position = |number: usize| {
            number
                .checked_sub(1)
                .filter(|index| *index < paragraphs.len())
        };
        let (Some(start), Some(end)) = (position(start_number), position(end_number)) else {
            return result(input, PdfLookupStatus::NotFound, None);
        };
        let pages: HashMap<_, _> = document
            .pages
            .iter()
            .map(|page| (page.index, page))
            .collect();
        return finish(document, input, paragraphs.len(), start, end, |index| {
            paragraph_unit(&text, &pages, paragraphs[index], index + 1)
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
        section_unit(&text, &pages, sections[index], index)
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

fn attach_rendered_text(
    paragraphs: &[Paragraph],
    pages: &[Page],
    footnotes: &[Footnote],
    structure: &mut DocumentStructure,
) {
    let mut text = String::with_capacity(structure.text.len());
    let mut id_ranges = HashMap::new();
    let mut line_ranges = HashMap::new();
    let mut page_ranges: HashMap<usize, ScalarRange> = HashMap::new();
    let mut position = 0;
    for paragraph in paragraphs {
        let paragraph_text = clean_text(&paragraph.text);
        if paragraph_text.is_empty() {
            continue;
        }
        if !text.is_empty() {
            append(&mut text, &mut position, "\n\n");
        }
        let start = position;
        append(&mut text, &mut position, &paragraph_text);
        let range = ScalarRange {
            start,
            end: position,
        };
        if !paragraph.id.is_empty() {
            id_ranges.insert(paragraph.id.as_str(), range);
        }
        for line_id in &paragraph.line_ids {
            line_ranges.insert(line_id.as_str(), range);
        }
        page_ranges
            .entry(paragraph.page_index)
            .and_modify(|page| {
                page.start = page.start.min(range.start);
                page.end = page.end.max(range.end);
            })
            .or_insert(range);
    }
    for page in pages {
        if !page.id.is_empty() {
            if let Some(range) = page_ranges.get(&page.index) {
                id_ranges.insert(page.id.as_str(), *range);
            }
        }
    }

    for note in footnotes {
        let lines_range = note
            .body_line_ids
            .iter()
            .filter_map(|id| line_ranges.get(id.as_str()))
            .copied()
            .fold(None, extend_range);
        let range = if let Some(range) = lines_range {
            Some(range)
        } else {
            let body = clean_text(&note.body);
            if body.is_empty() {
                None
            } else if let Some(start_byte) = text.find(body.as_ref()) {
                let start = utf16_len(&text[..start_byte]);
                Some(ScalarRange {
                    start,
                    end: start + utf16_len(&body),
                })
            } else {
                if !text.is_empty() {
                    append(&mut text, &mut position, "\n\n");
                }
                let start = position;
                append(&mut text, &mut position, &body);
                Some(ScalarRange {
                    start,
                    end: position,
                })
            }
        };
        if let Some(range) = range {
            id_ranges.insert(note.pair_id.as_str(), range);
            for line_id in &note.body_line_ids {
                line_ranges.entry(line_id.as_str()).or_insert(range);
            }
        }
    }

    for note in &structure.notes {
        if let Some(range) = id_ranges.get(note.id.as_str()).copied() {
            id_ranges.insert(note.node_id.as_str(), range);
        }
    }
    for node in &mut structure.nodes {
        let source_range = id_ranges.get(node.id.as_str()).copied().or_else(|| {
            node.line_ids
                .iter()
                .filter_map(|id| line_ranges.get(id.as_str()))
                .copied()
                .fold(None, extend_range)
        });
        node.rendered_range = source_range.or_else(|| {
            if matches!(node.kind, NodeKind::Prose | NodeKind::Heading) {
                None
            } else {
                node.page_indexes
                    .first()
                    .and_then(|page| page_ranges.get(page))
                    .copied()
            }
        });
    }
    structure.revision = format!("{:x}", Sha256::digest(text.as_bytes()));
    structure.rendered_text = (text != structure.text).then_some(text);
}

#[cfg(test)]
mod tests {
    use super::*;
    use legal_pdf_core::model::{LegalDocument, ScalarRange, SCHEMA_VERSION};
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

    fn pdf_summary() -> PdfSummary {
        PdfSummary {
            sha256: "00".repeat(32),
            parser_version: PARSER_VERSION.to_owned(),
            cache_key: "cache".to_owned(),
            page_count: 1,
            projection_page_count: 1,
            status: "ready".to_owned(),
            pages_needing_ocr: vec![],
            ocr_routed_pages: vec![],
        }
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
                lines: vec![],
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
                    line_ids: vec![],
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
        let lookup = document.lookup(&PdfLookupRequest::new("paragraph", "par1"));
        assert!(matches!(lookup.status, PdfLookupStatus::Found));
        assert_eq!(lookup.units[0].id, "p");
        assert_eq!(lookup.units[0].text, "text");
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
                lines: vec![],
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
                line_ids: vec![],
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
                pages_needing_ocr: vec![],
                ocr_routed_pages: vec![],
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
