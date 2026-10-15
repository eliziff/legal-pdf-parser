pub use legal_structure_model::{
    Derivation, DocumentStructure, PdfStructureTagWitness, NodeKind, NotePairSupportV2, ScalarRange, StructureNode,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;

pub const SCHEMA_VERSION: &str = "legalpdf.document.v4";
pub const PARSER_VERSION: &str = "0.3.1";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Word {
    pub id: String,
    pub text: String,
    pub bbox: [f64; 4],
    pub start: usize,
    pub end: usize,
    /// A rule is drawn under the word: painted in the PDF, or dark on a recognized page's image.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub underlined: bool,
}

/// Mark the words a horizontal rule on their page underlines: painted in the PDF, or drawn on a
/// recognized page's image.
pub fn mark_rule_underlines(page: &mut Page) {
    let rules = page
        .regions
        .iter()
        .filter(|region| region.kind == "rule")
        .map(|region| region.bbox)
        .filter(|[x0, y0, x1, y1]| (y1 - y0).abs() <= 2.0 && (x1 - x0).abs() > 2.0)
        .collect::<Vec<_>>();
    if rules.is_empty() {
        return;
    }
    for line in &mut page.lines {
        let height = (line.bbox[3] - line.bbox[1]).max(1.0);
        for word in &mut line.words {
            word.underlined = rules.iter().any(|rule| {
                underlines(word.bbox, height, rule[0].min(rule[2]), rule[0].max(rule[2]), (rule[1] + rule[3]) / 2.0)
            });
        }
    }
}

/// Whether a horizontal stroke from `start` to `end` at height `at` underlines a word box: set
/// from a third of its line's height above the word's bottom to half below, across most of it.
pub fn underlines(word: [f64; 4], line_height: f64, start: f64, end: f64, at: f64) -> bool {
    let width = (word[2] - word[0]).max(1.0);
    at >= word[3] - 0.3 * line_height
        && at <= word[3] + 0.5 * line_height
        && end.min(word[2]) - start.max(word[0]) >= 0.6 * width
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Span {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mcid: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_role: Option<String>,
    pub id: String,
    pub text: String,
    pub bbox: [f64; 4],
    #[serde(default)]
    pub font: String,
    #[serde(default)]
    pub size: f64,
    #[serde(default)]
    pub flags: u32,
    #[serde(default)]
    pub superscript: bool,
    #[serde(default)]
    pub start: usize,
    #[serde(default)]
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DetachedReference {
    pub note_id: String,
    pub selected_text: String,
    pub start_offset: usize,
    pub end_offset: usize,
    pub source_line_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrMarkerEvidence {
    /// A physical association only; this does not classify a paragraph or note.
    pub body_line_id: Option<String>,
    pub sequence_line_ids: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrEvidence {
    pub confidence: f64,
    pub recovered: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub baseline: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub boundary: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub marker: Option<OcrMarkerEvidence>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Line {
    pub id: String,
    pub page_index: usize,
    pub page_number: u32,
    pub source_index: usize,
    pub reading_order: usize,
    pub block_index: usize,
    pub text: String,
    pub bbox: [f64; 4],
    #[serde(default)]
    pub spans: Vec<Span>,
    #[serde(default)]
    pub words: Vec<Word>,
    #[serde(default)]
    pub detached_references: Vec<DetachedReference>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ocr: Option<OcrEvidence>,
    #[serde(default)]
    pub exclude_from_body: bool,
    #[serde(default)]
    pub suppress_footnote_label: bool,
    /// A competing paragraph-layout witness; pairing may overturn it.
    #[serde(default)]
    pub paragraph_label_candidate: bool,
    #[serde(default)]
    pub note_region_mode: String,
    #[serde(default)]
    pub region_id: String,
    #[serde(default = "unknown_region")]
    pub region_type: String,
    #[serde(default = "native_source")]
    pub source: String,
}

/// Align actual marked-content tags only through retained source spans and the graph source index.
pub fn attach_structure_tags(graph: &mut DocumentStructure, pages: &[Page], native: &[PdfStructureTagWitness]) {
    let mut tags = std::collections::BTreeMap::new();
    for tag in native {
        let mut witness = tag.clone(); witness.ranges.clear(); witness.line_ids.clear();
        tags.insert((tag.page_index, tag.mcid, tag.role.clone()), witness);
    }
    let fragments: HashMap<_, _> = graph.source_fragments.iter().map(|fragment|
        ((fragment.page_index, fragment.id.as_str()), fragment.range)).collect();
    let source = legal_structure_model::ScalarText::new(&graph.text);
    for page in pages {
        for line in &page.lines {
            let coordinates = legal_structure_model::ScalarText::new(&line.text);
            for span in &line.spans {
                let (Some(mcid), Some(role)) = (span.mcid, span.structure_role.as_ref()) else { continue; };
                let witness = tags.entry((page.index, mcid, role.clone())).or_insert_with(|| PdfStructureTagWitness {
                    page_index: page.index, mcid, role: role.clone(), ranges: Vec::new(), line_ids: Vec::new(),
                });
                let Some(fragment) = fragments.get(&(page.index, line.id.as_str())) else { continue; };
                if source.slice_utf16(fragment.start..fragment.end) != Some(line.text.as_str())
                    || coordinates.slice(span.start..span.end) != Some(span.text.as_str()) { continue; }
                let range = ScalarRange { start: fragment.start + coordinates.utf16(span.start),
                    end: fragment.start + coordinates.utf16(span.end) };
                if range.start >= range.end { continue; }
                witness.ranges.push(range);
                if witness.line_ids.last() != Some(&line.id) { witness.line_ids.push(line.id.clone()); }
            }
        }
    }
    graph.structure_tags = tags.into_values().collect();
}

fn unknown_region() -> String {
    "unknown".to_owned()
}

fn native_source() -> String {
    "native".to_owned()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    pub id: String,
    pub page_index: usize,
    #[serde(rename = "type")]
    pub kind: String,
    pub line_ids: Vec<String>,
    pub bbox: [f64; 4],
    pub reading_order: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Page {
    pub id: String,
    pub index: usize,
    pub number: u32,
    pub width: f64,
    pub height: f64,
    pub lines: Vec<Line>,
    pub regions: Vec<Region>,
    #[serde(default = "native_source")]
    pub source: String,
    #[serde(default = "one")]
    pub text_quality: f64,
    #[serde(default)]
    pub printed_label: Option<String>,
    #[serde(default)]
    pub printed_label_source: Option<String>,
    #[serde(default)]
    pub printed_label_line_id: Option<String>,
}

const fn one() -> f64 {
    1.0
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParagraphAnchor {
    pub pair_id: String,
    pub label: String,
    pub offset: usize,
}

/// An unchanged UTF-16 run in produced text, anchored to an original physical line.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceTextSpan {
    pub start: usize,
    pub end: usize,
    pub line_id: String,
    pub source_start: usize,
    pub source_end: usize,
}

impl SourceTextSpan {
    /// Retain an exact input slice at a new output position. Insertions have no run.
    pub fn retain(&self, start: usize, end: usize, output_start: usize) -> Option<Self> {
        let left = self.start.max(start);
        let right = self.end.min(end);
        (left < right).then(|| Self { start: output_start + left - start,
            end: output_start + right - start, line_id: self.line_id.clone(),
            source_start: self.source_start + left - self.start,
            source_end: self.source_start + right - self.start })
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paragraph {
    #[serde(default)]
    pub source_spans: Vec<SourceTextSpan>,
    pub id: String,
    pub page_index: usize,
    pub region_type: String,
    pub text: String,
    pub line_ids: Vec<String>,
    #[serde(default)]
    pub anchors: Vec<ParagraphAnchor>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FootnoteCrossref {
    pub source_pair_id: String,
    pub kind: String,
    pub number: u32,
    pub shortform: String,
    pub start: usize,
    pub end: usize,
    pub resolved: bool,
    pub target_pair_id: String,
    pub target_count: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Footnote {
    #[serde(default)]
    pub source_spans: Vec<SourceTextSpan>,
    pub pair_id: String,
    pub label: String,
    pub occurrence: usize,
    pub restart_sequence: usize,
    pub reference_page: Option<u32>,
    pub body_pages: Vec<u32>,
    pub reference_line_id: Option<String>,
    pub body_line_ids: Vec<String>,
    pub body: String,
    pub sentence_proposition: String,
    pub passage_since_prior_note: String,
    pub confidence: f64,
    pub provenance: String,
    #[serde(default)]
    pub warnings: Vec<String>,
    #[serde(default)]
    pub crossrefs: Vec<FootnoteCrossref>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Diagnostic {
    pub code: String,
    pub severity: String,
    pub message: String,
    #[serde(default)]
    pub page_index: Option<usize>,
    #[serde(default)]
    pub line_ids: Vec<String>,
    #[serde(default)]
    pub details: Map<String, Value>,
}

impl Diagnostic {
    pub fn warning(code: &str, message: impl Into<String>, page_index: Option<usize>) -> Self {
        Self {
            code: code.to_owned(),
            severity: "warning".to_owned(),
            message: message.into(),
            page_index,
            line_ids: Vec::new(),
            details: Map::new(),
        }
    }

    pub fn info(code: &str, message: impl Into<String>, page_index: Option<usize>) -> Self {
        Self {
            code: code.to_owned(),
            severity: "info".to_owned(),
            message: message.into(),
            page_index,
            line_ids: Vec::new(),
            details: Map::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PdfSourceSpan {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PdfPairingAudit {
    pub markers: Vec<Value>,
    pub pairing_summary: Value,
}

/// One bookmark of the PDF's own outline, in outline order: a navigation hint whose title and
/// target may be stale. `level` counts from 1 at the outline's top; `page_index` is the target's
/// physical page, counted from 0.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PdfOutlineEntry {
    pub title: String,
    pub level: usize,
    pub page_index: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PdfExtractionMetadata {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub structure_tags: Vec<PdfStructureTagWitness>,
    #[serde(default)]
    pub embedded_page_labels: Vec<Option<String>>,
    pub pages_needing_ocr: Vec<usize>,
    pub ocr_routed_pages: Vec<usize>,
    /// Pages that are a picture of their text: an image covers at least half of the page, and the
    /// page's own visible text would not fill one line across it (a page number, a stamp, a
    /// heading). Their text layer, if any, is invisible. A page of ordinary rendered text is never
    /// one, however it reads.
    #[serde(default)]
    pub scanned_pages: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LegalDocument {
    pub document_id: String,
    pub source_name: String,
    pub source_sha256: String,
    pub page_count: usize,
    pub status: String,
    pub pages: Vec<Page>,
    pub paragraphs: Vec<Paragraph>,
    pub footnotes: Vec<Footnote>,
    pub structure_graph: DocumentStructure,
    pub diagnostics: Vec<Diagnostic>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
    #[serde(default)]
    pub provenance: Map<String, Value>,
    #[serde(default = "schema_version")]
    pub schema_version: String,
    #[serde(default = "parser_version")]
    pub parser_version: String,
}

fn schema_version() -> String {
    SCHEMA_VERSION.to_owned()
}

fn parser_version() -> String {
    PARSER_VERSION.to_owned()
}

impl LegalDocument {
    pub fn line_count(&self) -> usize {
        self.pages.iter().map(|page| page.lines.len()).sum()
    }
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub pair_id: String,
    pub label: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NotePairKind {
    Footnote,
    Endnote,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAnchor {
    pub line_id: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotePairClaim {
    pub pair_id: String,
    pub support: NotePairSupportV2,
    pub label: String,
    pub kind: NotePairKind,
    pub label_anchor: SourceAnchor,
    pub reference_anchors: Vec<SourceAnchor>,
    pub body_line_ids: Vec<String>,
}

pub struct PairingOutput {
    pub footnotes: Vec<Footnote>,
    pub diagnostics: Vec<Diagnostic>,
    pub anchors: HashMap<String, Vec<Anchor>>,
    pub pair_claims: Vec<NotePairClaim>,
    pub pairing_audit: Option<PdfPairingAudit>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_field_names_match_the_public_contract() {
        let region = Region {
            id: "p0001-r0001".to_owned(),
            page_index: 0,
            kind: "body".to_owned(),
            line_ids: vec![],
            bbox: [0.0; 4],
            reading_order: 1,
        };
        let value = serde_json::to_value(region).unwrap();
        assert_eq!(value["type"], "body");
        assert!(value.get("kind").is_none());
        assert_eq!(SCHEMA_VERSION, "legalpdf.document.v4");
    }
}
