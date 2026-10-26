pub use legal_layout::*;
use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "legalpdf.document.v4";
pub const PARSER_VERSION: &str = "0.3.1";

/// A standard PDF heading role: H, or Hn for a positive level.
pub fn is_heading_role(role: &str) -> bool {
    role == "H"
        || role.strip_prefix('H').is_some_and(|level| {
            !level.starts_with('0')
                && level.bytes().all(|byte| byte.is_ascii_digit())
                && level.parse::<usize>().is_ok_and(|level| level > 0)
        })
}

/// Heading labels from a layout provider or the PDF's own structure tree.
pub fn is_heading_region(region: &str) -> bool {
    matches!(region, "heading" | "paragraph_title" | "doc_title") || is_heading_role(region)
}

/// The heading role covering a whole source line. Mixed roles, gaps containing text and
/// unanchored spans cannot turn a run-in heading into a heading for the entire paragraph.
pub fn tagged_heading<'a>(
    line: &'a Line,
    role: impl Fn(&'a Span) -> Option<&'a str>,
) -> Option<&'a str> {
    tagged_line_role(line, |span| role(span).filter(|role| is_heading_role(role)))
}

/// One source role covering the line's exact visible text, including every anchored span.
pub fn tagged_line_role<'a>(
    line: &'a Line,
    role: impl Fn(&'a Span) -> Option<&'a str>,
) -> Option<&'a str> {
    let spans = line
        .spans
        .iter()
        .filter(|span| !span.text.trim().is_empty());
    let tag = role(spans.clone().next()?)?;
    let mut spans = spans.collect::<Vec<_>>();
    spans.sort_unstable_by_key(|span| (span.start, span.end));
    let bytes = line
        .text
        .char_indices()
        .map(|(byte, _)| byte)
        .chain(std::iter::once(line.text.len()))
        .collect::<Vec<_>>();
    let mut covered = 0;
    for span in spans {
        if role(span) != Some(tag) || span.start >= span.end {
            return None;
        }
        let (start, end) = (*bytes.get(span.start)?, *bytes.get(span.end)?);
        if line.text.get(start..end)? != span.text
            || (span.start > covered && !line.text[bytes[covered]..start].trim().is_empty())
        {
            return None;
        }
        covered = covered.max(span.end);
    }
    line.text[bytes[covered]..].trim().is_empty().then_some(tag)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParagraphAnchor {
    pub pair_id: String,
    pub label: String,
    pub offset: usize,
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
