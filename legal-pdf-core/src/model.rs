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
    let spans = line
        .spans
        .iter()
        .filter(|span| !span.text.trim().is_empty());
    let heading = role(spans.clone().next()?)?;
    if !is_heading_role(heading) {
        return None;
    }
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
        if role(span) != Some(heading) || span.start >= span.end {
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
    line.text[bytes[covered]..]
        .trim()
        .is_empty()
        .then_some(heading)
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
