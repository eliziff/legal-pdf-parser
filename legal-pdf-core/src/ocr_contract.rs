use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrWord {
    pub text: String,
    pub bbox: [f64; 4],
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    pub bbox: [f64; 4],
    pub confidence: f64,
    /// Recognized from ink omitted by the page segmenter.
    #[serde(default, skip_serializing_if = "is_false")]
    pub recovered: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub baseline: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub boundary: Vec<[f64; 2]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub words: Vec<OcrWord>,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub region_id: String,
    #[serde(default = "unknown_region", skip_serializing_if = "is_unknown_region")]
    pub region_type: String,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub block_index: usize,
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn unknown_region() -> String {
    "unknown".to_owned()
}

fn is_unknown_region(value: &str) -> bool {
    value == "unknown"
}

const fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, Clone)]
pub struct OcrPageRequest {
    pub page_index: usize,
    pub width: f64,
    pub height: f64,
    /// The parts of the page to recognize, `[x0, y0, x1, y1]` in points from its top-left
    /// corner, or none for the whole page. A provider may recognize the whole page instead;
    /// only its lines inside a part are read.
    pub regions: Vec<[f64; 4]>,
    /// Line boxes whose ink the page's render measures, `[x0, y0, x1, y1]` in points; none to
    /// measure nothing.
    pub lines: Vec<[f64; 4]>,
    /// Each measured line's word boxes, in points, where its words' strokes are measured too;
    /// none, or fewer than `lines`, to measure lines alone.
    pub words: Vec<Vec<[f64; 4]>>,
}

/// What a requested page's render shows the readers: the rule above its notes, its thin
/// horizontal strokes (`[x0, y, x1, y]` in points, read as underlines and rules), and for each
/// requested line its dark share, its stroke weight, the height of its capitals over its baseline
/// (points) and its words' stroke weights. A recognized page names no type; this is its type.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RasterReading {
    pub separator: Option<f64>,
    pub strokes: Vec<[f64; 4]>,
    #[serde(default)]
    pub inks: Vec<f64>,
    #[serde(default)]
    pub weights: Vec<f64>,
    #[serde(default)]
    pub cap_heights: Vec<f64>,
    #[serde(default)]
    pub word_weights: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPageResult {
    pub page_index: usize,
    pub lines: Vec<OcrLine>,
    pub separator_y: Option<f64>,
}

pub trait PdfOcrProvider {
    fn extract_pages(
        &mut self,
        pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>>;

    /// Where the rule printed above each requested page's notes runs, in points from the
    /// page's top, read from the page's image without recognizing it; none by default.
    fn separators(&mut self, _pdf: &[u8], requests: &[OcrPageRequest]) -> Vec<Option<f64>> {
        vec![None; requests.len()]
    }

    /// A part of a page read again: the lines the recognizer reads inside `region` ([x0, y0, x1, y1]
    /// in points from the page's top-left) rendered at `scale` times the page's size in points, their
    /// boxes in page points; none by default. A paragraph number a first reading lost is looked
    /// for this way, in the margin where its run says it stands.
    fn read_region(&mut self, _pdf: &[u8], _page_index: usize, _region: [f64; 4], _scale: f64) -> Result<Vec<OcrLine>> {
        Ok(Vec::new())
    }
}
