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

fn unknown_region() -> String {
    "unknown".to_owned()
}

fn is_unknown_region(value: &str) -> bool {
    value == "unknown"
}

const fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, Clone, Copy)]
pub struct OcrPageRequest {
    pub page_index: usize,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone)]
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
}
