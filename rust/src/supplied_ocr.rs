// Provider-neutral OCR results, tied to the exact original PDF. Recognition stays
// with the embedding host; extraction, structure and passage geometry stay here.
use legal_pdf_core::{Error, OcrLine, OcrPageRequest, OcrPageResult, PdfOcrProvider, Result};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SuppliedOcr {
    pub source_sha256: String,
    pub pages: Vec<SuppliedPage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SuppliedPage {
    pub page_index: usize,
    pub width: f64,
    pub height: f64,
    pub lines: Vec<OcrLine>,
}

impl SuppliedOcr {
    pub fn identity(&self, source: &str) -> Result<String> {
        if self.source_sha256 != source {
            return Err(Error::Message("OCR source does not match the PDF".into()));
        }
        let mut seen = BTreeSet::new();
        let mut text_bytes = 0usize;
        if self.pages.len() > 1_000 {
            return Err(Error::Message("Too many supplied OCR pages".into()));
        }
        for page in &self.pages {
            if !seen.insert(page.page_index)
                || !page.width.is_finite()
                || !page.height.is_finite()
                || page.width <= 0.0
                || page.height <= 0.0
                || page.lines.len() > 10_000
            {
                return Err(Error::Message("Invalid supplied OCR page".into()));
            }
            let valid_box = |b: &[f64; 4]| {
                b.iter().all(|v| v.is_finite())
                    && b[0] >= 0.0
                    && b[1] >= 0.0
                    && b[2] > b[0]
                    && b[3] > b[1]
                    && b[2] <= page.width
                    && b[3] <= page.height
            };
            for line in &page.lines {
                text_bytes = text_bytes.saturating_add(line.text.len());
                if text_bytes > 16 * 1024 * 1024
                    || !valid_box(&line.bbox)
                    || !line.confidence.is_finite()
                    || !(0.0..=1.0).contains(&line.confidence)
                    || line.baseline.len() > 10_000
                    || line.boundary.len() > 10_000
                    || line.baseline.iter().chain(&line.boundary).any(|p| {
                        !p[0].is_finite()
                            || !p[1].is_finite()
                            || p[0] < 0.0
                            || p[1] < 0.0
                            || p[0] > page.width
                            || p[1] > page.height
                    })
                    || line.words.iter().any(|w| {
                        !valid_box(&w.bbox) || w.start > w.end || w.end > line.text.chars().count()
                    })
                {
                    return Err(Error::Message(
                        "Invalid supplied OCR line geometry or text".into(),
                    ));
                }
            }
        }
        let bytes = serde_json::to_vec(self).map_err(|e| Error::Message(e.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

impl PdfOcrProvider for &SuppliedOcr {
    fn extract_pages(
        &mut self,
        _pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        requests
            .iter()
            .filter_map(|request| {
                let Some(page) = self
                    .pages
                    .iter()
                    .find(|p| p.page_index == request.page_index)
                else {
                    // Pictures on a page with text of its own are read only where the host
                    // recognized that page; the page itself needs no recognition.
                    return (request.regions.is_empty()).then(|| {
                        Err(Error::Message(format!(
                            "Missing OCR for page {}",
                            request.page_index + 1
                        )))
                    });
                };
                if (page.width - request.width).abs() > 0.5
                    || (page.height - request.height).abs() > 0.5
                {
                    return Some(Err(Error::Message(
                        "OCR page dimensions do not match the PDF".into(),
                    )));
                }
                Some(Ok(OcrPageResult {
                    page_index: page.page_index,
                    lines: page.lines.clone(),
                    separator_y: None,
                }))
            })
            .collect()
    }
}
