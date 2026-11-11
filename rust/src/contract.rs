//! A PDF's parse as a caller asks for it: one request, checked as it is read, that the parse reads as it stands.

use crate::engine::parse_pdf;
use crate::RecognitionProgress;
#[cfg(feature = "ocr")]
use crate::OcrOptions;
#[cfg(feature = "ppdoc")]
use crate::PPDocOptions;
use crate::Result;
pub use legal_pdf_support::{PdfDocument, PdfSummary};
use serde::de::Error as _;
use serde::{Deserialize, Deserializer};
use std::path::PathBuf;

const MAX_SELECTED_PAGES: usize = 1_000;

#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct PdfRequest {
    pub(crate) profile: Option<legal_structure_model::DetectionProfile>,
    pub(crate) cache_dir: Option<PathBuf>,
    /// Where page recognition (OCR) and its scanner rasters are cached: apart from the parse cache, keyed by the OCR
    /// code and settings alone, so no parser change discards them. Defaults to `ocr` in the cache directory.
    pub(crate) ocr_cache_dir: Option<PathBuf>,
    /// The parse cached under this key, and no other: none where it is no longer cached.
    #[serde(default, deserialize_with = "sha256")]
    pub cache_key: Option<String>,
    #[serde(default, deserialize_with = "sha256")]
    pub expected_source_sha256: Option<String>,
    pub(crate) max_output_bytes: Option<usize>,
    /// Zero-based, sorted, the pages eligible for OCR (`pages`, from 1). Native extraction still inspects the whole PDF
    /// so physical page numbering stays authoritative.
    #[serde(default, rename = "pages", deserialize_with = "selected_pages")]
    pub ocr_pages: Option<Vec<usize>>,
    pub(crate) supplied_ocr: Option<crate::supplied_ocr::SuppliedOcr>,
    pub id: Option<String>,
    pub(crate) url: Option<String>,
    #[cfg(feature = "ocr")]
    #[serde(default, deserialize_with = "ocr")]
    pub(crate) ocr: Option<OcrOptions>,
    #[cfg(not(feature = "ocr"))]
    #[serde(default, rename = "ocr", deserialize_with = "unsupported")]
    _ocr: Option<()>,
    #[cfg(feature = "ppdoc")]
    #[serde(default, rename = "layout", deserialize_with = "layout")]
    pub(crate) ppdoc: Option<PPDocOptions>,
    #[cfg(not(feature = "ppdoc"))]
    #[serde(default, rename = "layout", deserialize_with = "unsupported")]
    _layout: Option<()>,
    /// What scores the structure stage's moves in place of tier 0: DOC, tier 1, read only in the modes its census
    /// promoted (`engine::DOC_PROMOTED`).
    #[cfg(feature = "doc")]
    #[serde(default, rename = "structure", deserialize_with = "structure")]
    pub(crate) doc: Option<DocOptions>,
    #[cfg(not(feature = "doc"))]
    #[serde(default, rename = "structure", deserialize_with = "unsupported")]
    _structure: Option<()>,
}

/// DOC as a request names it: its model pack (the stages `docparse.py export` writes) and the ONNX Runtime it runs in
/// (else the one LEGALPDF_ONNX_RUNTIME names).
#[cfg(feature = "doc")]
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DocOptions {
    pub(crate) model_pack: PathBuf,
    pub(crate) runtime: Option<PathBuf>,
}

fn sha256<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<String>, D::Error> {
    let value = String::deserialize(deserializer)?;
    if value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase()) {
        Ok(Some(value))
    } else {
        Err(D::Error::custom("a document request's hash must be lowercase SHA-256"))
    }
}

fn selected_pages<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<Vec<usize>>, D::Error> {
    let pages = Vec::<usize>::deserialize(deserializer)?;
    if pages.is_empty() || pages.len() > MAX_SELECTED_PAGES {
        return Err(D::Error::custom(format!("document request pages requires 1 to {MAX_SELECTED_PAGES} pages")));
    }
    if pages.contains(&0) {
        return Err(D::Error::custom("document request pages must be positive integers"));
    }
    let mut selected = pages.into_iter().map(|page| page - 1).collect::<Vec<_>>();
    selected.sort_unstable();
    selected.dedup();
    Ok(Some(selected))
}

#[cfg(feature = "ocr")]
fn ocr<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<OcrOptions>, D::Error> {
    #[derive(Deserialize)]
    #[serde(tag = "provider", deny_unknown_fields)]
    enum Provider {
        #[serde(rename = "indra")]
        Indra {
            #[serde(default)]
            settings: OcrOptions,
        },
    }
    let Provider::Indra { settings } = Provider::deserialize(deserializer)?;
    Ok(Some(settings))
}

#[cfg(feature = "ppdoc")]
fn layout<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<PPDocOptions>, D::Error> {
    #[derive(Deserialize)]
    #[serde(tag = "provider", deny_unknown_fields)]
    enum Provider {
        #[serde(rename = "ppdoc")]
        Ppdoc {
            #[serde(default)]
            settings: PPDocOptions,
        },
    }
    let Provider::Ppdoc { settings } = Provider::deserialize(deserializer)?;
    Ok(Some(settings))
}

#[cfg(feature = "doc")]
fn structure<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<DocOptions>, D::Error> {
    #[derive(Deserialize)]
    #[serde(tag = "provider", deny_unknown_fields)]
    enum Provider {
        #[serde(rename = "doc")]
        Doc { settings: DocOptions },
    }
    let Provider::Doc { settings } = Provider::deserialize(deserializer)?;
    Ok(Some(settings))
}

#[cfg(any(not(feature = "ocr"), not(feature = "ppdoc"), not(feature = "doc")))]
fn unsupported<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Option<()>, D::Error> {
    serde::de::IgnoredAny::deserialize(deserializer)?;
    Err(D::Error::custom("this legalpdf binary was built without the provider the request names"))
}

/// The PDF's document as `request` asks for it: read from its parse cache, or, given its bytes, parsed and cached
/// (failing where the cache cannot be written, since its cache key is how the reading is found again), reporting
/// recognized pages of the pages to recognize as they finish. None where there are no bytes and nothing is cached, or
/// where the request names a cache key no longer cached.
pub fn read_pdf(bytes: Option<&[u8]>, request: &PdfRequest, progress: RecognitionProgress<'_>) -> Result<Option<PdfDocument>> {
    parse_pdf(bytes, request, progress, bytes.is_some())
}
