use crate::engine::{parse_pdf, ParseOptions};
#[cfg(feature = "kraken")]
use crate::KrakenOptions;
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
use crate::PPDocOptions;
use crate::{Error, PdfLookupRequest, PdfStructureLookup, Result};
#[cfg(feature = "ocr")]
use crate::{OcrOptions, TesseractOptions};
pub use legal_pdf_support::{PdfDocument, PdfSummary};
use serde::Deserialize;
use std::path::PathBuf;

const MAX_SELECTED_PAGES: usize = 1_000;

fn present<'de, D, T>(deserializer: D) -> std::result::Result<Option<T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
pub struct PdfRequest {
    cache_dir: Option<PathBuf>,
    #[serde(default, deserialize_with = "present")]
    cache_key: Option<String>,
    #[serde(default, deserialize_with = "present")]
    expected_source_sha256: Option<String>,
    #[serde(default, deserialize_with = "present")]
    max_output_bytes: Option<usize>,
    #[serde(default, deserialize_with = "present")]
    pages: Option<Vec<usize>>,
    id: Option<String>,
    url: Option<String>,
    #[cfg(feature = "ocr")]
    #[serde(default, deserialize_with = "present")]
    ocr: Option<OcrRequest>,
    #[cfg(not(feature = "ocr"))]
    #[serde(default, deserialize_with = "present")]
    ocr: Option<serde::de::IgnoredAny>,
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    #[serde(default, deserialize_with = "present")]
    layout: Option<LayoutRequest>,
    #[cfg(not(any(feature = "ppdoc-full", feature = "ppdoc-openvino")))]
    #[serde(default, deserialize_with = "present")]
    layout: Option<serde::de::IgnoredAny>,
}

#[cfg(feature = "ocr")]
#[derive(Deserialize)]
#[serde(tag = "provider", deny_unknown_fields)]
enum OcrRequest {
    #[serde(rename = "tesseract")]
    Tesseract {
        #[serde(default)]
        settings: TesseractOptions,
    },
    #[cfg(feature = "kraken")]
    #[serde(rename = "kraken-lite")]
    Kraken {
        #[serde(default)]
        settings: KrakenOptions,
    },
    #[cfg(not(feature = "kraken"))]
    #[serde(rename = "kraken-lite")]
    Kraken {
        #[serde(rename = "settings", default = "ignored_any")]
        _settings: serde::de::IgnoredAny,
    },
}

#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
#[derive(Deserialize)]
#[serde(tag = "provider", deny_unknown_fields)]
enum LayoutRequest {
    #[serde(rename = "ppdoc")]
    Ppdoc {
        #[serde(default)]
        settings: PPDocOptions,
    },
}

#[cfg(all(feature = "ocr", not(feature = "kraken")))]
fn ignored_any() -> serde::de::IgnoredAny {
    serde::de::IgnoredAny
}

fn selected_pages(pages: Option<&[usize]>) -> Result<Option<Vec<usize>>> {
    let Some(pages) = pages else {
        return Ok(None);
    };
    if pages.is_empty() || pages.len() > MAX_SELECTED_PAGES {
        return Err(Error::Message(format!(
            "document request pages requires 1 to {MAX_SELECTED_PAGES} pages"
        )));
    }
    if pages.contains(&0) {
        return Err(Error::Message(
            "document request pages must be positive integers".to_owned(),
        ));
    }
    let mut selected = pages.iter().map(|page| page - 1).collect::<Vec<_>>();
    selected.sort_unstable();
    selected.dedup();
    Ok(Some(selected))
}

fn sha256_field(value: &Option<String>, key: &str) -> Result<Option<String>> {
    value
        .as_ref()
        .map(|value| {
            if value.len() == 64
                && value
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
            {
                Ok(value.clone())
            } else {
                Err(Error::Message(format!(
                    "document request {key} must be lowercase SHA-256"
                )))
            }
        })
        .transpose()
}

fn parse_options(request: &PdfRequest) -> Result<ParseOptions> {
    let mut options = ParseOptions {
        cache_dir: request.cache_dir.clone(),
        ocr_pages: selected_pages(request.pages.as_deref())?,
        cache_key: sha256_field(&request.cache_key, "cache_key")?,
        max_output_bytes: request.max_output_bytes,
        use_cache: true,
        expected_source_sha256: sha256_field(
            &request.expected_source_sha256,
            "expected_source_sha256",
        )?,
        ..ParseOptions::default()
    };

    #[cfg(feature = "ocr")]
    if let Some(request) = &request.ocr {
        options.ocr = match request {
            OcrRequest::Tesseract { settings } => Some(OcrOptions::Tesseract(settings.clone())),
            #[cfg(feature = "kraken")]
            OcrRequest::Kraken { settings } => Some(OcrOptions::Kraken(settings.clone())),
            #[cfg(not(feature = "kraken"))]
            OcrRequest::Kraken { .. } => {
                return Err(Error::Message(
                    "kraken-lite requires a legalpdf binary built with the kraken feature"
                        .to_owned(),
                ));
            }
        };
    }
    #[cfg(not(feature = "ocr"))]
    if request.ocr.is_some() {
        return Err(Error::Message(
            "this legalpdf binary was built without the `ocr` feature".to_owned(),
        ));
    }

    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    if let Some(LayoutRequest::Ppdoc { settings }) = &request.layout {
        options.ppdoc = Some(settings.clone());
    }
    #[cfg(not(any(feature = "ppdoc-full", feature = "ppdoc-openvino")))]
    if request.layout.is_some() {
        return Err(Error::Message(
            "layout requires a legalpdf binary built with a layout feature".to_owned(),
        ));
    }
    Ok(options)
}

fn validate_selected_pages(selected: Option<&[usize]>, count: usize) -> Result<()> {
    let Some(selected) = selected else {
        return Ok(());
    };
    if selected.iter().any(|page| *page >= count) {
        return Err(Error::Message(
            "document request pages contains a page beyond the source PDF".to_owned(),
        ));
    }
    Ok(())
}

pub fn derive_pdf_document(bytes: &[u8], request: &PdfRequest) -> Result<PdfDocument> {
    let options = parse_options(request)?;
    let document = parse_pdf(Some(bytes), &options)?
        .ok_or_else(|| Error::Message("PDF cache miss after parsing source bytes".to_owned()))?;
    finish_pdf_document(document, request, &options)
}

pub fn prepare_pdf_document(bytes: &[u8], request: &PdfRequest) -> Result<PdfSummary> {
    let mut options = parse_options(request)?;
    options.require_cache_write = true;
    let document = parse_pdf(Some(bytes), &options)?
        .ok_or_else(|| Error::Message("PDF cache miss after parsing source bytes".to_owned()))?;
    validate_selected_pages(options.ocr_pages.as_deref(), document.page_count())?;
    Ok(document.summary().clone())
}

pub fn restore_pdf_document(request: &PdfRequest) -> Result<Option<PdfDocument>> {
    let options = parse_options(request)?;
    parse_pdf(None, &options)?
        .map(|document| finish_pdf_document(document, request, &options))
        .transpose()
}

fn finish_pdf_document(
    mut document: PdfDocument,
    request: &PdfRequest,
    options: &ParseOptions,
) -> Result<PdfDocument> {
    validate_selected_pages(options.ocr_pages.as_deref(), document.page_count())?;
    let structure = document.structure_mut();
    if let Some(id) = &request.id {
        structure.document_id = id.to_owned();
    }
    structure.url = request.url.clone();
    Ok(document)
}

pub fn query_pdf_document(document: &PdfDocument, query: &PdfLookupRequest) -> PdfStructureLookup {
    document.lookup(query)
}

pub fn pdf_document_summary(document: &PdfDocument) -> &PdfSummary {
    document.summary()
}
