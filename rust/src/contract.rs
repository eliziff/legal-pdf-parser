use crate::engine::{parse_pdf, ParseOptions};
#[cfg(feature = "kraken")]
use crate::KrakenOptions;
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
use crate::PPDocOptions;
use crate::{source_doc, structure_lookup, Error, Result};
#[cfg(feature = "ocr")]
use crate::{OcrOptions, TesseractOptions};
use serde_json::{json, Value};
use std::path::PathBuf;

const MAX_SELECTED_PAGES: usize = 1_000;

#[derive(Debug)]
pub struct PdfDocumentResult {
    document: crate::LegalDocument,
    source_doc: Option<Value>,
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| Error::Message(format!("document request has no {key}")))
}

fn provider_spec(value: &Value, key: &str) -> Result<Option<(String, Value)>> {
    let Some(spec) = value.get(key) else {
        return Ok(None);
    };
    let spec = spec
        .as_object()
        .ok_or_else(|| Error::Message(format!("document request {key} is not an object")))?;
    if let Some(field) = spec
        .keys()
        .find(|field| !matches!(field.as_str(), "provider" | "settings"))
    {
        return Err(Error::Message(format!(
            "unknown document request {key} field: {field}"
        )));
    }
    let provider = spec
        .get("provider")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty() && value.len() <= 100)
        .ok_or_else(|| Error::Message(format!("document request {key} has no provider")))?;
    let settings = spec.get("settings").cloned().unwrap_or_else(|| json!({}));
    Ok(Some((provider.to_owned(), settings)))
}

fn strict_settings(provider: &str, settings: &Value, allowed: &[&str]) -> Result<()> {
    let settings = settings.as_object().ok_or_else(|| {
        Error::Message(format!(
            "document request {provider} settings is not an object"
        ))
    })?;
    if let Some(field) = settings
        .keys()
        .find(|field| !allowed.contains(&field.as_str()))
    {
        return Err(Error::Message(format!(
            "unknown document request {provider} setting: {field}"
        )));
    }
    Ok(())
}

fn selected_pages(value: &Value) -> Result<Option<Vec<usize>>> {
    let Some(pages) = value.get("pages") else {
        return Ok(None);
    };
    let pages = pages
        .as_array()
        .filter(|pages| !pages.is_empty() && pages.len() <= MAX_SELECTED_PAGES)
        .ok_or_else(|| {
            Error::Message(format!(
                "document request pages requires 1 to {MAX_SELECTED_PAGES} pages"
            ))
        })?;
    let mut selected = Vec::with_capacity(pages.len());
    for page in pages {
        let page = page
            .as_u64()
            .and_then(|page| usize::try_from(page).ok())
            .filter(|page| *page > 0)
            .ok_or_else(|| {
                Error::Message("document request pages must be positive integers".to_owned())
            })?;
        let index = page - 1;
        if !selected.contains(&index) {
            selected.push(index);
        }
    }
    selected.sort_unstable();
    Ok(Some(selected))
}

fn parse_options(value: &Value) -> Result<ParseOptions> {
    let mut options = ParseOptions {
        cache_dir: value
            .get("cache_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        ocr_pages: selected_pages(value)?,
        use_cache: true,
        ..ParseOptions::default()
    };

    #[cfg(feature = "ocr")]
    if let Some((provider, settings)) = provider_spec(value, "ocr")? {
        options.ocr = match provider.as_str() {
            "tesseract" => {
                strict_settings(
                    &provider,
                    &settings,
                    &[
                        "command",
                        "language",
                        "dpi",
                        "psm",
                        "timeout_seconds",
                        "expected_identity",
                    ],
                )?;
                Some(OcrOptions::Tesseract(serde_json::from_value::<
                    TesseractOptions,
                >(settings)?))
            }
            "kraken-lite" => {
                strict_settings(
                    &provider,
                    &settings,
                    &[
                        "model",
                        "codec",
                        "runtime",
                        "runtime_wheel",
                        "python",
                        "blla_pack",
                        "recognizer_pack",
                        "tesseract_library",
                        "dpi",
                        "threads",
                        "workers",
                        "layout_workers",
                        "batch_size",
                        "runtime_batch_size",
                        "width_bucket",
                        "width_scale",
                        "tier",
                        "layout",
                        "backend",
                        "device",
                        "cpu_arena",
                        "timeout_seconds",
                        "expected_identity",
                    ],
                )?;
                #[cfg(feature = "kraken")]
                {
                    Some(OcrOptions::Kraken(serde_json::from_value::<KrakenOptions>(
                        settings,
                    )?))
                }
                #[cfg(not(feature = "kraken"))]
                {
                    return Err(Error::Message(
                        "kraken-lite requires a legalpdf binary built with the kraken feature"
                            .to_owned(),
                    ));
                }
            }
            _ => {
                return Err(Error::Message(format!(
                    "unsupported OCR provider: {provider}"
                )))
            }
        };
    }
    #[cfg(not(feature = "ocr"))]
    if value.get("ocr").is_some() {
        return Err(Error::Message(
            "this legalpdf binary was built without the `ocr` feature".to_owned(),
        ));
    }

    if let Some((provider, settings)) = provider_spec(value, "layout")? {
        if provider != "ppdoc" {
            return Err(Error::Message(format!(
                "unsupported layout provider: {provider}"
            )));
        }
        strict_settings(
            &provider,
            &settings,
            &[
                "model_pack",
                "runtime",
                "cache_dir",
                "threads",
                "threshold",
                "render_dpi",
                "onednn",
                "backend",
                "device",
                "cpu_fallback",
                "expected_identity",
            ],
        )?;
        #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
        {
            options.ppdoc = Some(serde_json::from_value::<PPDocOptions>(settings)?);
        }
        #[cfg(not(any(feature = "ppdoc-full", feature = "ppdoc-openvino")))]
        {
            return Err(Error::Message(
                "layout requires a legalpdf binary built with a layout feature".to_owned(),
            ));
        }
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

pub fn derive_pdf_document(
    value: &Value,
    include_source_doc: bool,
    include_pairing_audit: bool,
) -> Result<PdfDocumentResult> {
    let options = parse_options(value)?;
    let selected = options.ocr_pages.clone();
    let mut document = parse_pdf(string(value, "source_pdf")?, &options)?;
    validate_selected_pages(selected.as_deref(), document.page_count)?;
    let source_doc = include_source_doc.then(|| {
        source_doc(
            &document,
            value.get("id").and_then(Value::as_str),
            value.get("url").and_then(Value::as_str),
        )["source_doc"]
            .clone()
    });
    if !include_pairing_audit {
        document.pairing_audit = None;
    }
    Ok(PdfDocumentResult {
        document,
        source_doc,
    })
}

pub fn query_pdf_document(document: &PdfDocumentResult, query: &Value) -> Result<Value> {
    structure_lookup(&document.document, query)
}

pub fn pdf_document_snapshot(document: &PdfDocumentResult) -> Value {
    let pdf = document
        .document
        .metadata
        .get("pdf")
        .unwrap_or(&Value::Null);
    json!({
        "structure": &document.document.structure_graph,
        "pdf_source_map": &document.document.pdf_source_map,
        "source_doc": &document.source_doc,
        "pairing_audit": &document.document.pairing_audit,
        "source": {
            "sha256": &document.document.source_sha256,
            "parser_version": &document.document.parser_version,
            "cache_key": document.document.provenance.get("deterministic_cache_key"),
            "page_count": document.document.page_count,
            "status": &document.document.status,
            "pages_needing_ocr": pdf.get("pages_needing_ocr"),
            "ocr_routed_pages": pdf.get("ocr_routed_pages"),
        },
    })
}
