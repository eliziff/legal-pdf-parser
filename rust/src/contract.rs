use crate::engine::{parse_pdf, ParseOptions};
#[cfg(feature = "kraken")]
use crate::KrakenOptions;
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
use crate::PPDocOptions;
use crate::{structure_lookup, Error, Result};
#[cfg(feature = "ocr")]
use crate::{OcrOptions, TesseractOptions};
use legal_structure::{InstrumentCrossReferenceGraph, NodeKind};
use serde_json::{json, Value};
use std::path::PathBuf;

const MAX_SELECTED_PAGES: usize = 1_000;

pub struct PdfDocumentResult {
    document: crate::LegalDocument,
}

impl PdfDocumentResult {
    pub fn structure(&self) -> &legal_structure::DocumentStructure {
        &self.document.structure_graph
    }

    pub fn cross_references(&self) -> Option<&InstrumentCrossReferenceGraph> {
        self.document.structure_graph.cross_references.as_ref()
    }
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
    let expected_source_sha256 = value
        .get("expected_source_sha256")
        .map(|value| {
            value
                .as_str()
                .filter(|value| {
                    value.len() == 64
                        && value
                            .bytes()
                            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
                })
                .map(str::to_owned)
                .ok_or_else(|| {
                    Error::Message(
                        "document request expected_source_sha256 must be lowercase SHA-256"
                            .to_owned(),
                    )
                })
        })
        .transpose()?;
    let source_name = value
        .get("source_name")
        .map(|value| {
            value
                .as_str()
                .filter(|name| {
                    !name.is_empty()
                        && name.len() <= 260
                        && !name.contains('/')
                        && !name.contains('\\')
                        && !name.chars().any(char::is_control)
                })
                .map(str::to_owned)
                .ok_or_else(|| Error::Message("document request source_name is invalid".to_owned()))
        })
        .transpose()?;
    let mut options = ParseOptions {
        cache_dir: value
            .get("cache_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        ocr_pages: selected_pages(value)?,
        use_cache: true,
        expected_source_sha256,
        source_name,
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
    bytes: &[u8],
    value: &Value,
    include_pairing_audit: bool,
) -> Result<PdfDocumentResult> {
    let options = parse_options(value)?;
    let document = parse_pdf(Some(bytes), &options)?
        .ok_or_else(|| Error::Message("PDF cache miss after parsing source bytes".to_owned()))?;
    finish_pdf_document(document, value, include_pairing_audit, &options)
}

pub fn restore_pdf_document(
    value: &Value,
    include_pairing_audit: bool,
) -> Result<Option<PdfDocumentResult>> {
    let options = parse_options(value)?;
    parse_pdf(None, &options)?
        .map(|document| finish_pdf_document(document, value, include_pairing_audit, &options))
        .transpose()
}

fn finish_pdf_document(
    mut document: legal_pdf_core::model::LegalDocument,
    value: &Value,
    include_pairing_audit: bool,
    options: &ParseOptions,
) -> Result<PdfDocumentResult> {
    validate_selected_pages(options.ocr_pages.as_deref(), document.page_count)?;
    if let Some(id) = value.get("id").and_then(Value::as_str) {
        document.structure_graph.document_id = id.to_owned();
    }
    document.structure_graph.url = value.get("url").and_then(Value::as_str).map(str::to_owned);
    if !include_pairing_audit {
        document.pairing_audit = None;
    }
    Ok(PdfDocumentResult { document })
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

pub fn pdf_document_summary(document: &PdfDocumentResult) -> Value {
    let document = &document.document;
    let pdf = document.metadata.get("pdf").unwrap_or(&Value::Null);
    let nodes = &document.structure_graph.nodes;
    json!({
        "sha256": &document.source_sha256,
        "parserVersion": &document.parser_version,
        "cacheKey": document.provenance.get("deterministic_cache_key"),
        "pageCount": document.page_count,
        "projectionPageCount": document.pdf_source_map.pages.len(),
        "status": &document.status,
        "pagesNeedingOcr": pdf.get("pages_needing_ocr"),
        "ocrRoutedPages": pdf.get("ocr_routed_pages"),
        "counts": {
            "paragraphs": nodes.iter().filter(|node| node.kind == NodeKind::Paragraph).count(),
            "sections": nodes.iter().filter(|node| node.kind == NodeKind::Section).count(),
            "footnotes": document.structure_graph.notes.len(),
            "tables": document.pdf_source_map.table_ids.len(),
            "images": document.pdf_source_map.image_ids.len(),
        },
    })
}
