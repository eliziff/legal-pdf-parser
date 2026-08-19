use crate::engine::{parse_pdf, ParseOptions};
#[cfg(feature = "kraken")]
use crate::KrakenOptions;
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
use crate::PPDocOptions;
use crate::{
    source_doc, structure_lookup, Error, OcrOptions, Result, TesseractOptions, PARSER_VERSION,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const REQUEST_SCHEMA: &str = "legalpdf.document-request.v1";
const RESULT_SCHEMA: &str = "legalpdf.document-result.v1";
const MAX_SELECTED_PAGES: usize = 1_000;
const MAX_REQUEST_BYTES: usize = 64 * 1024;

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

fn parse_options(value: &Value, operation: &str) -> Result<ParseOptions> {
    if operation == "source_doc" && value.get("pages").is_some() {
        return Err(Error::Message(
            "selected pages are not supported by source_doc".to_owned(),
        ));
    }
    if operation == "structure_lookup"
        && value.get("pages").is_some()
        && value.pointer("/query/locator_kind").and_then(Value::as_str) != Some("page")
    {
        return Err(Error::Message(
            "selected pages require a page structure lookup".to_owned(),
        ));
    }
    let mut options = ParseOptions {
        cache_dir: value
            .get("cache_dir")
            .and_then(Value::as_str)
            .map(PathBuf::from),
        ocr_pages: selected_pages(value)?,
        use_cache: true,
        ..ParseOptions::default()
    };

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
                        "cpu_fallback",
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
        #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
        {
            options.ppdoc = Some(serde_json::from_value::<PPDocOptions>(settings)?);
        }
        #[cfg(not(any(feature = "ppdoc", feature = "ppdoc-openvino")))]
        {
            return Err(Error::Message(
                "layout requires a legalpdf binary built with a layout feature".to_owned(),
            ));
        }
    }
    Ok(options)
}

fn validate_selected_pages(value: &Value, selected: Option<&[usize]>, count: usize) -> Result<()> {
    let Some(selected) = selected else {
        return Ok(());
    };
    if selected.iter().any(|page| *page >= count) {
        return Err(Error::Message(
            "document request pages contains a page beyond the source PDF".to_owned(),
        ));
    }
    if value.get("operation").and_then(Value::as_str) != Some("structure_lookup") {
        return Ok(());
    }
    let query = value
        .get("query")
        .ok_or_else(|| Error::Message("document request has no query".to_owned()))?;
    let locator = query
        .get("locator")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Message("page lookup has no exact page locator".to_owned()))?;
    let inline = crate::projection::numeric_range("page", locator);
    let start = inline
        .map(|range| range.0)
        .or_else(|| crate::projection::parse_ordinal("page", locator))
        .ok_or_else(|| Error::Message("page lookup has no exact page locator".to_owned()))?;
    let end = query
        .get("end_locator")
        .filter(|value| !value.is_null())
        .map(|value| {
            value
                .as_str()
                .and_then(|locator| crate::projection::parse_ordinal("page", locator))
                .ok_or_else(|| Error::Message("page lookup has no exact end locator".to_owned()))
        })
        .transpose()?
        .unwrap_or_else(|| inline.map(|range| range.1).unwrap_or(start));
    let context = query
        .get("context_blocks")
        .map(|value| {
            value
                .as_u64()
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value <= 2)
                .ok_or_else(|| {
                    Error::Message("page lookup context_blocks must be 0 to 2".to_owned())
                })
        })
        .transpose()?
        .unwrap_or(0);
    if start == 0 || end < start || end > count {
        return Err(Error::Message(
            "page lookup range is outside the source PDF".to_owned(),
        ));
    }
    let required_start = start.saturating_sub(context).max(1);
    let required_end = end.saturating_add(context).min(count);
    if (required_start..=required_end).any(|page| !selected.contains(&(page - 1))) {
        return Err(Error::Message(
            "selected pages do not cover the requested page range and context".to_owned(),
        ));
    }
    Ok(())
}

fn source_hash(path: &Path) -> Result<String> {
    let file = File::open(path).map_err(|source| Error::io(path, source))?;
    let mut reader = BufReader::new(file);
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|source| Error::io(path, source))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn inspect_request(value: &Value) -> Result<Value> {
    let path = Path::new(string(value, "source_pdf")?);
    if !path.is_file()
        || path
            .extension()
            .and_then(|value| value.to_str())
            .is_none_or(|value| !value.eq_ignore_ascii_case("pdf"))
    {
        return Err(Error::Message("input must be a PDF".to_owned()));
    }
    let inspection = pdf_inspector::detector::detect_pdf_type(path)
        .map_err(|error| Error::Message(format!("PDF inspection failed: {error}")))?;
    let sha256 = source_hash(path)?;
    Ok(json!({
        "schema_version": RESULT_SCHEMA,
        "operation": "inspect",
        "source": {
            "sha256": sha256,
            "parser_version": PARSER_VERSION,
            "cache_key": Value::Null,
            "cache_hit": false,
            "page_count": inspection.page_count,
        },
        "result": {
            "page_count": inspection.page_count,
            "pdf_type": format!("{:?}", inspection.pdf_type),
            "confidence": inspection.confidence,
            "pages_needing_ocr": inspection.pages_needing_ocr
                .iter()
                .map(|page| page + 1)
                .collect::<Vec<_>>(),
        },
    }))
}

fn source_summary(document: &crate::LegalDocument) -> Value {
    json!({
        "sha256": document.source_sha256,
        "parser_version": document.parser_version,
        "cache_key": document.provenance.get("deterministic_cache_key"),
        "cache_hit": document.provenance.get("cache_hit").and_then(Value::as_bool).unwrap_or(false),
        "page_count": document.page_count,
    })
}

fn physical_pages(value: Option<&Value>) -> Value {
    Value::Array(
        value
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(Value::as_u64)
            .map(|page| Value::from(page + 1))
            .collect(),
    )
}

fn prepare_summary(document: &crate::LegalDocument, selected: Option<&[usize]>) -> Value {
    let pdf = document.metadata.get("pdf").unwrap_or(&Value::Null);
    let prepared_pages = selected.map_or_else(
        || json!({"selection": "full", "count": document.page_count}),
        |pages| {
            json!({
                "selection": "selected",
                "count": pages.len(),
                "pages": pages.iter().map(|page| page + 1).collect::<Vec<_>>(),
            })
        },
    );
    json!({
        "status": document.status,
        "page_count": document.page_count,
        "prepared_pages": prepared_pages,
        "cache_validated": true,
        "pdf_type": pdf.get("pdf_type"),
        "confidence": pdf.get("confidence"),
        "pages_needing_ocr": physical_pages(pdf.get("pages_needing_ocr")),
        "ocr_routed_pages": physical_pages(pdf.get("ocr_routed_pages")),
        "counts": {
            "paragraphs": document.paragraphs.len(),
            "sections": document.sections.len(),
            "footnotes": document.footnotes.len(),
            "tables": document.tables.len(),
            "images": document.images.len(),
            "diagnostics": document.diagnostics.len(),
        },
    })
}

pub fn document_request(value: &Value) -> Result<Value> {
    if serde_json::to_vec(value)?.len() > MAX_REQUEST_BYTES {
        return Err(Error::Message("document request exceeds 64 KiB".to_owned()));
    }
    if value.get("schema_version").and_then(Value::as_str) != Some(REQUEST_SCHEMA) {
        return Err(Error::Message(
            "unsupported document request schema".to_owned(),
        ));
    }
    let operation = string(value, "operation")?;
    if !matches!(
        operation,
        "inspect" | "prepare" | "source_doc" | "structure_lookup"
    ) {
        return Err(Error::Message(format!(
            "unsupported document operation: {operation}"
        )));
    }
    let allowed = match operation {
        "inspect" => &["schema_version", "operation", "source_pdf"][..],
        "prepare" => &[
            "schema_version",
            "operation",
            "source_pdf",
            "cache_dir",
            "pages",
            "ocr",
            "layout",
        ][..],
        "source_doc" => &[
            "schema_version",
            "operation",
            "source_pdf",
            "cache_dir",
            "ocr",
            "layout",
            "id",
            "url",
        ][..],
        "structure_lookup" => &[
            "schema_version",
            "operation",
            "source_pdf",
            "cache_dir",
            "pages",
            "ocr",
            "layout",
            "query",
        ][..],
        _ => unreachable!(),
    };
    if let Some(field) = value.as_object().and_then(|request| {
        request
            .keys()
            .find(|field| !allowed.contains(&field.as_str()))
    }) {
        return Err(Error::Message(format!(
            "unknown document request field: {field}"
        )));
    }
    for field in ["id", "url"] {
        if value
            .get(field)
            .is_some_and(|value| value.as_str().is_none_or(|value| value.len() > 2_048))
        {
            return Err(Error::Message(format!(
                "document request {field} must be a string of at most 2048 bytes"
            )));
        }
    }
    if operation == "inspect" {
        return inspect_request(value);
    }
    let options = parse_options(value, operation)?;
    let selected = options.ocr_pages.clone();
    let source = string(value, "source_pdf")?;
    let document = parse_pdf(source, &options)?;
    validate_selected_pages(value, selected.as_deref(), document.page_count)?;
    let result = match operation {
        "prepare" => prepare_summary(&document, selected.as_deref()),
        "source_doc" => source_doc(
            &document,
            value.get("id").and_then(Value::as_str),
            value.get("url").and_then(Value::as_str),
        ),
        "structure_lookup" => structure_lookup(
            &document,
            value
                .get("query")
                .ok_or_else(|| Error::Message("document request has no query".to_owned()))?,
        )?,
        _ => unreachable!(),
    };
    Ok(json!({
        "schema_version": RESULT_SCHEMA,
        "operation": operation,
        "source": source_summary(&document),
        "result": result,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::{dictionary, Document, Object};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn request_requires_a_source_pdf() {
        let error = document_request(&json!({
            "schema_version": REQUEST_SCHEMA,
            "operation": "source_doc",
        }))
        .unwrap_err();
        assert!(error.to_string().contains("source_pdf"));
    }

    #[test]
    fn selected_pages_are_bounded_deduplicated_indexes() {
        assert_eq!(
            selected_pages(&json!({"pages": [5, 1, 5]})).unwrap(),
            Some(vec![0, 4])
        );
        assert!(selected_pages(&json!({"pages": [0]})).is_err());
        assert!(selected_pages(&json!({"pages": []})).is_err());
    }

    #[test]
    fn selected_page_lookup_covers_its_range_and_context() {
        let request = json!({
            "operation": "structure_lookup",
            "query": {
                "locator_kind": "page",
                "locator": "page 5",
                "context_blocks": 1,
            },
        });
        assert!(validate_selected_pages(&request, Some(&[3, 4, 5]), 10).is_ok());
        assert!(validate_selected_pages(&request, Some(&[4]), 10).is_err());
    }

    #[test]
    fn structure_lookup_accepts_selected_pages_at_the_strict_boundary() {
        let error = document_request(&json!({
            "schema_version": REQUEST_SCHEMA,
            "operation": "structure_lookup",
            "source_pdf": "missing.pdf",
            "pages": [5],
            "query": {"locator_kind": "page", "locator": "5"},
        }))
        .unwrap_err();
        assert!(!error.to_string().contains("unknown document request field"));
    }

    #[test]
    fn selected_pages_reject_non_page_lookups() {
        let error = parse_options(
            &json!({
                "pages": [5],
                "query": {"locator_kind": "footnote", "locator": "5"},
            }),
            "structure_lookup",
        )
        .unwrap_err();
        assert!(error.to_string().contains("page structure lookup"));

        let error = parse_options(&json!({"pages": [5]}), "source_doc").unwrap_err();
        assert!(error.to_string().contains("source_doc"));
    }

    #[test]
    fn provider_settings_and_top_level_fields_are_strict() {
        assert!(parse_options(
            &json!({
                "ocr": {"provider": "tesseract", "settings": {"mystery": true}},
            }),
            "prepare",
        )
        .is_err());
        assert!(parse_options(
            &json!({
                "ocr": {
                    "provider": "tesseract",
                    "settings": {},
                    "legacy": true,
                },
            }),
            "prepare",
        )
        .is_err());
        assert!(parse_options(
            &json!({
                "layout": {
                    "provider": "ppdoc",
                    "settings": {"legacy": true},
                },
            }),
            "prepare",
        )
        .is_err());
        for operation in ["inspect", "prepare", "source_doc", "structure_lookup"] {
            let error = document_request(&json!({
                "schema_version": REQUEST_SCHEMA,
                "operation": operation,
                "source_pdf": "missing.pdf",
                "artifact": "legacy",
            }))
            .unwrap_err();
            assert!(error.to_string().contains("unknown document request field"));
        }
    }

    #[test]
    fn inspect_is_direct_and_has_no_cache_key() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "legalpdf-inspect-{}-{stamp}.pdf",
            std::process::id()
        ));
        let mut pdf = Document::with_version("1.4");
        let pages_id = pdf.new_object_id();
        let page_id = pdf.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => Object::Reference(pages_id),
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => dictionary! {},
        });
        pdf.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => vec![Object::Reference(page_id)],
                "Count" => 1,
            }),
        );
        let catalog = pdf.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference(pages_id),
        });
        pdf.trailer.set("Root", Object::Reference(catalog));
        pdf.save(&path).unwrap();

        let result = document_request(&json!({
            "schema_version": REQUEST_SCHEMA,
            "operation": "inspect",
            "source_pdf": path,
        }))
        .unwrap();
        assert_eq!(result["result"]["page_count"], 1);
        assert!(result["source"]["cache_key"].is_null());
        assert_eq!(result["source"]["cache_hit"], false);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn prepare_summary_reports_physical_ocr_pages_from_pdf_metadata() {
        let document: crate::LegalDocument = serde_json::from_value(json!({
            "document_id": "document",
            "source_name": "source.pdf",
            "source_sha256": "a".repeat(64),
            "page_count": 3,
            "status": "ready",
            "pages": [],
            "paragraphs": [],
            "sections": [],
            "footnotes": [],
            "tables": [],
            "images": [],
            "diagnostics": [],
            "metadata": {"pdf": {
                "pdf_type": "Scanned",
                "confidence": 0.95,
                "pages_needing_ocr": [2],
                "ocr_routed_pages": [0, 1]
            }},
            "provenance": {}
        }))
        .unwrap();

        let result = prepare_summary(&document, None);
        assert_eq!(result["pdf_type"], "Scanned");
        assert_eq!(result["pages_needing_ocr"], json!([3]));
        assert_eq!(result["ocr_routed_pages"], json!([1, 2]));
    }
}
