use crate::{
    apply_docx_links, assess_docx_route, deterministic_docx_intents, extract_citation_fields,
    extract_docx_gold, load_artifacts, load_projection_artifacts, lookup_artifact_footnote,
    lookup_footnote, plan_docx_links, repair_context, repair_identity, repair_scopes, source_doc,
    split_citations, split_citations_recall_first, structure_lookup, to_alr_payload,
    to_toa_text_units, validate_docx_response, validate_repair_response, write_artifacts,
    DocxPlanOptions, Error, Result,
};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Message(format!("contract input has no {key}")))
}

fn optional_u32(value: &Value, key: &str) -> Result<Option<u32>> {
    value
        .get(key)
        .filter(|item| !item.is_null())
        .map(|item| {
            item.as_u64()
                .and_then(|number| u32::try_from(number).ok())
                .ok_or_else(|| Error::Message(format!("contract input {key} is not an integer")))
        })
        .transpose()
}

fn optional_usize(value: &Value, key: &str) -> Result<Option<usize>> {
    value
        .get(key)
        .filter(|item| !item.is_null())
        .map(|item| {
            item.as_u64()
                .and_then(|number| usize::try_from(number).ok())
                .ok_or_else(|| Error::Message(format!("contract input {key} is not an integer")))
        })
        .transpose()
}

fn temporary_directory() -> Result<PathBuf> {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::Message(format!("system clock is invalid: {error}")))?
        .as_nanos();
    let root =
        std::env::temp_dir().join(format!("legalpdf-contract-{}-{stamp}", std::process::id()));
    fs::create_dir(&root).map_err(|source| Error::io(&root, source))?;
    Ok(root)
}

fn artifact_hashes(document: &crate::LegalDocument, compact: bool) -> Result<Value> {
    let root = temporary_directory()?;
    let result = (|| {
        write_artifacts(document, &root, compact)?;
        let mut hashes = serde_json::Map::new();
        for name in [
            "document.json",
            "pages.jsonl",
            "paragraphs.jsonl",
            "sections.jsonl",
            "footnotes.jsonl",
            "tables.jsonl",
            "images.jsonl",
            "diagnostics.jsonl",
            "repairs.jsonl",
        ] {
            let path = root.join(name);
            let bytes = fs::read(&path).map_err(|source| Error::io(&path, source))?;
            hashes.insert(
                name.to_owned(),
                json!({
                    "bytes": bytes.len(),
                    "sha256": format!("{:x}", Sha256::digest(&bytes)),
                }),
            );
        }
        Ok(Value::Object(hashes))
    })();
    let cleanup = fs::remove_dir_all(&root).map_err(|source| Error::io(&root, source));
    match (result, cleanup) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Ok(value), Ok(())) => Ok(value),
    }
}

fn docx_notes(gold: &Value) -> Vec<Value> {
    gold.get("footnotes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|note| {
            json!({
                "id": note["ooxml_id"],
                "label": note["label"],
                "text": note["body"],
                "proposition": note["passage_since_prior_note"],
            })
        })
        .collect()
}

fn stable_docx_plan(mut plan: Value) -> Value {
    if let Some(telemetry) = plan.get_mut("telemetry").and_then(Value::as_object_mut) {
        telemetry.remove("elapsed_seconds");
        if let Some(batches) = telemetry.get_mut("batches").and_then(Value::as_array_mut) {
            for batch in batches {
                if let Some(batch) = batch.as_object_mut() {
                    batch.remove("elapsed_seconds");
                }
            }
        }
    }
    plan
}

fn stable_docx_gold(mut gold: Value) -> Value {
    if let Some(gold) = gold.as_object_mut() {
        gold.remove("source_sha256");
    }
    gold
}

pub fn replay_contract(value: &Value) -> Result<Value> {
    if value.get("schema_version").and_then(Value::as_str) != Some("legalpdf.contract-input.v1") {
        return Err(Error::Message(
            "unsupported contract input schema".to_owned(),
        ));
    }
    let operation = string(value, "operation")?;
    let artifact = value.get("artifact").and_then(Value::as_str).map(Path::new);
    match operation {
        "separator_contract" => {
            let images = value
                .get("images")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no images".to_owned()))?;
            let mut scans = Vec::new();
            for image in images {
                let width = optional_usize(image, "width")?.unwrap_or_default();
                let height = optional_usize(image, "height")?.unwrap_or_default();
                let background = image
                    .get("background")
                    .and_then(Value::as_u64)
                    .and_then(|number| u8::try_from(number).ok())
                    .unwrap_or(255);
                let mut gray = vec![background; width.saturating_mul(height)];
                for fill in image
                    .get("fills")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let fields = fill.as_array().ok_or_else(|| {
                        Error::Message("separator fill is not an array".to_owned())
                    })?;
                    let numbers = fields
                        .iter()
                        .map(|field| field.as_u64().unwrap_or_default() as usize)
                        .collect::<Vec<_>>();
                    if numbers.len() != 5 {
                        return Err(Error::Message(
                            "separator fill must have five values".to_owned(),
                        ));
                    }
                    for y in numbers[1].min(height)..numbers[3].min(height) {
                        for x in numbers[0].min(width)..numbers[2].min(width) {
                            gray[y * width + x] = numbers[4] as u8;
                        }
                    }
                }
                for pattern in image
                    .get("dotted_rows")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                {
                    let fields = pattern
                        .as_array()
                        .ok_or_else(|| Error::Message("dotted row is not an array".to_owned()))?
                        .iter()
                        .map(|field| field.as_u64().unwrap_or_default() as usize)
                        .collect::<Vec<_>>();
                    if fields.len() != 9 {
                        return Err(Error::Message(
                            "dotted row must have nine values".to_owned(),
                        ));
                    }
                    for base_y in (fields[0]..fields[1].min(height)).step_by(fields[2].max(1)) {
                        for y in base_y..(base_y + fields[3]).min(height) {
                            for x in (fields[4]..fields[5].min(width)).step_by(fields[6].max(1)) {
                                gray[y * width + x] = fields[7] as u8;
                                for extra in 1..fields[8] {
                                    if x + extra < fields[5].min(width) {
                                        gray[y * width + x + extra] = fields[7] as u8;
                                    }
                                }
                            }
                        }
                    }
                }
                scans.push(json!({
                    "id": image.get("id").cloned().unwrap_or(Value::Null),
                    "record": crate::separator::scan_gray_page(&gray, width, height),
                }));
            }
            let classifications = value
                .get("classifications")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .map(|case| {
                    let rules = serde_json::from_value::<Vec<crate::separator::RuleRecord>>(
                        case.get("rules").cloned().unwrap_or_else(|| json!([])),
                    )?;
                    let verticals =
                        serde_json::from_value::<Vec<crate::separator::VerticalRuleRecord>>(
                            case.get("vertical_rules")
                                .cloned()
                                .unwrap_or_else(|| json!([])),
                        )?;
                    let (separators, status) = crate::separator::classify_separator(
                        &rules,
                        &verticals,
                        case.get("min_y_ratio")
                            .and_then(Value::as_f64)
                            .unwrap_or(0.30),
                    );
                    Ok(json!({
                        "id": case.get("id").cloned().unwrap_or(Value::Null),
                        "separators": separators,
                        "status": status,
                    }))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "scans": scans,
                "classifications": classifications,
            }))
        }
        "pairing_support" => {
            let headings = value
                .get("headings")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no headings".to_owned()))?;
            let texts = value
                .get("texts")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no texts".to_owned()))?;
            let enumerators = value
                .get("enumerators")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no enumerators".to_owned()))?;
            let ladders = value
                .get("ladders")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no ladders".to_owned()))?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "headings": headings.iter().map(|item| {
                    let text = item.as_str().unwrap_or_default();
                    json!({"text": text, "plausible": crate::pairing_support::heading_text_plausible(text)})
                }).collect::<Vec<_>>(),
                "texts": texts.iter().map(|item| {
                    let text = item.as_str().unwrap_or_default();
                    json!({
                        "text": text,
                        "cue": crate::pairing_support::has_legal_citation_cue(text),
                        "continuation": crate::pairing_support::is_legal_citation_continuation(text),
                        "signal": crate::pairing_support::has_citation_signal(text),
                        "protected_spans": crate::pairing_support::protected_citation_spans(text),
                    })
                }).collect::<Vec<_>>(),
                "enumerators": enumerators.iter().map(|item| {
                    let value = item.get("value").and_then(Value::as_str).unwrap_or_default();
                    let punct = item.get("punct").and_then(Value::as_str).unwrap_or_default();
                    json!({
                        "value": value,
                        "punct": punct,
                        "interpretations": crate::pairing_support::enumerator_interpretations(value, punct),
                    })
                }).collect::<Vec<_>>(),
                "ladders": ladders.iter().map(|item| {
                    crate::pairing_support::parse_heading_ladder(item.as_array().map(Vec::as_slice).unwrap_or_default())
                }).collect::<Vec<_>>(),
            }))
        }
        "ocr_tsv" => {
            let number = |key: &str| {
                value
                    .get(key)
                    .and_then(Value::as_f64)
                    .ok_or_else(|| Error::Message(format!("contract input {key} is not a number")))
            };
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "lines": crate::ocr::tsv_lines(
                    string(value, "tsv")?,
                    number("x_scale")?,
                    number("y_scale")?,
                    number("page_width")?,
                    number("page_height")?,
                ),
            }))
        }
        "repair_identity" => Ok(json!({
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "identity": repair_identity()?,
        })),
        "repair_contract" => {
            let document = load_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            let targets = value
                .get("target_pages")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no target_pages".to_owned()))?
                .iter()
                .map(|page| {
                    page.as_u64()
                        .and_then(|page| usize::try_from(page).ok())
                        .ok_or_else(|| Error::Message("target page is not an integer".to_owned()))
                })
                .collect::<Result<Vec<_>>>()?;
            let expected = targets
                .iter()
                .map(|page| {
                    Ok((
                        *page,
                        document
                            .pages
                            .get(*page)
                            .ok_or_else(|| {
                                Error::Message("target page is out of range".to_owned())
                            })?
                            .lines
                            .iter()
                            .map(|line| line.id.clone())
                            .collect(),
                    ))
                })
                .collect::<Result<std::collections::BTreeMap<_, _>>>()?;
            let validation = value.get("response").map(|response| {
                let (valid, error) = validate_repair_response(response, &targets, &expected);
                json!({"valid": valid, "error": error})
            });
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "identity": repair_identity()?,
                "scopes": repair_scopes(&document),
                "context": repair_context(&document, &targets)?,
                "validation": validation,
            }))
        }
        "docx_intents" => {
            let id = string(value, "footnote_id")?;
            let text = string(value, "text")?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "intents": deterministic_docx_intents(id, text)?,
            }))
        }
        "docx_route" => {
            let footnotes = value
                .get("footnotes")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no footnotes".to_owned()))?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "assessment": assess_docx_route(footnotes)?,
            }))
        }
        "docx_validate" => {
            let records = value
                .get("records")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no records".to_owned()))?;
            let response = value
                .get("response")
                .ok_or_else(|| Error::Message("contract input has no response".to_owned()))?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "validated": validate_docx_response(response, records)?,
            }))
        }
        "docx_extract" => {
            let path = Path::new(string(value, "docx")?);
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "gold": extract_docx_gold(path)?,
            }))
        }
        "docx_batch" => {
            let cases = value
                .get("cases")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no cases".to_owned()))?;
            let mut results = Vec::with_capacity(cases.len());
            for case in cases {
                let path = Path::new(string(case, "docx")?);
                let gold = extract_docx_gold(path)?;
                let notes = docx_notes(&gold);
                let intents = notes
                    .iter()
                    .map(|note| {
                        Ok(json!({
                            "id": note["id"],
                            "intents": deterministic_docx_intents(
                                note["id"].as_str().expect("note id"),
                                note["text"].as_str().expect("note text"),
                            )?,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?;
                results.push(json!({
                    "docx": path,
                    "gold": gold,
                    "assessment": assess_docx_route(&notes)?,
                    "intents": intents,
                }));
            }
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "results": results,
            }))
        }
        "docx_plan_hybrid" => {
            let path = Path::new(string(value, "docx")?);
            let options = DocxPlanOptions {
                strategy: "hybrid".to_owned(),
                ..DocxPlanOptions::default()
            };
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "plan": stable_docx_plan(plan_docx_links(path, &options)?),
            }))
        }
        "docx_apply" => {
            let path = Path::new(string(value, "docx")?);
            let output = Path::new(string(value, "output")?);
            let links = value
                .get("links")
                .ok_or_else(|| Error::Message("contract input has no links".to_owned()))?;
            let options = DocxPlanOptions {
                strategy: "hybrid".to_owned(),
                ..DocxPlanOptions::default()
            };
            let plan = plan_docx_links(path, &options)?;
            let applied = apply_docx_links(path, &plan, links, output)?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "plan": stable_docx_plan(plan),
                "applied": applied,
                "gold": stable_docx_gold(extract_docx_gold(output)?),
                "targets": crate::docx::link_targets(output)?,
            }))
        }
        "citation_batch" => {
            let cases = value
                .get("cases")
                .and_then(Value::as_array)
                .ok_or_else(|| Error::Message("contract input has no cases".to_owned()))?;
            let mut results = Vec::with_capacity(cases.len());
            for case in cases {
                let text = string(case, "text")?;
                let mode = case
                    .get("mode")
                    .and_then(Value::as_str)
                    .unwrap_or("recall_first");
                let split = match mode {
                    "conservative" => split_citations(text)?,
                    "recall_first" => split_citations_recall_first(text)?,
                    _ => {
                        return Err(Error::Message(format!(
                            "unsupported citation split mode: {mode}"
                        )))
                    }
                };
                let fields = split
                    .parts
                    .iter()
                    .map(extract_citation_fields)
                    .collect::<Result<Vec<_>>>()?;
                results.push(json!({"mode": mode, "text": text, "split": split, "fields": fields}));
            }
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "results": results,
            }))
        }
        "citation_split" | "citation_split_recall_first" => {
            let text = string(value, "text")?;
            let split = if operation == "citation_split" {
                split_citations(text)?
            } else {
                split_citations_recall_first(text)?
            };
            let fields = split
                .parts
                .iter()
                .map(extract_citation_fields)
                .collect::<Result<Vec<_>>>()?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "split": split,
                "fields": fields,
            }))
        }
        "load_document" => {
            let document = load_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "document": document,
            }))
        }
        "artifact_bytes" => {
            let document = load_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            let compact = value
                .get("compact")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "compact": compact,
                "artifacts": artifact_hashes(&document, compact)?,
            }))
        }
        "adapters" => {
            let document = load_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "alr": to_alr_payload(&document),
                "toa": to_toa_text_units(&document)?,
            }))
        }
        "source_doc" => {
            let document = load_projection_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            let projected = source_doc(
                &document,
                value.get("id").and_then(Value::as_str),
                value.get("url").and_then(Value::as_str),
            );
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "result": projected,
            }))
        }
        "structure_lookup" => {
            let document = load_projection_artifacts(
                artifact
                    .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?,
            )?;
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "result": structure_lookup(&document, value)?,
            }))
        }
        "lookup" | "artifact_lookup" => {
            let artifact = artifact
                .ok_or_else(|| Error::Message("contract input has no artifact".to_owned()))?;
            let query = string(value, "query")?;
            let page = optional_u32(value, "page")?;
            let occurrence = optional_usize(value, "occurrence")?;
            let proposition_mode = value
                .get("proposition_mode")
                .and_then(Value::as_str)
                .unwrap_or("sentence");
            let result = if operation == "lookup" {
                let document = load_artifacts(artifact)?;
                lookup_footnote(&document, query, page, occurrence, proposition_mode)?
            } else {
                lookup_artifact_footnote(artifact, query, page, occurrence, proposition_mode)?
            };
            Ok(json!({
                "schema_version": "legalpdf.contract-result.v1",
                "operation": operation,
                "result": result,
            }))
        }
        other => Err(Error::Message(format!(
            "unsupported contract operation: {other}"
        ))),
    }
}
