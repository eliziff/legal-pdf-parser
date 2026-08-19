use crate::artifact::write_json;
use crate::codex::{invoke, stable_hash};
use crate::model::{Diagnostic, LegalDocument, Region, RepairRecord};
use crate::ocr::TesseractOcr;
use crate::structure::{rebuild_document, validate_document};
use crate::{Error, Result};
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::Path;

pub const REPAIR_PROMPT_VERSION: &str = "legalpdf.codex.structure.r1.v2";
pub const CONTEXT_RADIUS: usize = 1;
pub const MAX_ATTEMPTS: usize = 3;
pub const MAX_LIVE_CALLS: usize = 6;
pub const MAX_SCOPE_PAGES: usize = 2;
const REPAIRABLE: [&str; 5] = [
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
];
const REGION_TYPES: [&str; 6] = ["body", "heading", "footnote", "header", "footer", "unknown"];

fn response_schema() -> Value {
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["pages"],
        "properties": {
            "pages": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["page_index", "regions"],
                    "properties": {
                        "page_index": {"type": "integer", "minimum": 0},
                        "regions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": false,
                                "required": ["region_type", "line_ids"],
                                "properties": {
                                    "region_type": {
                                        "type": "string",
                                        "enum": REGION_TYPES
                                    },
                                    "line_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {"type": "string", "minLength": 1}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

pub fn repair_identity() -> Result<Value> {
    let mut repairable = REPAIRABLE.to_vec();
    repairable.sort_unstable();
    Ok(json!({
        "schema_version": "legalpdf.codex.repair-identity.v1",
        "prompt_version": REPAIR_PROMPT_VERSION,
        "response_schema_sha256": stable_hash(&response_schema())?,
        "context_radius": CONTEXT_RADIUS,
        "max_attempts": MAX_ATTEMPTS,
        "max_live_calls": MAX_LIVE_CALLS,
        "max_scope_pages": MAX_SCOPE_PAGES,
        "repairable_diagnostics": repairable,
        "repairable_diagnostics_sha256": stable_hash(&json!(repairable))?,
    }))
}

pub fn repair_context(document: &LegalDocument, target_pages: &[usize]) -> Result<Value> {
    if target_pages.is_empty() {
        return Err(Error::Message(
            "repair context requires a target page".to_owned(),
        ));
    }
    let targets = target_pages.iter().copied().collect::<HashSet<_>>();
    let start = target_pages
        .iter()
        .min()
        .copied()
        .unwrap()
        .saturating_sub(CONTEXT_RADIUS);
    let end = document
        .page_count
        .min(target_pages.iter().max().copied().unwrap() + CONTEXT_RADIUS + 1);
    let pages = (start..end)
        .map(|index| {
            let page = &document.pages[index];
            json!({
                "page_index": page.index,
                "width": page.width,
                "height": page.height,
                "target": targets.contains(&page.index),
                "lines": page.lines.iter().map(|line| json!({
                    "id": line.id,
                    "text": line.text,
                    "bbox": line.bbox,
                    "current_region_type": line.region_type,
                    "current_reading_order": line.reading_order,
                })).collect::<Vec<_>>()
            })
        })
        .collect::<Vec<_>>();
    let diagnostics = document
        .diagnostics
        .iter()
        .filter(|diagnostic| {
            diagnostic
                .page_index
                .is_some_and(|page| targets.contains(&page))
                && REPAIRABLE.contains(&diagnostic.code.as_str())
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema_version": "legalpdf.codex.input.v1",
        "target_pages": target_pages,
        "pages": pages,
        "diagnostics": diagnostics,
    }))
}

fn prompt(context: &Value, previous_error: &str) -> Result<String> {
    let retry = if previous_error.is_empty() {
        String::new()
    } else {
        format!(
            "\nThe previous response was rejected for this reason: {previous_error}\nCorrect that exact contract failure."
        )
    };
    Ok(
        "You are repairing structure in a legal PDF. The input contains immutable \
line IDs and immutable text for one or more adjacent target pages with r=1 \
context. Return one output page for EVERY TARGET PAGE and no context pages. \
Region order and line order inside each region define reading order. Include \
every target-page line ID exactly once. You cannot edit glyph text \
because the output contract contains IDs only. Classify page furniture, \
headings, body text, and footnote/endnote material conservatively. \
Do not abstain and do not add commentary."
            .to_owned()
            + &retry
            + "\n\nINPUT:\n"
            + &crate::artifact::python_json(context)?,
    )
}

pub fn validate_repair_response(
    response: &Value,
    target_pages: &[usize],
    expected_line_ids: &BTreeMap<usize, Vec<String>>,
) -> (bool, String) {
    let Some(object) = response.as_object() else {
        return (false, "response is not an object".to_owned());
    };
    if object.len() != 1 || !object.contains_key("pages") {
        return (
            false,
            "response has missing or additional top-level properties".to_owned(),
        );
    }
    let Some(pages) = object["pages"].as_array().filter(|pages| !pages.is_empty()) else {
        return (false, "pages must be a non-empty list".to_owned());
    };
    let actual_pages = pages
        .iter()
        .filter_map(Value::as_object)
        .filter_map(|page| page.get("page_index"))
        .filter_map(Value::as_u64)
        .filter_map(|page| usize::try_from(page).ok())
        .collect::<Vec<_>>();
    if actual_pages.iter().collect::<HashSet<_>>().len() != actual_pages.len()
        || actual_pages.iter().copied().collect::<BTreeSet<_>>()
            != target_pages.iter().copied().collect()
    {
        return (
            false,
            "output pages do not exactly match the requested targets".to_owned(),
        );
    }
    for page in pages {
        let Some(page) = page.as_object() else {
            return (
                false,
                "an output page has missing or additional properties".to_owned(),
            );
        };
        if page.len() != 2 || !page.contains_key("page_index") || !page.contains_key("regions") {
            return (
                false,
                "an output page has missing or additional properties".to_owned(),
            );
        }
        let Some(page_index) = page["page_index"]
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
        else {
            return (
                false,
                "a page_index is not a non-negative integer".to_owned(),
            );
        };
        let Some(regions) = page["regions"]
            .as_array()
            .filter(|regions| !regions.is_empty())
        else {
            return (
                false,
                format!("page {page_index} regions must be a non-empty list"),
            );
        };
        let mut actual = Vec::new();
        for region in regions {
            let Some(region) = region.as_object() else {
                return (
                    false,
                    "a region has missing or additional properties".to_owned(),
                );
            };
            if region.len() != 2
                || !region.contains_key("region_type")
                || !region.contains_key("line_ids")
            {
                return (
                    false,
                    "a region has missing or additional properties".to_owned(),
                );
            }
            if !region["region_type"]
                .as_str()
                .is_some_and(|value| REGION_TYPES.contains(&value))
            {
                return (false, "a region_type is unsupported".to_owned());
            }
            let Some(line_ids) = region["line_ids"].as_array().filter(|ids| !ids.is_empty()) else {
                return (false, "a region has no line IDs".to_owned());
            };
            if !line_ids
                .iter()
                .all(|id| id.as_str().is_some_and(|id| !id.is_empty()))
            {
                return (false, "a line ID is not a non-empty string".to_owned());
            }
            actual.extend(line_ids.iter().map(|id| id.as_str().unwrap().to_owned()));
        }
        if actual.iter().collect::<HashSet<_>>().len() != actual.len() {
            return (
                false,
                format!("page {page_index} contains a duplicate line ID"),
            );
        }
        let expected = &expected_line_ids[&page_index];
        if actual.iter().collect::<BTreeSet<_>>() != expected.iter().collect()
            || actual.len() != expected.len()
        {
            let missing = expected
                .iter()
                .filter(|id| !actual.contains(id))
                .cloned()
                .collect::<Vec<_>>();
            let unknown = actual
                .iter()
                .filter(|id| !expected.contains(id))
                .cloned()
                .collect::<Vec<_>>();
            return (
                false,
                format!(
                    "page {page_index} line coverage mismatch; missing={missing:?}, unknown={unknown:?}"
                ),
            );
        }
    }
    (true, String::new())
}

pub fn repair_scopes(document: &LegalDocument) -> Vec<Vec<usize>> {
    let mut codes_by_page = BTreeMap::<usize, HashSet<String>>::new();
    for diagnostic in &document.diagnostics {
        if diagnostic
            .page_index
            .is_some_and(|page| page < document.page_count)
            && REPAIRABLE.contains(&diagnostic.code.as_str())
        {
            codes_by_page
                .entry(diagnostic.page_index.unwrap())
                .or_default()
                .insert(diagnostic.code.clone());
        }
    }
    let mut scopes = Vec::<Vec<usize>>::new();
    for (page_index, codes) in &codes_by_page {
        if scopes.last().is_some_and(|scope| {
            scope.len() < MAX_SCOPE_PAGES
                && *page_index == scope.last().copied().unwrap() + 1
                && !codes.is_disjoint(&codes_by_page[scope.last().unwrap()])
        }) {
            scopes.last_mut().unwrap().push(*page_index);
        } else {
            scopes.push(vec![*page_index]);
        }
    }
    scopes
}

fn replay_page(document: &mut LegalDocument, page_index: usize, regions: &[Value]) -> Result<()> {
    let page = document
        .pages
        .get_mut(page_index)
        .ok_or_else(|| Error::Message(format!("repair page is out of range: {page_index}")))?;
    let mut line_by_id = page
        .lines
        .drain(..)
        .map(|line| (line.id.clone(), line))
        .collect::<HashMap<_, _>>();
    let mut ordered = Vec::new();
    let mut result_regions = Vec::new();
    for (region_index, item) in regions.iter().enumerate() {
        let region_type = item["region_type"].as_str().unwrap().to_owned();
        let region_id = format!("p{:04}-r{:04}", page.number, region_index + 1);
        let mut lines = Vec::new();
        for line_id in item["line_ids"].as_array().unwrap() {
            let line_id = line_id.as_str().unwrap();
            let mut line = line_by_id
                .remove(line_id)
                .ok_or_else(|| Error::Message(format!("unknown repair line: {line_id}")))?;
            line.region_id = region_id.clone();
            line.region_type = region_type.clone();
            line.reading_order = ordered.len() + 1;
            lines.push(line.clone());
            ordered.push(line);
        }
        let mut bbox = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for line in &lines {
            bbox[0] = bbox[0].min(line.bbox[0]);
            bbox[1] = bbox[1].min(line.bbox[1]);
            bbox[2] = bbox[2].max(line.bbox[2]);
            bbox[3] = bbox[3].max(line.bbox[3]);
        }
        result_regions.push(Region {
            id: region_id,
            page_index,
            kind: region_type,
            line_ids: lines.iter().map(|line| line.id.clone()).collect(),
            bbox,
            reading_order: lines[0].reading_order,
        });
    }
    page.lines = ordered;
    page.regions = result_regions;
    Ok(())
}

pub fn replay_repair(document: &mut LegalDocument, response: &Value) -> Result<()> {
    for page in response["pages"].as_array().unwrap() {
        replay_page(
            document,
            page["page_index"].as_u64().unwrap() as usize,
            page["regions"].as_array().unwrap(),
        )?;
    }
    Ok(())
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).map_err(|source| Error::io(path, source))?;
    serde_json::from_slice(&bytes).map_err(Into::into)
}

fn round_four(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

pub fn improve_document(
    document: &LegalDocument,
    pdf_path: &Path,
    model: &str,
    effort: &str,
    cache_dir: &Path,
    timeout_seconds: u64,
) -> Result<LegalDocument> {
    if model.trim().is_empty() || effort.trim().is_empty() {
        return Err(Error::Message(
            "model and effort must be non-empty".to_owned(),
        ));
    }
    if !pdf_path.is_file() {
        return Err(Error::Message(format!(
            "PDF does not exist: {}",
            pdf_path.display()
        )));
    }
    let mut document = document.clone();
    let scopes = repair_scopes(&document);
    let targets = scopes.iter().flatten().copied().collect::<Vec<_>>();
    let identity = repair_identity()?;
    fs::create_dir_all(cache_dir).map_err(|source| Error::io(cache_dir, source))?;
    let schema_path = cache_dir.join(format!(
        "{REPAIR_PROMPT_VERSION}.{}.schema.json",
        &identity["response_schema_sha256"].as_str().unwrap()[..16]
    ));
    if !schema_path.is_file() || read_json(&schema_path).ok().as_ref() != Some(&response_schema()) {
        write_json(&schema_path, &response_schema())?;
    }
    let mut total_calls = 0;
    let mut skipped_pages = Vec::new();
    for target_pages in scopes {
        let context = repair_context(&document, &target_pages)?;
        let input_hash = stable_hash(&context)?;
        let cache_key = stable_hash(&json!({
            "source_sha256": document.source_sha256,
            "context_hash": input_hash,
            "prompt_version": REPAIR_PROMPT_VERSION,
            "response_schema_sha256": identity["response_schema_sha256"],
            "repairable_diagnostics_sha256": identity["repairable_diagnostics_sha256"],
            "max_live_calls": MAX_LIVE_CALLS,
            "max_scope_pages": MAX_SCOPE_PAGES,
            "model": model,
            "effort": effort,
        }))?;
        let cache_contract = json!({
            "schema_version": "legalpdf.codex.cache.v1",
            "cache_key": cache_key,
            "model": model,
            "effort": effort,
            "prompt_version": REPAIR_PROMPT_VERSION,
            "response_schema_sha256": identity["response_schema_sha256"],
            "repairable_diagnostics_sha256": identity["repairable_diagnostics_sha256"],
            "repairable_diagnostics": identity["repairable_diagnostics"],
            "context_radius": CONTEXT_RADIUS,
            "max_attempts": MAX_ATTEMPTS,
            "max_live_calls": MAX_LIVE_CALLS,
            "max_scope_pages": MAX_SCOPE_PAGES,
        });
        let entry = cache_dir.join(&cache_key);
        let response_path = entry.join("response.json");
        let metadata_path = entry.join("metadata.json");
        let expected = target_pages
            .iter()
            .map(|page| {
                (
                    *page,
                    document.pages[*page]
                        .lines
                        .iter()
                        .map(|line| line.id.clone())
                        .collect(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut response = None;
        let mut usage = BTreeMap::<String, i64>::new();
        let mut elapsed = 0.0;
        let mut attempts = 0;
        let mut error = String::new();
        if response_path.is_file() || metadata_path.is_file() {
            let cached = (|| {
                if !response_path.is_file() || !metadata_path.is_file() {
                    return Err(Error::Message("cache publication is incomplete".to_owned()));
                }
                let candidate = read_json(&response_path)?;
                let metadata = read_json(&metadata_path)?;
                for (key, value) in cache_contract.as_object().unwrap() {
                    if metadata.get(key) != Some(value) {
                        return Err(Error::Message(
                            "cache metadata contract mismatch".to_owned(),
                        ));
                    }
                }
                if metadata.get("response_sha256").and_then(Value::as_str)
                    != Some(&stable_hash(&candidate)?)
                {
                    return Err(Error::Message("cached response hash mismatch".to_owned()));
                }
                let (valid, message) =
                    validate_repair_response(&candidate, &target_pages, &expected);
                if !valid {
                    return Err(Error::Message(message));
                }
                let raw_usage = metadata["token_usage"]
                    .as_object()
                    .ok_or_else(|| Error::Message("cache token usage is invalid".to_owned()))?;
                let parsed_usage = raw_usage
                    .iter()
                    .map(|(key, value)| {
                        value
                            .as_i64()
                            .filter(|value| *value >= 0)
                            .map(|value| (key.clone(), value))
                    })
                    .collect::<Option<BTreeMap<_, _>>>()
                    .ok_or_else(|| Error::Message("cache token usage is invalid".to_owned()))?;
                let cached_attempts = metadata["attempts"].as_u64().unwrap_or_default() as usize;
                let cached_elapsed = metadata["elapsed_seconds"].as_f64().unwrap_or(f64::NAN);
                if !(1..=MAX_ATTEMPTS).contains(&cached_attempts)
                    || !cached_elapsed.is_finite()
                    || cached_elapsed < 0.0
                {
                    return Err(Error::Message(
                        "cache attempt metadata is invalid".to_owned(),
                    ));
                }
                Ok((candidate, parsed_usage, cached_elapsed, cached_attempts))
            })();
            match cached {
                Ok((candidate, cached_usage, cached_elapsed, cached_attempts)) => {
                    response = Some(candidate);
                    usage = cached_usage;
                    elapsed = cached_elapsed;
                    attempts = cached_attempts;
                }
                Err(_) => {
                    let _ = fs::remove_dir_all(&entry);
                }
            }
        }
        if response.is_none() && total_calls >= MAX_LIVE_CALLS {
            skipped_pages.extend(target_pages.iter().copied());
            error = "document live-call budget exhausted".to_owned();
            document.repairs.push(RepairRecord {
                page_index: target_pages[0],
                status: "skipped".to_owned(),
                model: model.to_owned(),
                effort: effort.to_owned(),
                prompt_version: REPAIR_PROMPT_VERSION.to_owned(),
                cache_key,
                attempts: 0,
                elapsed_seconds: 0.0,
                input_line_hash: input_hash,
                output_hash: String::new(),
                token_usage: Map::new(),
                error: error.clone(),
                scope_pages: target_pages.clone(),
            });
            let mut diagnostic = Diagnostic::warning(
                "CODEX_REPAIR_BUDGET_EXHAUSTED",
                "Codex structural repair skipped because the document live-call budget was exhausted.",
                Some(target_pages[0]),
            );
            diagnostic
                .details
                .insert("scope_pages".to_owned(), json!(target_pages));
            diagnostic
                .details
                .insert("live_calls".to_owned(), json!(total_calls));
            diagnostic
                .details
                .insert("max_live_calls".to_owned(), json!(MAX_LIVE_CALLS));
            document.diagnostics.push(diagnostic);
            continue;
        }
        if response.is_none() {
            fs::create_dir_all(&entry).map_err(|source| Error::io(&entry, source))?;
            let image_pages = (target_pages[0].saturating_sub(CONTEXT_RADIUS)
                ..document
                    .page_count
                    .min(target_pages[target_pages.len() - 1] + CONTEXT_RADIUS + 1))
                .collect::<Vec<_>>();
            let images = TesseractOcr::render_pages(
                pdf_path,
                &image_pages,
                &cache_dir.join("renders").join(&document.source_sha256),
            )?;
            let allowed_attempts = MAX_ATTEMPTS.min(MAX_LIVE_CALLS - total_calls);
            for attempt in 1..=allowed_attempts {
                attempts = attempt;
                total_calls += 1;
                match invoke(
                    &prompt(&context, &error)?,
                    &schema_path,
                    &images,
                    model,
                    effort,
                    &entry,
                    timeout_seconds,
                ) {
                    Ok((candidate, attempt_usage, attempt_elapsed)) => {
                        elapsed += attempt_elapsed;
                        usage = attempt_usage;
                        let (valid, message) =
                            validate_repair_response(&candidate, &target_pages, &expected);
                        if valid {
                            response = Some(candidate);
                            error.clear();
                            break;
                        }
                        error = message;
                    }
                    Err(failure) => error = failure.to_string(),
                }
            }
            if let Some(response) = &response {
                write_json(&response_path, response)?;
                let mut metadata = cache_contract.as_object().unwrap().clone();
                metadata.insert(
                    "response_sha256".to_owned(),
                    Value::String(stable_hash(response)?),
                );
                metadata.insert("attempts".to_owned(), json!(attempts));
                metadata.insert("elapsed_seconds".to_owned(), json!(elapsed));
                metadata.insert("token_usage".to_owned(), json!(usage));
                write_json(&metadata_path, &Value::Object(metadata))?;
            }
        }
        let Some(response) = response else {
            document.repairs.push(RepairRecord {
                page_index: target_pages[0],
                status: "failed".to_owned(),
                model: model.to_owned(),
                effort: effort.to_owned(),
                prompt_version: REPAIR_PROMPT_VERSION.to_owned(),
                cache_key,
                attempts,
                elapsed_seconds: round_four(elapsed),
                input_line_hash: input_hash,
                output_hash: String::new(),
                token_usage: usage
                    .into_iter()
                    .map(|(key, value)| (key, json!(value)))
                    .collect(),
                error: error.clone(),
                scope_pages: target_pages.clone(),
            });
            let mut diagnostic = Diagnostic::warning(
                "CODEX_REPAIR_FAILED",
                format!("Codex structural repair failed after {attempts} attempts: {error}"),
                Some(target_pages[0]),
            );
            diagnostic
                .details
                .insert("scope_pages".to_owned(), json!(target_pages));
            document.diagnostics.push(diagnostic);
            continue;
        };
        replay_repair(&mut document, &response)?;
        let output_hash = stable_hash(&response)?;
        document.repairs.push(RepairRecord {
            page_index: target_pages[0],
            status: "applied".to_owned(),
            model: model.to_owned(),
            effort: effort.to_owned(),
            prompt_version: REPAIR_PROMPT_VERSION.to_owned(),
            cache_key: cache_key.clone(),
            attempts,
            elapsed_seconds: round_four(elapsed),
            input_line_hash: input_hash,
            output_hash: output_hash.clone(),
            token_usage: usage
                .into_iter()
                .map(|(key, value)| (key, json!(value)))
                .collect(),
            error: String::new(),
            scope_pages: target_pages.clone(),
        });
        for diagnostic in &mut document.diagnostics {
            if diagnostic
                .page_index
                .is_some_and(|page| target_pages.contains(&page))
                && REPAIRABLE.contains(&diagnostic.code.as_str())
            {
                diagnostic.severity = "info".to_owned();
                diagnostic
                    .details
                    .insert("codex_repair_applied".to_owned(), Value::Bool(true));
                diagnostic.details.insert(
                    "repair_output_hash".to_owned(),
                    Value::String(output_hash.clone()),
                );
            }
        }
        for &target_page in &target_pages {
            let mut diagnostic = Diagnostic::info(
                "CODEX_REPAIR_APPLIED",
                "Validated Codex structural repair applied.",
                Some(target_page),
            );
            diagnostic
                .details
                .insert("model".to_owned(), Value::String(model.to_owned()));
            diagnostic
                .details
                .insert("effort".to_owned(), Value::String(effort.to_owned()));
            diagnostic
                .details
                .insert("cache_key".to_owned(), Value::String(cache_key.clone()));
            diagnostic
                .details
                .insert("scope_pages".to_owned(), json!(target_pages));
            document.diagnostics.push(diagnostic);
        }
    }
    if document
        .repairs
        .iter()
        .any(|repair| repair.status == "applied")
    {
        rebuild_document(&mut document)?;
    } else {
        validate_document(&document)?;
    }
    document.provenance.insert(
        "codex".to_owned(),
        json!({
            "model": model,
            "effort": effort,
            "prompt_version": REPAIR_PROMPT_VERSION,
            "response_schema_sha256": identity["response_schema_sha256"],
            "repairable_diagnostics_sha256": identity["repairable_diagnostics_sha256"],
            "repairable_diagnostics": identity["repairable_diagnostics"],
            "context_radius": CONTEXT_RADIUS,
            "max_attempts": MAX_ATTEMPTS,
            "max_live_calls": MAX_LIVE_CALLS,
            "max_scope_pages": MAX_SCOPE_PAGES,
            "target_pages": targets,
            "skipped_pages": skipped_pages,
            "live_calls": total_calls,
        }),
    );
    Ok(document)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Line, Page};

    fn line(id: &str, order: usize) -> crate::Line {
        Line {
            id: id.to_owned(),
            page_index: 0,
            page_number: 1,
            source_index: order,
            reading_order: order,
            block_index: 0,
            text: id.to_owned(),
            bbox: [0.0, order as f64, 1.0, order as f64 + 1.0],
            spans: vec![],
            words: vec![],
            detached_references: vec![],
            exclude_from_body: false,
            suppress_footnote_label: false,
            note_region_mode: String::new(),
            region_id: String::new(),
            region_type: "unknown".to_owned(),
            source: "native".to_owned(),
        }
    }

    #[test]
    fn validation_requires_exact_page_and_line_coverage() {
        let expected = BTreeMap::from([(0, vec!["a".to_owned(), "b".to_owned()])]);
        let response = json!({"pages": [{"page_index": 0, "regions": [
            {"region_type": "body", "line_ids": ["b", "a"]}
        ]}]});
        assert_eq!(
            validate_repair_response(&response, &[0], &expected),
            (true, String::new())
        );
        let duplicate = json!({"pages": [{"page_index": 0, "regions": [
            {"region_type": "body", "line_ids": ["a", "a"]}
        ]}]});
        assert_eq!(
            validate_repair_response(&duplicate, &[0], &expected).1,
            "page 0 contains a duplicate line ID"
        );
    }

    #[test]
    fn replay_preserves_text_and_applies_order() {
        let mut document = LegalDocument {
            document_id: "d".to_owned(),
            source_name: "x.pdf".to_owned(),
            source_sha256: "0".repeat(64),
            page_count: 1,
            status: "degraded".to_owned(),
            pages: vec![Page {
                id: "p0001".to_owned(),
                index: 0,
                number: 1,
                width: 10.0,
                height: 10.0,
                lines: vec![line("a", 1), line("b", 2)],
                regions: vec![],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![],
            sections: vec![],
            footnotes: vec![],
            tables: vec![],
            images: vec![],
            diagnostics: vec![],
            repairs: vec![],
            metadata: Map::new(),
            provenance: Map::new(),
            schema_version: crate::SCHEMA_VERSION.to_owned(),
            parser_version: crate::PARSER_VERSION.to_owned(),
        };
        replay_repair(&mut document, &json!({"pages": [{"page_index": 0, "regions": [
            {"region_type": "heading", "line_ids": ["b"]}, {"region_type": "body", "line_ids": ["a"]}
        ]}]})).unwrap();
        assert_eq!(
            document.pages[0]
                .lines
                .iter()
                .map(|line| line.id.as_str())
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
        assert_eq!(
            document.pages[0]
                .lines
                .iter()
                .map(|line| line.text.as_str())
                .collect::<Vec<_>>(),
            ["b", "a"]
        );
    }
}
