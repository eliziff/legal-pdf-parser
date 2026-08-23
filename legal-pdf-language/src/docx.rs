use crate::deterministic_citations::{extract_fields, split_footnote};
use crate::grammar_tables::compile_table_entry;
use crate::{Error, Result};
use fancy_regex::Regex as FancyRegex;
use legal_pdf_core::{atomic_write_with, python_json, write_json};
use quick_xml::events::{BytesDecl, BytesEnd, BytesStart, BytesText, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;
use quick_xml::Writer;
use regex::Regex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::env;
use std::fs::{self, File};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, LazyLock, OnceLock};
use std::time::Instant;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

pub const DOCX_PROMPT_VERSION: &str = "legalpdf.docx.citation-intents.v1";
pub const DEFAULT_DOCX_MODEL: &str = "gpt-5.6-sol";
pub const DEFAULT_DOCX_EFFORT: &str = "none";
pub const MAX_FOOTNOTES: usize = 400;
pub const MAX_DOCX_SUPRA_BYTES: usize = 25 * 1024 * 1024;
const MAX_BATCH_FOOTNOTES: usize = 32;
const MAX_BATCH_CHARS: usize = 45_000;
const MAX_BATCHES: usize = 13;
const W_NS: &str = "http://schemas.openxmlformats.org/wordprocessingml/2006/main";
const R_NS: &str = "http://schemas.openxmlformats.org/officeDocument/2006/relationships";
const PKG_REL_NS: &str = "http://schemas.openxmlformats.org/package/2006/relationships";
const XML_NS: &str = "http://www.w3.org/XML/1998/namespace";
const HYPERLINK_REL: &str =
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink";
const KINDS: [&str; 11] = [
    "statute",
    "gazette",
    "case",
    "unreported",
    "parliamentary_paper",
    "non_parliamentary",
    "journal",
    "book",
    "essay_collection",
    "report",
    "other",
];

#[derive(Debug, Clone)]
pub struct DocxPlanOptions {
    pub strategy: String,
    pub model: String,
    pub effort: String,
    pub cache_dir: Option<PathBuf>,
    pub timeout_seconds: u64,
}

impl Default for DocxPlanOptions {
    fn default() -> Self {
        Self {
            strategy: "auto".to_owned(),
            model: DEFAULT_DOCX_MODEL.to_owned(),
            effort: DEFAULT_DOCX_EFFORT.to_owned(),
            cache_dir: None,
            timeout_seconds: 600,
        }
    }
}

fn grammar(
    slot: &'static OnceLock<std::result::Result<Arc<FancyRegex>, String>>,
    id: &'static str,
) -> Result<&'static FancyRegex> {
    slot.get_or_init(|| {
        compile_table_entry(id)
            .map(Arc::new)
            .map_err(|error| error.to_string())
    })
    .as_ref()
    .map(Arc::as_ref)
    .map_err(|message| Error::Message(message.clone()))
}

fn reference_re() -> Result<&'static FancyRegex> {
    static SLOT: OnceLock<std::result::Result<Arc<FancyRegex>, String>> = OnceLock::new();
    grammar(&SLOT, "ref.token")
}

fn supra_note_re() -> Result<&'static FancyRegex> {
    static SLOT: OnceLock<std::result::Result<Arc<FancyRegex>, String>> = OnceLock::new();
    grammar(&SLOT, "ref.supra-note.linking")
}

fn url_re() -> Result<&'static FancyRegex> {
    static SLOT: OnceLock<std::result::Result<Arc<FancyRegex>, String>> = OnceLock::new();
    grammar(&SLOT, "cite.url.prefix")
}

fn fancy_match(pattern: &FancyRegex, text: &str) -> Result<bool> {
    pattern
        .is_match(text)
        .map_err(|error| Error::Message(format!("regex search failed: {error}")))
}

fn sha256_path(path: &Path) -> Result<String> {
    let mut file = File::open(path).map_err(|source| Error::io(path, source))?;
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|source| Error::io(path, source))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn py_string(value: Option<&Value>) -> String {
    let Some(value) = value.filter(|value| py_truthy(value)) else {
        return String::new();
    };
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        Value::Array(_) | Value::Object(_) => python_json(value).unwrap_or_default(),
    }
}

fn required<'a>(value: &'a Value, key: &str) -> Result<&'a Value> {
    value
        .as_object()
        .and_then(|object| object.get(key))
        .ok_or_else(|| Error::Message(format!("missing required property: {key}")))
}

fn object_with_exact_keys(value: &Value, keys: &[&str]) -> bool {
    value.as_object().is_some_and(|object| {
        object.len() == keys.len() && keys.iter().all(|key| object.contains_key(*key))
    })
}

#[allow(clippy::too_many_arguments)]
fn make_intent(
    part_id: String,
    verbatim: String,
    corrected: String,
    kind: String,
    pinpoint_fragments: &[Value],
    page_pinpoints: &[Value],
    short_form: String,
    bare_citation: String,
    citation_with_style: String,
    support_quote: String,
    route: &str,
) -> Value {
    let fragments = pinpoint_fragments
        .iter()
        .map(|value| py_string(Some(value)).trim().to_owned())
        .filter(|value| !value.is_empty())
        .collect::<Vec<_>>();
    let pages = page_pinpoints
        .iter()
        .filter_map(|value| value.as_i64())
        .filter(|value| *value > 0)
        .collect::<Vec<_>>();
    let (locator_kind, locator) = if let Some(first) = fragments.first() {
        let lower = first.to_lowercase();
        if lower.starts_with("par") {
            ("paragraph", first.get(3..).unwrap_or_default().to_owned())
        } else if lower.starts_with("sec") {
            ("section", first.get(3..).unwrap_or_default().to_owned())
        } else if let Some(page) = pages.first() {
            ("page", page.to_string())
        } else {
            ("none", String::new())
        }
    } else if let Some(page) = pages.first() {
        ("page", page.to_string())
    } else {
        ("none", String::new())
    };
    let corrected = if corrected.is_empty() {
        verbatim.clone()
    } else {
        corrected
    };
    let kind = if KINDS.contains(&kind.as_str()) {
        kind
    } else {
        "other".to_owned()
    };
    json!({
        "part_id": part_id,
        "verbatim": verbatim,
        "corrected": corrected,
        "kind": kind,
        "pinpoint_fragments": fragments,
        "page_pinpoints": pages,
        "short_form": short_form,
        "bare_citation": if bare_citation.is_empty() { verbatim.clone() } else { bare_citation },
        "citation_with_style": if citation_with_style.is_empty() { verbatim } else { citation_with_style },
        "support_quote": support_quote,
        "locator_kind": locator_kind,
        "locator": locator,
        "route": route,
    })
}

pub fn deterministic_docx_intents(footnote_id: &str, text: &str) -> Result<Option<Vec<Value>>> {
    let split = split_footnote(text)?;
    if split.status != "deterministic_complete" || split.parts.is_empty() {
        return Ok(None);
    }
    let fields = split
        .parts
        .iter()
        .map(extract_fields)
        .collect::<Result<Vec<_>>>()?;
    if fields.iter().any(|field| field.status != "complete") {
        return Ok(None);
    }
    for part in &split.parts {
        if fancy_match(reference_re()?, &part.text)? {
            return Ok(None);
        }
    }
    if fields.iter().any(|field| {
        matches!(field.kind.as_str(), "case" | "unreported" | "statute")
            && field.bare_citation.trim().is_empty()
    }) {
        return Ok(None);
    }
    Ok(Some(
        split
            .parts
            .iter()
            .zip(fields)
            .enumerate()
            .map(|(index, (part, field))| {
                make_intent(
                    format!("{footnote_id}:{}", index + 1),
                    part.text.clone(),
                    field.corrected,
                    field.kind,
                    &field
                        .pinpoint_fragments
                        .into_iter()
                        .map(Value::String)
                        .collect::<Vec<_>>(),
                    &field
                        .page_pinpoints
                        .into_iter()
                        .map(Value::from)
                        .collect::<Vec<_>>(),
                    field.short_form,
                    field.bare_citation,
                    field.citation_with_style,
                    String::new(),
                    "deterministic",
                )
            })
            .collect(),
    ))
}

fn normalize_with_map(value: &str) -> (String, Vec<(usize, usize)>) {
    let mut out = String::new();
    let mut positions = Vec::new();
    let mut previous_space = true;
    for (start, character) in value.char_indices() {
        let end = start + character.len_utf8();
        if character.is_whitespace() {
            if previous_space {
                continue;
            }
            out.push(' ');
            positions.push((start, end));
            previous_space = true;
        } else {
            let translated = match character {
                '‘' | '’' => '\'',
                '“' | '”' => '"',
                '–' | '—' => '-',
                other => other,
            };
            for lowered in translated.to_lowercase() {
                out.push(lowered);
                positions.push((start, end));
            }
            previous_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
        positions.pop();
    }
    (out, positions)
}

fn char_index(value: &str, byte_index: usize) -> usize {
    value[..byte_index].chars().count()
}

fn core_without_separators(value: &str) -> String {
    value
        .chars()
        .filter(|character| !character.is_whitespace() && *character != ';')
        .collect()
}

fn snap_parts(text: &str, parts: &[Value]) -> Result<Vec<Value>> {
    let (normalized, positions) = normalize_with_map(text);
    let mut spans = Vec::<(usize, usize)>::new();
    let mut cursor = 0;
    for part in parts {
        let (wanted, _) = normalize_with_map(&py_string(
            part.as_object().and_then(|object| object.get("verbatim")),
        ));
        let wanted = wanted.trim();
        if wanted.is_empty() {
            return Err(Error::Message(
                "worker returned an empty citation part".to_owned(),
            ));
        }
        let start = normalized[cursor..]
            .find(wanted)
            .map(|offset| cursor + offset)
            .or_else(|| normalized.find(wanted))
            .ok_or_else(|| {
                Error::Message("worker part is not an exact footnote substring".to_owned())
            })?;
        spans.push((start, start + wanted.len()));
        cursor = start + wanted.len();
    }
    for index in 0..spans.len().saturating_sub(1) {
        if spans[index].1 > spans[index + 1].0 {
            spans[index].1 = spans[index + 1].0;
        }
        if spans[index].1 <= spans[index].0 {
            return Err(Error::Message(
                "worker returned overlapping citation parts".to_owned(),
            ));
        }
    }
    if let Some(last) = spans.last_mut() {
        let trailing = &normalized[last.1..];
        if !trailing.is_empty()
            && trailing
                .chars()
                .all(|character| ".,:!?'\"])}’”".contains(character))
        {
            last.1 = normalized.len();
        }
    }
    let mut snapped = Vec::with_capacity(parts.len());
    for (part, (start, end)) in parts.iter().zip(spans) {
        if end <= start {
            return Err(Error::Message(
                "worker returned an empty citation span".to_owned(),
            ));
        }
        let first = char_index(&normalized, start);
        let last = char_index(&normalized, end) - 1;
        let source_start = positions[first].0;
        let source_end = positions[last].1;
        let mut item = part
            .as_object()
            .cloned()
            .ok_or_else(|| Error::Message("worker part is not an object".to_owned()))?;
        item.insert(
            "verbatim".to_owned(),
            Value::String(text[source_start..source_end].trim().to_owned()),
        );
        snapped.push(Value::Object(item));
    }
    let actual = snapped
        .iter()
        .map(|part| py_string(part.get("verbatim")))
        .collect::<String>();
    if core_without_separators(text) != core_without_separators(&actual) {
        return Err(Error::Message(
            "worker split lost, gained, or reordered footnote characters".to_owned(),
        ));
    }
    Ok(snapped)
}

pub fn validate_docx_response(response: &Value, records: &[Value]) -> Result<Value> {
    if fancy_match(url_re()?, &python_json(response)?)? {
        return Err(Error::Message("worker output contains a URL".to_owned()));
    }
    if !object_with_exact_keys(response, &["results"]) {
        return Err(Error::Message(
            "worker response has the wrong top-level shape".to_owned(),
        ));
    }
    let results = response["results"]
        .as_array()
        .ok_or_else(|| Error::Message("worker results is not an array".to_owned()))?;
    let record_by_id = records
        .iter()
        .map(|record| Ok((py_string(Some(required(record, "id")?)), record)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let result_ids = results
        .iter()
        .filter_map(Value::as_object)
        .map(|item| py_string(item.get("id")))
        .collect::<Vec<_>>();
    let unique = result_ids.iter().collect::<HashSet<_>>();
    if result_ids.len() != unique.len()
        || result_ids.iter().cloned().collect::<BTreeSet<_>>()
            != record_by_id.keys().cloned().collect()
    {
        return Err(Error::Message(
            "worker result ids do not exactly match the request".to_owned(),
        ));
    }
    let expected_part_keys = [
        "verbatim",
        "corrected",
        "kind",
        "pinpoint_fragments",
        "page_pinpoints",
        "short_form",
        "bare_citation",
        "citation_with_style",
        "support_quote",
    ];
    let mut validated = Map::new();
    for raw in results {
        if !object_with_exact_keys(raw, &["id", "parts"]) {
            return Err(Error::Message(
                "worker result has an unsupported property".to_owned(),
            ));
        }
        let id = py_string(raw.get("id"));
        let record = record_by_id
            .get(&id)
            .ok_or_else(|| Error::Message("worker result id is unknown".to_owned()))?;
        let parts = raw["parts"]
            .as_array()
            .filter(|parts| (1..=20).contains(&parts.len()))
            .ok_or_else(|| Error::Message("worker returned an invalid part count".to_owned()))?;
        let snapped = snap_parts(&py_string(record.get("text")), parts)?;
        let allowed_quote_text = format!(
            "{} {}",
            py_string(record.get("text")),
            py_string(record.get("proposition"))
        );
        for part in &snapped {
            if !object_with_exact_keys(part, &expected_part_keys) {
                return Err(Error::Message(
                    "worker part has an unsupported property".to_owned(),
                ));
            }
            let kind = part.get("kind").and_then(Value::as_str).unwrap_or_default();
            if !KINDS.contains(&kind) {
                return Err(Error::Message(
                    "worker returned an unsupported citation kind".to_owned(),
                ));
            }
            let quote = py_string(part.get("support_quote")).trim().to_owned();
            if !quote.is_empty() && !allowed_quote_text.contains(&quote) {
                return Err(Error::Message(
                    "worker support_quote is not copied from the input".to_owned(),
                ));
            }
        }
        validated.insert(id, Value::Array(snapped));
    }
    Ok(Value::Object(validated))
}

fn response_schema() -> Value {
    let mut kinds = KINDS.to_vec();
    kinds.sort_unstable();
    let part = json!({
        "type": "object",
        "additionalProperties": false,
        "required": [
            "verbatim", "corrected", "kind", "pinpoint_fragments",
            "page_pinpoints", "short_form", "bare_citation",
            "citation_with_style", "support_quote"
        ],
        "properties": {
            "verbatim": {"type": "string", "minLength": 1},
            "corrected": {"type": "string"},
            "kind": {"type": "string", "enum": kinds},
            "pinpoint_fragments": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 80}
            },
            "page_pinpoints": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "integer", "minimum": 1}
            },
            "short_form": {"type": "string", "maxLength": 240},
            "bare_citation": {"type": "string", "maxLength": 1000},
            "citation_with_style": {"type": "string", "maxLength": 1600},
            "support_quote": {"type": "string", "maxLength": 1200}
        }
    });
    json!({
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": false,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": false,
                    "required": ["id", "parts"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "parts": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 20,
                            "items": part
                        }
                    }
                }
            }
        }
    })
}

fn prompt(records: &[Value]) -> Result<String> {
    Ok(
        "You are a bounded citation-intent worker for legal DOCX footnotes. \
For every record, split its footnote into source-level parts using the \
McGill-style rule: top-level semicolons normally split sources, missing \
semicolons between distinct authorities still split, and semicolons \
inside one citation do not. Isolate every supra or ibid reference. \
Preserve each verbatim part as an exact, non-overlapping substring; do \
not invent or drop characters. Classify the source and extract only \
compact deterministic lookup fields. pinpoint_fragments use parN for \
case paragraphs and secN for legislation sections/rules/articles; keep \
all separate pinpoints but only the first endpoint of a range. \
page_pinpoints contains integer reporter/PDF pages, never paragraph \
numbers. support_quote is either an exact quotation copied from the \
record's proposition/footnote that the cited source is said to support, \
or an empty string. NEVER output or construct a URL. Mike resolves every \
identity and locator through verified provider tools after this call. \
Return each requested id exactly once and no commentary.\n\nINPUT:\n"
            .to_owned()
            + &python_json(&json!({"records": records}))?,
    )
}

fn fixed_codex_tokens() -> Result<usize> {
    static VALUE: OnceLock<std::result::Result<usize, String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            env::var("LEGALPDF_CODEX_FIXED_TOKENS")
                .unwrap_or_else(|_| "14500".to_owned())
                .parse()
                .map_err(|error| format!("LEGALPDF_CODEX_FIXED_TOKENS is not an integer: {error}"))
        })
        .clone()
        .map_err(Error::Message)
}

fn minimum_route_savings() -> Result<isize> {
    static VALUE: OnceLock<std::result::Result<isize, String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            env::var("LEGALPDF_ROUTE_MIN_TOKEN_SAVINGS")
                .unwrap_or_else(|_| "512".to_owned())
                .parse()
                .map_err(|error| {
                    format!("LEGALPDF_ROUTE_MIN_TOKEN_SAVINGS is not an integer: {error}")
                })
        })
        .clone()
        .map_err(Error::Message)
}

fn batch(records: &[Value]) -> Result<Vec<Vec<Value>>> {
    let mut batches = Vec::<Vec<Value>>::new();
    let mut current = Vec::<Value>::new();
    let mut chars = 0;
    for record in records {
        let size = py_string(record.get("text")).chars().count()
            + py_string(record.get("proposition")).chars().count();
        if !current.is_empty()
            && (current.len() >= MAX_BATCH_FOOTNOTES || chars + size > MAX_BATCH_CHARS)
        {
            batches.push(std::mem::take(&mut current));
            chars = 0;
        }
        current.push(record.clone());
        chars += size;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    if batches.len() > MAX_BATCHES {
        return Err(Error::Message(format!(
            "citation linking requires {} Codex batches; limit is {MAX_BATCHES}",
            batches.len()
        )));
    }
    Ok(batches)
}

fn token_estimate(records: &[Value]) -> Result<usize> {
    let batches = batch(records)?;
    let chars = batches
        .iter()
        .map(|records| prompt(records).map(|value| value.chars().count()))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    Ok(batches.len() * fixed_codex_tokens()? + chars.div_ceil(4))
}

fn route_assessment(footnotes: &[Value]) -> Result<Value> {
    let mut deterministic = Map::new();
    let mut fallback = Vec::new();
    for note in footnotes {
        let note_id = py_string(Some(required(note, "id")?));
        let text = py_string(Some(required(note, "text")?));
        if let Some(intents) = deterministic_docx_intents(&note_id, &text)? {
            deterministic.insert(note_id, Value::Array(intents));
        } else {
            fallback.push(note.clone());
        }
    }
    let direct_tokens = token_estimate(footnotes)?;
    let hybrid_tokens = if fallback.is_empty() {
        0
    } else {
        token_estimate(&fallback)?
    };
    let savings = direct_tokens as isize - hybrid_tokens as isize;
    let recommended = if !deterministic.is_empty() && savings >= minimum_route_savings()? {
        "hybrid"
    } else {
        "direct"
    };
    Ok(json!({
        "recommended_strategy": recommended,
        "footnote_count": footnotes.len(),
        "deterministic_count": deterministic.len(),
        "fallback_count": fallback.len(),
        "estimated_direct_tokens": direct_tokens,
        "estimated_hybrid_tokens": hybrid_tokens,
        "estimated_token_savings": savings,
        "fixed_codex_tokens_per_batch": fixed_codex_tokens()?,
        "minimum_route_savings": minimum_route_savings()?,
        "_deterministic": deterministic,
        "_fallback": fallback,
    }))
}

pub fn assess_docx_route(footnotes: &[Value]) -> Result<Value> {
    route_assessment(footnotes)
}

fn home_dir() -> Option<PathBuf> {
    env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }).map(PathBuf::from)
}

fn expand_user(path: &Path) -> PathBuf {
    let text = path.as_os_str().to_string_lossy();
    if text == "~" {
        return home_dir().unwrap_or_else(|| path.to_owned());
    }
    if let Some(rest) = text.strip_prefix("~/").or_else(|| text.strip_prefix("~\\")) {
        if let Some(home) = home_dir() {
            return home.join(rest);
        }
    }
    path.to_owned()
}

#[cfg(windows)]
fn clean_canonical(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        path
    }
}

#[cfg(not(windows))]
fn clean_canonical(path: PathBuf) -> PathBuf {
    path
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    let expanded = expand_user(path);
    let absolute = if expanded.is_absolute() {
        expanded
    } else {
        env::current_dir()
            .map_err(|source| Error::io(".", source))?
            .join(expanded)
    };
    Ok(clean_canonical(
        fs::canonicalize(&absolute).unwrap_or(absolute),
    ))
}

fn cache_root(cache_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(cache_dir) = cache_dir {
        return absolute_path(cache_dir);
    }
    let base = if cfg!(windows) {
        env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|home| home.join("AppData/Local")))
    } else {
        env::var_os("XDG_CACHE_HOME")
            .map(PathBuf::from)
            .or_else(|| home_dir().map(|home| home.join(".cache")))
    }
    .ok_or_else(|| Error::Message("could not determine the local cache directory".to_owned()))?;
    Ok(base.join("OpenLegalProducts/LegalData/cache/docx-linking"))
}

fn invoke_batch(
    records: &[Value],
    model: &str,
    effort: &str,
    cache_dir: &Path,
    timeout_seconds: u64,
) -> Result<(Map<String, Value>, Value)> {
    let prompt_text = prompt(records)?;
    let key = crate::codex::stable_hash(&json!({
        "prompt_version": DOCX_PROMPT_VERSION,
        "prompt": &prompt_text,
        "model": model,
        "effort": effort,
    }))?;
    let entry = cache_dir.join(&key);
    let response_path = entry.join("last-message.json");
    let metadata_path = entry.join("metadata.json");
    if response_path.is_file() && metadata_path.is_file() {
        let response = serde_json::from_slice(
            &fs::read(&response_path).map_err(|source| Error::io(&response_path, source))?,
        )?;
        let validated = validate_docx_response(&response, records)?;
        let mut metadata = serde_json::from_slice::<Value>(
            &fs::read(&metadata_path).map_err(|source| Error::io(&metadata_path, source))?,
        )?
        .as_object()
        .cloned()
        .ok_or_else(|| Error::Message("cached DOCX metadata is not an object".to_owned()))?;
        metadata.insert("cache_hit".to_owned(), Value::Bool(true));
        return Ok((
            validated.as_object().cloned().expect("validated object"),
            Value::Object(metadata),
        ));
    }
    fs::create_dir_all(&entry).map_err(|source| Error::io(&entry, source))?;
    let schema_path = cache_dir.join(format!("{DOCX_PROMPT_VERSION}.schema.json"));
    if !schema_path.is_file() {
        write_json(&schema_path, &response_schema())?;
    }
    let (response, usage, elapsed) = crate::codex::invoke(
        &prompt_text,
        &schema_path,
        &[],
        model,
        effort,
        &entry,
        timeout_seconds,
    )?;
    let validated = validate_docx_response(&response, records)?;
    let metadata = json!({
        "schema_version": "legalpdf.docx_link_batch.v1",
        "prompt_version": DOCX_PROMPT_VERSION,
        "cache_key": key,
        "model": model,
        "effort": effort,
        "elapsed_seconds": round_four(elapsed),
        "token_usage": usage,
        "record_count": records.len(),
        "input_chars": prompt_text.chars().count(),
        "cache_hit": false,
    });
    write_json(&metadata_path, &metadata)?;
    Ok((
        validated.as_object().cloned().expect("validated object"),
        metadata,
    ))
}

fn round_four(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn contains_ibid(text: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"(?i)\bibid\b").expect("literal ibid regex"))
        .is_match(text)
}

fn resolve_references(footnotes: &mut [Value]) -> Result<()> {
    let by_label = footnotes
        .iter()
        .map(|note| (py_string(note.get("label")), note.clone()))
        .collect::<BTreeMap<_, _>>();
    let mut previous: Option<Value> = None;
    for note in footnotes {
        let parts = note
            .get_mut("parts")
            .and_then(Value::as_array_mut)
            .ok_or_else(|| Error::Message("planned footnote has no parts".to_owned()))?;
        for part in parts {
            let text = py_string(part.get("verbatim"));
            let mut origin = None;
            if contains_ibid(&text) {
                origin = previous.clone();
            } else if let Some(captures) = supra_note_re()?
                .captures(&text)
                .map_err(|error| Error::Message(format!("supra-note search failed: {error}")))?
            {
                if let Some(note_match) = captures.get(1) {
                    let candidates = by_label
                        .get(note_match.as_str())
                        .and_then(|note| note.get("parts"))
                        .and_then(Value::as_array)
                        .cloned()
                        .unwrap_or_default();
                    let whole = captures.get(0).expect("whole supra-note match");
                    let hint = text[..whole.start()]
                        .trim_matches(&[' ', ',', '.', ';', '[', ']', '(', ')'][..]);
                    let hint_folded = hint.to_lowercase();
                    let matching = candidates
                        .iter()
                        .filter(|candidate| {
                            !hint.is_empty()
                                && format!(
                                    "{} {}",
                                    py_string(candidate.get("short_form")),
                                    py_string(candidate.get("citation_with_style"))
                                )
                                .to_lowercase()
                                .contains(&hint_folded)
                        })
                        .cloned()
                        .collect::<Vec<_>>();
                    origin = if matching.len() == 1 {
                        matching.into_iter().next()
                    } else if candidates.len() == 1 {
                        candidates.into_iter().next()
                    } else {
                        None
                    };
                }
            }
            let object = part.as_object_mut().ok_or_else(|| {
                Error::Message("planned citation part is not an object".to_owned())
            })?;
            if let Some(origin) = origin {
                for key in ["kind", "bare_citation", "citation_with_style"] {
                    object.insert(key.to_owned(), required(&origin, key)?.clone());
                }
                object.insert(
                    "origin_part_id".to_owned(),
                    required(&origin, "part_id")?.clone(),
                );
            } else {
                object.insert("origin_part_id".to_owned(), Value::String(String::new()));
            }
            let current = Value::Object(object.clone());
            let kind = py_string(current.get("kind"));
            if kind != "other" || !fancy_match(reference_re()?, &text)? {
                previous = Some(current);
            }
        }
    }
    Ok(())
}

pub fn plan_footnotes(notes: &[Value], options: &DocxPlanOptions) -> Result<Value> {
    let started = Instant::now();
    let mut normalized = Vec::new();
    for note in notes.iter().take(MAX_FOOTNOTES + 1) {
        let id = py_string(Some(required(note, "id")?));
        let label = py_string(note.get("label"));
        let label = if label.is_empty() { id.clone() } else { label };
        normalized.push(json!({
            "id": id,
            "label": label,
            "text": py_string(Some(required(note, "text")?)),
            "proposition": py_string(note.get("proposition")),
        }));
    }
    if normalized.len() > MAX_FOOTNOTES {
        return Err(Error::Message(format!(
            "DOCX has more than {MAX_FOOTNOTES} linkable footnotes"
        )));
    }
    let assessment = route_assessment(&normalized)?;
    let selected = if options.strategy == "auto" {
        py_string(assessment.get("recommended_strategy"))
    } else {
        options.strategy.clone()
    };
    let deterministic = if selected == "hybrid" {
        assessment["_deterministic"]
            .as_object()
            .cloned()
            .unwrap_or_default()
    } else {
        Map::new()
    };
    let model_records = if selected == "hybrid" {
        assessment["_fallback"]
            .as_array()
            .cloned()
            .unwrap_or_default()
    } else {
        normalized.clone()
    };
    let root = cache_root(options.cache_dir.as_deref())?;
    let mut model_results = Map::new();
    let mut telemetry = Vec::new();
    for records in batch(&model_records)? {
        let (results, metadata) = invoke_batch(
            &records,
            &options.model,
            &options.effort,
            &root,
            options.timeout_seconds,
        )?;
        model_results.extend(results);
        telemetry.push(metadata);
    }
    let mut planned = Vec::with_capacity(normalized.len());
    for note in normalized {
        let note_id = py_string(note.get("id"));
        let raw_parts = if let Some(parts) = deterministic.get(&note_id).and_then(Value::as_array) {
            parts.clone()
        } else {
            let parts = model_results
                .get(&note_id)
                .and_then(Value::as_array)
                .ok_or_else(|| {
                    Error::Message(format!("citation worker returned no result for {note_id}"))
                })?;
            parts
                .iter()
                .enumerate()
                .map(|(index, part)| {
                    make_intent(
                        format!("{note_id}:{}", index + 1),
                        py_string(part.get("verbatim")),
                        py_string(part.get("corrected")),
                        py_string(part.get("kind")),
                        part.get("pinpoint_fragments")
                            .and_then(Value::as_array)
                            .map(Vec::as_slice)
                            .unwrap_or_default(),
                        part.get("page_pinpoints")
                            .and_then(Value::as_array)
                            .map(Vec::as_slice)
                            .unwrap_or_default(),
                        py_string(part.get("short_form")),
                        py_string(part.get("bare_citation")),
                        py_string(part.get("citation_with_style")),
                        py_string(part.get("support_quote")),
                        "codex",
                    )
                })
                .collect()
        };
        let mut note = note.as_object().cloned().expect("normalized note object");
        note.insert("parts".to_owned(), Value::Array(raw_parts));
        planned.push(Value::Object(note));
    }
    resolve_references(&mut planned)?;
    let mut token_usage = Map::new();
    for batch in &telemetry {
        if let Some(usage) = batch.get("token_usage").and_then(Value::as_object) {
            for (key, value) in usage {
                let amount = value.as_i64().unwrap_or_default();
                let total = token_usage
                    .get(key)
                    .and_then(Value::as_i64)
                    .unwrap_or_default();
                token_usage.insert(key.clone(), Value::from(total + amount));
            }
        }
    }
    let assessment_public = assessment
        .as_object()
        .expect("assessment object")
        .iter()
        .filter(|(key, _)| !key.starts_with('_'))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect::<Map<_, _>>();
    Ok(json!({
        "schema_version": "legalpdf.footnote_link_plan.v1",
        "model": options.model,
        "effort": options.effort,
        "strategy_requested": options.strategy,
        "strategy_used": selected,
        "assessment": assessment_public,
        "footnotes": planned,
        "telemetry": {
            "elapsed_seconds": round_four(started.elapsed().as_secs_f64()),
            "codex_batches": telemetry.len(),
            "live_codex_batches": telemetry.iter().filter(|item| {
                !item.get("cache_hit").and_then(Value::as_bool).unwrap_or(false)
            }).count(),
            "token_usage": token_usage,
            "batches": telemetry,
        }
    }))
}

#[derive(Clone)]
struct XmlAttribute {
    qname: String,
    namespace: Option<String>,
    local: String,
    value: String,
}

#[derive(Clone)]
struct XmlElement {
    qname: String,
    namespace: Option<String>,
    local: String,
    attributes: Vec<XmlAttribute>,
    children: Vec<XmlNode>,
    self_closing: bool,
}

#[derive(Clone)]
enum XmlNode {
    Element(XmlElement),
    Raw(Event<'static>),
}

struct XmlDocument {
    nodes: Vec<XmlNode>,
}

fn namespace_value(value: ResolveResult<'_>) -> Option<String> {
    match value {
        ResolveResult::Bound(namespace) => {
            Some(String::from_utf8_lossy(namespace.as_ref()).into_owned())
        }
        ResolveResult::Unbound | ResolveResult::Unknown(_) => None,
    }
}

fn decode_attribute(
    raw: &quick_xml::events::attributes::Attribute<'_>,
    decoder: quick_xml::encoding::Decoder,
) -> Result<String> {
    #[allow(deprecated)]
    raw.decode_and_unescape_value(decoder)
        .map(|value| value.into_owned())
        .map_err(|error| Error::Message(format!("XML attribute decoding failed: {error}")))
}

fn parse_element(
    reader: &NsReader<&[u8]>,
    start: &BytesStart<'_>,
    namespace: Option<String>,
    self_closing: bool,
) -> Result<XmlElement> {
    let qname = std::str::from_utf8(start.name().as_ref())
        .map_err(|error| Error::Message(format!("XML name is not UTF-8: {error}")))?
        .to_owned();
    let local = std::str::from_utf8(start.local_name().as_ref())
        .map_err(|error| Error::Message(format!("XML name is not UTF-8: {error}")))?
        .to_owned();
    let mut attributes = Vec::new();
    for raw in start.attributes().with_checks(false) {
        let raw = raw.map_err(|error| Error::Message(format!("XML attribute failed: {error}")))?;
        let qname = std::str::from_utf8(raw.key.as_ref())
            .map_err(|error| Error::Message(format!("XML attribute name is not UTF-8: {error}")))?
            .to_owned();
        let (resolved, local_name) = reader.resolver().resolve_attribute(raw.key);
        let local = std::str::from_utf8(local_name.as_ref())
            .map_err(|error| Error::Message(format!("XML attribute name is not UTF-8: {error}")))?
            .to_owned();
        attributes.push(XmlAttribute {
            qname,
            namespace: namespace_value(resolved),
            local,
            value: decode_attribute(&raw, reader.decoder())?,
        });
    }
    Ok(XmlElement {
        qname,
        namespace,
        local,
        attributes,
        children: Vec::new(),
        self_closing,
    })
}

fn parse_xml(raw: &[u8]) -> Result<XmlDocument> {
    let mut reader = NsReader::from_reader(raw);
    reader.config_mut().trim_text(false);
    let mut buffer = Vec::new();
    let mut roots = Vec::new();
    let mut stack = Vec::<XmlElement>::new();
    loop {
        let event = reader.read_event_into(&mut buffer)?;
        match event {
            Event::Start(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                stack.push(parse_element(&reader, &start, namespace, false)?);
            }
            Event::Empty(start) => {
                let namespace = namespace_value(reader.resolver().resolve_element(start.name()).0);
                let node = XmlNode::Element(parse_element(&reader, &start, namespace, true)?);
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
            Event::End(_) => {
                let element = stack
                    .pop()
                    .ok_or_else(|| Error::Message("XML has an unmatched end tag".to_owned()))?;
                let node = XmlNode::Element(element);
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
            Event::Eof => break,
            Event::Decl(_) => {}
            other => {
                let node = XmlNode::Raw(other.into_owned());
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(node);
                } else {
                    roots.push(node);
                }
            }
        }
        buffer.clear();
    }
    if !stack.is_empty() {
        return Err(Error::Message("XML has an unclosed element".to_owned()));
    }
    Ok(XmlDocument { nodes: roots })
}

fn write_element<W: Write>(writer: &mut Writer<W>, element: &XmlElement) -> Result<()> {
    let mut start = BytesStart::new(element.qname.as_str());
    for attribute in &element.attributes {
        start.push_attribute((attribute.qname.as_str(), attribute.value.as_str()));
    }
    if element.children.is_empty() {
        writer
            .write_event(Event::Empty(start))
            .map_err(|source| Error::Message(format!("XML writing failed: {source}")))?;
        return Ok(());
    }
    writer
        .write_event(Event::Start(start))
        .map_err(|source| Error::Message(format!("XML writing failed: {source}")))?;
    for child in &element.children {
        write_node(writer, child)?;
    }
    writer
        .write_event(Event::End(BytesEnd::new(element.qname.as_str())))
        .map_err(|source| Error::Message(format!("XML writing failed: {source}")))?;
    Ok(())
}

fn write_node<W: Write>(writer: &mut Writer<W>, node: &XmlNode) -> Result<()> {
    match node {
        XmlNode::Element(element) => write_element(writer, element),
        XmlNode::Raw(event) => writer
            .write_event(event.borrow())
            .map_err(|source| Error::Message(format!("XML writing failed: {source}"))),
    }
}

fn serialize_xml(document: &XmlDocument) -> Result<Vec<u8>> {
    let mut writer = Writer::new(Vec::new());
    writer
        .write_event(Event::Decl(BytesDecl::new("1.0", Some("utf-8"), None)))
        .map_err(|source| Error::Message(format!("XML writing failed: {source}")))?;
    for node in &document.nodes {
        write_node(&mut writer, node)?;
    }
    Ok(writer.into_inner())
}

impl XmlDocument {
    fn root(&self) -> Result<&XmlElement> {
        self.nodes
            .iter()
            .find_map(|node| match node {
                XmlNode::Element(element) => Some(element),
                XmlNode::Raw(_) => None,
            })
            .ok_or_else(|| Error::Message("XML has no root element".to_owned()))
    }

    fn root_mut(&mut self) -> Result<&mut XmlElement> {
        self.nodes
            .iter_mut()
            .find_map(|node| match node {
                XmlNode::Element(element) => Some(element),
                XmlNode::Raw(_) => None,
            })
            .ok_or_else(|| Error::Message("XML has no root element".to_owned()))
    }
}

impl XmlElement {
    fn is(&self, namespace: &str, local: &str) -> bool {
        self.namespace.as_deref() == Some(namespace) && self.local == local
    }

    fn attribute(&self, namespace: Option<&str>, local: &str) -> Option<&str> {
        self.attributes
            .iter()
            .find(|attribute| {
                attribute.local == local
                    && namespace
                        .is_none_or(|namespace| attribute.namespace.as_deref() == Some(namespace))
            })
            .map(|attribute| attribute.value.as_str())
    }

    fn set_attribute(&mut self, qname: &str, namespace: Option<&str>, local: &str, value: &str) {
        if let Some(attribute) = self.attributes.iter_mut().find(|attribute| {
            attribute.local == local && attribute.namespace.as_deref() == namespace
        }) {
            attribute.qname = qname.to_owned();
            attribute.value = value.to_owned();
            return;
        }
        self.attributes.push(XmlAttribute {
            qname: qname.to_owned(),
            namespace: namespace.map(str::to_owned),
            local: local.to_owned(),
            value: value.to_owned(),
        });
    }

    fn direct_elements(&self) -> impl Iterator<Item = &XmlElement> {
        self.children.iter().filter_map(|node| match node {
            XmlNode::Element(element) => Some(element),
            XmlNode::Raw(_) => None,
        })
    }
}

fn raw_text(event: &Event<'_>) -> Result<Option<String>> {
    match event {
        Event::Text(text) => {
            let decoded = text
                .decode()
                .map_err(|error| Error::Message(format!("XML text decoding failed: {error}")))?;
            quick_xml::escape::unescape(&decoded)
                .map(|value| Some(value.into_owned()))
                .map_err(|error| Error::Message(format!("XML text unescaping failed: {error}")))
        }
        Event::CData(text) => text
            .decode()
            .map(|value| Some(value.into_owned()))
            .map_err(|error| Error::Message(format!("XML text decoding failed: {error}"))),
        Event::GeneralRef(reference) => {
            if let Some(value) = reference
                .resolve_char_ref()
                .map_err(|error| Error::Message(format!("XML reference failed: {error}")))?
            {
                return Ok(Some(value.to_string()));
            }
            let name = reference
                .decode()
                .map_err(|error| Error::Message(format!("XML reference failed: {error}")))?;
            Ok(Some(
                match name.as_ref() {
                    "lt" => "<",
                    "gt" => ">",
                    "amp" => "&",
                    "apos" => "'",
                    "quot" => "\"",
                    _ => return Ok(Some(format!("&{name};"))),
                }
                .to_owned(),
            ))
        }
        _ => Ok(None),
    }
}

fn element_text(element: &XmlElement) -> Result<String> {
    let mut output = String::new();
    for child in &element.children {
        match child {
            XmlNode::Raw(event) => {
                if let Some(value) = raw_text(event)? {
                    output.push_str(&value);
                }
            }
            XmlNode::Element(child) => output.push_str(&element_text(child)?),
        }
    }
    Ok(output)
}

fn walk_elements<'a>(element: &'a XmlElement, output: &mut Vec<&'a XmlElement>) {
    output.push(element);
    for child in element.direct_elements() {
        walk_elements(child, output);
    }
}

fn paragraph_text(paragraph: &XmlElement) -> Result<(String, Vec<String>)> {
    let mut values = String::new();
    let mut references = Vec::new();
    let mut elements = Vec::new();
    walk_elements(paragraph, &mut elements);
    for element in elements {
        match element.local.as_str() {
            "t" => values.push_str(&element_text(element)?),
            "tab" => values.push('\t'),
            "br" | "cr" => values.push('\n'),
            "footnoteReference" | "endnoteReference" => {
                if let Some(id) = element.attribute(None, "id") {
                    if !id.is_empty() {
                        let kind = if element.local == "footnoteReference" {
                            "footnote"
                        } else {
                            "endnote"
                        };
                        let key = format!("{kind}:{id}");
                        values.push_str(&format!("⟦FN:{key}⟧"));
                        references.push(key);
                    }
                }
            }
            _ => {}
        }
    }
    Ok((values, references))
}

fn paragraph_style(paragraph: &XmlElement) -> String {
    let mut elements = Vec::new();
    walk_elements(paragraph, &mut elements);
    elements
        .into_iter()
        .find(|element| element.local == "pStyle")
        .and_then(|element| element.attribute(None, "val"))
        .unwrap_or_default()
        .to_owned()
}

fn docx_paragraph_text(element: &XmlElement, output: &mut String) -> Result<()> {
    if element.is(W_NS, "del") {
        return Ok(());
    }
    if element.is(W_NS, "t") {
        output.push_str(&element_text(element)?);
        return Ok(());
    }
    for child in element.direct_elements() {
        docx_paragraph_text(child, output)?;
    }
    Ok(())
}

fn normalize_docx_text(text: &str) -> String {
    let text = text
        .replace(['\u{201c}', '\u{201d}'], "\"")
        .replace(['\u{2018}', '\u{2019}'], "'")
        .replace(['\u{00a0}', '\u{2007}', '\u{202f}'], " ");
    legal_structure::normalize_javascript_whitespace(&text)
}

fn normalized_docx_paragraph(paragraph: &XmlElement) -> Result<String> {
    let mut text = String::new();
    docx_paragraph_text(paragraph, &mut text)?;
    Ok(normalize_docx_text(&text))
}

fn tolerant_docx_paragraphs(xml: &str) -> Result<Vec<String>> {
    static PARAGRAPHS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:p\b[^>]*>.*?</w:p>").expect("literal DOCX regex"));
    static DELETIONS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:del\b[^>]*>.*?</w:del>").expect("literal DOCX regex"));
    static TEXTS: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?s)<w:t\b[^>]*>(.*?)</w:t>").expect("literal DOCX regex"));
    PARAGRAPHS
        .find_iter(xml)
        .map(|paragraph| {
            let accepted = DELETIONS.replace_all(paragraph.as_str(), "");
            let mut value = String::new();
            for captures in TEXTS.captures_iter(&accepted) {
                value.push_str(
                    &quick_xml::escape::unescape(captures.get(1).unwrap().as_str())
                        .map_err(|error| Error::Message(error.to_string()))?,
                );
            }
            Ok(normalize_docx_text(&value))
        })
        .collect()
}

static SUPRA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        &[
            r"(?i)supra,?",
            legal_structure::JS_WHITESPACE_CLASS,
            r"{1,4}(?:note|nn?\.?)",
            legal_structure::JS_WHITESPACE_CLASS,
            r"{1,4}([0-9]+)",
        ]
        .concat(),
    )
    .expect("literal DOCX supra regex")
});
static NUMBERED_SUPRA: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)(supra)[^\n\r\u{2028}\u{2029}]{0,40}?[0-9]+")
        .expect("literal numbered supra regex")
});
static NUMBERING_RESTART: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<w:numRestart\b").expect("literal DOCX numbering regex"));
static PARAGRAPH: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:p\b.*?</w:p>").expect("literal DOCX paragraph regex"));
static RUN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:r\b([^>]*)>(.*?)</w:r>").expect("literal DOCX run regex"));
static TEXT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:t\b[^>]*>(.*?)</w:t>").expect("literal DOCX text regex"));
static FIELD_MARKER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:fldChar\b[^>]*\bw:fldCharType=(?:"(begin|end)"|'(begin|end)')[^>]*/?>"#)
        .expect("literal DOCX field regex")
});
static NOTEREF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)NOTEREF").expect("literal NOTEREF regex"));
static RUN_PROPERTIES: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)<w:rPr\b.*?</w:rPr>").expect("literal run properties regex"));
static FOOTNOTE_REFERENCE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:footnoteReference\b[^>]*\bw:id=(?:"(-?[0-9]+)"|'(-?[0-9]+)')[^>]*/?>"#)
        .expect("literal DOCX footnote reference regex")
});
static CUSTOM_MARK: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?-u:\b)w:customMarkFollows=").expect("literal custom mark regex")
});
static BOOKMARK_ID: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:bookmark(?:Start|End)\b[^>]*\bw:id=(?:"([0-9]+)"|'([0-9]+)')"#)
        .expect("literal DOCX bookmark regex")
});
static BOOKMARK_NAME: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"<w:bookmarkStart\b[^>]*\bw:name=(?:"([^"]*)"|'([^']*)')"#)
        .expect("literal DOCX bookmark name regex")
});

#[derive(Debug)]
pub struct DocxSupraCleanup {
    pub bytes: Vec<u8>,
    pub detected: usize,
    pub converted: usize,
    pub already_linked: usize,
    pub review_required: usize,
    pub bookmarks_added: usize,
    pub restarted_numbering: bool,
    pub unsafe_or_split_fields: usize,
}

struct SupraAnalysis {
    detected: usize,
    already_linked: usize,
    ordinals: BTreeSet<usize>,
}

#[derive(Debug)]
struct ParagraphTextNode {
    text: String,
    visible_start: usize,
    visible_end: usize,
    xml_start: usize,
    run_start: usize,
    run_end: usize,
    run_attributes: String,
    run_properties: String,
    safe_to_replace: bool,
}

fn xml_text(value: &str) -> Result<String> {
    quick_xml::escape::unescape(value)
        .map(|value| value.into_owned())
        .map_err(|error| Error::Message(format!("XML text unescaping failed: {error}")))
}

fn escape_xml_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn element_is_open(xml: &str, offset: usize, tag: &str) -> bool {
    let prior = &xml[..offset];
    let open = prior
        .rfind(&format!("<{tag} "))
        .max(prior.rfind(&format!("<{tag}>")));
    open.is_some_and(|open| {
        prior
            .rfind(&format!("</{tag}>"))
            .is_none_or(|close| open > close)
    })
}

fn javascript_iu_word(character: Option<char>) -> bool {
    character.is_some_and(|character| {
        character.is_ascii_alphanumeric()
            || character == '_'
            || matches!(character, '\u{017f}' | '\u{212a}')
    })
}

fn javascript_iu_word_bounded(text: &str, start: usize, end: usize) -> bool {
    javascript_iu_word(text[..start].chars().next_back())
        != javascript_iu_word(text[start..].chars().next())
        && javascript_iu_word(text[..end].chars().next_back())
            != javascript_iu_word(text[end..].chars().next())
}

fn field_spans(paragraph: &str) -> Vec<(usize, usize)> {
    let mut stack = Vec::new();
    let mut spans = Vec::new();
    for marker in FIELD_MARKER.captures_iter(paragraph) {
        let whole = marker.get(0).unwrap();
        if marker.get(1).or_else(|| marker.get(2)).unwrap().as_str() == "begin" {
            stack.push(whole.start());
        } else if let Some(start) = stack.pop() {
            let field = &paragraph[start..whole.start()];
            if NOTEREF
                .find_iter(field)
                .any(|value| javascript_iu_word_bounded(field, value.start(), value.end()))
            {
                spans.push((start, whole.end()));
            }
        }
    }
    spans
}

fn paragraph_text_nodes(
    xml: &str,
    paragraph: &str,
    paragraph_offset: usize,
) -> Result<Vec<ParagraphTextNode>> {
    let mut nodes = Vec::new();
    let mut visible = 0;
    for run in RUN.captures_iter(paragraph) {
        let whole = run.get(0).unwrap();
        let body = run.get(2).unwrap();
        let texts = TEXT.captures_iter(body.as_str()).collect::<Vec<_>>();
        let properties = RUN_PROPERTIES
            .find(body.as_str())
            .map_or("", |value| value.as_str());
        let only_text = texts.len() == 1
            && legal_structure::normalize_javascript_whitespace(
                &body.as_str().replacen(properties, "", 1).replacen(
                    texts[0].get(0).unwrap().as_str(),
                    "",
                    1,
                ),
            )
            .is_empty();
        let safe = only_text
            && ["w:hyperlink", "w:fldSimple", "w:ins", "w:del"]
                .iter()
                .all(|tag| !element_is_open(xml, paragraph_offset + whole.start(), tag));
        for text in texts {
            let value = xml_text(text.get(1).unwrap().as_str())?;
            nodes.push(ParagraphTextNode {
                visible_start: visible,
                visible_end: visible + value.len(),
                xml_start: whole.start()
                    + (body.start() - whole.start())
                    + text.get(0).unwrap().start(),
                run_start: whole.start(),
                run_end: whole.end(),
                run_attributes: run.get(1).unwrap().as_str().to_owned(),
                run_properties: properties.to_owned(),
                text: value,
                safe_to_replace: safe,
            });
            visible = nodes.last().unwrap().visible_end;
        }
    }
    Ok(nodes)
}

fn analyze_supras(xml: &str) -> Result<SupraAnalysis> {
    let mut analysis = SupraAnalysis {
        detected: 0,
        already_linked: 0,
        ordinals: BTreeSet::new(),
    };
    for paragraph in PARAGRAPH.find_iter(xml) {
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes
            .iter()
            .map(|node| node.text.as_str())
            .collect::<String>();
        let fields = field_spans(paragraph.as_str());
        for matched in SUPRA.captures_iter(&visible) {
            let whole = matched.get(0).unwrap();
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            analysis.detected += 1;
            let number = matched.get(1).unwrap();
            let node = nodes.iter().find(|node| {
                node.visible_start <= number.start() && node.visible_end >= number.end()
            });
            if node.is_some_and(|node| {
                fields
                    .iter()
                    .any(|&(start, end)| start <= node.xml_start && node.xml_start < end)
            }) {
                analysis.already_linked += 1;
            } else if let Ok(ordinal) = number.as_str().parse::<usize>() {
                if ordinal > 0 {
                    analysis.ordinals.insert(ordinal);
                }
            }
        }
    }
    Ok(analysis)
}

fn contains_numbered_supra(xml: &str) -> Result<bool> {
    for paragraph in PARAGRAPH.find_iter(xml) {
        let mut visible = String::new();
        for text in TEXT.captures_iter(paragraph.as_str()) {
            visible.push_str(&xml_text(text.get(1).unwrap().as_str())?);
        }
        if NUMBERED_SUPRA.captures_iter(&visible).any(|capture| {
            let whole = capture.get(0).unwrap();
            let word = capture.get(1).unwrap();
            javascript_iu_word_bounded(&visible, word.start(), word.end())
                && javascript_iu_word_bounded(&visible, word.start(), whole.end())
        }) {
            return Ok(true);
        }
    }
    Ok(false)
}

fn read_docx_files(bytes: &[u8], wanted: Option<&[&str]>) -> Result<Vec<(String, Vec<u8>)>> {
    const MAX_EXPANDED_BYTES: u64 = 96 * 1024 * 1024;
    const MAX_XML_PART_BYTES: u64 = 16 * 1024 * 1024;
    const MAX_XML_TOTAL_BYTES: u64 = 32 * 1024 * 1024;
    if bytes.is_empty() || bytes.len() > MAX_DOCX_SUPRA_BYTES {
        return Err(Error::Message(
            "DOCX is empty or exceeds the read limit".to_owned(),
        ));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    let mut files = Vec::with_capacity(wanted.map_or(archive.len(), |names| names.len()));
    let mut seen = HashSet::with_capacity(archive.len());
    let mut expanded = 0_u64;
    let mut xml_expanded = 0_u64;
    let mut declared_expanded = 0_u64;
    let mut declared_xml_expanded = 0_u64;
    let mut file_count = 0;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().replace('\\', "/");
        if name.contains('\0')
            || name.starts_with('/')
            || name.as_bytes().get(1) == Some(&b':')
            || name.split('/').any(|component| component == "..")
        {
            return Err(Error::Message(
                "DOCX contains an unsafe package path".to_owned(),
            ));
        }
        let lower_name = name.to_ascii_lowercase();
        let xml = lower_name.ends_with(".xml") || lower_name.ends_with(".xml.rels");
        if !entry.is_dir() {
            file_count += 1;
            if file_count > 2_048 {
                return Err(Error::Message(
                    "DOCX has too many package entries".to_owned(),
                ));
            }
            declared_expanded = declared_expanded
                .checked_add(entry.size())
                .filter(|&size| size <= MAX_EXPANDED_BYTES)
                .ok_or_else(|| Error::Message("DOCX exceeds the expanded read limit".to_owned()))?;
            if xml {
                if entry.size() > MAX_XML_PART_BYTES {
                    return Err(Error::Message(
                        "DOCX XML part exceeds the read limit".to_owned(),
                    ));
                }
                declared_xml_expanded = declared_xml_expanded
                    .checked_add(entry.size())
                    .filter(|&size| size <= MAX_XML_TOTAL_BYTES)
                    .ok_or_else(|| {
                        Error::Message("DOCX XML parts exceed the read limit".to_owned())
                    })?;
            }
        }
        if !seen.insert(name.clone()) {
            return Err(Error::Message(format!(
                "DOCX contains duplicate package part {name}"
            )));
        }
        if wanted.is_some_and(|wanted| !wanted.contains(&name.as_str())) {
            continue;
        }
        let remaining = MAX_EXPANDED_BYTES.saturating_sub(expanded);
        let limit = if xml {
            remaining
                .min(MAX_XML_PART_BYTES)
                .min(MAX_XML_TOTAL_BYTES.saturating_sub(xml_expanded))
        } else {
            remaining
        };
        let mut value = Vec::with_capacity(entry.size().min(limit).min(1024 * 1024) as usize);
        entry
            .by_ref()
            .take(limit + 1)
            .read_to_end(&mut value)
            .map_err(|error| Error::io(&name, error))?;
        if value.len() as u64 > limit {
            let message = if xml {
                "DOCX XML part exceeds the read limit"
            } else {
                "DOCX exceeds the expanded read limit"
            };
            return Err(Error::Message(message.to_owned()));
        }
        expanded += value.len() as u64;
        if xml {
            xml_expanded += value.len() as u64;
        }
        files.push((name, value));
    }
    Ok(files)
}

fn docx_part(files: &[(String, Vec<u8>)], name: &str) -> Option<String> {
    file_bytes(files, name).map(|bytes| String::from_utf8_lossy(bytes).into_owned())
}

fn footnote_reference_ids(document: &str) -> Vec<usize> {
    FOOTNOTE_REFERENCE
        .captures_iter(document)
        .filter(|matched| !CUSTOM_MARK.is_match(matched.get(0).unwrap().as_str()))
        .filter_map(|matched| {
            matched
                .get(1)
                .or_else(|| matched.get(2))?
                .as_str()
                .parse()
                .ok()
        })
        .filter(|id| *id > 0)
        .collect()
}

fn add_target_bookmarks(
    mut xml: String,
    reference_ids: &[usize],
    ordinals: &BTreeSet<usize>,
) -> (String, HashMap<usize, String>, usize) {
    let mut bookmark_id = BOOKMARK_ID
        .captures_iter(&xml)
        .filter_map(|matched| {
            matched
                .get(1)
                .or_else(|| matched.get(2))?
                .as_str()
                .parse()
                .ok()
        })
        .max()
        .unwrap_or(0)
        + 1;
    let existing = BOOKMARK_NAME
        .captures_iter(&xml)
        .filter_map(|matched| matched.get(1).or_else(|| matched.get(2)))
        .map(|name| name.as_str().to_owned())
        .collect::<HashSet<_>>();
    let mut targets = HashMap::<usize, (usize, usize)>::new();
    for run in RUN.find_iter(&xml) {
        for matched in FOOTNOTE_REFERENCE.captures_iter(run.as_str()) {
            let Some(reference_id) = matched
                .get(1)
                .or_else(|| matched.get(2))
                .and_then(|id| id.as_str().parse().ok())
            else {
                continue;
            };
            targets
                .entry(reference_id)
                .or_insert((run.start(), run.end()));
        }
    }
    let mut names = HashMap::new();
    let mut edits = BTreeMap::<(usize, usize), Vec<(usize, String)>>::new();
    let mut added = 0;
    for &ordinal in ordinals {
        let Some(&reference_id) = ordinal
            .checked_sub(1)
            .and_then(|index| reference_ids.get(index))
        else {
            continue;
        };
        let name = format!("MikeSupraNote{ordinal}");
        names.insert(ordinal, name.clone());
        if existing.contains(&name) {
            continue;
        }
        let Some(&target) = targets.get(&reference_id) else {
            names.remove(&ordinal);
            continue;
        };
        edits.entry(target).or_default().push((bookmark_id, name));
        bookmark_id += 1;
        added += 1;
    }
    for ((start, end), bookmarks) in edits.into_iter().rev() {
        let mut replacement = String::new();
        for (id, name) in &bookmarks {
            replacement.push_str(&format!(
                r#"<w:bookmarkStart w:id="{id}" w:name="{name}"/>"#
            ));
        }
        replacement.push_str(&xml[start..end]);
        for (id, _) in bookmarks.iter().rev() {
            replacement.push_str(&format!(r#"<w:bookmarkEnd w:id="{id}"/>"#));
        }
        xml.replace_range(start..end, &replacement);
    }
    (xml, names, added)
}

fn plain_run(attributes: &str, properties: &str, text: &str) -> String {
    if text.is_empty() {
        String::new()
    } else {
        format!(
            r#"<w:r{attributes}>{properties}<w:t xml:space="preserve">{}</w:t></w:r>"#,
            escape_xml_text(text)
        )
    }
}

fn noteref_field(attributes: &str, properties: &str, name: &str, number: &str) -> String {
    format!(
        concat!(
            r#"<w:r{attributes}>{properties}<w:fldChar w:fldCharType="begin"/></w:r>"#,
            r#"<w:r{attributes}>{properties}<w:instrText xml:space="preserve"> NOTEREF {name} \h </w:instrText></w:r>"#,
            r#"<w:r{attributes}>{properties}<w:fldChar w:fldCharType="separate"/></w:r>"#,
            r#"{number_run}<w:r{attributes}>{properties}<w:fldChar w:fldCharType="end"/></w:r>"#
        ),
        attributes = attributes,
        properties = properties,
        name = name,
        number_run = plain_run(attributes, properties, number)
    )
}

fn convert_safe_paragraphs(xml: &str, names: &HashMap<usize, String>) -> Result<(String, usize)> {
    let mut output = String::with_capacity(xml.len());
    let mut cursor = 0;
    let mut converted = 0;
    for paragraph in PARAGRAPH.find_iter(xml) {
        output.push_str(&xml[cursor..paragraph.start()]);
        let nodes = paragraph_text_nodes(xml, paragraph.as_str(), paragraph.start())?;
        let visible = nodes
            .iter()
            .map(|node| node.text.as_str())
            .collect::<String>();
        let fields = field_spans(paragraph.as_str());
        let mut candidates = BTreeMap::<usize, Vec<(&ParagraphTextNode, usize, &str, &str)>>::new();
        for matched in SUPRA.captures_iter(&visible) {
            let whole = matched.get(0).unwrap();
            if !javascript_iu_word_bounded(&visible, whole.start(), whole.end()) {
                continue;
            }
            let number = matched.get(1).unwrap();
            let Ok(ordinal) = number.as_str().parse::<usize>() else {
                continue;
            };
            let Some(name) = names.get(&ordinal) else {
                continue;
            };
            let Some(node) = nodes.iter().find(|node| {
                node.visible_start <= number.start() && node.visible_end >= number.end()
            }) else {
                continue;
            };
            if !node.safe_to_replace
                || fields
                    .iter()
                    .any(|&(start, end)| start <= node.xml_start && node.xml_start < end)
            {
                continue;
            }
            candidates.entry(node.run_start).or_default().push((
                node,
                number.start(),
                number.as_str(),
                name,
            ));
        }
        let mut edits = Vec::new();
        for rows in candidates.values_mut() {
            rows.sort_by_key(|row| row.1);
            let node = rows[0].0;
            let mut replacement = String::new();
            let mut text_cursor = 0;
            for (_, start, number, name) in rows {
                let local = *start - node.visible_start;
                replacement.push_str(&plain_run(
                    &node.run_attributes,
                    &node.run_properties,
                    &node.text[text_cursor..local],
                ));
                replacement.push_str(&noteref_field(
                    &node.run_attributes,
                    &node.run_properties,
                    name,
                    number,
                ));
                text_cursor = local + number.len();
                converted += 1;
            }
            replacement.push_str(&plain_run(
                &node.run_attributes,
                &node.run_properties,
                &node.text[text_cursor..],
            ));
            edits.push((node.run_start, node.run_end, replacement));
        }
        let mut next = paragraph.as_str().to_owned();
        edits.sort_by_key(|edit| std::cmp::Reverse(edit.0));
        for (start, end, replacement) in edits {
            next.replace_range(start..end, &replacement);
        }
        output.push_str(&next);
        cursor = paragraph.end();
    }
    output.push_str(&xml[cursor..]);
    Ok((output, converted))
}

pub fn has_docx_supra_references(bytes: &[u8]) -> Result<bool> {
    let files = read_docx_files(bytes, Some(&["word/footnotes.xml"]))?;
    if let Some(xml) = docx_part(&files, "word/footnotes.xml") {
        if contains_numbered_supra(&xml)? {
            return Ok(true);
        }
    }
    let files = read_docx_files(bytes, Some(&["word/document.xml"]))?;
    docx_part(&files, "word/document.xml").map_or(Ok(false), |xml| contains_numbered_supra(&xml))
}

pub fn fix_docx_supra_cross_references(bytes: &[u8]) -> Result<DocxSupraCleanup> {
    let files = read_docx_files(
        bytes,
        Some(&[
            "word/document.xml",
            "word/footnotes.xml",
            "word/settings.xml",
        ]),
    )?;
    let document = docx_part(&files, "word/document.xml").ok_or_else(|| {
        Error::Message("DOCX does not contain ordinary Word footnotes".to_owned())
    })?;
    let footnotes = docx_part(&files, "word/footnotes.xml").ok_or_else(|| {
        Error::Message("DOCX does not contain ordinary Word footnotes".to_owned())
    })?;
    let body = analyze_supras(&document)?;
    let notes = analyze_supras(&footnotes)?;
    let detected = body.detected + notes.detected;
    let already_linked = body.already_linked + notes.already_linked;
    let restarted = NUMBERING_RESTART.is_match(&document)
        || docx_part(&files, "word/settings.xml")
            .is_some_and(|settings| NUMBERING_RESTART.is_match(&settings));
    let unchanged = |review_required, restarted_numbering| DocxSupraCleanup {
        bytes: bytes.to_vec(),
        detected,
        converted: 0,
        already_linked,
        review_required,
        bookmarks_added: 0,
        restarted_numbering,
        unsafe_or_split_fields: review_required,
    };
    if detected == 0 || restarted {
        return Ok(unchanged(
            detected.saturating_sub(already_linked),
            restarted,
        ));
    }
    let ordinals = body.ordinals.union(&notes.ordinals).copied().collect();
    let reference_ids = footnote_reference_ids(&document);
    let (bookmarked, names, bookmarks_added) =
        add_target_bookmarks(document, &reference_ids, &ordinals);
    let (next_document, body_converted) = convert_safe_paragraphs(&bookmarked, &names)?;
    let (next_footnotes, note_converted) = convert_safe_paragraphs(&footnotes, &names)?;
    let converted = body_converted + note_converted;
    let review_required = detected.saturating_sub(converted + already_linked);
    if converted == 0 {
        return Ok(unchanged(review_required, false));
    }
    let mut files = read_docx_files(bytes, None)?;
    replace_file(&mut files, "word/document.xml", next_document.into_bytes());
    replace_file(
        &mut files,
        "word/footnotes.xml",
        next_footnotes.into_bytes(),
    );
    let writer = Cursor::new(Vec::new());
    let mut archive = ZipWriter::new(writer);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, value) in files {
        if name.ends_with('/') {
            archive.add_directory(name, options)?;
        } else {
            archive.start_file(name, options)?;
            archive
                .write_all(&value)
                .map_err(|error| Error::io("DOCX output", error))?;
        }
    }
    let output = archive.finish()?.into_inner();
    Ok(DocxSupraCleanup {
        bytes: output,
        detected,
        converted,
        already_linked,
        review_required,
        bookmarks_added,
        restarted_numbering: false,
        unsafe_or_split_fields: review_required,
    })
}

fn nested_word_elements<'a>(element: &'a XmlElement, wanted: &str) -> Vec<&'a XmlElement> {
    let mut found = Vec::new();
    for child in element.direct_elements() {
        if child.is(W_NS, wanted) {
            found.push(child);
        } else if child.is(W_NS, "sdt") || child.is(W_NS, "sdtContent") {
            found.extend(nested_word_elements(child, wanted));
        }
    }
    found
}

fn paragraphs_under<'a>(
    element: &'a XmlElement,
    indexed: &HashMap<*const XmlElement, (usize, usize)>,
) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    let mut pending = vec![element];
    while let Some(current) = pending.pop() {
        if let Some(index) = indexed.get(&(current as *const XmlElement)) {
            found.push(*index);
        } else {
            let children = current.direct_elements().collect::<Vec<_>>();
            pending.extend(children.into_iter().rev());
        }
    }
    found
}

fn positive_word_int(element: Option<&XmlElement>, name: &str) -> usize {
    element
        .and_then(|element| element.direct_elements().find(|child| child.is(W_NS, name)))
        .and_then(|element| element.attribute(None, "val"))
        .and_then(|value| value.parse().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

fn docx_document_xml(bytes: &[u8]) -> Result<Vec<u8>> {
    const MAX_DOCX_BYTES: usize = 50 * 1024 * 1024;
    const MAX_XML_BYTES: u64 = 32 * 1024 * 1024;
    if bytes.is_empty() || bytes.len() > MAX_DOCX_BYTES {
        return Err(Error::Message(
            "DOCX is empty or exceeds the read limit".to_owned(),
        ));
    }
    let mut archive = ZipArchive::new(Cursor::new(bytes))?;
    if archive.len() > 10_000 {
        return Err(Error::Message(
            "DOCX has too many package entries".to_owned(),
        ));
    }
    let mut document = archive
        .by_name("word/document.xml")
        .map_err(|_| Error::Message("DOCX has no word/document.xml".to_owned()))?;
    if document.size() > MAX_XML_BYTES {
        return Err(Error::Message(
            "DOCX document XML exceeds the read limit".to_owned(),
        ));
    }
    let mut xml = Vec::with_capacity(document.size() as usize);
    document
        .read_to_end(&mut xml)
        .map_err(|source| Error::io("word/document.xml", source))?;
    Ok(xml)
}

/// Parse the accepted DOCX text and authoritative table coordinates once, then
/// feed the canonical Rust detector directly.
pub fn analyze_docx_bytes(
    bytes: &[u8],
    document_id: String,
) -> Result<legal_structure::DocumentStructure> {
    let xml = docx_document_xml(bytes)?;
    let document = match parse_xml(&xml) {
        Ok(document) => document,
        Err(_) => {
            let xml = std::str::from_utf8(&xml)
                .map_err(|error| Error::Message(format!("DOCX XML is not UTF-8: {error}")))?;
            let paragraphs = tolerant_docx_paragraphs(xml)?;
            let structure = legal_structure::analyze_docx(document_id, paragraphs, &[])
                .map_err(|error| Error::Message(error.to_string()))?;
            return Ok(structure);
        }
    };
    let root = document.root()?;
    let body = root
        .direct_elements()
        .find(|element| element.is(W_NS, "body"))
        .ok_or_else(|| Error::Message("DOCX has no document body".to_owned()))?;

    let mut all = Vec::new();
    walk_elements(root, &mut all);
    let canonical_elements = all
        .iter()
        .copied()
        .filter(|element| element.is(W_NS, "p") && !element.self_closing)
        .collect::<Vec<_>>();
    let paragraphs = canonical_elements
        .iter()
        .map(|paragraph| normalized_docx_paragraph(paragraph))
        .collect::<Result<Vec<_>>>()?;
    let mut starts = Vec::with_capacity(paragraphs.len());
    let mut text_length = 0;
    for (index, paragraph) in paragraphs.iter().enumerate() {
        text_length += usize::from(index > 0);
        starts.push(text_length);
        text_length += legal_structure::utf16_len(paragraph);
    }

    let mut by_element = canonical_elements
        .iter()
        .enumerate()
        .map(|(index, paragraph)| {
            let start = starts[index];
            (
                *paragraph as *const XmlElement,
                (
                    start,
                    start + legal_structure::utf16_len(&paragraphs[index]),
                ),
            )
        })
        .collect::<HashMap<_, _>>();
    let mut entry_paragraph = HashMap::new();
    let mut preceding = 0;
    for element in &all {
        entry_paragraph.insert(*element as *const XmlElement, preceding);
        if element.is(W_NS, "p") {
            if element.self_closing {
                let offset = starts.get(preceding).copied().unwrap_or(text_length);
                by_element.insert(*element as *const XmlElement, (offset, offset));
            } else {
                preceding += 1;
            }
        }
    }

    let mut table_cells = Vec::new();
    for (table_index, table) in nested_word_elements(body, "tbl").into_iter().enumerate() {
        for (row_index, row) in nested_word_elements(table, "tr").into_iter().enumerate() {
            let row_properties = row
                .direct_elements()
                .find(|element| element.is(W_NS, "trPr"));
            let mut column = positive_word_int(row_properties, "gridBefore");
            for cell in nested_word_elements(row, "tc") {
                let cell_properties = cell
                    .direct_elements()
                    .find(|element| element.is(W_NS, "tcPr"));
                let column_span = positive_word_int(cell_properties, "gridSpan");
                let merged = cell_properties.is_some_and(|properties| {
                    ["vMerge", "hMerge"].into_iter().any(|name| {
                        properties
                            .direct_elements()
                            .find(|element| element.is(W_NS, name))
                            .is_some_and(|element| {
                                !element
                                    .attribute(None, "val")
                                    .unwrap_or_default()
                                    .eq_ignore_ascii_case("restart")
                            })
                    })
                });
                if !merged {
                    let contents = paragraphs_under(cell, &by_element);
                    let empty_at = || {
                        let preceding = entry_paragraph
                            .get(&(cell as *const XmlElement))
                            .copied()
                            .unwrap_or(paragraphs.len());
                        preceding.checked_sub(1).map_or(0, |index| {
                            starts[index] + legal_structure::utf16_len(&paragraphs[index])
                        })
                    };
                    let start = contents.first().map_or_else(empty_at, |(start, _)| *start);
                    let end = contents.last().map_or_else(empty_at, |(_, end)| *end);
                    table_cells.push(legal_structure::AuthoritativeTableCell {
                        table: table_index + 1,
                        table_name: None,
                        row: row_index + 1,
                        column,
                        row_span: None,
                        column_span: Some(column_span),
                        address: None,
                        start,
                        end,
                    });
                }
                column += column_span;
            }
        }
    }
    legal_structure::analyze_docx(document_id, paragraphs, &table_cells)
        .map_err(|error| Error::Message(error.to_string()))
}

fn marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"⟦FN:(?P<id>[^⟧]+)⟧").expect("literal marker regex"))
}

fn sentence_boundary_re() -> &'static FancyRegex {
    static RE: OnceLock<FancyRegex> = OnceLock::new();
    RE.get_or_init(|| {
        FancyRegex::new(r#"[.!?](?:[\"'”’)\]]*)(?=\s|⟦FN:|$)"#)
            .expect("literal sentence boundary regex")
    })
}

fn sentence_at(text: &str, offset: usize) -> Result<String> {
    let boundaries = sentence_boundary_re()
        .find_iter(text)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| Error::Message(format!("sentence boundary search failed: {error}")))?;
    let previous = boundaries
        .iter()
        .filter(|matched| matched.end() <= offset)
        .collect::<Vec<_>>();
    let (start, end) = if previous
        .last()
        .is_some_and(|matched| text[matched.end()..offset].trim().is_empty())
    {
        let end = previous.last().expect("previous boundary").end();
        let start = if previous.len() > 1 {
            previous[previous.len() - 2].end()
        } else {
            0
        };
        (start, end)
    } else {
        let start = previous.last().map_or(0, |matched| matched.end());
        let end = boundaries
            .iter()
            .find(|matched| matched.start() >= offset)
            .map_or(text.len(), |matched| matched.end());
        (start, end)
    };
    Ok(marker_re()
        .replace_all(&text[start..end], "")
        .trim()
        .to_owned())
}

fn citation_strings(text: &str) -> Vec<String> {
    static NEUTRAL: OnceLock<Regex> = OnceLock::new();
    static REPORTER: OnceLock<Regex> = OnceLock::new();
    let neutral = NEUTRAL.get_or_init(|| {
        Regex::new(
            r"\b(?:18|19|20)\d{2}\s+(?:SCC|FC|FCA|ABCA|ABKB|ONCA|ONSC|BCCA|BCSC|QCCA|QCCS|NSCA|NBCA|MBCA|SKCA|NLCA|PECA|YKCA|NWTCA|NUCA)\s+\d+\b",
        )
        .expect("literal neutral citation regex")
    });
    let reporter = REPORTER.get_or_init(|| {
        Regex::new(r"\b\d+\s+[A-Z][A-Za-z.'’& -]{1,50}\s+\d+\b")
            .expect("literal reporter citation regex")
    });
    neutral
        .find_iter(text)
        .chain(reporter.find_iter(text))
        .map(|matched| {
            matched
                .as_str()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn zip_read_all(path: &Path) -> Result<Vec<(String, Vec<u8>)>> {
    let file = File::open(path).map_err(|source| Error::io(path, source))?;
    let mut archive = ZipArchive::new(file)?;
    let mut files = Vec::<(String, Vec<u8>)>::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_owned();
        let mut bytes = Vec::new();
        entry
            .read_to_end(&mut bytes)
            .map_err(|source| Error::io(path, source))?;
        if let Some((_, existing)) = files.iter_mut().find(|(existing, _)| *existing == name) {
            *existing = bytes;
        } else {
            files.push((name, bytes));
        }
    }
    Ok(files)
}

fn file_bytes<'a>(files: &'a [(String, Vec<u8>)], name: &str) -> Option<&'a [u8]> {
    files
        .iter()
        .find(|(entry, _)| entry == name)
        .map(|(_, bytes)| bytes.as_slice())
}

pub fn extract_docx_gold(path: impl AsRef<Path>) -> Result<Value> {
    let source = absolute_path(path.as_ref())?;
    if source
        .extension()
        .and_then(|value| value.to_str())
        .is_none_or(|value| !value.eq_ignore_ascii_case("docx"))
        || !source.is_file()
    {
        return Err(Error::Message(format!(
            "DOCX not found: {}",
            source.display()
        )));
    }
    let files = zip_read_all(&source)?;
    let document_raw = file_bytes(&files, "word/document.xml")
        .ok_or_else(|| Error::Message("DOCX has no word/document.xml".to_owned()))?;
    let document = parse_xml(document_raw)?;
    let mut all = Vec::new();
    walk_elements(document.root()?, &mut all);
    let mut paragraphs = Vec::new();
    let mut reference_order = Vec::new();
    for paragraph in all.into_iter().filter(|element| element.local == "p") {
        let (text, references) = paragraph_text(paragraph)?;
        if text.trim().is_empty() {
            continue;
        }
        reference_order.extend(references.iter().cloned());
        paragraphs.push(json!({
            "text": text,
            "style": paragraph_style(paragraph),
            "footnote_ids": references
                .iter()
                .filter_map(|key| key.strip_prefix("footnote:"))
                .collect::<Vec<_>>(),
            "endnote_ids": references
                .iter()
                .filter_map(|key| key.strip_prefix("endnote:"))
                .collect::<Vec<_>>(),
        }));
    }
    let mut note_text = Vec::<(String, String)>::new();
    for (kind, member_name) in [
        ("footnote", "word/footnotes.xml"),
        ("endnote", "word/endnotes.xml"),
    ] {
        let Some(raw) = file_bytes(&files, member_name) else {
            continue;
        };
        let notes = parse_xml(raw)?;
        let mut all = Vec::new();
        walk_elements(notes.root()?, &mut all);
        for note in all.into_iter().filter(|element| element.local == kind) {
            let Some(id) = note.attribute(None, "id") else {
                continue;
            };
            if id.parse::<i64>().unwrap_or_default() <= 0 {
                continue;
            }
            let mut descendants = Vec::new();
            walk_elements(note, &mut descendants);
            let mut parts = Vec::new();
            for paragraph in descendants
                .into_iter()
                .filter(|element| element.local == "p")
            {
                let (value, _) = paragraph_text(paragraph)?;
                let value = value.trim();
                if !value.is_empty() {
                    parts.push(value.to_owned());
                }
            }
            let text = parts.join(" ");
            if !text.is_empty() {
                let key = format!("{kind}:{id}");
                if let Some((_, prior)) = note_text.iter_mut().find(|(prior, _)| prior == &key) {
                    *prior = text;
                } else {
                    note_text.push((key, text));
                }
            }
        }
    }
    let mut unique_order = Vec::<String>::new();
    for id in reference_order {
        if !unique_order.contains(&id) {
            unique_order.push(id);
        }
    }
    for (key, _) in &note_text {
        if !unique_order.contains(key) {
            unique_order.push(key.clone());
        }
    }
    let mut display_by_key = BTreeMap::new();
    let mut counters = BTreeMap::<&str, usize>::new();
    for key in &unique_order {
        let kind = key.split_once(':').map_or("", |value| value.0);
        let counter = counters.entry(kind).or_default();
        *counter += 1;
        display_by_key.insert(key.clone(), counter.to_string());
    }
    let mut propositions = BTreeMap::<String, (String, String)>::new();
    let mut passage_parts = Vec::<String>::new();
    for paragraph in &paragraphs {
        let text = paragraph["text"].as_str().expect("paragraph text");
        let mut previous_offset = 0;
        for captures in marker_re().captures_iter(text) {
            let matched = captures.get(0).expect("marker match");
            let id = captures.name("id").expect("marker id").as_str();
            let segment = marker_re()
                .replace_all(&text[previous_offset..matched.start()], "")
                .trim()
                .to_owned();
            if !segment.is_empty() {
                passage_parts.push(segment);
            }
            propositions.insert(
                id.to_owned(),
                (
                    sentence_at(text, matched.start())?,
                    passage_parts.join("\n\n"),
                ),
            );
            passage_parts.clear();
            previous_offset = matched.end();
        }
        let tail = marker_re()
            .replace_all(&text[previous_offset..], "")
            .trim()
            .to_owned();
        if !tail.is_empty() {
            passage_parts.push(tail);
        }
    }
    let text_by_key = note_text.into_iter().collect::<BTreeMap<_, _>>();
    let footnotes = unique_order
        .iter()
        .filter_map(|key| {
            text_by_key.get(key).map(|body| {
                let (kind, id) = key.split_once(':').unwrap_or_default();
                let proposition = propositions.get(key);
                json!({
                    "ooxml_id": id,
                    "kind": kind,
                    "label": display_by_key[key],
                    "occurrence": 1,
                    "body": body,
                    "sentence_proposition": proposition.map(|value| value.0.as_str()).unwrap_or_default(),
                    "passage_since_prior_note": proposition.map(|value| value.1.as_str()).unwrap_or_default(),
                })
            })
        })
        .collect::<Vec<_>>();
    let body_text = paragraphs
        .iter()
        .map(|paragraph| paragraph["text"].as_str().expect("paragraph text"))
        .collect::<Vec<_>>()
        .join("\n\n");
    let citation_input = format!(
        "{body_text}\n{}",
        text_by_key.values().cloned().collect::<Vec<_>>().join("\n")
    );
    let citations = citation_strings(&citation_input);
    let mut note_counts = BTreeMap::<&str, usize>::new();
    for note in &footnotes {
        *note_counts
            .entry(note["kind"].as_str().unwrap_or_default())
            .or_default() += 1;
    }
    let result = json!({
        "schema_version": "legalpdf.docx_gold.v2",
        "source_name": source.file_name().and_then(|value| value.to_str()).unwrap_or_default(),
        "source_sha256": sha256_path(&source)?,
        "paragraphs": paragraphs,
        "footnotes": footnotes,
        "note_counts": note_counts,
        "citations": citations,
    });
    Ok(result)
}

pub fn plan_docx_links(path: impl AsRef<Path>, options: &DocxPlanOptions) -> Result<Value> {
    let source = absolute_path(path.as_ref())?;
    let gold = extract_docx_gold(&source)?;
    let notes = gold["footnotes"]
        .as_array()
        .expect("DOCX gold footnotes")
        .iter()
        .map(|note| {
            json!({
                "id": note["ooxml_id"],
                "label": note["label"],
                "text": note["body"],
                "proposition": note["passage_since_prior_note"],
            })
        })
        .collect::<Vec<_>>();
    let mut plan = plan_footnotes(&notes, options)?;
    let object = plan.as_object_mut().expect("DOCX plan object");
    object.insert(
        "schema_version".to_owned(),
        Value::String("legalpdf.docx_link_plan.v1".to_owned()),
    );
    object.insert(
        "source".to_owned(),
        Value::String(source.to_string_lossy().into_owned()),
    );
    object.insert(
        "source_sha256".to_owned(),
        Value::String(sha256_path(&source)?),
    );
    Ok(plan)
}

fn new_element(qname: &str, namespace: &str, local: &str) -> XmlElement {
    XmlElement {
        qname: qname.to_owned(),
        namespace: Some(namespace.to_owned()),
        local: local.to_owned(),
        attributes: Vec::new(),
        children: Vec::new(),
        self_closing: true,
    }
}

fn simple_run(run: &XmlElement) -> bool {
    run.direct_elements()
        .all(|child| child.is(W_NS, "rPr") || child.is(W_NS, "t"))
}

fn run_text(run: &XmlElement) -> Result<String> {
    let mut elements = Vec::new();
    walk_elements(run, &mut elements);
    let mut text = String::new();
    for element in elements.into_iter().filter(|element| element.is(W_NS, "t")) {
        text.push_str(&element_text(element)?);
    }
    Ok(text)
}

fn new_run(source: &XmlElement, text: &str) -> XmlElement {
    let mut run = new_element("w:r", W_NS, "r");
    if let Some(properties) = source.direct_elements().find(|child| child.is(W_NS, "rPr")) {
        run.children.push(XmlNode::Element(properties.clone()));
    }
    let mut value = new_element("w:t", W_NS, "t");
    if text.chars().next().is_some_and(char::is_whitespace)
        || text.chars().next_back().is_some_and(char::is_whitespace)
    {
        value.set_attribute("xml:space", Some(XML_NS), "space", "preserve");
    }
    value
        .children
        .push(XmlNode::Raw(Event::Text(BytesText::new(text).into_owned())));
    run.children.push(XmlNode::Element(value));
    run
}

fn link_paragraph(
    paragraph: &mut XmlElement,
    spans: &[(usize, usize, String)],
    relationship_ids: &BTreeMap<String, String>,
) -> Result<usize> {
    let children = std::mem::take(&mut paragraph.children);
    let mut cursor = 0;
    let mut linked = 0;
    let mut rebuilt = Vec::new();
    for child in children {
        let XmlNode::Element(run) = child else {
            rebuilt.push(child);
            continue;
        };
        if !run.is(W_NS, "r") {
            rebuilt.push(XmlNode::Element(run));
            continue;
        }
        let text = run_text(&run)?;
        let start = cursor;
        let end = cursor + text.len();
        cursor = end;
        let mut intersections = spans
            .iter()
            .filter_map(|(left, right, url)| {
                let left = start.max(*left);
                let right = end.min(*right);
                (left < right).then(|| (left, right, url.clone()))
            })
            .collect::<Vec<_>>();
        if intersections.is_empty() {
            rebuilt.push(XmlNode::Element(run));
            continue;
        }
        if !simple_run(&run) {
            return Err(Error::Message(
                "citation crosses a complex Word run".to_owned(),
            ));
        }
        intersections.sort();
        let mut local = 0;
        for (left, right, url) in intersections {
            let left = left - start;
            let right = right - start;
            if left > local {
                rebuilt.push(XmlNode::Element(new_run(&run, &text[local..left])));
            }
            let mut hyperlink = new_element("w:hyperlink", W_NS, "hyperlink");
            hyperlink.set_attribute(
                "r:id",
                Some(R_NS),
                "id",
                relationship_ids
                    .get(&url)
                    .expect("relationship created for every link"),
            );
            hyperlink
                .children
                .push(XmlNode::Element(new_run(&run, &text[left..right])));
            rebuilt.push(XmlNode::Element(hyperlink));
            linked += 1;
            local = right;
        }
        if local < text.len() {
            rebuilt.push(XmlNode::Element(new_run(&run, &text[local..])));
        }
    }
    paragraph.children = rebuilt;
    Ok(linked)
}

fn link_note_paragraphs(
    element: &mut XmlElement,
    parts: &[Value],
    links: &BTreeMap<String, String>,
    relationship_ids: &BTreeMap<String, String>,
    found_part_ids: &mut HashSet<String>,
    linked_parts: &mut usize,
) -> Result<()> {
    if element.is(W_NS, "p") {
        let text = element
            .direct_elements()
            .filter(|child| child.is(W_NS, "r"))
            .map(run_text)
            .collect::<Result<Vec<_>>>()?
            .join("");
        let mut spans = Vec::new();
        let mut cursor = 0;
        for part in parts {
            let part_id = py_string(part.get("part_id"));
            let verbatim = py_string(part.get("verbatim"));
            let start = text[cursor..]
                .find(&verbatim)
                .map(|offset| cursor + offset)
                .or_else(|| text.find(&verbatim));
            if let Some(start) = start {
                cursor = start + verbatim.len();
                if let Some(url) = links.get(&part_id) {
                    spans.push((start, cursor, url.clone()));
                    found_part_ids.insert(part_id);
                }
            }
        }
        if !spans.is_empty() {
            link_paragraph(element, &spans, relationship_ids)?;
            *linked_parts += spans.len();
        }
    }
    for child in &mut element.children {
        if let XmlNode::Element(child) = child {
            link_note_paragraphs(
                child,
                parts,
                links,
                relationship_ids,
                found_part_ids,
                linked_parts,
            )?;
        }
    }
    Ok(())
}

fn ensure_namespace(root: &mut XmlElement, prefix: &str, namespace: &str) {
    let qname = if prefix.is_empty() {
        "xmlns".to_owned()
    } else {
        format!("xmlns:{prefix}")
    };
    if !root
        .attributes
        .iter()
        .any(|attribute| attribute.qname == qname)
    {
        root.attributes.push(XmlAttribute {
            qname,
            namespace: None,
            local: if prefix.is_empty() {
                "xmlns".to_owned()
            } else {
                prefix.to_owned()
            },
            value: namespace.to_owned(),
        });
    }
}

fn relationships_document(raw: Option<&[u8]>) -> Result<XmlDocument> {
    if let Some(raw) = raw {
        parse_xml(raw)
    } else {
        let mut root = new_element("Relationships", PKG_REL_NS, "Relationships");
        ensure_namespace(&mut root, "", PKG_REL_NS);
        Ok(XmlDocument {
            nodes: vec![XmlNode::Element(root)],
        })
    }
}

fn replace_file(files: &mut Vec<(String, Vec<u8>)>, name: &str, bytes: Vec<u8>) {
    if let Some((_, value)) = files.iter_mut().find(|(entry, _)| entry == name) {
        *value = bytes;
    } else {
        files.push((name.to_owned(), bytes));
    }
}

pub fn apply_docx_links(
    docx_path: impl AsRef<Path>,
    plan: &Value,
    resolved_links: &Value,
    output_path: impl AsRef<Path>,
) -> Result<Value> {
    let source = absolute_path(docx_path.as_ref())?;
    let target = absolute_path(output_path.as_ref())?;
    if plan.get("source_sha256").and_then(Value::as_str) != Some(&sha256_path(&source)?) {
        return Err(Error::Message(
            "link plan does not match the DOCX bytes".to_owned(),
        ));
    }
    let links = resolved_links
        .as_object()
        .into_iter()
        .flat_map(|object| object.iter())
        .filter_map(|(key, value)| {
            value.as_str().and_then(|url| {
                (url.starts_with("https://") || url.starts_with("http://"))
                    .then(|| (key.clone(), url.to_owned()))
            })
        })
        .collect::<BTreeMap<_, _>>();
    if links.values().any(|url| url.chars().count() > 8000) {
        return Err(Error::Message(
            "resolved provider URL is too long".to_owned(),
        ));
    }
    let mut files = zip_read_all(&source)?;
    let footnotes_raw = file_bytes(&files, "word/footnotes.xml")
        .ok_or_else(|| Error::Message("DOCX has no footnotes.xml".to_owned()))?
        .to_vec();
    let mut footnotes = parse_xml(&footnotes_raw)?;
    ensure_namespace(footnotes.root_mut()?, "w", W_NS);
    ensure_namespace(footnotes.root_mut()?, "r", R_NS);
    let rel_path = "word/_rels/footnotes.xml.rels";
    let mut relationships = relationships_document(file_bytes(&files, rel_path))?;
    let relationship_root = relationships.root_mut()?;
    ensure_namespace(relationship_root, "", PKG_REL_NS);
    let mut existing = BTreeMap::<String, String>::new();
    let mut used_ids = HashSet::<String>::new();
    for relationship in relationship_root.direct_elements() {
        let id = relationship.attribute(None, "Id").unwrap_or_default();
        used_ids.insert(id.to_owned());
        if relationship.attribute(None, "Type") == Some(HYPERLINK_REL) {
            existing.insert(
                relationship
                    .attribute(None, "Target")
                    .unwrap_or_default()
                    .to_owned(),
                id.to_owned(),
            );
        }
    }
    let mut relationship_ids = BTreeMap::<String, String>::new();
    for url in links.values().cloned().collect::<BTreeSet<_>>() {
        if let Some(id) = existing.get(&url) {
            relationship_ids.insert(url, id.clone());
            continue;
        }
        let mut number = 1;
        while used_ids.contains(&format!("rId{number}")) {
            number += 1;
        }
        let id = format!("rId{number}");
        used_ids.insert(id.clone());
        relationship_ids.insert(url.clone(), id.clone());
        let mut relationship = new_element("Relationship", PKG_REL_NS, "Relationship");
        relationship.set_attribute("Id", None, "Id", &id);
        relationship.set_attribute("Type", None, "Type", HYPERLINK_REL);
        relationship.set_attribute("Target", None, "Target", &url);
        relationship.set_attribute("TargetMode", None, "TargetMode", "External");
        relationship_root
            .children
            .push(XmlNode::Element(relationship));
    }
    let root = footnotes.root_mut()?;
    let mut note_indices = HashMap::<String, usize>::new();
    for (index, node) in root.children.iter().enumerate() {
        if let XmlNode::Element(note) = node {
            if note.is(W_NS, "footnote") {
                note_indices.insert(
                    note.attribute(Some(W_NS), "id")
                        .or_else(|| note.attribute(None, "id"))
                        .unwrap_or_default()
                        .to_owned(),
                    index,
                );
            }
        }
    }
    let mut linked_parts = 0;
    let mut skipped_parts = 0;
    for note in plan
        .get("footnotes")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
    {
        let id = py_string(note.get("id"));
        let parts = note
            .get("parts")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        let Some(index) = note_indices.get(&id).copied() else {
            skipped_parts += parts.len();
            continue;
        };
        let XmlNode::Element(node) = &mut root.children[index] else {
            unreachable!("footnote index points to an element")
        };
        let mut found = HashSet::new();
        link_note_paragraphs(
            node,
            &parts,
            &links,
            &relationship_ids,
            &mut found,
            &mut linked_parts,
        )?;
        skipped_parts += parts
            .iter()
            .filter(|part| {
                let id = py_string(part.get("part_id"));
                links.contains_key(&id) && !found.contains(&id)
            })
            .count();
    }
    replace_file(&mut files, "word/footnotes.xml", serialize_xml(&footnotes)?);
    replace_file(&mut files, rel_path, serialize_xml(&relationships)?);
    let zip_options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    atomic_write_with(&target, |writer| {
        let mut archive = ZipWriter::new(writer);
        for (name, bytes) in &files {
            archive
                .start_file(name, zip_options)
                .map_err(|source| legal_pdf_core::Error::Message(format!("ZIP error: {source}")))?;
            archive
                .write_all(bytes)
                .map_err(|source| legal_pdf_core::Error::io(&target, source))?;
        }
        archive
            .finish()
            .map_err(|source| legal_pdf_core::Error::Message(format!("ZIP error: {source}")))?;
        Ok(())
    })?;
    Ok(json!({
        "output": target.to_string_lossy(),
        "linked_parts": linked_parts,
        "skipped_parts": skipped_parts,
        "resolved_link_count": links.len(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let root = env::temp_dir().join(format!(
            "legalpdf-docx-test-{}-{}",
            std::process::id(),
            TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        root
    }

    fn fixture(path: &Path, footnote: &str) {
        let document = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>Proposition</w:t></w:r>
<w:r><w:footnoteReference w:id="2"/></w:r></w:p></w:body></w:document>"#
        );
        let notes = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="{W_NS}" xmlns:r="{R_NS}">
<w:footnote w:id="-1" w:type="separator"><w:p/></w:footnote>
<w:footnote w:id="2"><w:p><w:r><w:footnoteRef/></w:r>
<w:r><w:rPr><w:i/></w:rPr><w:t>{footnote}</w:t></w:r></w:p></w:footnote>
</w:footnotes>"#
        );
        let content_types = r#"<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>
</Types>"#;
        let file = File::create(path).unwrap();
        let mut archive = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, value) in [
            ("word/document.xml", document.as_bytes()),
            ("word/footnotes.xml", notes.as_bytes()),
            ("[Content_Types].xml", content_types.as_bytes()),
        ] {
            archive.start_file(name, options).unwrap();
            archive.write_all(value).unwrap();
        }
        archive.finish().unwrap();
    }

    fn mixed_note_fixture(path: &Path) {
        let document = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>First</w:t></w:r>
<w:r><w:footnoteReference w:id="2"/></w:r><w:r><w:t> second</w:t></w:r>
<w:r><w:endnoteReference w:id="7"/></w:r></w:p></w:body></w:document>"#
        );
        let footnotes = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="{W_NS}"><w:footnote w:id="2"><w:p><w:r><w:t>Foot body</w:t></w:r></w:p></w:footnote></w:footnotes>"#
        );
        let endnotes = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<w:endnotes xmlns:w="{W_NS}"><w:endnote w:id="7"><w:p><w:r><w:t>End body</w:t></w:r></w:p></w:endnote></w:endnotes>"#
        );
        let file = File::create(path).unwrap();
        let mut archive = ZipWriter::new(file);
        let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
        for (name, value) in [
            ("word/document.xml", document.as_bytes()),
            ("word/footnotes.xml", footnotes.as_bytes()),
            ("word/endnotes.xml", endnotes.as_bytes()),
        ] {
            archive.start_file(name, options).unwrap();
            archive.write_all(value).unwrap();
        }
        archive.finish().unwrap();
    }

    #[test]
    fn docx_gold_v2_preserves_footnotes_and_endnotes() {
        let root = temporary_directory();
        let source = root.join("mixed.docx");
        mixed_note_fixture(&source);
        let gold = extract_docx_gold(&source).unwrap();
        assert_eq!(gold["schema_version"], "legalpdf.docx_gold.v2");
        assert_eq!(gold["paragraphs"][0]["footnote_ids"], json!(["2"]));
        assert_eq!(gold["paragraphs"][0]["endnote_ids"], json!(["7"]));
        assert_eq!(
            gold["paragraphs"][0]["text"],
            "First⟦FN:footnote:2⟧ second⟦FN:endnote:7⟧"
        );
        assert_eq!(gold["footnotes"][0]["kind"], "footnote");
        assert_eq!(gold["footnotes"][1]["kind"], "endnote");
        assert_eq!(gold["footnotes"][0]["label"], "1");
        assert_eq!(gold["footnotes"][1]["label"], "1");
        assert_eq!(gold["note_counts"], json!({"endnote": 1, "footnote": 1}));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn deterministic_gate_ports_complete_split() {
        let text = "Criminal Code, RSC 1985, c C-46, s 7; R v Example, 2024 SCC 1";
        let intents = deterministic_docx_intents("2", text).unwrap().unwrap();
        assert_eq!(intents[0]["kind"], "statute");
        assert_eq!(intents[1]["kind"], "case");
        assert_eq!(intents[0]["locator"], "7");
        assert!(intents[0].get("link").is_none());
    }

    #[test]
    fn direct_route_is_kept_when_hybrid_saves_nothing() {
        let notes = vec![json!({
            "id": "2",
            "text": "A difficult prose footnote.",
            "proposition": ""
        })];
        let assessment = assess_docx_route(&notes).unwrap();
        assert_eq!(assessment["recommended_strategy"], "direct");
        assert_eq!(assessment["estimated_token_savings"], 0);
    }

    #[test]
    fn worker_contract_rejects_urls_and_snaps_terminal_punctuation() {
        let records = vec![json!({
            "id": "2",
            "text": "R v Example, 2024 SCC 1.",
            "proposition": ""
        })];
        let citation = "R v Example, 2024 SCC 1";
        let part = json!({
            "verbatim": citation,
            "corrected": citation,
            "kind": "case",
            "pinpoint_fragments": [],
            "page_pinpoints": [],
            "short_form": "Example",
            "bare_citation": "2024 SCC 1",
            "citation_with_style": citation,
            "support_quote": ""
        });
        let validated = validate_docx_response(
            &json!({"results": [{"id": "2", "parts": [part.clone()]}]}),
            &records,
        )
        .unwrap();
        assert_eq!(validated["2"][0]["verbatim"], records[0]["text"]);
        let mut url_part = part;
        url_part["support_quote"] = Value::String("https://example.test".to_owned());
        assert!(validate_docx_response(
            &json!({"results": [{"id": "2", "parts": [url_part]}]}),
            &records,
        )
        .unwrap_err()
        .to_string()
        .contains("URL"));
    }

    #[test]
    fn provider_urls_are_applied_without_changing_text() {
        let root = temporary_directory();
        let source = root.join("source.docx");
        let output = root.join("linked.docx");
        let text = "Criminal Code, RSC 1985, c C-46, s 7; R v Example, 2024 SCC 1";
        fixture(&source, text);
        let options = DocxPlanOptions {
            strategy: "hybrid".to_owned(),
            ..DocxPlanOptions::default()
        };
        let plan = plan_docx_links(&source, &options).unwrap();
        assert_eq!(plan["telemetry"]["codex_batches"], 0);
        let links = json!({
            "2:1": "https://laws.example.test/code#sec7",
            "2:2": "https://cases.example.test/example"
        });
        let result = apply_docx_links(&source, &plan, &links, &output).unwrap();
        assert_eq!(result["linked_parts"], 2);
        let gold = extract_docx_gold(&output).unwrap();
        assert_eq!(gold["footnotes"][0]["body"], text);
        let files = zip_read_all(&output).unwrap();
        let relationships =
            parse_xml(file_bytes(&files, "word/_rels/footnotes.xml.rels").unwrap()).unwrap();
        let targets = relationships
            .root()
            .unwrap()
            .direct_elements()
            .filter_map(|element| element.attribute(None, "Target"))
            .collect::<BTreeSet<_>>();
        assert_eq!(
            targets,
            BTreeSet::from([
                "https://laws.example.test/code#sec7",
                "https://cases.example.test/example"
            ])
        );
        fs::remove_dir_all(root).unwrap();
    }
}
