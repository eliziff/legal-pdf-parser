use crate::error::{Error, Result};
use crate::lookup::{normal_label, validate_proposition_mode};
use crate::model::{
    Diagnostic, Footnote, FootnoteLookup, ImageBlock, LegalDocument, Line, Page, Paragraph,
    RepairRecord, Section, TableBlock, GEOMETRY_SCHEMA_VERSION, PARSER_VERSION, SCHEMA_VERSION,
};
use flate2::bufread::GzDecoder;
use flate2::write::GzEncoder;
use flate2::{Compression, GzBuilder};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::ser::Formatter;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn io<T>(path: &Path, result: std::io::Result<T>) -> Result<T> {
    result.map_err(|source| Error::io(path, source))
}

fn temporary_path(path: &Path, attempt: u64) -> Result<PathBuf> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| Error::Message(format!("unsafe artifact path: {}", path.display())))?;
    Ok(path.with_file_name(format!(
        ".{name}.{}.{}.tmp",
        std::process::id(),
        TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed) + attempt
    )))
}

pub(crate) fn atomic_write_with(
    path: &Path,
    write: impl FnOnce(&mut BufWriter<File>) -> Result<()>,
) -> Result<()> {
    if let Some(parent) = path.parent() {
        io(parent, fs::create_dir_all(parent))?;
    }
    let mut chosen = None;
    for attempt in 0..32 {
        let candidate = temporary_path(path, attempt)?;
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => {
                chosen = Some((candidate, file));
                break;
            }
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(source) => return Err(Error::io(candidate, source)),
        }
    }
    let (temporary, file) = chosen.ok_or_else(|| {
        Error::Message(format!(
            "could not allocate an artifact temporary for {}",
            path.display()
        ))
    })?;
    let result = (|| {
        let mut writer = BufWriter::new(file);
        write(&mut writer)?;
        io(&temporary, writer.flush())?;
        io(&temporary, writer.get_ref().sync_all())?;
        drop(writer);
        if path.exists() {
            io(path, fs::remove_file(path))?;
        }
        io(path, fs::rename(&temporary, path))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn write_json(path: &Path, value: &Value) -> Result<()> {
    atomic_write_with(path, |writer| {
        serde_json::to_writer_pretty(&mut *writer, value)?;
        writer
            .write_all(b"\n")
            .map_err(|source| Error::io(path, source))
    })
}

fn write_jsonl<T: Serialize>(path: &Path, values: impl IntoIterator<Item = T>) -> Result<()> {
    atomic_write_with(path, |writer| {
        for value in values {
            write_python_json_line(writer, &value, path)?;
        }
        Ok(())
    })
}

pub(crate) struct PythonLineFormatter;

pub(crate) fn python_json(value: &Value) -> Result<String> {
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, PythonLineFormatter);
    value.serialize(&mut serializer)?;
    String::from_utf8(bytes).map_err(|error| Error::Message(format!("JSON is not UTF-8: {error}")))
}

impl Formatter for PythonLineFormatter {
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + Write,
    {
        writer.write_all(b": ")
    }
}

fn write_python_json_line<W: Write, T: Serialize>(
    writer: &mut W,
    value: &T,
    path: &Path,
) -> Result<()> {
    let sorted = serde_json::to_value(value)?;
    let mut serializer = serde_json::Serializer::with_formatter(&mut *writer, PythonLineFormatter);
    sorted.serialize(&mut serializer)?;
    writer
        .write_all(b"\n")
        .map_err(|source| Error::io(path, source))
}

fn compact_page(page: &Page) -> Value {
    json!({
        "id": page.id,
        "index": page.index,
        "number": page.number,
        "printed_label": page.printed_label,
        "printed_label_source": page.printed_label_source,
        "source": page.source,
        "text_quality": page.text_quality,
        "lines": page.lines.iter().map(|line| json!({
            "reading_order": line.reading_order,
            "text": line.text,
        })).collect::<Vec<_>>(),
    })
}

#[derive(Deserialize)]
struct CompactLine {
    reading_order: usize,
    text: String,
}

#[derive(Deserialize)]
struct CompactPage {
    id: String,
    index: usize,
    number: u32,
    source: String,
    text_quality: f64,
    printed_label: Option<String>,
    printed_label_source: Option<String>,
    lines: Vec<CompactLine>,
}

fn read_compact_pages(path: &Path) -> Result<Vec<Page>> {
    read_jsonl::<CompactPage>(path).map(|pages| {
        pages
            .into_iter()
            .map(|page| Page {
                lines: page
                    .lines
                    .into_iter()
                    .enumerate()
                    .map(|(index, line)| Line {
                        id: format!("{}-line-{}", page.id, index + 1),
                        page_index: page.index,
                        page_number: page.number,
                        source_index: index,
                        reading_order: line.reading_order,
                        block_index: index,
                        text: line.text,
                        bbox: [0.0; 4],
                        spans: Vec::new(),
                        words: Vec::new(),
                        detached_references: Vec::new(),
                        exclude_from_body: false,
                        suppress_footnote_label: false,
                        note_region_mode: String::new(),
                        region_id: String::new(),
                        region_type: "unknown".to_owned(),
                        source: page.source.clone(),
                    })
                    .collect(),
                id: page.id,
                index: page.index,
                number: page.number,
                width: 0.0,
                height: 0.0,
                regions: Vec::new(),
                source: page.source,
                text_quality: page.text_quality,
                printed_label: page.printed_label,
                printed_label_source: page.printed_label_source,
                printed_label_line_id: None,
            })
            .collect()
    })
}

fn hash_file(path: &Path) -> Result<String> {
    let file = io(path, File::open(path))?;
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

fn write_jsonl_gzip<T: Serialize>(
    path: &Path,
    values: impl IntoIterator<Item = T>,
) -> Result<String> {
    atomic_write_with(path, |writer| {
        let mut gzip: GzEncoder<&mut BufWriter<File>> =
            GzBuilder::new().mtime(0).write(writer, Compression::fast());
        for value in values {
            write_python_json_line(&mut gzip, &value, path)?;
        }
        gzip.finish().map_err(|source| Error::io(path, source))?;
        Ok(())
    })?;
    hash_file(path)
}

/// Write every collection first and publish `document.json` last.
pub fn write_artifacts(
    document: &LegalDocument,
    output_dir: impl AsRef<Path>,
    compact_pages: bool,
) -> Result<PathBuf> {
    let root = output_dir.as_ref();
    io(root, fs::create_dir_all(root))?;
    let manifest = root.join("document.json");
    if manifest.exists() {
        io(&manifest, fs::remove_file(&manifest))?;
    }
    if compact_pages {
        write_jsonl(
            &root.join("pages.jsonl"),
            document.pages.iter().map(compact_page),
        )?;
    } else {
        write_jsonl(&root.join("pages.jsonl"), &document.pages)?;
    }
    write_jsonl(&root.join("paragraphs.jsonl"), &document.paragraphs)?;
    write_jsonl(&root.join("sections.jsonl"), &document.sections)?;
    write_jsonl(&root.join("footnotes.jsonl"), &document.footnotes)?;
    write_jsonl(&root.join("tables.jsonl"), &document.tables)?;
    write_jsonl(&root.join("images.jsonl"), &document.images)?;
    write_jsonl(&root.join("diagnostics.jsonl"), &document.diagnostics)?;
    write_jsonl(&root.join("repairs.jsonl"), &document.repairs)?;
    write_json(&manifest, &document.manifest(compact_pages))?;
    Ok(manifest)
}

fn read_value(path: &Path) -> Result<Value> {
    let file = io(path, File::open(path))?;
    serde_json::from_reader(BufReader::new(file)).map_err(Error::from)
}

fn read_jsonl<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = io(path, File::open(path))?;
    let mut values = Vec::new();
    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line.map_err(|source| Error::io(path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(serde_json::from_str(&line).map_err(|error| {
            Error::Message(format!(
                "invalid {} line {}: {error}",
                path.display(),
                index + 1
            ))
        })?);
    }
    Ok(values)
}

fn require_fields(value: &Value, kind: &str, fields: &[&str]) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| Error::Message(format!("{kind} artifact is not an object")))?;
    let mut missing = fields
        .iter()
        .filter(|field| !object.contains_key(**field))
        .copied()
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    missing.sort_unstable();
    Err(Error::Message(format!(
        "{kind} artifact is missing fields: {}",
        missing.join(", ")
    )))
}

const WORD_FIELDS: &[&str] = &["id", "text", "bbox", "start", "end"];
const SPAN_FIELDS: &[&str] = &[
    "id",
    "text",
    "bbox",
    "font",
    "size",
    "flags",
    "superscript",
    "start",
    "end",
];
const LINE_FIELDS: &[&str] = &[
    "id",
    "page_index",
    "page_number",
    "source_index",
    "reading_order",
    "block_index",
    "text",
    "bbox",
    "spans",
    "words",
    "detached_references",
    "exclude_from_body",
    "suppress_footnote_label",
    "note_region_mode",
    "region_id",
    "region_type",
    "source",
];
const REGION_FIELDS: &[&str] = &[
    "id",
    "page_index",
    "type",
    "line_ids",
    "bbox",
    "reading_order",
];
const PAGE_FIELDS: &[&str] = &[
    "id",
    "index",
    "number",
    "width",
    "height",
    "lines",
    "regions",
    "source",
    "text_quality",
    "printed_label",
    "printed_label_source",
    "printed_label_line_id",
];
const PARAGRAPH_FIELDS: &[&str] = &[
    "id",
    "page_index",
    "region_type",
    "text",
    "line_ids",
    "anchors",
];
const SECTION_FIELDS: &[&str] = &[
    "id",
    "heading_paragraph_id",
    "heading",
    "locator",
    "locator_kind",
    "aliases",
    "text",
    "paragraph_ids",
    "page_indexes",
    "line_ids",
    "provenance",
];
const FOOTNOTE_FIELDS: &[&str] = &[
    "pair_id",
    "label",
    "occurrence",
    "restart_sequence",
    "reference_page",
    "body_pages",
    "reference_line_id",
    "body_line_ids",
    "body",
    "sentence_proposition",
    "passage_since_prior_note",
    "confidence",
    "provenance",
    "warnings",
    "crossrefs",
];
const TABLE_FIELDS: &[&str] = &[
    "id",
    "page_index",
    "page_number",
    "bbox",
    "cells",
    "provenance",
    "confidence",
];
const IMAGE_FIELDS: &[&str] = &[
    "id",
    "page_index",
    "page_number",
    "bbox",
    "source_name",
    "area_ratio",
    "route",
    "route_reason",
];
const DIAGNOSTIC_FIELDS: &[&str] = &[
    "code",
    "severity",
    "message",
    "page_index",
    "line_ids",
    "details",
];
const REPAIR_FIELDS: &[&str] = &[
    "page_index",
    "status",
    "model",
    "effort",
    "prompt_version",
    "cache_key",
    "attempts",
    "elapsed_seconds",
    "input_line_hash",
    "output_hash",
    "token_usage",
    "error",
    "scope_pages",
];

fn page_from_value(value: Value) -> Result<Page> {
    require_fields(&value, "Page", PAGE_FIELDS)?;
    for line in value
        .get("lines")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Message("Page artifact lines is not an array".to_owned()))?
    {
        require_fields(line, "Line", LINE_FIELDS)?;
        for span in line
            .get("spans")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Message("Line artifact spans is not an array".to_owned()))?
        {
            require_fields(span, "Span", SPAN_FIELDS)?;
        }
        for word in line
            .get("words")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Message("Line artifact words is not an array".to_owned()))?
        {
            require_fields(word, "Word", WORD_FIELDS)?;
        }
    }
    for region in value
        .get("regions")
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Message("Page artifact regions is not an array".to_owned()))?
    {
        require_fields(region, "Region", REGION_FIELDS)?;
    }
    serde_json::from_value(value).map_err(Error::from)
}

fn read_pages(path: &Path) -> Result<Vec<Page>> {
    read_jsonl::<Value>(path)?
        .into_iter()
        .map(page_from_value)
        .collect()
}

fn read_required<T: DeserializeOwned>(path: &Path, kind: &str, fields: &[&str]) -> Result<Vec<T>> {
    read_jsonl::<Value>(path)?
        .into_iter()
        .map(|value| {
            require_fields(&value, kind, fields)?;
            serde_json::from_value(value).map_err(Error::from)
        })
        .collect()
}

fn read_jsonl_reader<T: DeserializeOwned>(path: &Path, reader: impl BufRead) -> Result<Vec<T>> {
    let mut values = Vec::new();
    for (index, line) in reader.lines().enumerate() {
        let line = line.map_err(|source| Error::io(path, source))?;
        if line.trim().is_empty() {
            continue;
        }
        values.push(serde_json::from_str(&line).map_err(|error| {
            Error::Message(format!(
                "invalid {} line {}: {error}",
                path.display(),
                index + 1
            ))
        })?);
    }
    Ok(values)
}

fn read_jsonl_gzip<T: DeserializeOwned>(path: &Path) -> Result<Vec<T>> {
    let file = io(path, File::open(path))?;
    read_jsonl_reader(path, BufReader::new(GzDecoder::new(BufReader::new(file))))
}

pub(crate) fn write_gzip_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    atomic_write_with(path, |writer| {
        let mut gzip = GzBuilder::new().mtime(0).write(writer, Compression::fast());
        serde_json::to_writer(&mut gzip, value)?;
        gzip.finish().map_err(|source| Error::io(path, source))?;
        Ok(())
    })
}

pub(crate) fn read_gzip_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let file = io(path, File::open(path))?;
    serde_json::from_reader(GzDecoder::new(BufReader::new(file))).map_err(Error::from)
}

fn manifest_path(path: &Path) -> PathBuf {
    if path.is_dir() {
        path.join("document.json")
    } else {
        path.to_owned()
    }
}

fn artifact_path(root: &Path, manifest: &Value, key: &str) -> Result<PathBuf> {
    let name = manifest
        .get("artifacts")
        .and_then(|value| value.get(key))
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Message(format!("document manifest has no {key} artifact")))?;
    let candidate = Path::new(name);
    let mut components = candidate.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(Error::Message(format!(
            "unsafe {key} artifact path: {name}"
        )));
    }
    Ok(root.join(candidate))
}

fn required_string<'a>(manifest: &'a Value, key: &str) -> Result<&'a str> {
    manifest
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Message(format!("document manifest has no valid {key}")))
}

fn required_usize(manifest: &Value, key: &str) -> Result<usize> {
    manifest
        .get(key)
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Error::Message(format!("document manifest has no valid {key}")))
}

fn load_manifest(path: &Path) -> Result<(PathBuf, Value)> {
    let path = manifest_path(path);
    let manifest = read_value(&path)?;
    let required = [
        "schema_version",
        "parser_version",
        "document_id",
        "source_name",
        "source_sha256",
        "page_count",
        "status",
        "metadata",
        "provenance",
        "counts",
        "artifacts",
    ];
    let object = manifest
        .as_object()
        .ok_or_else(|| Error::Message("Document manifest is not an object".to_owned()))?;
    let mut missing = required
        .iter()
        .filter(|key| !object.contains_key(**key))
        .copied()
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        missing.sort_unstable();
        return Err(Error::Message(format!(
            "Document manifest is missing fields: {}",
            missing.join(", ")
        )));
    }
    if required_string(&manifest, "schema_version")? != SCHEMA_VERSION {
        return Err(Error::Message(format!(
            "unsupported document schema: {:?}",
            manifest.get("schema_version")
        )));
    }
    let parser = required_string(&manifest, "parser_version")?;
    if parser != PARSER_VERSION {
        return Err(Error::Message(format!(
            "unsupported parser version: {parser}"
        )));
    }
    Ok((path, manifest))
}

fn one_component_path(root: &Path, name: &str, kind: &str) -> Result<PathBuf> {
    let candidate = Path::new(name);
    let mut components = candidate.components();
    if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
        return Err(Error::Message(format!(
            "unsafe {kind} artifact path: {name}"
        )));
    }
    Ok(root.join(candidate))
}

pub fn write_geometry_artifacts(
    pages: &[Page],
    output_dir: impl AsRef<Path>,
    source_sha256: &str,
    engine_code: &Value,
    deterministic_cache_key: &str,
) -> Result<PathBuf> {
    let root = output_dir.as_ref();
    io(root, fs::create_dir_all(root))?;
    let manifest = root.join("geometry.json");
    if manifest.exists() {
        io(&manifest, fs::remove_file(&manifest))?;
    }
    let pages_path = root.join("pages.jsonl.gz");
    let pages_sha256 = write_jsonl_gzip(&pages_path, pages)?;
    write_json(
        &manifest,
        &json!({
            "schema_version": GEOMETRY_SCHEMA_VERSION,
            "parser_version": PARSER_VERSION,
            "source_sha256": source_sha256,
            "engine_code": engine_code,
            "deterministic_cache_key": deterministic_cache_key,
            "page_count": pages.len(),
            "pages_sha256": pages_sha256,
            "artifacts": {"pages": "pages.jsonl.gz"},
        }),
    )?;
    Ok(manifest)
}

pub fn add_geometry_to_compact(
    pages: &[Page],
    document: impl AsRef<Path>,
    output_dir: impl AsRef<Path>,
    source_sha256: &str,
    engine_code: &Value,
    deterministic_cache_key: &str,
) -> Result<PathBuf> {
    let (manifest_path, manifest) = load_manifest(document.as_ref())?;
    if manifest.get("artifact_profile").and_then(Value::as_str) != Some("compact-source") {
        return Err(Error::Message(
            "geometry can only extend a compact-source artifact".to_owned(),
        ));
    }
    if manifest.get("source_sha256").and_then(Value::as_str) != Some(source_sha256)
        || manifest.get("parser_version").and_then(Value::as_str) != Some(PARSER_VERSION)
        || manifest
            .pointer("/provenance/engine_code")
            .is_none_or(|value| value != engine_code)
        || manifest
            .pointer("/provenance/deterministic_cache_key")
            .and_then(Value::as_str)
            != Some(deterministic_cache_key)
    {
        return Err(Error::Message(
            "compact artifact parse identity does not match this engine".to_owned(),
        ));
    }
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let compact: Vec<Value> = read_jsonl(&artifact_path(root, &manifest, "pages")?)?;
    let extracted: Vec<Value> = pages.iter().map(compact_page).collect();
    if compact != extracted {
        return Err(Error::Message(
            "extracted geometry does not match the compact page text".to_owned(),
        ));
    }
    write_geometry_artifacts(
        pages,
        output_dir,
        source_sha256,
        engine_code,
        deterministic_cache_key,
    )
}

pub fn load_geometry_artifacts(
    document: impl AsRef<Path>,
    geometry: impl AsRef<Path>,
) -> Result<Vec<Page>> {
    let (document_path, document_manifest) = load_manifest(document.as_ref())?;
    if document_manifest
        .get("artifact_profile")
        .and_then(Value::as_str)
        != Some("compact-source")
    {
        return Err(Error::Message(
            "geometry requires a compact-source artifact".to_owned(),
        ));
    }
    let geometry_path = if geometry.as_ref().is_dir() {
        geometry.as_ref().join("geometry.json")
    } else {
        geometry.as_ref().to_owned()
    };
    let geometry_manifest = read_value(&geometry_path)?;
    if geometry_manifest
        .get("schema_version")
        .and_then(Value::as_str)
        != Some(GEOMETRY_SCHEMA_VERSION)
        || geometry_manifest
            .get("parser_version")
            .and_then(Value::as_str)
            != Some(PARSER_VERSION)
        || geometry_manifest.get("source_sha256") != document_manifest.get("source_sha256")
        || geometry_manifest.get("engine_code")
            != document_manifest.pointer("/provenance/engine_code")
        || geometry_manifest.get("deterministic_cache_key")
            != document_manifest.pointer("/provenance/deterministic_cache_key")
    {
        return Err(Error::Message(
            "geometry sidecar does not match the compact artifact".to_owned(),
        ));
    }
    let pages_name = geometry_manifest
        .pointer("/artifacts/pages")
        .and_then(Value::as_str)
        .ok_or_else(|| Error::Message("geometry sidecar has no pages artifact".to_owned()))?;
    let root = geometry_path.parent().unwrap_or_else(|| Path::new("."));
    let pages_path = one_component_path(root, pages_name, "geometry pages")?;
    if geometry_manifest
        .get("pages_sha256")
        .and_then(Value::as_str)
        != Some(hash_file(&pages_path)?.as_str())
    {
        return Err(Error::Message(
            "geometry sidecar payload hash does not match".to_owned(),
        ));
    }
    let pages = read_jsonl_gzip::<Value>(&pages_path)?
        .into_iter()
        .map(page_from_value)
        .collect::<Result<Vec<_>>>()?;
    let document_root = document_path.parent().unwrap_or_else(|| Path::new("."));
    let compact_count =
        read_jsonl::<Value>(&artifact_path(document_root, &document_manifest, "pages")?)?.len();
    if geometry_manifest.get("page_count").and_then(Value::as_u64) != Some(pages.len() as u64)
        || compact_count != pages.len()
    {
        return Err(Error::Message(
            "geometry sidecar page count does not match".to_owned(),
        ));
    }
    Ok(pages)
}

pub fn load_artifacts(path: impl AsRef<Path>) -> Result<LegalDocument> {
    let (manifest_path, manifest) = load_manifest(path.as_ref())?;
    if manifest.get("artifact_profile").and_then(Value::as_str) == Some("compact-source") {
        return Err(Error::Message(
            "compact-source artifacts require their geometry sidecar".to_owned(),
        ));
    }
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let pages = read_pages(&artifact_path(root, &manifest, "pages")?)?;
    load_document(manifest_path, manifest, pages)
}

/// Load the immutable fields available in either full or compact-source artifacts.
/// Compact pages intentionally receive no synthetic geometry; projection consumers
/// may use their ordered text but cannot mistake them for renderable pages.
pub fn load_projection_artifacts(path: impl AsRef<Path>) -> Result<LegalDocument> {
    let (manifest_path, manifest) = load_manifest(path.as_ref())?;
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let pages_path = artifact_path(root, &manifest, "pages")?;
    let pages =
        if manifest.get("artifact_profile").and_then(Value::as_str) == Some("compact-source") {
            read_compact_pages(&pages_path)?
        } else {
            read_pages(&pages_path)?
        };
    load_document(manifest_path, manifest, pages)
}

fn load_document(
    manifest_path: PathBuf,
    manifest: Value,
    pages: Vec<Page>,
) -> Result<LegalDocument> {
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let paragraphs: Vec<Paragraph> = read_required(
        &artifact_path(root, &manifest, "paragraphs")?,
        "Paragraph",
        PARAGRAPH_FIELDS,
    )?;
    let sections: Vec<Section> = read_required(
        &artifact_path(root, &manifest, "sections")?,
        "Section",
        SECTION_FIELDS,
    )?;
    let footnotes: Vec<Footnote> = read_required(
        &artifact_path(root, &manifest, "footnotes")?,
        "Footnote",
        FOOTNOTE_FIELDS,
    )?;
    let tables: Vec<TableBlock> = read_required(
        &artifact_path(root, &manifest, "tables")?,
        "TableBlock",
        TABLE_FIELDS,
    )?;
    let images: Vec<ImageBlock> = read_required(
        &artifact_path(root, &manifest, "images")?,
        "ImageBlock",
        IMAGE_FIELDS,
    )?;
    let diagnostics: Vec<Diagnostic> = read_required(
        &artifact_path(root, &manifest, "diagnostics")?,
        "Diagnostic",
        DIAGNOSTIC_FIELDS,
    )?;
    let repairs: Vec<RepairRecord> = read_required(
        &artifact_path(root, &manifest, "repairs")?,
        "RepairRecord",
        REPAIR_FIELDS,
    )?;

    let page_count = required_usize(&manifest, "page_count")?;
    let counts = manifest
        .get("counts")
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Message("document manifest has no counts".to_owned()))?;
    let actual = [
        ("pages", pages.len()),
        ("lines", pages.iter().map(|page| page.lines.len()).sum()),
        ("paragraphs", paragraphs.len()),
        ("sections", sections.len()),
        ("footnotes", footnotes.len()),
        ("tables", tables.len()),
        ("images", images.len()),
        ("diagnostics", diagnostics.len()),
        ("repairs", repairs.len()),
    ];
    if page_count != pages.len()
        || actual
            .iter()
            .any(|(key, value)| counts.get(*key).and_then(Value::as_u64) != Some(*value as u64))
    {
        return Err(Error::Message(
            "document artifact counts do not match the manifest".to_owned(),
        ));
    }

    Ok(LegalDocument {
        document_id: required_string(&manifest, "document_id")?.to_owned(),
        source_name: required_string(&manifest, "source_name")?.to_owned(),
        source_sha256: required_string(&manifest, "source_sha256")?.to_owned(),
        page_count,
        status: required_string(&manifest, "status")?.to_owned(),
        pages,
        paragraphs,
        sections,
        footnotes,
        tables,
        images,
        diagnostics,
        repairs,
        metadata: manifest
            .get("metadata")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| Error::Message("document manifest has no metadata".to_owned()))?,
        provenance: manifest
            .get("provenance")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| Error::Message("document manifest has no provenance".to_owned()))?,
        schema_version: required_string(&manifest, "schema_version")?.to_owned(),
        parser_version: required_string(&manifest, "parser_version")?.to_owned(),
    })
}

pub fn lookup_artifact_footnote(
    path: impl AsRef<Path>,
    query: &str,
    page: Option<u32>,
    occurrence: Option<usize>,
    proposition_mode: &str,
) -> Result<FootnoteLookup> {
    validate_proposition_mode(proposition_mode)?;
    let (manifest_path, manifest) = load_manifest(path.as_ref())?;
    let root = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let query = query.trim();
    let normalized = normal_label(query);
    let footnotes: Vec<Footnote> = read_jsonl(&artifact_path(root, &manifest, "footnotes")?)?;
    let matches: Vec<Footnote> = footnotes
        .into_iter()
        .filter(|footnote| {
            (footnote.pair_id == query || footnote.label == normalized)
                && page.is_none_or(|number| {
                    footnote.reference_page == Some(number) || footnote.body_pages.contains(&number)
                })
                && occurrence.is_none_or(|number| footnote.occurrence == number)
        })
        .collect();
    if matches.is_empty() {
        return Ok(FootnoteLookup {
            status: "not_found".to_owned(),
            query: query.to_owned(),
            matches: Vec::new(),
            footnote: None,
            proposition_mode: proposition_mode.to_owned(),
            proposition: String::new(),
            context: String::new(),
        });
    }
    if matches.len() > 1 {
        return Ok(FootnoteLookup {
            status: "ambiguous".to_owned(),
            query: query.to_owned(),
            matches: matches.into_iter().map(|item| item.pair_id).collect(),
            footnote: None,
            proposition_mode: proposition_mode.to_owned(),
            proposition: String::new(),
            context: String::new(),
        });
    }
    let footnote = matches.into_iter().next().expect("one match");
    let proposition = if proposition_mode == "sentence" {
        footnote.sentence_proposition.clone()
    } else {
        footnote.passage_since_prior_note.clone()
    };
    let marker = format!("⟦FN:{}⟧", footnote.pair_id);
    let paragraphs: Vec<Paragraph> = read_jsonl(&artifact_path(root, &manifest, "paragraphs")?)?;
    let context = paragraphs
        .into_iter()
        .find(|paragraph| paragraph.text.contains(&marker))
        .map(|paragraph| paragraph.text.chars().take(2_000).collect())
        .unwrap_or_default();
    Ok(FootnoteLookup {
        status: "found".to_owned(),
        query: query.to_owned(),
        matches: vec![footnote.pair_id.clone()],
        footnote: Some(footnote),
        proposition_mode: proposition_mode.to_owned(),
        proposition,
        context,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Line, Region, Span, Word};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_dir() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("legalpdf-artifact-{}-{stamp}", std::process::id()))
    }

    fn document() -> LegalDocument {
        let line = Line {
            id: "p0001-l0001".to_owned(),
            page_index: 0,
            page_number: 1,
            source_index: 1,
            reading_order: 1,
            block_index: 1,
            text: "Text".to_owned(),
            bbox: [1.0, 2.0, 3.0, 4.0],
            spans: Vec::<Span>::new(),
            words: Vec::<Word>::new(),
            detached_references: vec![],
            exclude_from_body: false,
            suppress_footnote_label: false,
            note_region_mode: String::new(),
            region_id: "p0001-r0001".to_owned(),
            region_type: "body".to_owned(),
            source: "native".to_owned(),
        };
        LegalDocument {
            document_id: "doc-test".to_owned(),
            source_name: "test.pdf".to_owned(),
            source_sha256: "00".repeat(32),
            page_count: 1,
            status: "ready".to_owned(),
            pages: vec![Page {
                id: "p0001".to_owned(),
                index: 0,
                number: 1,
                width: 612.0,
                height: 792.0,
                lines: vec![line],
                regions: vec![Region {
                    id: "p0001-r0001".to_owned(),
                    page_index: 0,
                    kind: "body".to_owned(),
                    line_ids: vec!["p0001-l0001".to_owned()],
                    bbox: [1.0, 2.0, 3.0, 4.0],
                    reading_order: 1,
                }],
                source: "native".to_owned(),
                text_quality: 1.0,
                printed_label: None,
                printed_label_source: None,
                printed_label_line_id: None,
            }],
            paragraphs: vec![Paragraph {
                id: "para-000001".to_owned(),
                page_index: 0,
                region_type: "body".to_owned(),
                text: "Text".to_owned(),
                line_ids: vec!["p0001-l0001".to_owned()],
                anchors: vec![],
            }],
            sections: vec![],
            footnotes: vec![],
            tables: vec![],
            images: vec![],
            diagnostics: vec![],
            repairs: vec![],
            metadata: Default::default(),
            provenance: Default::default(),
            schema_version: SCHEMA_VERSION.to_owned(),
            parser_version: PARSER_VERSION.to_owned(),
        }
    }

    #[test]
    fn publication_round_trips_and_manifest_is_last_marker() {
        let root = temp_dir();
        let document = document();
        let manifest = write_artifacts(&document, &root, false).unwrap();
        assert!(manifest.is_file());
        assert_eq!(load_artifacts(&manifest).unwrap(), document);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compact_geometry_sidecar_is_identity_bound_and_round_trips() {
        let root = temp_dir();
        let mut document = document();
        let engine_code = json!({"engine": "test"});
        document
            .provenance
            .insert("engine_code".to_owned(), engine_code.clone());
        document.provenance.insert(
            "deterministic_cache_key".to_owned(),
            Value::String("cache-key".to_owned()),
        );
        let compact = write_artifacts(&document, root.join("compact"), true).unwrap();
        let projected = load_projection_artifacts(&compact).unwrap();
        assert_eq!(projected.pages[0].lines[0].text, "Text");
        assert_eq!(
            projected.pages[0].lines[0].reading_order,
            document.pages[0].lines[0].reading_order
        );
        assert_eq!(projected.pages[0].width, 0.0);
        assert_eq!(projected.paragraphs, document.paragraphs);
        let geometry = add_geometry_to_compact(
            &document.pages,
            &compact,
            root.join("geometry"),
            &document.source_sha256,
            &engine_code,
            "cache-key",
        )
        .unwrap();
        assert_eq!(
            load_geometry_artifacts(&compact, &geometry).unwrap(),
            document.pages
        );
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn compressed_cache_is_one_round_trippable_file() {
        let root = temp_dir();
        let path = root.join("document.json.gz");
        let value = json!({"text": "legal text", "pages": [1, 2, 3]});
        write_gzip_json(&path, &value).unwrap();
        assert_eq!(read_gzip_json::<Value>(&path).unwrap(), value);
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
