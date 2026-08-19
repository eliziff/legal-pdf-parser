use crate::artifact::{add_geometry_to_compact, read_gzip_json, write_gzip_json, write_json};
use crate::error::{Error, Result};
use crate::model::{
    Diagnostic, ImageBlock, LegalDocument, TableBlock, PARSER_VERSION, SCHEMA_VERSION,
};
use crate::ocr::{OcrOptions, OcrProvider};
use crate::pdf::{extract_pdf, ExtractedPdf};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
use crate::ppdoc::{PPDocLayout, PPDocOptions};
use crate::structure::{
    derive, prepare_pages, replay_derive, status, validate_document, validate_pages,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{self, File, FileTimes};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::time::SystemTime;

const EXTRACTION_CACHE_SCHEMA: &str = "legalpdf.extraction-cache.v1";
const PARSE_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct CachedExtraction {
    schema_version: String,
    source_sha256: String,
    cache_key: String,
    extraction: ExtractedPdf,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ParseMode {
    #[default]
    Local,
    Codex,
}

#[derive(Debug, Clone)]
pub struct ParseOptions {
    pub cache_dir: Option<PathBuf>,
    pub use_cache: bool,
    pub ocr: Option<OcrOptions>,
    /// Zero-based pages eligible for OCR. Native extraction still inspects the
    /// complete PDF so physical page numbering stays authoritative.
    pub ocr_pages: Option<Vec<usize>>,
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    pub ppdoc: Option<PPDocOptions>,
    pub mode: ParseMode,
    pub model: Option<String>,
    pub effort: Option<String>,
    pub repair_timeout_seconds: u64,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            use_cache: true,
            ocr: None,
            ocr_pages: None,
            #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
            ppdoc: None,
            mode: ParseMode::Local,
            model: None,
            effort: None,
            repair_timeout_seconds: 600,
        }
    }
}

pub fn default_cache_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("OPEN_LEGAL_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(root)
            .join("apps")
            .join("legalpdf")
            .join("cache");
    }
    if cfg!(windows) {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        return base
            .join("OpenLegalProducts")
            .join("LegalData")
            .join("apps")
            .join("legalpdf")
            .join("cache");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("OpenLegalData").join("legalpdf")
}

fn sha256_file(path: &Path) -> Result<String> {
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

fn engine_identity() -> Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY
        .get_or_init(|| {
            let sources: &[(&str, &[u8])] = &[
                ("artifact.rs", include_bytes!("artifact.rs")),
                ("engine.rs", include_bytes!("engine.rs")),
                ("grammar_tables.rs", include_bytes!("grammar_tables.rs")),
                ("grammar_word.rs", include_bytes!("grammar_word.rs")),
                #[cfg(feature = "kraken")]
                ("kraken.rs", include_bytes!("kraken.rs")),
                ("model.rs", include_bytes!("model.rs")),
                ("ocr.rs", include_bytes!("ocr.rs")),
                #[cfg(any(feature = "kraken", feature = "ppdoc"))]
                ("ort_runtime.rs", include_bytes!("ort_runtime.rs")),
                ("pairing.rs", include_bytes!("pairing.rs")),
                ("pairing_support.rs", include_bytes!("pairing_support.rs")),
                ("pdf.rs", include_bytes!("pdf.rs")),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                ("ppdoc.rs", include_bytes!("ppdoc.rs")),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                (
                    "ppdoc_postprocess.rs",
                    include_bytes!("ppdoc_postprocess.rs"),
                ),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                ("ppdoc_openvino.rs", include_bytes!("ppdoc_openvino.rs")),
                ("separator.rs", include_bytes!("separator.rs")),
                ("structure.rs", include_bytes!("structure.rs")),
                #[cfg(feature = "kraken")]
                ("tesseract_layout.rs", include_bytes!("tesseract_layout.rs")),
                (
                    "data/mcgill_reporters.json",
                    include_bytes!("../../src/legalpdf/data/mcgill_reporters.json"),
                ),
                (
                    "data/legal-grammar-tables/grammar-corpus.json",
                    include_bytes!("../../data/legal-grammar-tables/grammar-corpus.json"),
                ),
                ("Cargo.lock", include_bytes!("../../Cargo.lock")),
                (
                    "pdf-inspector/.cargo_vcs_info.json",
                    include_bytes!("../../vendor/pdf-inspector/.cargo_vcs_info.json"),
                ),
            ];
            let mut identity = BTreeMap::from([
                (
                    "engine".to_owned(),
                    Value::String("legal-pdf-parser-rust".to_owned()),
                ),
                (
                    "engine_version".to_owned(),
                    Value::String(env!("CARGO_PKG_VERSION").to_owned()),
                ),
                (
                    "parser_version".to_owned(),
                    Value::String(PARSER_VERSION.to_owned()),
                ),
                (
                    "native_extractor".to_owned(),
                    Value::String("pdf-inspector 1.14.0".to_owned()),
                ),
                (
                    "ocr_renderer".to_owned(),
                    Value::String(if cfg!(feature = "ocr") {
                        "hayro 0.7.1".to_owned()
                    } else {
                        "disabled".to_owned()
                    }),
                ),
            ]);
            for (name, bytes) in sources {
                identity.insert(
                    (*name).to_owned(),
                    Value::String(format!("{:x}", Sha256::digest(bytes))),
                );
            }
            json!(identity)
        })
        .clone()
}

fn extraction_identity() -> Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY
        .get_or_init(|| {
            let sources: &[(&str, &[u8])] = &[
                ("model.rs", include_bytes!("model.rs")),
                ("ocr.rs", include_bytes!("ocr.rs")),
                ("pdf.rs", include_bytes!("pdf.rs")),
                #[cfg(feature = "kraken")]
                ("kraken.rs", include_bytes!("kraken.rs")),
                #[cfg(feature = "kraken")]
                ("kraken_process.rs", include_bytes!("kraken_process.rs")),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                ("ppdoc.rs", include_bytes!("ppdoc.rs")),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                (
                    "ppdoc_postprocess.rs",
                    include_bytes!("ppdoc_postprocess.rs"),
                ),
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                ("ppdoc_openvino.rs", include_bytes!("ppdoc_openvino.rs")),
                #[cfg(feature = "kraken")]
                ("tesseract_layout.rs", include_bytes!("tesseract_layout.rs")),
                (
                    "pdf-inspector/.cargo_vcs_info.json",
                    include_bytes!("../../vendor/pdf-inspector/.cargo_vcs_info.json"),
                ),
            ];
            let mut identity = BTreeMap::from([
                (
                    "schema".to_owned(),
                    Value::String(EXTRACTION_CACHE_SCHEMA.to_owned()),
                ),
                (
                    "native_extractor".to_owned(),
                    Value::String("pdf-inspector 1.14.0; lopdf 0.42.0".to_owned()),
                ),
            ]);
            for (name, bytes) in sources {
                identity.insert(
                    (*name).to_owned(),
                    Value::String(format!("{:x}", Sha256::digest(bytes))),
                );
            }
            json!(identity)
        })
        .clone()
}

fn cache_key(
    source_hash: &str,
    identity: &Value,
    ocr_provider: Option<&OcrProvider>,
    ppdoc_identity: Option<&str>,
    ocr_pages: Option<&[usize]>,
) -> Result<String> {
    let value = json!({
        "source_sha256": source_hash,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "engine_code": identity,
        "ocr_provider": ocr_provider.map(OcrProvider::name),
        "ocr_provider_identity": ocr_provider.map(OcrProvider::identity),
        "layout_provider": ppdoc_identity.map(|_| "ppdoc-lite"),
        "layout_provider_identity": ppdoc_identity,
        "ocr_pages": ocr_pages,
    });
    let bytes = serde_json::to_vec(&value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn extraction_cache_key(
    source_hash: &str,
    ocr_provider: Option<&OcrProvider>,
    ppdoc_identity: Option<&str>,
    ocr_pages: Option<&[usize]>,
) -> Result<String> {
    let value = json!({
        "source_sha256": source_hash,
        "extractor": extraction_identity(),
        "ocr_provider": ocr_provider.map(OcrProvider::name),
        "ocr_provider_identity": ocr_provider.map(OcrProvider::identity),
        "layout_provider": ppdoc_identity.map(|_| "ppdoc-lite"),
        "layout_provider_identity": ppdoc_identity,
        "ocr_pages": ocr_pages,
    });
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(&value)?)))
}

fn build_document(
    path: &Path,
    source_hash: &str,
    key: &str,
    identity: &Value,
    providers: (Option<&OcrProvider>, Option<&str>, Option<&str>),
    mut extracted: ExtractedPdf,
) -> Result<LegalDocument> {
    let (ocr_provider, layout_variant, layout_identity) = providers;
    let provider_name = ocr_provider.map(OcrProvider::name).map(str::to_owned);
    let provider_identity = ocr_provider.map(OcrProvider::identity).map(str::to_owned);
    let derived = derive(&mut extracted.pages, &extracted.separators);
    extracted.diagnostics.extend(derived.diagnostics);
    let mut metadata = Map::new();
    metadata.insert("pdf".to_owned(), Value::Object(extracted.metadata));
    metadata.insert("pairing".to_owned(), derived.pairing_summary);
    let mut provenance = Map::new();
    provenance.insert("engine".to_owned(), Value::String("legalpdf".to_owned()));
    provenance.insert(
        "native_extractor".to_owned(),
        Value::String("pdf-inspector 1.14.0 (Rust)".to_owned()),
    );
    provenance.insert(
        "ocr_provider".to_owned(),
        provider_name.map_or(Value::Null, Value::String),
    );
    provenance.insert(
        "ocr_provider_identity".to_owned(),
        provider_identity.map_or(Value::Null, Value::String),
    );
    provenance.insert(
        "layout_provider".to_owned(),
        layout_identity.map_or(Value::Null, |_| Value::String("ppdoc-lite".to_owned())),
    );
    provenance.insert(
        "layout_variant".to_owned(),
        layout_variant.map_or(Value::Null, |value| Value::String(value.to_owned())),
    );
    provenance.insert(
        "layout_provider_identity".to_owned(),
        layout_identity.map_or(Value::Null, |value| Value::String(value.to_owned())),
    );
    provenance.insert("cache_hit".to_owned(), Value::Bool(false));
    provenance.insert("cache_enabled".to_owned(), Value::Bool(true));
    provenance.insert(
        "deterministic_cache_key".to_owned(),
        Value::String(key.to_owned()),
    );
    provenance.insert("engine_code".to_owned(), identity.clone());
    let status = status(&extracted.diagnostics, &extracted.pages);
    let source_name = path
        .file_name()
        .map_or_else(String::new, |value| value.to_string_lossy().into_owned());
    let document = LegalDocument {
        document_id: format!("doc-{}", &source_hash[..20]),
        source_name,
        source_sha256: source_hash.to_owned(),
        page_count: extracted.pages.len(),
        status,
        pages: extracted.pages,
        paragraphs: derived.paragraphs,
        sections: derived.sections,
        footnotes: derived.footnotes,
        tables: extracted.tables,
        images: extracted.images,
        diagnostics: extracted.diagnostics,
        repairs: vec![],
        metadata,
        provenance,
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    validate_document(&document)?;
    Ok(document)
}

fn parse_cache_root(root: &Path) -> PathBuf {
    root.join("parse-v1")
}

fn cache_limit() -> u64 {
    std::env::var("LEGALPDF_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(PARSE_CACHE_MAX_BYTES)
}

fn touch_cache(path: &Path) {
    if let Ok(file) = File::options().write(true).open(path) {
        let _ = file.set_times(FileTimes::new().set_modified(SystemTime::now()));
    }
}

fn prune_parse_cache(root: &Path) {
    let root = parse_cache_root(root);
    let mut files = ["extractions", "documents"]
        .into_iter()
        .flat_map(|directory| {
            fs::read_dir(root.join(directory))
                .into_iter()
                .flatten()
                .flatten()
        })
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            metadata.is_file().then(|| {
                (
                    metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                    metadata.len(),
                    entry.path(),
                )
            })
        })
        .collect::<Vec<_>>();
    let mut total = files.iter().map(|(_, bytes, _)| bytes).sum::<u64>();
    let limit = cache_limit();
    if total <= limit {
        return;
    }
    files.sort_unstable_by_key(|(modified, _, _)| *modified);
    for (_, bytes, path) in files {
        if total <= limit {
            break;
        }
        if fs::remove_file(path).is_ok() {
            total = total.saturating_sub(bytes);
        }
    }
}

pub fn parse_pdf(path: impl AsRef<Path>, options: &ParseOptions) -> Result<LegalDocument> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(Error::Message(format!(
            "PDF does not exist: {}",
            path.display()
        )));
    }
    if path
        .extension()
        .and_then(|value| value.to_str())
        .map(str::to_lowercase)
        .as_deref()
        != Some("pdf")
    {
        return Err(Error::Message(format!(
            "input must be a PDF: {}",
            path.display()
        )));
    }
    let path = fs::canonicalize(path).map_err(|source| Error::io(path, source))?;
    let source_hash = sha256_file(&path)?;
    let identity = engine_identity();
    let mut ocr_provider = options.ocr.as_ref().map(OcrProvider::new).transpose()?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let mut ppdoc = options.ppdoc.as_ref().map(PPDocLayout::new).transpose()?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let ppdoc_identity = ppdoc
        .as_ref()
        .map(|provider| provider.identity().to_owned());
    #[cfg(not(any(feature = "ppdoc", feature = "ppdoc-openvino")))]
    let ppdoc_identity: Option<String> = None;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let ppdoc_variant = ppdoc
        .as_ref()
        .map(|provider| provider.variant_id().to_owned());
    #[cfg(not(any(feature = "ppdoc", feature = "ppdoc-openvino")))]
    let ppdoc_variant: Option<String> = None;
    let key = cache_key(
        &source_hash,
        &identity,
        ocr_provider.as_ref(),
        ppdoc_identity.as_deref(),
        options.ocr_pages.as_deref(),
    )?;
    let extraction_key = extraction_cache_key(
        &source_hash,
        ocr_provider.as_ref(),
        ppdoc_identity.as_deref(),
        options.ocr_pages.as_deref(),
    )?;
    let selected_cache_root = options.cache_dir.clone().unwrap_or_else(default_cache_dir);
    let cache_root = options.use_cache.then(|| selected_cache_root.clone());
    let mut document = None;
    if let Some(root) = &cache_root {
        let path = parse_cache_root(root)
            .join("documents")
            .join(format!("{key}.json.gz"));
        if path.is_file() {
            if let Ok(mut cached) = read_gzip_json::<LegalDocument>(&path) {
                let valid = cached.source_sha256 == source_hash
                    && cached.schema_version == SCHEMA_VERSION
                    && cached.parser_version == PARSER_VERSION
                    && cached
                        .provenance
                        .get("deterministic_cache_key")
                        .and_then(Value::as_str)
                        == Some(key.as_str())
                    && validate_document(&cached).is_ok();
                if valid {
                    cached
                        .provenance
                        .insert("cache_hit".to_owned(), Value::Bool(true));
                    cached
                        .provenance
                        .insert("cache_enabled".to_owned(), Value::Bool(true));
                    touch_cache(&path);
                    document = Some(cached);
                }
            }
            if document.is_none() {
                let _ = fs::remove_file(path);
            }
        }
    }
    if document.is_none() {
        let mut extracted = cache_root.as_ref().and_then(|root| {
            let path = parse_cache_root(root)
                .join("extractions")
                .join(format!("{extraction_key}.json.gz"));
            if !path.is_file() {
                return None;
            }
            match read_gzip_json::<CachedExtraction>(&path) {
                Ok(cached)
                    if cached.schema_version == EXTRACTION_CACHE_SCHEMA
                        && cached.source_sha256 == source_hash
                        && cached.cache_key == extraction_key
                        && cached.extraction.pages.len() == cached.extraction.separators.len() =>
                {
                    touch_cache(&path);
                    Some(cached.extraction)
                }
                _ => {
                    let _ = fs::remove_file(path);
                    None
                }
            }
        });
        if extracted.is_none() {
            let mut fresh = extract_pdf(
                &path,
                ocr_provider.as_mut(),
                options.ocr_pages.as_deref(),
            )?;
            #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
            if let Some(provider) = ppdoc.as_mut() {
                fresh
                    .diagnostics
                    .extend(provider.annotate_pdf(&path, &mut fresh.pages)?);
            }
            if let Some(root) = &cache_root {
                let cached = CachedExtraction {
                    schema_version: EXTRACTION_CACHE_SCHEMA.to_owned(),
                    source_sha256: source_hash.clone(),
                    cache_key: extraction_key.clone(),
                    extraction: fresh,
                };
                let cache_path = parse_cache_root(root)
                    .join("extractions")
                    .join(format!("{extraction_key}.json.gz"));
                let _ = write_gzip_json(&cache_path, &cached);
                fresh = cached.extraction;
            }
            extracted = Some(fresh);
        }
        let mut parsed = build_document(
            &path,
            &source_hash,
            &key,
            &identity,
            (
                ocr_provider.as_ref(),
                ppdoc_variant.as_deref(),
                ppdoc_identity.as_deref(),
            ),
            extracted.expect("cache or parse produced an extraction"),
        )?;
        parsed
            .provenance
            .insert("cache_enabled".to_owned(), Value::Bool(options.use_cache));
        if let Some(root) = &cache_root {
            let cache_path = parse_cache_root(root)
                .join("documents")
                .join(format!("{key}.json.gz"));
            let _ = write_gzip_json(&cache_path, &parsed);
            prune_parse_cache(root);
        }
        document = Some(parsed);
    }
    let document = document.expect("cache or parse produced a document");
    if options.mode == ParseMode::Local {
        return Ok(document);
    }
    let model = options
        .model
        .clone()
        .or_else(|| std::env::var("LEGALPDF_CODEX_MODEL").ok())
        .filter(|value| !value.is_empty());
    let effort = options
        .effort
        .clone()
        .or_else(|| std::env::var("LEGALPDF_CODEX_EFFORT").ok())
        .filter(|value| !value.is_empty());
    let (model, effort) = model.zip(effort).ok_or_else(|| {
        Error::Message(
            "Codex mode requires model and effort arguments or LEGALPDF_CODEX_MODEL and LEGALPDF_CODEX_EFFORT."
                .to_owned(),
        )
    })?;
    crate::repair::improve_document(
        &document,
        &path,
        &model,
        &effort,
        &selected_cache_root.join("codex"),
        options.repair_timeout_seconds,
    )
}

pub fn page_count(path: impl AsRef<Path>) -> Result<usize> {
    Ok(lopdf::Document::load(path)?.get_pages().len())
}

#[doc(hidden)]
pub fn extract_common_input(path: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<PathBuf> {
    extract_layout_input(path, output, None)
}

pub fn extract_layout_input(
    path: impl AsRef<Path>,
    output: impl AsRef<Path>,
    ocr: Option<&OcrOptions>,
) -> Result<PathBuf> {
    let path = path.as_ref();
    let source_hash = sha256_file(path)?;
    let mut ocr_provider = ocr.map(OcrProvider::new).transpose()?;
    let extracted = extract_pdf(path, ocr_provider.as_mut(), None)?;
    let value = json!({
        "schema_version": "legalpdf.common-input.v1",
        "source_name": path.file_name().map_or_else(String::new, |value| value.to_string_lossy().into_owned()),
        "source_sha256": source_hash,
        "pages": extracted.pages,
        "separators": extracted.separators,
        "metadata": extracted.metadata,
        "tables": extracted.tables,
        "images": extracted.images,
        "diagnostics": extracted.diagnostics,
    });
    let output = output.as_ref();
    write_json(output, &value)?;
    Ok(output.to_path_buf())
}

pub fn add_pdf_geometry(
    path: impl AsRef<Path>,
    document: impl AsRef<Path>,
    output: impl AsRef<Path>,
    options: &ParseOptions,
) -> Result<PathBuf> {
    let path = path.as_ref();
    if !path.is_file()
        || path
            .extension()
            .and_then(|value| value.to_str())
            .map(str::to_lowercase)
            .as_deref()
            != Some("pdf")
    {
        return Err(Error::Message(format!(
            "input must be a PDF: {}",
            path.display()
        )));
    }
    let source_hash = sha256_file(path)?;
    let identity = engine_identity();
    let mut ocr_provider = options.ocr.as_ref().map(OcrProvider::new).transpose()?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let mut ppdoc = options.ppdoc.as_ref().map(PPDocLayout::new).transpose()?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let ppdoc_identity = ppdoc.as_ref().map(PPDocLayout::identity);
    #[cfg(not(any(feature = "ppdoc", feature = "ppdoc-openvino")))]
    let ppdoc_identity: Option<&str> = None;
    let key = cache_key(
        &source_hash,
        &identity,
        ocr_provider.as_ref(),
        ppdoc_identity,
        options.ocr_pages.as_deref(),
    )?;
    let mut extracted = extract_pdf(path, ocr_provider.as_mut(), options.ocr_pages.as_deref())?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    if let Some(provider) = ppdoc.as_mut() {
        extracted
            .diagnostics
            .extend(provider.annotate_pdf(path, &mut extracted.pages)?);
    }
    prepare_pages(&mut extracted.pages, &extracted.separators);
    validate_pages(&extracted.pages)?;
    add_geometry_to_compact(
        &extracted.pages,
        document,
        output,
        &source_hash,
        &identity,
        &key,
    )
}

#[derive(Deserialize)]
struct CommonInput {
    schema_version: String,
    source_name: String,
    source_sha256: String,
    pages: Vec<crate::model::Page>,
    separators: Vec<Option<f64>>,
    #[serde(default)]
    metadata: Map<String, Value>,
    #[serde(default)]
    tables: Vec<TableBlock>,
    #[serde(default)]
    images: Vec<ImageBlock>,
    #[serde(default)]
    diagnostics: Vec<Diagnostic>,
}

#[derive(Deserialize)]
struct LayoutAssignments {
    schema_version: String,
    source_sha256: String,
    provider: String,
    model: String,
    identity: String,
    pages: Vec<LayoutAssignmentPage>,
}

#[derive(Deserialize)]
struct LayoutAssignmentPage {
    page_index: usize,
    regions: Vec<LayoutAssignmentRegion>,
}

#[derive(Deserialize)]
struct LayoutAssignmentRegion {
    #[serde(rename = "type")]
    kind: String,
    reading_order: usize,
    line_ids: Vec<String>,
}

const EXTERNAL_LAYOUT_TYPES: &[&str] = &[
    "abstract",
    "content",
    "display_formula",
    "doc_title",
    "figure_title",
    "footer",
    "footnote",
    "header",
    "image",
    "paragraph_title",
    "reference",
    "table",
    "text",
];

pub fn apply_external_layout(
    input: impl AsRef<Path>,
    assignments: impl AsRef<Path>,
) -> Result<LegalDocument> {
    let input = input.as_ref();
    let mut common: CommonInput = serde_json::from_reader(BufReader::new(
        File::open(input).map_err(|source| Error::io(input, source))?,
    ))?;
    if common.schema_version != "legalpdf.common-input.v1"
        || common.separators.len() != common.pages.len()
    {
        return Err(Error::Message(
            "external layout requires a valid legalpdf common input".to_owned(),
        ));
    }
    let assignments_path = assignments.as_ref();
    let assignment_bytes =
        fs::read(assignments_path).map_err(|source| Error::io(assignments_path, source))?;
    let layout: LayoutAssignments = serde_json::from_slice(&assignment_bytes)?;
    if layout.schema_version != "legalpdf.layout-assignments.v1"
        || layout.source_sha256 != common.source_sha256
        || layout.provider.trim().is_empty()
        || layout.model.trim().is_empty()
        || layout.identity.trim().is_empty()
        || [
            layout.provider.as_str(),
            layout.model.as_str(),
            layout.identity.as_str(),
        ]
        .iter()
        .any(|value| value.len() > 1_024 || value.contains(['\r', '\n', '\0']))
    {
        return Err(Error::Message(
            "external layout assignments have invalid provenance".to_owned(),
        ));
    }

    let mut locations = HashMap::new();
    let mut required = HashSet::new();
    for (page_slot, page) in common.pages.iter().enumerate() {
        for (line_slot, line) in page.lines.iter().enumerate() {
            if locations
                .insert(line.id.clone(), (page_slot, line_slot))
                .is_some()
            {
                return Err(Error::Message(format!(
                    "common input contains duplicate line ID: {}",
                    line.id
                )));
            }
            if !line.exclude_from_body && !line.text.trim().is_empty() {
                required.insert(line.id.clone());
            }
        }
    }
    let mut assigned = HashSet::new();
    let mut pages_seen = HashSet::new();
    for page in &layout.pages {
        if !pages_seen.insert(page.page_index) {
            return Err(Error::Message(format!(
                "external layout repeats page index {}",
                page.page_index
            )));
        }
        let page_slot = common
            .pages
            .iter()
            .position(|candidate| candidate.index == page.page_index)
            .ok_or_else(|| {
                Error::Message(format!(
                    "external layout page index is out of range: {}",
                    page.page_index
                ))
            })?;
        let mut regions = page.regions.iter().collect::<Vec<_>>();
        regions.sort_unstable_by_key(|region| region.reading_order);
        for (region_slot, region) in regions.into_iter().enumerate() {
            if region.reading_order == 0
                || region.reading_order > 100_000
                || region.line_ids.is_empty()
                || !EXTERNAL_LAYOUT_TYPES.contains(&region.kind.as_str())
            {
                return Err(Error::Message(format!(
                    "external layout page {} has an invalid region",
                    page.page_index
                )));
            }
            for (line_order, line_id) in region.line_ids.iter().enumerate() {
                let &(candidate_page, line_slot) = locations.get(line_id).ok_or_else(|| {
                    Error::Message(format!("external layout names unknown line ID: {line_id}"))
                })?;
                if candidate_page != page_slot || !required.contains(line_id) {
                    return Err(Error::Message(format!(
                        "external layout line is not assignable on page {}: {line_id}",
                        page.page_index
                    )));
                }
                if !assigned.insert(line_id.clone()) {
                    return Err(Error::Message(format!(
                        "external layout repeats line ID: {line_id}"
                    )));
                }
                let page_id = common.pages[page_slot].id.clone();
                let line = &mut common.pages[page_slot].lines[line_slot];
                line.region_type = region.kind.clone();
                line.region_id = format!("{}-external-r{:04}", page_id, region_slot + 1);
                line.reading_order = region.reading_order * 10_000 + line_order;
            }
        }
    }
    if assigned != required {
        let mut missing = required.difference(&assigned).cloned().collect::<Vec<_>>();
        missing.sort();
        missing.truncate(20);
        return Err(Error::Message(format!(
            "external layout did not assign every visible line; missing: {}",
            missing.join(", ")
        )));
    }

    let derived = derive(&mut common.pages, &common.separators);
    common.diagnostics.extend(derived.diagnostics);
    let mut metadata = common.metadata;
    metadata.insert("pairing".to_owned(), derived.pairing_summary);
    let assignment_sha256 = format!("{:x}", Sha256::digest(&assignment_bytes));
    let provenance = Map::from_iter([
        ("engine".to_owned(), Value::String("legalpdf".to_owned())),
        (
            "layout_provider".to_owned(),
            Value::String(layout.provider.clone()),
        ),
        (
            "layout_variant".to_owned(),
            Value::String(layout.model.clone()),
        ),
        (
            "layout_provider_identity".to_owned(),
            Value::String(layout.identity.clone()),
        ),
        (
            "layout_assignments_sha256".to_owned(),
            Value::String(assignment_sha256),
        ),
    ]);
    let document_token = common.source_sha256.get(..20).ok_or_else(|| {
        Error::Message("common input source_sha256 is not a SHA-256 digest".to_owned())
    })?;
    let document = LegalDocument {
        document_id: format!("doc-{document_token}"),
        source_name: common.source_name,
        source_sha256: common.source_sha256,
        page_count: common.pages.len(),
        status: status(&common.diagnostics, &common.pages),
        pages: common.pages,
        paragraphs: derived.paragraphs,
        sections: derived.sections,
        footnotes: derived.footnotes,
        tables: common.tables,
        images: common.images,
        diagnostics: common.diagnostics,
        repairs: vec![],
        metadata,
        provenance,
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    validate_document(&document)?;
    Ok(document)
}

#[cfg(feature = "ocr")]
pub fn render_pdf_pages(
    path: impl AsRef<Path>,
    output: impl AsRef<Path>,
    dpi: u16,
) -> Result<Vec<PathBuf>> {
    use hayro::hayro_interpret::InterpreterSettings;
    use hayro::hayro_syntax::Pdf;
    use hayro::vello_cpu::color::palette::css::WHITE;
    use hayro::{render, RenderCache, RenderSettings};

    if !(72..=300).contains(&dpi) {
        return Err(Error::Message(
            "layout page rendering DPI must be between 72 and 300".to_owned(),
        ));
    }
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|source| Error::io(path, source))?;
    let pdf = Pdf::new(bytes).map_err(|error| {
        Error::Message(format!("layout renderer could not open PDF: {error:?}"))
    })?;
    let output = output.as_ref();
    fs::create_dir_all(output).map_err(|source| Error::io(output, source))?;
    let cache = RenderCache::new();
    let interpreter = InterpreterSettings::default();
    let scale = f32::from(dpi) / 72.0;
    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: WHITE,
        ..Default::default()
    };
    let mut paths = Vec::new();
    for (index, page) in pdf.pages().iter().enumerate() {
        let pixmap = render(page, &cache, &interpreter, &settings);
        let width = u32::from(pixmap.width());
        let height = u32::from(pixmap.height());
        let pixels = pixmap
            .data_as_u8_slice()
            .chunks_exact(4)
            .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
            .collect();
        let image = image::RgbImage::from_raw(width, height, pixels)
            .ok_or_else(|| Error::Message("layout renderer returned invalid pixels".to_owned()))?;
        let destination = output.join(format!("page-{index:06}.png"));
        image.save(&destination).map_err(|error| {
            Error::Message(format!("could not save {}: {error}", destination.display()))
        })?;
        paths.push(destination);
    }
    Ok(paths)
}

#[doc(hidden)]
pub fn replay_common_input(input: impl AsRef<Path>, output: impl AsRef<Path>) -> Result<PathBuf> {
    let input = input.as_ref();
    let file = File::open(input).map_err(|source| Error::io(input, source))?;
    let mut common: CommonInput = serde_json::from_reader(BufReader::new(file))?;
    if common.schema_version != "legalpdf.common-input.v1" {
        return Err(Error::Message(format!(
            "unsupported common-input schema: {:?}",
            common.schema_version
        )));
    }
    if common.separators.len() != common.pages.len() {
        return Err(Error::Message(
            "common input must contain one separator value per page".to_owned(),
        ));
    }
    let document_token = common.source_sha256.get(..20).ok_or_else(|| {
        Error::Message("common input source_sha256 is not a SHA-256 digest".to_owned())
    })?;
    let replay = replay_derive(&mut common.pages, &common.separators);
    let document = LegalDocument {
        document_id: format!("doc-{document_token}"),
        source_name: common.source_name,
        source_sha256: common.source_sha256.clone(),
        page_count: common.pages.len(),
        status: status(&replay.derived.diagnostics, &common.pages),
        pages: common.pages,
        paragraphs: replay.derived.paragraphs,
        sections: replay.derived.sections,
        footnotes: replay.derived.footnotes,
        tables: vec![],
        images: vec![],
        diagnostics: replay.derived.diagnostics,
        repairs: vec![],
        metadata: Map::new(),
        provenance: Map::new(),
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    validate_document(&document)?;
    let value = json!({
        "schema_version": "legalpdf.common-input-result.v1",
        "source_sha256": document.source_sha256,
        "prepared_pages": replay.prepared_pages,
        "derived_pages": document.pages,
        "markers": replay.derived.markers,
        "marker_summary": replay.derived.marker_summary,
        "pairing_summary": replay.derived.pairing_summary,
        "paragraphs": document.paragraphs,
        "sections": document.sections,
        "footnotes": document.footnotes,
        "diagnostics": document.diagnostics,
        "status": document.status,
        "validation": "ok",
    });
    let output = output.as_ref();
    write_json(output, &value)?;
    Ok(output.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_identity_is_stable() {
        let identity = engine_identity();
        assert_eq!(
            cache_key(&"00".repeat(32), &identity, None, None, None).unwrap(),
            cache_key(&"00".repeat(32), &identity, None, None, None).unwrap()
        );
    }

    #[test]
    fn external_layout_is_source_bound_and_covers_lines_before_derivation() {
        let root = std::env::temp_dir().join(format!(
            "legalpdf-layout-contract-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let input = root.join("input.json");
        let assignments = root.join("assignments.json");
        let source_sha256 = "12".repeat(32);
        let line = |id: &str, text: &str, order: usize, y: f64| {
            json!({
                "id": id,
                "page_index": 0,
                "page_number": 1,
                "source_index": order,
                "reading_order": order,
                "block_index": order,
                "text": text,
                "bbox": [10.0, y, 500.0, y + 20.0],
                "region_id": "",
                "region_type": "unknown",
                "source": "native"
            })
        };
        write_json(
            &input,
            &json!({
                "schema_version": "legalpdf.common-input.v1",
                "source_name": "fixture.pdf",
                "source_sha256": source_sha256,
                "pages": [{
                    "id": "page-1",
                    "index": 0,
                    "number": 1,
                    "width": 612.0,
                    "height": 792.0,
                    "lines": [
                        line("line-heading", "Reasons for Judgment", 1, 50.0),
                        line("line-body", "The appeal is dismissed.", 2, 90.0)
                    ],
                    "regions": [],
                    "source": "native",
                    "text_quality": 1.0
                }],
                "separators": [null],
                "metadata": {},
                "tables": [],
                "images": [],
                "diagnostics": []
            }),
        )
        .unwrap();
        write_json(
            &assignments,
            &json!({
                "schema_version": "legalpdf.layout-assignments.v1",
                "source_sha256": source_sha256,
                "provider": "mllm",
                "model": "fixture-vision",
                "identity": "fixture-identity",
                "pages": [{
                    "page_index": 0,
                    "regions": [
                        {"type": "paragraph_title", "reading_order": 1, "line_ids": ["line-heading"]},
                        {"type": "text", "reading_order": 2, "line_ids": ["line-body"]}
                    ]
                }]
            }),
        )
        .unwrap();
        let document = apply_external_layout(&input, &assignments).unwrap();
        assert_eq!(document.provenance["layout_provider"], "mllm");
        assert!(document.pages[0]
            .lines
            .iter()
            .all(|line| line.region_type != "unknown"));
        let _ = fs::remove_dir_all(root);
    }
}
