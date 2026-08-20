use crate::error::{Error, Result};
use crate::model::{
    Diagnostic, Footnote, LegalDocument, Line, Page, Paragraph, Region, Section, Span, Word,
    PARSER_VERSION, SCHEMA_VERSION,
};
use crate::ocr::{OcrOptions, OcrProvider};
use crate::pdf::{extract_pdf, ExtractedPdf};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
use crate::ppdoc::{PPDocLayout, PPDocOptions};
use crate::storage::{read_gzip_json, write_gzip_json};
use crate::structure::{derive, replay, status, validate_document};
use serde::{ser::SerializeMap, Deserialize, Serialize, Serializer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, FileTimes};
use std::io::{BufReader, BufWriter, Read, Write};
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

#[derive(Debug, Clone)]
pub(crate) struct ParseOptions {
    pub cache_dir: Option<PathBuf>,
    pub use_cache: bool,
    pub ocr: Option<OcrOptions>,
    /// Zero-based pages eligible for OCR. Native extraction still inspects the
    /// complete PDF so physical page numbering stays authoritative.
    pub ocr_pages: Option<Vec<usize>>,
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    pub ppdoc: Option<PPDocOptions>,
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
        }
    }
}

fn default_cache_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("OPEN_LEGAL_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(root).join("apps/legalpdf/cache");
    }
    if cfg!(windows) {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        return base.join("OpenLegalProducts/LegalData/apps/legalpdf/cache");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("OpenLegalData/legalpdf")
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
                ("storage.rs", include_bytes!("storage.rs")),
                ("structure.rs", include_bytes!("structure.rs")),
                #[cfg(feature = "kraken")]
                ("tesseract_layout.rs", include_bytes!("tesseract_layout.rs")),
                (
                    "data/mcgill_reporters.json",
                    include_bytes!("../../data/mcgill_reporters.json"),
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
    let derived = derive(&mut extracted.pages, &extracted.separators)?;
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

pub(crate) fn parse_pdf(path: impl AsRef<Path>, options: &ParseOptions) -> Result<LegalDocument> {
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
            let mut fresh =
                extract_pdf(&path, ocr_provider.as_mut(), options.ocr_pages.as_deref())?;
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
    Ok(document.expect("cache or parse produced a document"))
}

struct CommonInput {
    source_name: String,
    source_sha256: String,
    pages: Vec<crate::model::Page>,
    separators: Vec<Option<f64>>,
}

struct ReplayOutput {
    derived_pages: Vec<Page>,
    diagnostics: Vec<Diagnostic>,
    footnotes: Vec<Footnote>,
    marker_summary: Value,
    markers: Vec<Value>,
    pairing_summary: Value,
    paragraphs: Vec<Paragraph>,
    prepared_pages: Vec<Page>,
    schema_version: &'static str,
    sections: Vec<Section>,
    source_sha256: String,
    status: String,
    validation: &'static str,
}

trait FrozenOrder {
    fn serialize_frozen<S: Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error>;
}

struct Frozen<'a, T>(&'a T);

impl<T: FrozenOrder> Serialize for Frozen<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        self.0.serialize_frozen(serializer)
    }
}

struct FrozenSlice<'a, T>(&'a [T]);

impl<T: FrozenOrder> Serialize for FrozenSlice<'_, T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        serializer.collect_seq(self.0.iter().map(Frozen))
    }
}

macro_rules! frozen_fields {
    ($map:ident $value:ident;) => {};
    ($map:ident $value:ident; [$field:ident] $($rest:tt)*) => {
        $map.serialize_entry(stringify!($field), &FrozenSlice(&$value.$field))?;
        frozen_fields!($map $value; $($rest)*);
    };
    ($map:ident $value:ident; $field:ident => $key:literal $($rest:tt)*) => {
        $map.serialize_entry($key, &$value.$field)?;
        frozen_fields!($map $value; $($rest)*);
    };
    ($map:ident $value:ident; $field:ident $($rest:tt)*) => {
        $map.serialize_entry(stringify!($field), &$value.$field)?;
        frozen_fields!($map $value; $($rest)*);
    };
}

macro_rules! frozen_type {
    ($kind:ident: $($fields:tt)*) => {
        impl FrozenOrder for $kind {
            fn serialize_frozen<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_map(None)?;
                frozen_fields!(map self; $($fields)*);
                map.end()
            }
        }
    };
}

frozen_type!(Word: bbox end id start text);
frozen_type!(Span: bbox end flags font id size start superscript text);
frozen_type!(Line: bbox block_index detached_references exclude_from_body id note_region_mode page_index page_number reading_order region_id region_type source source_index [spans] suppress_footnote_label text [words]);
frozen_type!(Region: bbox id line_ids page_index reading_order kind => "type");
frozen_type!(Page: height id index [lines] number printed_label printed_label_line_id printed_label_source [regions] source text_quality width);
frozen_type!(Paragraph: anchors id line_ids page_index region_type text);
frozen_type!(Section: aliases heading heading_paragraph_id id line_ids locator locator_kind page_indexes paragraph_ids provenance text);
frozen_type!(Footnote: body body_line_ids body_pages confidence crossrefs label occurrence pair_id passage_since_prior_note provenance reference_line_id reference_page restart_sequence sentence_proposition warnings);
frozen_type!(Diagnostic: code details line_ids message page_index severity);
frozen_type!(ReplayOutput: [derived_pages] [diagnostics] [footnotes] marker_summary markers pairing_summary [paragraphs] [prepared_pages] schema_version [sections] source_sha256 status validation);

fn replay_common(mut common: CommonInput) -> Result<ReplayOutput> {
    let document_token = common.source_sha256.get(..20).ok_or_else(|| {
        Error::Message("common input source_sha256 is not a SHA-256 digest".to_owned())
    })?;
    let replay = replay(&mut common.pages, &common.separators)?;
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
    Ok(ReplayOutput {
        derived_pages: document.pages,
        diagnostics: document.diagnostics,
        footnotes: document.footnotes,
        marker_summary: replay.derived.marker_summary,
        markers: replay.derived.markers,
        pairing_summary: replay.derived.pairing_summary,
        paragraphs: document.paragraphs,
        prepared_pages: replay.prepared_pages,
        schema_version: "legalpdf.common-input-result.v1",
        sections: document.sections,
        source_sha256: document.source_sha256,
        status: document.status,
        validation: "ok",
    })
}

#[derive(Debug, Default)]
struct DigestWriter {
    bytes: u64,
    digest: Sha256,
}

impl Write for DigestWriter {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.bytes += bytes.len() as u64;
        self.digest.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[doc(hidden)]
pub fn digest_cached_extraction(input: impl AsRef<Path>, source_name: String) -> Result<Value> {
    let cached: CachedExtraction = read_gzip_json(input.as_ref())?;
    if cached.schema_version != EXTRACTION_CACHE_SCHEMA
        || cached.extraction.pages.len() != cached.extraction.separators.len()
    {
        return Err(Error::Message("invalid extraction cache".to_owned()));
    }
    let common = CommonInput {
        source_name,
        source_sha256: cached.source_sha256,
        pages: cached.extraction.pages,
        separators: cached.extraction.separators,
    };
    let input_lines: usize = common.pages.iter().map(|page| page.lines.len()).sum();
    let mut writer = BufWriter::with_capacity(1024 * 1024, DigestWriter::default());
    let value = replay_common(common)?;
    let source_sha256 = value.source_sha256.clone();
    serde_json::to_writer_pretty(&mut writer, &Frozen(&value))?;
    writer.write_all(b"\n").expect("digest writer cannot fail");
    let writer = writer.into_inner().expect("digest writer cannot fail");
    Ok(
        json!({"input_lines": input_lines, "output_bytes": writer.bytes, "output_sha256": format!("{:x}", writer.digest.finalize()), "source_sha256": source_sha256}),
    )
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
}
