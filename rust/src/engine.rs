use crate::structure_engine::{derive_pdf_pages, PdfReplayProjection};
use legal_pdf_core::model::{
    Diagnostic, Footnote, LegalDocument, Line, Page, Paragraph, Region, Span, Word, PARSER_VERSION,
    SCHEMA_VERSION,
};
use legal_pdf_core::{read_gzip_json, write_gzip_json, Error, Result};
use legal_pdf_extraction::{extract_pdf, ExtractedPdf};
#[cfg(feature = "ocr")]
use legal_pdf_ocr::{OcrOptions, OcrProvider, PreparedOcrProvider};
use legal_pdf_structure::{
    derive, status, validate_document, validate_pdf_components, StructureIdentity, StructureOutput,
};
use legal_pdf_support::{profile, PdfDocument};
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
use legal_pdf_support::{PPDocLayout, PPDocOptions};
use legal_structure::{ScalarText, DOCUMENT_STRUCTURE_SCHEMA};
use serde::{ser::SerializeMap, Deserialize, Serialize, Serializer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, FileTimes};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};
use std::time::SystemTime;

const EXTRACTION_CACHE_SCHEMA: &str = "legalpdf.extraction-cache.v1";
const PARSE_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;

#[cfg(feature = "ocr")]
struct DeferredOcrProvider<'a> {
    options: &'a OcrOptions,
    prepared: Option<PreparedOcrProvider>,
    runtime: Option<OcrProvider>,
}

#[cfg(feature = "ocr")]
impl legal_pdf_core::PdfOcrProvider for DeferredOcrProvider<'_> {
    fn extract_pages(
        &mut self,
        pdf: &[u8],
        requests: &[legal_pdf_core::OcrPageRequest],
    ) -> Result<Vec<legal_pdf_core::OcrPageResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        if self.runtime.is_none() {
            let prepared = self.prepared.take().ok_or_else(|| {
                Error::Message("OCR runtime preparation was already consumed".to_owned())
            })?;
            self.runtime = Some(profile::measure("provider_runtime_ocr", || {
                OcrProvider::from_prepared(self.options, prepared)
            })?);
        }
        legal_pdf_core::PdfOcrProvider::extract_pages(
            self.runtime.as_mut().expect("OCR runtime initialized"),
            pdf,
            requests,
        )
    }
}

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
    pub cache_key: Option<String>,
    pub require_cache_write: bool,
    pub use_cache: bool,
    pub expected_source_sha256: Option<String>,
    pub max_output_bytes: Option<usize>,
    #[cfg(feature = "ocr")]
    pub ocr: Option<OcrOptions>,
    /// Zero-based pages eligible for OCR. Native extraction still inspects the
    /// complete PDF so physical page numbering stays authoritative.
    pub ocr_pages: Option<Vec<usize>>,
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    pub ppdoc: Option<PPDocOptions>,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            cache_dir: None,
            cache_key: None,
            require_cache_write: false,
            use_cache: true,
            expected_source_sha256: None,
            max_output_bytes: None,
            #[cfg(feature = "ocr")]
            ocr: None,
            ocr_pages: None,
            #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
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

fn validate_output_size(document: &PdfDocument, options: &ParseOptions) -> Result<()> {
    if options
        .max_output_bytes
        .is_some_and(|limit| document.structure().query_text().len() > limit)
    {
        return Err(Error::Message(
            "PDF document text exceeds the read limit".to_owned(),
        ));
    }
    Ok(())
}

fn engine_identity() -> &'static Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        json!({
            "engine": "legal-pdf-parser-rust",
            "engine_version": env!("CARGO_PKG_VERSION"),
            "parser_version": PARSER_VERSION,
            "native_extractor": "pdf-inspector",
            "ocr_renderer": if cfg!(feature = "ocr") { "hayro 0.7.1" } else { "disabled" },
            "engine_source_sha256": env!("LEGAL_PDF_ENGINE_SHA256"),
            "structure_source_sha256": legal_structure::ENGINE_SOURCE_SHA256,
        })
    })
}

fn cache_key(
    source_hash: &str,
    identity: &Value,
    ocr_provider: Option<(&str, &str)>,
    ppdoc_identity: Option<&str>,
    ocr_pages: Option<&[usize]>,
) -> Result<String> {
    let value = json!({
        "source_sha256": source_hash,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "engine_code": identity,
        "ocr_provider": ocr_provider.map(|provider| provider.0),
        "ocr_provider_identity": ocr_provider.map(|provider| provider.1),
        "layout_provider": ppdoc_identity.map(|_| "ppdoc-lite"),
        "layout_provider_identity": ppdoc_identity,
        "ocr_pages": ocr_pages,
    });
    serialization_sha256(&value)
}

fn derive_extracted(
    extracted: &mut ExtractedPdf,
    document_id: &str,
    source_hash: &str,
) -> Result<StructureOutput> {
    let mut derived = derive(
        &mut extracted.pages,
        &extracted.separators,
        StructureIdentity {
            document_id: document_id.to_owned(),
            source_sha256: source_hash.to_owned(),
        },
    )?;
    extracted.diagnostics.append(&mut derived.diagnostics);
    Ok(derived)
}

fn build_document(
    source_name: String,
    source_hash: &str,
    key: &str,
    identity: &Value,
    providers: (Option<(&str, &str)>, Option<&str>, Option<&str>),
    mut extracted: ExtractedPdf,
) -> Result<LegalDocument> {
    let (ocr_provider, layout_variant, layout_identity) = providers;
    let provider_name = ocr_provider.map(|provider| provider.0.to_owned());
    let provider_identity = ocr_provider.map(|provider| provider.1.to_owned());
    let document_id = format!("doc-{}", &source_hash[..20]);
    let derived = derive_extracted(&mut extracted, &document_id, source_hash)?;
    let mut metadata = Map::new();
    metadata.insert("pdf".to_owned(), serde_json::to_value(extracted.metadata)?);
    let mut provenance = Map::new();
    provenance.insert("engine".to_owned(), Value::String("legalpdf".to_owned()));
    provenance.insert(
        "native_extractor".to_owned(),
        Value::String("pdf-inspector".to_owned()),
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
    provenance.insert(
        "deterministic_cache_key".to_owned(),
        Value::String(key.to_owned()),
    );
    provenance.insert("engine_code".to_owned(), identity.clone());
    let status = status(&extracted.diagnostics, &extracted.pages);
    let document = LegalDocument {
        document_id,
        source_name,
        source_sha256: source_hash.to_owned(),
        page_count: extracted.pages.len(),
        status,
        pages: extracted.pages,
        paragraphs: derived.paragraphs,
        footnotes: derived.footnotes,
        structure_graph: derived.structure_graph,
        diagnostics: extracted.diagnostics,
        metadata,
        provenance,
        schema_version: SCHEMA_VERSION.to_owned(),
        parser_version: PARSER_VERSION.to_owned(),
    };
    profile::measure("build.validate_document", || validate_document(&document))?;
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

fn prune_document_cache(root: &Path) {
    let mut files = fs::read_dir(parse_cache_root(root).join("documents"))
        .into_iter()
        .flatten()
        .flatten()
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

fn maybe_prune_document_cache(root: &Path) {
    static WRITES: AtomicUsize = AtomicUsize::new(0);
    // ponytail: amortized maintenance; rescan sooner if cache overshoot becomes material.
    if WRITES.fetch_add(1, Ordering::Relaxed) % 32 == 0 {
        prune_document_cache(root);
    }
}

fn cached_document(root: &Path, key: &str, source_hash: &str) -> Option<PdfDocument> {
    let path = parse_cache_root(root)
        .join("documents")
        .join(format!("{key}.json.gz"));
    if !path.is_file() {
        return None;
    }
    let cached = profile::measure("document_cache_read_decode", || {
        read_gzip_json::<PdfDocument>(&path).ok()
    });
    let cached = match cached.filter(|cached| {
        cached.structure().schema_version.as_ref() == DOCUMENT_STRUCTURE_SCHEMA
            && cached.structure().source_sha256.as_deref() == Some(source_hash)
            && cached.summary().sha256 == source_hash
            && cached.summary().parser_version == PARSER_VERSION
            && cached.summary().cache_key == key
            && cached.summary().page_count == cached.page_count()
            && cached.summary().projection_page_count == cached.page_count()
    }) {
        Some(cached) => cached,
        None => {
            let _ = fs::remove_file(path);
            return None;
        }
    };
    touch_cache(&path);
    Some(cached)
}

pub(crate) fn parse_pdf(
    bytes: Option<&[u8]>,
    options: &ParseOptions,
) -> Result<Option<PdfDocument>> {
    let _profile = profile::scope("parse_pdf");
    let source_hash = if let Some(bytes) = bytes {
        if bytes.is_empty()
            || bytes.len() > 100 * 1024 * 1024
            || !bytes[..bytes.len().min(1024)]
                .windows(5)
                .any(|window| window == b"%PDF-")
        {
            return Err(Error::Message("PDF source bytes are invalid".to_owned()));
        }
        let actual = profile::measure("source_sha256", || format!("{:x}", Sha256::digest(bytes)));
        if options
            .expected_source_sha256
            .as_deref()
            .is_some_and(|expected| expected != actual)
        {
            return Err(Error::Message(
                "PDF source changed after preparation began".to_owned(),
            ));
        }
        actual
    } else {
        options.expected_source_sha256.clone().ok_or_else(|| {
            Error::Message("PDF cache lookup requires expected_source_sha256".to_owned())
        })?
    };
    let cache_root = options
        .use_cache
        .then(|| options.cache_dir.clone().unwrap_or_else(default_cache_dir));
    if let Some(key) = options.cache_key.as_deref() {
        let cached = cache_root
            .as_deref()
            .and_then(|root| cached_document(root, key, &source_hash));
        if let Some(document) = &cached {
            validate_output_size(document, options)?;
        }
        return Ok(cached);
    }
    let identity = engine_identity();
    #[cfg(feature = "ocr")]
    let mut ocr_prepared = profile::measure("provider_identity_ocr", || {
        options.ocr.as_ref().map(OcrProvider::prepare).transpose()
    })?;
    #[cfg(feature = "ocr")]
    let ocr_identity = ocr_prepared
        .as_ref()
        .map(|provider| (provider.name().to_owned(), provider.identity().to_owned()));
    #[cfg(not(feature = "ocr"))]
    let ocr_identity: Option<(String, String)> = None;
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    let mut ppdoc_prepared = profile::measure("provider_identity_ppdoc", || {
        options.ppdoc.as_ref().map(PPDocLayout::prepare).transpose()
    })?;
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    let ppdoc_identity = ppdoc_prepared
        .as_ref()
        .map(|provider| provider.identity().to_owned());
    #[cfg(not(any(feature = "ppdoc-full", feature = "ppdoc-openvino")))]
    let ppdoc_identity: Option<String> = None;
    let key = cache_key(
        &source_hash,
        identity,
        ocr_identity
            .as_ref()
            .map(|provider| (provider.0.as_str(), provider.1.as_str())),
        ppdoc_identity.as_deref(),
        options.ocr_pages.as_deref(),
    )?;
    if let Some(root) = &cache_root {
        if let Some(cached) = cached_document(root, &key, &source_hash) {
            validate_output_size(&cached, options)?;
            return Ok(Some(cached));
        }
    }
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    #[cfg(feature = "ocr")]
    let mut ocr_provider =
        options
            .ocr
            .as_ref()
            .zip(ocr_prepared.take())
            .map(|(options, prepared)| DeferredOcrProvider {
                options,
                prepared: Some(prepared),
                runtime: None,
            });
    #[cfg(feature = "ocr")]
    let selected_ocr = ocr_provider
        .as_mut()
        .map(|provider| provider as &mut dyn legal_pdf_core::PdfOcrProvider);
    #[cfg(not(feature = "ocr"))]
    let selected_ocr = None;
    let mut extracted = profile::measure("extract_pdf", || {
        extract_pdf(bytes, selected_ocr, options.ocr_pages.as_deref())
    })?;
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    let mut ppdoc = profile::measure("provider_runtime_ppdoc", || {
        options
            .ppdoc
            .as_ref()
            .zip(ppdoc_prepared.take())
            .map(|(options, prepared)| PPDocLayout::from_prepared(options, prepared))
            .transpose()
    })?;
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    if let Some(provider) = ppdoc.as_mut() {
        extracted
            .diagnostics
            .extend(profile::measure("ppdoc_annotate", || {
                provider.annotate_pdf(bytes, &mut extracted.pages)
            })?);
    }
    let document_id = format!("doc-{}", &source_hash[..20]);
    let derived = profile::measure("derive_document", || {
        derive_extracted(&mut extracted, &document_id, &source_hash)
    })?;
    profile::measure("build.validate_document", || {
        validate_pdf_components(
            &document_id,
            &source_hash,
            &extracted.pages,
            &derived.paragraphs,
            &derived.footnotes,
            &derived.structure_graph,
        )
    })?;
    let document_status = status(&extracted.diagnostics, &extracted.pages);
    let parsed = profile::measure("project_document", || {
        PdfDocument::from_parts(
            &source_hash,
            &key,
            document_status,
            extracted.metadata,
            extracted.pages,
            derived.paragraphs,
            derived.footnotes,
            derived.structure_graph,
        )
    });
    validate_output_size(&parsed, options)?;
    if let Some(root) = &cache_root {
        let cache_path = parse_cache_root(root)
            .join("documents")
            .join(format!("{key}.json.gz"));
        match profile::measure("document_cache_write", || {
            write_gzip_json(&cache_path, &parsed)
        }) {
            Ok(()) => maybe_prune_document_cache(root),
            Err(error) if options.require_cache_write => return Err(error),
            Err(_) => {}
        }
    }
    Ok(Some(parsed))
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
frozen_type!(Footnote: body body_line_ids body_pages confidence crossrefs label occurrence pair_id passage_since_prior_note provenance reference_line_id reference_page restart_sequence sentence_proposition warnings);
frozen_type!(Diagnostic: code details line_ids message page_index severity);
struct FrozenReplay<'a>(&'a PdfReplayProjection);

impl Serialize for FrozenReplay<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        let value = self.0;
        let mut map = serializer.serialize_map(None)?;
        map.serialize_entry("derived_pages", &FrozenSlice(&value.derived_pages))?;
        map.serialize_entry("diagnostics", &FrozenSlice(&value.diagnostics))?;
        map.serialize_entry("footnotes", &FrozenSlice(&value.footnotes))?;
        map.serialize_entry("marker_summary", &value.pairing_audit.pairing_summary)?;
        map.serialize_entry("markers", &value.pairing_audit.markers)?;
        map.serialize_entry("pairing_summary", &value.pairing_audit.pairing_summary)?;
        map.serialize_entry("paragraphs", &FrozenSlice(&value.paragraphs))?;
        map.serialize_entry("prepared_pages", &FrozenSlice(&value.prepared_pages))?;
        map.serialize_entry("schema_version", "legalpdf.common-input-result.v1")?;
        map.serialize_entry("source_sha256", &value.source_sha256)?;
        map.serialize_entry("status", &value.status)?;
        map.serialize_entry("validation", "ok")?;
        map.end()
    }
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

fn serialization_sha256(value: &impl Serialize) -> Result<String> {
    let mut writer = DigestWriter::default();
    serde_json::to_writer(&mut writer, value)?;
    Ok(format!("{:x}", writer.digest.finalize()))
}

fn structure_examples(document: &LegalDocument) -> Value {
    let text = ScalarText::new(&document.structure_graph.text);
    let mut examples = BTreeMap::<String, Vec<Value>>::new();
    for kind in [
        legal_structure::NodeKind::Heading,
        legal_structure::NodeKind::Section,
        legal_structure::NodeKind::Paragraph,
        legal_structure::NodeKind::ListItem,
        legal_structure::NodeKind::Footnote,
        legal_structure::NodeKind::Endnote,
    ] {
        let nodes = document
            .structure_graph
            .nodes
            .iter()
            .filter(|node| node.kind == kind)
            .collect::<Vec<_>>();
        if nodes.is_empty() {
            continue;
        }
        let mut slots = vec![0, nodes.len() / 2, nodes.len() - 1];
        slots.sort_unstable();
        slots.dedup();
        let values = slots
            .into_iter()
            .map(|slot| {
                let node = nodes[slot];
                let value = text
                    .slice_utf16(node.range.start..node.range.end)
                    .unwrap_or_default()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .chars()
                    .take(240)
                    .collect::<String>();
                json!({
                    "id": node.id,
                    "text": value,
                    "label": node.label,
                    "locator_kind": node.locator_kind,
                    "parent_id": node.parent_id,
                    "page_indexes": node.page_indexes,
                    "line_ids": node.line_ids,
                    "rule": node.proof.as_ref().map(|proof| proof.rule),
                })
            })
            .collect();
        let name = serde_json::to_value(kind)
            .ok()
            .and_then(|value| value.as_str().map(str::to_owned))
            .unwrap_or_else(|| "unknown".to_owned());
        examples.insert(name, values);
    }
    json!(examples)
}

#[doc(hidden)]
pub fn corpus_check_cached_extraction(
    input: impl AsRef<Path>,
    source_name: String,
) -> Result<Value> {
    let cached: CachedExtraction = read_gzip_json(input.as_ref())?;
    if cached.schema_version != EXTRACTION_CACHE_SCHEMA
        || cached.source_sha256.len() != 64
        || cached.extraction.pages.len() != cached.extraction.separators.len()
    {
        return Err(Error::Message("invalid extraction cache".to_owned()));
    }
    let page_count = cached.extraction.pages.len();
    let line_count = cached
        .extraction
        .pages
        .iter()
        .map(|page| page.lines.len())
        .sum::<usize>();
    let source_sha256 = cached.source_sha256.clone();
    let cache_key = cached.cache_key.clone();
    let document = build_document(
        source_name,
        &source_sha256,
        &cache_key,
        engine_identity(),
        (None, None, None),
        cached.extraction,
    )?;
    let mut by_kind = BTreeMap::<String, usize>::new();
    let mut sections_by_locator_kind = BTreeMap::<String, usize>::new();
    for node in &document.structure_graph.nodes {
        let kind = serde_json::to_value(node.kind)?
            .as_str()
            .unwrap_or("unknown")
            .to_owned();
        *by_kind.entry(kind).or_default() += 1;
        if node.kind == legal_structure::NodeKind::Section {
            *sections_by_locator_kind
                .entry(
                    node.locator_kind
                        .clone()
                        .unwrap_or_else(|| "unknown".to_owned()),
                )
                .or_default() += 1;
        }
    }
    let proofs = document
        .structure_graph
        .nodes
        .iter()
        .filter_map(|node| node.proof.as_ref().map(|proof| (node.id.as_str(), proof)))
        .collect::<Vec<_>>();
    let mut by_rule = BTreeMap::<String, usize>::new();
    for (_, proof) in &proofs {
        let rule = serde_json::to_value(proof.rule)?
            .as_str()
            .unwrap_or("unknown")
            .to_owned();
        *by_rule.entry(rule).or_default() += 1;
    }
    let abstentions = document
        .structure_graph
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "structure_run_abstained")
        .count();
    let partial_resolutions = document
        .structure_graph
        .diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == "structure_run_partially_resolved")
        .count();
    let heading_derived_sections = document
        .structure_graph
        .nodes
        .iter()
        .filter(|node| {
            node.kind == legal_structure::NodeKind::Section
                && node.grammar.as_deref() == Some("accepted_heading")
        })
        .count();
    let mut diagnostics_by_code = BTreeMap::<String, usize>::new();
    for diagnostic in &document.structure_graph.diagnostics {
        *diagnostics_by_code
            .entry(diagnostic.code.clone())
            .or_default() += 1;
    }
    let structure = json!({
        "schema_version": document.structure_graph.schema_version,
        "node_count": document.structure_graph.nodes.len(),
        "note_count": document.structure_graph.notes.len(),
        "diagnostic_count": document.structure_graph.diagnostics.len(),
        "by_kind": by_kind,
        "by_rule": by_rule,
        "sections_by_locator_kind": sections_by_locator_kind,
        "diagnostics_by_code": diagnostics_by_code,
        "heading_derived_section_count": heading_derived_sections,
        "abstention_count": abstentions,
        "partial_resolution_count": partial_resolutions,
        "graph_sha256": serialization_sha256(&document.structure_graph)?,
        "proofs_sha256": serialization_sha256(&proofs)?,
        "examples": structure_examples(&document),
    });
    Ok(json!({
        "source_sha256": source_sha256,
        "page_count": page_count,
        "line_count": line_count,
        "product_sha256": serialization_sha256(&document)?,
        "structure": structure,
    }))
}

#[doc(hidden)]
pub fn digest_cached_extraction(input: impl AsRef<Path>, source_name: String) -> Result<Value> {
    let cached: CachedExtraction = read_gzip_json(input.as_ref())?;
    if cached.schema_version != EXTRACTION_CACHE_SCHEMA
        || cached.extraction.pages.len() != cached.extraction.separators.len()
    {
        return Err(Error::Message("invalid extraction cache".to_owned()));
    }
    let input_lines: usize = cached
        .extraction
        .pages
        .iter()
        .map(|page| page.lines.len())
        .sum();
    let document_id = format!(
        "doc-{}",
        cached
            .source_sha256
            .get(..20)
            .ok_or_else(|| Error::Message("cached source SHA-256 is invalid".to_owned()))?
    );
    let value = derive_pdf_pages(
        document_id,
        cached.source_sha256,
        cached.extraction.pages,
        cached.extraction.separators,
    )
    .map_err(|error| Error::Message(error.message))?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, DigestWriter::default());
    let source_sha256 = value.source_sha256.clone();
    let _ = source_name;
    let structure = json!({
        "paragraph_count": value.paragraphs.len(),
        "paragraphs_sha256": serialization_sha256(&FrozenSlice(&value.paragraphs))?,
        "graph_sha256": serialization_sha256(&value.structure_graph)?,
    });
    serde_json::to_writer_pretty(&mut writer, &FrozenReplay(&value))?;
    writer.write_all(b"\n").expect("digest writer cannot fail");
    let writer = writer.into_inner().expect("digest writer cannot fail");
    Ok(
        json!({"input_lines": input_lines, "output_bytes": writer.bytes, "output_sha256": format!("{:x}", writer.digest.finalize()), "source_sha256": source_sha256, "structure": structure}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_identity_is_stable() {
        let identity = engine_identity();
        assert_eq!(
            cache_key(&"00".repeat(32), identity, None, None, None).unwrap(),
            cache_key(&"00".repeat(32), identity, None, None, None).unwrap()
        );
    }
}
