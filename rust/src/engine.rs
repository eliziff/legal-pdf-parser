use crate::structure::{derive, status, validate_document, StructureIdentity};
use crate::structure_engine::{derive_pdf_pages, PdfReplayProjection};
use legal_pdf_core::model::{
    Diagnostic, Footnote, LegalDocument, Line, Page, Paragraph, Region, Span, Word, PARSER_VERSION,
    SCHEMA_VERSION,
};
use legal_pdf_core::{read_gzip_json, write_gzip_bytes, Error, Result};
use legal_pdf_extraction::{extract_pdf, ExtractedPdf};
#[cfg(feature = "ocr")]
use legal_pdf_ocr::{OcrOptions, OcrProvider, PreparedOcrProvider};
use legal_pdf_support::{profile, project_structure};
#[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
use legal_pdf_support::{PPDocLayout, PPDocOptions};
use serde::{ser::SerializeMap, Deserialize, Serialize, Serializer};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, FileTimes};
use std::io::{BufReader, BufWriter, Read, Write};
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
        pdf_path: &Path,
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
            pdf_path,
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
    pub use_cache: bool,
    pub verified_source_sha256: Option<String>,
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
            use_cache: true,
            verified_source_sha256: None,
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

fn insert_digests(identity: &mut BTreeMap<String, Value>, digests: &[(&str, &str)]) {
    for (name, digest) in digests {
        identity.insert((*name).to_owned(), Value::String((*digest).to_owned()));
    }
}

fn engine_identity() -> &'static Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY.get_or_init(|| {
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
        insert_digests(
            &mut identity,
            &[
                (
                    "engine.rs",
                    "a464a8b25a5a9eaa1e58ee4682a7ce993ecbce589fbafcb7c7fff3a205e24362",
                ),
                (
                    "model.rs",
                    "b3c02c1c938601f2a277a5a54022fdd79b939c7f8d9cbb43361600da9f5f2b09",
                ),
                (
                    "pairing.rs",
                    "bb0bf8def5fbb3f9a12a1ebe4a8a44c3aff9fa397f14f3efa382b5a28f69ab71",
                ),
                (
                    "pairing_support.rs",
                    "45cb53bd9daae49ab58c0614cb61077695751318acc22bb9ae1d20c8ed3c5d80",
                ),
                (
                    "legal-pdf-pairing/lib.rs",
                    "cec953dcd716b02b956637a6eb0c512719387af48ac42f465608ade3ac08965e",
                ),
                (
                    "legal-pdf-core/lib.rs",
                    "73beb349a58ff8ee1727e42978cbd3d2f260f8275506058285ae5eefe67911b8",
                ),
                (
                    "legal-pdf-extraction/lib.rs",
                    "fb0c3551d6041667b17d5aadd04f65ca21df98b2258330fdc612087f90501712",
                ),
                (
                    "legal-pdf-support/lib.rs",
                    "741860d3c2ad26f4fea0daf2876949d8b4e6aeea2f94e67a131405d3b45439db",
                ),
                (
                    "legal-pdf-structure/lib.rs",
                    "48b7f3234b5de89eba0f22160cb7829417accc2f8cdc539df433562926f6fe04",
                ),
                (
                    "legal-structure/lib.rs",
                    "4cab0f76821275e5457bcf0aa9cb2c24aafc3e320b6d05f1a8b70e85599b7b22",
                ),
                (
                    "pdf.rs",
                    "5df9f637cfe2426c752389a6d13163c360b92b08d00e1d608cadd08956a8298e",
                ),
                (
                    "storage.rs",
                    "3da41897710cdaf1992703a20a08ca5466fbe385c183d15a3b372b8bff0a76e6",
                ),
                (
                    "structure.rs",
                    "0d190979fee784669e5061c404ae0ed4bc9d7aaa4470cc0fbd04445c163ef914",
                ),
                (
                    "structure-adapter.rs",
                    "cbe430bd0e0f577a32b99e715164f447348760def362ecfb51426adb5c8cea4a",
                ),
                (
                    "data/mcgill_reporters.json",
                    "946e7554e8e9134d9b148d244d825e999080dd900c666cc4cf43235fa5ec9e2f",
                ),
                (
                    "pdf-inspector/.cargo_vcs_info.json",
                    "e3d92d9d90501ff4f7b0f83b20b537789163bc833b0ad96820ea9be7049ae8fa",
                ),
            ],
        );
        #[cfg(feature = "language")]
        insert_digests(
            &mut identity,
            &[
                (
                    "grammar_tables.rs",
                    "089893ecc62a1965d9729ff67e85c632efb44944e2dd4dc1826c57d4f83506b7",
                ),
                (
                    "grammar_word.rs",
                    "e26a45bc99bcb6c7829338fd16d63ebebee29e09b11e8d0d0a1b38054c486e39",
                ),
                (
                    "legal-pdf-language/lib.rs",
                    "34d868a69379e6be5dd65331b4f8398c0821df48ca06a4c2bcb45118dcbc56e2",
                ),
                (
                    "data/legal-grammar-tables/grammar-corpus.json",
                    "8e6da9011c1cf78c609d54d53abb67b7a3e50f9a67cbf48cd72ab8136b16606f",
                ),
            ],
        );
        #[cfg(feature = "ocr")]
        insert_digests(
            &mut identity,
            &[
                (
                    "ocr.rs",
                    "077f42d364cd18fe3e401b4deb933493e69fcbaaf060e58036c6863564a97dcc",
                ),
                (
                    "legal-pdf-ocr/lib.rs",
                    "92e732ac7c66484793b62c0a2c808072a26c7892afd91043daff7a455fcf52a6",
                ),
                (
                    "separator.rs",
                    "600736a243d5ba7c22c1f5b7ef9f2dd40e871047d3f417acac293f414a1440e8",
                ),
            ],
        );
        #[cfg(feature = "kraken")]
        insert_digests(
            &mut identity,
            &[
                (
                    "kraken.rs",
                    "86b253e9680652c7e8abfd8bb28520a6abae80fa22c4b80cde1a12516752f718",
                ),
                (
                    "ort_runtime.rs",
                    "4939b2199a0b67cba01c5a80b8a906b83ac67e48181a7665f18248b4cf3a1784",
                ),
                (
                    "tesseract_layout.rs",
                    "6706a12f83cb56740028ee02f99fb37d370c992fb93b3e051b7d923c480618e6",
                ),
            ],
        );
        #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
        insert_digests(
            &mut identity,
            &[
                (
                    "ppdoc.rs",
                    "412e8dec23c786fed282d536013180a9c757e5b6d46501e2bd9635e0ee38e480",
                ),
                (
                    "ppdoc_postprocess.rs",
                    "89c4a0f0a5d2a53532668fded947553dbac3bc79d5a7ffaf20a5718fbaf25a27",
                ),
                (
                    "ppdoc_openvino.rs",
                    "dc8de55bc05c4b859de273a71db061f1baf972d6325bd2dc5ce8dd89091d317c",
                ),
            ],
        );
        #[cfg(feature = "ppdoc-full")]
        insert_digests(
            &mut identity,
            &[(
                "ort_runtime.rs",
                "4939b2199a0b67cba01c5a80b8a906b83ac67e48181a7665f18248b4cf3a1784",
            )],
        );
        #[cfg(feature = "full")]
        insert_digests(
            &mut identity,
            &[(
                "Cargo.lock",
                "90b3561eeeaad70e651c0a91ec5f5e5932c406660ca52319516c24899bedfba5",
            )],
        );
        json!(identity)
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
    let bytes = serde_json::to_vec(&value)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn build_document(
    path: &Path,
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
    let derived = derive(
        &mut extracted.pages,
        &extracted.separators,
        StructureIdentity {
            document_id: document_id.clone(),
            source_sha256: source_hash.to_owned(),
        },
    )?;
    extracted.diagnostics.extend(derived.diagnostics);
    let mut metadata = Map::new();
    metadata.insert("pdf".to_owned(), Value::Object(extracted.metadata));
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
    let mut pdf_source_map = derived.pdf_source_map;
    pdf_source_map.table_ids = extracted
        .tables
        .iter()
        .map(|table| table.id.clone())
        .collect();
    pdf_source_map.image_ids = extracted
        .images
        .iter()
        .map(|image| image.id.clone())
        .collect();
    let document = LegalDocument {
        document_id,
        source_name,
        source_sha256: source_hash.to_owned(),
        page_count: extracted.pages.len(),
        status,
        pages: extracted.pages,
        paragraphs: derived.paragraphs,
        footnotes: derived.footnotes,
        tables: extracted.tables,
        images: extracted.images,
        structure_graph: derived.structure_graph,
        pdf_source_map,
        pairing_audit: Some(derived.pairing_audit),
        diagnostics: extracted.diagnostics,
        repairs: vec![],
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

pub(crate) fn parse_pdf(path: impl AsRef<Path>, options: &ParseOptions) -> Result<LegalDocument> {
    let _profile = profile::scope("parse_pdf");
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
        .map(|value| value.eq_ignore_ascii_case("pdf"))
        != Some(true)
    {
        return Err(Error::Message(format!(
            "input must be a PDF: {}",
            path.display()
        )));
    }
    let path = fs::canonicalize(path).map_err(|source| Error::io(path, source))?;
    let source_hash = match &options.verified_source_sha256 {
        Some(value) => value.clone(),
        None => profile::measure("source_sha256", || sha256_file(&path))?,
    };
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
    #[cfg(any(feature = "ppdoc-full", feature = "ppdoc-openvino"))]
    let ppdoc_variant = ppdoc_prepared
        .as_ref()
        .map(|provider| provider.variant_id().to_owned());
    #[cfg(not(any(feature = "ppdoc-full", feature = "ppdoc-openvino")))]
    let ppdoc_variant: Option<String> = None;
    let key = cache_key(
        &source_hash,
        identity,
        ocr_identity
            .as_ref()
            .map(|provider| (provider.0.as_str(), provider.1.as_str())),
        ppdoc_identity.as_deref(),
        options.ocr_pages.as_deref(),
    )?;
    let cache_root = options
        .use_cache
        .then(|| options.cache_dir.clone().unwrap_or_else(default_cache_dir));
    if let Some(root) = &cache_root {
        let path = parse_cache_root(root)
            .join("documents")
            .join(format!("{key}.json.gz"));
        if path.is_file() {
            let cached = profile::measure("document_cache_read_decode", || {
                read_gzip_json::<LegalDocument>(&path).ok()
            });
            let cached = cached.filter(|cached| {
                cached.source_sha256 == source_hash
                    && cached.schema_version == SCHEMA_VERSION
                    && cached.parser_version == PARSER_VERSION
                    && cached
                        .provenance
                        .get("deterministic_cache_key")
                        .and_then(Value::as_str)
                        == Some(key.as_str())
                    && profile::measure("document_cache_validate", || {
                        validate_document(cached).is_ok()
                    })
            });
            if let Some(mut cached) = cached {
                profile::measure("project_structure", || project_structure(&mut cached));
                cached
                    .provenance
                    .insert("cache_hit".to_owned(), Value::Bool(true));
                cached
                    .provenance
                    .insert("cache_enabled".to_owned(), Value::Bool(true));
                touch_cache(&path);
                return Ok(cached);
            }
            let _ = fs::remove_file(path);
        }
    }
    #[cfg(feature = "ocr")]
    let mut ocr_provider = options
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
        extract_pdf(&path, selected_ocr, options.ocr_pages.as_deref())
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
                provider.annotate_pdf(&path, &mut extracted.pages)
            })?);
    }
    let mut parsed = profile::measure("build_document", || {
        build_document(
            &path,
            &source_hash,
            &key,
            identity,
            (
                ocr_identity
                    .as_ref()
                    .map(|provider| (provider.0.as_str(), provider.1.as_str())),
                ppdoc_variant.as_deref(),
                ppdoc_identity.as_deref(),
            ),
            extracted,
        )
    })?;
    parsed
        .provenance
        .insert("cache_enabled".to_owned(), Value::Bool(options.use_cache));
    profile::measure("project_structure", || project_structure(&mut parsed));
    if let Some(root) = &cache_root {
        let cache_path = parse_cache_root(root)
            .join("documents")
            .join(format!("{key}.json.gz"));
        if let Ok(bytes) = profile::measure("document_cache_serialize", || {
            serde_json::to_vec(&parsed)
        }) {
            let root = root.clone();
            std::thread::spawn(move || {
                let _profile = profile::scope("document_cache_background_write");
                let _ = profile::measure("document_cache_compress_write", || {
                    write_gzip_bytes(&cache_path, &bytes)
                });
                maybe_prune_document_cache(&root);
            });
        }
    }
    Ok(parsed)
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
        map.serialize_entry("marker_summary", &value.marker_summary)?;
        map.serialize_entry("markers", &value.markers)?;
        map.serialize_entry("pairing_summary", &value.pairing_summary)?;
        map.serialize_entry("paragraphs", &FrozenSlice(&value.paragraphs))?;
        map.serialize_entry("prepared_pages", &FrozenSlice(&value.prepared_pages))?;
        map.serialize_entry("schema_version", "legalpdf.common-input-result.v1")?;
        map.serialize_entry("source_sha256", &value.source_sha256)?;
        map.serialize_entry("status", &value.status)?;
        map.serialize_entry("validation", value.validation)?;
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
    let index = legal_pdf_structure::PdfTextIndex::from_pages(&document.pages);
    let text = index.text().chars().collect::<Vec<_>>();
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
                let extent = document
                    .pdf_source_map
                    .nodes
                    .iter()
                    .find(|extent| extent.id == node.id);
                let end = node.range.end.min(text.len());
                let start = node.range.start.min(end);
                let value = text[start..end]
                    .iter()
                    .collect::<String>()
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
                    "page_indexes": extent.map(|value| &value.page_indexes).unwrap_or(&node.page_indexes),
                    "line_ids": extent.map(|value| &value.line_ids).unwrap_or(&node.line_ids),
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
        Path::new(&source_name),
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
