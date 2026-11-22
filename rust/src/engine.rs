use crate::{PdfRequest, RecognitionProgress};
use legal_pdf_core::model::{PARSER_VERSION, SCHEMA_VERSION};
use legal_pdf_core::{read_gzip_json, read_gzip_packed, write_gzip_json, write_gzip_packed, Error, FailureKind, Result};
use legal_pdf_extraction_processor::{extract_pdf, recognize_pdf, ExtractedPdf};
#[cfg(feature = "ocr")]
use legal_pdf_ocr::{OcrOptions, OcrProvider, PreparedOcrProvider};
use legal_pdf_structure::{Doc, StructureIdentity, StructureOutput};
use legal_pdf_support::{profile, PdfDocument, PdfStatus};
#[cfg(feature = "ppdoc")]
use legal_pdf_support::PPDocLayout;
use legal_structure_model::{serialization_sha256, DetectionProfile, DocumentStructure, DOCUMENT_STRUCTURE_SCHEMA};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, FileTimes};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    OnceLock,
};
use std::time::SystemTime;

// v4: a blank page (no text, no image) is not a page needing recognition.
// v5: text painted inside a clip of no area is hidden, and no longer read.
#[doc(hidden)]
pub const EXTRACTION_CACHE_SCHEMA: &str = "legalpdf.extraction-cache.v5";
/// What the layout model's reading (PP-DocLayout) gives, as its caches are keyed: raised when it changes.
const LAYOUT_VERSION: &str = "1";
/// How a page is drawn for recognition, as the rasters cached for it are keyed: raised when the drawing changes.
const RASTER_VERSION: &str = "1";
/// The recognizer's code a page's reading is cached under, alongside its provider's identity: a version, as extraction's
/// schema is, never a fingerprint of source files, which discarded every recognized page on edits that read no page
/// differently. Its value is the fingerprint the cache was written under (legal-pdf-parser f67cff9); a change to how a
/// page is read gives it a new one.
const RECOGNITION_CODE: &str = "a15fdddbf3186a9c7e19d1d5d524bf730b92ce19ad6ad2c17d108cd21b1a2d7b";
const PARSE_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;

#[cfg(feature = "ocr")]
struct DeferredOcrProvider<'a> {
    options: &'a OcrOptions,
    prepared: Option<PreparedOcrProvider>,
    runtime: Option<OcrProvider>,
    cache_root: Option<PathBuf>,
    cache_identity: String,
    /// What the raster measures are cached under: this source and the raster code, not recognition.
    raster_identity: String,
    /// The page last rendered for regions read again, by page and scale: a page's regions share it.
    region_render: Option<(usize, i64, legal_pdf_ocr::GrayRender)>,
    progress: RecognitionProgress<'a>,
    /// What each measured page's render shows, by page.
    rasters: BTreeMap<usize, legal_pdf_core::RasterReading>,
}

#[cfg(feature = "ocr")]
#[derive(Serialize, Deserialize)]
struct CachedRaster {
    key: String,
    #[serde(flatten)]
    reading: legal_pdf_core::RasterReading,
}

#[cfg(feature = "ocr")]
#[derive(Serialize, Deserialize)]
struct CachedOcrPage {
    key: String,
    page: legal_pdf_core::OcrPageResult,
}

#[cfg(feature = "ocr")]
#[derive(Serialize, Deserialize)]
struct CachedRegion {
    key: String,
    lines: Vec<legal_pdf_core::OcrLine>,
}

#[cfg(feature = "ocr")]
fn cached_rasters(
    root: Option<&Path>,
    identity: &str,
    pdf: &[u8],
    requests: &[legal_pdf_core::OcrPageRequest],
) -> Vec<CachedRaster> {
    let keys = requests.iter().map(|request| {
        serialization_sha256(&("raster-v4", identity, request.page_index, request.width, request.height, &request.lines, &request.words))
    }).collect::<Vec<_>>();
    let mut found = keys.iter().map(|key| {
        root.and_then(|root| read_gzip_json::<CachedRaster>(&root.join(format!("{key}.json.gz"))).ok())
            .filter(|cached| &cached.key == key)
    }).collect::<Vec<_>>();
    let missing = (0..requests.len()).filter(|&n| found[n].is_none()).collect::<Vec<_>>();
    if !missing.is_empty() {
        let wanted = missing.iter().map(|&n| requests[n].clone()).collect::<Vec<_>>();
        for (&n, reading) in missing.iter().zip(legal_pdf_ocr::raster_page_rules(pdf, &wanted)) {
            let cached = CachedRaster { key: keys[n].clone(), reading };
            if let Some(root) = root {
                let _ = write_gzip_json(&root.join(format!("{}.json.gz", keys[n])), &cached);
            }
            found[n] = Some(cached);
        }
    }
    found.into_iter().map(|cached| cached.unwrap_or(CachedRaster { key: String::new(), reading: Default::default() })).collect()
}

#[cfg(feature = "ocr")]
impl DeferredOcrProvider<'_> {
    /// The recognizer, made from its preparation the first time it is asked for.
    fn runtime(&mut self) -> Result<&mut OcrProvider> {
        if self.runtime.is_none() {
            let prepared = self.prepared.take().ok_or_else(|| Error::Message("OCR runtime preparation was already consumed".to_owned()))?;
            self.runtime = Some(profile::measure("provider_runtime_ocr", || OcrProvider::from_prepared(self.options, prepared))
                .map_err(|error| error.failure(FailureKind::OcrRuntime))?);
        }
        Ok(self.runtime.as_mut().expect("OCR runtime initialized"))
    }
}

#[cfg(feature = "ocr")]
impl legal_pdf_core::PdfOcrProvider for DeferredOcrProvider<'_> {
    fn extract_pages(&mut self, pdf: &[u8], requests: &[legal_pdf_core::OcrPageRequest]) -> Result<Vec<legal_pdf_core::OcrPageResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let (mut results, mut missing, mut paths) = (Vec::new(), Vec::new(), BTreeMap::new());
        for request in requests {
            let key = serialization_sha256(&(&self.cache_identity, request.page_index, request.width, request.height, &request.regions));
            let path = self.cache_root.as_ref().map(|root| root.join(format!("{key}.json.gz")));
            let cached = path
                .as_ref()
                .and_then(|path| read_gzip_json::<CachedOcrPage>(path).ok())
                .filter(|cached| cached.key == key && cached.page.page_index == request.page_index);
            if let Some(cached) = cached {
                if let Some(path) = &path {
                    touch_cache(path);
                }
                results.push(cached.page);
            } else {
                missing.push(request.clone());
                paths.insert(request.page_index, (key, path));
            }
        }
        let progress = self.progress;
        let report = |done: usize| {
            if let Some(progress) = progress {
                progress(done, requests.len());
            }
        };
        report(results.len());
        if missing.is_empty() {
            return Ok(results);
        }
        self.runtime()?;
        // Progress is reported per window of 4, 8 and then 16 pages, one runtime throughout.
        let mut windows = Vec::new();
        let mut rest = missing.as_slice();
        let mut size = if progress.is_some() { 4 } else { rest.len() };
        while !rest.is_empty() {
            let (window, after) = rest.split_at(size.min(rest.len()));
            windows.push(window);
            rest = after;
            size = (size * 2).min(16);
        }
        let raster_root = self.cache_root.as_deref().and_then(Path::parent).map(|root| root.join("raster"));
        let raster_requests = missing.iter().filter(|request| request.regions.is_empty()).cloned().collect::<Vec<_>>();
        let raster_identity = self.raster_identity.clone();
        std::thread::scope(|scope| -> Result<()> {
            // Keep the exact 150 dpi scanner, overlapping its existing cache fill with recognition.
            let raster = raster_root.map(|root| scope.spawn(move || cached_rasters(Some(&root), &raster_identity, pdf, &raster_requests)));
            for window in windows {
                let recognized = legal_pdf_core::PdfOcrProvider::extract_pages(self.runtime.as_mut().expect("OCR runtime initialized"), pdf, window)
                    .map_err(|error| error.failure(FailureKind::OcrRuntime))?;
                for page in recognized {
                    let Some((key, path)) = paths.remove(&page.page_index) else {
                        return Err(Error::Message("OCR returned an unexpected or repeated page".to_owned()));
                    };
                    if let Some(path) = path {
                        write_gzip_json(&path, &CachedOcrPage { key, page: page.clone() })?;
                    }
                    results.push(page);
                }
                report(results.len());
            }
            if let Some(raster) = raster {
                raster.join().map_err(|_| Error::Message("raster scan worker panicked".to_owned()))?;
            }
            Ok(())
        })?;
        if !paths.is_empty() {
            return Err(Error::Message("OCR did not return every requested page".to_owned()));
        }
        results.sort_by_key(|page| page.page_index);
        Ok(results)
    }

    /// A region read again, through the recognition cache as a page reading is: the same OCR code
    /// and provider read the same region of the same page the same way.
    fn read_region(&mut self, pdf: &[u8], page_index: usize, region: [f64; 4], scale: f64) -> Result<Vec<legal_pdf_core::OcrLine>> {
        let key = serialization_sha256(&("region-v1", &self.cache_identity, page_index, region.map(|value| (value * 100.0).round() as i64), (scale * 100.0).round() as i64));
        let path = self.cache_root.as_ref().map(|root| root.join(format!("region-{key}.json.gz")));
        if let Some(cached) = path.as_ref().and_then(|path| read_gzip_json::<CachedRegion>(path).ok()).filter(|cached| cached.key == key) {
            return Ok(cached.lines);
        }
        self.runtime()?;
        let scaled = (scale * 100.0).round() as i64;
        if self.region_render.as_ref().is_none_or(|(page, at, _)| (*page, *at) != (page_index, scaled)) {
            self.region_render = legal_pdf_ocr::render_gray_page(pdf, page_index, scale * 72.0).map(|render| (page_index, scaled, render));
        }
        let Some((_, _, render)) = &self.region_render else {
            return Ok(Vec::new());
        };
        let lines = self.runtime.as_mut().expect("OCR runtime initialized").read_rendered_region(render, region, scale)
            .map_err(|error| error.failure(FailureKind::OcrRuntime))?;
        if let Some(path) = &path {
            write_gzip_json(path, &CachedRegion { key, lines: lines.clone() })?;
        }
        Ok(lines)
    }

    fn separators(
        &mut self,
        pdf: &[u8],
        requests: &[legal_pdf_core::OcrPageRequest],
    ) -> Vec<Option<f64>> {
        profile::measure("extract.raster_separators", || {
            let root = self.cache_root.as_ref().and_then(|root| root.parent()).map(|parse| parse.join("raster"));
            let found = cached_rasters(root.as_deref(), &self.raster_identity, pdf, requests);
            requests
                .iter()
                .zip(found)
                .map(|(request, cached)| {
                    let separator = cached.reading.separator;
                    self.rasters.insert(request.page_index, cached.reading);
                    separator
                })
                .collect()
        })
    }
}

#[doc(hidden)]
#[derive(Serialize, Deserialize)]
pub struct CachedExtraction<T = ExtractedPdf> {
    pub schema_version: String,
    pub source_sha256: String,
    pub extraction: T,
}

fn default_cache_dir() -> PathBuf {
    if let Some(root) = std::env::var_os("OPEN_LEGAL_DATA_HOME").filter(|value| !value.is_empty()) {
        return PathBuf::from(root).join("apps/legalpdf/cache");
    }
    if cfg!(windows) {
        let base = std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        return base.join("OpenLegalData/apps/legalpdf/cache");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("OpenLegalData/legalpdf")
}

fn validate_output_size(document: &PdfDocument, request: &PdfRequest) -> Result<()> {
    match request.max_output_bytes {
        Some(limit) if document.structure().query_text().len() > limit => Err(Error::Message("PDF document text exceeds the read limit".to_owned())),
        _ => Ok(()),
    }
}

fn document_identity() -> &'static Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        let mut identity = json!({
            "parse_version": legal_pdf_structure::PARSE_VERSION,
            "extraction_schema": EXTRACTION_CACHE_SCHEMA,
            "structure_version": legal_structure::ENGINE_VERSION,
        });
        // TEMP (remove before landing): a reading made with LEGALPDF_LEVERS set is cached apart from every other, under
        // what it is set to; unset, the key is the one it was.
        if let Ok(levers) = std::env::var("LEGALPDF_LEVERS") {
            identity["levers"] = json!(levers);
        }
        identity
    })
}

fn cache_key(
    source_hash: &str,
    identity: &Value,
    ocr_provider: Option<(&str, &str)>,
    ppdoc_identity: Option<&str>,
    ocr_pages: Option<&[usize]>,
    semantic_profile: Option<DetectionProfile>,
    doc_identity: Option<&str>,
) -> String {
    let mut value = json!({
        "source_sha256": source_hash,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "engine_code": identity,
        "ocr_provider": ocr_provider.map(|provider| provider.0),
        "ocr_provider_identity": ocr_provider.map(|provider| provider.1),
        "ocr_source_sha256": ocr_provider
            .filter(|provider| provider.0 != "supplied")
            .map(|_| RECOGNITION_CODE),
        "layout_provider": ppdoc_identity.map(|_| "ppdoc-lite"),
        "layout_provider_identity": ppdoc_identity,
        "layout_version": ppdoc_identity.map(|_| LAYOUT_VERSION),
        "ocr_pages": ocr_pages,
        "profile": semantic_profile,
    });
    // A request naming DOC: its stages and the modes it reads in. One naming none keeps the key it had.
    if let Some(identity) = doc_identity {
        value["structure_provider"] = json!({ "doc": identity, "modes": DOC_PROMOTED });
    }
    serialization_sha256(&value)
}

/// The modes ("born-digital", or "recognized" where recognition read a page) whose census promoted DOC, the structure
/// stage's tier 1, over tier 0.
const DOC_PROMOTED: &[&str] = &["born-digital", "recognized"];

/// What a request's DOC is keyed by: its pure-Rust bundle (`doc.safetensors`), which each fit writes anew.
fn doc_identity(doc: &crate::contract::DocOptions) -> Result<String> {
    let path = if doc.model_pack.is_file() {
        doc.model_pack.clone()
    } else {
        doc.model_pack.join("doc.safetensors")
    };
    fs::read(&path).map(|bundle| Sha256::digest(bundle).iter().map(|byte| format!("{byte:02x}")).collect::<String>()).map_err(|source| Error::Io { path, source })
}

/// DOC's stages loaded once for every parse that reads with them, by their bundle path: an app's parse reads one
/// document at a time, on every core.
fn loaded_doc(doc: &crate::contract::DocOptions) -> Result<std::sync::Arc<Doc>> {
    type Loaded = Vec<(PathBuf, std::sync::Arc<Doc>)>;
    static LOADED: std::sync::Mutex<Loaded> = std::sync::Mutex::new(Vec::new());
    let mut loaded = LOADED.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some((_, reader)) = loaded.iter().find(|(known, _)| *known == doc.model_pack) {
        return Ok(std::sync::Arc::clone(reader));
    }
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(8);
    let reader = std::sync::Arc::new(Doc::load(&doc.model_pack, None, false, threads)
        .map_err(|error| Error::Message(error).failure(FailureKind::Parser))?);
    loaded.push((doc.model_pack.clone(), std::sync::Arc::clone(&reader)));
    Ok(reader)
}

/// The extraction's structure, its moves DOC's where the request names `doc` and the PDF's mode is promoted, else tier 0's;
/// its dump entry written to `dump` where named (the structure gold's).
fn derive_extracted(
    extracted: &mut ExtractedPdf,
    source_hash: &str,
    semantic_profile: Option<DetectionProfile>,
    doc: Option<&crate::contract::DocOptions>,
    dump: Option<legal_pdf_structure::Keep<'_>>,
) -> Result<StructureOutput> {
    let mode = if extracted.pages.iter().any(|page| page.source != "native") { "recognized" } else { "born-digital" };
    let doc = doc.filter(|_| DOC_PROMOTED.contains(&mode)).map(loaded_doc).transpose()?;
    let mut derived = legal_pdf_structure::derive(
        &mut extracted.pages,
        &extracted.separators,
        &extracted.outline,
        StructureIdentity {
            source_sha256: source_hash.to_owned(),
            profile: semantic_profile,
            page_labels: extracted.metadata.embedded_page_labels.clone(),
        },
        std::env::var_os("LEGALPDF_DECISION_TRACE").map(PathBuf::from).as_deref(),
        doc.as_deref(),
        dump,
    )?;
    extracted.diagnostics.append(&mut derived.diagnostics);
    Ok(derived)
}

/// Recognized pages made ready for the readers: the strokes each recognized page's image shows,
/// as rule regions, the words a rule underlines, each measured line's ink, stroke weight and
/// capital height, and its words' stroke weights.
fn finish_recognition(
    pages: &mut [legal_pdf_core::model::Page],
    rasters: BTreeMap<usize, legal_pdf_core::RasterReading>,
) {
    for (page_index, reading) in rasters {
        let Some(page) = pages.get_mut(page_index) else {
            continue;
        };
        if page.lines.len() == reading.weights.len() {
            for (slot, line) in page.lines.iter_mut().enumerate() {
                line.weight = reading.weights[slot];
                line.ink = reading.inks.get(slot).copied().unwrap_or(0.0);
                line.cap_height = reading.cap_heights.get(slot).copied().unwrap_or(0.0);
                if let Some(weights) = reading.word_weights.get(slot).filter(|weights| weights.len() == line.words.len()) {
                    for (word, weight) in line.words.iter_mut().zip(weights) {
                        word.weight = *weight;
                    }
                }
            }
        }
        let number = page.number;
        page.regions.extend(reading.strokes.into_iter().enumerate().map(|(n, bbox)| {
            let id = format!("p{number:04}-stroke{:04}", n + 1);
            legal_pdf_core::model::Region { id, page_index, kind: "rule".to_owned(), line_ids: Vec::new(), bbox, reading_order: 0 }
        }));
    }
    for page in pages {
        legal_pdf_extraction_processor::mark_rule_underlines(page);
    }
}

/// A recognized extraction's structure, projected as the document consumers read, over the pages as the structure
/// stage left them: its source fragments and line labels name those lines (a transcript's line number split off its line).
#[doc(hidden)]
pub fn derive_document(
    mut extracted: ExtractedPdf,
    source_hash: &str,
    key: &str,
    semantic_profile: Option<DetectionProfile>,
    doc: Option<&crate::contract::DocOptions>,
    dump: Option<legal_pdf_structure::Keep<'_>>,
) -> Result<PdfDocument> {
    let derived = profile::measure("derive_document", || {
        derive_extracted(&mut extracted, source_hash, semantic_profile, doc, dump)
    })?;
    let document_status = if extracted.diagnostics.is_empty() { PdfStatus::Ready } else { PdfStatus::Degraded };
    let qualities = extracted.pages.iter().map(|page| page.text_quality).collect();
    Ok(profile::measure("project_document", || {
        PdfDocument::from_parts(source_hash, key, document_status, extracted.metadata, extracted.outline, qualities, extracted.pages, derived.authority_text_units, derived.structure_graph)
    }))
}

/// A cached extraction through the structure stage, as a parse reads it, checked: the pages as the stage left them and the
/// structure, its revision taken, with the extraction's diagnostics.
#[doc(hidden)]
pub fn corpus_structure(source_hash: &str, mut extracted: ExtractedPdf) -> Result<(ExtractedPdf, DocumentStructure)> {
    let mut derived = derive_extracted(&mut extracted, source_hash, None, None, None)?;
    derived.structure_graph.refresh_revision()?;
    Ok((extracted, derived.structure_graph))
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

/// The OCR cache's root for a parse cache root: the request's own, else `ocr` beside `parse-v1`.
fn ocr_cache_root(root: &Path, options: &PdfRequest) -> PathBuf {
    options.ocr_cache_dir.clone().unwrap_or_else(|| root.join("ocr"))
}

/// Oldest files out of the given folders until they fit the cache limit.
fn prune_cache(directories: &[PathBuf]) {
    let mut files = directories
        .iter()
        .flat_map(fs::read_dir)
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

fn maybe_prune_document_cache(root: &Path, prepared: &Path, ocr_root: &Path) {
    static WRITES: AtomicUsize = AtomicUsize::new(0);
    // ponytail: amortized maintenance; rescan sooner if cache overshoot becomes material.
    if WRITES.fetch_add(1, Ordering::Relaxed) % 32 == 0 {
        // Each is bounded apart: a parse never evicts an extraction or recognition.
        prune_cache(&[parse_cache_root(root).join("documents")]);
        prune_cache(&[prepared.to_owned()]);
        prune_cache(&[ocr_root.join("recognition")]);
    }
}

fn cached_document(root: &Path, key: &str, source_hash: &str) -> Option<PdfDocument> {
    let path = parse_cache_root(root).join("documents").join(format!("{key}.json.gz"));
    if !path.is_file() {
        return None;
    }
    let cached = profile::measure("document_cache_read_decode", || read_gzip_json::<PdfDocument>(&path).ok());
    let current = |cached: &PdfDocument| {
        let (structure, summary) = (cached.structure(), cached.summary());
        structure.schema_version.as_ref() == DOCUMENT_STRUCTURE_SCHEMA
            && summary.sha256 == source_hash
            && summary.parser_version == PARSER_VERSION
    };
    let Some(cached) = cached.filter(current) else {
        let _ = fs::remove_file(path);
        return None;
    };
    touch_cache(&path);
    Some(cached)
}

/// The PDF's document as `options` asks for it, named by its id and URL there: read from the parse cache, or parsed from
/// `bytes` and cached (where `require_cache_write`, failing where it cannot be). None where no bytes are given and none
/// is cached, or where the cache key asked for is no longer cached.
pub(crate) fn parse_pdf(
    bytes: Option<&[u8]>,
    options: &PdfRequest,
    progress: RecognitionProgress<'_>,
    require_cache_write: bool,
) -> Result<Option<PdfDocument>> {
    #[cfg(feature = "ocr")]
    let provider_named = options.ocr.is_some();
    #[cfg(not(feature = "ocr"))]
    let provider_named = false;
    if options.supplied_ocr.is_some() && (provider_named || options.cache_key.is_some()) {
        return Err(Error::Message("Supplied OCR cannot be combined with a provider or cache key".into()));
    }
    let Some(mut document) = parsed_pdf(bytes, options, progress, require_cache_write)? else {
        return Ok(None);
    };
    if options.ocr_pages.as_ref().is_some_and(|pages| pages.iter().any(|page| *page >= document.page_count())) {
        return Err(Error::Message("document request pages contains a page beyond the source PDF".to_owned()));
    }
    let structure = document.structure_mut();
    if let Some(id) = &options.id {
        structure.document_id = id.to_owned();
    }
    structure.url = options.url.clone();
    Ok(Some(document))
}

/// TEMP lever "oracle-text" (remove before landing): the oracle. Each extracted line keeps its place, box, fonts and order,
/// and reads the gold's text for it, the gold's correction where it has one: the gold of this source in the store
/// LEGALPDF_ORACLE_STORE names (`<store>/<sha>/structure/**/gold.json.gz`, the last by name), joined by line id. Under
/// it, whatever the parser misses is the parser's, no reading of the page to blame.
fn oracle_text(extracted: &mut ExtractedPdf, source_hash: &str) {
    if !legal_structure_model::lever("oracle-text") {
        return;
    }
    let Some(store) = std::env::var_os("LEGALPDF_ORACLE_STORE").map(PathBuf::from) else { return };
    fn golds(dir: &Path, found: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                golds(&path, found);
            } else if path.file_name().is_some_and(|name| name == "gold.json.gz") {
                found.push(path);
            }
        }
    }
    let mut found = Vec::new();
    golds(&store.join(source_hash).join("structure"), &mut found);
    found.sort();
    let Some(path) = found.last() else { return };
    let Ok(gold) = read_gzip_json::<Value>(path) else { return };
    // (Each gold line's text, page and box: an id names the same line only where the page is the same and the boxes
    // overlap by half the smaller one; an extraction changed since the gold was made may number its lines otherwise.)
    let texts: BTreeMap<String, (String, i64, [f64; 4])> = gold["lines"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|line| {
            let id = line["id"].as_str()?;
            let text = line["corrected"].as_str().filter(|text| !text.trim().is_empty()).or_else(|| line["text"].as_str())?;
            let page = line["page"].as_i64()?;
            let bbox = line["bbox"].as_array().filter(|b| b.len() == 4).map(|b| [0, 1, 2, 3].map(|k| b[k].as_f64().unwrap_or(0.0)))?;
            Some((id.to_owned(), (text.to_owned(), page, bbox)))
        })
        .collect();
    let overlap = |a: [f64; 4], b: [f64; 4]| {
        let (w, h) = ((a[2].min(b[2]) - a[0].max(b[0])).max(0.0), (a[3].min(b[3]) - a[1].max(b[1])).max(0.0));
        let smaller = ((a[2] - a[0]) * (a[3] - a[1])).min((b[2] - b[0]) * (b[3] - b[1]));
        if smaller > 0.0 { w * h / smaller } else { 0.0 }
    };
    let (mut replaced, mut refused) = (0usize, 0usize);
    let seen: std::collections::HashSet<String> = extracted.pages.iter().flat_map(|page| page.lines.iter().map(|line| line.id.clone())).collect();
    let unmatched = texts.keys().filter(|id| !seen.contains(*id)).count();
    for line in extracted.pages.iter_mut().flat_map(|page| page.lines.iter_mut()) {
        let Some((text, page, bbox)) = texts.get(&line.id).filter(|(text, ..)| *text != line.text) else { continue };
        if *page != line.page_index as i64 + 1 || overlap(*bbox, line.bbox) < 0.5 {
            refused += 1;
            continue;
        }
        replaced += 1;
        line.text = text.clone();
        if let Some(mut span) = line.spans.first().cloned() {
            span.text = text.clone();
            span.bbox = line.bbox;
            line.spans = vec![span];
        }
        // Words cut again over the new text, each a share of the line's box by its characters.
        if let Some(template) = line.words.first().cloned() {
            let characters: Vec<char> = text.chars().collect();
            let [x0, y0, x1, y1] = line.bbox;
            let share = if characters.is_empty() { 0.0 } else { (x1 - x0) / characters.len() as f64 };
            let mut words = Vec::new();
            let mut at = 0;
            while at < characters.len() {
                if characters[at].is_whitespace() {
                    at += 1;
                    continue;
                }
                let mut end = at;
                while end < characters.len() && !characters[end].is_whitespace() {
                    end += 1;
                }
                let mut word = template.clone();
                word.id = format!("{}-w{}", line.id, words.len());
                word.text = characters[at..end].iter().collect();
                word.bbox = [x0 + share * at as f64, y0, x0 + share * end as f64, y1];
                word.start = at;
                word.end = end;
                words.push(word);
                at = end;
            }
            line.words = words;
        }
    }
    eprintln!("ORACLE {source_hash} replaced {replaced} refused {refused} gold-unmatched {unmatched} of {}", texts.len());
}

fn parsed_pdf(
    bytes: Option<&[u8]>,
    options: &PdfRequest,
    #[cfg_attr(not(feature = "ocr"), allow(unused_variables))] progress: RecognitionProgress<'_>,
    require_cache_write: bool,
) -> Result<Option<PdfDocument>> {
    let source_hash = source_hash(bytes, options)?;
    let cache_root = options.cache_dir.clone().unwrap_or_else(default_cache_dir);
    if let Some(key) = options.cache_key.as_deref() {
        let cached = cached_document(&cache_root, key, &source_hash);
        if let Some(document) = &cached {
            validate_output_size(document, options)?;
        }
        return Ok(cached);
    }
    let (key, prepared, providers) = reading(&source_hash, options)?;
    if let Some(cached) = cached_document(&cache_root, &key, &source_hash) {
        validate_output_size(&cached, options)?;
        return Ok(Some(cached));
    }
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let mut extracted = recognized_extraction(bytes, &source_hash, options, Some(&cache_root), providers, progress, require_cache_write, &prepared)?;
    oracle_text(&mut extracted, &source_hash);
    let parsed = derive_document(extracted, &source_hash, &key, options.profile, options.doc.as_ref(), None)?;
    validate_output_size(&parsed, options)?;
    let cache_path = parse_cache_root(&cache_root).join("documents").join(format!("{key}.json.gz"));
    match profile::measure("document_cache_write", || write_gzip_json(&cache_path, &parsed)) {
        Ok(()) => maybe_prune_document_cache(&cache_root, &cache_root.join(&prepared), &ocr_cache_root(&cache_root, options)),
        Err(error) if require_cache_write => return Err(error),
        Err(_) => {}
    }
    // The parse's phases, written once it is read and kept.
    profile::tallied();
    Ok(Some(parsed))
}

/// The cache key the reading `options` asks for of the PDF with `source_hash` is kept under, the identity its finished
/// extraction is kept under (`prepared_identity`), and the providers that reading is made with, prepared (their
/// identities are part of both).
fn reading(source_hash: &str, options: &PdfRequest) -> Result<(String, String, Providers)> {
    #[cfg(feature = "ocr")]
    let ocr_prepared = profile::measure("provider_identity_ocr", || options.ocr.as_ref().map(OcrProvider::prepare).transpose())
        .map_err(|error| error.failure(FailureKind::OcrRuntime))?;
    #[cfg(feature = "ocr")]
    let ocr_identity = ocr_prepared.as_ref().map(|provider| (provider.name().to_owned(), provider.identity().to_owned()));
    #[cfg(not(feature = "ocr"))]
    let ocr_identity: Option<(String, String)> = None;
    #[cfg(feature = "ppdoc")]
    let ppdoc_prepared = profile::measure("provider_identity_ppdoc", || options.ppdoc.as_ref().map(PPDocLayout::new).transpose())
        .map_err(|error| error.failure(FailureKind::LayoutRuntime))?;
    #[cfg(feature = "ppdoc")]
    let ppdoc_identity = ppdoc_prepared.as_ref().map(|provider| provider.identity().to_owned());
    #[cfg(not(feature = "ppdoc"))]
    let ppdoc_identity: Option<String> = None;
    let ocr_identity = match &options.supplied_ocr {
        Some(supplied) => Some(("supplied".to_owned(), supplied.identity(source_hash)?)),
        None => ocr_identity,
    };
    let doc = options.doc.as_ref().map(doc_identity).transpose()?;
    let ocr_pair = ocr_identity.as_ref().map(|provider| (provider.0.as_str(), provider.1.as_str()));
    let key = cache_key(source_hash, document_identity(), ocr_pair, ppdoc_identity.as_deref(), options.ocr_pages.as_deref(), options.profile, doc.as_deref());
    let prepared = prepared_identity(ocr_pair, ppdoc_identity.as_deref(), options.ocr_pages.as_deref());
    Ok((key, prepared, Providers {
        #[cfg(feature = "ocr")]
        ocr_identity,
        #[cfg(feature = "ocr")]
        ocr: ocr_prepared,
        #[cfg(feature = "ppdoc")]
        ppdoc: ppdoc_prepared,
    }))
}

/// The pages a PDF's recognition would read, by index from 0: those its extraction (cached) finds no text layer reads
/// well, as a parse with no recognizer leaves them, its structure not read.
pub fn pages_needing_ocr(bytes: &[u8], options: &PdfRequest) -> Result<Vec<usize>> {
    let source_hash = source_hash(Some(bytes), options)?;
    let cache_root = options.cache_dir.clone().unwrap_or_else(default_cache_dir);
    let providers = Providers {
        #[cfg(feature = "ocr")]
        ocr_identity: None,
        #[cfg(feature = "ocr")]
        ocr: None,
        #[cfg(feature = "ppdoc")]
        ppdoc: None,
    };
    let prepared = prepared_identity(None, None, options.ocr_pages.as_deref());
    Ok(recognized_extraction(bytes, &source_hash, options, Some(&cache_root), providers, None, false, &prepared)?.metadata.pages_needing_ocr)
}

/// The cache key a PDF's reading is kept under: of `bytes` where given (they must be the source expected), else of
/// the source expected.
pub fn pdf_cache_key(bytes: Option<&[u8]>, options: &PdfRequest) -> Result<String> {
    Ok(reading(&source_hash(bytes, options)?, options)?.0)
}

/// Where a PDF's finished extraction is kept: under the identity of what made it, by the PDF, packed
/// (`write_gzip_packed`).
#[doc(hidden)]
pub fn prepared_path(root: &Path, identity: &str, source_hash: &str) -> PathBuf {
    root.join(identity).join(format!("{source_hash}.pack.gz"))
}

/// A finished extraction kept at `path`, read back (and its time touched, for the cache's pruning), where it is whole.
#[doc(hidden)]
pub fn kept_extraction(path: &Path) -> Option<ExtractedPdf> {
    let kept = profile::measure("extraction_cache_read", || read_gzip_packed::<ExtractedPdf>(path).ok())
        .filter(|kept| !kept.pages.is_empty() && kept.pages.len() == kept.separators.len())?;
    touch_cache(path);
    Some(kept)
}

/// What a finished extraction is kept under: the extraction's schema and the providers that made it (the recognizer's
/// and the layout model's identities, the pages recognition was asked to read).
fn prepared_identity(ocr: Option<(&str, &str)>, ppdoc: Option<&str>, ocr_pages: Option<&[usize]>) -> String {
    let identity = serialization_sha256(&("legalpdf.prepared.v1", EXTRACTION_CACHE_SCHEMA, ocr, ppdoc, ocr_pages));
    identity[..16.min(identity.len())].to_owned()
}

/// The source's SHA-256: of its bytes, a PDF's, where given, which must be the source expected; else the source
/// expected.
fn source_hash(bytes: Option<&[u8]>, options: &PdfRequest) -> Result<String> {
    let Some(bytes) = bytes else {
        return options.expected_source_sha256.clone().ok_or_else(|| Error::Message("PDF cache lookup requires expected_source_sha256".to_owned()));
    };
    if bytes.is_empty() || bytes.len() > 100 * 1024 * 1024 || !bytes[..bytes.len().min(1024)].windows(5).any(|window| window == b"%PDF-") {
        return Err(Error::Message("PDF source bytes are invalid".to_owned()).failure(FailureKind::Corrupt));
    }
    let actual = profile::measure("source_sha256", || Sha256::digest(bytes).iter().map(|byte| format!("{byte:02x}")).collect::<String>());
    if options.expected_source_sha256.as_deref().is_some_and(|expected| expected != actual) {
        return Err(Error::Message("PDF source changed after preparation began".to_owned()).failure(FailureKind::SourceChanged));
    }
    Ok(actual)
}

/// The providers a parse reads its pages with, prepared: its recognizer and the identity its
/// readings are cached under, and its layout model.
#[doc(hidden)]
pub struct Providers {
    #[cfg(feature = "ocr")]
    pub ocr_identity: Option<(String, String)>,
    #[cfg(feature = "ocr")]
    pub ocr: Option<PreparedOcrProvider>,
    #[cfg(feature = "ppdoc")]
    pub ppdoc: Option<PPDocLayout>,
}

/// A PDF's pages as the structure stage receives them: extracted through the extraction cache under
/// `cache_root`, recognized where they need it through the recognition cache (keyed by the OCR code
/// and its provider alone, so no parser change discards a reading), and laid out by the layout
/// model where one is given. A parse and the structure gold read their pages through this one path.
#[doc(hidden)]
pub fn recognized_extraction(
    bytes: &[u8],
    source_hash: &str,
    options: &PdfRequest,
    cache_root: Option<&Path>,
    #[cfg_attr(not(feature = "ocr"), allow(unused_variables, unused_mut))] mut providers: Providers,
    #[cfg_attr(not(feature = "ocr"), allow(unused_variables))] progress: RecognitionProgress<'_>,
    require_cache_write: bool,
    prepared: &str,
) -> Result<ExtractedPdf> {
    #[cfg(feature = "ocr")]
    let mut ocr_provider = {
        let identity = providers.ocr_identity.as_ref().map(|provider| (provider.0.as_str(), provider.1.as_str()));
        let recognition_key = serialization_sha256(&("legalpdf.recognition-cache.v1", source_hash, RECOGNITION_CODE, identity));
        let raster_key = serialization_sha256(&("legalpdf.raster-cache.v1", source_hash, RASTER_VERSION));
        let recognition_root = cache_root.map(|root| ocr_cache_root(root, options).join("recognition"));
        options.ocr.as_ref().zip(providers.ocr).map(|(options, prepared)| DeferredOcrProvider {
            options,
            prepared: Some(prepared),
            runtime: None,
            cache_root: recognition_root,
            cache_identity: recognition_key,
            raster_identity: raster_key,
            region_render: None,
            progress,
            rasters: BTreeMap::new(),
        })
    };
    #[cfg(feature = "ocr")]
    let selected_ocr = ocr_provider.as_mut().map(|provider| provider as &mut dyn legal_pdf_core::PdfOcrProvider);
    #[cfg(not(feature = "ocr"))]
    let selected_ocr = None;
    let mut supplied = options.supplied_ocr.as_ref();
    let selected_ocr = match supplied.as_mut() {
        Some(provider) => Some(provider as &mut dyn legal_pdf_core::PdfOcrProvider),
        None => selected_ocr,
    };
    // The finished extraction (the PDF's text layer, recognition where it reads pages, the layout model's annotations),
    // kept once under what made it (`prepared`: the schema and the providers' identities, or the caller's), by the
    // PDF: a PDF read before by the same extraction is read back, its pages neither recognized nor laid out again. Never
    // keyed by a fingerprint of source files: work after extraction never re-extracts, and a change to extraction
    // itself bumps EXTRACTION_CACHE_SCHEMA.
    let prepared_path = cache_root.map(|root| prepared_path(root, prepared, source_hash));
    if let Some(cached) = prepared_path.as_deref().and_then(kept_extraction) {
        return Ok(cached);
    }
    let mut extracted = profile::measure("extract_pdf", || extract_pdf(bytes))?;
    recognize_pdf(bytes, &mut extracted, selected_ocr, options.ocr_pages.as_deref())?;
    #[cfg(feature = "ocr")]
    let rasters = ocr_provider.as_mut().map(|provider| std::mem::take(&mut provider.rasters)).unwrap_or_default();
    #[cfg(not(feature = "ocr"))]
    let rasters = BTreeMap::new();
    finish_recognition(&mut extracted.pages, rasters);
    #[cfg(feature = "ppdoc")]
    if let Some(provider) = providers.ppdoc.as_mut() {
        profile::measure("ppdoc_annotate", || provider.annotate_pdf(bytes, &mut extracted.pages))
            .map_err(|error| error.failure(FailureKind::LayoutRuntime))?;
    }
    if let Some(path) = &prepared_path {
        let result = profile::measure("extraction_cache_write", || write_gzip_packed(path, &extracted));
        if require_cache_write {
            result?;
        }
    }
    Ok(extracted)
}


