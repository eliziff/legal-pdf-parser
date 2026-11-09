use legal_pdf_core::model::{PARSER_VERSION, SCHEMA_VERSION};
use legal_pdf_core::{read_gzip_json, write_gzip_json, Error, FailureKind, Result};
use legal_pdf_extraction_processor::{extract_pdf, recognize_pdf, ExtractedPdf};
#[cfg(feature = "ocr")]
use legal_pdf_ocr::{OcrOptions, OcrProvider, PreparedOcrProvider};
use legal_pdf_diagnostics::validate_pdf_components;
use legal_pdf_structure::{StructureIdentity, StructureOutput};
use legal_pdf_support::{profile, PdfDocument, PdfStatus};
#[cfg(feature = "ppdoc")]
use legal_pdf_support::{PPDocLayout, PPDocOptions};
use legal_structure_model::{DetectionProfile, DocumentStructure, ScalarText, DOCUMENT_STRUCTURE_SCHEMA};
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

const EXTRACTION_CACHE_SCHEMA: &str = "legalpdf.extraction-cache.v1";
const PARSE_CACHE_MAX_BYTES: u64 = 1024 * 1024 * 1024;

/// Recognized pages so far, and pages to recognize, as recognition proceeds.
pub type RecognitionProgress<'a> = Option<&'a (dyn Fn(usize, usize) + Sync)>;

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
            .unwrap_or_default()
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
            let key = serialization_sha256(&(&self.cache_identity, request.page_index, request.width, request.height, &request.regions))?;
            let path = self.cache_root.as_ref().map(|root| root.join(format!("{key}.json.gz")));
            let cached = path
                .as_ref()
                .and_then(|path| read_gzip_json::<CachedOcrPage>(path).ok())
                .filter(|cached| cached.key == key && cached.page.page_index == request.page_index);
            // LEGALPDF_RECOGNITION_ADOPT: an OCR code change that leaves recognition as it was
            // (its raster pass, its plumbing) adopts the page's newest earlier reading.
            let cached = cached.or_else(|| {
                std::env::var_os("LEGALPDF_RECOGNITION_ADOPT")?;
                let root = self.cache_root.as_ref()?;
                let mut earlier = fs::read_dir(root)
                    .ok()?
                    .flatten()
                    .filter_map(|entry| Some((entry.metadata().ok()?.modified().ok()?, entry.path())))
                    .collect::<Vec<_>>();
                earlier.sort_by(|a, b| b.0.cmp(&a.0));
                let page = earlier.into_iter().find_map(|(_, file)| {
                    read_gzip_json::<CachedOcrPage>(&file).ok().filter(|cached| cached.page.page_index == request.page_index)
                })?;
                let adopted = CachedOcrPage { key: key.clone(), page: page.page };
                if let Some(path) = &path {
                    let _ = write_gzip_json(path, &adopted);
                }
                Some(adopted)
            });
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
        // LEGALPDF_RECOGNITION_TRACE: how many of the pages asked for the recognition cache held.
        if std::env::var_os("LEGALPDF_RECOGNITION_TRACE").is_some() {
            eprintln!("recognition\t{} cached\t{} to read", results.len(), missing.len());
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
        let key = serialization_sha256(&("region-v1", &self.cache_identity, page_index, region.map(|value| (value * 100.0).round() as i64), (scale * 100.0).round() as i64))?;
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

#[derive(Serialize, Deserialize)]
struct CachedExtraction<T = ExtractedPdf> {
    schema_version: String,
    source_sha256: String,
    extraction: T,
}

#[derive(Debug, Clone)]
pub(crate) struct ParseOptions {
    pub profile: Option<DetectionProfile>,
    pub cache_dir: Option<PathBuf>,
    /// Where page recognition (OCR) and its scanner rasters are cached: apart from the parse
    /// cache, keyed by the OCR code and settings alone, so no parser change discards them.
    /// Defaults to `ocr` in the cache directory.
    pub ocr_cache_dir: Option<PathBuf>,
    pub cache_key: Option<String>,
    pub require_cache_write: bool,
    pub use_cache: bool,
    pub expected_source_sha256: Option<String>,
    pub max_output_bytes: Option<usize>,
    pub supplied_ocr: Option<crate::supplied_ocr::SuppliedOcr>,
    #[cfg(feature = "ocr")]
    pub ocr: Option<OcrOptions>,
    /// Zero-based pages eligible for OCR. Native extraction still inspects the
    /// complete PDF so physical page numbering stays authoritative.
    pub ocr_pages: Option<Vec<usize>>,
    #[cfg(feature = "ppdoc")]
    pub ppdoc: Option<PPDocOptions>,
}

impl Default for ParseOptions {
    fn default() -> Self {
        Self {
            profile: None,
            cache_dir: None,
            ocr_cache_dir: None,
            cache_key: None,
            require_cache_write: false,
            use_cache: true,
            expected_source_sha256: None,
            max_output_bytes: None,
            supplied_ocr: None,
            #[cfg(feature = "ocr")]
            ocr: None,
            ocr_pages: None,
            #[cfg(feature = "ppdoc")]
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
        return base.join("OpenLegalData/apps/legalpdf/cache");
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."));
    base.join("OpenLegalData/legalpdf")
}

fn validate_output_size(document: &PdfDocument, options: &ParseOptions) -> Result<()> {
    match options.max_output_bytes {
        Some(limit) if document.structure().query_text().len() > limit => Err(Error::Message("PDF document text exceeds the read limit".to_owned())),
        _ => Ok(()),
    }
}

fn document_identity() -> &'static Value {
    static IDENTITY: OnceLock<Value> = OnceLock::new();
    IDENTITY.get_or_init(|| {
        json!({
            "document_source_sha256": env!("LEGAL_PDF_DOCUMENT_SHA256"),
            "extraction_source_sha256": env!("LEGAL_PDF_EXTRACTION_SHA256"),
            "structure_source_sha256": legal_pdf_core::layout_analysis().source_sha256(),
        })
    })
}

fn cache_key(
    source_hash: &str,
    identity: &Value,
    ocr_provider: Option<(&str, &str)>,
    ppdoc_identity: Option<&str>,
    ocr_pages: Option<&[usize]>,
    semantic_profile: Option<DetectionProfile>,
) -> Result<String> {
    let value = json!({
        "source_sha256": source_hash,
        "schema_version": SCHEMA_VERSION,
        "parser_version": PARSER_VERSION,
        "engine_code": identity,
        "ocr_provider": ocr_provider.map(|provider| provider.0),
        "ocr_provider_identity": ocr_provider.map(|provider| provider.1),
        "ocr_source_sha256": ocr_provider
            .filter(|provider| provider.0 != "supplied")
            .map(|_| env!("LEGAL_PDF_OCR_SHA256")),
        "layout_provider": ppdoc_identity.map(|_| "ppdoc-lite"),
        "layout_provider_identity": ppdoc_identity,
        "layout_source_sha256": ppdoc_identity.map(|_| env!("LEGAL_PDF_LAYOUT_SHA256")),
        "ocr_pages": ocr_pages,
        "profile": semantic_profile,
    });
    serialization_sha256(&value)
}

fn derive_extracted(
    extracted: &mut ExtractedPdf,
    source_hash: &str,
    cache_root: Option<&Path>,
    semantic_profile: Option<DetectionProfile>,
) -> Result<StructureOutput> {
    // Reader state depends on source pages, navigation hints and the engine's layout services.
    let reader_cache = cache_root.and_then(|root| {
        let analysis = legal_pdf_core::layout_analysis().source_sha256();
        let ablated = std::env::var("LEGALPDF_ABLATE").ok();
        let key = serialization_sha256(&(env!("LEGAL_PDF_READER_SHA256"), analysis, semantic_profile, &extracted.pages, &extracted.separators, &extracted.outline, ablated)).ok()?;
        Some(parse_cache_root(root).join("reader").join(format!("{key}.json.gz")))
    });
    let mut derived = legal_pdf_structure::derive(
        &mut extracted.pages,
        &extracted.separators,
        &extracted.outline,
        StructureIdentity {
            source_sha256: source_hash.to_owned(),
            profile: semantic_profile,
            page_labels: extracted.metadata.embedded_page_labels.clone(),
        },
        reader_cache.as_deref(),
        std::env::var_os("LEGALPDF_DECISION_TRACE").map(PathBuf::from).as_deref(),
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

/// A recognized extraction's structure, checked and projected as the document consumers read;
/// `pages`, where given, receives the pages as the structure stage left them.
fn derive_document(
    mut extracted: ExtractedPdf,
    source_hash: &str,
    key: &str,
    cache_root: Option<&Path>,
    semantic_profile: Option<DetectionProfile>,
    pages: Option<&mut Vec<legal_pdf_core::model::Page>>,
) -> Result<PdfDocument> {
    let passage_pages = PdfDocument::passage_evidence(&extracted.pages);
    let derived = profile::measure("derive_document", || {
        derive_extracted(&mut extracted, source_hash, cache_root, semantic_profile)
    })?;
    profile::measure("build.validate_document", || {
        validate_pdf_components(&extracted.pages, &derived.structure_graph)
    })?;
    let document_status = if extracted.diagnostics.is_empty() { PdfStatus::Ready } else { PdfStatus::Degraded };
    if let Some(pages) = pages {
        pages.clone_from(&extracted.pages);
    }
    Ok(profile::measure("project_document", || {
        PdfDocument::from_parts(source_hash, key, document_status, extracted.metadata, extracted.pages.iter().map(|page| page.text_quality).collect(), passage_pages, derived.authority_text_units, derived.structure_graph)
    }))
}

/// The structure gold's inputs: `pdf`'s pages exactly as a parse of this build feeds its structure
/// stage (`recognized_extraction`), cached under `cache_dir` as a parse caches them. Recognition is
/// Indra's where LEGALPDF_INDRA_MODEL_PACK and LEGALPDF_ONNXRUNTIME name it, through the product's
/// recognition cache (a page read before by the same OCR code is not read again); the layout model
/// PP-DocLayout's where LEGALPDF_PPDOC_MODEL_PACK names a pack. The gold is aligned to whatever
/// lines these are, so nothing here is chosen by the gold.
#[cfg(feature = "ocr")]
#[doc(hidden)]
pub fn gold_structure_inputs(pdf: &[u8], cache_dir: &Path) -> Result<ExtractedPdf> {
    let source_hash = format!("{:x}", Sha256::digest(pdf));
    let recognizer = std::env::var_os("LEGALPDF_INDRA_MODEL_PACK").is_some() && std::env::var_os("LEGALPDF_ONNXRUNTIME").is_some();
    let options = ParseOptions {
        cache_dir: Some(cache_dir.to_owned()),
        ocr: recognizer.then(OcrOptions::default),
        #[cfg(feature = "ppdoc")]
        // The product's layout model (OpenVINO), on the same pack and runtime: LEGALPDF_PPDOC_DEVICE its device.
        ppdoc: std::env::var_os("LEGALPDF_PPDOC_MODEL_PACK").map(|pack| PPDocOptions {
            model_pack: Some(PathBuf::from(&pack)),
            runtime: std::env::var_os("LEGALPDF_OPENVINO_RUNTIME").map(PathBuf::from),
            threads: std::env::var("LEGALPDF_PPDOC_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1),
            device: std::env::var("LEGALPDF_PPDOC_DEVICE").ok(),
            ..PPDocOptions::default()
        }),
        ..ParseOptions::default()
    };
    let ocr = options.ocr.as_ref().map(OcrProvider::prepare).transpose()?;
    let providers = Providers {
        ocr_identity: ocr.as_ref().map(|provider| (provider.name().to_owned(), provider.identity().to_owned())),
        ocr,
        #[cfg(feature = "ppdoc")]
        ppdoc: options.ppdoc.as_ref().map(PPDocLayout::new).transpose()?,
    };
    recognized_extraction(pdf, &source_hash, &options, Some(cache_dir), providers, None)
}

/// What the structure gold's inputs are keyed by: the extraction schema and the models the
/// environment names. Work after extraction never re-extracts the gold; recognition is cached on
/// its own, and a change to it that should reach the inputs bumps the version here.
#[doc(hidden)]
pub fn gold_inputs_identity() -> String {
    let models = ["LEGALPDF_INDRA_MODEL_PACK", "LEGALPDF_ONNXRUNTIME", "LEGALPDF_PPDOC_MODEL_PACK"].map(|name| std::env::var(name).ok());
    let identity = serialization_sha256(&("legalpdf.gold-inputs.v2", EXTRACTION_CACHE_SCHEMA, models)).unwrap_or_default();
    identity[..16.min(identity.len())].to_owned()
}
/// The structure gold's replay: a recognized extraction through the structure stage, uncached;
/// `pages`, where given, receives the pages as the structure stage left them.
#[doc(hidden)]
pub fn gold_structure_document(
    extracted: ExtractedPdf,
    source_hash: &str,
    pages: Option<&mut Vec<legal_pdf_core::model::Page>>,
) -> Result<PdfDocument> {
    derive_document(extracted, source_hash, "structure-gold", None, None, pages)
}

/// A cached extraction through the structure stage, as a parse reads it, checked: the pages as the stage left them and the
/// structure, its revision taken, with the extraction's diagnostics.
fn corpus_structure(source_hash: &str, mut extracted: ExtractedPdf) -> Result<(ExtractedPdf, DocumentStructure)> {
    let mut derived = derive_extracted(&mut extracted, source_hash, None, None)?;
    derived.structure_graph.refresh_revision()?;
    profile::measure("build.validate_document", || validate_pdf_components(&extracted.pages, &derived.structure_graph))?;
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
fn ocr_cache_root(root: &Path, options: &ParseOptions) -> PathBuf {
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

fn maybe_prune_document_cache(root: &Path, ocr_root: &Path) {
    static WRITES: AtomicUsize = AtomicUsize::new(0);
    // ponytail: amortized maintenance; rescan sooner if cache overshoot becomes material.
    if WRITES.fetch_add(1, Ordering::Relaxed) % 32 == 0 {
        // Each is bounded apart: a parse never evicts an extraction or recognition.
        prune_cache(&[parse_cache_root(root).join("documents")]);
        prune_cache(&[parse_cache_root(root).join("extractions")]);
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

pub(crate) fn parse_pdf(
    bytes: Option<&[u8]>,
    options: &ParseOptions,
    #[cfg_attr(not(feature = "ocr"), allow(unused_variables))] progress: RecognitionProgress<'_>,
) -> Result<Option<PdfDocument>> {
    let source_hash = if let Some(bytes) = bytes {
        if bytes.is_empty() || bytes.len() > 100 * 1024 * 1024 || !bytes[..bytes.len().min(1024)].windows(5).any(|window| window == b"%PDF-") {
            return Err(Error::Message("PDF source bytes are invalid".to_owned()).failure(FailureKind::Corrupt));
        }
        let actual = profile::measure("source_sha256", || format!("{:x}", Sha256::digest(bytes)));
        if options.expected_source_sha256.as_deref().is_some_and(|expected| expected != actual) {
            return Err(Error::Message("PDF source changed after preparation began".to_owned()).failure(FailureKind::SourceChanged));
        }
        actual
    } else {
        options.expected_source_sha256.clone().ok_or_else(|| Error::Message("PDF cache lookup requires expected_source_sha256".to_owned()))?
    };
    let cache_root = options.use_cache.then(|| options.cache_dir.clone().unwrap_or_else(default_cache_dir));
    if let Some(key) = options.cache_key.as_deref() {
        let cached = cache_root.as_deref().and_then(|root| cached_document(root, key, &source_hash));
        if let Some(document) = &cached {
            validate_output_size(document, options)?;
        }
        return Ok(cached);
    }
    let identity = document_identity();
    #[cfg(feature = "ocr")]
    let mut ocr_prepared = profile::measure("provider_identity_ocr", || options.ocr.as_ref().map(OcrProvider::prepare).transpose())
        .map_err(|error| error.failure(FailureKind::OcrRuntime))?;
    #[cfg(feature = "ocr")]
    let ocr_identity = ocr_prepared.as_ref().map(|provider| (provider.name().to_owned(), provider.identity().to_owned()));
    #[cfg(not(feature = "ocr"))]
    let ocr_identity: Option<(String, String)> = None;
    #[cfg(feature = "ppdoc")]
    let mut ppdoc_prepared = profile::measure("provider_identity_ppdoc", || options.ppdoc.as_ref().map(PPDocLayout::new).transpose())
        .map_err(|error| error.failure(FailureKind::LayoutRuntime))?;
    #[cfg(feature = "ppdoc")]
    let ppdoc_identity = ppdoc_prepared.as_ref().map(|provider| provider.identity().to_owned());
    #[cfg(not(feature = "ppdoc"))]
    let ppdoc_identity: Option<String> = None;
    let ocr_identity = match &options.supplied_ocr {
        Some(supplied) => Some(("supplied".to_owned(), supplied.identity(&source_hash)?)),
        None => ocr_identity,
    };
    let ocr_pair = ocr_identity.as_ref().map(|provider| (provider.0.as_str(), provider.1.as_str()));
    let key = cache_key(&source_hash, identity, ocr_pair, ppdoc_identity.as_deref(), options.ocr_pages.as_deref(), options.profile)?;
    if let Some(cached) = cache_root.as_deref().and_then(|root| cached_document(root, &key, &source_hash)) {
        validate_output_size(&cached, options)?;
        return Ok(Some(cached));
    }
    let Some(bytes) = bytes else {
        return Ok(None);
    };
    let providers = Providers {
        ocr_identity,
        #[cfg(feature = "ocr")]
        ocr: ocr_prepared.take(),
        #[cfg(feature = "ppdoc")]
        ppdoc: ppdoc_prepared.take(),
    };
    let extracted = recognized_extraction(bytes, &source_hash, options, cache_root.as_deref(), providers, progress)?;
    let parsed = derive_document(extracted, &source_hash, &key, cache_root.as_deref(), options.profile, None)?;
    validate_output_size(&parsed, options)?;
    if let Some(root) = &cache_root {
        let cache_path = parse_cache_root(root).join("documents").join(format!("{key}.json.gz"));
        match profile::measure("document_cache_write", || write_gzip_json(&cache_path, &parsed)) {
            Ok(()) => maybe_prune_document_cache(root, &ocr_cache_root(root, options)),
            Err(error) if options.require_cache_write => return Err(error),
            Err(_) => {}
        }
    }
    Ok(Some(parsed))
}

/// The providers a parse reads its pages with, prepared: its recognizer and the identity its
/// readings are cached under, and its layout model.
struct Providers {
    ocr_identity: Option<(String, String)>,
    #[cfg(feature = "ocr")]
    ocr: Option<PreparedOcrProvider>,
    #[cfg(feature = "ppdoc")]
    ppdoc: Option<PPDocLayout>,
}

/// A PDF's pages as the structure stage receives them: extracted through the extraction cache under
/// `cache_root`, recognized where they need it through the recognition cache (keyed by the OCR code
/// and its provider alone, so no parser change discards a reading), and laid out by the layout
/// model where one is given. A parse and the structure gold read their pages through this one path.
fn recognized_extraction(
    bytes: &[u8],
    source_hash: &str,
    options: &ParseOptions,
    cache_root: Option<&Path>,
    mut providers: Providers,
    #[cfg_attr(not(feature = "ocr"), allow(unused_variables))] progress: RecognitionProgress<'_>,
) -> Result<ExtractedPdf> {
    #[cfg(feature = "ocr")]
    let mut ocr_provider = {
        let identity = providers.ocr_identity.as_ref().map(|provider| (provider.0.as_str(), provider.1.as_str()));
        let recognition_key = serialization_sha256(&("legalpdf.recognition-cache.v1", source_hash, env!("LEGAL_PDF_OCR_SHA256"), identity))?;
        let raster_key = serialization_sha256(&("legalpdf.raster-cache.v1", source_hash, env!("LEGAL_PDF_RASTER_SHA256")))?;
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
    // A PDF's extraction is keyed by the PDF and the schema, never by a fingerprint of source files: work after
    // extraction never re-extracts, and a change to extraction itself bumps EXTRACTION_CACHE_SCHEMA.
    let extraction_key = serialization_sha256(&(EXTRACTION_CACHE_SCHEMA, source_hash))?;
    let extraction_path = cache_root.map(|root| parse_cache_root(root).join("extractions").join(format!("{extraction_key}.json.gz")));
    let cached = extraction_path
        .as_ref()
        .and_then(|path| profile::measure("extraction_cache_read", || read_gzip_json::<CachedExtraction>(path)).ok())
        .filter(|cached| {
            let extraction = &cached.extraction;
            cached.schema_version == EXTRACTION_CACHE_SCHEMA && cached.source_sha256 == source_hash && !extraction.pages.is_empty() && extraction.pages.len() == extraction.separators.len()
        });
    let mut extracted = if let Some(cached) = cached {
        if let Some(path) = &extraction_path {
            touch_cache(path);
        }
        cached.extraction
    } else {
        let extracted = profile::measure("extract_pdf", || extract_pdf(bytes, None, None))?;
        if let Some(path) = &extraction_path {
            let entry = CachedExtraction { schema_version: EXTRACTION_CACHE_SCHEMA.to_owned(), source_sha256: source_hash.to_owned(), extraction: &extracted };
            let result = profile::measure("extraction_cache_write", || write_gzip_json(path, &entry));
            if options.require_cache_write {
                result?;
            }
        }
        extracted
    };
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
    Ok(extracted)
}

fn serialization_sha256(value: &impl Serialize) -> Result<String> {
    Ok(legal_structure_model::serialization_sha256(value))
}

fn structure_examples(graph: &DocumentStructure) -> Value {
    use legal_structure_model::NodeKind;
    let text = ScalarText::new(&graph.text);
    let mut examples = BTreeMap::<String, Vec<Value>>::new();
    for kind in [NodeKind::Heading, NodeKind::Section, NodeKind::Paragraph, NodeKind::ListItem, NodeKind::Footnote, NodeKind::Endnote] {
        let nodes = graph.nodes.iter().filter(|node| node.kind == kind).collect::<Vec<_>>();
        if nodes.is_empty() {
            continue;
        }
        let mut slots = vec![0, nodes.len() / 2, nodes.len() - 1];
        slots.dedup();
        let example = |slot: usize| {
            let node = nodes[slot];
            let words = text.slice_utf16(node.range.start..node.range.end).unwrap_or_default().split_whitespace().collect::<Vec<_>>().join(" ");
            json!({
                "id": node.id,
                "text": words.chars().take(240).collect::<String>(),
                "label": node.label,
                "locator_kind": node.locator_kind,
                "parent_id": node.parent_id,
                "page_indexes": node.page_indexes,
                "line_ids": node.line_ids,
            })
        };
        examples.insert(kind.name().to_owned(), slots.into_iter().map(example).collect());
    }
    json!(examples)
}

#[doc(hidden)]
pub fn corpus_check_cached_extraction(input: impl AsRef<Path>) -> Result<Value> {
    let cached: CachedExtraction = read_gzip_json(input.as_ref())?;
    if cached.schema_version != EXTRACTION_CACHE_SCHEMA || cached.source_sha256.len() != 64 || cached.extraction.pages.len() != cached.extraction.separators.len() {
        return Err(Error::Message("invalid extraction cache".to_owned()));
    }
    let page_count = cached.extraction.pages.len();
    let line_count = cached.extraction.pages.iter().map(|page| page.lines.len()).sum::<usize>();
    let source_sha256 = cached.source_sha256.clone();
    let (extracted, graph) = corpus_structure(&source_sha256, cached.extraction)?;
    let graph = &graph;
    // Each document's nodes, a line each, for reading two engines' graphs side by side.
    if let Some(directory) = std::env::var_os("LEGALPDF_CORPUS_NODES") {
        let line = |node: &legal_structure_model::StructureNode| {
            format!("{} {} {:?} {:?}\n", serde_json::to_value(node.kind).unwrap_or_default(), node.label.as_deref().unwrap_or("_"), node.range, node.parent_id)
        };
        let lines = graph.nodes.iter().map(line).collect::<String>();
        std::fs::write(Path::new(&directory).join(format!("{}.nodes.txt", &source_sha256[..16])), lines).map_err(|error| Error::Message(error.to_string()))?;
    }
    let count = |names: &mut dyn Iterator<Item = String>| names.fold(BTreeMap::<String, usize>::new(), |mut counts, name| {
        *counts.entry(name).or_default() += 1;
        counts
    });
    let by_kind = count(&mut graph.nodes.iter().map(|node| node.kind.name().to_owned()));
    let sections = graph.nodes.iter().filter(|node| node.kind == legal_structure_model::NodeKind::Section);
    let sections_by_locator_kind = count(&mut sections.map(|node| node.locator_kind.clone().unwrap_or_else(|| "unknown".to_owned())));
    let heading_derived_sections = graph
        .nodes
        .iter()
        .filter(|node| node.kind == legal_structure_model::NodeKind::Section && node.grammar.as_deref() == Some("accepted_heading"))
        .count();
    let structure = json!({
        "schema_version": graph.schema_version,
        "node_count": graph.nodes.len(),
        "note_count": graph.nodes.iter().filter(|node| matches!(node.kind, legal_structure_model::NodeKind::Footnote | legal_structure_model::NodeKind::Endnote)).count(),
        "by_kind": by_kind,
        "sections_by_locator_kind": sections_by_locator_kind,
        "heading_derived_section_count": heading_derived_sections,
        "graph_sha256": serialization_sha256(graph)?,
        "examples": structure_examples(graph),
    });
    Ok(json!({
        "source_sha256": source_sha256,
        "page_count": page_count,
        "line_count": line_count,
        "product_sha256": serialization_sha256(&(&extracted.pages, graph, &extracted.diagnostics))?,
        "structure": structure,
    }))
}
