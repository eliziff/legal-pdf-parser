//! Indra, the recognizer: a PDF's pages or grey images read into lines and words. Pages are drawn
//! once and stood upright; the line detector, the separator and the region checks read the page in
//! luma grey, and lines are cut for reading from it in the recognizer's own grey (`gray_and_ink_of`).
use crate::indra_native::{IndraNative, IndraScratch};
use crate::line_detector::LineDetector;
use crate::model_pack::ModelPack;
use crate::orientation::{place_back, turn, Orientation};
use crate::planning::{self, RuntimePlan};
use image::{imageops, GrayImage, Luma, RgbaImage};
use legal_pdf_core::profile::measure;
use legal_pdf_core::render::{open_pdf, PageRenderer};
use legal_pdf_core::{canonical_file, provider_asset_sha256, required_path, Error, OrtBackend, Result};
use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord};
use ort::{
    execution_providers::{cuda::CuDNNConvAlgorithmSearch, CPUExecutionProvider, CUDAExecutionProvider},
    session::{
        builder::GraphOptimizationLevel,
        builder::PrepackedWeights,
        run_options::{HasSelectedOutputs, OutputSelector, RunOptions},
        Session,
    },
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::{Arc, Mutex};

/// The recognizer reads lines this many pixels high.
const INPUT_HEIGHT: usize = 32;
const BLANK_LABEL: usize = 0;
/// The white margin set above and below a line's ink before it is read, as a share of its height.
const VERTICAL_MARGIN: f64 = 0.2;
/// How sure the recognizer must be of a line the page analysis left out: a number or a word reads
/// at 0.95 and more; a signature, a seal's lettering or a speckle at less.
const RECOVERED_CONFIDENCE: f64 = 0.95;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndraTier {
    Quality,
    #[default]
    Balanced,
    Fast,
    Turbo,
    Extreme,
}
impl IndraTier {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Fast => "fast",
            Self::Turbo => "turbo",
            Self::Extreme => "extreme",
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OcrOptions {
    pub model_pack: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub dpi: u16,
    pub threads: usize,
    pub workers: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub width_bucket: usize,
    pub tier: IndraTier,
    /// Omitted selects a CPU/GPU plan from this machine's capabilities.
    pub backend: Option<OrtBackend>,
    pub device: Option<String>,
    pub cpu_arena: bool,
    pub expected_identity: Option<String>,
    pub orientation: Option<PathBuf>,
}
impl Default for OcrOptions {
    fn default() -> Self {
        Self {
            model_pack: None,
            runtime: None,
            dpi: 144,
            threads: 0,
            workers: 0,
            layout_workers: 0,
            batch_size: 0,
            width_bucket: 0,
            tier: IndraTier::Balanced,
            backend: None,
            device: None,
            cpu_arena: true,
            expected_identity: None,
            orientation: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct LineBox {
    pub(crate) left: usize,
    pub(crate) top: usize,
    pub(crate) right: usize,
    pub(crate) bottom: usize,
    /// Ink the page analysis left out, read as a line of its own.
    pub(crate) recovered: bool,
}

impl LineBox {
    pub(crate) fn width(self) -> usize {
        self.right.saturating_sub(self.left)
    }

    pub(crate) fn height(self) -> usize {
        self.bottom.saturating_sub(self.top)
    }
}

/// A line cut from its page, padded and scaled to the recognizer's height.
struct PreparedLine {
    bbox: LineBox,
    confidence_required: bool,
    width: usize,
    pixels: Vec<u8>,
    pixel_x_origin: f64,
    pixel_x_scale: f64,
}

/// A line's recognized text, its confidence and where its words stand on the page.
struct Recognition {
    text: String,
    confidence: f64,
    words: Vec<OcrWord>,
}

/// Where a recognized image stands on its page: its page, the rule above the page's notes, the
/// points each pixel spans, its top-left corner on the page in points, and the quarter turns
/// counter-clockwise that stood it upright with its size as drawn, in pixels.
struct Placement {
    page_index: usize,
    separator: Option<f64>,
    scale: [f64; 2],
    origin: [f64; 2],
    turns: u8,
    drawn: [f64; 2],
}

struct Codec {
    labels: BTreeMap<usize, Vec<(Vec<usize>, String)>>,
}

impl Codec {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        let mapping: BTreeMap<String, Vec<usize>> = serde_json::from_reader(BufReader::new(file))?;
        if mapping.is_empty() || mapping.values().any(Vec::is_empty) {
            return Err(Error::Message("Indra codec is empty or malformed".to_owned()));
        }
        let mut labels = mapping.into_iter().map(|(text, labels)| (labels, text)).collect::<Vec<_>>();
        labels.sort_by_key(|(labels, _)| std::cmp::Reverse(labels.len()));
        for (index, (prefix, _)) in labels.iter().enumerate() {
            if labels.iter().enumerate().any(|(other, (candidate, _))| {
                index != other && prefix.len() <= candidate.len() && candidate.starts_with(prefix)
            }) {
                return Err(Error::Message("Indra codec is not prefix-free".to_owned()));
            }
        }
        let mut grouped: BTreeMap<usize, Vec<(Vec<usize>, String)>> = BTreeMap::new();
        for (sequence, text) in labels {
            grouped.entry(sequence[0]).or_default().push((sequence, text));
        }
        Ok(Self { labels: grouped })
    }

    /// The text `decode` reads from these labels, piece by piece, with the first and last
    /// label each piece was read from.
    fn pieces(&self, labels: &[usize]) -> Vec<(usize, usize, &str)> {
        let mut pieces = Vec::new();
        let mut index = 0;
        while index < labels.len() {
            let Some((sequence, value)) = self
                .labels
                .get(&labels[index])
                .into_iter()
                .flatten()
                .find(|(sequence, _)| labels[index..].starts_with(sequence))
            else {
                index += 1;
                continue;
            };
            pieces.push((index, index + sequence.len() - 1, value.as_str()));
            index += sequence.len();
        }
        pieces
    }

    fn decode(&self, spans: &[(usize, f32)]) -> (String, f64) {
        let labels = spans.iter().map(|(label, _)| *label).collect::<Vec<_>>();
        let mut text = String::new();
        let mut confidence = 0.0_f64;
        let mut characters = 0_usize;
        let mut index = 0;
        while index < spans.len() {
            let Some((sequence, value)) = self
                .labels
                .get(&labels[index])
                .into_iter()
                .flatten()
                .find(|(sequence, _)| labels[index..].starts_with(sequence))
            else {
                index += 1;
                continue;
            };
            let score = spans[index..index + sequence.len()].iter().map(|(_, score)| f64::from(*score)).sum::<f64>()
                / sequence.len() as f64;
            text.push_str(value);
            confidence += score * value.chars().count() as f64;
            characters += value.chars().count();
            index += sequence.len();
        }
        (text, if characters == 0 { 0.0 } else { confidence / characters as f64 })
    }
}

enum RecognizerEngine {
    Native {
        model: Arc<IndraNative>,
        workers: usize,
    },
    Ort(Vec<Session>),
}

fn open_recognizer(
    model: &Path,
    plan: &RuntimePlan,
    cpu_arena: bool,
    independent_pool: bool,
) -> Result<RecognizerEngine> {
    if plan.backend == OrtBackend::Cpu {
        let native = IndraNative::from_onnx_path(model).map_err(Error::Message)?;
        return Ok(RecognizerEngine::Native {
            model: Arc::new(native),
            workers: plan.workers.max(1),
        });
    }
    let weights = PrepackedWeights::new();
    let sessions = (0..plan.workers)
        .map(|_| open_session(model, plan.threads, plan.backend, &plan.device, cpu_arena, &weights, independent_pool))
        .collect::<Result<Vec<_>>>()?;
    Ok(RecognizerEngine::Ort(sessions))
}

pub struct OcrProvider {
    engine: RecognizerEngine,
    codec: Codec,
    dpi: u16,
    layouts: Vec<LineDetector>,
    plan: RuntimePlan,
    orientation: Option<Orientation>,
    identity: String,
    fallback: Option<(PathBuf, PathBuf)>,
    fallback_engine: Option<(RecognizerEngine, Codec)>,
    cpu_arena: bool,
    independent_pool: bool,
}

/// A recognizer's models found, checked and planned for this machine, and its identity known,
/// before any is loaded; a GPU plan's recognizer already loaded where opening it verified the plan.
pub struct PreparedOcrProvider {
    pub(crate) pack: ModelPack,
    pub(crate) runtime: PathBuf,
    pub(crate) plan: RuntimePlan,
    pub(crate) ready: Option<Box<OcrProvider>>,
    pub(crate) identity: String,
}
impl PreparedOcrProvider {
    pub fn identity(&self) -> &str {
        &self.identity
    }
    pub fn name(&self) -> &str {
        "indra"
    }
}
impl OcrProvider {
    pub fn new(options: &OcrOptions) -> Result<Self> {
        Self::from_prepared(options, Self::prepare(options)?)
    }
    pub fn prepare(options: &OcrOptions) -> Result<PreparedOcrProvider> {
        validate_options(options)?;
        let root = required_path(&options.model_pack, "LEGALPDF_INDRA_MODEL_PACK", "Indra model pack")?;
        let runtime = canonical_file(
            &required_path(&options.runtime, "LEGALPDF_ONNXRUNTIME", "ONNX Runtime library")?,
            "ONNX Runtime library",
        )?;
        legal_pdf_core::init_ort_runtime(&runtime, None)?;
        let prepared = planning::prepare(options, &root, &runtime)?;
        if options.expected_identity.as_deref().is_some_and(|expected| expected != prepared.identity) {
            return Err(Error::Message("Indra identity changed before OCR began".into()));
        }
        Ok(prepared)
    }
    pub(crate) fn prepare_plan(options: &OcrOptions, root: &Path, runtime: PathBuf, plan: RuntimePlan) -> Result<PreparedOcrProvider> {
        let device = &plan.device;
        let mut pack = ModelPack::open(root, options.tier, plan.backend)?;
        if let Some(path) = &options.orientation {
            pack.orientation = Some(canonical_file(path, "orientation model")?);
        }
        let layout_identity = provider_asset_sha256(&pack.detector)?;
        let identity=format!("indra-native-v1:backend={}:device={device}:model={}:codec={}:runtime={}:layout={}:layout-model={layout_identity}:side={}:orientation={}:tier={}:dpi={}:batch={}:bucket={}:workers={}:threads={}:layout-workers={}:cpu-arena={}",
            plan.backend.name(),provider_asset_sha256(&pack.model)?,provider_asset_sha256(&pack.codec)?,provider_asset_sha256(&runtime)?,"indra",pack.detector_side,
            pack.orientation.as_deref().map(provider_asset_sha256).transpose()?.unwrap_or_default(),
            options.tier.name(),options.dpi,plan.batch_size,plan.width_bucket,plan.workers,plan.threads,plan.layout_workers,options.cpu_arena);
        Ok(PreparedOcrProvider { pack, runtime, plan, ready: None, identity })
    }
    pub fn from_prepared(options: &OcrOptions, mut prepared: PreparedOcrProvider) -> Result<Self> {
        if let Some(ready) = prepared.ready.take() {
            return Ok(*ready);
        }
        Self::open_prepared(options, &prepared)
    }
    pub(crate) fn open_prepared(options: &OcrOptions, prepared: &PreparedOcrProvider) -> Result<Self> {
        let independent_pool = legal_pdf_core::init_ort_runtime(&prepared.runtime, None)?.is_some();
        let (plan, pack) = (&prepared.plan, &prepared.pack);
        let (layouts, engine) = std::thread::scope(|scope| -> Result<_> {
            // Independent model pools can initialize together; CPU recognition loads weights directly.
            let layouts = scope.spawn(|| {
                (0..plan.layout_workers)
                    .map(|_| LineDetector::open(&pack.detector, pack.detector_side, plan.backend, &plan.device, independent_pool))
                    .collect::<Result<Vec<_>>>()
            });
            let engine = open_recognizer(&pack.model, plan, options.cpu_arena, independent_pool);
            let layouts = layouts.join().map_err(|_| Error::Message("layout initialization worker panicked".into()))??;
            Ok((layouts, engine?))
        })?;
        Ok(Self {
            engine,
            codec: Codec::load(&pack.codec)?,
            dpi: options.dpi,
            layouts,
            orientation: pack.orientation.as_deref().map(Orientation::open).transpose()?,
            plan: plan.clone(),
            identity: prepared.identity.clone(),
            fallback: pack.fallback.clone(),
            fallback_engine: None,
            cpu_arena: options.cpu_arena,
            independent_pool,
        })
    }
    pub fn configuration(&self) -> &RuntimePlan {
        &self.plan
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn name(&self) -> &str {
        "indra"
    }

    /// Each image's lines, its pages prepared a window at a time while the window before is read.
    pub fn recognize_images(&mut self, images: &[GrayImage]) -> Result<Vec<Vec<OcrLine>>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let mut use_fallback = vec![false; images.len()];
        let adjusted: Vec<std::borrow::Cow<'_, GrayImage>> = images
            .iter()
            .enumerate()
            .map(|(idx, img)| {
                let prof = crate::simd_kernel::analyze_gray_page(img);
                let is_crisp_legal = prof.white_frac >= 0.78 && prof.mid_frac <= 0.015 && prof.line_energy >= 3.50;
                if self.fallback.is_some() && !is_crisp_legal {
                    use_fallback[idx] = true;
                }
                match crate::simd_kernel::upright_and_deskew(img, prof) {
                    Some(fixed) => std::borrow::Cow::Owned(fixed),
                    None => std::borrow::Cow::Borrowed(img),
                }
            })
            .collect();
        let refs: Vec<&GrayImage> = adjusted.iter().map(std::borrow::Cow::as_ref).collect();
        let mut prep_all = {
            let layouts = &mut self.layouts;
            let window_size = if refs.len() <= 4 { refs.len() } else { preparation_window_size(refs.len()) };
            let mut prepared = Vec::with_capacity(refs.len());
            for chunk in refs.chunks(window_size) {
                prepared.extend(prepare_window_refs(layouts, chunk, chunk)?);
            }
            prepared
        };

        let mut pages = vec![Vec::new(); images.len()];
        let primary_indices: Vec<usize> = (0..images.len()).filter(|&i| !use_fallback[i]).collect();
        if !primary_indices.is_empty() {
            let primary_prep: Vec<Vec<PreparedLine>> = primary_indices.iter().map(|&i| std::mem::take(&mut prep_all[i])).collect();
            let recognized = recognize_pages(&mut self.engine, &self.codec, &self.plan, primary_prep)?;
            for (idx, lines) in primary_indices.into_iter().zip(recognized) {
                pages[idx] = lines;
            }
        }
        let fallback_indices: Vec<usize> = (0..images.len()).filter(|&i| use_fallback[i]).collect();
        if !fallback_indices.is_empty() {
            if self.fallback_engine.is_none() {
                if let Some((ref model_path, ref codec_path)) = self.fallback {
                    let fb_engine = open_recognizer(model_path, &self.plan, self.cpu_arena, self.independent_pool)?;
                    let fb_codec = Codec::load(codec_path)?;
                    self.fallback_engine = Some((fb_engine, fb_codec));
                }
            }
            if let Some((ref mut fb_engine, ref fb_codec)) = self.fallback_engine {
                let fb_prep: Vec<Vec<PreparedLine>> = fallback_indices.iter().map(|&i| std::mem::take(&mut prep_all[i])).collect();
                let recognized = recognize_pages(fb_engine, fb_codec, &self.plan, fb_prep)?;
                for (idx, lines) in fallback_indices.into_iter().zip(recognized) {
                    pages[idx] = lines;
                }
            }
        }

        for (i, page_lines) in pages.iter_mut().enumerate() {
            if crate::simd_kernel::needs_180_check(page_lines) {
                let flipped = imageops::rotate180(refs[i]);
                let flip_ref = [&flipped];
                let prep = prepare_window_refs(&mut self.layouts, &flip_ref, &flip_ref)?;
                let re = if use_fallback[i] {
                    if let Some((ref mut fb_engine, ref fb_codec)) = self.fallback_engine {
                        recognize_pages(fb_engine, fb_codec, &self.plan, prep)?
                    } else {
                        Vec::new()
                    }
                } else {
                    recognize_pages(&mut self.engine, &self.codec, &self.plan, prep)?
                };
                if let Some(re_lines) = re.into_iter().next() {
                    if crate::simd_kernel::text_quality_score(&re_lines) > crate::simd_kernel::text_quality_score(page_lines) {
                        *page_lines = re_lines;
                    }
                }
            }
        }
        Ok(pages)
    }

    /// The requested pages of `bytes` recognized, in request order: drawn by up to four workers,
    /// their lines found and prepared, and read, each stage a window of pages ahead of the next.
    pub(crate) fn recognize_pdf(&mut self, bytes: &[u8], requests: &[OcrPageRequest]) -> Result<Vec<OcrPageResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let window_size = preparation_window_size(requests.len());
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let occupied = self.plan.workers * self.plan.threads + self.plan.layout_workers;
        let render_workers = available.saturating_sub(occupied + 1).clamp(1, 4).min(window_size);
        let dpi = f64::from(self.dpi);
        // Picture regions use real confidence to distinguish print from seals and chart lettering.
        let confidence_pages =
            requests.iter().filter(|request| !request.regions.is_empty()).map(|request| request.page_index).collect::<BTreeSet<_>>();
        let pdf = open_pdf(bytes)?;
        let Self { engine, codec, layouts, plan, orientation, .. } = self;
        let orientation = orientation.as_mut().map(Mutex::new);
        let mut results: Vec<OcrPageResult> = requests
            .iter()
            .map(|request| OcrPageResult { page_index: request.page_index, lines: Vec::new(), separator_y: None })
            .collect();
        std::thread::scope(|scope| -> Result<()> {
            let (render_sender, render_receiver) = sync_channel(1);
            let (prepared_sender, prepared_receiver) = sync_channel(1);
            let (pdf, orientation) = (&pdf, orientation.as_ref());
            scope.spawn(move || {
                std::thread::scope(|render_scope| {
                    let mut queues = Vec::new();
                    for worker in 0..render_workers {
                        let (sender, receiver) = sync_channel(0);
                        queues.push(receiver);
                        render_scope.spawn(move || {
                            // Share the parsed PDF; each worker draws with its own render cache.
                            let pages = PageRenderer::new(pdf, dpi);
                            for window in requests.chunks(window_size) {
                                let start = worker * window.len() / render_workers;
                                let end = (worker + 1) * window.len() / render_workers;
                                let rendered = measure("ocr.render", || render_pdf_pages(&pages, &window[start..end], orientation));
                                let failed = rendered.is_err();
                                if sender.send(rendered).is_err() || failed {
                                    break;
                                }
                            }
                        });
                    }
                    for _ in requests.chunks(window_size) {
                        let value = queues.iter().try_fold(Vec::new(), |mut pages, queue| {
                            pages.extend(queue.recv().map_err(|_| Error::Message("PDF render worker stopped before returning its pages".into()))??);
                            Ok(pages)
                        });
                        let failed = value.is_err();
                        if render_sender.send(value).is_err() || failed {
                            break;
                        }
                    }
                    // Closing every rendezvous queue also unblocks workers if a downstream stage fails.
                    drop(queues);
                });
            });
            scope.spawn(move || {
                for window in render_receiver {
                    let value = window.and_then(|pages: Vec<(Placement, GrayImage, Option<GrayImage>)>| {
                        let (mut placements, mut seen, mut ink) = (Vec::new(), Vec::new(), Vec::new());
                        for (placement, page, read) in pages {
                            placements.push(placement);
                            seen.push(page);
                            ink.push(read);
                        }
                        let mut prepared = prepare_window(layouts, &seen, &ink)?;
                        for (placement, lines) in placements.iter().zip(&mut prepared) {
                            if confidence_pages.contains(&placement.page_index) {
                                lines.iter_mut().for_each(|line| line.confidence_required = true);
                            }
                        }
                        Ok((placements, prepared))
                    });
                    let failed = value.is_err();
                    if prepared_sender.send(value).is_err() || failed {
                        break;
                    }
                }
            });
            place_pages(&mut results, prepared_receiver, engine, codec, plan)
        })?;
        Ok(results)
    }
}

/// Each window's images read and their lines placed back on their pages, in request order: the
/// parts of a page recognized apart are placed back on it, and a page none of whose parts reads as
/// print has no lines.
fn place_pages(
    results: &mut [OcrPageResult],
    windows: Receiver<Result<(Vec<Placement>, Vec<Vec<PreparedLine>>)>>,
    engine: &mut RecognizerEngine,
    codec: &Codec,
    plan: &RuntimePlan,
) -> Result<()> {
    let mut at = 0;
    for window in windows {
        let (placements, prepared) = window?;
        for (placement, mut lines) in placements.into_iter().zip(recognize_pages(engine, codec, plan, prepared)?) {
            place_back(&mut lines, placement.turns, placement.drawn[0], placement.drawn[1]);
            scale_lines(&mut lines, placement.scale);
            translate_lines(&mut lines, placement.origin);
            while results[at].page_index != placement.page_index {
                at += 1;
            }
            results[at].lines.extend(lines);
            results[at].separator_y = results[at].separator_y.or(placement.separator);
        }
    }
    Ok(())
}

/// A window's images' lines found on `seen` and cut for reading from `ink`, falling back to `seen`
/// on monochrome pages where `gray == ink`.
fn prepare_window(layouts: &mut [LineDetector], seen: &[GrayImage], ink: &[Option<GrayImage>]) -> Result<Vec<Vec<PreparedLine>>> {
    let seen_refs: Vec<&GrayImage> = seen.iter().collect();
    let ink_refs: Vec<&GrayImage> = seen.iter().zip(ink).map(|(gray, separate)| separate.as_ref().unwrap_or(gray)).collect();
    prepare_window_refs(layouts, &seen_refs, &ink_refs)
}

fn prepare_window_refs(layouts: &mut [LineDetector], seen: &[&GrayImage], ink: &[&GrayImage]) -> Result<Vec<Vec<PreparedLine>>> {
    let boxes = measure("ocr.layout", || layout_boxes_refs(layouts, seen))?;
    measure("ocr.prepare_lines", || prepare_pages_refs(ink, &boxes))
}

/// Pages' prepared lines read, each kept where it reads as text: a line the page analysis left
/// out only when the recognizer is sure of it.
fn recognize_pages(engine: &mut RecognizerEngine, codec: &Codec, plan: &RuntimePlan, pages: Vec<Vec<PreparedLine>>) -> Result<Vec<Vec<OcrLine>>> {
    let recognized = measure("ocr.recognize", || recognize_prepared_pages(engine, codec, plan, &pages))?;
    Ok(pages
        .into_iter()
        .zip(recognized)
        .map(|(prepared, recognized)| {
            prepared
                .into_iter()
                .zip(recognized)
                .filter_map(|(line, Recognition { text, confidence, words })| {
                    // Retain confidently read source fragments, including isolated letters and symbols.
                    let kept = !text.trim().is_empty() && (!line.bbox.recovered || confidence >= RECOVERED_CONFIDENCE);
                    kept.then_some(OcrLine {
                        recovered: line.bbox.recovered,
                        text,
                        bbox: [line.bbox.left as f64, line.bbox.top as f64, line.bbox.right as f64, line.bbox.bottom as f64],
                        confidence,
                        baseline: vec![],
                        boundary: vec![],
                        words,
                        region_id: String::new(),
                        region_type: "unknown".to_owned(),
                        block_index: 0,
                    })
                })
                .collect()
        })
        .collect())
}

/// Each image's line boxes, the images shared among the line detectors.
fn layout_boxes_refs(layouts: &mut [LineDetector], images: &[&GrayImage]) -> Result<Vec<Vec<LineBox>>> {
    let workers = images.len().min(layouts.len());
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let handles = layouts
            .iter_mut()
            .take(workers)
            .map(|layout| {
                let sender = sender.clone();
                let next = &next;
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= images.len() {
                        break;
                    }
                    let value = layout.lines(images[index]);
                    let failed = value.is_err();
                    if sender.send((index, value)).is_err() || failed {
                        break;
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        let mut output = std::iter::repeat_with(|| None).take(images.len()).collect::<Vec<_>>();
        let mut failure = None;
        for (index, value) in receiver {
            match value {
                Ok(value) => output[index] = Some(value),
                Err(error) => {
                    failure.get_or_insert(error);
                }
            }
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Message("layout worker panicked".to_owned()))?;
        }
        if let Some(error) = failure {
            return Err(error);
        }
        output.into_iter().map(|value| value.ok_or_else(|| Error::Message("layout worker returned no page".to_owned()))).collect()
    })
}

/// Pages' prepared lines read by the active recognizer (pure-Rust SIMD on CPU, ORT on GPU).
fn recognize_prepared_pages(
    engine: &mut RecognizerEngine,
    codec: &Codec,
    plan: &RuntimePlan,
    pages: &[Vec<PreparedLine>],
) -> Result<Vec<Vec<Recognition>>> {
    let counts = pages.iter().map(Vec::len).collect::<Vec<_>>();
    let lines = pages.iter().flatten().collect::<Vec<_>>();
    if lines.is_empty() {
        return Ok(counts.into_iter().map(|_| Vec::new()).collect());
    }
    match engine {
        RecognizerEngine::Native { model, workers } => {
            Ok(recognize_prepared_native(model, codec, *workers, &lines, &counts))
        }
        RecognizerEngine::Ort(sessions) => {
            recognize_prepared_ort(sessions, codec, plan, &lines, &counts)
        }
    }
}

fn recognize_single_native(model: &IndraNative, codec: &Codec, line: &PreparedLine, scratch: &mut IndraScratch) -> Recognition {
    let need_confidence = line.bbox.recovered || line.confidence_required;
    let (timesteps, step_labels, ctc_spans) = model.recognize_u8(&line.pixels, line.width, need_confidence, scratch);
    let (text, confidence) = codec.decode(&ctc_spans);
    let words = line_words(codec, line, &emissions(step_labels.into_iter()), timesteps);
    Recognition { text, confidence, words }
}

fn recognize_prepared_native(
    model: &IndraNative,
    codec: &Codec,
    workers: usize,
    lines: &[&PreparedLine],
    counts: &[usize],
) -> Vec<Vec<Recognition>> {
    let total = lines.len();
    let worker_count = workers.min(total).max(1);
    let mut output: Vec<Option<Recognition>> = (0..total).map(|_| None).collect();
    if worker_count == 1 {
        let mut scratch = IndraScratch::default();
        for (slot, line) in output.iter_mut().zip(lines.iter()) {
            *slot = Some(recognize_single_native(model, codec, line, &mut scratch));
        }
    } else {
        let next = AtomicUsize::new(0);
        let (sender, receiver) = std::sync::mpsc::channel();
        std::thread::scope(|scope| {
            for _ in 0..worker_count {
                let sender = sender.clone();
                let next = &next;
                scope.spawn(move || {
                    let mut scratch = IndraScratch::default();
                    let mut local = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= total {
                            break;
                        }
                        local.push((index, recognize_single_native(model, codec, lines[index], &mut scratch)));
                    }
                    let _ = sender.send(local);
                });
            }
            drop(sender);
            for chunk in receiver {
                for (index, rec) in chunk {
                    output[index] = Some(rec);
                }
            }
        });
    }
    let mut flat = output.into_iter().map(|v| v.expect("every line recognized"));
    counts.iter().map(|&count| flat.by_ref().take(count).collect()).collect()
}

fn recognize_prepared_ort(
    sessions: &mut [Session],
    codec: &Codec,
    plan: &RuntimePlan,
    lines: &[&PreparedLine],
    counts: &[usize],
) -> Result<Vec<Vec<Recognition>>> {
    let jobs = recognition_jobs(lines, plan.batch_size.max(1), plan.width_bucket);
    if jobs.is_empty() {
        return Ok(counts.iter().map(|_| Vec::new()).collect());
    }
    let workers = jobs.len().min(sessions.len()).max(1);
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let handles = sessions
            .iter_mut()
            .take(workers)
            .map(|session| {
                let sender = sender.clone();
                let (next, jobs) = (&next, &jobs);
                let has_class_ids = session.outputs.iter().any(|output| output.name == "class_ids");
                scope.spawn(move || {
                    let mut pixels = Vec::new();
                    let mut lengths = Vec::new();
                    let run_opts_ids = has_class_ids
                        .then(|| {
                            RunOptions::new()
                                .map(|opt| opt.with_outputs(OutputSelector::no_default().with("class_ids").with("lengths")))
                        })
                        .transpose();
                    let run_opts_logits = RunOptions::new()
                        .map(|opt| opt.with_outputs(OutputSelector::no_default().with("logits").with("lengths")));
                    let (Ok(run_opts_ids), Ok(run_opts_logits)) = (run_opts_ids, run_opts_logits) else {
                        let _ = sender.send(Err(Error::Message("Indra ONNX RunOptions setup failed".to_owned())));
                        return;
                    };
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= jobs.len() {
                            break;
                        }
                        let (need_logits, ref batch) = jobs[index];
                        let opts = if !need_logits {
                            if let Some(ref ids_opt) = run_opts_ids {
                                ids_opt
                            } else {
                                &run_opts_logits
                            }
                        } else {
                            &run_opts_logits
                        };
                        let value = recognize_batch(session, opts, codec, lines, batch, &mut pixels, &mut lengths);
                        let failed = value.is_err();
                        if sender.send(value).is_err() || failed {
                            break;
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        let mut output = std::iter::repeat_with(|| None).take(lines.len()).collect::<Vec<_>>();
        let mut failure = None;
        for value in receiver {
            match value {
                Ok(values) => {
                    for (index, value) in values {
                        output[index] = Some(value);
                    }
                }
                Err(error) => {
                    failure.get_or_insert(error);
                }
            };
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Message("Indra recognition worker panicked".to_owned()))?;
        }
        if let Some(error) = failure {
            return Err(error);
        }
        let mut output = output
            .into_iter()
            .map(|value| value.ok_or_else(|| Error::Message("Indra worker returned no line".to_owned())))
            .collect::<Result<Vec<_>>>()?
            .into_iter();
        Ok(counts.iter().map(|&count| output.by_ref().take(count).collect()).collect())
    })
}

fn push_width_batches(
    lines: &[&PreparedLine],
    mut order: Vec<usize>,
    batch_size: usize,
    width_slack: usize,
    need_logits: bool,
    jobs: &mut Vec<(bool, Vec<usize>)>,
) {
    order.sort_unstable_by_key(|&i| lines[i].width);
    let mut cur = Vec::with_capacity(batch_size);
    let mut min_w = 0usize;
    for idx in order {
        let w = lines[idx].width;
        if !cur.is_empty() && (cur.len() >= batch_size || cur.len() * w.saturating_sub(min_w) > width_slack) {
            jobs.push((need_logits, std::mem::take(&mut cur)));
        }
        if cur.is_empty() {
            min_w = w;
        }
        cur.push(idx);
    }
    if !cur.is_empty() {
        jobs.push((need_logits, cur));
    }
}

fn recognition_jobs(lines: &[&PreparedLine], batch_size: usize, width_bucket: usize) -> Vec<(bool, Vec<usize>)> {
    let width_slack = if width_bucket > 0 { width_bucket } else { 128 };
    let mut fast = Vec::new();
    let mut conf = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.bbox.recovered || line.confidence_required {
            conf.push(i);
        } else {
            fast.push(i);
        }
    }
    let mut jobs = Vec::new();
    push_width_batches(lines, fast, batch_size, width_slack, false, &mut jobs);
    push_width_batches(lines, conf, batch_size, width_slack, true, &mut jobs);
    jobs.sort_unstable_by_key(|(_, job)| std::cmp::Reverse(job.len() * lines[*job.last().unwrap()].width));
    jobs
}

/// One batch of lines read: each with its text, confidence and words.
fn recognize_batch(
    session: &mut Session,
    run_options: &RunOptions<HasSelectedOutputs>,
    codec: &Codec,
    lines: &[&PreparedLine],
    batch: &[usize],
    pixels: &mut Vec<f32>,
    lengths: &mut Vec<i64>,
) -> Result<Vec<(usize, Recognition)>> {
    let maximum = batch.iter().map(|&index| lines[index].width).max().unwrap_or(1).div_ceil(8) * 8;
    pixels.resize(batch.len() * INPUT_HEIGHT * maximum, 0.0);
    lengths.clear();
    for (local, &index) in batch.iter().enumerate() {
        let line = lines[index];
        lengths.push(line.width as i64);
        for row in 0..INPUT_HEIGHT {
            let source = row * line.width;
            let target = (local * INPUT_HEIGHT + row) * maximum;
            crate::simd_kernel::u8_to_norm_f32_row(
                &line.pixels[source..source + line.width],
                &mut pixels[target..target + line.width],
            );
            if line.width < maximum {
                pixels[target + line.width..target + maximum].fill(0.0);
            }
        }
    }
    let image = TensorRef::from_array_view(([batch.len(), 1, INPUT_HEIGHT, maximum], pixels.as_slice())).map_err(ort_error)?;
    let sequence_lengths = TensorRef::from_array_view(([batch.len()], lengths.as_slice())).map_err(ort_error)?;
    let outputs = session
        .run_with_options(ort::inputs!["image" => image, "widths" => sequence_lengths], run_options)
        .map_err(ort_error)?;
    let (length_shape, output_lengths) = if let Ok((shape, values)) = outputs["lengths"].try_extract_tensor::<i64>() {
        (shape.iter().copied().collect::<Vec<_>>(), values.iter().map(|&value| usize::try_from(value).unwrap_or(0)).collect::<Vec<_>>())
    } else {
        let (shape, values) = outputs["lengths"].try_extract_tensor::<i32>().map_err(ort_error)?;
        (shape.iter().copied().collect::<Vec<_>>(), values.iter().map(|&value| usize::try_from(value).unwrap_or(0)).collect::<Vec<_>>())
    };
    if let Some(class_ids) = outputs.get("class_ids") {
        let (shape, ids) = class_ids.try_extract_tensor::<i64>().map_err(ort_error)?;
        let dimensions = shape.iter().map(|value| *value as usize).collect::<Vec<_>>();
        if dimensions.len() != 2 || dimensions[0] != batch.len() || length_shape.first().copied() != Some(batch.len() as i64) {
            return Err(Error::Message(format!("Indra model returned unexpected class shapes {dimensions:?} and {length_shape:?}")));
        }
        let timesteps = dimensions[1];
        return Ok(batch
            .iter()
            .enumerate()
            .map(|(local, &index)| {
                let length = output_lengths[local].clamp(1, timesteps);
                let (text, confidence) = codec.decode(&greedy_ids(ids, local, timesteps, length));
                let steps = ids[local * timesteps..local * timesteps + length]
                    .iter()
                    .map(|&label| usize::try_from(label).ok().filter(|label| *label > BLANK_LABEL));
                let words = line_words(codec, lines[index], &emissions(steps), length);
                (index, Recognition { text, confidence, words })
            })
            .collect());
    }
    let logits = outputs.get("logits").ok_or_else(|| Error::Message("Indra model returned neither logits nor class_ids".to_owned()))?;
    let (shape, logits) = logits.try_extract_tensor::<f32>().map_err(ort_error)?;
    let dimensions = shape.iter().map(|value| *value as usize).collect::<Vec<_>>();
    if dimensions.len() != 3 || dimensions[0] != batch.len() || length_shape.first().copied() != Some(batch.len() as i64) {
        return Err(Error::Message(format!("Indra model returned unexpected shapes {dimensions:?} and {length_shape:?}")));
    }
    let (timesteps, classes) = (dimensions[1], dimensions[2]);
    Ok(batch
        .iter()
        .enumerate()
        .map(|(local, &index)| {
            let length = output_lengths[local].clamp(1, timesteps);
            let (spans, steps) = greedy_ctc_and_steps(logits, local, classes, timesteps, length);
            let (text, confidence) = codec.decode(&spans);
            let words = line_words(codec, lines[index], &emissions(steps.into_iter()), length);
            (index, Recognition { text, confidence, words })
        })
        .collect())
}

/// The requested pages drawn in grey: a whole page stood upright, with the rule above its notes; or
/// each requested part of it that reads as print, on a white margin.
fn render_pdf_pages(
    pages: &PageRenderer,
    requests: &[OcrPageRequest],
    orientation: Option<&Mutex<&mut Orientation>>,
) -> Result<Vec<(Placement, GrayImage, Option<GrayImage>)>> {
    // White margin around a part of a page, so the layout reads its first and last lines whole.
    const MARGIN: usize = 16;
    let mut rendered = Vec::new();
    for request in requests {
        let pixmap = pages
            .render(request.page_index)
            .ok_or_else(|| Error::Message(format!("OCR renderer could not draw page {}", request.page_index + 1)))?;
        let (width, height) = (usize::from(pixmap.width()), usize::from(pixmap.height()));
        let scale = [request.width / width as f64, request.height / height as f64];
        let (gray_raw, ink_raw) = crate::ocr::gray_and_ink_of(&pixmap);
        let gray = GrayImage::from_raw(width as u32, height as u32, gray_raw)
            .expect("renderer dimensions match its pixel buffer");
        let ink = ink_raw.map(|buf| {
            GrayImage::from_raw(width as u32, height as u32, buf).expect("renderer dimensions match its pixel buffer")
        });
        if request.regions.is_empty() {
            let turns = match orientation {
                Some(orientation) => {
                    let page = RgbaImage::from_raw(width as u32, height as u32, pixmap.data_as_u8_slice().to_vec())
                        .expect("renderer dimensions match its pixel buffer");
                    orientation.lock().map_err(|_| Error::Message("OCR orientation lock poisoned".into()))?.upright_turns(&page)?
                }
                None => 0,
            };
            // A turned page's rule runs across its print, not across the page as drawn.
            let separator = (turns == 0).then(|| crate::separator::separator_y(gray.as_raw(), width, height, request.height)).flatten();
            let (image, ink) = if turns == 0 { (gray, ink) } else { (turn(&gray, turns), ink.map(|img| turn(&img, turns))) };
            let drawn = [width as f64, height as f64];
            rendered.push((Placement { page_index: request.page_index, separator, scale, origin: [0.0, 0.0], turns, drawn }, image, ink));
            continue;
        }
        // Each part of the page is recognized on its own, at the page's resolution.
        for region in &request.regions {
            let x0 = ((region[0] / scale[0]).floor().max(0.0) as usize).min(width);
            let y0 = ((region[1] / scale[1]).floor().max(0.0) as usize).min(height);
            let x1 = ((region[2] / scale[0]).ceil().max(0.0) as usize).min(width);
            let y1 = ((region[3] / scale[1]).ceil().max(0.0) as usize).min(height);
            if x1 <= x0 || y1 <= y0 || !dark_print_on_light_ground(&gray, [x0, y0, x1, y1]) {
                continue;
            }
            let framed = |image: &GrayImage| {
                let mut crop = GrayImage::from_pixel((x1 - x0 + 2 * MARGIN) as u32, (y1 - y0 + 2 * MARGIN) as u32, Luma([255]));
                let part = imageops::crop_imm(image, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image();
                imageops::replace(&mut crop, &part, MARGIN as i64, MARGIN as i64);
                crop
            };
            let origin = [(x0 as f64 - MARGIN as f64) * scale[0], (y0 as f64 - MARGIN as f64) * scale[1]];
            rendered.push((
                Placement { page_index: request.page_index, separator: None, scale, origin, turns: 0, drawn: [0.0, 0.0] },
                framed(&gray),
                ink.as_ref().map(|img| framed(img)),
            ));
        }
    }
    Ok(rendered)
}

/// Whether a part of a grey page reads as print: mostly light ground, with dark ink on at most a
/// third of it. A photograph, a dark band or a filled graphic is not recognized.
fn dark_print_on_light_ground(gray: &GrayImage, [x0, y0, x1, y1]: [usize; 4]) -> bool {
    let width = gray.width() as usize;
    let (mut light, mut dark, mut count) = (0usize, 0usize, 0usize);
    for row in y0..y1 {
        for &value in &gray.as_raw()[row * width + x0..row * width + x1] {
            light += usize::from(value >= 235);
            dark += usize::from(value <= 110);
            count += 1;
        }
    }
    light * 100 >= count * 45 && dark * 500 >= count && dark * 100 <= count * 35
}

fn preparation_window_size(page_count: usize) -> usize {
    page_count.div_ceil(page_count.div_ceil(4).max(2)).max(1)
}

fn scale_lines(lines: &mut [OcrLine], [x_scale, y_scale]: [f64; 2]) {
    for line in lines {
        for bbox in std::iter::once(&mut line.bbox).chain(line.words.iter_mut().map(|word| &mut word.bbox)) {
            bbox[0] *= x_scale;
            bbox[1] *= y_scale;
            bbox[2] *= x_scale;
            bbox[3] *= y_scale;
        }
        for point in line.baseline.iter_mut().chain(&mut line.boundary) {
            point[0] *= x_scale;
            point[1] *= y_scale;
        }
    }
}

fn translate_lines(lines: &mut [OcrLine], origin: [f64; 2]) {
    if origin == [0.0, 0.0] {
        return;
    }
    for line in lines {
        for bbox in std::iter::once(&mut line.bbox).chain(line.words.iter_mut().map(|word| &mut word.bbox)) {
            bbox[0] += origin[0];
            bbox[1] += origin[1];
            bbox[2] += origin[0];
            bbox[3] += origin[1];
        }
        for point in line.baseline.iter_mut().chain(&mut line.boundary) {
            point[0] += origin[0];
            point[1] += origin[1];
        }
    }
}

fn optimized_cache_path(model: &Path) -> Option<PathBuf> {
    let meta = std::fs::metadata(model).ok()?;
    let mtime = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| d.as_secs());
    let stem = model.file_stem()?.to_string_lossy();
    let dir = std::env::temp_dir().join("legalocr-ort-cache");
    let _ = std::fs::create_dir_all(&dir);
    Some(dir.join(format!("{stem}_{}_{}_v3.opt.onnx", meta.len(), mtime)))
}

pub(crate) fn open_session(
    model: &Path,
    threads: usize,
    backend: OrtBackend,
    device: &str,
    cpu_arena: bool,
    prepacked_weights: &PrepackedWeights,
    independent_pool: bool,
) -> Result<Session> {
    let cache_path = (backend == OrtBackend::Cpu).then(|| optimized_cache_path(model)).flatten();
    let use_cached = cache_path.as_ref().is_some_and(|p| p.is_file());
    let (load_path, opt_level, write_tmp) = if use_cached {
        (cache_path.as_deref().unwrap(), GraphOptimizationLevel::Disable, None)
    } else {
        let tmp = cache_path.as_ref().map(|p| p.with_extension(format!("tmp.{}.onnx", std::process::id())));
        (model, GraphOptimizationLevel::Level3, tmp)
    };
    let mut builder = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(opt_level)
        .map_err(ort_error)?
        .with_intra_threads(threads)
        .map_err(ort_error)?
        .with_inter_threads(1)
        .and_then(|builder| builder.with_prepacked_weights(prepacked_weights))
        .and_then(|builder| builder.with_config_entry("session.set_denormal_as_zero", "1"))
        .map_err(ort_error)?;
    if let Some(ref tmp_path) = write_tmp {
        builder = builder.with_optimized_model_path(tmp_path).map_err(ort_error)?;
    }
    if independent_pool {
        builder = builder.with_independent_thread_pool().map_err(ort_error)?;
    }
    if backend == OrtBackend::DirectMl || (backend == OrtBackend::Cpu && threads == 1) {
        builder = builder
            .with_memory_pattern(false)
            .and_then(|builder| builder.with_parallel_execution(false))
            .map_err(ort_error)?;
        if backend == OrtBackend::Cpu {
            builder = builder
                .with_config_entry("session.intra_op.allow_spinning", "0")
                .and_then(|builder| builder.with_config_entry("session.inter_op.allow_spinning", "0"))
                .map_err(ort_error)?;
        }
    }
    if backend == OrtBackend::Cpu && !cpu_arena {
        builder = builder
            .with_execution_providers(vec![CPUExecutionProvider::default().with_arena_allocator(false).build()])
            .map_err(ort_error)?;
    }
    // Avoid an exhaustive cuDNN search for every changing OCR line width.
    let providers = if backend == OrtBackend::Cuda {
        vec![CUDAExecutionProvider::default()
            .with_device_id(device.parse().expect("normalized device"))
            .with_conv_algorithm_search(CuDNNConvAlgorithmSearch::Heuristic)
            .with_tf32(true)
            .build()
            .error_on_failure()]
    } else {
        backend.providers(device)
    };
    if !providers.is_empty() {
        builder = builder.with_execution_providers(providers).map_err(ort_error)?;
    }
    let session = builder.commit_from_file(load_path).map_err(ort_error)?;
    if let (Some(tmp_path), Some(final_path)) = (write_tmp, cache_path) {
        if tmp_path.is_file() {
            let _ = std::fs::rename(&tmp_path, &final_path);
        }
    }
    Ok(session)
}

fn validate_options(options: &OcrOptions) -> Result<()> {
    if !(72..=600).contains(&options.dpi)
        || options.batch_size > 256
        || options.width_bucket > 1024
        || options.workers > 16
        || options.layout_workers > 16
        || options.threads > 16
    {
        return Err(Error::Message("Indra DPI, batch or worker counts are outside supported bounds".to_owned()));
    }
    if options.backend.is_some_and(|backend| backend != OrtBackend::Cpu) && !options.cpu_arena {
        return Err(Error::Message("Indra low-memory mode requires the CPU backend".to_owned()));
    }
    if options.device.is_some() && options.backend.is_none() {
        return Err(Error::Message("an explicit device requires an explicit backend".to_owned()));
    }
    if let Some(backend) = options.backend {
        backend.normalized_device(options.device.as_deref())?;
    }
    Ok(())
}

pub(crate) fn ort_error(error: ort::Error) -> Error {
    Error::Message(format!("Indra ONNX inference failed: {error}"))
}

/// A line of a grey page cut out and prepared for reading: its ink's bounds padded with white (a
/// third of its height at each side, `VERTICAL_MARGIN` of it above and below) and scaled to the
/// recognizer's height.
fn prepare_line(image: &GrayImage, bbox: LineBox, scratch: &mut crate::simd_kernel::DownsampleScratch) -> Option<PreparedLine> {
    let bw = bbox.width();
    let bh = bbox.height();
    if bw == 0 || bh == 0 {
        return None;
    }
    let stride = image.width() as usize;
    let raw = image.as_raw();
    let (x0, y0, x1, y1, px, py) = if let Some((ix0, iy0, ix1, iy1)) =
        crate::simd_kernel::find_ink_bbox(raw, stride, bbox.left, bbox.top, bw, bh, 220)
    {
        (
            ix0,
            iy0,
            ix1,
            iy1,
            ((iy1 - iy0) as f64 * 0.3).ceil() as usize,
            ((iy1 - iy0) as f64 * VERTICAL_MARGIN).ceil() as usize,
        )
    } else {
        (0, 0, bw, bh, 0, 0)
    };
    let cw = x1 - x0;
    let ch = y1 - y0;
    let padded_width = cw + 2 * px;
    let padded_height = ch + 2 * py;
    let width = (padded_width as f32 * INPUT_HEIGHT as f32 / padded_height as f32).round().clamp(8.0, 4096.0) as usize;
    let pixels = if padded_height > 48 {
        crate::simd_kernel::resize_cropped_padded_triangle_u8(
            raw,
            stride,
            bbox.left + x0,
            bbox.top + y0,
            cw,
            ch,
            px,
            py,
            width,
            INPUT_HEIGHT,
            scratch,
        )
    } else {
        crate::simd_kernel::resize_cropped_padded_bilinear(
            raw,
            stride,
            bbox.left + x0,
            bbox.top + y0,
            cw,
            ch,
            px,
            py,
            width,
            INPUT_HEIGHT,
        )
    };
    Some(PreparedLine {
        bbox,
        confidence_required: false,
        width,
        pixels,
        pixel_x_origin: bbox.left as f64 + x0 as f64 - px as f64,
        pixel_x_scale: padded_width as f64 / width as f64,
    })
}

/// Each image's lines prepared, the images shared among the machine's threads.
fn prepare_pages_refs(images: &[&GrayImage], boxes: &[Vec<LineBox>]) -> Result<Vec<Vec<PreparedLine>>> {
    let workers = images.len().min(std::thread::available_parallelism().map_or(1, |value| value.get()).max(1));
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                let sender = sender.clone();
                let next = &next;
                scope.spawn(move || {
                    let mut scratch = crate::simd_kernel::DownsampleScratch::default();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= images.len() {
                            break;
                        }
                        let lines = boxes[index].iter().filter_map(|&bbox| prepare_line(images[index], bbox, &mut scratch)).collect();
                        if sender.send((index, lines)).is_err() {
                            break;
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        let mut output = std::iter::repeat_with(|| None).take(images.len()).collect::<Vec<_>>();
        for (index, lines) in receiver {
            output[index] = Some(lines);
        }
        for handle in handles {
            handle.join().map_err(|_| Error::Message("line preparation worker panicked".to_owned()))?;
        }
        output.into_iter().map(|lines| lines.ok_or_else(|| Error::Message("line preparation returned no page".to_owned()))).collect()
    })
}

/// The labels a greedy CTC reading emits, each with the first and last timestep of its run:
/// a label is emitted where it differs from the step before, and blanks emit nothing.
fn emissions(steps: impl Iterator<Item = Option<usize>>) -> Vec<(usize, usize, usize)> {
    let mut output: Vec<(usize, usize, usize)> = Vec::new();
    let mut previous: Option<usize> = None;
    for (timestep, label) in steps.enumerate() {
        match label {
            Some(label) if previous == Some(label) => {
                if let Some(last) = output.last_mut() {
                    last.2 = timestep;
                }
            }
            Some(label) => output.push((label, timestep, timestep)),
            None => {}
        }
        previous = label;
    }
    output
}

/// Where a recognized line's words stand on the page. The model reads the line image at a
/// fixed stride, so each emitted character has a column; a CTC reading emits late and
/// narrow, so a column only says roughly where a space falls. Each space is placed in the
/// widest gap of the line's ink near its column, each word runs between its spaces' gaps,
/// drawn in to its own ink, and is mapped back onto the line's box. Offsets count
/// characters of the decoded text.
fn line_words(codec: &Codec, line: &PreparedLine, emitted: &[(usize, usize, usize)], length: usize) -> Vec<OcrWord> {
    let width = line.width;
    if emitted.is_empty() || width == 0 || length == 0 || line.pixels.is_empty() {
        return Vec::new();
    }
    let rows = line.pixels.len() / width;
    let labels: Vec<usize> = emitted.iter().map(|(label, _, _)| *label).collect();
    // Each character of the decoded text with the column it was emitted at.
    let stride = line.width as f64 / length as f64;
    let mut characters: Vec<(char, f64)> = Vec::new();
    for (first, last, value) in codec.pieces(&labels) {
        let column = (emitted[first].1 + emitted[last].2 + 1) as f64 / 2.0 * stride;
        characters.extend(value.chars().map(|character| (character, column)));
    }
    let ink: Vec<bool> = (0..width).map(|column| (0..rows).any(|row| line.pixels[row * width + column] < 128)).collect();
    let (Some(first_ink), Some(last_ink)) = (ink.iter().position(|value| *value), ink.iter().rposition(|value| *value)) else {
        return Vec::new();
    };
    // Runs of blank columns between the line's first and last ink.
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    let mut column = first_ink;
    while column <= last_ink {
        if ink[column] {
            column += 1;
            continue;
        }
        let start = column;
        while column <= last_ink && !ink[column] {
            column += 1;
        }
        gaps.push((start, column));
    }
    let pitch = (last_ink + 1 - first_ink) as f64 / characters.len().max(1) as f64;
    // Words as character ranges, and between them the spaces' estimated columns.
    let mut words: Vec<(usize, usize)> = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        if characters[index].0.is_whitespace() {
            index += 1;
            continue;
        }
        let begin = index;
        while index < characters.len() && !characters[index].0.is_whitespace() {
            index += 1;
        }
        words.push((begin, index));
    }
    let estimates: Vec<f64> = words
        .windows(2)
        .map(|pair| {
            let spaces = &characters[pair[0].1..pair[1].0];
            spaces.iter().map(|(_, column)| *column).sum::<f64>() / spaces.len() as f64
        })
        .collect();
    // How far a space read at `estimate` stands from a gap: nothing when it falls inside.
    let distance = |gap: &(usize, usize), estimate: f64| (gap.0 as f64 - estimate).max(estimate - gap.1 as f64).max(0.0);
    // The spaces fall in the line's widest gaps when those stand where the spaces were read.
    let mut widest = gaps.clone();
    widest.sort_by(|left, right| (right.1 - right.0).cmp(&(left.1 - left.0)));
    widest.truncate(estimates.len());
    widest.sort_unstable();
    let mut bounds: Vec<(usize, usize)> = vec![(first_ink, first_ink)];
    if widest.len() == estimates.len() && widest.iter().zip(&estimates).all(|(gap, estimate)| distance(gap, *estimate) <= 3.0 * pitch) {
        bounds.extend(widest);
    } else {
        // Otherwise each space takes the widest gap near where it was read.
        for estimate in &estimates {
            let after = bounds.last().map_or(0, |bound| bound.1);
            let chosen = gaps
                .iter()
                .filter(|gap| gap.0 >= after && distance(gap, *estimate) <= 1.5 * pitch)
                .max_by(|left, right| {
                    (left.1 - left.0).cmp(&(right.1 - right.0)).then(distance(right, *estimate).total_cmp(&distance(left, *estimate)))
                })
                .copied()
                .unwrap_or_else(|| {
                    let at = (estimate.round().max(0.0) as usize).clamp(after, last_ink + 1);
                    (at, at)
                });
            bounds.push(chosen);
        }
    }
    bounds.push((last_ink + 1, last_ink + 1));
    let scale = line.pixel_x_scale;
    let letter_gap = rows.max(1);
    words
        .iter()
        .enumerate()
        .map(|(slot, (begin, end))| {
            let (from, to) = (bounds[slot].1, bounds[slot + 1].0.max(bounds[slot].1));
            // The word's ink: its largest run of marks and those that follow on within a
            // letter's gap, leaving out specks across the gutter beside it.
            let mut runs: Vec<(usize, usize, usize)> = Vec::new();
            for column in (from..to).filter(|column| ink[*column]) {
                match runs.last_mut() {
                    Some(run) if column - run.1 <= letter_gap => {
                        run.1 = column + 1;
                        run.2 += 1;
                    }
                    _ => runs.push((column, column + 1, 1)),
                }
            }
            let (left, right) = runs.iter().max_by_key(|run| run.2).map_or((from, to), |run| (run.0, run.1));
            OcrWord {
                text: characters[*begin..*end].iter().map(|(character, _)| *character).collect(),
                bbox: [
                    line.pixel_x_origin + left as f64 * scale,
                    line.bbox.top as f64,
                    line.pixel_x_origin + right as f64 * scale,
                    line.bbox.bottom as f64,
                ],
                start: *begin,
                end: *end,
            }
        })
        .collect()
}

fn greedy_ids(ids: &[i64], batch: usize, timesteps: usize, length: usize) -> Vec<(usize, f32)> {
    let mut output = Vec::new();
    let mut previous = i64::MIN;
    for &label in &ids[batch * timesteps..batch * timesteps + length] {
        if label != previous && label > BLANK_LABEL as i64 {
            output.push((label as usize, 1.0));
        }
        previous = label;
    }
    output
}

/// Single-pass argmax + softmax confidence over a `[length, classes]` logit slice.
fn greedy_ctc_and_steps(
    logits: &[f32],
    batch: usize,
    classes: usize,
    timesteps: usize,
    length: usize,
) -> (Vec<(usize, f32)>, Vec<Option<usize>>) {
    let mut output: Vec<(usize, f32)> = Vec::new();
    let mut steps = Vec::with_capacity(length);
    let mut previous = usize::MAX;
    for step in logits[batch * timesteps * classes..][..length * classes].chunks_exact(classes) {
        let (label, best_val) = step
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |best, (idx, &val)| if val > best.1 { (idx, val) } else { best });
        steps.push((label != BLANK_LABEL).then_some(label));
        if label != BLANK_LABEL {
            let denom: f32 = step.iter().map(|&val| (val - best_val).exp()).sum();
            let confidence = 1.0 / denom.max(f32::MIN_POSITIVE);
            if label != previous {
                output.push((label, confidence));
            } else if let Some((_, score)) = output.last_mut() {
                *score = score.max(confidence);
            }
        }
        previous = label;
    }
    (output, steps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn synthetic_text_line(width: usize) -> Vec<u8> {
        let mut buf = vec![255u8; INPUT_HEIGHT * width];
        let mut x = 8usize;
        let mut step = 0usize;
        while x + 10 < width {
            let h0 = 6 + (step % 4);
            let h1 = 24 - (step % 3);
            for y in h0..h1 {
                for dx in 0..2 {
                    buf[y * width + x + dx] = 16;
                }
                if step % 2 == 0 && (y == h0 || y == (h0 + h1) / 2 || y == h1 - 1) {
                    for dx in 2..7 {
                        buf[y * width + x + dx] = 24;
                    }
                }
                if step % 3 == 0 {
                    for dx in 6..8 {
                        buf[y * width + x + dx] = 28;
                    }
                }
            }
            x += if step % 5 == 4 { 18 } else { 11 };
            step += 1;
        }
        buf
    }

    #[test]
    fn indra_native_matches_ort_and_benchmarks() {
        let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let runtime_dir = manifest_dir.parent().unwrap().join("runtime");
        let ort_dll = runtime_dir.join("onnx/onnxruntime.dll");
        if !ort_dll.is_file() {
            eprintln!("skipping ORT parity test: {} not found", ort_dll.display());
            return;
        }
        legal_pdf_core::init_ort_runtime(&ort_dll, None).expect("init ort runtime");

        let weights = PrepackedWeights::new();
        let browser_followup = manifest_dir
            .parent()
            .unwrap()
            .parent()
            .unwrap()
            .join("legal-browser-ocr/assets/indra-followup");
        let models = [
            (
                "student8cropids",
                browser_followup.join("student8cropids/model.onnx"),
                browser_followup.join("student8cropids/codec.json"),
            ),
            (
                "student10c64cropids",
                browser_followup.join("student10c64cropids/model.onnx"),
                browser_followup.join("student10c64cropids/codec.json"),
            ),
            (
                "wise_a075",
                runtime_dir.join("indra/assets/markers/model.onnx"),
                runtime_dir.join("indra/assets/markers/codec.json"),
            ),
        ];

        for (label, model_path, codec_path) in &models {
            if !model_path.is_file() || !codec_path.is_file() {
                continue;
            }
            let codec = Codec::load(codec_path).expect("load codec");
            let native = IndraNative::from_onnx_path(model_path).expect("load IndraNative");
            let mut session = open_session(model_path, 1, OrtBackend::Cpu, "cpu", true, &weights, false)
                .expect("open ORT session");
            let mut scratch = IndraScratch::default();

            for &width in &[128usize, 133, 320, 640] {
                let u8_line = synthetic_text_line(width);
                let maximum = width.div_ceil(8) * 8;
                let mut f32_input = vec![0.0_f32; INPUT_HEIGHT * maximum];
                for r in 0..INPUT_HEIGHT {
                    crate::simd_kernel::u8_to_norm_f32_row(
                        &u8_line[r * width..(r + 1) * width],
                        &mut f32_input[r * maximum..r * maximum + width],
                    );
                }

                let width_arr = [width as i64];
                let image = TensorRef::from_array_view(([1usize, 1, INPUT_HEIGHT, maximum], f32_input.as_slice())).unwrap();
                let widths = TensorRef::from_array_view(([1usize], width_arr.as_slice())).unwrap();
                let outputs = session.run(ort::inputs!["image" => image, "widths" => widths]).unwrap();
                let (l_shape, l_data) = outputs["logits"].try_extract_tensor::<f32>().unwrap();
                let (_, ids_data) = outputs["class_ids"].try_extract_tensor::<i64>().unwrap();
                let (_, len_data) = outputs["lengths"].try_extract_tensor::<i64>().unwrap();

                let ort_len = len_data[0] as usize;
                let ort_t = l_shape[1] as usize;
                let vocab = l_shape[2] as usize;

                let (nat_t_f32, nat_logits) = native.forward_logits_f32(&f32_input, maximum, width, &mut scratch);
                assert_eq!(nat_t_f32, ort_len, "{label} width={width}: timesteps mismatch");
                assert_eq!(vocab, native.vocab_size());

                let mut max_diff = 0.0_f32;
                for t in 0..ort_len {
                    for c in 0..vocab {
                        let d = (nat_logits[t * vocab + c] - l_data[t * vocab + c]).abs();
                        if d > max_diff {
                            max_diff = d;
                        }
                    }
                }
                assert!(
                    max_diff < 2e-4,
                    "{label} width={width}: max logit diff {max_diff} >= 2e-4 (ort_t={ort_t})"
                );

                let (nat_t_u8, step_labels, ctc_spans) = native.recognize_u8(&u8_line, width, true, &mut scratch);
                assert_eq!(nat_t_u8, ort_len);
                let (fast_t_u8, fast_labels, _) = native.recognize_u8(&u8_line, width, false, &mut scratch);
                assert_eq!(fast_t_u8, ort_len);
                assert_eq!(fast_labels, step_labels, "{label} width={width}: shortlist class_ids mismatch");
                for t in 0..ort_len {
                    let ort_id = usize::try_from(ids_data[t]).unwrap();
                    let nat_id = step_labels[t].unwrap_or(0);
                    assert_eq!(nat_id, ort_id, "{label} width={width} t={t}: class_id mismatch");
                }
                let ort_text = codec.decode(&greedy_ids(ids_data, 0, ort_t, ort_len)).0;
                let (nat_text, nat_conf) = codec.decode(&ctc_spans);
                assert_eq!(nat_text, ort_text, "{label} width={width}: decoded text mismatch");
                let ort_conf = codec.decode(&greedy_ctc_and_steps(l_data, 0, vocab, ort_t, ort_len).0).1;
                assert!(
                    (nat_conf - ort_conf).abs() < 1e-4,
                    "{label} width={width}: confidence diff {} vs {}",
                    nat_conf,
                    ort_conf
                );
                eprintln!("[parity] {label} width={width}: max_logit_diff={max_diff:.2e} text={nat_text:?}");
            }

            if matches!(*label, "student8cropids" | "wise_a075") {
                let width = 320usize;
                let u8_line = synthetic_text_line(width);
                let mut f32_input = vec![0.0_f32; INPUT_HEIGHT * width];
                for r in 0..INPUT_HEIGHT {
                    crate::simd_kernel::u8_to_norm_f32_row(
                        &u8_line[r * width..(r + 1) * width],
                        &mut f32_input[r * width..(r + 1) * width],
                    );
                }
                let run_opts = RunOptions::new()
                    .unwrap()
                    .with_outputs(OutputSelector::no_default().with("class_ids").with("lengths"));
                let width_arr = [width as i64];
                for _ in 0..10 {
                    let image = TensorRef::from_array_view(([1usize, 1, INPUT_HEIGHT, width], f32_input.as_slice())).unwrap();
                    let widths = TensorRef::from_array_view(([1usize], width_arr.as_slice())).unwrap();
                    let _ = session.run_with_options(ort::inputs!["image" => image, "widths" => widths], &run_opts).unwrap();
                    let _ = native.recognize_u8(&u8_line, width, false, &mut scratch);
                }
                let iters = 150usize;
                let t0 = Instant::now();
                for _ in 0..iters {
                    for r in 0..INPUT_HEIGHT {
                        crate::simd_kernel::u8_to_norm_f32_row(
                            &u8_line[r * width..(r + 1) * width],
                            &mut f32_input[r * width..(r + 1) * width],
                        );
                    }
                    let image = TensorRef::from_array_view(([1usize, 1, INPUT_HEIGHT, width], f32_input.as_slice())).unwrap();
                    let widths = TensorRef::from_array_view(([1usize], width_arr.as_slice())).unwrap();
                    let outputs = session.run_with_options(ort::inputs!["image" => image, "widths" => widths], &run_opts).unwrap();
                    let _ = outputs["class_ids"].try_extract_tensor::<i64>().unwrap();
                }
                let ort_ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;

                let t1 = Instant::now();
                for _ in 0..iters {
                    let _ = native.recognize_u8(&u8_line, width, false, &mut scratch);
                }
                let native_ms = t1.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                eprintln!(
                    "[bench] {label} single-line width={width}: ORT={ort_ms:.3}ms/line, IndraNative={native_ms:.3}ms/line, speedup={:.2}x",
                    ort_ms / native_ms.max(1e-9)
                );
            }
        }
    }

    #[test]
    fn triangle_downscaler_matches_imageops_resize() {
        let mut page = GrayImage::from_pixel(900, 120, Luma([255]));
        for y in 20..88usize {
            for x in 30..820usize {
                if (x / 7 + y / 5) % 2 == 0 {
                    page.put_pixel(x as u32, y as u32, Luma([((x * 13 + y * 7) % 180) as u8]));
                }
            }
        }
        let bbox = LineBox { left: 20, top: 15, right: 830, bottom: 95, recovered: false };
        let stride = page.width() as usize;
        let raw = page.as_raw();
        let (ix0, iy0, ix1, iy1) = crate::simd_kernel::find_ink_bbox(raw, stride, bbox.left, bbox.top, bbox.width(), bbox.height(), 220).unwrap();
        let px = ((iy1 - iy0) as f64 * 0.3).ceil() as usize;
        let py = ((iy1 - iy0) as f64 * VERTICAL_MARGIN).ceil() as usize;
        let cw = ix1 - ix0;
        let ch = iy1 - iy0;
        let pw = cw + 2 * px;
        let ph = ch + 2 * py;
        assert!(ph > 48);
        let dst_w = (pw as f32 * INPUT_HEIGHT as f32 / ph as f32).round().clamp(8.0, 4096.0) as usize;

        let mut padded = GrayImage::from_pixel(pw as u32, ph as u32, Luma([255]));
        for row in 0..ch {
            let s = (bbox.top + iy0 + row) * stride + bbox.left + ix0;
            let d = (py + row) * pw + px;
            padded.as_mut()[d..d + cw].copy_from_slice(&raw[s..s + cw]);
        }
        let expected = imageops::resize(&padded, dst_w as u32, INPUT_HEIGHT as u32, imageops::FilterType::Triangle).into_raw();

        let mut scratch = crate::simd_kernel::DownsampleScratch::default();
        let actual = crate::simd_kernel::resize_cropped_padded_triangle_u8(
            raw,
            stride,
            bbox.left + ix0,
            bbox.top + iy0,
            cw,
            ch,
            px,
            py,
            dst_w,
            INPUT_HEIGHT,
            &mut scratch,
        );
        assert_eq!(actual.len(), expected.len());
        let max_px_diff = actual.iter().zip(&expected).map(|(&a, &b)| a.abs_diff(b)).max().unwrap_or(0);
        assert!(max_px_diff <= 1, "max triangle resize pixel diff {max_px_diff} > 1");
    }

    #[test]
    fn gray_and_ink_of_monochrome_and_color() {
        let mut mono = Vec::with_capacity(8 * 4);
        for idx in 0..8usize {
            let v = (idx * 31) as u8;
            mono.extend_from_slice(&[v, v, v, 255]);
        }
        let (gray, ink) = crate::ocr::gray_and_ink_of_rgba(&mono);
        assert!(ink.is_none(), "monochrome rgba should skip duplicate ink buffer");
        assert_eq!(gray, crate::ocr::gray_of_rgba(&mono));

        let colored: Vec<u8> = [255u8, 255, 255, 255, 200, 100, 40, 255, 50, 50, 50, 255].into();
        let (gray_c, ink_c) = crate::ocr::gray_and_ink_of_rgba(&colored);
        let ink_c = ink_c.expect("colored rgba should compute separate ink buffer");
        assert_eq!(gray_c, crate::ocr::gray_of_rgba(&colored));
        assert_eq!(ink_c[0], 255);
        assert_eq!(ink_c[1], ((200 + 100 + 40 + 1) / 3) as u8);
        assert_eq!(ink_c[2], 50);
    }
}
