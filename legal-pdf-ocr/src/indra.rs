//! Indra, the recognizer: a PDF's pages or grey images read into lines and words. Pages are drawn,
//! stood upright and turned grey once; the line detector, line preparation, the separator and the
//! region checks all read that grey image.
use crate::line_detector::LineDetector;
use crate::model_pack::ModelPack;
use crate::orientation::{place_back, turn, Orientation};
use crate::planning::{self, RuntimePlan};
use image::{imageops, GrayImage, Luma, RgbaImage};
use legal_pdf_core::profile::measure;
use legal_pdf_core::render::{open_pdf, PageRenderer};
use legal_pdf_core::{provider_asset_sha256, Error, OrtBackend, Result};
use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord};
use ort::{
    execution_providers::{cuda::CuDNNConvAlgorithmSearch, CPUExecutionProvider, CUDAExecutionProvider},
    session::{
        builder::GraphOptimizationLevel,
        builder::PrepackedWeights,
        run_options::{OutputSelector, RunOptions},
        Session,
    },
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{sync_channel, Receiver};
use std::sync::Mutex;

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

pub struct OcrProvider {
    sessions: Vec<Session>,
    codec: Codec,
    dpi: u16,
    layouts: Vec<LineDetector>,
    plan: RuntimePlan,
    orientation: Option<Orientation>,
    identity: String,
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
        let (layouts, sessions) = std::thread::scope(|scope| -> Result<_> {
            // Independent model pools can initialize together; recognition weights stay shared.
            let layouts = scope.spawn(|| {
                (0..plan.layout_workers)
                    .map(|_| LineDetector::open(&pack.detector, pack.detector_side, plan.backend, &plan.device, independent_pool))
                    .collect::<Result<Vec<_>>>()
            });
            let weights = PrepackedWeights::new();
            let sessions = (0..plan.workers)
                .map(|_| open_session(&pack.model, plan.threads, plan.backend, &plan.device, options.cpu_arena, &weights, independent_pool))
                .collect::<Result<Vec<_>>>();
            let layouts = layouts.join().map_err(|_| Error::Message("layout initialization worker panicked".into()))??;
            Ok((layouts, sessions?))
        })?;
        Ok(Self {
            sessions,
            codec: Codec::load(&pack.codec)?,
            dpi: options.dpi,
            layouts,
            orientation: pack.orientation.as_deref().map(Orientation::open).transpose()?,
            plan: plan.clone(),
            identity: prepared.identity.clone(),
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
        // Balance bounded windows so overlap never leaves a one-page tail.
        let window_size = if images.len() <= 4 { images.len() } else { preparation_window_size(images.len()) };
        let Self { sessions, codec, layouts, plan, .. } = self;
        std::thread::scope(|scope| {
            let (sender, receiver) = sync_channel(1);
            scope.spawn(move || {
                for window in images.chunks(window_size) {
                    let value = prepare_window(layouts, window);
                    let failed = value.is_err();
                    if sender.send(value).is_err() || failed {
                        break;
                    }
                }
            });
            let mut pages = Vec::with_capacity(images.len());
            for window in receiver {
                pages.extend(recognize_pages(sessions, codec, plan, window?)?);
            }
            Ok(pages)
        })
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
        let Self { sessions, codec, layouts, plan, orientation, .. } = self;
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
                    let value = window.and_then(|pages: Vec<(Placement, GrayImage)>| {
                        let (placements, images): (Vec<_>, Vec<_>) = pages.into_iter().unzip();
                        let mut prepared = prepare_window(layouts, &images)?;
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
            place_pages(&mut results, prepared_receiver, sessions, codec, plan)
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
    sessions: &mut [Session],
    codec: &Codec,
    plan: &RuntimePlan,
) -> Result<()> {
    let mut at = 0;
    for window in windows {
        let (placements, prepared) = window?;
        for (placement, mut lines) in placements.into_iter().zip(recognize_pages(sessions, codec, plan, prepared)?) {
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

/// A window's images' lines found and prepared for reading.
fn prepare_window(layouts: &mut [LineDetector], images: &[GrayImage]) -> Result<Vec<Vec<PreparedLine>>> {
    let boxes = measure("ocr.layout", || layout_boxes(layouts, images))?;
    measure("ocr.prepare_lines", || prepare_pages(images, &boxes))
}

/// Pages' prepared lines read, each kept where it reads as text: a line the page analysis left
/// out only when the recognizer is sure of it.
fn recognize_pages(sessions: &mut [Session], codec: &Codec, plan: &RuntimePlan, pages: Vec<Vec<PreparedLine>>) -> Result<Vec<Vec<OcrLine>>> {
    let recognized = measure("ocr.recognize", || recognize_prepared_pages(sessions, codec, plan, &pages))?;
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
fn layout_boxes(layouts: &mut [LineDetector], images: &[GrayImage]) -> Result<Vec<Vec<LineBox>>> {
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
                    let value = layout.lines(&images[index]);
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

/// Pages' prepared lines read by the recognizer's sessions, batched by width.
fn recognize_prepared_pages(
    sessions: &mut [Session],
    codec: &Codec,
    plan: &RuntimePlan,
    pages: &[Vec<PreparedLine>],
) -> Result<Vec<Vec<Recognition>>> {
    let counts = pages.iter().map(Vec::len).collect::<Vec<_>>();
    let lines = pages.iter().flatten().collect::<Vec<_>>();
    let jobs = recognition_jobs(&lines, plan.batch_size, plan.width_bucket);
    if jobs.is_empty() {
        return Ok(counts.into_iter().map(|_| Vec::new()).collect());
    }
    let workers = jobs.len().min(sessions.len());
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let handles = sessions
            .iter_mut()
            .take(workers)
            .map(|session| {
                let sender = sender.clone();
                let (next, jobs, lines) = (&next, &jobs, &lines);
                scope.spawn(move || {
                    let mut pixels = Vec::new();
                    let mut lengths = Vec::new();
                    loop {
                        let index = next.fetch_add(1, Ordering::Relaxed);
                        if index >= jobs.len() {
                            break;
                        }
                        let value = recognize_batch(session, codec, lines, &jobs[index], &mut pixels, &mut lengths);
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
        Ok(counts.into_iter().map(|count| output.by_ref().take(count).collect()).collect())
    })
}

fn recognition_jobs(lines: &[&PreparedLine], batch_size: usize, width_bucket: usize) -> Vec<Vec<usize>> {
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, line) in lines.iter().enumerate() {
        groups.entry(line.width.div_ceil(width_bucket)).or_default().push(index);
    }
    let mut jobs = groups
        .into_values()
        .flat_map(|indices| indices.chunks(batch_size).map(<[usize]>::to_vec).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    jobs.sort_unstable_by_key(|job| std::cmp::Reverse(job.len() * lines[job[0]].width.div_ceil(width_bucket)));
    jobs
}

/// One batch of lines read: each with its text, confidence and words.
fn recognize_batch(
    session: &mut Session,
    codec: &Codec,
    lines: &[&PreparedLine],
    batch: &[usize],
    pixels: &mut Vec<f32>,
    lengths: &mut Vec<i64>,
) -> Result<Vec<(usize, Recognition)>> {
    let maximum = batch.iter().map(|&index| lines[index].width).max().unwrap_or(1).div_ceil(8) * 8;
    pixels.resize(batch.len() * INPUT_HEIGHT * maximum, 0.0);
    pixels.fill(0.0);
    lengths.clear();
    for (local, &index) in batch.iter().enumerate() {
        let line = lines[index];
        lengths.push(line.width as i64);
        for row in 0..INPUT_HEIGHT {
            let source = row * line.width;
            let target = (local * INPUT_HEIGHT + row) * maximum;
            for (&value, pixel) in line.pixels[source..source + line.width].iter().zip(&mut pixels[target..target + line.width]) {
                *pixel = 1.0 - f32::from(value) / 255.0;
            }
        }
    }
    let image = TensorRef::from_array_view(([batch.len(), 1, INPUT_HEIGHT, maximum], pixels.as_slice())).map_err(ort_error)?;
    let sequence_lengths = TensorRef::from_array_view(([batch.len()], lengths.as_slice())).map_err(ort_error)?;
    let use_ids = !batch.iter().any(|&index| lines[index].bbox.recovered || lines[index].confidence_required);
    let run_options = RunOptions::new()
        .map_err(ort_error)?
        .with_outputs(OutputSelector::no_default().with(if use_ids { "class_ids" } else { "logits" }).with("lengths"));
    let outputs = session
        .run_with_options(ort::inputs!["image" => image, "widths" => sequence_lengths], &run_options)
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
            let (text, confidence) = codec.decode(&greedy_ctc(logits, local, classes, timesteps, length));
            let steps =
                best_labels(logits, local, classes, timesteps, length).into_iter().map(|label| (label != BLANK_LABEL).then_some(label));
            let words = line_words(codec, lines[index], &emissions(steps), length);
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
) -> Result<Vec<(Placement, GrayImage)>> {
    // White margin around a part of a page, so the layout reads its first and last lines whole.
    const MARGIN: usize = 16;
    let mut rendered = Vec::new();
    for request in requests {
        let pixmap = pages
            .render(request.page_index)
            .ok_or_else(|| Error::Message(format!("OCR renderer could not draw page {}", request.page_index + 1)))?;
        let (width, height) = (usize::from(pixmap.width()), usize::from(pixmap.height()));
        let scale = [request.width / width as f64, request.height / height as f64];
        let gray = GrayImage::from_raw(width as u32, height as u32, crate::ocr::gray_of(&pixmap))
            .expect("renderer dimensions match its pixel buffer");
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
            let image = if turns == 0 { gray } else { turn(&gray, turns) };
            let drawn = [width as f64, height as f64];
            rendered.push((Placement { page_index: request.page_index, separator, scale, origin: [0.0, 0.0], turns, drawn }, image));
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
            let mut crop = GrayImage::from_pixel((x1 - x0 + 2 * MARGIN) as u32, (y1 - y0 + 2 * MARGIN) as u32, Luma([255]));
            let part = imageops::crop_imm(&gray, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image();
            imageops::replace(&mut crop, &part, MARGIN as i64, MARGIN as i64);
            let origin = [(x0 as f64 - MARGIN as f64) * scale[0], (y0 as f64 - MARGIN as f64) * scale[1]];
            rendered.push((Placement { page_index: request.page_index, separator: None, scale, origin, turns: 0, drawn: [0.0, 0.0] }, crop));
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

pub(crate) fn open_session(
    model: &Path,
    threads: usize,
    backend: OrtBackend,
    device: &str,
    cpu_arena: bool,
    prepacked_weights: &PrepackedWeights,
    independent_pool: bool,
) -> Result<Session> {
    let mut builder = Session::builder()
        .map_err(ort_error)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_error)?
        .with_intra_threads(threads)
        .map_err(ort_error)?
        .with_inter_threads(1)
        .and_then(|builder| builder.with_prepacked_weights(prepacked_weights))
        .map_err(ort_error)?;
    if independent_pool {
        builder = builder.with_independent_thread_pool().map_err(ort_error)?;
    }
    if backend == OrtBackend::DirectMl {
        builder = builder.with_memory_pattern(false).and_then(|builder| builder.with_parallel_execution(false)).map_err(ort_error)?;
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
    builder.commit_from_file(model).map_err(ort_error)
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

fn required_path(value: &Option<PathBuf>, variable: &str, label: &str) -> Result<PathBuf> {
    value
        .clone()
        .or_else(|| std::env::var_os(variable).map(PathBuf::from))
        .filter(|path| !path.as_os_str().is_empty())
        .ok_or_else(|| Error::Message(format!("{label} is required; pass it explicitly or set {variable}")))
}

pub(crate) fn canonical_file(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::Message(format!("{label} does not exist: {}", path.display())));
    }
    fs::canonicalize(path).map_err(|source| Error::io(path, source))
}

pub(crate) fn ort_error(error: ort::Error) -> Error {
    Error::Message(format!("Indra ONNX inference failed: {error}"))
}

/// A line of a grey page cut out and prepared for reading: its ink's bounds padded with white (a
/// third of its height at each side, `VERTICAL_MARGIN` of it above and below) and scaled to the
/// recognizer's height.
fn prepare_line(image: &GrayImage, bbox: LineBox) -> Option<PreparedLine> {
    if bbox.width() == 0 || bbox.height() == 0 {
        return None;
    }
    let crop = imageops::crop_imm(image, bbox.left as u32, bbox.top as u32, bbox.width() as u32, bbox.height() as u32).to_image();
    let (mut x0, mut y0, mut x1, mut y1) = (crop.width(), crop.height(), 0, 0);
    for (x, y, pixel) in crop.enumerate_pixels() {
        if pixel[0] < 220 {
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x + 1);
            y1 = y1.max(y + 1);
        }
    }
    let (px, py) = if x1 > x0 {
        (((y1 - y0) as f64 * 0.3).ceil() as u32, ((y1 - y0) as f64 * VERTICAL_MARGIN).ceil() as u32)
    } else {
        (x0, y0, x1, y1) = (0, 0, crop.width(), crop.height());
        (0, 0)
    };
    let padded_width = x1 - x0 + 2 * px;
    let padded_height = y1 - y0 + 2 * py;
    let mut padded = GrayImage::from_pixel(padded_width, padded_height, Luma([255]));
    imageops::replace(&mut padded, &imageops::crop_imm(&crop, x0, y0, x1 - x0, y1 - y0).to_image(), i64::from(px), i64::from(py));
    let width = (padded_width as f32 * INPUT_HEIGHT as f32 / padded_height as f32).round().clamp(8.0, 4096.0) as usize;
    let resized = imageops::resize(&padded, width as u32, INPUT_HEIGHT as u32, imageops::FilterType::Triangle);
    Some(PreparedLine {
        bbox,
        confidence_required: false,
        width,
        pixels: resized.into_raw(),
        pixel_x_origin: bbox.left as f64 + x0 as f64 - px as f64,
        pixel_x_scale: padded_width as f64 / width as f64,
    })
}

/// Each image's lines prepared, the images shared among the machine's threads.
fn prepare_pages(images: &[GrayImage], boxes: &[Vec<LineBox>]) -> Result<Vec<Vec<PreparedLine>>> {
    let workers = images.len().min(std::thread::available_parallelism().map_or(1, |value| value.get()).max(1));
    let next = AtomicUsize::new(0);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let handles = (0..workers)
            .map(|_| {
                let sender = sender.clone();
                let next = &next;
                scope.spawn(move || loop {
                    let index = next.fetch_add(1, Ordering::Relaxed);
                    if index >= images.len() {
                        break;
                    }
                    let lines = boxes[index].iter().filter_map(|&bbox| prepare_line(&images[index], bbox)).collect();
                    if sender.send((index, lines)).is_err() {
                        break;
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

/// Each timestep's most likely label, as `greedy_ctc` reads it.
fn best_labels(logits: &[f32], batch: usize, classes: usize, timesteps: usize, length: usize) -> Vec<usize> {
    logits[batch * timesteps * classes..][..length * classes]
        .chunks_exact(classes)
        .map(|step| {
            step.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (label, &value)| if value > best.1 { (label, value) } else { best })
                .0
        })
        .collect()
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

fn greedy_ctc(logits: &[f32], batch: usize, classes: usize, timesteps: usize, length: usize) -> Vec<(usize, f32)> {
    let labels = best_labels(logits, batch, classes, timesteps, length);
    let denominators = logits[batch * timesteps * classes..][..length * classes]
        .chunks_exact(classes)
        .zip(&labels)
        .map(|(step, &label)| step.iter().map(|&value| (value - step[label]).exp()).sum::<f32>())
        .collect::<Vec<_>>();
    let mut output = Vec::new();
    let mut previous = usize::MAX;
    for (timestep, &label) in labels.iter().enumerate() {
        if label != previous && label != BLANK_LABEL {
            let confidence = 1.0 / denominators[timestep].max(f32::MIN_POSITIVE);
            output.push((label, confidence));
        } else if label == previous && label != BLANK_LABEL {
            if let Some((_, score)) = output.last_mut() {
                let confidence = 1.0 / denominators[timestep].max(f32::MIN_POSITIVE);
                *score = score.max(confidence);
            }
        }
        previous = label;
    }
    output
}
