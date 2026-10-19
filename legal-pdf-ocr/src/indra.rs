use crate::line_detector::LineDetector;
use crate::model_pack::ModelPack;
use crate::orientation::{place_back, turn, Orientation};
use crate::planning::{self, RuntimePlan};
#[cfg(feature = "ocr")]
use hayro::hayro_interpret::InterpreterSettings;
#[cfg(feature = "ocr")]
use hayro::hayro_syntax::Pdf;
#[cfg(feature = "ocr")]
use hayro::vello_cpu::color::palette::css::WHITE;
#[cfg(feature = "ocr")]
use hayro::{render, RenderCache, RenderSettings};
use image::{imageops, GrayImage, ImageReader, RgbaImage};
use legal_pdf_core::{provider_asset_sha256, Error, Result};
use legal_pdf_core::{OcrLine, OcrPageRequest, OcrPageResult, OcrWord};
use ort::{
    execution_providers::{
        cuda::CuDNNConvAlgorithmSearch, CPUExecutionProvider, CUDAExecutionProvider,
    },
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
use std::time::Instant;

// Store padding in the shared batch tensor, not in every prepared line.
const INPUT_PADDING: usize = 0;
const BLANK_LABEL: usize = 0;
/// How sure the recognizer must be of a line the page analysis left out: a number or a word reads
/// at 0.95 and more; a signature, a seal's lettering or a speckle at less.
pub(crate) const RECOVERED_CONFIDENCE: f64 = 0.95;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IndraLayout {
    #[default]
    Indra,
    Tesseract,
}

impl IndraLayout {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "indra" => Some(Self::Indra),
            "tesseract" => Some(Self::Tesseract),
            _ => None,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Indra => "indra",
            Self::Tesseract => "tesseract",
        }
    }
}

pub use legal_pdf_core::OrtBackend as IndraBackend;

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
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "quality" => Some(Self::Quality),
            "balanced" => Some(Self::Balanced),
            "fast" => Some(Self::Fast),
            "turbo" => Some(Self::Turbo),
            "extreme" => Some(Self::Extreme),
            _ => None,
        }
    }
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
pub struct IndraOptions {
    pub model_pack: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub tesseract_library: Option<PathBuf>,
    pub dpi: u16,
    pub threads: usize,
    pub workers: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub width_bucket: usize,
    #[serde(skip, default = "default_input_height")]
    pub input_height: usize,
    pub tier: IndraTier,
    pub layout: IndraLayout,
    /// Omitted selects a CPU/GPU plan from this machine's capabilities.
    pub backend: Option<IndraBackend>,
    pub device: Option<String>,
    pub cpu_arena: bool,
    pub expected_identity: Option<String>,
    pub orientation: Option<PathBuf>,
}
impl Default for IndraOptions {
    fn default() -> Self {
        Self {
            model_pack: None,
            runtime: None,
            tesseract_library: None,
            dpi: 144,
            threads: 0,
            workers: 0,
            layout_workers: 0,
            batch_size: 0,
            width_bucket: 0,
            input_height: 32,
            tier: IndraTier::Balanced,
            layout: IndraLayout::Indra,
            backend: None,
            device: None,
            cpu_arena: true,
            expected_identity: None,
            orientation: None,
        }
    }
}
fn default_input_height() -> usize {
    32
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

#[derive(Debug, Clone, Serialize)]
pub struct IndraImageDiagnostics {
    pub lines: Vec<OcrLine>,
    pub layout_boxes: Vec<[usize; 4]>,
    pub layout_seconds: f64,
    pub recognition_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct IndraBatchPerformance {
    pub detailed: bool,
    pub pages: usize,
    pub input_pixels: usize,
    pub layout_boxes: usize,
    pub prepared_lines: usize,
    pub output_lines: usize,
    pub layout_seconds: f64,
    pub render_seconds: f64,
    pub line_prepare_seconds: f64,
    pub schedule_seconds: f64,
    pub recognition_wall_seconds: f64,
    pub output_assembly_seconds: f64,
    pub total_seconds: f64,
    pub recognition_workers: usize,
    pub batches: usize,
    pub batch_fill_ratio: f64,
    pub tensor_fill_ratio: f64,
    pub line_width_p50: usize,
    pub line_width_p95: usize,
    pub line_width_max: usize,
    pub batch_lines_p50: usize,
    pub batch_lines_p95: usize,
    pub batch_lines_max: usize,
    pub batch_seconds_p50: f64,
    pub batch_seconds_p95: f64,
    pub batch_seconds_max: f64,
    pub worker_busy_seconds_min: f64,
    pub worker_busy_seconds_max: f64,
    pub worker_busy_seconds_sum: f64,
    pub pack_seconds_sum: f64,
    pub inference_seconds_sum: f64,
    pub decode_seconds_sum: f64,
    pub prepared_bytes: usize,
    pub peak_tensor_bytes: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct IndraBatchDiagnostics {
    pub pages: Vec<IndraImageDiagnostics>,
    pub performance: IndraBatchPerformance,
}

impl LineBox {
    pub(crate) fn width(self) -> usize {
        self.right.saturating_sub(self.left)
    }

    pub(crate) fn height(self) -> usize {
        self.bottom.saturating_sub(self.top)
    }
}

struct PreparedLine {
    bbox: LineBox,
    confidence_required: bool,
    width: usize,
    pixels: Vec<u8>,
    pixel_x_origin: f64,
    pixel_x_scale: f64,
}

#[derive(Clone, Copy)]
enum PagePixels<'a> {
    Gray(&'a GrayImage),
    Rgba {
        pixels: &'a [u8],
        width: u32,
        height: u32,
    },
}

impl PagePixels<'_> {
    fn width(self) -> u32 {
        match self {
            Self::Gray(image) => image.width(),
            Self::Rgba { width, .. } => width,
        }
    }

    fn height(self) -> u32 {
        match self {
            Self::Gray(image) => image.height(),
            Self::Rgba { height, .. } => height,
        }
    }

    fn layout(self, engine: &mut LineDetector) -> Result<Vec<LineBox>> {
        match self {
            Self::Gray(image) => engine.lines(image),
            Self::Rgba {
                pixels,
                width,
                height,
            } => engine.lines_rgba(pixels, width, height),
        }
    }
}

#[derive(Default)]
struct RecognitionPerformance {
    schedule_seconds: f64,
    wall_seconds: f64,
    workers: usize,
    batches: usize,
    batch_fill_ratio: f64,
    tensor_fill_ratio: f64,
    line_width_p50: usize,
    line_width_p95: usize,
    line_width_max: usize,
    batch_lines_p50: usize,
    batch_lines_p95: usize,
    batch_lines_max: usize,
    batch_seconds_p50: f64,
    batch_seconds_p95: f64,
    batch_seconds_max: f64,
    worker_busy_seconds_min: f64,
    worker_busy_seconds_max: f64,
    worker_busy_seconds_sum: f64,
    pack_seconds_sum: f64,
    inference_seconds_sum: f64,
    decode_seconds_sum: f64,
    peak_tensor_bytes: usize,
    tensor_elements: usize,
    useful_elements: usize,
    batch_lines_values: Vec<usize>,
    batch_seconds_values: Vec<f64>,
    worker_busy_values: Vec<f64>,
}

struct PreparedWindow {
    layout_boxes: Vec<Vec<[usize; 4]>>,
    prepared: Vec<Vec<PreparedLine>>,
    layout_seconds: f64,
    line_prepare_seconds: f64,
}

struct PreparedDiagnostics {
    pages: Vec<IndraImageDiagnostics>,
    layout_seconds: f64,
    line_prepare_seconds: f64,
    layout_box_count: usize,
    total_lines: usize,
    prepared_bytes: usize,
    output_assembly_seconds: f64,
    recognition: RecognitionPerformance,
}

/// A line's recognized text, its confidence and where its words stand on the page.
struct Recognition {
    text: String,
    confidence: f64,
    words: Vec<OcrWord>,
}

struct RecognizedBatch {
    values: Vec<(usize, Recognition)>,
    worker: usize,
    lines: usize,
    tensor_elements: usize,
    useful_elements: usize,
    total_seconds: f64,
    pack_seconds: f64,
    inference_seconds: f64,
    decode_seconds: f64,
}

struct Codec {
    labels: BTreeMap<usize, Vec<(Vec<usize>, String)>>,
}

impl Codec {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        let mapping: BTreeMap<String, Vec<usize>> = serde_json::from_reader(BufReader::new(file))?;
        if mapping.is_empty() || mapping.values().any(Vec::is_empty) {
            return Err(Error::Message(
                "Indra codec is empty or malformed".to_owned(),
            ));
        }
        let mut labels = mapping
            .into_iter()
            .map(|(text, labels)| (labels, text))
            .collect::<Vec<_>>();
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
            grouped
                .entry(sequence[0])
                .or_default()
                .push((sequence, text));
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
            let score = spans[index..index + sequence.len()]
                .iter()
                .map(|(_, score)| f64::from(*score))
                .sum::<f64>()
                / sequence.len() as f64;
            text.push_str(value);
            confidence += score * value.chars().count() as f64;
            characters += value.chars().count();
            index += sequence.len();
        }
        (
            text,
            if characters == 0 {
                0.0
            } else {
                confidence / characters as f64
            },
        )
    }
}

pub struct IndraOcr {
    sessions: Vec<Session>,
    codec: Option<Codec>,
    dpi: u16,
    batch_size: usize,
    width_bucket: usize,
    input_height: usize,
    width_scale: f32,
    layouts: Vec<LineDetector>,
    plan: RuntimePlan,
    last_performance: Option<IndraBatchPerformance>,
    orientation: Option<Orientation>,
    identity: String,
    name: String,
}

pub(crate) struct PreparedIndra {
    pub(crate) pack: ModelPack,
    pub(crate) runtime: PathBuf,
    pub(crate) tesseract_library: Option<PathBuf>,
    pub(crate) plan: RuntimePlan,
    pub(crate) ready: Option<Box<IndraOcr>>,
    pub(crate) identity: String,
    pub(crate) name: String,
}
impl PreparedIndra {
    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }
    pub(crate) fn name(&self) -> &str {
        &self.name
    }
}
impl IndraOcr {
    pub fn new(options: &IndraOptions) -> Result<Self> {
        Self::from_prepared(options, Self::prepare(options)?)
    }
    pub(crate) fn prepare(options: &IndraOptions) -> Result<PreparedIndra> {
        validate_options(options)?;
        let root = required_path(
            &options.model_pack,
            "LEGALPDF_INDRA_MODEL_PACK",
            "Indra model pack",
        )?;
        let runtime = canonical_file(
            &required_path(
                &options.runtime,
                "LEGALPDF_ONNXRUNTIME",
                "ONNX Runtime library",
            )?,
            "ONNX Runtime library",
        )?;
        legal_pdf_core::init_ort_runtime(&runtime, None)?;
        let prepared = planning::prepare(options, &root, &runtime)?;
        if options
            .expected_identity
            .as_deref()
            .is_some_and(|expected| expected != prepared.identity)
        {
            return Err(Error::Message(
                "Indra identity changed before OCR began".into(),
            ));
        }
        Ok(prepared)
    }
    pub(crate) fn prepare_plan(
        options: &IndraOptions,
        root: &Path,
        runtime: PathBuf,
        plan: RuntimePlan,
    ) -> Result<PreparedIndra> {
        let device = &plan.device;
        let mut pack = ModelPack::open(root, options.tier, plan.backend)?;
        if let Some(path) = &options.orientation {
            pack.orientation = Some(canonical_file(path, "orientation model")?);
        }
        let tesseract_library = if options.layout == IndraLayout::Tesseract {
            Some(canonical_file(
                &required_path(
                    &options.tesseract_library,
                    "LEGALPDF_TESSERACT_LIBRARY",
                    "Tesseract layout library",
                )?,
                "Tesseract layout library",
            )?)
        } else {
            None
        };
        let layout_identity =
            provider_asset_sha256(tesseract_library.as_deref().unwrap_or(&pack.detector))?;
        let identity=format!("indra-native-v1:backend={}:device={device}:model={}:codec={}:runtime={}:layout={}:layout-model={layout_identity}:side={}:orientation={}:tier={}:dpi={}:batch={}:bucket={}:workers={}:threads={}:layout-workers={}:cpu-arena={}",
            plan.backend.name(),provider_asset_sha256(&pack.model)?,provider_asset_sha256(&pack.codec)?,provider_asset_sha256(&runtime)?,options.layout.name(),pack.detector_side,
            pack.orientation.as_deref().map(provider_asset_sha256).transpose()?.unwrap_or_default(),
            options.tier.name(),options.dpi,plan.batch_size,plan.width_bucket,plan.workers,plan.threads,plan.layout_workers,options.cpu_arena);
        Ok(PreparedIndra {
            pack,
            runtime,
            tesseract_library,
            plan,
            ready: None,
            name: identity.clone(),
            identity,
        })
    }
    pub(crate) fn from_prepared(
        options: &IndraOptions,
        mut prepared: PreparedIndra,
    ) -> Result<Self> {
        if options
            .expected_identity
            .as_deref()
            .is_some_and(|expected| expected != prepared.identity)
        {
            return Err(Error::Message(
                "Indra identity changed before OCR began".to_owned(),
            ));
        }
        if let Some(ready) = prepared.ready.take() {
            return Ok(*ready);
        }
        Self::open_prepared(options, &prepared)
    }
    pub(crate) fn open_prepared(options: &IndraOptions, prepared: &PreparedIndra) -> Result<Self> {
        let independent_pool = legal_pdf_core::init_ort_runtime(&prepared.runtime, None)?.is_some();
        let (plan, pack, tesseract_library) =
            (&prepared.plan, &prepared.pack, &prepared.tesseract_library);
        let (layouts, sessions) = std::thread::scope(|scope| -> Result<_> {
            // Independent model pools can initialize together; recognition weights stay shared.
            let layouts = scope.spawn(|| {
                (0..plan.layout_workers)
                    .map(|_| {
                        if let Some(path) = tesseract_library {
                            LineDetector::tesseract(path, options.dpi)
                        } else {
                            LineDetector::indra(
                                &pack.detector,
                                pack.detector_side,
                                plan.backend,
                                &plan.device,
                                independent_pool,
                            )
                        }
                    })
                    .collect::<Result<Vec<_>>>()
            });
            let weights = PrepackedWeights::new();
            let sessions = (0..prepared.plan.workers)
                .map(|_| {
                    open_session(
                        &prepared.pack.model,
                        prepared.plan.threads,
                        prepared.plan.backend,
                        &prepared.plan.device,
                        options.cpu_arena,
                        &weights,
                        independent_pool,
                    )
                })
                .collect::<Result<Vec<_>>>();
            let layouts = layouts
                .join()
                .map_err(|_| Error::Message("layout initialization worker panicked".into()))??;
            Ok((layouts, sessions?))
        })?;
        let codec = Codec::load(&prepared.pack.codec)?;
        let orientation = prepared
            .pack
            .orientation
            .as_deref()
            .map(Orientation::open)
            .transpose()?;
        Ok(Self {
            sessions,
            codec: Some(codec),
            dpi: options.dpi,
            batch_size: prepared.plan.batch_size,
            width_bucket: prepared.plan.width_bucket,
            input_height: 32,
            width_scale: 1.0,
            layouts,
            orientation,
            plan: prepared.plan.clone(),
            last_performance: None,
            identity: prepared.identity.clone(),
            name: prepared.name.clone(),
        })
    }
    pub fn last_performance(&self) -> Option<&IndraBatchPerformance> {
        self.last_performance.as_ref()
    }
    pub fn configuration(&self) -> &RuntimePlan {
        &self.plan
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn recognize_image(&mut self, path: impl AsRef<Path>) -> Result<Vec<OcrLine>> {
        let path = path.as_ref();
        let image = ImageReader::open(path)
            .map_err(|source| Error::io(path, source))?
            .decode()
            .map_err(|source| {
                Error::Message(format!("could not decode {}: {source}", path.display()))
            })?
            .into_luma8();
        self.recognize_gray_image(&image)
    }

    pub fn recognize_gray_image(&mut self, image: &GrayImage) -> Result<Vec<OcrLine>> {
        self.recognize_gray(image, 1.0, 1.0)
    }

    pub fn recognize_gray_image_diagnostics(
        &mut self,
        image: &GrayImage,
    ) -> Result<IndraImageDiagnostics> {
        self.diagnose_gray(image, 1.0, 1.0)
    }

    pub fn recognize_gray_images_diagnostics(
        &mut self,
        images: &[GrayImage],
    ) -> Result<Vec<IndraImageDiagnostics>> {
        self.diagnose_gray_many(images)
    }

    pub fn recognize_rgba_images_diagnostics(
        &mut self,
        images: &[RgbaImage],
    ) -> Result<Vec<IndraImageDiagnostics>> {
        self.diagnose_rgba_many(images)
    }

    pub fn recognize_rgba_images_profile(
        &mut self,
        images: &[RgbaImage],
    ) -> Result<IndraBatchDiagnostics> {
        let pages = images
            .iter()
            .map(|image| PagePixels::Rgba {
                pixels: image.as_raw(),
                width: image.width(),
                height: image.height(),
            })
            .collect::<Vec<_>>();
        self.diagnose_pages_profile(&pages)
    }

    pub fn recognize_gray_images_profile(
        &mut self,
        images: &[GrayImage],
    ) -> Result<IndraBatchDiagnostics> {
        self.diagnose_gray_many_profile(images)
    }

    pub fn warmup_gray_image(&mut self, image: &GrayImage) -> Result<()> {
        let boxes = self.layouts[0].lines(image)?;
        for layout in &mut self.layouts[1..] {
            let _ = layout.lines(image)?;
        }
        let lines = boxes
            .into_iter()
            .filter_map(|bbox| prepare_line(image, bbox, self.width_scale, self.input_height))
            .collect::<Vec<_>>();
        for session in &mut self.sessions {
            let _ = recognize_prepared(
                session,
                self.codec.as_ref().expect("fast Indra codec"),
                &lines,
                self.batch_size,
                self.width_bucket,
                self.input_height,
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "ocr")]
    pub fn extract_pages(
        &mut self,
        bytes: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        let document_started = Instant::now();
        self.last_performance = Some(IndraBatchPerformance::default());
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        // Separate bounded stages keep rendering from stalling detection and GPU recognition.
        let window_size = preparation_window_size(requests.len());
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        let occupied = self.plan.workers * self.plan.threads + self.plan.layout_workers;
        let render_workers = available
            .saturating_sub(occupied + 1)
            .clamp(1, 4)
            .min(window_size);
        let width_scale = self.width_scale;
        let input_height = self.input_height;
        let batch_size = self.batch_size;
        let width_bucket = self.width_bucket;
        let dpi = self.dpi;
        // Picture regions use real confidence to distinguish print from seals and chart lettering.
        let confidence_pages = requests
            .iter()
            .filter(|request| !request.regions.is_empty())
            .map(|request| request.page_index)
            .collect::<BTreeSet<_>>();
        let codec = self.codec.as_ref().expect("fast Indra codec");
        let layouts = &mut self.layouts;
        let sessions = &mut self.sessions;
        let orientation = self.orientation.as_mut();
        let (pages, performance) = std::thread::scope(|scope| -> Result<_> {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let (metadata_sender, metadata_receiver) = std::sync::mpsc::channel();
            let (render_sender, render_receiver) = std::sync::mpsc::sync_channel(1);
            let renderer = scope.spawn(move || {
                let orientation = orientation.map(std::sync::Mutex::new);
                let pdf = match Pdf::new(bytes.to_vec()) {
                    Ok(pdf) => pdf,
                    Err(error) => {
                        let _ = render_sender.send(Err(Error::Message(format!(
                            "OCR renderer could not open PDF: {error:?}"
                        ))));
                        return;
                    }
                };
                std::thread::scope(|render_scope| {
                    let mut queues = Vec::new();
                    for worker in 0..render_workers {
                        let (sender, receiver) = std::sync::mpsc::sync_channel(0);
                        queues.push(receiver);
                        let orientation = orientation.as_ref();
                        let pdf = &pdf;
                        render_scope.spawn(move || {
                            // Share the parsed PDF; Hayro's render caches stay local to each worker.
                            let cache = RenderCache::new();
                            let interpreter = InterpreterSettings::default();
                            let settings = pdf_render_settings(dpi);
                            for window in requests.chunks(window_size) {
                                let start = worker * window.len() / render_workers;
                                let end = (worker + 1) * window.len() / render_workers;
                                let started = Instant::now();
                                let rendered = render_pdf_pages(
                                    pdf,
                                    &cache,
                                    &interpreter,
                                    &settings,
                                    &window[start..end],
                                    orientation,
                                );
                                let failed = rendered.is_err();
                                let value =
                                    rendered.map(|pages| (pages, started.elapsed().as_secs_f64()));
                                if sender.send(value).is_err() || failed {
                                    break;
                                }
                            }
                        });
                    }
                    for _ in requests.chunks(window_size) {
                        let value = (|| -> Result<_> {
                            let (mut pages, mut seconds) = (Vec::new(), 0.0_f64);
                            for queue in &queues {
                                let (part, elapsed) = queue.recv().map_err(|_| {
                                    Error::Message(
                                        "PDF render worker stopped before returning its pages"
                                            .into(),
                                    )
                                })??;
                                pages.extend(part);
                                seconds = seconds.max(elapsed);
                            }
                            Ok((pages, seconds))
                        })();
                        let failed = value.is_err();
                        if render_sender.send(value).is_err() || failed {
                            break;
                        }
                    }
                    // Closing every rendezvous queue also unblocks workers if a downstream stage fails.
                    drop(queues);
                });
            });
            let preparer = scope.spawn(move || {
                for value in render_receiver {
                    let (rendered, render_seconds) = match value {
                        Ok(value) => value,
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            break;
                        }
                    };
                    let (metadata, page_images): (Vec<_>, Vec<_>) = rendered.into_iter().unzip();
                    let input_pixels = page_images
                        .iter()
                        .map(|image| image.width() as usize * image.height() as usize)
                        .sum::<usize>();
                    let images = page_images
                        .iter()
                        .map(|image| PagePixels::Rgba {
                            pixels: image.as_raw(),
                            width: image.width(),
                            height: image.height(),
                        })
                        .collect::<Vec<_>>();
                    let mut value =
                        Self::prepare_window(layouts, &images, width_scale, input_height);
                    if let Ok(window) = value.as_mut() {
                        for (metadata, lines) in metadata.iter().zip(&mut window.prepared) {
                            if confidence_pages.contains(&metadata.0) {
                                for line in lines {
                                    line.confidence_required = true;
                                }
                            }
                        }
                    }
                    let failed = value.is_err();
                    if !failed
                        && metadata_sender
                            .send((metadata, render_seconds, input_pixels))
                            .is_err()
                    {
                        break;
                    }
                    if sender.send(value).is_err() || failed {
                        break;
                    }
                }
            });
            let result = Self::recognize_prepared_windows(
                receiver,
                sessions,
                codec,
                batch_size,
                width_bucket,
                input_height,
                requests.len(),
            );
            renderer
                .join()
                .map_err(|_| Error::Message("PDF render worker panicked".into()))?;
            preparer
                .join()
                .map_err(|_| Error::Message("PDF preparation worker panicked".into()))?;
            let mut metadata = Vec::new();
            let (mut render_seconds, mut input_pixels) = (0.0, 0);
            for (window, seconds, pixels) in metadata_receiver {
                metadata.extend(window);
                render_seconds += seconds;
                input_pixels += pixels;
            }
            let diagnostics = finish_diagnostics(
                result?,
                input_pixels,
                document_started.elapsed().as_secs_f64(),
                render_seconds,
            );
            let pages = finish_recognized_pdf_pages(requests, metadata, diagnostics.pages)?;
            Ok((pages, diagnostics.performance))
        })?;
        self.last_performance = Some(performance);
        Ok(pages)
    }

    fn recognize_gray(
        &mut self,
        image: &GrayImage,
        x_scale: f64,
        y_scale: f64,
    ) -> Result<Vec<OcrLine>> {
        Ok(self.diagnose_gray(image, x_scale, y_scale)?.lines)
    }

    fn diagnose_gray(
        &mut self,
        image: &GrayImage,
        x_scale: f64,
        y_scale: f64,
    ) -> Result<IndraImageDiagnostics> {
        let mut page = self
            .diagnose_gray_many(std::slice::from_ref(image))?
            .pop()
            .expect("single-page OCR returns one page");
        scale_lines(&mut page.lines, x_scale, y_scale);
        Ok(page)
    }

    fn diagnose_gray_many(&mut self, images: &[GrayImage]) -> Result<Vec<IndraImageDiagnostics>> {
        Ok(self.diagnose_gray_many_profile(images)?.pages)
    }

    fn diagnose_gray_many_profile(
        &mut self,
        images: &[GrayImage],
    ) -> Result<IndraBatchDiagnostics> {
        if images.is_empty() {
            return Ok(IndraBatchDiagnostics {
                pages: Vec::new(),
                performance: IndraBatchPerformance::default(),
            });
        }
        let pages = images.iter().map(PagePixels::Gray).collect::<Vec<_>>();
        self.diagnose_pages_profile(&pages)
    }

    fn diagnose_rgba_many(&mut self, images: &[RgbaImage]) -> Result<Vec<IndraImageDiagnostics>> {
        let pages = images
            .iter()
            .map(|image| PagePixels::Rgba {
                pixels: image.as_raw(),
                width: image.width(),
                height: image.height(),
            })
            .collect::<Vec<_>>();
        Ok(self.diagnose_pages_profile(&pages)?.pages)
    }

    fn diagnose_pages_profile(
        &mut self,
        images: &[PagePixels<'_>],
    ) -> Result<IndraBatchDiagnostics> {
        if images.is_empty() {
            return Ok(IndraBatchDiagnostics {
                pages: Vec::new(),
                performance: IndraBatchPerformance::default(),
            });
        }
        let total_started = Instant::now();
        let input_pixels = images
            .iter()
            .map(|image| image.width() as usize * image.height() as usize)
            .sum();
        // Balance bounded windows so overlap never leaves a one-page tail.
        let window_size = if images.len() <= 4 {
            images.len()
        } else {
            preparation_window_size(images.len())
        };
        let width_scale = self.width_scale;
        let input_height = self.input_height;
        let batch_size = self.batch_size;
        let width_bucket = self.width_bucket;
        let codec = self.codec.as_ref().expect("fast Indra codec");
        let layouts = &mut self.layouts;
        let sessions = &mut self.sessions;
        let prepared = std::thread::scope(|scope| -> Result<_> {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let producer = scope.spawn(move || {
                for window in images.chunks(window_size) {
                    let value = Self::prepare_window(layouts, window, width_scale, input_height);
                    let failed = value.is_err();
                    if sender.send(value).is_err() || failed {
                        break;
                    }
                }
            });
            let result = Self::recognize_prepared_windows(
                receiver,
                sessions,
                codec,
                batch_size,
                width_bucket,
                input_height,
                images.len(),
            );
            producer
                .join()
                .map_err(|_| Error::Message("Indra preparation worker panicked".to_owned()))?;
            result
        })?;
        Ok(finish_diagnostics(
            prepared,
            input_pixels,
            total_started.elapsed().as_secs_f64(),
            0.0,
        ))
    }

    fn recognize_prepared_windows(
        receiver: std::sync::mpsc::Receiver<Result<PreparedWindow>>,
        sessions: &mut [Session],
        codec: &Codec,
        batch_size: usize,
        width_bucket: usize,
        input_height: usize,
        page_capacity: usize,
    ) -> Result<PreparedDiagnostics> {
        let mut pages = Vec::with_capacity(page_capacity);
        let mut layout_seconds = 0.0;
        let mut line_prepare_seconds = 0.0;
        let mut layout_box_count = 0;
        let mut total_lines = 0;
        let mut prepared_bytes = 0;
        let mut output_assembly_seconds = 0.0;
        let mut recognition_parts = Vec::new();
        let mut widths = Vec::new();
        for window in receiver {
            let window = window?;
            let page_count = window.prepared.len();
            let counts = window.prepared.iter().map(Vec::len).collect::<Vec<_>>();
            let window_lines = counts.iter().sum::<usize>();
            widths.extend(window.prepared.iter().flatten().map(|line| line.width));
            prepared_bytes += window
                .prepared
                .iter()
                .flatten()
                .map(|line| line.pixels.len())
                .sum::<usize>();
            layout_box_count += window.layout_boxes.iter().map(Vec::len).sum::<usize>();
            layout_seconds += window.layout_seconds;
            line_prepare_seconds += window.line_prepare_seconds;
            total_lines += window_lines;
            let (recognized, part) = Self::recognize_prepared_pages(
                sessions,
                codec,
                batch_size,
                width_bucket,
                input_height,
                &window.prepared,
            )?;
            let window_recognition_seconds = window.line_prepare_seconds + part.wall_seconds;
            recognition_parts.push(part);
            let started = Instant::now();
            pages.extend(
                window
                    .layout_boxes
                    .into_iter()
                    .zip(window.prepared)
                    .zip(recognized)
                    .zip(counts)
                    .map(|(((layout_boxes, prepared), recognized), count)| {
                        let lines = prepared
                            .into_iter()
                            .zip(recognized)
                            .filter_map(|(line, recognition)| {
                                let Recognition {
                                    text,
                                    confidence,
                                    words,
                                } = recognition;
                                // Retain confidently read source fragments, including isolated letters and symbols.
                                let kept = !text.trim().is_empty()
                                    && (!line.bbox.recovered || confidence >= RECOVERED_CONFIDENCE);
                                kept.then_some(OcrLine {
                                    recovered: line.bbox.recovered,
                                    text,
                                    bbox: [
                                        line.bbox.left as f64,
                                        line.bbox.top as f64,
                                        line.bbox.right as f64,
                                        line.bbox.bottom as f64,
                                    ],
                                    confidence,
                                    baseline: vec![],
                                    boundary: vec![],
                                    words,
                                    region_id: String::new(),
                                    region_type: "unknown".to_owned(),
                                    block_index: 0,
                                })
                            })
                            .collect();
                        IndraImageDiagnostics {
                            lines,
                            layout_boxes,
                            layout_seconds: window.layout_seconds / page_count as f64,
                            recognition_seconds: if window_lines == 0 {
                                window_recognition_seconds / page_count as f64
                            } else {
                                window_recognition_seconds * count as f64 / window_lines as f64
                            },
                        }
                    }),
            );
            output_assembly_seconds += started.elapsed().as_secs_f64();
        }
        Ok(PreparedDiagnostics {
            pages,
            layout_seconds,
            line_prepare_seconds,
            layout_box_count,
            total_lines,
            prepared_bytes,
            output_assembly_seconds,
            recognition: merge_recognition(recognition_parts, batch_size, widths),
        })
    }

    fn layout_boxes_many(
        layouts: &mut [LineDetector],
        images: &[PagePixels<'_>],
    ) -> Result<Vec<Vec<LineBox>>> {
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
                        let value = images[index].layout(layout);
                        let failed = value.is_err();
                        if sender.send((index, value)).is_err() || failed {
                            break;
                        }
                    })
                })
                .collect::<Vec<_>>();
            drop(sender);
            let mut output = std::iter::repeat_with(|| None)
                .take(images.len())
                .collect::<Vec<_>>();
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
                handle
                    .join()
                    .map_err(|_| Error::Message("Tesseract layout worker panicked".to_owned()))?;
            }
            if let Some(error) = failure {
                return Err(error);
            }
            output
                .into_iter()
                .map(|value| {
                    value.ok_or_else(|| Error::Message("layout worker returned no page".to_owned()))
                })
                .collect()
        })
    }

    fn prepare_window(
        layouts: &mut [LineDetector],
        images: &[PagePixels<'_>],
        width_scale: f32,
        input_height: usize,
    ) -> Result<PreparedWindow> {
        let started = Instant::now();
        let boxes = Self::layout_boxes_many(layouts, images)?;
        let layout_seconds = started.elapsed().as_secs_f64();
        let layout_boxes = boxes
            .iter()
            .map(|page| {
                page.iter()
                    .map(|bbox| [bbox.left, bbox.top, bbox.right, bbox.bottom])
                    .collect()
            })
            .collect();
        let started = Instant::now();
        let prepared = prepare_pages(images, &boxes, width_scale, input_height)?;
        Ok(PreparedWindow {
            layout_boxes,
            prepared,
            layout_seconds,
            line_prepare_seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn recognize_prepared_pages(
        sessions: &mut [Session],
        codec: &Codec,
        batch_size: usize,
        width_bucket: usize,
        input_height: usize,
        pages: &[Vec<PreparedLine>],
    ) -> Result<(Vec<Vec<Recognition>>, RecognitionPerformance)> {
        let counts = pages.iter().map(Vec::len).collect::<Vec<_>>();
        let lines = pages.iter().flatten().collect::<Vec<_>>();
        let schedule_started = Instant::now();
        let jobs = recognition_jobs(&lines, batch_size, width_bucket);
        let schedule_seconds = schedule_started.elapsed().as_secs_f64();
        if jobs.is_empty() {
            return Ok((
                counts.into_iter().map(|_| Vec::new()).collect(),
                RecognitionPerformance {
                    schedule_seconds,
                    ..RecognitionPerformance::default()
                },
            ));
        }
        let workers = jobs.len().min(sessions.len());
        let next = AtomicUsize::new(0);
        let (sender, receiver) = std::sync::mpsc::channel();
        let recognition_started = Instant::now();
        std::thread::scope(|scope| {
            let handles = sessions
                .iter_mut()
                .take(workers)
                .enumerate()
                .map(|(worker, session)| {
                    let sender = sender.clone();
                    let next = &next;
                    let jobs = &jobs;
                    let lines = &lines;
                    scope.spawn(move || {
                        let mut pixels = Vec::new();
                        let mut lengths = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            if index >= jobs.len() {
                                break;
                            }
                            let value = recognize_batch(
                                session,
                                codec,
                                lines,
                                &jobs[index],
                                &mut pixels,
                                &mut lengths,
                                worker,
                                input_height,
                            );
                            let failed = value.is_err();
                            if sender.send(value).is_err() || failed {
                                break;
                            }
                        }
                    })
                })
                .collect::<Vec<_>>();
            drop(sender);
            let mut output = std::iter::repeat_with(|| None)
                .take(lines.len())
                .collect::<Vec<_>>();
            let mut batches = Vec::with_capacity(jobs.len());
            let mut failure = None;
            for value in receiver {
                match value {
                    Ok(mut batch) => {
                        for (index, value) in std::mem::take(&mut batch.values) {
                            output[index] = Some(value);
                        }
                        batches.push(batch);
                    }
                    Err(error) => {
                        failure.get_or_insert(error);
                    }
                };
            }
            for handle in handles {
                handle
                    .join()
                    .map_err(|_| Error::Message("Indra recognition worker panicked".to_owned()))?;
            }
            if let Some(error) = failure {
                return Err(error);
            }
            let mut output = output
                .into_iter()
                .map(|value| {
                    value.ok_or_else(|| Error::Message("Indra worker returned no line".to_owned()))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter();
            let pages = counts
                .into_iter()
                .map(|count| output.by_ref().take(count).collect())
                .collect();
            let wall_seconds = recognition_started.elapsed().as_secs_f64();
            Ok((
                pages,
                summarize_recognition(
                    schedule_seconds,
                    wall_seconds,
                    workers,
                    batch_size,
                    &lines,
                    &batches,
                ),
            ))
        })
    }
}

fn finish_diagnostics(
    prepared: PreparedDiagnostics,
    input_pixels: usize,
    total_seconds: f64,
    render_seconds: f64,
) -> IndraBatchDiagnostics {
    let PreparedDiagnostics {
        mut pages,
        layout_seconds,
        line_prepare_seconds,
        layout_box_count,
        total_lines,
        prepared_bytes,
        output_assembly_seconds,
        recognition,
    } = prepared;
    // Receipts report additive per-page time. Preserve real layout work and assign
    // overlapped preparation/recognition only the remaining wall time.
    let reported_recognition = (total_seconds - layout_seconds).max(0.0);
    let measured_recognition = pages
        .iter()
        .map(|page| page.recognition_seconds)
        .sum::<f64>();
    if measured_recognition > 0.0 {
        let scale = reported_recognition / measured_recognition;
        for page in &mut pages {
            page.recognition_seconds *= scale;
        }
    }
    let output_lines = pages.iter().map(|page| page.lines.len()).sum();
    let performance = IndraBatchPerformance {
        detailed: true,
        pages: pages.len(),
        input_pixels,
        layout_boxes: layout_box_count,
        prepared_lines: total_lines,
        output_lines,
        layout_seconds,
        render_seconds,
        line_prepare_seconds,
        schedule_seconds: recognition.schedule_seconds,
        recognition_wall_seconds: recognition.wall_seconds,
        output_assembly_seconds,
        total_seconds,
        recognition_workers: recognition.workers,
        batches: recognition.batches,
        batch_fill_ratio: recognition.batch_fill_ratio,
        tensor_fill_ratio: recognition.tensor_fill_ratio,
        line_width_p50: recognition.line_width_p50,
        line_width_p95: recognition.line_width_p95,
        line_width_max: recognition.line_width_max,
        batch_lines_p50: recognition.batch_lines_p50,
        batch_lines_p95: recognition.batch_lines_p95,
        batch_lines_max: recognition.batch_lines_max,
        batch_seconds_p50: recognition.batch_seconds_p50,
        batch_seconds_p95: recognition.batch_seconds_p95,
        batch_seconds_max: recognition.batch_seconds_max,
        worker_busy_seconds_min: recognition.worker_busy_seconds_min,
        worker_busy_seconds_max: recognition.worker_busy_seconds_max,
        worker_busy_seconds_sum: recognition.worker_busy_seconds_sum,
        pack_seconds_sum: recognition.pack_seconds_sum,
        inference_seconds_sum: recognition.inference_seconds_sum,
        decode_seconds_sum: recognition.decode_seconds_sum,
        prepared_bytes,
        peak_tensor_bytes: recognition.peak_tensor_bytes,
    };
    IndraBatchDiagnostics { pages, performance }
}

fn recognition_jobs(
    lines: &[&PreparedLine],
    batch_size: usize,
    width_bucket: usize,
) -> Vec<Vec<usize>> {
    let mut groups = BTreeMap::<usize, Vec<usize>>::new();
    for (index, line) in lines.iter().enumerate() {
        groups
            .entry(line.width.div_ceil(width_bucket))
            .or_default()
            .push(index);
    }
    let mut jobs = groups
        .into_values()
        .flat_map(|indices| {
            indices
                .chunks(batch_size)
                .map(<[usize]>::to_vec)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    jobs.sort_unstable_by_key(|job| {
        std::cmp::Reverse(job.len() * lines[job[0]].width.div_ceil(width_bucket))
    });
    jobs
}

fn percentile_index(length: usize, percentile: usize) -> usize {
    (length.saturating_sub(1) * percentile).div_ceil(100)
}

fn summarize_recognition(
    schedule_seconds: f64,
    wall_seconds: f64,
    workers: usize,
    batch_size: usize,
    lines: &[&PreparedLine],
    batches: &[RecognizedBatch],
) -> RecognitionPerformance {
    let mut widths = lines.iter().map(|line| line.width).collect::<Vec<_>>();
    widths.sort_unstable();
    let mut batch_lines = batches.iter().map(|batch| batch.lines).collect::<Vec<_>>();
    batch_lines.sort_unstable();
    let mut batch_seconds = batches
        .iter()
        .map(|batch| batch.total_seconds)
        .collect::<Vec<_>>();
    batch_seconds.sort_by(f64::total_cmp);
    let mut worker_busy = vec![0.0_f64; workers];
    for batch in batches {
        worker_busy[batch.worker] += batch.total_seconds;
    }
    let tensor_elements = batches
        .iter()
        .map(|batch| batch.tensor_elements)
        .sum::<usize>();
    RecognitionPerformance {
        schedule_seconds,
        wall_seconds,
        workers,
        batches: batches.len(),
        batch_fill_ratio: if batches.is_empty() {
            0.0
        } else {
            lines.len() as f64 / (batches.len() * batch_size) as f64
        },
        tensor_fill_ratio: if tensor_elements == 0 {
            0.0
        } else {
            batches
                .iter()
                .map(|batch| batch.useful_elements)
                .sum::<usize>() as f64
                / tensor_elements as f64
        },
        line_width_p50: widths
            .get(percentile_index(widths.len(), 50))
            .copied()
            .unwrap_or(0),
        line_width_p95: widths
            .get(percentile_index(widths.len(), 95))
            .copied()
            .unwrap_or(0),
        line_width_max: widths.last().copied().unwrap_or(0),
        batch_lines_p50: batch_lines
            .get(percentile_index(batch_lines.len(), 50))
            .copied()
            .unwrap_or(0),
        batch_lines_p95: batch_lines
            .get(percentile_index(batch_lines.len(), 95))
            .copied()
            .unwrap_or(0),
        batch_lines_max: batch_lines.last().copied().unwrap_or(0),
        batch_seconds_p50: batch_seconds
            .get(percentile_index(batch_seconds.len(), 50))
            .copied()
            .unwrap_or(0.0),
        batch_seconds_p95: batch_seconds
            .get(percentile_index(batch_seconds.len(), 95))
            .copied()
            .unwrap_or(0.0),
        batch_seconds_max: batch_seconds.last().copied().unwrap_or(0.0),
        worker_busy_seconds_min: worker_busy
            .iter()
            .copied()
            .min_by(f64::total_cmp)
            .unwrap_or(0.0),
        worker_busy_seconds_max: worker_busy
            .iter()
            .copied()
            .max_by(f64::total_cmp)
            .unwrap_or(0.0),
        worker_busy_seconds_sum: worker_busy.iter().sum(),
        pack_seconds_sum: batches.iter().map(|batch| batch.pack_seconds).sum(),
        inference_seconds_sum: batches.iter().map(|batch| batch.inference_seconds).sum(),
        decode_seconds_sum: batches.iter().map(|batch| batch.decode_seconds).sum(),
        peak_tensor_bytes: batches
            .iter()
            .map(|batch| {
                batch.tensor_elements * std::mem::size_of::<f32>()
                    + batch.lines * std::mem::size_of::<i64>()
            })
            .max()
            .unwrap_or(0),
        tensor_elements,
        useful_elements: batches.iter().map(|batch| batch.useful_elements).sum(),
        batch_lines_values: batch_lines,
        batch_seconds_values: batch_seconds,
        worker_busy_values: worker_busy,
    }
}

fn merge_recognition(
    parts: Vec<RecognitionPerformance>,
    batch_size: usize,
    mut widths: Vec<usize>,
) -> RecognitionPerformance {
    let workers = parts.iter().map(|part| part.workers).max().unwrap_or(0);
    let mut worker_busy = vec![0.0; workers];
    let mut batch_lines = Vec::new();
    let mut batch_seconds = Vec::new();
    let mut merged = RecognitionPerformance {
        workers,
        ..RecognitionPerformance::default()
    };
    for mut part in parts {
        merged.schedule_seconds += part.schedule_seconds;
        merged.wall_seconds += part.wall_seconds;
        merged.batches += part.batches;
        merged.pack_seconds_sum += part.pack_seconds_sum;
        merged.inference_seconds_sum += part.inference_seconds_sum;
        merged.decode_seconds_sum += part.decode_seconds_sum;
        merged.peak_tensor_bytes = merged.peak_tensor_bytes.max(part.peak_tensor_bytes);
        merged.tensor_elements += part.tensor_elements;
        merged.useful_elements += part.useful_elements;
        batch_lines.append(&mut part.batch_lines_values);
        batch_seconds.append(&mut part.batch_seconds_values);
        for (target, value) in worker_busy.iter_mut().zip(part.worker_busy_values) {
            *target += value;
        }
    }
    widths.sort_unstable();
    batch_lines.sort_unstable();
    batch_seconds.sort_by(f64::total_cmp);
    let lines = widths.len();
    merged.batch_fill_ratio = if merged.batches == 0 {
        0.0
    } else {
        lines as f64 / (merged.batches * batch_size) as f64
    };
    merged.tensor_fill_ratio = if merged.tensor_elements == 0 {
        0.0
    } else {
        merged.useful_elements as f64 / merged.tensor_elements as f64
    };
    merged.line_width_p50 = widths
        .get(percentile_index(widths.len(), 50))
        .copied()
        .unwrap_or(0);
    merged.line_width_p95 = widths
        .get(percentile_index(widths.len(), 95))
        .copied()
        .unwrap_or(0);
    merged.line_width_max = widths.last().copied().unwrap_or(0);
    merged.batch_lines_p50 = batch_lines
        .get(percentile_index(batch_lines.len(), 50))
        .copied()
        .unwrap_or(0);
    merged.batch_lines_p95 = batch_lines
        .get(percentile_index(batch_lines.len(), 95))
        .copied()
        .unwrap_or(0);
    merged.batch_lines_max = batch_lines.last().copied().unwrap_or(0);
    merged.batch_seconds_p50 = batch_seconds
        .get(percentile_index(batch_seconds.len(), 50))
        .copied()
        .unwrap_or(0.0);
    merged.batch_seconds_p95 = batch_seconds
        .get(percentile_index(batch_seconds.len(), 95))
        .copied()
        .unwrap_or(0.0);
    merged.batch_seconds_max = batch_seconds.last().copied().unwrap_or(0.0);
    merged.worker_busy_seconds_min = worker_busy
        .iter()
        .copied()
        .min_by(f64::total_cmp)
        .unwrap_or(0.0);
    merged.worker_busy_seconds_max = worker_busy
        .iter()
        .copied()
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    merged.worker_busy_seconds_sum = worker_busy.iter().sum();
    merged.batch_lines_values = batch_lines;
    merged.batch_seconds_values = batch_seconds;
    merged.worker_busy_values = worker_busy;
    merged
}

fn recognize_batch(
    session: &mut Session,
    codec: &Codec,
    lines: &[&PreparedLine],
    batch: &[usize],
    pixels: &mut Vec<f32>,
    lengths: &mut Vec<i64>,
    worker: usize,
    input_height: usize,
) -> Result<RecognizedBatch> {
    let total_started = Instant::now();
    let pack_started = Instant::now();
    let maximum = batch
        .iter()
        .map(|&index| lines[index].width)
        .max()
        .unwrap_or(1)
        .div_ceil(8)
        * 8;
    pixels.resize(batch.len() * input_height * maximum, 0.0);
    pixels.fill(0.0);
    lengths.clear();
    for (local, &index) in batch.iter().enumerate() {
        let line = lines[index];
        lengths.push(line.width as i64);
        let content_width = line.width - INPUT_PADDING * 2;
        for row in 0..input_height {
            let source = row * content_width;
            let target = (local * input_height + row) * maximum + INPUT_PADDING;
            for (&value, pixel) in line.pixels[source..source + content_width]
                .iter()
                .zip(&mut pixels[target..target + content_width])
            {
                *pixel = 1.0 - f32::from(value) / 255.0;
            }
        }
    }
    let image =
        TensorRef::from_array_view(([batch.len(), 1, input_height, maximum], pixels.as_slice()))
            .map_err(ort_error)?;
    let sequence_lengths =
        TensorRef::from_array_view(([batch.len()], lengths.as_slice())).map_err(ort_error)?;
    let pack_seconds = pack_started.elapsed().as_secs_f64();
    let inference_started = Instant::now();
    let use_ids = !batch
        .iter()
        .any(|&index| lines[index].bbox.recovered || lines[index].confidence_required);
    let run_options = RunOptions::new().map_err(ort_error)?.with_outputs(
        OutputSelector::no_default()
            .with(if use_ids { "class_ids" } else { "logits" })
            .with("lengths"),
    );
    let outputs = session
        .run_with_options(
            ort::inputs![
                "image" => image,
                "widths" => sequence_lengths,
            ],
            &run_options,
        )
        .map_err(ort_error)?;
    let inference_seconds = inference_started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let (length_shape, output_lengths) =
        if let Ok((shape, values)) = outputs["lengths"].try_extract_tensor::<i64>() {
            (
                shape.iter().copied().collect::<Vec<_>>(),
                values
                    .iter()
                    .map(|&value| usize::try_from(value).unwrap_or(0))
                    .collect::<Vec<_>>(),
            )
        } else {
            let (shape, values) = outputs["lengths"]
                .try_extract_tensor::<i32>()
                .map_err(ort_error)?;
            (
                shape.iter().copied().collect::<Vec<_>>(),
                values
                    .iter()
                    .map(|&value| usize::try_from(value).unwrap_or(0))
                    .collect::<Vec<_>>(),
            )
        };
    if let Some(class_ids) = outputs.get("class_ids") {
        let (shape, ids) = class_ids.try_extract_tensor::<i64>().map_err(ort_error)?;
        let dimensions = shape
            .iter()
            .map(|value| *value as usize)
            .collect::<Vec<_>>();
        if dimensions.len() != 2
            || dimensions[0] != batch.len()
            || length_shape.first().copied() != Some(batch.len() as i64)
        {
            return Err(Error::Message(format!(
                "Indra model returned unexpected class shapes {dimensions:?} and {length_shape:?}"
            )));
        }
        let timesteps = dimensions[1];
        let values = batch
            .iter()
            .enumerate()
            .map(|(local, &index)| {
                let length = output_lengths[local].clamp(1, timesteps);
                let spans = greedy_ids(ids, local, timesteps, length);
                let (text, confidence) = codec.decode(&spans);
                let steps = ids[local * timesteps..local * timesteps + length]
                    .iter()
                    .map(|&label| {
                        usize::try_from(label)
                            .ok()
                            .filter(|label| *label > BLANK_LABEL)
                    });
                let words = line_words(codec, lines[index], &emissions(steps), length);
                (
                    index,
                    Recognition {
                        text,
                        confidence,
                        words,
                    },
                )
            })
            .collect();
        return Ok(RecognizedBatch {
            values,
            worker,
            lines: batch.len(),
            tensor_elements: batch.len() * input_height * maximum,
            useful_elements: batch
                .iter()
                .map(|&index| lines[index].width * input_height)
                .sum(),
            total_seconds: total_started.elapsed().as_secs_f64(),
            pack_seconds,
            inference_seconds,
            decode_seconds: decode_started.elapsed().as_secs_f64(),
        });
    }
    let logits = outputs.get("logits").ok_or_else(|| {
        Error::Message("Indra model returned neither logits nor class_ids".to_owned())
    })?;
    let (shape, logits) = logits.try_extract_tensor::<f32>().map_err(ort_error)?;
    let dimensions = shape
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    if dimensions.len() != 3
        || dimensions[0] != batch.len()
        || length_shape.first().copied() != Some(batch.len() as i64)
    {
        return Err(Error::Message(format!(
            "Indra model returned unexpected shapes {dimensions:?} and {length_shape:?}"
        )));
    }
    let classes = dimensions[2];
    let timesteps = dimensions[1];
    let values = batch
        .iter()
        .enumerate()
        .map(|(local, &index)| {
            let length = output_lengths[local].clamp(1, timesteps);
            let spans = greedy_ctc(logits, local, classes, timesteps, length);
            let (text, confidence) = codec.decode(&spans);
            let steps = best_labels(logits, local, classes, timesteps, length)
                .into_iter()
                .map(|label| (label != BLANK_LABEL).then_some(label));
            let words = line_words(codec, lines[index], &emissions(steps), length);
            (
                index,
                Recognition {
                    text,
                    confidence,
                    words,
                },
            )
        })
        .collect();
    Ok(RecognizedBatch {
        values,
        worker,
        lines: batch.len(),
        tensor_elements: batch.len() * input_height * maximum,
        useful_elements: batch
            .iter()
            .map(|&index| lines[index].width * input_height)
            .sum(),
        total_seconds: total_started.elapsed().as_secs_f64(),
        pack_seconds,
        inference_seconds,
        decode_seconds: decode_started.elapsed().as_secs_f64(),
    })
}

fn recognize_prepared(
    session: &mut Session,
    codec: &Codec,
    lines: &[PreparedLine],
    batch_size: usize,
    width_bucket: usize,
    input_height: usize,
) -> Result<Vec<(String, f64)>> {
    let lines = lines.iter().collect::<Vec<_>>();
    let mut output = vec![(String::new(), 0.0); lines.len()];
    let mut pixels = Vec::new();
    let mut lengths = Vec::new();
    for job in recognition_jobs(&lines, batch_size, width_bucket) {
        for (index, value) in recognize_batch(
            session,
            codec,
            &lines,
            &job,
            &mut pixels,
            &mut lengths,
            0,
            input_height,
        )?
        .values
        {
            output[index] = (value.text, value.confidence);
        }
    }
    Ok(output)
}

#[cfg(feature = "ocr")]
/// A rendered image's page, footnote separator, points per pixel, top-left corner on the page in
/// points, and the quarter turns counter-clockwise that stood it upright with its drawn size in pixels.
type PdfPageMetadata = (usize, Option<f64>, f64, f64, [f64; 2], (u8, f64, f64));
#[cfg(feature = "ocr")]
type RenderedPdfPage = (PdfPageMetadata, RgbaImage);

#[cfg(feature = "ocr")]
fn pdf_render_settings(dpi: u16) -> RenderSettings {
    let scale = f32::from(dpi) / 72.0;
    RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: WHITE,
        ..Default::default()
    }
}

#[cfg(feature = "ocr")]
fn render_pdf_pages<'a>(
    pdf: &'a Pdf,
    cache: &RenderCache<'a>,
    interpreter: &InterpreterSettings,
    settings: &RenderSettings,
    requests: &[OcrPageRequest],
    orientation: Option<&std::sync::Mutex<&mut Orientation>>,
) -> Result<Vec<RenderedPdfPage>> {
    // White margin around a part of a page, so the layout reads its first and last lines whole.
    const MARGIN: usize = 16;
    let mut rendered = Vec::new();
    for request in requests {
        let page = pdf.pages().iter().nth(request.page_index).ok_or_else(|| {
            Error::Message(format!(
                "PDF page index is out of range: {}",
                request.page_index
            ))
        })?;
        let pixmap = render(page, cache, interpreter, settings);
        let width = usize::from(pixmap.width());
        let height = usize::from(pixmap.height());
        if width == 0 || height == 0 {
            return Err(Error::Message(format!(
                "OCR renderer produced an empty page image for page {}",
                request.page_index + 1
            )));
        }
        let x_scale = request.width / width as f64;
        let y_scale = request.height / height as f64;
        let pixels = pixmap.data_as_u8_slice();
        if request.regions.is_empty() {
            let mut page = RgbaImage::from_raw(width as u32, height as u32, pixels.to_vec())
                .expect("renderer dimensions match its pixel buffer");
            let turns = match orientation {
                Some(orientation) => orientation
                    .lock()
                    .map_err(|_| Error::Message("OCR orientation lock poisoned".into()))?
                    .upright_turns(&page)?,
                None => 0,
            };
            if turns != 0 {
                page = turn(&page, turns);
            }
            let metadata = (
                request.page_index,
                // A turned page's rule runs across its print, not across the page as drawn.
                (turns == 0)
                    .then(|| super::ocr::raster_separator_y(&pixmap, request.height))
                    .flatten(),
                x_scale,
                y_scale,
                [0.0, 0.0],
                (turns, width as f64, height as f64),
            );
            let (turned_width, turned_height) = (page.width() as usize, page.height() as usize);
            rendered.push((
                metadata,
                rendered_image(page.into_raw(), turned_width, turned_height),
            ));
            continue;
        }
        // Each part of the page is recognized on its own, at the page's resolution.
        for region in &request.regions {
            let x0 = ((region[0] / x_scale).floor().max(0.0) as usize).min(width);
            let y0 = ((region[1] / y_scale).floor().max(0.0) as usize).min(height);
            let x1 = ((region[2] / x_scale).ceil().max(0.0) as usize).min(width);
            let y1 = ((region[3] / y_scale).ceil().max(0.0) as usize).min(height);
            if x1 <= x0 || y1 <= y0 || !dark_print_on_light_ground(pixels, width, [x0, y0, x1, y1])
            {
                continue;
            }
            let crop_width = x1 - x0 + 2 * MARGIN;
            let crop_height = y1 - y0 + 2 * MARGIN;
            let mut crop = vec![255u8; crop_width * crop_height * 4];
            for row in y0..y1 {
                let from = (row * width + x0) * 4;
                let to = ((row - y0 + MARGIN) * crop_width + MARGIN) * 4;
                crop[to..to + (x1 - x0) * 4].copy_from_slice(&pixels[from..from + (x1 - x0) * 4]);
            }
            let metadata = (
                request.page_index,
                None,
                x_scale,
                y_scale,
                [
                    (x0 as f64 - MARGIN as f64) * x_scale,
                    (y0 as f64 - MARGIN as f64) * y_scale,
                ],
                (0, 0.0, 0.0),
            );
            rendered.push((metadata, rendered_image(crop, crop_width, crop_height)));
        }
    }
    Ok(rendered)
}

/// Whether a part of an RGBA page reads as print: mostly light ground, with dark ink on at most a
/// third of it. A photograph, a dark band or a filled graphic is not recognized.
#[cfg(feature = "ocr")]
fn dark_print_on_light_ground(pixels: &[u8], width: usize, [x0, y0, x1, y1]: [usize; 4]) -> bool {
    let (mut light, mut dark, mut count) = (0usize, 0usize, 0usize);
    for row in y0..y1 {
        for pixel in pixels[(row * width + x0) * 4..(row * width + x1) * 4].chunks_exact(4) {
            let luminance =
                (u32::from(pixel[0]) * 3 + u32::from(pixel[1]) * 6 + u32::from(pixel[2])) / 10;
            light += usize::from(luminance >= 235);
            dark += usize::from(luminance <= 110);
            count += 1;
        }
    }
    light * 100 >= count * 45 && dark * 500 >= count && dark * 100 <= count * 35
}

#[cfg(feature = "ocr")]
fn rendered_image(rgba: Vec<u8>, width: usize, height: usize) -> RgbaImage {
    RgbaImage::from_raw(width as u32, height as u32, rgba)
        .expect("renderer dimensions match its pixel buffer")
}

/// One result per requested page, in request order: the parts of a page recognized apart are
/// placed back on it, and a page none of whose parts reads as print has no lines.
#[cfg(feature = "ocr")]
fn finish_recognized_pdf_pages(
    requests: &[OcrPageRequest],
    metadata: Vec<PdfPageMetadata>,
    pages: Vec<IndraImageDiagnostics>,
) -> Result<Vec<OcrPageResult>> {
    let mut results: Vec<OcrPageResult> = requests
        .iter()
        .map(|request| OcrPageResult {
            page_index: request.page_index,
            lines: Vec::new(),
            separator_y: None,
        })
        .collect();
    let mut at = 0;
    for ((page_index, separator_y, x_scale, y_scale, origin, (turns, width, height)), mut page) in
        metadata.into_iter().zip(pages)
    {
        place_back(&mut page.lines, turns, width, height);
        scale_lines(&mut page.lines, x_scale, y_scale);
        translate_lines(&mut page.lines, origin);
        while results[at].page_index != page_index {
            at += 1;
        }
        results[at].lines.extend(page.lines);
        results[at].separator_y = results[at].separator_y.or(separator_y);
    }
    Ok(results)
}

fn preparation_window_size(page_count: usize) -> usize {
    page_count.div_ceil(page_count.div_ceil(4).max(2)).max(1)
}

fn scale_lines(lines: &mut [OcrLine], x_scale: f64, y_scale: f64) {
    for line in lines {
        line.bbox[0] *= x_scale;
        line.bbox[1] *= y_scale;
        line.bbox[2] *= x_scale;
        line.bbox[3] *= y_scale;
        for point in &mut line.baseline {
            point[0] *= x_scale;
            point[1] *= y_scale;
        }
        for point in &mut line.boundary {
            point[0] *= x_scale;
            point[1] *= y_scale;
        }
        for word in &mut line.words {
            word.bbox[0] *= x_scale;
            word.bbox[1] *= y_scale;
            word.bbox[2] *= x_scale;
            word.bbox[3] *= y_scale;
        }
    }
}

#[cfg(feature = "ocr")]
fn translate_lines(lines: &mut [OcrLine], origin: [f64; 2]) {
    if origin == [0.0, 0.0] {
        return;
    }
    for line in lines {
        for bbox in
            std::iter::once(&mut line.bbox).chain(line.words.iter_mut().map(|word| &mut word.bbox))
        {
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
    backend: IndraBackend,
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
    if backend == IndraBackend::DirectMl {
        builder = builder
            .with_memory_pattern(false)
            .and_then(|builder| builder.with_parallel_execution(false))
            .map_err(ort_error)?;
    }
    if backend == IndraBackend::Cpu && !cpu_arena {
        builder = builder
            .with_execution_providers(vec![CPUExecutionProvider::default()
                .with_arena_allocator(false)
                .build()])
            .map_err(ort_error)?;
    }
    // Avoid an exhaustive cuDNN search for every changing OCR line width.
    let providers = if backend == IndraBackend::Cuda {
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
        builder = builder
            .with_execution_providers(providers)
            .map_err(ort_error)?;
    }
    builder.commit_from_file(model).map_err(ort_error)
}

fn validate_options(options: &IndraOptions) -> Result<()> {
    if !(72..=600).contains(&options.dpi)
        || options.input_height != 32
        || options.batch_size > 256
        || options.width_bucket > 1024
        || options.workers > 16
        || options.layout_workers > 16
        || options.threads > 16
    {
        return Err(Error::Message(
            "Indra DPI, height, batch or worker counts are outside supported bounds".to_owned(),
        ));
    }
    if options
        .backend
        .is_some_and(|backend| backend != IndraBackend::Cpu)
        && !options.cpu_arena
    {
        return Err(Error::Message(
            "Indra low-memory mode requires the CPU backend".to_owned(),
        ));
    }
    if options.device.is_some() && options.backend.is_none() {
        return Err(Error::Message(
            "an explicit device requires an explicit backend".to_owned(),
        ));
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
        .ok_or_else(|| {
            Error::Message(format!(
                "{label} is required; pass it explicitly or set {variable}"
            ))
        })
}

pub(crate) fn canonical_file(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::Message(format!(
            "{label} does not exist: {}",
            path.display()
        )));
    }
    fs::canonicalize(path).map_err(|source| Error::io(path, source))
}

pub(crate) fn ort_error(error: ort::Error) -> Error {
    Error::Message(format!("Indra ONNX inference failed: {error}"))
}

fn prepare_line(
    image: &GrayImage,
    bbox: LineBox,
    width_scale: f32,
    input_height: usize,
) -> Option<PreparedLine> {
    if bbox.width() == 0 || bbox.height() == 0 {
        return None;
    }
    let crop = imageops::crop_imm(
        image,
        bbox.left as u32,
        bbox.top as u32,
        bbox.width() as u32,
        bbox.height() as u32,
    )
    .to_image();
    prepare_crop(crop, bbox, width_scale, input_height)
}

fn prepare_crop(
    crop: GrayImage,
    bbox: LineBox,
    width_scale: f32,
    input_height: usize,
) -> Option<PreparedLine> {
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
        (
            ((y1 - y0) as f64 * 0.3).ceil() as u32,
            ((y1 - y0) as f64 * 0.2).ceil() as u32,
        )
    } else {
        x0 = 0;
        y0 = 0;
        x1 = crop.width();
        y1 = crop.height();
        (0, 0)
    };
    let padded_width = x1 - x0 + 2 * px;
    let padded_height = y1 - y0 + 2 * py;
    let mut padded = GrayImage::from_pixel(padded_width, padded_height, image::Luma([255]));
    imageops::replace(
        &mut padded,
        &imageops::crop_imm(&crop, x0, y0, x1 - x0, y1 - y0).to_image(),
        i64::from(px),
        i64::from(py),
    );
    let width = (padded_width as f32 * input_height as f32 / padded_height as f32 * width_scale)
        .round()
        .clamp(8.0, 4096.0) as usize;
    let resized = imageops::resize(
        &padded,
        width as u32,
        input_height as u32,
        imageops::FilterType::Triangle,
    );
    Some(PreparedLine {
        bbox,
        confidence_required: false,
        width,
        pixels: resized.into_raw(),
        pixel_x_origin: bbox.left as f64 + x0 as f64 - px as f64,
        pixel_x_scale: padded_width as f64 / width as f64,
    })
}

fn prepare_rgba_line(
    source: &[u8],
    page_width: usize,
    bbox: LineBox,
    width_scale: f32,
    input_height: usize,
) -> Option<PreparedLine> {
    if bbox.width() == 0 || bbox.height() == 0 {
        return None;
    }
    let mut gray = GrayImage::new(bbox.width() as u32, bbox.height() as u32);
    for (row, target) in gray.as_mut().chunks_exact_mut(bbox.width()).enumerate() {
        let start = ((bbox.top + row) * page_width + bbox.left) * 4;
        for (pixel, value) in source[start..start + bbox.width() * 4]
            .chunks_exact(4)
            .zip(target)
        {
            *value =
                ((u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2]) + 1) / 3) as u8;
        }
    }
    prepare_crop(gray, bbox, width_scale, input_height)
}

fn prepare_page_line(
    image: PagePixels<'_>,
    bbox: LineBox,
    width_scale: f32,
    input_height: usize,
) -> Option<PreparedLine> {
    match image {
        PagePixels::Gray(image) => prepare_line(image, bbox, width_scale, input_height),
        PagePixels::Rgba { pixels, width, .. } => {
            prepare_rgba_line(pixels, width as usize, bbox, width_scale, input_height)
        }
    }
}

fn prepare_pages(
    images: &[PagePixels<'_>],
    boxes: &[Vec<LineBox>],
    width_scale: f32,
    input_height: usize,
) -> Result<Vec<Vec<PreparedLine>>> {
    let workers = images.len().min(
        std::thread::available_parallelism()
            .map_or(1, |value| value.get())
            .max(1),
    );
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
                    let lines = boxes[index]
                        .iter()
                        .filter_map(|&bbox| {
                            prepare_page_line(images[index], bbox, width_scale, input_height)
                        })
                        .collect();
                    if sender.send((index, lines)).is_err() {
                        break;
                    }
                })
            })
            .collect::<Vec<_>>();
        drop(sender);
        let mut output = std::iter::repeat_with(|| None)
            .take(images.len())
            .collect::<Vec<_>>();
        for (index, lines) in receiver {
            output[index] = Some(lines);
        }
        for handle in handles {
            handle
                .join()
                .map_err(|_| Error::Message("line preparation worker panicked".to_owned()))?;
        }
        output
            .into_iter()
            .map(|lines| {
                lines.ok_or_else(|| Error::Message("line preparation returned no page".to_owned()))
            })
            .collect()
    })
}

/// Each timestep's most likely label, as `greedy_ctc` reads it.
fn best_labels(
    logits: &[f32],
    batch: usize,
    classes: usize,
    timesteps: usize,
    length: usize,
) -> Vec<usize> {
    logits[batch * timesteps * classes..][..length * classes]
        .chunks_exact(classes)
        .map(|step| {
            step.iter()
                .enumerate()
                .fold((0, f32::NEG_INFINITY), |best, (label, &value)| {
                    if value > best.1 {
                        (label, value)
                    } else {
                        best
                    }
                })
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
fn line_words(
    codec: &Codec,
    line: &PreparedLine,
    emitted: &[(usize, usize, usize)],
    length: usize,
) -> Vec<OcrWord> {
    let content_width = line.width.saturating_sub(INPUT_PADDING * 2);
    if emitted.is_empty() || content_width == 0 || length == 0 || line.pixels.is_empty() {
        return Vec::new();
    }
    let rows = line.pixels.len() / content_width;
    let labels: Vec<usize> = emitted.iter().map(|(label, _, _)| *label).collect();
    // Each character of the decoded text with the column it was emitted at.
    let stride = line.width as f64 / length as f64;
    let mut characters: Vec<(char, f64)> = Vec::new();
    for (first, last, value) in codec.pieces(&labels) {
        let step = (emitted[first].1 + emitted[last].2 + 1) as f64 / 2.0;
        let column = step * stride - INPUT_PADDING as f64;
        characters.extend(value.chars().map(|character| (character, column)));
    }
    let ink: Vec<bool> = (0..content_width)
        .map(|column| (0..rows).any(|row| line.pixels[row * content_width + column] < 128))
        .collect();
    let (Some(first_ink), Some(last_ink)) = (
        ink.iter().position(|value| *value),
        ink.iter().rposition(|value| *value),
    ) else {
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
    let distance = |gap: &(usize, usize), estimate: f64| {
        (gap.0 as f64 - estimate)
            .max(estimate - gap.1 as f64)
            .max(0.0)
    };
    // The spaces fall in the line's widest gaps when those stand where the spaces were read.
    let mut widest = gaps.clone();
    widest.sort_by(|left, right| (right.1 - right.0).cmp(&(left.1 - left.0)));
    widest.truncate(estimates.len());
    widest.sort_unstable();
    let mut bounds: Vec<(usize, usize)> = vec![(first_ink, first_ink)];
    if widest.len() == estimates.len()
        && widest
            .iter()
            .zip(&estimates)
            .all(|(gap, estimate)| distance(gap, *estimate) <= 3.0 * pitch)
    {
        bounds.extend(widest);
    } else {
        // Otherwise each space takes the widest gap near where it was read.
        for estimate in &estimates {
            let after = bounds.last().map_or(0, |bound| bound.1);
            let chosen = gaps
                .iter()
                .filter(|gap| gap.0 >= after && distance(gap, *estimate) <= 1.5 * pitch)
                .max_by(|left, right| {
                    (left.1 - left.0)
                        .cmp(&(right.1 - right.0))
                        .then(distance(right, *estimate).total_cmp(&distance(left, *estimate)))
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
            let (left, right) = runs
                .iter()
                .max_by_key(|run| run.2)
                .map_or((from, to), |run| (run.0, run.1));
            OcrWord {
                text: characters[*begin..*end]
                    .iter()
                    .map(|(character, _)| *character)
                    .collect(),
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

fn greedy_ctc(
    logits: &[f32],
    batch: usize,
    classes: usize,
    timesteps: usize,
    length: usize,
) -> Vec<(usize, f32)> {
    let labels = best_labels(logits, batch, classes, timesteps, length);
    let denominators = logits[batch * timesteps * classes..][..length * classes]
        .chunks_exact(classes)
        .zip(&labels)
        .map(|(step, &label)| {
            step.iter()
                .map(|&value| (value - step[label]).exp())
                .sum::<f32>()
        })
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires local model/runtime and the independently invented Lite OCR fixture"]
    fn partial_page_regions_return_real_character_confidence() {
        let backend = std::env::var("LEGALPDF_OCR_TEST_BACKEND").unwrap_or_else(|_| "cpu".into());
        let options = IndraOptions {
            model_pack: Some(
                std::env::var_os("LEGALPDF_OCR_TEST_MODEL_PACK")
                    .expect("set model pack")
                    .into(),
            ),
            runtime: Some(
                std::env::var_os("LEGALPDF_OCR_TEST_RUNTIME")
                    .expect("set runtime")
                    .into(),
            ),
            backend: Some(IndraBackend::parse(&backend).expect("unknown test backend")),
            workers: 1,
            threads: 1,
            layout_workers: 1,
            batch_size: 16,
            width_bucket: 24,
            ..Default::default()
        };
        let bytes = std::fs::read(
            std::env::var_os("LEGALPDF_OCR_TEST_PDF").expect("set invented fixture PDF"),
        )
        .unwrap();
        let mut engine = IndraOcr::new(&options).unwrap();
        let pages = engine
            .extract_pages(
                &bytes,
                &[OcrPageRequest {
                    page_index: 0,
                    width: 612.0,
                    height: 792.0,
                    regions: vec![[0.0, 30.0, 612.0, 65.0]],
                }],
            )
            .unwrap();
        let line = pages[0]
            .lines
            .iter()
            .find(|line| {
                !line.recovered && line.text.contains("Northlake") && line.text.contains("Society")
            })
            .unwrap_or_else(|| {
                panic!(
                    "the cropped printed title must be read: {:?}",
                    pages[0].lines
                )
            });
        assert!(
            line.confidence > 0.0 && line.confidence < 1.0,
            "region confidence must come from logits, not compact IDs"
        );
    }

    const TEST_INPUT_HEIGHT: usize = 32;

    #[test]
    fn indra_options_fail_closed_without_assets() {
        let options = IndraOptions::default();
        assert!(IndraOcr::new(&options).is_err());
    }

    #[test]
    fn ctc_collapses_repeats_but_not_blank_separated_labels() {
        let logits = [0.0, 5.0, 0.0, 5.0, 5.0, 0.0, 5.0, 0.0, 0.0, 5.0];
        let spans = greedy_ctc(&logits, 0, 2, 5, 5);
        assert_eq!(spans.iter().map(|span| span.0).collect::<Vec<_>>(), [1, 1]);
        let expected = 1.0 / (1.0 + (-5.0_f32).exp());
        assert!(spans
            .iter()
            .all(|(_, confidence)| (confidence - expected).abs() < f32::EPSILON));
        assert_eq!(greedy_ids(&[0, 1, 1, 0, 1], 0, 5, 5), [(1, 1.0), (1, 1.0)]);
    }

    #[test]
    fn recognition_jobs_batch_every_line_once_by_width() {
        let lines = [23, 24, 25, 47, 48]
            .into_iter()
            .map(|width| PreparedLine {
                confidence_required: false,
                bbox: LineBox {
                    left: 0,
                    top: 0,
                    right: width,
                    bottom: 1,
                    ..LineBox::default()
                },
                width,
                pixels: Vec::new(),
                pixel_x_origin: 0.0,
                pixel_x_scale: 1.0,
            })
            .collect::<Vec<_>>();
        let lines = lines.iter().collect::<Vec<_>>();
        let jobs = recognition_jobs(&lines, 2, 24);
        let mut indices = jobs.iter().flatten().copied().collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, (0..lines.len()).collect::<Vec<_>>());
        let costs = jobs
            .iter()
            .map(|job| job.len() * lines[job[0]].width.div_ceil(24))
            .collect::<Vec<_>>();
        assert!(costs.windows(2).all(|pair| pair[0] >= pair[1]));
        assert!(jobs.iter().all(|job| {
            job.len() <= 2
                && job.iter().all(|&index| {
                    lines[index].width.div_ceil(24) == lines[job[0]].width.div_ceil(24)
                })
        }));
    }

    #[test]
    fn recognition_profile_accounts_for_batch_padding_and_worker_balance() {
        let prepared = [100, 120, 200].map(|width| PreparedLine {
            confidence_required: false,
            bbox: LineBox {
                left: 0,
                top: 0,
                right: width,
                bottom: 1,
                ..LineBox::default()
            },
            width,
            pixels: Vec::new(),
            pixel_x_origin: 0.0,
            pixel_x_scale: 1.0,
        });
        let lines = prepared.iter().collect::<Vec<_>>();
        let batches = [
            RecognizedBatch {
                values: Vec::new(),
                worker: 0,
                lines: 2,
                tensor_elements: 2 * TEST_INPUT_HEIGHT * 120,
                useful_elements: TEST_INPUT_HEIGHT * 220,
                total_seconds: 0.2,
                pack_seconds: 0.01,
                inference_seconds: 0.18,
                decode_seconds: 0.01,
            },
            RecognizedBatch {
                values: Vec::new(),
                worker: 1,
                lines: 1,
                tensor_elements: TEST_INPUT_HEIGHT * 200,
                useful_elements: TEST_INPUT_HEIGHT * 200,
                total_seconds: 0.3,
                pack_seconds: 0.02,
                inference_seconds: 0.27,
                decode_seconds: 0.01,
            },
        ];
        let profile = summarize_recognition(0.001, 0.31, 2, 2, &lines, &batches);
        assert_eq!(profile.batches, 2);
        assert_eq!(profile.line_width_p50, 120);
        assert_eq!(profile.line_width_p95, 200);
        assert_eq!(profile.batch_lines_p50, 2);
        assert_eq!(profile.batch_lines_p95, 2);
        assert!((profile.batch_fill_ratio - 0.75).abs() < f64::EPSILON);
        assert!((profile.tensor_fill_ratio - 420.0 / 440.0).abs() < f64::EPSILON);
        assert!((profile.worker_busy_seconds_min - 0.2).abs() < f64::EPSILON);
        assert!((profile.worker_busy_seconds_max - 0.3).abs() < f64::EPSILON);
        assert_eq!(
            profile.peak_tensor_bytes,
            2 * TEST_INPUT_HEIGHT * 120 * std::mem::size_of::<f32>()
                + 2 * std::mem::size_of::<i64>()
        );
    }

    #[test]
    fn backends_parse_and_normalize_devices_fail_closed() {
        assert_eq!(IndraBackend::parse("cpu"), Some(IndraBackend::Cpu));
        assert_eq!(
            IndraBackend::parse("tensorrt"),
            Some(IndraBackend::TensorRt)
        );
        assert_eq!(IndraBackend::parse("auto"), None);
        assert_eq!(IndraBackend::Cuda.normalized_device(None).unwrap(), "0");
        assert_eq!(
            IndraBackend::OpenVino
                .normalized_device(Some("GPU.1"))
                .unwrap(),
            "GPU.1"
        );
        assert!(IndraBackend::Cpu.normalized_device(Some("0")).is_err());
        assert!(IndraBackend::DirectMl
            .normalized_device(Some("-1"))
            .is_err());
        let options = IndraOptions::default();
        assert!(validate_options(&options).is_ok());
    }
}
