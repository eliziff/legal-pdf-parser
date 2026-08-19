use crate::error::{Error, Result};
use crate::kraken_process::BllaRuntime;
use crate::ocr::{OcrLine, OcrPageRequest, OcrPageResult};
use crate::tesseract_layout::TesseractLayout;
#[cfg(feature = "ocr")]
use hayro::hayro_interpret::InterpreterSettings;
#[cfg(feature = "ocr")]
use hayro::hayro_syntax::Pdf;
#[cfg(feature = "ocr")]
use hayro::vello_cpu::color::palette::css::WHITE;
#[cfg(feature = "ocr")]
use hayro::{render, RenderCache, RenderSettings};
use image::{imageops, GrayImage, ImageReader, RgbaImage};
use ort::{
    execution_providers::CPUExecutionProvider,
    session::{builder::GraphOptimizationLevel, builder::PrepackedWeights, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

const INPUT_HEIGHT: usize = 48;
// Store padding in the shared batch tensor, not in every prepared line.
const INPUT_PADDING: usize = 16;
const BLANK_LABEL: usize = 0;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KrakenLayout {
    #[default]
    Tesseract,
    Blla,
}

impl KrakenLayout {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "tesseract" => Some(Self::Tesseract),
            "blla" => Some(Self::Blla),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Tesseract => "tesseract",
            Self::Blla => "blla",
        }
    }
}

pub use crate::ort_backend::OrtBackend as KrakenBackend;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KrakenTier {
    #[default]
    Quality,
    Balanced,
    Turbo,
    Extreme,
}

impl KrakenTier {
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "quality" => Some(Self::Quality),
            "balanced" => Some(Self::Balanced),
            "turbo" => Some(Self::Turbo),
            "extreme" => Some(Self::Extreme),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Quality => "quality",
            Self::Balanced => "balanced",
            Self::Turbo => "turbo",
            Self::Extreme => "extreme",
        }
    }

    fn width_scale(self) -> f32 {
        match self {
            Self::Quality => 1.0,
            Self::Balanced => 0.85,
            Self::Turbo => 0.76,
            Self::Extreme => 0.70,
        }
    }
}

#[derive(Debug, Clone)]
pub struct KrakenOptions {
    pub model: Option<PathBuf>,
    pub codec: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub runtime_wheel: Option<PathBuf>,
    pub python: Option<PathBuf>,
    pub blla_pack: Option<PathBuf>,
    pub recognizer_pack: Option<PathBuf>,
    pub tesseract_library: Option<PathBuf>,
    pub dpi: u16,
    pub threads: usize,
    pub workers: usize,
    pub layout_workers: usize,
    pub batch_size: usize,
    pub runtime_batch_size: usize,
    pub width_bucket: usize,
    pub width_scale: Option<f32>,
    pub tier: KrakenTier,
    pub layout: KrakenLayout,
    pub backend: KrakenBackend,
    pub device: Option<String>,
    pub cpu_fallback: bool,
    pub cpu_arena: bool,
    pub timeout_seconds: u64,
    pub expected_identity: Option<String>,
}

impl Default for KrakenOptions {
    fn default() -> Self {
        Self {
            model: None,
            codec: None,
            runtime: None,
            runtime_wheel: None,
            python: None,
            blla_pack: None,
            recognizer_pack: None,
            tesseract_library: None,
            dpi: 200,
            threads: 0,
            workers: 0,
            layout_workers: std::thread::available_parallelism()
                .map_or(1, |value| value.get().div_ceil(2))
                .min(8),
            batch_size: 32,
            runtime_batch_size: 0,
            width_bucket: 24,
            width_scale: None,
            tier: KrakenTier::Quality,
            layout: KrakenLayout::Tesseract,
            backend: KrakenBackend::Cpu,
            device: None,
            cpu_fallback: false,
            cpu_arena: true,
            timeout_seconds: 3600,
            expected_identity: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LineBox {
    pub(crate) left: usize,
    pub(crate) top: usize,
    pub(crate) right: usize,
    pub(crate) bottom: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct KrakenImageDiagnostics {
    pub lines: Vec<OcrLine>,
    pub layout_boxes: Vec<[usize; 4]>,
    pub layout_seconds: f64,
    pub recognition_seconds: f64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KrakenBatchPerformance {
    pub detailed: bool,
    pub pages: usize,
    pub input_pixels: usize,
    pub layout_boxes: usize,
    pub prepared_lines: usize,
    pub output_lines: usize,
    pub layout_seconds: f64,
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
pub struct KrakenBatchDiagnostics {
    pub pages: Vec<KrakenImageDiagnostics>,
    pub performance: KrakenBatchPerformance,
}

impl LineBox {
    fn width(self) -> usize {
        self.right.saturating_sub(self.left)
    }

    pub(crate) fn height(self) -> usize {
        self.bottom.saturating_sub(self.top)
    }
}

struct PreparedLine {
    bbox: LineBox,
    width: usize,
    pixels: Vec<u8>,
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

    fn layout(self, engine: &mut TesseractLayout) -> Result<Vec<LineBox>> {
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
    pages: Vec<KrakenImageDiagnostics>,
    layout_seconds: f64,
    line_prepare_seconds: f64,
    layout_box_count: usize,
    total_lines: usize,
    prepared_bytes: usize,
    output_assembly_seconds: f64,
    recognition: RecognitionPerformance,
}

struct RecognizedBatch {
    values: Vec<(usize, (String, f64))>,
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
    labels: Vec<(Vec<usize>, String)>,
}

impl Codec {
    fn load(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|source| Error::io(path, source))?;
        let mapping: BTreeMap<String, Vec<usize>> = serde_json::from_reader(BufReader::new(file))?;
        if mapping.is_empty() || mapping.values().any(Vec::is_empty) {
            return Err(Error::Message(
                "Kraken codec is empty or malformed".to_owned(),
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
                return Err(Error::Message("Kraken codec is not prefix-free".to_owned()));
            }
        }
        Ok(Self { labels })
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
                .iter()
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

pub struct KrakenOcr {
    sessions: Vec<Session>,
    codec: Option<Codec>,
    dpi: u16,
    batch_size: usize,
    width_bucket: usize,
    width_scale: f32,
    tesseract_layouts: Vec<TesseractLayout>,
    blla: Option<BllaRuntime>,
    identity: String,
    name: String,
}

impl KrakenOcr {
    pub fn new(options: &KrakenOptions) -> Result<Self> {
        validate_options(options)?;
        if options.layout == KrakenLayout::Blla {
            let blla = BllaRuntime::new(options)?;
            let identity = blla.identity().to_owned();
            let name = blla.name().to_owned();
            return Ok(Self {
                sessions: vec![],
                codec: None,
                dpi: options.dpi,
                batch_size: options.batch_size,
                width_bucket: options.width_bucket,
                width_scale: 1.0,
                tesseract_layouts: vec![],
                blla: Some(blla),
                identity,
                name,
            });
        }
        let device = options
            .backend
            .normalized_device(options.device.as_deref())?;
        let model = required_path(&options.model, "LEGALPDF_KRAKEN_MODEL", "Kraken model")?;
        let codec = options
            .codec
            .clone()
            .or_else(|| std::env::var_os("LEGALPDF_KRAKEN_CODEC").map(PathBuf::from))
            .unwrap_or_else(|| model.parent().unwrap_or(Path::new(".")).join("codec.json"));
        let runtime = required_path(
            &options.runtime,
            "LEGALPDF_ONNXRUNTIME",
            "ONNX Runtime library",
        )?;
        let model = canonical_file(&model, "Kraken model")?;
        let codec_path = canonical_file(&codec, "Kraken codec")?;
        let runtime = canonical_file(&runtime, "ONNX Runtime library")?;
        let tesseract_library = required_path(
            &options.tesseract_library,
            "LEGALPDF_TESSERACT_LIBRARY",
            "Tesseract layout library",
        )
        .and_then(|path| canonical_file(&path, "Tesseract layout library"))?;
        let (workers, threads) =
            recognition_schedule(options.workers, options.threads, options.backend);
        let automatic_cpu =
            options.backend == KrakenBackend::Cpu && options.workers == 0 && options.threads == 0;
        let requested_global_threads = automatic_cpu.then_some(workers);
        let environment_global_threads =
            crate::ort_runtime::init(&runtime, requested_global_threads)?;
        let independent_pool = environment_global_threads.is_some() && !uses_global_pool(options);
        let session_global_threads = (!independent_pool)
            .then_some(environment_global_threads)
            .flatten();
        let tesseract_layouts = (0..options.layout_workers)
            .map(|_| TesseractLayout::new(&tesseract_library, options.dpi))
            .collect::<Result<Vec<_>>>()?;
        let layout_identity = sha256_file(&tesseract_library)?;
        let identity = format!(
            "kraken-lite-rust-v2:backend={}:device={device}:fallback={}:model={}:codec={}:runtime={}:layout={layout_identity}",
            options.backend.name(),
            if options.cpu_fallback { "cpu" } else { "none" },
            sha256_file(&model)?,
            sha256_file(&codec_path)?,
            sha256_file(&runtime)?,
        );
        if options
            .expected_identity
            .as_deref()
            .is_some_and(|expected| expected != identity)
        {
            return Err(Error::Message(format!(
                "Kraken identity changed before OCR began: expected {}, found {identity}",
                options.expected_identity.as_deref().unwrap_or_default()
            )));
        }
        let prepacked_weights = PrepackedWeights::new();
        let sessions = (0..workers)
            .map(|_| {
                open_session(
                    &model,
                    threads,
                    options.backend,
                    &device,
                    options.cpu_fallback,
                    options.cpu_arena,
                    &prepacked_weights,
                    independent_pool,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let codec = Codec::load(&codec_path)?;
        let width_scale = options
            .width_scale
            .unwrap_or_else(|| options.tier.width_scale());
        let name = format!(
            "{identity}:tier={}:dpi={}:batch={}:bucket={}:scale={width_scale:.3}:layout={}:workers={workers}:threads={}:layout-workers={}:ort-global-threads={}:cpu-arena={}",
            options.tier.name(),
            options.dpi,
            options.batch_size,
            options.width_bucket,
            options.layout.name(),
            threads,
            options.layout_workers,
            session_global_threads.unwrap_or(0),
            options.cpu_arena,
        );
        Ok(Self {
            sessions,
            codec: Some(codec),
            dpi: options.dpi,
            batch_size: options.batch_size,
            width_bucket: options.width_bucket,
            width_scale,
            tesseract_layouts,
            blla: None,
            identity,
            name,
        })
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
    ) -> Result<KrakenImageDiagnostics> {
        self.diagnose_gray(image, 1.0, 1.0)
    }

    pub fn recognize_gray_images_diagnostics(
        &mut self,
        images: &[GrayImage],
    ) -> Result<Vec<KrakenImageDiagnostics>> {
        self.diagnose_gray_many(images)
    }

    pub fn recognize_rgba_images_diagnostics(
        &mut self,
        images: &[RgbaImage],
    ) -> Result<Vec<KrakenImageDiagnostics>> {
        self.diagnose_rgba_many(images)
    }

    pub fn recognize_gray_images_profile(
        &mut self,
        images: &[GrayImage],
    ) -> Result<KrakenBatchDiagnostics> {
        self.diagnose_gray_many_profile(images)
    }

    pub fn warmup_gray_image(&mut self, image: &GrayImage) -> Result<()> {
        if let Some(blla) = &mut self.blla {
            let _ = blla.recognize(std::slice::from_ref(image))?;
            return Ok(());
        }
        let boxes = self.tesseract_layouts[0].lines(image)?;
        for layout in &mut self.tesseract_layouts[1..] {
            let _ = layout.lines(image)?;
        }
        let lines = boxes
            .into_iter()
            .filter_map(|bbox| prepare_line(image, bbox, self.width_scale))
            .collect::<Vec<_>>();
        for session in &mut self.sessions {
            let _ = recognize_prepared(
                session,
                self.codec.as_ref().expect("fast Kraken codec"),
                &lines,
                self.batch_size,
                self.width_bucket,
            )?;
        }
        Ok(())
    }

    #[cfg(feature = "ocr")]
    pub(crate) fn extract_pages(
        &mut self,
        pdf_path: &Path,
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let bytes = fs::read(pdf_path).map_err(|source| Error::io(pdf_path, source))?;
        let blla = self.blla.is_some();
        if blla || requests.len() < 24 {
            let pdf = Pdf::new(bytes).map_err(|error| {
                Error::Message(format!("OCR renderer could not open PDF: {error:?}"))
            })?;
            let cache = RenderCache::new();
            let interpreter = InterpreterSettings::default();
            let settings = pdf_render_settings(self.dpi);
            let rendered = render_pdf_pages(&pdf, &cache, &interpreter, &settings, requests, blla)?;
            return finish_pdf_pages(rendered, |images| self.diagnose_gray_many(images));
        }

        // Preserve the established OCR batches while extending their producer
        // upstream to render only the next bounded page window.
        let window_size = preparation_window_size(requests.len());
        let width_scale = self.width_scale;
        let batch_size = self.batch_size;
        let width_bucket = self.width_bucket;
        let dpi = self.dpi;
        let codec = self.codec.as_ref().expect("fast Kraken codec");
        let layouts = &mut self.tesseract_layouts;
        let sessions = &mut self.sessions;
        std::thread::scope(|scope| -> Result<_> {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let (metadata_sender, metadata_receiver) = std::sync::mpsc::channel();
            let producer = scope.spawn(move || {
                let result = Pdf::new(bytes).map_err(|error| {
                    Error::Message(format!("OCR renderer could not open PDF: {error:?}"))
                });
                let pdf = match result {
                    Ok(pdf) => pdf,
                    Err(error) => {
                        let _ = sender.send(Err(error));
                        return;
                    }
                };
                let cache = RenderCache::new();
                let interpreter = InterpreterSettings::default();
                let settings = pdf_render_settings(dpi);
                for window in requests.chunks(window_size) {
                    let rendered = match render_pdf_pages(
                        &pdf,
                        &cache,
                        &interpreter,
                        &settings,
                        window,
                        false,
                    ) {
                        Ok(rendered) => rendered,
                        Err(error) => {
                            let _ = sender.send(Err(error));
                            break;
                        }
                    };
                    let (metadata, gray_images): (Vec<_>, Vec<_>) = rendered.into_iter().unzip();
                    let images = gray_images.iter().map(PagePixels::Gray).collect::<Vec<_>>();
                    let value = Self::prepare_window(layouts, &images, width_scale);
                    let failed = value.is_err();
                    if !failed && metadata_sender.send(metadata).is_err() {
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
                requests.len(),
            );
            producer
                .join()
                .map_err(|_| Error::Message("PDF render worker panicked".to_owned()))?;
            finish_recognized_pdf_pages(
                metadata_receiver.into_iter().flatten().collect(),
                result?.pages,
            )
        })
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
    ) -> Result<KrakenImageDiagnostics> {
        let mut page = self
            .diagnose_gray_many(std::slice::from_ref(image))?
            .pop()
            .expect("single-page OCR returns one page");
        scale_lines(&mut page.lines, x_scale, y_scale);
        Ok(page)
    }

    fn diagnose_gray_many(&mut self, images: &[GrayImage]) -> Result<Vec<KrakenImageDiagnostics>> {
        Ok(self.diagnose_gray_many_profile(images)?.pages)
    }

    fn diagnose_gray_many_profile(
        &mut self,
        images: &[GrayImage],
    ) -> Result<KrakenBatchDiagnostics> {
        if images.is_empty() {
            return Ok(KrakenBatchDiagnostics {
                pages: Vec::new(),
                performance: KrakenBatchPerformance::default(),
            });
        }
        if let Some(blla) = &mut self.blla {
            let started = Instant::now();
            let pages = blla.recognize(images)?;
            let performance = KrakenBatchPerformance {
                pages: pages.len(),
                input_pixels: images
                    .iter()
                    .map(|image| image.width() as usize * image.height() as usize)
                    .sum(),
                layout_boxes: pages.iter().map(|page| page.layout_boxes.len()).sum(),
                prepared_lines: pages.iter().map(|page| page.layout_boxes.len()).sum(),
                output_lines: pages.iter().map(|page| page.lines.len()).sum(),
                layout_seconds: pages.iter().map(|page| page.layout_seconds).sum(),
                recognition_wall_seconds: pages.iter().map(|page| page.recognition_seconds).sum(),
                total_seconds: started.elapsed().as_secs_f64(),
                ..KrakenBatchPerformance::default()
            };
            return Ok(KrakenBatchDiagnostics { pages, performance });
        }
        let pages = images.iter().map(PagePixels::Gray).collect::<Vec<_>>();
        self.diagnose_pages_profile(&pages)
    }

    fn diagnose_rgba_many(&mut self, images: &[RgbaImage]) -> Result<Vec<KrakenImageDiagnostics>> {
        if self.blla.is_some() {
            return Err(Error::Message(
                "direct RGBA pages require Tesseract layout".to_owned(),
            ));
        }
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
    ) -> Result<KrakenBatchDiagnostics> {
        if images.is_empty() {
            return Ok(KrakenBatchDiagnostics {
                pages: Vec::new(),
                performance: KrakenBatchPerformance::default(),
            });
        }
        let total_started = Instant::now();
        let input_pixels = images
            .iter()
            .map(|image| image.width() as usize * image.height() as usize)
            .sum();
        // Balance bounded windows so overlap never leaves a one-page tail.
        let window_size = preparation_window_size(images.len());
        let width_scale = self.width_scale;
        let batch_size = self.batch_size;
        let width_bucket = self.width_bucket;
        let codec = self.codec.as_ref().expect("fast Kraken codec");
        let layouts = &mut self.tesseract_layouts;
        let sessions = &mut self.sessions;
        let prepared = std::thread::scope(|scope| -> Result<_> {
            let (sender, receiver) = std::sync::mpsc::sync_channel(1);
            let producer = scope.spawn(move || {
                for window in images.chunks(window_size) {
                    let value = Self::prepare_window(layouts, window, width_scale);
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
                images.len(),
            );
            producer
                .join()
                .map_err(|_| Error::Message("Kraken preparation worker panicked".to_owned()))?;
            result
        })?;
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
        let total_seconds = total_started.elapsed().as_secs_f64();
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
        let performance = KrakenBatchPerformance {
            detailed: true,
            pages: pages.len(),
            input_pixels,
            layout_boxes: layout_box_count,
            prepared_lines: total_lines,
            output_lines,
            layout_seconds,
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
        Ok(KrakenBatchDiagnostics { pages, performance })
    }

    fn recognize_prepared_windows(
        receiver: std::sync::mpsc::Receiver<Result<PreparedWindow>>,
        sessions: &mut [Session],
        codec: &Codec,
        batch_size: usize,
        width_bucket: usize,
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
                            .filter_map(|(line, (text, confidence))| {
                                (!text.trim().is_empty()).then_some(OcrLine {
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
                                    words: vec![],
                                    region_id: String::new(),
                                    region_type: "unknown".to_owned(),
                                    block_index: 0,
                                })
                            })
                            .collect();
                        KrakenImageDiagnostics {
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
        layouts: &mut [TesseractLayout],
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
        layouts: &mut [TesseractLayout],
        images: &[PagePixels<'_>],
        width_scale: f32,
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
        let prepared = prepare_pages(images, &boxes, width_scale)?;
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
        pages: &[Vec<PreparedLine>],
    ) -> Result<(Vec<Vec<(String, f64)>>, RecognitionPerformance)> {
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
                    .map_err(|_| Error::Message("Kraken recognition worker panicked".to_owned()))?;
            }
            if let Some(error) = failure {
                return Err(error);
            }
            let mut output = output
                .into_iter()
                .map(|value| {
                    value.ok_or_else(|| Error::Message("Kraken worker returned no line".to_owned()))
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
) -> Result<RecognizedBatch> {
    let total_started = Instant::now();
    let pack_started = Instant::now();
    let maximum = batch
        .iter()
        .map(|&index| lines[index].width)
        .max()
        .unwrap_or(1);
    pixels.resize(batch.len() * INPUT_HEIGHT * maximum, 0.0);
    pixels.fill(0.0);
    lengths.clear();
    for (local, &index) in batch.iter().enumerate() {
        let line = lines[index];
        lengths.push(line.width as i64);
        let content_width = line.width - INPUT_PADDING * 2;
        for row in 0..INPUT_HEIGHT {
            let source = row * content_width;
            let target = (local * INPUT_HEIGHT + row) * maximum + INPUT_PADDING;
            for (&value, pixel) in line.pixels[source..source + content_width]
                .iter()
                .zip(&mut pixels[target..target + content_width])
            {
                *pixel = 1.0 - f32::from(value) / 255.0;
            }
        }
    }
    let image =
        TensorRef::from_array_view(([batch.len(), 1, INPUT_HEIGHT, maximum], pixels.as_slice()))
            .map_err(ort_error)?;
    let sequence_lengths =
        TensorRef::from_array_view(([batch.len()], lengths.as_slice())).map_err(ort_error)?;
    let pack_seconds = pack_started.elapsed().as_secs_f64();
    let inference_started = Instant::now();
    let outputs = session
        .run(ort::inputs![
            "image" => image,
            "sequence_lengths" => sequence_lengths,
        ])
        .map_err(ort_error)?;
    let inference_seconds = inference_started.elapsed().as_secs_f64();
    let decode_started = Instant::now();
    let (length_shape, output_lengths) =
        if let Ok((shape, values)) = outputs["output_lengths"].try_extract_tensor::<i64>() {
            (
                shape.iter().copied().collect::<Vec<_>>(),
                values
                    .iter()
                    .map(|&value| usize::try_from(value).unwrap_or(0))
                    .collect::<Vec<_>>(),
            )
        } else {
            let (shape, values) = outputs["output_lengths"]
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
        if dimensions.len() != 3
            || dimensions[0] != batch.len()
            || dimensions[1] != 1
            || length_shape.first().copied() != Some(batch.len() as i64)
        {
            return Err(Error::Message(format!(
                "Kraken model returned unexpected class shapes {dimensions:?} and {length_shape:?}"
            )));
        }
        let timesteps = dimensions[2];
        let values = batch
            .iter()
            .enumerate()
            .map(|(local, &index)| {
                let length = output_lengths[local].clamp(1, timesteps);
                let spans = greedy_ids(ids, local, timesteps, length);
                (index, codec.decode(&spans))
            })
            .collect();
        return Ok(RecognizedBatch {
            values,
            worker,
            lines: batch.len(),
            tensor_elements: batch.len() * INPUT_HEIGHT * maximum,
            useful_elements: batch
                .iter()
                .map(|&index| lines[index].width * INPUT_HEIGHT)
                .sum(),
            total_seconds: total_started.elapsed().as_secs_f64(),
            pack_seconds,
            inference_seconds,
            decode_seconds: decode_started.elapsed().as_secs_f64(),
        });
    }
    let logits = outputs.get("logits").ok_or_else(|| {
        Error::Message("Kraken model returned neither logits nor class_ids".to_owned())
    })?;
    let (shape, logits) = logits.try_extract_tensor::<f32>().map_err(ort_error)?;
    let dimensions = shape
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    if dimensions.len() < 3
        || dimensions[0] != batch.len()
        || dimensions[2..dimensions.len() - 1]
            .iter()
            .any(|&value| value != 1)
        || length_shape.first().copied() != Some(batch.len() as i64)
    {
        return Err(Error::Message(format!(
            "Kraken model returned unexpected shapes {dimensions:?} and {length_shape:?}"
        )));
    }
    let classes = dimensions[1];
    let timesteps = *dimensions.last().unwrap_or(&0);
    let values = batch
        .iter()
        .enumerate()
        .map(|(local, &index)| {
            let length = output_lengths[local].clamp(1, timesteps);
            let spans = greedy_ctc(logits, local, classes, timesteps, length);
            (index, codec.decode(&spans))
        })
        .collect();
    Ok(RecognizedBatch {
        values,
        worker,
        lines: batch.len(),
        tensor_elements: batch.len() * INPUT_HEIGHT * maximum,
        useful_elements: batch
            .iter()
            .map(|&index| lines[index].width * INPUT_HEIGHT)
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
) -> Result<Vec<(String, f64)>> {
    let lines = lines.iter().collect::<Vec<_>>();
    let mut output = vec![(String::new(), 0.0); lines.len()];
    let mut pixels = Vec::new();
    let mut lengths = Vec::new();
    for job in recognition_jobs(&lines, batch_size, width_bucket) {
        for (index, value) in
            recognize_batch(session, codec, &lines, &job, &mut pixels, &mut lengths, 0)?.values
        {
            output[index] = value;
        }
    }
    Ok(output)
}

#[cfg(feature = "ocr")]
type PdfPageMetadata = (usize, Option<f64>, f64, f64);
#[cfg(feature = "ocr")]
type RenderedPdfPage = (PdfPageMetadata, GrayImage);

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
    blla: bool,
) -> Result<Vec<RenderedPdfPage>> {
    requests
        .iter()
        .map(|request| {
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
            let metadata = (
                request.page_index,
                super::ocr::raster_separator_y(&pixmap, request.height),
                request.width / width as f64,
                request.height / height as f64,
            );
            let pixels = pixmap
                .data_as_u8_slice()
                .chunks_exact(4)
                .map(|pixel| {
                    if blla {
                        ((u32::from(pixel[0]) * 77
                            + u32::from(pixel[1]) * 150
                            + u32::from(pixel[2]) * 29)
                            / 256) as u8
                    } else {
                        ((u32::from(pixel[0]) + u32::from(pixel[1]) + u32::from(pixel[2]) + 1) / 3)
                            as u8
                    }
                })
                .collect();
            let image = GrayImage::from_raw(width as u32, height as u32, pixels)
                .expect("renderer dimensions match its pixel buffer");
            Ok((metadata, image))
        })
        .collect()
}

#[cfg(feature = "ocr")]
fn finish_pdf_pages(
    rendered: Vec<RenderedPdfPage>,
    recognize: impl FnOnce(&[GrayImage]) -> Result<Vec<KrakenImageDiagnostics>>,
) -> Result<Vec<OcrPageResult>> {
    let (metadata, images): (Vec<_>, Vec<_>) = rendered.into_iter().unzip();
    finish_recognized_pdf_pages(metadata, recognize(&images)?)
}

#[cfg(feature = "ocr")]
fn finish_recognized_pdf_pages(
    metadata: Vec<PdfPageMetadata>,
    pages: Vec<KrakenImageDiagnostics>,
) -> Result<Vec<OcrPageResult>> {
    Ok(metadata
        .into_iter()
        .zip(pages)
        .map(|((page_index, separator_y, x_scale, y_scale), mut page)| {
            scale_lines(&mut page.lines, x_scale, y_scale);
            OcrPageResult {
                page_index,
                lines: page.lines,
                separator_y,
            }
        })
        .collect())
}

fn preparation_window_size(page_count: usize) -> usize {
    if page_count < 24 {
        page_count
    } else {
        page_count.div_ceil(page_count.div_ceil(32).max(2))
    }
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

fn recognition_schedule(
    requested_workers: usize,
    requested_threads: usize,
    backend: KrakenBackend,
) -> (usize, usize) {
    if backend != KrakenBackend::Cpu {
        return (requested_workers.max(1), requested_threads.max(1));
    }
    let available = std::thread::available_parallelism().map_or(1, |value| value.get());
    let automatic_threads = if requested_threads == 0 {
        available.min(2)
    } else {
        requested_threads
    };
    let workers = if requested_workers == 0 {
        (available / automatic_threads).max(1)
    } else {
        requested_workers
    };
    let threads = if requested_threads == 0 {
        (available / workers).clamp(1, 2)
    } else {
        automatic_threads
    };
    (workers, threads)
}

fn uses_global_pool(options: &KrakenOptions) -> bool {
    options.backend == KrakenBackend::Cpu
        && options.workers == 0
        && options.threads == 0
        && options.tier != KrakenTier::Quality
}

fn open_session(
    model: &Path,
    threads: usize,
    backend: KrakenBackend,
    device: &str,
    cpu_fallback: bool,
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
    if backend == KrakenBackend::DirectMl {
        builder = builder
            .with_memory_pattern(false)
            .and_then(|builder| builder.with_parallel_execution(false))
            .map_err(ort_error)?;
    }
    if backend == KrakenBackend::Cpu && !cpu_arena {
        builder = builder
            .with_execution_providers(vec![CPUExecutionProvider::default()
                .with_arena_allocator(false)
                .build()])
            .map_err(ort_error)?;
    }
    let providers = backend.providers(device);
    if !providers.is_empty() {
        builder = builder
            .with_execution_providers(providers)
            .map_err(ort_error)?;
        if !cpu_fallback && backend != KrakenBackend::Cpu {
            builder = builder
                .with_config_entry("session.disable_cpu_ep_fallback", "1")
                .map_err(ort_error)?;
        }
    }
    builder.commit_from_file(model).map_err(ort_error)
}

fn validate_options(options: &KrakenOptions) -> Result<()> {
    if options.backend == KrakenBackend::Cpu && options.cpu_fallback {
        return Err(Error::Message(
            "Kraken CPU fallback only applies to accelerator backends".to_owned(),
        ));
    }
    if options.backend != KrakenBackend::Cpu && !options.cpu_arena {
        return Err(Error::Message(
            "Kraken low-memory mode only applies to the CPU backend".to_owned(),
        ));
    }
    if !(72..=600).contains(&options.dpi) {
        return Err(Error::Message(
            "OCR DPI must be between 72 and 600".to_owned(),
        ));
    }
    if !(1..=256).contains(&options.batch_size)
        || options.runtime_batch_size > 256
        || !(1..=1024).contains(&options.width_bucket)
        || !(1..=16).contains(&options.layout_workers)
        || options.workers > 16
        || options.threads > 16
    {
        return Err(Error::Message(
            "Kraken batch, bucket, and layout-worker counts must be positive and bounded"
                .to_owned(),
        ));
    }
    if !(1..=86_400).contains(&options.timeout_seconds) {
        return Err(Error::Message(
            "Kraken timeout must be between 1 and 86400 seconds".to_owned(),
        ));
    }
    let has_blla_assets = options.runtime_wheel.is_some()
        || options.python.is_some()
        || options.blla_pack.is_some()
        || options.recognizer_pack.is_some();
    if options.layout == KrakenLayout::Tesseract && has_blla_assets {
        return Err(Error::Message(
            "Kraken BLLA runtime options require --kraken-layout blla".to_owned(),
        ));
    }
    if options.layout == KrakenLayout::Blla
        && (options.model.is_some()
            || options.codec.is_some()
            || options.runtime.is_some()
            || options.tesseract_library.is_some()
            || options.width_scale.is_some())
    {
        return Err(Error::Message(
            "Kraken native Tesseract-layout options cannot be combined with BLLA orchestration"
                .to_owned(),
        ));
    }
    let width_scale = options
        .width_scale
        .unwrap_or_else(|| options.tier.width_scale());
    if !(0.5..=1.25).contains(&width_scale) || !width_scale.is_finite() {
        return Err(Error::Message(
            "Kraken width scale must be between 0.5 and 1.25".to_owned(),
        ));
    }
    options
        .backend
        .normalized_device(options.device.as_deref())?;
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

fn ort_error(error: ort::Error) -> Error {
    Error::Message(format!("Kraken ONNX inference failed: {error}"))
}

pub(crate) fn sha256_file(path: &Path) -> Result<String> {
    let mut reader = BufReader::new(File::open(path).map_err(|source| Error::io(path, source))?);
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

fn prepare_line(image: &GrayImage, bbox: LineBox, width_scale: f32) -> Option<PreparedLine> {
    if bbox.width() == 0 || bbox.height() == 0 {
        return None;
    }
    let crop = imageops::crop_imm(
        image,
        bbox.left as u32,
        bbox.top as u32,
        bbox.width() as u32,
        bbox.height() as u32,
    );
    let content_width = ((bbox.width() as f32 * INPUT_HEIGHT as f32 / bbox.height() as f32)
        * width_scale)
        .round()
        .max(1.0) as usize;
    let resized = imageops::resize(
        &*crop,
        content_width as u32,
        INPUT_HEIGHT as u32,
        imageops::FilterType::Lanczos3,
    );
    let width = content_width + INPUT_PADDING * 2;
    Some(PreparedLine {
        bbox,
        width,
        pixels: resized.into_raw(),
    })
}

fn prepare_rgba_line(
    source: &[u8],
    page_width: usize,
    bbox: LineBox,
    width_scale: f32,
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
    let content_width = ((bbox.width() as f32 * INPUT_HEIGHT as f32 / bbox.height() as f32)
        * width_scale)
        .round()
        .max(1.0) as usize;
    let resized = imageops::resize(
        &gray,
        content_width as u32,
        INPUT_HEIGHT as u32,
        imageops::FilterType::Lanczos3,
    );
    Some(PreparedLine {
        bbox,
        width: content_width + INPUT_PADDING * 2,
        pixels: resized.into_raw(),
    })
}

fn prepare_page_line(
    image: PagePixels<'_>,
    bbox: LineBox,
    width_scale: f32,
) -> Option<PreparedLine> {
    match image {
        PagePixels::Gray(image) => prepare_line(image, bbox, width_scale),
        PagePixels::Rgba { pixels, width, .. } => {
            prepare_rgba_line(pixels, width as usize, bbox, width_scale)
        }
    }
}

fn prepare_pages(
    images: &[PagePixels<'_>],
    boxes: &[Vec<LineBox>],
    width_scale: f32,
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
                        .filter_map(|&bbox| prepare_page_line(images[index], bbox, width_scale))
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
    let offset = batch * classes * timesteps;
    let mut labels = vec![0; length];
    let mut maxima = vec![f32::NEG_INFINITY; length];
    for class in 0..classes {
        let values = &logits[offset + class * timesteps..offset + class * timesteps + length];
        for (timestep, &value) in values.iter().enumerate() {
            if value > maxima[timestep] {
                maxima[timestep] = value;
                labels[timestep] = class;
            }
        }
    }
    let mut denominators = vec![0.0_f32; length];
    for class in 0..classes {
        let values = &logits[offset + class * timesteps..offset + class * timesteps + length];
        for (timestep, &value) in values.iter().enumerate() {
            if labels[timestep] != BLANK_LABEL {
                denominators[timestep] += (value - maxima[timestep]).exp();
            }
        }
    }
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
    fn kraken_options_fail_closed_without_assets() {
        let options = KrakenOptions::default();
        assert!(KrakenOcr::new(&options).is_err());
    }

    #[test]
    fn ctc_collapses_repeats_but_not_blank_separated_labels() {
        let logits = [0.0, 0.0, 5.0, 5.0, 0.0, 5.0, 5.0, 0.0, 0.0, 5.0];
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
                bbox: LineBox {
                    left: 0,
                    top: 0,
                    right: width,
                    bottom: 1,
                },
                width,
                pixels: Vec::new(),
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
    fn recognition_schedule_uses_the_measured_cpu_shape_without_oversubscribing() {
        let available = std::thread::available_parallelism().map_or(1, |value| value.get());
        let expected_workers = (available / 2).max(1);
        let expected_threads = (available / expected_workers).clamp(1, 2);
        assert_eq!(
            recognition_schedule(0, 0, KrakenBackend::Cpu),
            (expected_workers, expected_threads)
        );
        assert_eq!(recognition_schedule(7, 0, KrakenBackend::Cpu).0, 7);
        assert_eq!(recognition_schedule(0, 0, KrakenBackend::Cuda), (1, 1));
        assert_eq!(recognition_schedule(4, 2, KrakenBackend::Cuda), (4, 2));
    }

    #[test]
    fn shared_pool_is_reserved_for_automatic_speed_tiers() {
        let mut options = KrakenOptions::default();
        assert!(!uses_global_pool(&options));
        options.tier = KrakenTier::Balanced;
        assert!(uses_global_pool(&options));
        options.workers = 4;
        assert!(!uses_global_pool(&options));
    }

    #[test]
    fn recognition_profile_accounts_for_batch_padding_and_worker_balance() {
        let prepared = [100, 120, 200].map(|width| PreparedLine {
            bbox: LineBox {
                left: 0,
                top: 0,
                right: width,
                bottom: 1,
            },
            width,
            pixels: Vec::new(),
        });
        let lines = prepared.iter().collect::<Vec<_>>();
        let batches = [
            RecognizedBatch {
                values: Vec::new(),
                worker: 0,
                lines: 2,
                tensor_elements: 2 * INPUT_HEIGHT * 120,
                useful_elements: INPUT_HEIGHT * 220,
                total_seconds: 0.2,
                pack_seconds: 0.01,
                inference_seconds: 0.18,
                decode_seconds: 0.01,
            },
            RecognizedBatch {
                values: Vec::new(),
                worker: 1,
                lines: 1,
                tensor_elements: INPUT_HEIGHT * 200,
                useful_elements: INPUT_HEIGHT * 200,
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
            2 * INPUT_HEIGHT * 120 * std::mem::size_of::<f32>() + 2 * std::mem::size_of::<i64>()
        );
    }

    #[test]
    fn backends_parse_and_normalize_devices_fail_closed() {
        assert_eq!(KrakenBackend::parse("cpu"), Some(KrakenBackend::Cpu));
        assert_eq!(
            KrakenBackend::parse("tensorrt"),
            Some(KrakenBackend::TensorRt)
        );
        assert_eq!(KrakenBackend::parse("auto"), None);
        assert_eq!(KrakenBackend::Cuda.normalized_device(None).unwrap(), "0");
        assert_eq!(
            KrakenBackend::OpenVino
                .normalized_device(Some("GPU.1"))
                .unwrap(),
            "GPU.1"
        );
        assert!(KrakenBackend::Cpu.normalized_device(Some("0")).is_err());
        assert!(KrakenBackend::DirectMl
            .normalized_device(Some("-1"))
            .is_err());
        let mut options = KrakenOptions::default();
        options.cpu_fallback = true;
        assert!(validate_options(&options).is_err());
    }
}
