use crate::ppdoc_openvino::OpenVinoSession;
use crate::ppdoc_postprocess::{
    best_region_index, postprocess_document, scale_detections, RegionDetection,
};
use image::{imageops, ImageReader, RgbImage};
use legal_pdf_core::model::{Diagnostic, Page};
pub use legal_pdf_core::OrtBackend as PPDocBackend;
use legal_pdf_core::{Error, Result};
#[cfg(feature = "ppdoc")]
use libloading::Library;
#[cfg(feature = "ppdoc")]
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputValue},
    value::Tensor,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
#[cfg(feature = "ppdoc")]
use std::ffi::{c_char, c_void, CStr};
use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};

const PACK_FORMAT: &str = "legalpdf.ppdoc-lite-model/1";

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PPDocOptions {
    pub model_pack: Option<PathBuf>,
    pub runtime: Option<PathBuf>,
    pub cache_dir: Option<PathBuf>,
    pub threads: usize,
    pub threshold: f32,
    pub render_dpi: u16,
    pub onednn: bool,
    pub backend: PPDocBackend,
    pub device: Option<String>,
    pub cpu_fallback: bool,
    pub expected_identity: Option<String>,
}

impl Default for PPDocOptions {
    fn default() -> Self {
        Self {
            model_pack: None,
            runtime: None,
            cache_dir: None,
            threads: 0,
            threshold: 0.10,
            render_dpi: 72,
            onednn: false,
            backend: PPDocBackend::Cpu,
            device: None,
            cpu_fallback: false,
            expected_identity: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct PPDocDetection {
    pub label_id: usize,
    pub label: String,
    pub score: f32,
    pub bbox: [f32; 4],
    pub order: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct Manifest {
    format: String,
    variant_id: String,
    labels: Vec<String>,
    model: ModelManifest,
    input: InputManifest,
    #[serde(default)]
    postprocess: PostprocessManifest,
}

#[derive(Debug, Deserialize)]
struct ModelManifest {
    #[serde(default = "default_backend")]
    backend: String,
    file: String,
    sha256: String,
    params: Option<String>,
    params_sha256: Option<String>,
    #[serde(default)]
    files: Vec<CompanionManifest>,
    inputs: Vec<String>,
    outputs: BTreeMap<String, String>,
    #[serde(default = "default_detections_per_image")]
    detections_per_image: usize,
    #[serde(default = "default_output_width")]
    output_width: usize,
}

#[derive(Debug, Deserialize)]
struct CompanionManifest {
    file: String,
    sha256: String,
}

#[derive(Debug, Default, Deserialize)]
struct PostprocessManifest {
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    model_nms: ModelNmsManifest,
}

#[derive(Debug, Deserialize)]
struct ModelNmsManifest {
    #[serde(default = "default_nms_score_threshold")]
    score_threshold: f32,
    #[serde(default = "default_nms_threshold")]
    nms_threshold: f32,
    #[serde(default = "default_nms_top_k")]
    nms_top_k: usize,
}

impl Default for ModelNmsManifest {
    fn default() -> Self {
        Self {
            score_threshold: default_nms_score_threshold(),
            nms_threshold: default_nms_threshold(),
            nms_top_k: default_nms_top_k(),
        }
    }
}

const fn default_nms_score_threshold() -> f32 {
    0.01
}

const fn default_nms_threshold() -> f32 {
    0.7
}

const fn default_nms_top_k() -> usize {
    1_000
}

fn default_backend() -> String {
    "onnx".to_owned()
}

const fn default_detections_per_image() -> usize {
    300
}

const fn default_output_width() -> usize {
    7
}

#[derive(Debug, Deserialize)]
struct InputManifest {
    target_size: [usize; 2],
    #[serde(default = "default_interpolation")]
    interpolation: String,
    #[serde(default = "default_scale")]
    scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
}

const fn default_scale() -> f32 {
    1.0 / 255.0
}

fn default_interpolation() -> String {
    "opencv_cubic".to_owned()
}

struct ModelPack {
    manifest: Manifest,
    model: PathBuf,
    model_sha256: String,
    #[cfg(feature = "ppdoc")]
    params: Option<PathBuf>,
    params_sha256: Option<String>,
    companion_sha256: Vec<String>,
}

impl ModelPack {
    fn load(path: &Path) -> Result<Self> {
        let root = fs::canonicalize(path).map_err(|source| Error::io(path, source))?;
        let manifest_path = root.join("manifest.json");
        let manifest: Manifest = serde_json::from_reader(BufReader::new(
            File::open(&manifest_path).map_err(|source| Error::io(&manifest_path, source))?,
        ))?;
        if manifest.format != PACK_FORMAT || manifest.variant_id.trim().is_empty() {
            return Err(Error::Message(
                "PPdoc model pack has an unsupported format or empty variant ID".to_owned(),
            ));
        }
        if manifest.labels.is_empty() || manifest.labels.iter().any(|label| label.trim().is_empty())
        {
            return Err(Error::Message(
                "PPdoc model pack has no usable labels".to_owned(),
            ));
        }
        if manifest.input.target_size.contains(&0)
            || manifest.input.mean.iter().any(|value| !value.is_finite())
            || manifest
                .input
                .std
                .iter()
                .any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(Error::Message(
                "PPdoc model pack has invalid preprocessing values".to_owned(),
            ));
        }
        if manifest
            .model
            .inputs
            .iter()
            .any(|name| name.trim().is_empty())
            || manifest
                .model
                .inputs
                .iter()
                .enumerate()
                .any(|(index, name)| manifest.model.inputs[index + 1..].contains(name))
            || !matches!(
                manifest.input.interpolation.as_str(),
                "opencv_cubic" | "bilinear"
            )
        {
            return Err(Error::Message(
                "PPdoc model pack has invalid input names or preprocessing interpolation"
                    .to_owned(),
            ));
        }
        if !matches!(manifest.model.inputs.as_slice(), [_image])
            && !matches!(manifest.model.inputs.as_slice(), [image, scale] if image == "image" && scale == "scale_factor")
            && !matches!(manifest.model.inputs.as_slice(), [shape, image, scale] if shape == "im_shape" && image == "image" && scale == "scale_factor")
        {
            return Err(Error::Message(format!(
                "PPdoc model inputs are unsupported: {:?}",
                manifest.model.inputs
            )));
        }
        match manifest.model.outputs.get("contract").map(String::as_str) {
            Some("decoded_boxes") if manifest.model.outputs.contains_key("boxes") => {}
            Some("rtdetr_raw")
                if matches!(manifest.model.backend.as_str(), "onnx" | "openvino")
                    && manifest.model.outputs.contains_key("boxes")
                    && manifest.model.outputs.contains_key("logits") => {}
            Some("ppyoloe_raw")
                if matches!(manifest.model.backend.as_str(), "onnx" | "openvino")
                    && manifest.model.outputs.contains_key("boxes")
                    && manifest.model.outputs.contains_key("scores") => {}
            contract => {
                return Err(Error::Message(format!(
                    "unsupported PPdoc output contract: {contract:?}"
                )))
            }
        }
        if !matches!(
            manifest.model.backend.as_str(),
            "onnx" | "paddle" | "openvino"
        ) {
            return Err(Error::Message(format!(
                "unsupported PPdoc backend: {}",
                manifest.model.backend
            )));
        }
        if manifest.model.detections_per_image == 0 || manifest.model.output_width < 6 {
            return Err(Error::Message(
                "PPdoc model has invalid decoded-box dimensions".to_owned(),
            ));
        }
        if manifest.model.outputs.get("contract").map(String::as_str) == Some("ppyoloe_raw")
            && (!manifest.postprocess.model_nms.score_threshold.is_finite()
                || !(0.0..=1.0).contains(&manifest.postprocess.model_nms.score_threshold)
                || !manifest.postprocess.model_nms.nms_threshold.is_finite()
                || !(0.0..=1.0).contains(&manifest.postprocess.model_nms.nms_threshold)
                || manifest.postprocess.model_nms.nms_top_k == 0)
        {
            return Err(Error::Message(
                "PPdoc PP-YOLOE pack has invalid model NMS settings".to_owned(),
            ));
        }
        if manifest
            .postprocess
            .score_threshold
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(Error::Message(
                "PPdoc model pack has an invalid postprocess score threshold".to_owned(),
            ));
        }
        let model = pack_file(&root, &manifest.model.file, "model")?;
        let model_sha256 = sha256_file(&model)?;
        if model_sha256 != manifest.model.sha256.to_ascii_lowercase() {
            return Err(Error::Message(format!(
                "PPdoc model hash mismatch: expected {}, found {model_sha256}",
                manifest.model.sha256
            )));
        }
        let companion_sha256 = manifest
            .model
            .files
            .iter()
            .map(|companion| {
                let path = pack_file(&root, &companion.file, "companion")?;
                let actual = sha256_file(&path)?;
                if actual != companion.sha256.to_ascii_lowercase() {
                    return Err(Error::Message(format!(
                        "PPdoc companion hash mismatch for {}: expected {}, found {actual}",
                        companion.file, companion.sha256
                    )));
                }
                Ok(actual)
            })
            .collect::<Result<Vec<_>>>()?;
        if manifest.model.backend == "openvino" && companion_sha256.is_empty() {
            return Err(Error::Message(
                "OpenVINO IR model packs require a hash-checked companion file".to_owned(),
            ));
        }
        let (params, params_sha256) = match (
            manifest.model.params.as_deref(),
            manifest.model.params_sha256.as_deref(),
        ) {
            (Some(file), Some(expected)) => {
                let path = pack_file(&root, file, "parameter")?;
                let actual = sha256_file(&path)?;
                if actual != expected.to_ascii_lowercase() {
                    return Err(Error::Message(format!(
                        "PPdoc parameter hash mismatch: expected {expected}, found {actual}"
                    )));
                }
                (Some(path), Some(actual))
            }
            (None, None) if matches!(manifest.model.backend.as_str(), "onnx" | "openvino") => {
                (None, None)
            }
            _ => {
                return Err(Error::Message(
                    "Paddle model packs require params and params_sha256".to_owned(),
                ))
            }
        };
        #[cfg(not(feature = "ppdoc"))]
        let _ = params;
        Ok(Self {
            manifest,
            model,
            model_sha256,
            #[cfg(feature = "ppdoc")]
            params,
            params_sha256,
            companion_sha256,
        })
    }
}

enum Backend {
    #[cfg(feature = "ppdoc")]
    Onnx(Session),
    #[cfg(feature = "ppdoc")]
    Paddle(PaddleSession),
    OpenVino(OpenVinoSession),
}

pub struct PPDocLayout {
    backend: Backend,
    pack: ModelPack,
    threshold: f32,
    render_dpi: u16,
    identity: String,
}

pub struct PreparedPPDoc {
    pack: ModelPack,
    runtime: PathBuf,
    cache_dir: Option<PathBuf>,
    device: String,
    threshold: f32,
    identity: String,
}

impl PreparedPPDoc {
    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn variant_id(&self) -> &str {
        &self.pack.manifest.variant_id
    }
}

impl PPDocLayout {
    pub fn new(options: &PPDocOptions) -> Result<Self> {
        Self::from_prepared(options, Self::prepare(options)?)
    }

    pub fn prepare(options: &PPDocOptions) -> Result<PreparedPPDoc> {
        if !(0.0..=1.0).contains(&options.threshold) || !options.threshold.is_finite() {
            return Err(Error::Message(
                "PPdoc threshold must be between zero and one".to_owned(),
            ));
        }
        if !(72..=600).contains(&options.render_dpi) {
            return Err(Error::Message(
                "PPdoc render DPI must be between 72 and 600".to_owned(),
            ));
        }
        let pack_path = required_path(
            &options.model_pack,
            "LEGALPDF_PPDOC_MODEL_PACK",
            "PPdoc model pack",
        )?;
        let pack = ModelPack::load(&pack_path)?;
        let native_openvino = options.backend == PPDocBackend::OpenVino;
        #[cfg(not(feature = "ppdoc"))]
        if !native_openvino || pack.manifest.model.backend == "paddle" {
            return Err(Error::Message(
                "this thin build supports only direct OpenVINO inference; rebuild with --features ppdoc-full for ONNX, Paddle, or GPU execution providers"
                    .to_owned(),
            ));
        }
        if pack.manifest.model.backend == "openvino" && !native_openvino {
            return Err(Error::Message(
                "OpenVINO IR model packs require --backend openvino".to_owned(),
            ));
        }
        let (runtime_variable, runtime_label) = if pack.manifest.model.backend == "paddle" {
            ("LEGALPDF_PPDOC_RUNTIME", "Paddle runtime library")
        } else if native_openvino {
            ("LEGALPDF_OPENVINO_RUNTIME", "OpenVINO C runtime library")
        } else {
            ("LEGALPDF_ONNXRUNTIME", "ONNX Runtime library")
        };
        let runtime = required_path(&options.runtime, runtime_variable, runtime_label)?;
        let runtime = canonical_file(&runtime, runtime_label)?;
        let cache_dir = options
            .cache_dir
            .clone()
            .or_else(|| std::env::var_os("LEGALPDF_PPDOC_CACHE_DIR").map(PathBuf::from))
            .filter(|path| !path.as_os_str().is_empty())
            .map(|path| {
                fs::create_dir_all(&path).map_err(|source| Error::io(&path, source))?;
                fs::canonicalize(&path).map_err(|source| Error::io(&path, source))
            })
            .transpose()?;
        if pack.manifest.model.backend == "paddle"
            && (options.backend != PPDocBackend::Cpu
                || options.device.is_some()
                || options.cpu_fallback)
        {
            return Err(Error::Message(
                "PPdoc accelerator options require an ONNX model pack".to_owned(),
            ));
        }
        if native_openvino && pack.manifest.model.backend == "paddle" {
            return Err(Error::Message(
                "OpenVINO requires an ONNX or OpenVINO IR model pack".to_owned(),
            ));
        }
        if native_openvino && options.cpu_fallback {
            return Err(Error::Message(
                "native OpenVINO does not use ONNX Runtime CPU fallback".to_owned(),
            ));
        }
        if options.backend == PPDocBackend::Cpu && options.cpu_fallback {
            return Err(Error::Message(
                "PPdoc CPU fallback only applies to accelerator backends".to_owned(),
            ));
        }
        let device = options
            .backend
            .normalized_device(options.device.as_deref())?;
        let threshold = options.threshold.max(
            pack.manifest
                .postprocess
                .score_threshold
                .unwrap_or_default(),
        );
        let identity = format!(
            "ppdoc-lite-rust-v5:variant={}:model_backend={}:execution_backend={}:device={}:fallback={}:model={}:companions={}:params={}:runtime={}:onednn={}:threshold={:.6}:render_dpi={}:input={}:interpolation={}",
            pack.manifest.variant_id,
            pack.manifest.model.backend,
            options.backend.name(),
            device,
            if options.cpu_fallback { "cpu" } else { "none" },
            pack.model_sha256,
            if pack.companion_sha256.is_empty() {
                "none".to_owned()
            } else {
                pack.companion_sha256.join(",")
            },
            pack.params_sha256.as_deref().unwrap_or("none"),
            sha256_file(&runtime)?,
            options.onednn,
            threshold,
            options.render_dpi,
            pack.manifest.model.inputs.join(","),
            pack.manifest.input.interpolation,
        );
        if options
            .expected_identity
            .as_deref()
            .is_some_and(|expected| expected != identity)
        {
            return Err(Error::Message(format!(
                "PPdoc identity changed before inference: expected {}, found {identity}",
                options.expected_identity.as_deref().unwrap_or_default()
            )));
        }
        Ok(PreparedPPDoc {
            pack,
            runtime,
            cache_dir,
            device,
            threshold,
            identity,
        })
    }

    pub fn from_prepared(options: &PPDocOptions, prepared: PreparedPPDoc) -> Result<Self> {
        let PreparedPPDoc {
            pack,
            runtime,
            cache_dir,
            device,
            threshold,
            identity,
        } = prepared;
        let native_openvino = options.backend == PPDocBackend::OpenVino;
        #[cfg(feature = "ppdoc")]
        let backend = if pack.manifest.model.backend == "paddle" {
            Backend::Paddle(PaddleSession::new(
                &runtime,
                &pack.model,
                pack.params.as_deref().expect("Paddle params validated"),
                options.threads,
                options.onednn,
            )?)
        } else if native_openvino {
            Backend::OpenVino(OpenVinoSession::new(
                &runtime,
                &pack.model,
                &device,
                options.threads,
                cache_dir.as_deref(),
            )?)
        } else {
            let environment_global_threads = legal_pdf_core::init_ort_runtime(&runtime, None)?;
            let mut builder = Session::builder()
                .map_err(ort_error)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_error)?;
            if environment_global_threads.is_some() {
                builder = builder.with_independent_thread_pool().map_err(ort_error)?;
            }
            if options.threads > 0 {
                builder = builder
                    .with_intra_threads(options.threads)
                    .map_err(ort_error)?;
            }
            builder = builder.with_inter_threads(1).map_err(ort_error)?;
            if options.backend == PPDocBackend::DirectMl {
                builder = builder
                    .with_memory_pattern(false)
                    .and_then(|builder| builder.with_parallel_execution(false))
                    .map_err(ort_error)?;
            }
            let providers = options.backend.providers(&device);
            if !providers.is_empty() {
                builder = builder
                    .with_execution_providers(providers)
                    .map_err(ort_error)?;
                if !options.cpu_fallback {
                    builder = builder
                        .with_config_entry("session.disable_cpu_ep_fallback", "1")
                        .map_err(ort_error)?;
                }
            }
            Backend::Onnx(builder.commit_from_file(&pack.model).map_err(ort_error)?)
        };
        #[cfg(not(feature = "ppdoc"))]
        let backend = Backend::OpenVino(OpenVinoSession::new(
            &runtime,
            &pack.model,
            &device,
            options.threads,
            cache_dir.as_deref(),
        )?);
        Ok(Self {
            backend,
            pack,
            threshold,
            render_dpi: options.render_dpi,
            identity,
        })
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn variant_id(&self) -> &str {
        &self.pack.manifest.variant_id
    }

    pub fn detect_image(&mut self, path: impl AsRef<Path>) -> Result<Vec<PPDocDetection>> {
        let path = path.as_ref();
        let image = ImageReader::open(path)
            .map_err(|source| Error::io(path, source))?
            .decode()
            .map_err(|source| {
                Error::Message(format!("could not decode {}: {source}", path.display()))
            })?
            .into_rgb8();
        self.detect_rgb(&image)
    }

    pub fn detect_rgb(&mut self, image: &RgbImage) -> Result<Vec<PPDocDetection>> {
        if image.width() == 0 || image.height() == 0 {
            return Err(Error::Message("PPdoc input image is empty".to_owned()));
        }
        let [target_height, target_width] = self.pack.manifest.input.target_size;
        let input = &self.pack.manifest.input;
        let pixels = match input.interpolation.as_str() {
            "opencv_cubic" => resize_opencv_cubic_nchw(
                image,
                target_width as u32,
                target_height as u32,
                input.scale,
                input.mean,
                input.std,
            ),
            "bilinear" => resize_bilinear_nchw(
                image,
                target_width as u32,
                target_height as u32,
                input.scale,
                input.mean,
                input.std,
            ),
            _ => unreachable!("interpolation validated when the pack was loaded"),
        };
        let im_shape_values = [target_height as f32, target_width as f32];
        let scale_values = [
            target_height as f32 / image.height() as f32,
            target_width as f32 / image.width() as f32,
        ];
        let (values, width, count) = match &mut self.backend {
            #[cfg(feature = "ppdoc")]
            Backend::Paddle(session) => session.run(
                &pixels,
                target_height,
                target_width,
                &im_shape_values,
                &scale_values,
                self.pack.manifest.model.detections_per_image,
                self.pack.manifest.model.output_width,
            )?,
            #[cfg(feature = "ppdoc")]
            Backend::Onnx(session) => run_onnx(
                session,
                &self.pack.manifest.model,
                &self.pack.manifest.postprocess.model_nms,
                self.pack.manifest.labels.len(),
                self.threshold,
                pixels,
                target_height,
                target_width,
                im_shape_values,
                scale_values,
            )?,
            Backend::OpenVino(session) => run_openvino(
                session,
                &self.pack.manifest.model,
                &self.pack.manifest.postprocess.model_nms,
                self.pack.manifest.labels.len(),
                self.threshold,
                pixels,
                target_height,
                target_width,
                im_shape_values,
                scale_values,
            )?,
        };
        Ok(postprocess(
            &values,
            width,
            count,
            &self.pack.manifest.labels,
            image.width(),
            image.height(),
            self.threshold,
        ))
    }

    pub fn annotate_pdf(&mut self, pdf_path: &Path, pages: &mut [Page]) -> Result<Vec<Diagnostic>> {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};

        let bytes = fs::read(pdf_path).map_err(|source| Error::io(pdf_path, source))?;
        let pdf = Pdf::new(bytes).map_err(|error| {
            Error::Message(format!("PPdoc renderer could not open PDF: {error:?}"))
        })?;
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        let scale = f32::from(self.render_dpi) / 72.0;
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: WHITE,
            ..Default::default()
        };
        let mut regions_by_page = vec![Vec::<RegionDetection>::new(); pages.len()];
        let mut detection_count = 0;

        for (page_position, page) in pages.iter().enumerate() {
            if !page
                .lines
                .iter()
                .any(|line| !line.exclude_from_body && !line.text.trim().is_empty())
            {
                continue;
            }
            let pdf_page = pdf.pages().iter().nth(page.index).ok_or_else(|| {
                Error::Message(format!("PDF page index is out of range: {}", page.index))
            })?;
            let pixmap = render(pdf_page, &cache, &interpreter, &settings);
            let width = u32::from(pixmap.width());
            let height = u32::from(pixmap.height());
            if width == 0 || height == 0 {
                return Err(Error::Message(format!(
                    "PPdoc renderer produced an empty page image for page {}",
                    page.number
                )));
            }
            let pixels = pixmap
                .data_as_u8_slice()
                .chunks_exact(4)
                .flat_map(|pixel| [pixel[0], pixel[1], pixel[2]])
                .collect();
            let image = RgbImage::from_raw(width, height, pixels)
                .expect("renderer dimensions match its pixel buffer");
            let detections = self.detect_rgb(&image)?;
            detection_count += detections.len();
            regions_by_page[page_position] =
                scale_detections(page.width, page.height, width, height, &detections);
        }

        crate::profile::measure("ppdoc_postprocess", || {
            postprocess_document(pages, &mut regions_by_page)
        });
        let mut pending = Vec::<(usize, usize, String, String)>::new();
        let mut unmatched = Vec::new();
        for (page_position, page) in pages.iter().enumerate() {
            for (line_index, line) in page.lines.iter().enumerate() {
                if line.exclude_from_body || line.text.trim().is_empty() {
                    continue;
                }
                let Some(region_index) =
                    best_region_index(line.bbox, &regions_by_page[page_position])
                else {
                    unmatched.push(line.id.clone());
                    continue;
                };
                let region = &regions_by_page[page_position][region_index];
                pending.push((
                    page_position,
                    line_index,
                    region.label.clone(),
                    format!("{}-ppdoc-r{:04}", page.id, region.raw_index),
                ));
            }
        }

        if !unmatched.is_empty() {
            let mut diagnostic = Diagnostic::warning(
                "PPDOC_LAYOUT_INCOMPLETE",
                "PPdoc did not cover every text line; model regions were discarded.",
                None,
            );
            diagnostic.line_ids = unmatched;
            diagnostic
                .details
                .insert("detections".to_owned(), serde_json::json!(detection_count));
            diagnostic
                .details
                .insert("matched_lines".to_owned(), serde_json::json!(pending.len()));
            return Ok(vec![diagnostic]);
        }

        for (page_index, line_index, label, region_id) in pending {
            let line = &mut pages[page_index].lines[line_index];
            line.region_type = label;
            line.region_id = region_id;
        }
        Ok(Vec::new())
    }
}

#[allow(clippy::too_many_arguments)]
fn run_openvino(
    session: &mut OpenVinoSession,
    model: &ModelManifest,
    model_nms: &ModelNmsManifest,
    class_count: usize,
    threshold: f32,
    pixels: Vec<f32>,
    target_height: usize,
    target_width: usize,
    im_shape_values: [f32; 2],
    scale_values: [f32; 2],
) -> Result<(Vec<f32>, usize, usize)> {
    let contract = model.outputs.get("contract").map(String::as_str);
    if contract == Some("decoded_boxes") {
        let outputs = session.run_decoded(
            &model.inputs,
            &model.outputs["boxes"],
            model.outputs.get("counts").map(String::as_str),
            pixels,
            target_height,
            target_width,
            im_shape_values,
            scale_values,
        )?;
        let (width, count) =
            decoded_output_dimensions(&outputs.boxes_shape, outputs.boxes.len(), outputs.count)?;
        return Ok((outputs.boxes, width, count));
    }
    if contract == Some("ppyoloe_raw") {
        let outputs = session.run_raw(
            &model.inputs,
            &model.outputs["boxes"],
            &model.outputs["scores"],
            pixels,
            target_height,
            target_width,
            im_shape_values,
            scale_values,
        )?;
        if outputs.boxes_shape.len() != 3
            || outputs.logits_shape.len() != 3
            || outputs.boxes_shape[0] != 1
            || outputs.logits_shape[0] != 1
            || outputs.boxes_shape[2] != 4
            || outputs.logits_shape[1] != class_count as i64
            || outputs.boxes_shape[1] != outputs.logits_shape[2]
        {
            return Err(Error::Message(format!(
                "PPdoc PP-YOLOE outputs have unexpected shapes: boxes={:?}, scores={:?}",
                outputs.boxes_shape, outputs.logits_shape
            )));
        }
        let query_count = usize::try_from(outputs.boxes_shape[1]).unwrap_or(0);
        let decoded = decode_ppyoloe_raw(
            &outputs.boxes,
            &outputs.logits,
            query_count,
            class_count,
            model.detections_per_image,
            model_nms,
            threshold,
        )?;
        let count = decoded.len() / 7;
        return Ok((decoded, 7, count));
    }
    if contract != Some("rtdetr_raw") {
        return Err(Error::Message(
            "native OpenVINO requires a decoded-box, RT-DETR raw, or PP-YOLOE raw output contract"
                .to_owned(),
        ));
    }
    let outputs = session.run_raw(
        &model.inputs,
        &model.outputs["boxes"],
        &model.outputs["logits"],
        pixels,
        target_height,
        target_width,
        im_shape_values,
        scale_values,
    )?;
    if outputs.boxes_shape.len() != 3
        || outputs.logits_shape.len() != 3
        || outputs.boxes_shape[0] != 1
        || outputs.logits_shape[0] != 1
        || outputs.boxes_shape[2] != 4
        || outputs.boxes_shape[1] != outputs.logits_shape[1]
        || outputs.logits_shape[2] != class_count as i64
    {
        return Err(Error::Message(format!(
            "PPdoc OpenVINO outputs have unexpected shapes: boxes={:?}, logits={:?}",
            outputs.boxes_shape, outputs.logits_shape
        )));
    }
    let query_count = usize::try_from(outputs.boxes_shape[1]).unwrap_or(0);
    let image_height = (im_shape_values[0] / scale_values[0] + 0.5).floor();
    let image_width = (im_shape_values[1] / scale_values[1] + 0.5).floor();
    let decoded = decode_rtdetr_raw(
        &outputs.boxes,
        &outputs.logits,
        query_count,
        class_count,
        model.detections_per_image,
        image_width,
        image_height,
    )?;
    Ok((
        decoded,
        7,
        model.detections_per_image.min(query_count * class_count),
    ))
}

#[cfg(feature = "ppdoc")]
fn run_onnx(
    session: &mut Session,
    model: &ModelManifest,
    model_nms: &ModelNmsManifest,
    class_count: usize,
    threshold: f32,
    pixels: Vec<f32>,
    target_height: usize,
    target_width: usize,
    im_shape_values: [f32; 2],
    scale_values: [f32; 2],
) -> Result<(Vec<f32>, usize, usize)> {
    let image_tensor = Tensor::from_array((
        [1, 3, target_height, target_width],
        pixels.into_boxed_slice(),
    ))
    .map_err(ort_error)?;
    let im_shape = Tensor::from_array(([1, 2], im_shape_values.to_vec().into_boxed_slice()))
        .map_err(ort_error)?;
    let scale_factor = Tensor::from_array(([1, 2], scale_values.to_vec().into_boxed_slice()))
        .map_err(ort_error)?;
    let mut image_tensor = Some(image_tensor);
    let mut im_shape = Some(im_shape);
    let mut scale_factor = Some(scale_factor);
    let mut values: Vec<(String, SessionInputValue<'_>)> = Vec::new();
    for name in &model.inputs {
        let value = match input_kind(&model.inputs, name) {
            ModelInput::Image => image_tensor.take().expect("unique input").into(),
            ModelInput::Shape => im_shape.take().expect("unique input").into(),
            ModelInput::Scale => scale_factor.take().expect("unique input").into(),
        };
        values.push((name.clone(), value));
    }
    let outputs = session.run(values).map_err(ort_error)?;
    let boxes_name = &model.outputs["boxes"];
    let (shape, values) = outputs
        .get(boxes_name)
        .ok_or_else(|| Error::Message(format!("PPdoc output is missing {boxes_name}")))?
        .try_extract_tensor::<f32>()
        .map_err(ort_error)?;
    let contract = model.outputs.get("contract").map(String::as_str);
    if contract == Some("ppyoloe_raw") {
        let scores_name = &model.outputs["scores"];
        let (scores_shape, scores) = outputs
            .get(scores_name)
            .ok_or_else(|| Error::Message(format!("PPdoc output is missing {scores_name}")))?
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        if shape.len() != 3
            || scores_shape.len() != 3
            || shape[0] != 1
            || scores_shape[0] != 1
            || shape[2] != 4
            || scores_shape[1] != class_count as i64
            || shape[1] != scores_shape[2]
        {
            return Err(Error::Message(format!(
                "PPdoc PP-YOLOE outputs have unexpected shapes: boxes={shape:?}, scores={scores_shape:?}"
            )));
        }
        let query_count = usize::try_from(shape[1]).unwrap_or(0);
        let decoded = decode_ppyoloe_raw(
            values,
            scores,
            query_count,
            class_count,
            model.detections_per_image,
            model_nms,
            threshold,
        )?;
        let count = decoded.len() / 7;
        return Ok((decoded, 7, count));
    }
    if contract == Some("rtdetr_raw") {
        let logits_name = &model.outputs["logits"];
        let (logits_shape, logits) = outputs
            .get(logits_name)
            .ok_or_else(|| Error::Message(format!("PPdoc output is missing {logits_name}")))?
            .try_extract_tensor::<f32>()
            .map_err(ort_error)?;
        if shape.len() != 3
            || logits_shape.len() != 3
            || shape[0] != 1
            || logits_shape[0] != 1
            || shape[2] != 4
            || shape[1] != logits_shape[1]
            || logits_shape[2] != class_count as i64
        {
            return Err(Error::Message(format!(
                "PPdoc raw outputs have unexpected shapes: boxes={shape:?}, logits={logits_shape:?}"
            )));
        }
        let query_count = usize::try_from(shape[1]).unwrap_or(0);
        let image_height = (im_shape_values[0] / scale_values[0] + 0.5).floor();
        let image_width = (im_shape_values[1] / scale_values[1] + 0.5).floor();
        let decoded = decode_rtdetr_raw(
            values,
            logits,
            query_count,
            class_count,
            model.detections_per_image,
            image_width,
            image_height,
        )?;
        return Ok((
            decoded,
            7,
            model.detections_per_image.min(query_count * class_count),
        ));
    }
    let requested_count = if let Some(counts_name) = model.outputs.get("counts") {
        let output = outputs
            .get(counts_name)
            .ok_or_else(|| Error::Message(format!("PPdoc output is missing {counts_name}")))?;
        Some(
            if let Ok((_, counts)) = output.try_extract_tensor::<i32>() {
                counts.first().copied().unwrap_or(0).max(0) as usize
            } else {
                let (_, counts) = output.try_extract_tensor::<i64>().map_err(ort_error)?;
                usize::try_from(counts.first().copied().unwrap_or(0).max(0)).unwrap_or(0)
            },
        )
    } else {
        None
    };
    let (width, count) = decoded_output_dimensions(shape, values.len(), requested_count)?;
    Ok((values.to_vec(), width, count))
}

fn decoded_output_dimensions(
    shape: &[i64],
    value_count: usize,
    requested_count: Option<usize>,
) -> Result<(usize, usize)> {
    let width = usize::try_from(*shape.last().unwrap_or(&0)).unwrap_or(0);
    if width < 6 || value_count % width != 0 {
        return Err(Error::Message(format!(
            "PPdoc boxes output has an unexpected shape: {shape:?}"
        )));
    }
    let available = value_count / width;
    Ok((width, requested_count.unwrap_or(available).min(available)))
}

#[derive(Clone, Copy)]
struct PPYoloeCandidate {
    label: usize,
    query: usize,
    score: f32,
    bbox: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn decode_ppyoloe_raw(
    boxes: &[f32],
    scores: &[f32],
    query_count: usize,
    class_count: usize,
    keep_top_k: usize,
    nms: &ModelNmsManifest,
    requested_threshold: f32,
) -> Result<Vec<f32>> {
    if boxes.len() != query_count * 4 || scores.len() != class_count * query_count {
        return Err(Error::Message(
            "PPdoc PP-YOLOE tensor lengths do not match their shapes".to_owned(),
        ));
    }
    let score_threshold = nms.score_threshold.max(requested_threshold);
    let mut candidates_by_class = Vec::with_capacity(class_count);
    for label in 0..class_count {
        let mut candidates = (0..query_count)
            .filter_map(|query| {
                let score = scores[label * query_count + query];
                let offset = query * 4;
                let bbox: [f32; 4] = boxes[offset..offset + 4].try_into().expect("four values");
                (score.is_finite()
                    && score > score_threshold
                    && bbox.iter().all(|value| value.is_finite())
                    && bbox[2] > bbox[0]
                    && bbox[3] > bbox[1])
                    .then_some(PPYoloeCandidate {
                        label,
                        query,
                        score,
                        bbox,
                    })
            })
            .collect::<Vec<_>>();
        candidates.sort_unstable_by(|left, right| {
            right
                .score
                .total_cmp(&left.score)
                .then_with(|| left.query.cmp(&right.query))
        });
        candidates.truncate(nms.nms_top_k.min(candidates.len()));
        candidates_by_class.extend(candidates);
    }
    candidates_by_class.sort_unstable_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then_with(|| left.label.cmp(&right.label))
            .then_with(|| left.query.cmp(&right.query))
    });
    let mut selected_by_class = vec![Vec::new(); class_count];
    let mut kept = Vec::with_capacity(keep_top_k);
    for candidate in candidates_by_class {
        if selected_by_class[candidate.label]
            .iter()
            .all(|other: &PPYoloeCandidate| {
                bbox_iou(candidate.bbox, other.bbox) <= nms.nms_threshold
            })
        {
            selected_by_class[candidate.label].push(candidate);
            kept.push(candidate);
            if kept.len() == keep_top_k {
                break;
            }
        }
    }

    let mut decoded = Vec::with_capacity(kept.len() * 7);
    for detection in kept {
        decoded.extend_from_slice(&[
            detection.label as f32,
            detection.score,
            detection.bbox[0],
            detection.bbox[1],
            detection.bbox[2],
            detection.bbox[3],
            -1.0,
        ]);
    }
    Ok(decoded)
}

fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let intersection_width = (left[2].min(right[2]) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left[3].min(right[3]) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let left_area = (left[2] - left[0]).max(0.0) * (left[3] - left[1]).max(0.0);
    let right_area = (right[2] - right[0]).max(0.0) * (right[3] - right[1]).max(0.0);
    let union = left_area + right_area - intersection;
    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}

fn decode_rtdetr_raw(
    boxes: &[f32],
    logits: &[f32],
    query_count: usize,
    class_count: usize,
    top_k: usize,
    image_width: f32,
    image_height: f32,
) -> Result<Vec<f32>> {
    if boxes.len() != query_count * 4 || logits.len() != query_count * class_count {
        return Err(Error::Message(
            "PPdoc raw tensor lengths do not match their shapes".to_owned(),
        ));
    }
    let scores: Vec<f32> = logits
        .iter()
        .map(|logit| 1.0 / (1.0 + (-logit).exp()))
        .collect();
    let mut ranked: Vec<usize> = (0..scores.len()).collect();
    ranked.sort_unstable_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then_with(|| left.cmp(right))
    });
    ranked.truncate(top_k.min(ranked.len()));

    let mut decoded = Vec::with_capacity(ranked.len() * 7);
    for flat_index in ranked {
        let query = flat_index / class_count;
        let label = flat_index % class_count;
        let cx = boxes[query * 4] * image_width;
        let cy = boxes[query * 4 + 1] * image_height;
        let half_width = boxes[query * 4 + 2] * image_width * 0.5;
        let half_height = boxes[query * 4 + 3] * image_height * 0.5;
        decoded.extend_from_slice(&[
            label as f32,
            scores[flat_index],
            cx - half_width,
            cy - half_height,
            cx + half_width,
            cy + half_height,
            -1.0,
        ]);
    }
    Ok(decoded)
}

#[cfg(feature = "ppdoc")]
type PaddleCreate = unsafe extern "C" fn(*const c_char, *const c_char, i32, i32) -> *mut c_void;
#[cfg(feature = "ppdoc")]
type PaddleRun = unsafe extern "C" fn(
    *mut c_void,
    *const f32,
    i32,
    i32,
    *const f32,
    *const f32,
    *mut f32,
    usize,
    *mut usize,
    *mut i32,
) -> i32;
#[cfg(feature = "ppdoc")]
type PaddleLastError = unsafe extern "C" fn() -> *const c_char;
#[cfg(feature = "ppdoc")]
type PaddleDestroy = unsafe extern "C" fn(*mut c_void);

#[cfg(feature = "ppdoc")]
struct PaddleFunctions {
    run: PaddleRun,
    last_error: PaddleLastError,
    destroy: PaddleDestroy,
}

#[cfg(feature = "ppdoc")]
struct PaddleSession {
    handle: *mut c_void,
    functions: PaddleFunctions,
    _library: Library,
}

#[cfg(feature = "ppdoc")]
impl PaddleSession {
    fn new(
        runtime: &Path,
        model: &Path,
        params: &Path,
        threads: usize,
        onednn: bool,
    ) -> Result<Self> {
        // SAFETY: The library is retained for the lifetime of all copied function pointers.
        let library = unsafe { load_library(runtime) }.map_err(|error| {
            Error::Message(format!(
                "could not load Paddle runtime library {}: {error}",
                runtime.display()
            ))
        })?;
        // Windows Paddle accepts narrow paths. Reject interior NULs rather than truncating them.
        let model = std::ffi::CString::new(model.to_string_lossy().as_bytes())
            .map_err(|_| Error::Message("Paddle model path contains a NUL byte".to_owned()))?;
        let params = std::ffi::CString::new(params.to_string_lossy().as_bytes())
            .map_err(|_| Error::Message("Paddle parameter path contains a NUL byte".to_owned()))?;
        // SAFETY: Signatures match the versioned C ABI in ppdoc_paddle.h.
        unsafe {
            let abi: unsafe extern "C" fn() -> i32 =
                paddle_symbol(&library, b"ppdoc_paddle_abi_version\0", runtime)?;
            if abi() != 1 {
                return Err(Error::Message("unsupported PPdoc Paddle ABI".to_owned()));
            }
            let create: PaddleCreate = paddle_symbol(&library, b"ppdoc_paddle_create\0", runtime)?;
            let last_error: PaddleLastError =
                paddle_symbol(&library, b"ppdoc_paddle_last_error\0", runtime)?;
            let handle = create(
                model.as_ptr(),
                params.as_ptr(),
                i32::try_from(threads).unwrap_or(i32::MAX),
                i32::from(onednn),
            );
            if handle.is_null() {
                return Err(paddle_error(
                    last_error,
                    "could not create Paddle predictor",
                ));
            }
            Ok(Self {
                handle,
                functions: PaddleFunctions {
                    run: paddle_symbol(&library, b"ppdoc_paddle_run\0", runtime)?,
                    last_error,
                    destroy: paddle_symbol(&library, b"ppdoc_paddle_destroy\0", runtime)?,
                },
                _library: library,
            })
        }
    }

    fn run(
        &mut self,
        pixels: &[f32],
        target_height: usize,
        target_width: usize,
        im_shape: &[f32; 2],
        scale_factor: &[f32; 2],
        detections_per_image: usize,
        output_width: usize,
    ) -> Result<(Vec<f32>, usize, usize)> {
        let mut boxes = vec![0.0; detections_per_image * output_width];
        let mut length = 0;
        let mut count = 0;
        // SAFETY: all buffers remain live and correctly sized for this synchronous call.
        let status = unsafe {
            (self.functions.run)(
                self.handle,
                pixels.as_ptr(),
                i32::try_from(target_height)
                    .map_err(|_| Error::Message("PPdoc target height is too large".to_owned()))?,
                i32::try_from(target_width)
                    .map_err(|_| Error::Message("PPdoc target width is too large".to_owned()))?,
                im_shape.as_ptr(),
                scale_factor.as_ptr(),
                boxes.as_mut_ptr(),
                boxes.len(),
                &mut length,
                &mut count,
            )
        };
        if status != 0 {
            return Err(paddle_error(
                self.functions.last_error,
                "Paddle inference failed",
            ));
        }
        if length > boxes.len() || length % output_width != 0 {
            return Err(Error::Message(format!(
                "Paddle returned an invalid decoded-box length: {length}"
            )));
        }
        boxes.truncate(length);
        Ok((
            boxes,
            output_width,
            usize::try_from(count.max(0))
                .unwrap_or(0)
                .min(length / output_width),
        ))
    }
}

#[cfg(feature = "ppdoc")]
impl Drop for PaddleSession {
    fn drop(&mut self) {
        // SAFETY: handle was created by this live library and is destroyed exactly once.
        unsafe { (self.functions.destroy)(self.handle) };
    }
}

#[cfg(feature = "ppdoc")]
unsafe fn paddle_symbol<T: Copy>(library: &Library, name: &[u8], path: &Path) -> Result<T> {
    // SAFETY: The caller supplies the C declaration matching this exported symbol.
    unsafe { library.get::<T>(name) }
        .map(|symbol| *symbol)
        .map_err(|error| {
            Error::Message(format!(
                "Paddle runtime library {} is missing {}: {error}",
                path.display(),
                String::from_utf8_lossy(name).trim_end_matches('\0')
            ))
        })
}

#[cfg(feature = "ppdoc")]
fn paddle_error(last_error: PaddleLastError, fallback: &str) -> Error {
    // SAFETY: the ABI returns a thread-local NUL-terminated string valid until the next call.
    let message = unsafe {
        let pointer = last_error();
        (!pointer.is_null())
            .then(|| CStr::from_ptr(pointer).to_string_lossy().into_owned())
            .filter(|value| !value.is_empty())
    };
    Error::Message(format!(
        "PPdoc Paddle inference failed: {}",
        message.as_deref().unwrap_or(fallback)
    ))
}

#[cfg(all(feature = "ppdoc", windows))]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    use libloading::os::windows::{Library as WindowsLibrary, LOAD_WITH_ALTERED_SEARCH_PATH};
    // SAFETY: Forwarded to the caller; altered search also resolves sibling Paddle DLLs.
    unsafe { WindowsLibrary::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH) }.map(Into::into)
}

#[cfg(all(feature = "ppdoc", not(windows)))]
unsafe fn load_library(path: &Path) -> std::result::Result<Library, libloading::Error> {
    // SAFETY: Forwarded to the caller.
    unsafe { Library::new(path) }
}

const INTER_RESIZE_COEF_SCALE: f32 = 2048.0;

#[derive(Clone, Copy)]
pub(crate) enum ModelInput {
    Image,
    Shape,
    Scale,
}

pub(crate) fn input_kind(inputs: &[String], name: &str) -> ModelInput {
    if inputs.len() == 1 {
        ModelInput::Image
    } else {
        match name {
            "image" => ModelInput::Image,
            "im_shape" => ModelInput::Shape,
            "scale_factor" => ModelInput::Scale,
            _ => unreachable!("model inputs validated when the pack was loaded"),
        }
    }
}

#[derive(Clone, Copy)]
struct CubicSample {
    source: [u32; 4],
    coefficients: [i16; 4],
}

fn resize_opencv_cubic_nchw(
    image: &RgbImage,
    width: u32,
    height: u32,
    normalization_scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let x_samples = cubic_samples(image.width(), width);
    let y_samples = cubic_samples(image.height(), height);
    let source = image.as_raw();
    let source_width = image.width() as usize;
    let width = width as usize;
    let height = height as usize;
    let plane = width * height;
    let mut output = vec![0.0_f32; 3 * plane];
    let coefficient_scale = 1.0 / (INTER_RESIZE_COEF_SCALE * INTER_RESIZE_COEF_SCALE);
    for (y, y_sample) in y_samples.iter().copied().enumerate() {
        for (x, x_sample) in x_samples.iter().copied().enumerate() {
            let destination = y * width + x;
            for channel in 0..3 {
                let mut horizontal = [0_i32; 4];
                for (row, source_y) in y_sample.source.iter().enumerate() {
                    horizontal[row] = x_sample
                        .source
                        .iter()
                        .zip(x_sample.coefficients)
                        .map(|(source_x, coefficient)| {
                            let offset = ((*source_y as usize * source_width + *source_x as usize)
                                * 3)
                                + channel;
                            i32::from(source[offset]) * i32::from(coefficient)
                        })
                        .sum();
                }
                let weighted = (horizontal[0] as f32).mul_add(
                    f32::from(y_sample.coefficients[0]) * coefficient_scale,
                    (horizontal[1] as f32).mul_add(
                        f32::from(y_sample.coefficients[1]) * coefficient_scale,
                        (horizontal[2] as f32).mul_add(
                            f32::from(y_sample.coefficients[2]) * coefficient_scale,
                            horizontal[3] as f32
                                * f32::from(y_sample.coefficients[3])
                                * coefficient_scale,
                        ),
                    ),
                );
                let value = weighted.round_ties_even().clamp(0.0, 255.0);
                output[channel * plane + destination] =
                    (value * normalization_scale - mean[channel]) / std[channel];
            }
        }
    }
    output
}

fn resize_bilinear_nchw(
    image: &RgbImage,
    width: u32,
    height: u32,
    normalization_scale: f32,
    mean: [f32; 3],
    std: [f32; 3],
) -> Vec<f32> {
    let resized = imageops::resize(image, width, height, imageops::FilterType::Triangle);
    let plane = width as usize * height as usize;
    let mut output = vec![0.0_f32; 3 * plane];
    for (offset, pixel) in resized.pixels().enumerate() {
        for channel in 0..3 {
            output[channel * plane + offset] =
                (f32::from(pixel[channel]) * normalization_scale - mean[channel]) / std[channel];
        }
    }
    output
}

fn cubic_samples(source_size: u32, target_size: u32) -> Vec<CubicSample> {
    let scale = f64::from(source_size) / f64::from(target_size);
    (0..target_size)
        .map(|destination| {
            let mut fraction = ((f64::from(destination) + 0.5) * scale - 0.5) as f32;
            let base = fraction.floor() as i32;
            fraction -= base as f32;
            let coefficients = cubic_coefficients(fraction).map(|value| {
                (value * INTER_RESIZE_COEF_SCALE)
                    .round_ties_even()
                    .clamp(f32::from(i16::MIN), f32::from(i16::MAX)) as i16
            });
            let maximum = source_size.saturating_sub(1) as i32;
            CubicSample {
                source: std::array::from_fn(|index| {
                    (base - 1 + index as i32).clamp(0, maximum) as u32
                }),
                coefficients,
            }
        })
        .collect()
}

fn cubic_coefficients(x: f32) -> [f32; 4] {
    const A: f32 = -0.75;
    let x1 = x + 1.0;
    let inverse = 1.0 - x;
    let first = ((A * x1 - 5.0 * A) * x1 + 8.0 * A) * x1 - 4.0 * A;
    let second = ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0;
    let third = ((A + 2.0) * inverse - (A + 3.0)) * inverse * inverse + 1.0;
    [first, second, third, 1.0 - first - second - third]
}

fn postprocess(
    values: &[f32],
    row_width: usize,
    count: usize,
    labels: &[String],
    image_width: u32,
    image_height: u32,
    threshold: f32,
) -> Vec<PPDocDetection> {
    let mut detections: Vec<(f32, PPDocDetection)> = values
        .chunks_exact(row_width)
        .take(count)
        .enumerate()
        .filter_map(|(index, row)| {
            let label_id = row[0] as isize;
            if label_id < 0 || row[1] <= threshold || label_id as usize >= labels.len() {
                return None;
            }
            let bbox = [
                row[2].round_ties_even().clamp(0.0, image_width as f32),
                row[3].round_ties_even().clamp(0.0, image_height as f32),
                row[4].round_ties_even().clamp(0.0, image_width as f32),
                row[5].round_ties_even().clamp(0.0, image_height as f32),
            ];
            (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then(|| {
                (
                    if row_width >= 7 { row[6] } else { index as f32 },
                    PPDocDetection {
                        label_id: label_id as usize,
                        label: labels[label_id as usize].clone(),
                        score: row[1],
                        bbox,
                        order: None,
                    },
                )
            })
        })
        .collect();
    if row_width >= 7 {
        detections.sort_by(|left, right| left.0.total_cmp(&right.0));
    }
    if detections.len() > 1 {
        let area_threshold = if image_width > image_height {
            0.82
        } else {
            0.93
        };
        let filtered: Vec<_> = detections
            .iter()
            .filter(|(_, detection)| {
                detection.label != "image"
                    || (detection.bbox[2] - detection.bbox[0])
                        * (detection.bbox[3] - detection.bbox[1])
                        <= area_threshold * image_width as f32 * image_height as f32
            })
            .cloned()
            .collect();
        if !filtered.is_empty() {
            detections = filtered;
        }
    }
    let mut next_order = 1;
    detections
        .into_iter()
        .map(|(_, mut detection)| {
            if !skips_reading_order(&detection.label) {
                detection.order = Some(next_order);
                next_order += 1;
            }
            detection
        })
        .collect()
}

fn skips_reading_order(label: &str) -> bool {
    matches!(
        label,
        "figure_title"
            | "vision_footnote"
            | "image"
            | "chart"
            | "table"
            | "header"
            | "header_image"
            | "footer"
            | "footer_image"
            | "footnote"
            | "aside_text"
    )
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

fn pack_file(root: &Path, relative: &str, label: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    if relative.is_absolute() {
        return Err(Error::Message(format!(
            "PPdoc {label} path must be relative to its pack"
        )));
    }
    let path = fs::canonicalize(root.join(relative))
        .map_err(|source| Error::io(root.join(relative), source))?;
    if !path.starts_with(root) {
        return Err(Error::Message(format!(
            "PPdoc {label} path escapes its pack"
        )));
    }
    Ok(path)
}

fn canonical_file(path: &Path, label: &str) -> Result<PathBuf> {
    if !path.is_file() {
        return Err(Error::Message(format!(
            "{label} does not exist: {}",
            path.display()
        )));
    }
    fs::canonicalize(path).map_err(|source| Error::io(path, source))
}

fn sha256_file(path: &Path) -> Result<String> {
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

#[cfg(feature = "ppdoc")]
fn ort_error(error: ort::Error) -> Error {
    Error::Message(format!("PPdoc ONNX inference failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_boxes_are_clipped_filtered_and_keep_sequence_order() {
        let labels = vec!["text".to_owned(), "footnote".to_owned()];
        let rows = [
            0.0, 0.9, -2.0, 4.2, 110.0, 60.0, 1.0, 1.0, 0.05, 1.0, 2.0, 3.0, 4.0, 2.0, 8.0, 0.9,
            1.0, 2.0, 3.0, 4.0, 3.0,
        ];
        let detections = postprocess(&rows, 7, 3, &labels, 100, 50, 0.10);
        assert_eq!(detections.len(), 1);
        assert_eq!(detections[0].label, "text");
        assert_eq!(detections[0].bbox, [0.0, 4.0, 100.0, 50.0]);
        assert_eq!(detections[0].order, Some(1));
    }

    #[test]
    fn decoded_output_dimensions_accept_paddledetection_rows_and_clamp_count() {
        assert_eq!(
            decoded_output_dimensions(&[1, 300, 6], 1_800, Some(287)).unwrap(),
            (6, 287)
        );
        assert_eq!(
            decoded_output_dimensions(&[300, 6], 1_800, Some(999)).unwrap(),
            (6, 300)
        );
        assert!(decoded_output_dimensions(&[300, 5], 1_500, None).is_err());
    }

    #[test]
    fn ppyoloe_raw_decode_uses_class_major_scores_and_classwise_nms() {
        let boxes = [
            0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 9.0, 9.0, 20.0, 20.0, 30.0, 30.0, 0.0, 0.0, 10.0, 10.0,
        ];
        let scores = [0.90, 0.80, 0.70, 0.01, 0.02, 0.03, 0.04, 0.95];
        let nms = ModelNmsManifest {
            score_threshold: 0.01,
            nms_threshold: 0.5,
            nms_top_k: 1_000,
        };
        let decoded = decode_ppyoloe_raw(&boxes, &scores, 4, 2, 3, &nms, 0.10).unwrap();
        assert_eq!(decoded.len(), 21);
        assert_eq!(&decoded[0..7], &[1.0, 0.95, 0.0, 0.0, 10.0, 10.0, -1.0]);
        assert_eq!(&decoded[7..14], &[0.0, 0.90, 0.0, 0.0, 10.0, 10.0, -1.0]);
        assert_eq!(&decoded[14..21], &[0.0, 0.70, 20.0, 20.0, 30.0, 30.0, -1.0]);
    }

    #[test]
    fn cubic_resize_matches_opencv_u8_contract() {
        let image = RgbImage::from_raw(
            3,
            2,
            vec![
                0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170,
            ],
        )
        .unwrap();
        let expected = vec![
            0, 0, 8, 0, 10, 20, 21, 31, 41, 41, 51, 61, 53, 63, 73, 18, 28, 38, 30, 40, 50, 50, 60,
            70, 71, 81, 91, 83, 93, 103, 67, 77, 87, 79, 89, 99, 100, 110, 120, 120, 130, 140, 132,
            142, 152, 97, 107, 117, 109, 119, 129, 129, 139, 149, 150, 160, 170, 162, 172, 182,
        ];
        let resized = resize_opencv_cubic_nchw(&image, 5, 4, 1.0, [0.0; 3], [1.0; 3]);
        let interleaved: Vec<u8> = (0..20)
            .flat_map(|offset| {
                [
                    resized[offset] as u8,
                    resized[20 + offset] as u8,
                    resized[40 + offset] as u8,
                ]
            })
            .collect();
        assert_eq!(interleaved, expected);
        let resized = resize_opencv_cubic_nchw(&image, 20, 4, 1.0, [0.0; 3], [1.0; 3]);
        let interleaved: Vec<u8> = (0..80)
            .flat_map(|offset| {
                [
                    resized[offset] as u8,
                    resized[80 + offset] as u8,
                    resized[160 + offset] as u8,
                ]
            })
            .collect();
        assert_eq!(
            format!("{:x}", Sha256::digest(interleaved)),
            "310895b2ad24f63cc379b7c9e6987599174685d841d0acc9d6101c433aa37006"
        );
    }

    #[test]
    fn bilinear_resize_is_planar_and_normalized() {
        let image = RgbImage::from_raw(1, 1, vec![10, 20, 30]).unwrap();
        let resized = resize_bilinear_nchw(&image, 2, 2, 0.5, [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]);
        assert_eq!(resized.len(), 12);
        assert_eq!(&resized[0..4], &[2.0; 4]);
        assert_eq!(&resized[4..8], &[8.0 / 3.0; 4]);
        assert_eq!(&resized[8..12], &[3.0; 4]);
    }
}
