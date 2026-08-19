#[cfg(feature = "fast-allocator")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legalpdf::{
    add_pdf_geometry, apply_docx_links, apply_external_layout, default_cache_dir,
    extract_common_input, improve_document, load_artifacts, lookup_artifact_footnote, page_count,
    parse_pdf, plan_docx_links, repair_identity, replay_common_input, replay_contract,
    write_artifacts, DocxPlanOptions, Error, OcrOptions, OcrProvider, ParseMode, ParseOptions,
    Result, TesseractOptions,
};
#[cfg(feature = "ocr")]
use legalpdf::{extract_layout_input, render_pdf_pages};
#[cfg(feature = "kraken")]
use legalpdf::{KrakenBackend, KrakenLayout, KrakenOcr, KrakenOptions, KrakenTier};
#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
use legalpdf::{PPDocBackend, PPDocLayout, PPDocOptions};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};

fn usage() -> &'static str {
    "usage:\n  legalpdf parse <pdf> --output <dir> [--mode local|codex] [--cache] [--cache-dir <dir>] [--model <name>] [--effort <level>] [--compact-pages] [--ocr-pages <1,2,...>] [OCR OPTIONS] [PPDOC OPTIONS]\n  legalpdf inspect <pdf>\n  legalpdf layout-input <pdf> --output <json> --images <dir> [--image-dpi <72..300>] [OCR OPTIONS]\n  legalpdf apply-layout <layout-input> --assignments <json> --output <dir> [--compact-pages]\n  legalpdf add-geometry <pdf> --document <compact-document> --output <dir> [OCR OPTIONS] [PPDOC OPTIONS]\n  legalpdf page-count <pdf>\n  legalpdf contract <request.json>\n  legalpdf ocr-identity --provider tesseract|kraken-lite [OCR OPTIONS]\n  legalpdf ppdoc-images <image>... --model-pack <dir> --runtime <dll|so|dylib> [--backend cpu|cuda|tensorrt|directml|openvino|onednn] [--device <id|OpenVINO-device>] [--cpu-fallback] [--threads <n>] [--cache-dir <dir>] [--threshold <0..1>] [--onednn]\n  legalpdf repair-identity\n  legalpdf improve <pdf> --document <artifact> --output <dir> --model <name> --effort <level> [--cache-dir <dir>] [--timeout-seconds <n>]\n  legalpdf footnote <document> <label-or-pair-id> [--page <n>] [--occurrence <n>] [--proposition sentence|passage_since_prior_note]\n  legalpdf docx-link-plan <docx> --output <json> [--strategy auto|direct|hybrid] [--model <name>] [--effort <level>] [--cache-dir <dir>] [--timeout-seconds <n>]\n  legalpdf docx-apply-links <docx> --plan <json> --links <json> --output <docx>\n\nOCR OPTIONS:\n  --ocr-provider tesseract [--tesseract-command <path>] [--ocr-language <code>] [--ocr-dpi <72..600>] [--ocr-psm <0..13>] [--ocr-timeout <1..3600>]\n  --ocr-provider kraken-lite --kraken-layout tesseract --kraken-model <onnx> [--kraken-codec <json>] --onnx-runtime <dll|so|dylib> --kraken-tesseract-library <dll|so|dylib> [KRAKEN COMMON OPTIONS]\n  --ocr-provider kraken-lite --kraken-layout blla --kraken-runtime-wheel <whl> --kraken-blla-pack <dir> --kraken-recognizer-pack <dir> [--kraken-python <path>] [--kraken-timeout <1..86400>] [KRAKEN COMMON OPTIONS]\n  KRAKEN COMMON OPTIONS: [--kraken-backend cpu|cuda|tensorrt|directml|openvino|onednn] [--kraken-device <id|OpenVINO-device>] [--kraken-cpu-fallback] [--kraken-low-memory] [--kraken-tier quality|balanced|turbo|extreme] [--kraken-workers <n>] [--kraken-threads <n>] [--kraken-layout-workers <n>] [--kraken-batch-size <n>] [--kraken-width-bucket <n>] [--kraken-width-scale <0.5..1.25>]\n  Both providers accept --expected-ocr-identity <identity>.\n\nPPDOC OPTIONS:\n  --ppdoc-model-pack <dir> --ppdoc-runtime <dll|so|dylib> [--ppdoc-backend cpu|cuda|tensorrt|directml|openvino|onednn] [--ppdoc-device <id|OpenVINO-device>] [--ppdoc-cpu-fallback] [--ppdoc-threads <n>] [--ppdoc-cache-dir <dir>] [--ppdoc-threshold <0..1>] [--ppdoc-dpi <72..600>] [--ppdoc-expected-identity <identity>]"
}

#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
fn parse_ppdoc_option(
    arguments: &[String],
    index: &mut usize,
    option: &str,
    prefixed: bool,
    configured: &mut Option<PPDocOptions>,
) -> Result<bool> {
    let name = if prefixed {
        option.strip_prefix("--ppdoc-")
    } else {
        option.strip_prefix("--")
    };
    let Some(name) = name else {
        return Ok(false);
    };
    if !matches!(
        name,
        "model-pack"
            | "runtime"
            | "onnx-runtime"
            | "onednn"
            | "backend"
            | "device"
            | "cache-dir"
            | "cpu-fallback"
            | "threads"
            | "threshold"
            | "dpi"
            | "expected-identity"
    ) {
        return Ok(false);
    }
    let options = configured.get_or_insert_with(PPDocOptions::default);
    match name {
        "model-pack" => {
            options.model_pack = Some(PathBuf::from(take_value(arguments, index, option)?))
        }
        "runtime" | "onnx-runtime" => {
            options.runtime = Some(PathBuf::from(take_value(arguments, index, option)?))
        }
        "onednn" => options.onednn = true,
        "backend" => {
            let value = take_value(arguments, index, option)?;
            options.backend = PPDocBackend::parse(&value)
                .ok_or_else(|| Error::Message(format!("unsupported PPdoc backend: {value}")))?;
        }
        "device" => options.device = Some(take_value(arguments, index, option)?),
        "cache-dir" => {
            options.cache_dir = Some(PathBuf::from(take_value(arguments, index, option)?))
        }
        "cpu-fallback" => options.cpu_fallback = true,
        "threads" => {
            options.threads = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message(format!("{option} must be an integer")))?
        }
        "threshold" => {
            options.threshold = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message(format!("{option} must be a number")))?
        }
        "dpi" => {
            options.render_dpi = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message(format!("{option} must be an integer")))?
        }
        "expected-identity" => {
            options.expected_identity = Some(take_value(arguments, index, option)?)
        }
        _ => unreachable!("PPdoc option names were validated"),
    }
    Ok(true)
}

#[derive(Default)]
struct OcrCli {
    provider: Option<String>,
    tesseract: TesseractOptions,
    #[cfg(feature = "kraken")]
    kraken: KrakenOptions,
    touched: bool,
    tesseract_only: bool,
    kraken_only: bool,
    expected_identity: Option<String>,
}

impl OcrCli {
    fn finish(mut self) -> Result<Option<OcrOptions>> {
        let Some(provider) = self.provider else {
            return if self.touched {
                Err(Error::Message(
                    "OCR options require --ocr-provider".to_owned(),
                ))
            } else {
                Ok(None)
            };
        };
        match provider.as_str() {
            "tesseract" if !self.kraken_only => {
                self.tesseract.expected_identity = self.expected_identity;
                Ok(Some(self.tesseract.into()))
            }
            "tesseract" => Err(Error::Message(
                "Kraken options cannot be used with Tesseract".to_owned(),
            )),
            "kraken" | "kraken-lite" if !self.tesseract_only => {
                #[cfg(feature = "kraken")]
                {
                    self.kraken.expected_identity = self.expected_identity;
                    Ok(Some(self.kraken.into()))
                }
                #[cfg(not(feature = "kraken"))]
                {
                    Err(Error::Message(
                        "Kraken-lite requires a legalpdf binary built with --features kraken"
                            .to_owned(),
                    ))
                }
            }
            "kraken" | "kraken-lite" => Err(Error::Message(
                "Tesseract options cannot be used with Kraken-lite".to_owned(),
            )),
            _ => Err(Error::Message(format!(
                "unsupported OCR provider: {provider}"
            ))),
        }
    }
}

fn parse_ocr_option(
    arguments: &[String],
    index: &mut usize,
    option: &str,
    options: &mut OcrCli,
) -> Result<bool> {
    match option {
        "--ocr-provider" => {
            options.provider = Some(take_value(arguments, index, option)?);
            options.touched = true;
        }
        "--tesseract-command" => {
            options.tesseract.command = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.tesseract_only = true;
        }
        "--ocr-language" => {
            options.tesseract.language = take_value(arguments, index, option)?;
            options.tesseract_only = true;
        }
        "--ocr-dpi" => {
            let dpi = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--ocr-dpi must be an integer".to_owned()))?;
            options.tesseract.dpi = dpi;
            #[cfg(feature = "kraken")]
            {
                options.kraken.dpi = dpi;
            }
        }
        "--ocr-psm" => {
            options.tesseract.psm = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--ocr-psm must be an integer".to_owned()))?;
            options.tesseract_only = true;
        }
        "--ocr-timeout" => {
            options.tesseract.timeout_seconds = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--ocr-timeout must be an integer".to_owned()))?;
            options.tesseract_only = true;
        }
        "--expected-ocr-identity" => {
            options.expected_identity = Some(take_value(arguments, index, option)?);
        }
        #[cfg(feature = "kraken")]
        "--kraken-model" => {
            options.kraken.model = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-codec" => {
            options.kraken.codec = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--onnx-runtime" => {
            options.kraken.runtime = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-runtime-wheel" => {
            options.kraken.runtime_wheel =
                Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-python" => {
            options.kraken.python = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-blla-pack" => {
            options.kraken.blla_pack = Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-recognizer-pack" => {
            options.kraken.recognizer_pack =
                Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-tesseract-library" => {
            options.kraken.tesseract_library =
                Some(PathBuf::from(take_value(arguments, index, option)?));
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-layout" => {
            let value = take_value(arguments, index, option)?;
            options.kraken.layout = KrakenLayout::parse(&value).ok_or_else(|| {
                Error::Message("--kraken-layout must be tesseract or blla".to_owned())
            })?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-backend" => {
            let value = take_value(arguments, index, option)?;
            options.kraken.backend = KrakenBackend::parse(&value).ok_or_else(|| {
                Error::Message(
                    "--kraken-backend must be cpu, cuda, tensorrt, directml, openvino, or onednn"
                        .to_owned(),
                )
            })?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-device" => {
            options.kraken.device = Some(take_value(arguments, index, option)?);
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-cpu-fallback" => {
            options.kraken.cpu_fallback = true;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-low-memory" => {
            options.kraken.cpu_arena = false;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-tier" => {
            let value = take_value(arguments, index, option)?;
            options.kraken.tier = KrakenTier::parse(&value).ok_or_else(|| {
                Error::Message(
                    "--kraken-tier must be quality, balanced, turbo, or extreme".to_owned(),
                )
            })?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-threads" => {
            options.kraken.threads = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--kraken-threads must be an integer".to_owned()))?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-workers" => {
            options.kraken.workers = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--kraken-workers must be an integer".to_owned()))?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-layout-workers" => {
            options.kraken.layout_workers =
                take_value(arguments, index, option)?.parse().map_err(|_| {
                    Error::Message("--kraken-layout-workers must be an integer".to_owned())
                })?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-batch-size" => {
            options.kraken.batch_size = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--kraken-batch-size must be an integer".to_owned()))?;
            options.kraken.runtime_batch_size = options.kraken.batch_size;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-timeout" => {
            options.kraken.timeout_seconds = take_value(arguments, index, option)?
                .parse()
                .map_err(|_| Error::Message("--kraken-timeout must be an integer".to_owned()))?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-width-bucket" => {
            options.kraken.width_bucket =
                take_value(arguments, index, option)?.parse().map_err(|_| {
                    Error::Message("--kraken-width-bucket must be an integer".to_owned())
                })?;
            options.kraken_only = true;
        }
        #[cfg(feature = "kraken")]
        "--kraken-width-scale" => {
            options.kraken.width_scale =
                Some(take_value(arguments, index, option)?.parse().map_err(|_| {
                    Error::Message("--kraken-width-scale must be a number".to_owned())
                })?);
            options.kraken_only = true;
        }
        _ => return Ok(false),
    }
    options.touched = true;
    Ok(true)
}

fn take_value(arguments: &[String], index: &mut usize, option: &str) -> Result<String> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| Error::Message(format!("{option} requires a value")))
}

fn parse_ocr_pages(value: &str) -> Result<Vec<usize>> {
    let mut pages = Vec::new();
    for raw in value.split(',') {
        let page = raw.parse::<usize>().ok().filter(|page| *page > 0)
            .ok_or_else(|| Error::Message("--ocr-pages must be comma-separated positive integers".to_owned()))?;
        let index = page - 1;
        if !pages.contains(&index) {
            pages.push(index);
        }
    }
    if pages.is_empty() || pages.len() > 1_000 {
        return Err(Error::Message("--ocr-pages requires 1 to 1000 pages".to_owned()));
    }
    pages.sort_unstable();
    Ok(pages)
}

fn parse_command(arguments: &[String]) -> Result<i32> {
    let pdf = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut output = None;
    let mut options = ParseOptions {
        use_cache: false,
        ..ParseOptions::default()
    };
    let mut compact = false;
    let mut ocr = OcrCli::default();
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let mut ppdoc = None;
    let mut mode = "local".to_owned();
    let mut model = None;
    let mut effort = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?))
            }
            "--cache-dir" => {
                options.cache_dir = Some(PathBuf::from(take_value(
                    arguments,
                    &mut index,
                    "--cache-dir",
                )?));
                options.use_cache = true;
            }
            "--cache" => options.use_cache = true,
            "--no-cache" => options.use_cache = false,
            "--compact-pages" => compact = true,
            "--ocr-pages" => options.ocr_pages = Some(parse_ocr_pages(
                &take_value(arguments, &mut index, "--ocr-pages")?,
            )?),
            "--mode" => {
                mode = take_value(arguments, &mut index, "--mode")?;
                if !matches!(mode.as_str(), "local" | "codex") {
                    return Err(Error::Message(format!("unknown parsing mode: {mode}")));
                }
            }
            "--model" => model = Some(take_value(arguments, &mut index, "--model")?),
            "--effort" => effort = Some(take_value(arguments, &mut index, "--effort")?),
            option if option.starts_with('-') => {
                let handled = parse_ocr_option(arguments, &mut index, option, &mut ocr)?;
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                let handled =
                    handled || parse_ppdoc_option(arguments, &mut index, option, true, &mut ppdoc)?;
                if !handled {
                    return Err(Error::Message(format!("unknown parse option: {option}")));
                }
            }
            value => {
                return Err(Error::Message(format!(
                    "unexpected parse argument: {value}"
                )))
            }
        }
        index += 1;
    }
    let output =
        output.ok_or_else(|| Error::Message("parse requires --output <dir>".to_owned()))?;
    options.ocr = ocr.finish()?;
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    {
        options.ppdoc = ppdoc;
    }
    options.mode = if mode == "codex" {
        ParseMode::Codex
    } else {
        ParseMode::Local
    };
    options.model = model;
    options.effort = effort;
    let document = parse_pdf(&pdf, &options)?;
    let manifest = write_artifacts(&document, output, compact)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "document": manifest,
            "status": document.status,
            "pages": document.page_count,
            "footnotes": document.footnotes.len(),
            "cache_hit": document.provenance.get("cache_hit").and_then(serde_json::Value::as_bool).unwrap_or(false),
        }))?
    );
    Ok(0)
}

fn geometry_command(arguments: &[String]) -> Result<i32> {
    let pdf = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut document = None;
    let mut output = None;
    let mut ocr = OcrCli::default();
    #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
    let mut ppdoc = None;
    let mut index = 1;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--document" => {
                document = Some(PathBuf::from(take_value(arguments, &mut index, option)?))
            }
            "--output" => output = Some(PathBuf::from(take_value(arguments, &mut index, option)?)),
            _ if option.starts_with('-') => {
                let handled = parse_ocr_option(arguments, &mut index, option, &mut ocr)?;
                #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
                let handled =
                    handled || parse_ppdoc_option(arguments, &mut index, option, true, &mut ppdoc)?;
                if !handled {
                    return Err(Error::Message(format!(
                        "unknown add-geometry option: {option}"
                    )));
                }
            }
            _ => {
                return Err(Error::Message(format!(
                    "unexpected add-geometry argument: {option}"
                )))
            }
        }
        index += 1;
    }
    let document = document
        .ok_or_else(|| Error::Message("add-geometry requires --document <path>".to_owned()))?;
    let output =
        output.ok_or_else(|| Error::Message("add-geometry requires --output <dir>".to_owned()))?;
    let options = ParseOptions {
        use_cache: false,
        ocr: ocr.finish()?,
        #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
        ppdoc,
        ..ParseOptions::default()
    };
    let manifest = add_pdf_geometry(pdf, document, output, &options)?;
    println!("{}", serde_json::to_string(&json!({"geometry": manifest}))?);
    Ok(0)
}

#[cfg(feature = "ocr")]
fn layout_input_command(arguments: &[String]) -> Result<i32> {
    let pdf = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut output = None;
    let mut images = None;
    let mut image_dpi = 120_u16;
    let mut ocr = OcrCli::default();
    let mut index = 1;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--output" => output = Some(PathBuf::from(take_value(arguments, &mut index, option)?)),
            "--images" => images = Some(PathBuf::from(take_value(arguments, &mut index, option)?)),
            "--image-dpi" => {
                image_dpi = take_value(arguments, &mut index, option)?
                    .parse()
                    .map_err(|_| Error::Message("--image-dpi must be an integer".to_owned()))?
            }
            _ if option.starts_with('-') => {
                if !parse_ocr_option(arguments, &mut index, option, &mut ocr)? {
                    return Err(Error::Message(format!(
                        "unknown layout-input option: {option}"
                    )));
                }
            }
            _ => {
                return Err(Error::Message(format!(
                    "unexpected layout-input argument: {option}"
                )))
            }
        }
        index += 1;
    }
    let output =
        output.ok_or_else(|| Error::Message("layout-input requires --output <json>".to_owned()))?;
    let images =
        images.ok_or_else(|| Error::Message("layout-input requires --images <dir>".to_owned()))?;
    let ocr = ocr.finish()?;
    extract_layout_input(&pdf, &output, ocr.as_ref())?;
    let rendered = render_pdf_pages(&pdf, &images, image_dpi)?;
    println!(
        "{}",
        serde_json::to_string(&json!({"input": output, "images": rendered}))?
    );
    Ok(0)
}

#[cfg(not(feature = "ocr"))]
fn layout_input_command(_arguments: &[String]) -> Result<i32> {
    Err(Error::Message(
        "layout-input requires a legalpdf binary built with --features ocr".to_owned(),
    ))
}

fn apply_layout_command(arguments: &[String]) -> Result<i32> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut assignments = None;
    let mut output = None;
    let mut pdf = None;
    let mut model = None;
    let mut effort = None;
    let mut cache_dir = None;
    let mut timeout_seconds = 600_u64;
    let mut compact = false;
    let mut index = 1;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--assignments" => {
                assignments = Some(PathBuf::from(take_value(arguments, &mut index, option)?))
            }
            "--output" => output = Some(PathBuf::from(take_value(arguments, &mut index, option)?)),
            "--pdf" => pdf = Some(PathBuf::from(take_value(arguments, &mut index, option)?)),
            "--model" => model = Some(take_value(arguments, &mut index, option)?),
            "--effort" => effort = Some(take_value(arguments, &mut index, option)?),
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(take_value(arguments, &mut index, option)?))
            }
            "--timeout-seconds" => {
                timeout_seconds =
                    take_value(arguments, &mut index, option)?
                        .parse()
                        .map_err(|_| {
                            Error::Message("--timeout-seconds must be an integer".to_owned())
                        })?
            }
            "--compact-pages" => compact = true,
            _ => {
                return Err(Error::Message(format!(
                    "unknown apply-layout option: {option}"
                )))
            }
        }
        index += 1;
    }
    let assignments = assignments
        .ok_or_else(|| Error::Message("apply-layout requires --assignments <json>".to_owned()))?;
    let output =
        output.ok_or_else(|| Error::Message("apply-layout requires --output <dir>".to_owned()))?;
    let document = apply_external_layout(input, assignments)?;
    let repair_requested = pdf.is_some() || model.is_some() || effort.is_some();
    let document = if repair_requested {
        let pdf = pdf.ok_or_else(|| {
            Error::Message("apply-layout repair requires --pdf, --model, and --effort".to_owned())
        })?;
        let model = model.ok_or_else(|| {
            Error::Message("apply-layout repair requires --pdf, --model, and --effort".to_owned())
        })?;
        let effort = effort.ok_or_else(|| {
            Error::Message("apply-layout repair requires --pdf, --model, and --effort".to_owned())
        })?;
        improve_document(
            &document,
            &pdf,
            &model,
            &effort,
            &cache_dir.unwrap_or_else(|| default_cache_dir().join("codex")),
            timeout_seconds,
        )?
    } else {
        document
    };
    let manifest = write_artifacts(&document, output, compact)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "document": manifest,
            "status": document.status,
            "pages": document.page_count,
            "footnotes": document.footnotes.len(),
        }))?
    );
    Ok(0)
}

fn ocr_identity_command(arguments: &[String]) -> Result<i32> {
    let mut options = OcrCli::default();
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if option == "--provider" {
            options.provider = Some(take_value(arguments, &mut index, option)?);
            options.touched = true;
        } else {
            if option == "--ocr-provider"
                || !parse_ocr_option(arguments, &mut index, option, &mut options)?
            {
                return Err(Error::Message(format!(
                    "unknown ocr-identity option: {option}"
                )));
            }
        }
        index += 1;
    }
    let configured = options.finish()?.ok_or_else(|| {
        Error::Message("ocr-identity requires --provider tesseract|kraken-lite".to_owned())
    })?;
    let provider_name = match &configured {
        OcrOptions::Tesseract(_) => "tesseract",
        #[cfg(feature = "kraken")]
        OcrOptions::Kraken(_) => "kraken-lite",
    };
    let provider = OcrProvider::new(&configured)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "provider": provider_name,
            "identity": provider.identity(),
        }))?
    );
    Ok(0)
}

#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
fn ppdoc_identity_command(arguments: &[String]) -> Result<i32> {
    let mut configured = None;
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if !parse_ppdoc_option(arguments, &mut index, option, false, &mut configured)? {
            return Err(Error::Message(format!(
                "unknown ppdoc-identity option: {option}"
            )));
        }
        index += 1;
    }
    let options = configured.ok_or_else(|| {
        Error::Message("ppdoc-identity requires model-pack and runtime options".to_owned())
    })?;
    let provider = PPDocLayout::new(&options)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "provider": "local-layout",
            "variant": provider.variant_id(),
            "identity": provider.identity(),
        }))?
    );
    Ok(0)
}

#[cfg(feature = "kraken")]
fn kraken_images_command(arguments: &[String]) -> Result<i32> {
    let mut images = Vec::new();
    let mut options = OcrCli {
        provider: Some("kraken-lite".to_owned()),
        touched: true,
        ..OcrCli::default()
    };
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        if option == "--list" {
            let list = PathBuf::from(take_value(arguments, &mut index, option)?);
            let contents =
                std::fs::read_to_string(&list).map_err(|source| Error::io(&list, source))?;
            images.extend(
                contents
                    .lines()
                    .filter(|line| !line.trim().is_empty())
                    .map(|line| {
                        let mut path = PathBuf::from(line.trim());
                        path.set_extension("png");
                        path
                    }),
            );
        } else if option.starts_with('-') {
            if !parse_ocr_option(arguments, &mut index, option, &mut options)? {
                return Err(Error::Message(format!(
                    "unknown _kraken-images option: {option}"
                )));
            }
        } else {
            images.push(PathBuf::from(option));
        }
        index += 1;
    }
    if images.is_empty() {
        return Err(Error::Message(
            "_kraken-images requires image paths or --list <xml-list>".to_owned(),
        ));
    }
    let configured = options.finish()?.expect("preset provider");
    let OcrOptions::Kraken(configured) = configured else {
        return Err(Error::Message(
            "_kraken-images only supports Kraken-lite".to_owned(),
        ));
    };
    let mut provider = KrakenOcr::new(&configured)?;
    let ocr_identity = provider.identity().to_owned();
    let ocr_engine = provider.name().to_owned();
    let first = image::ImageReader::open(&images[0])
        .map_err(|source| Error::io(&images[0], source))?
        .decode()
        .map_err(|source| {
            Error::Message(format!(
                "could not decode {}: {source}",
                images[0].display()
            ))
        })?
        .into_luma8();
    provider.warmup_gray_image(&first)?;
    let mut completed = 0;
    for paths in images.chunks(32) {
        let decoded = paths
            .iter()
            .map(|path| {
                image::ImageReader::open(path)
                    .map_err(|source| Error::io(path, source))?
                    .decode()
                    .map_err(|source| {
                        Error::Message(format!("could not decode {}: {source}", path.display()))
                    })
                    .map(|image| image.into_luma8())
            })
            .collect::<Result<Vec<_>>>()?;
        let results = provider.recognize_gray_images_diagnostics(&decoded)?;
        for (path, result) in paths.iter().zip(results) {
            completed += 1;
            let seconds = result.layout_seconds + result.recognition_seconds;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "image": std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()),
                    "text": join_text_lines(result.lines.iter().map(|line| line.text.as_str())),
                    "lines": result.lines,
                    "layout_boxes": result.layout_boxes,
                    "layout_seconds": result.layout_seconds,
                    "recognition_seconds": result.recognition_seconds,
                    "seconds": seconds,
                    "ocr_identity": ocr_identity.as_str(),
                    "ocr_engine": ocr_engine.as_str(),
                    "progress": [completed, images.len()],
                }))?
            );
        }
    }
    Ok(0)
}

#[cfg(feature = "kraken")]
fn join_text_lines<'a>(lines: impl Iterator<Item = &'a str>) -> String {
    lines.collect::<Vec<_>>().join("\n")
}

#[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
fn ppdoc_images_command(arguments: &[String]) -> Result<i32> {
    let mut images = Vec::new();
    let mut configured = Some(PPDocOptions::default());
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--list" => {
                let list = PathBuf::from(take_value(arguments, &mut index, option)?);
                let contents =
                    std::fs::read_to_string(&list).map_err(|source| Error::io(&list, source))?;
                images.extend(
                    contents
                        .lines()
                        .filter(|line| !line.trim().is_empty())
                        .map(|line| PathBuf::from(line.trim())),
                );
            }
            value if value.starts_with('-') => {
                if !parse_ppdoc_option(arguments, &mut index, option, false, &mut configured)? {
                    return Err(Error::Message(format!(
                        "unknown ppdoc-images option: {value}"
                    )));
                }
            }
            value => images.push(PathBuf::from(value)),
        }
        index += 1;
    }
    if images.is_empty() {
        return Err(Error::Message(
            "ppdoc-images requires image paths or --list <path-list>".to_owned(),
        ));
    }
    let mut provider = PPDocLayout::new(configured.as_ref().expect("PPdoc defaults exist"))?;
    let identity = provider.identity().to_owned();
    let variant_id = provider.variant_id().to_owned();
    for (offset, path) in images.iter().enumerate() {
        let started = std::time::Instant::now();
        let detections = provider.detect_image(path)?;
        println!(
            "{}",
            serde_json::to_string(&json!({
                "image": std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()),
                "variant_id": variant_id,
                "identity": identity,
                "detections": detections,
                "seconds": started.elapsed().as_secs_f64(),
                "progress": [offset + 1, images.len()],
            }))?
        );
    }
    Ok(0)
}

fn footnote_command(arguments: &[String]) -> Result<i32> {
    let document = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let query = arguments
        .get(1)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut page = None;
    let mut occurrence = None;
    let mut proposition = "sentence".to_owned();
    let mut index = 2;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--page" => {
                page = Some(
                    take_value(arguments, &mut index, "--page")?
                        .parse()
                        .map_err(|_| Error::Message("--page must be an integer".to_owned()))?,
                );
            }
            "--occurrence" => {
                occurrence = Some(
                    take_value(arguments, &mut index, "--occurrence")?
                        .parse()
                        .map_err(|_| {
                            Error::Message("--occurrence must be an integer".to_owned())
                        })?,
                );
            }
            "--proposition" => proposition = take_value(arguments, &mut index, "--proposition")?,
            option => return Err(Error::Message(format!("unknown footnote option: {option}"))),
        }
        index += 1;
    }
    let result = lookup_artifact_footnote(document, query, page, occurrence, &proposition)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(if result.status == "found" { 0 } else { 2 })
}

fn parity_replay_command(arguments: &[String]) -> Result<i32> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message("_parity-replay requires an input".to_owned()))?;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?));
            }
            option => {
                return Err(Error::Message(format!(
                    "unknown _parity-replay option: {option}"
                )));
            }
        }
        index += 1;
    }
    let output = output
        .ok_or_else(|| Error::Message("_parity-replay requires --output <path>".to_owned()))?;
    let path = replay_common_input(input, output)?;
    println!("{}", serde_json::to_string(&json!({"result": path}))?);
    Ok(0)
}

fn parity_extract_command(arguments: &[String]) -> Result<i32> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message("_parity-extract requires an input".to_owned()))?;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?));
            }
            option => {
                return Err(Error::Message(format!(
                    "unknown _parity-extract option: {option}"
                )));
            }
        }
        index += 1;
    }
    let output = output
        .ok_or_else(|| Error::Message("_parity-extract requires --output <path>".to_owned()))?;
    let path = extract_common_input(input, output)?;
    println!("{}", serde_json::to_string(&json!({"result": path}))?);
    Ok(0)
}

fn contract_replay_command(arguments: &[String]) -> Result<i32> {
    let input = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message("contract requires an input".to_owned()))?;
    if arguments.len() != 1 {
        return Err(Error::Message(
            "contract accepts exactly one input".to_owned(),
        ));
    }
    let bytes = std::fs::read(&input).map_err(|source| Error::io(&input, source))?;
    let mut value: serde_json::Value = serde_json::from_slice(&bytes)?;
    if let Some(artifact) = value.get("artifact").and_then(serde_json::Value::as_str) {
        let artifact = std::path::Path::new(artifact);
        if artifact.is_relative() {
            value["artifact"] = serde_json::Value::String(
                input
                    .parent()
                    .unwrap_or_else(|| std::path::Path::new("."))
                    .join(artifact)
                    .to_string_lossy()
                    .into_owned(),
            );
        }
    }
    println!("{}", serde_json::to_string(&replay_contract(&value)?)?);
    Ok(0)
}

fn read_json(path: &Path) -> Result<Value> {
    let bytes = std::fs::read(path).map_err(|source| Error::io(path, source))?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn improve_command(arguments: &[String]) -> Result<i32> {
    let pdf = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut document = None;
    let mut output = None;
    let mut cache_dir = None;
    let mut model = None;
    let mut effort = None;
    let mut timeout_seconds = 600;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--document" => {
                document = Some(PathBuf::from(take_value(
                    arguments,
                    &mut index,
                    "--document",
                )?))
            }
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?))
            }
            "--cache-dir" => {
                cache_dir = Some(PathBuf::from(take_value(
                    arguments,
                    &mut index,
                    "--cache-dir",
                )?))
            }
            "--model" => model = Some(take_value(arguments, &mut index, "--model")?),
            "--effort" => effort = Some(take_value(arguments, &mut index, "--effort")?),
            "--timeout-seconds" => {
                timeout_seconds = take_value(arguments, &mut index, "--timeout-seconds")?
                    .parse()
                    .map_err(|_| {
                        Error::Message("--timeout-seconds must be an integer".to_owned())
                    })?
            }
            option => return Err(Error::Message(format!("unknown improve option: {option}"))),
        }
        index += 1;
    }
    let document_path =
        document.ok_or_else(|| Error::Message("improve requires --document <path>".to_owned()))?;
    let output =
        output.ok_or_else(|| Error::Message("improve requires --output <dir>".to_owned()))?;
    let model =
        model.ok_or_else(|| Error::Message("improve requires --model <name>".to_owned()))?;
    let effort =
        effort.ok_or_else(|| Error::Message("improve requires --effort <level>".to_owned()))?;
    let cache = cache_dir.unwrap_or_else(|| default_cache_dir().join("codex"));
    let source = load_artifacts(document_path)?;
    let document = improve_document(&source, &pdf, &model, &effort, &cache, timeout_seconds)?;
    let manifest = write_artifacts(&document, output, false)?;
    println!(
        "{}",
        serde_json::to_string(&json!({
            "document": manifest,
            "status": document.status,
            "repairs": document.repairs,
        }))?
    );
    Ok(0)
}

fn docx_plan_command(arguments: &[String]) -> Result<i32> {
    let docx = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut output = None;
    let mut options = DocxPlanOptions::default();
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?))
            }
            "--strategy" => options.strategy = take_value(arguments, &mut index, "--strategy")?,
            "--model" => options.model = take_value(arguments, &mut index, "--model")?,
            "--effort" => options.effort = take_value(arguments, &mut index, "--effort")?,
            "--cache-dir" => {
                options.cache_dir = Some(PathBuf::from(take_value(
                    arguments,
                    &mut index,
                    "--cache-dir",
                )?))
            }
            "--timeout-seconds" => {
                options.timeout_seconds = take_value(arguments, &mut index, "--timeout-seconds")?
                    .parse()
                    .map_err(|_| {
                        Error::Message("--timeout-seconds must be an integer".to_owned())
                    })?
            }
            option => {
                return Err(Error::Message(format!(
                    "unknown docx-link-plan option: {option}"
                )))
            }
        }
        index += 1;
    }
    if !matches!(options.strategy.as_str(), "auto" | "direct" | "hybrid") {
        return Err(Error::Message(format!(
            "unknown DOCX strategy: {}",
            options.strategy
        )));
    }
    let output = output
        .ok_or_else(|| Error::Message("docx-link-plan requires --output <path>".to_owned()))?;
    let plan = plan_docx_links(docx, &options)?;
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent).map_err(|source| Error::io(parent, source))?;
    }
    let mut bytes = serde_json::to_vec_pretty(&plan)?;
    bytes.push(b'\n');
    std::fs::write(&output, bytes).map_err(|source| Error::io(&output, source))?;
    println!(
        "{}",
        std::fs::canonicalize(&output).unwrap_or(output).display()
    );
    Ok(0)
}

fn docx_apply_command(arguments: &[String]) -> Result<i32> {
    let docx = arguments
        .first()
        .map(PathBuf::from)
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    let mut plan = None;
    let mut links = None;
    let mut output = None;
    let mut index = 1;
    while index < arguments.len() {
        match arguments[index].as_str() {
            "--plan" => plan = Some(PathBuf::from(take_value(arguments, &mut index, "--plan")?)),
            "--links" => links = Some(PathBuf::from(take_value(arguments, &mut index, "--links")?)),
            "--output" => {
                output = Some(PathBuf::from(take_value(
                    arguments, &mut index, "--output",
                )?))
            }
            option => {
                return Err(Error::Message(format!(
                    "unknown docx-apply-links option: {option}"
                )))
            }
        }
        index += 1;
    }
    let plan =
        read_json(&plan.ok_or_else(|| {
            Error::Message("docx-apply-links requires --plan <path>".to_owned())
        })?)?;
    let links =
        read_json(&links.ok_or_else(|| {
            Error::Message("docx-apply-links requires --links <path>".to_owned())
        })?)?;
    let links = links.get("links").unwrap_or(&links);
    if !links.is_object() {
        return Err(Error::Message(
            "links JSON must be an object or contain a links object".to_owned(),
        ));
    }
    let output = output
        .ok_or_else(|| Error::Message("docx-apply-links requires --output <path>".to_owned()))?;
    println!(
        "{}",
        serde_json::to_string(&apply_docx_links(docx, &plan, links, output)?)?
    );
    Ok(0)
}

fn run() -> Result<i32> {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = arguments
        .split_first()
        .ok_or_else(|| Error::Message(usage().to_owned()))?;
    match command.as_str() {
        "parse" => parse_command(rest),
        "layout-input" => layout_input_command(rest),
        "apply-layout" => apply_layout_command(rest),
        "add-geometry" => geometry_command(rest),
        "ocr-identity" => ocr_identity_command(rest),
        #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
        "ppdoc-identity" => ppdoc_identity_command(rest),
        "repair-identity" => {
            if !rest.is_empty() {
                return Err(Error::Message(
                    "repair-identity accepts no arguments".to_owned(),
                ));
            }
            println!("{}", serde_json::to_string(&repair_identity()?)?);
            Ok(0)
        }
        "improve" => improve_command(rest),
        "page-count" => {
            if rest.len() != 1 {
                return Err(Error::Message(usage().to_owned()));
            }
            let pdf = rest
                .first()
                .ok_or_else(|| Error::Message(usage().to_owned()))?;
            println!(
                "{}",
                serde_json::to_string(&json!({"pages": page_count(pdf)?}))?
            );
            Ok(0)
        }
        "inspect" => {
            if rest.len() != 1 {
                return Err(Error::Message(usage().to_owned()));
            }
            let result = pdf_inspector::detector::detect_pdf_type(
                rest.first().ok_or_else(|| Error::Message(usage().to_owned()))?,
            ).map_err(|error| Error::Message(format!("PDF inspection failed: {error}")))?;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "pages": result.page_count,
                    "pdf_type": format!("{:?}", result.pdf_type),
                    "confidence": result.confidence,
                    "pages_needing_ocr": result.pages_needing_ocr,
                }))?
            );
            Ok(0)
        }
        "footnote" => footnote_command(rest),
        "docx-link-plan" => docx_plan_command(rest),
        "docx-apply-links" => docx_apply_command(rest),
        "_parity-extract" => parity_extract_command(rest),
        "_parity-replay" => parity_replay_command(rest),
        "contract" => contract_replay_command(rest),
        #[cfg(feature = "kraken")]
        "_kraken-images" => kraken_images_command(rest),
        #[cfg(any(feature = "ppdoc", feature = "ppdoc-openvino"))]
        "ppdoc-images" => ppdoc_images_command(rest),
        #[cfg(not(any(feature = "ppdoc", feature = "ppdoc-openvino")))]
        "ppdoc-images" => Err(Error::Message(
            "ppdoc-images requires --features ppdoc-openvino or --features ppdoc".to_owned(),
        )),
        "--version" | "-V" => {
            println!("legalpdf {}", env!("CARGO_PKG_VERSION"));
            Ok(0)
        }
        "--help" | "-h" => {
            println!("{}", usage());
            Ok(0)
        }
        _ => Err(Error::Message(usage().to_owned())),
    }
}

fn main() {
    match run() {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("legalpdf: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(test, feature = "kraken"))]
mod tests {
    use super::{join_text_lines, parse_ocr_pages};

    #[test]
    fn kraken_page_text_preserves_line_delimiters() {
        assert_eq!(
            join_text_lines(["body", "1 note"].into_iter()),
            "body\n1 note"
        );
    }

    #[test]
    fn targeted_ocr_pages_are_bounded_deduplicated_indexes() {
        assert_eq!(parse_ocr_pages("5,1,5").unwrap(), vec![0, 4]);
        assert!(parse_ocr_pages("0").is_err());
        assert!(parse_ocr_pages("").is_err());
    }
}
