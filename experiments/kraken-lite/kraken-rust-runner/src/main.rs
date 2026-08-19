use kraken_rust_runner::error::{Error, Result};
use kraken_rust_runner::kraken::{
    KrakenBackend, KrakenLayout, KrakenOcr, KrakenOptions, KrakenTier,
};
use serde_json::json;
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

const IMAGE_CHUNK: usize = 32;

fn take(arguments: &[String], index: &mut usize, option: &str) -> Result<String> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| Error::Message(format!("{option} requires a value")))
}

fn number<T>(arguments: &[String], index: &mut usize, option: &str) -> Result<T>
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    take(arguments, index, option)?
        .parse()
        .map_err(|error| Error::Message(format!("invalid {option}: {error}")))
}

fn run() -> Result<()> {
    let program_started = Instant::now();
    let mut arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if arguments
        .first()
        .is_some_and(|value| value == "_kraken-images")
    {
        arguments.remove(0);
    }
    let mut images = Vec::new();
    let mut options = KrakenOptions::default();
    let mut profile = false;
    let mut rgba_input = false;
    let mut skip_warmup = false;
    let mut index = 0;
    while index < arguments.len() {
        let option = arguments[index].as_str();
        match option {
            "--list" => {
                let path = PathBuf::from(take(&arguments, &mut index, option)?);
                let contents =
                    std::fs::read_to_string(&path).map_err(|source| Error::io(&path, source))?;
                images.extend(contents.lines().filter_map(|line| {
                    let line = line.trim();
                    if line.is_empty() {
                        None
                    } else {
                        let mut path = PathBuf::from(line);
                        path.set_extension("png");
                        Some(path)
                    }
                }));
            }
            "--kraken-model" => {
                options.model = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-codec" => {
                options.codec = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--onnx-runtime" => {
                options.runtime = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-runtime-wheel" => {
                options.runtime_wheel = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-python" => {
                options.python = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-blla-pack" => {
                options.blla_pack = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-recognizer-pack" => {
                options.recognizer_pack = Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-tesseract-library" => {
                options.tesseract_library =
                    Some(PathBuf::from(take(&arguments, &mut index, option)?))
            }
            "--kraken-layout" => {
                let value = take(&arguments, &mut index, option)?;
                options.layout = KrakenLayout::parse(&value)
                    .ok_or_else(|| Error::Message(format!("invalid {option}: {value}")))?;
            }
            "--kraken-tier" => {
                let value = take(&arguments, &mut index, option)?;
                options.tier = KrakenTier::parse(&value)
                    .ok_or_else(|| Error::Message(format!("invalid {option}: {value}")))?;
            }
            "--kraken-backend" => {
                let value = take(&arguments, &mut index, option)?;
                options.backend = KrakenBackend::parse(&value)
                    .ok_or_else(|| Error::Message(format!("invalid {option}: {value}")))?;
            }
            "--kraken-device" => options.device = Some(take(&arguments, &mut index, option)?),
            "--kraken-low-memory" => options.cpu_arena = false,
            "--kraken-workers" => options.workers = number(&arguments, &mut index, option)?,
            "--kraken-threads" => options.threads = number(&arguments, &mut index, option)?,
            "--kraken-layout-workers" => {
                options.layout_workers = number(&arguments, &mut index, option)?
            }
            "--kraken-batch-size" => options.batch_size = number(&arguments, &mut index, option)?,
            "--kraken-runtime-batch-size" => {
                options.runtime_batch_size = number(&arguments, &mut index, option)?
            }
            "--kraken-width-bucket" => {
                options.width_bucket = number(&arguments, &mut index, option)?
            }
            "--kraken-input-height" => {
                options.input_height = number(&arguments, &mut index, option)?
            }
            "--kraken-width-scale" => {
                options.width_scale = Some(number(&arguments, &mut index, option)?)
            }
            "--ocr-dpi" => options.dpi = number(&arguments, &mut index, option)?,
            "--kraken-timeout" => options.timeout_seconds = number(&arguments, &mut index, option)?,
            "--expected-ocr-identity" => {
                options.expected_identity = Some(take(&arguments, &mut index, option)?)
            }
            "--profile" => profile = true,
            "--rgba-input" => rgba_input = true,
            "--skip-warmup" => skip_warmup = true,
            value if value.starts_with('-') => {
                return Err(Error::Message(format!("unknown option: {value}")))
            }
            value => images.push(PathBuf::from(value)),
        }
        index += 1;
    }
    if images.is_empty() {
        return Err(Error::Message(
            "kraken-rust-runner requires image paths or --list <xml-list>".to_owned(),
        ));
    }
    let provider_started = Instant::now();
    let mut provider = KrakenOcr::new(&options)?;
    let provider_init_seconds = provider_started.elapsed().as_secs_f64();
    let identity = provider.identity().to_owned();
    let engine = provider.name().to_owned();
    let warmup_decode_started = Instant::now();
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
    let warmup_decode_seconds = warmup_decode_started.elapsed().as_secs_f64();
    let warmup_started = Instant::now();
    if !skip_warmup {
        provider.warmup_gray_image(&first)?;
    }
    let warmup_seconds = warmup_started.elapsed().as_secs_f64();

    let mut completed = 0;
    let measured_started = Instant::now();
    let mut first_output_seconds = None;
    let mut chunks = Vec::new();
    let input_file_bytes = images
        .iter()
        .filter_map(|path| std::fs::metadata(path).ok())
        .map(|metadata| metadata.len())
        .sum::<u64>();
    for paths in images.chunks(IMAGE_CHUNK) {
        let decode_started = Instant::now();
        let decoded = paths
            .iter()
            .map(|path| {
                image::ImageReader::open(path)
                    .map_err(|source| Error::io(path, source))?
                    .decode()
                    .map_err(|source| {
                        Error::Message(format!("could not decode {}: {source}", path.display()))
                    })
            })
            .collect::<Result<Vec<_>>>()?;
        let decode_seconds = decode_started.elapsed().as_secs_f64();
        let ocr_started = Instant::now();
        let (results, performance) = if rgba_input {
            let decoded = decoded
                .into_iter()
                .map(|image| image.into_rgba8())
                .collect::<Vec<_>>();
            if profile {
                let batch = provider.recognize_rgba_images_profile(&decoded)?;
                (batch.pages, Some(batch.performance))
            } else {
                (provider.recognize_rgba_images_diagnostics(&decoded)?, None)
            }
        } else {
            let decoded = decoded
                .into_iter()
                .map(|image| image.into_luma8())
                .collect::<Vec<_>>();
            if profile {
                let batch = provider.recognize_gray_images_profile(&decoded)?;
                (batch.pages, Some(batch.performance))
            } else {
                (provider.recognize_gray_images_diagnostics(&decoded)?, None)
            }
        };
        let ocr_call_seconds = ocr_started.elapsed().as_secs_f64();
        let output_started = Instant::now();
        for (path, result) in paths.iter().zip(results) {
            completed += 1;
            println!(
                "{}",
                serde_json::to_string(&json!({
                    "image": std::fs::canonicalize(path).unwrap_or_else(|_| path.clone()),
                    "text": result.lines.iter().map(|line| line.text.as_str()).collect::<Vec<_>>().join("\n"),
                    "lines": result.lines,
                    "layout_boxes": result.layout_boxes,
                    "layout_seconds": result.layout_seconds,
                    "recognition_seconds": result.recognition_seconds,
                    "seconds": result.layout_seconds + result.recognition_seconds,
                    "ocr_identity": identity,
                    "ocr_engine": engine,
                    "progress": [completed, images.len()],
                }))?
            );
            first_output_seconds.get_or_insert_with(|| program_started.elapsed().as_secs_f64());
        }
        if profile {
            chunks.push(json!({
                "pages": paths.len(),
                "decode_seconds": decode_seconds,
                "ocr_call_seconds": ocr_call_seconds,
                "output_seconds": output_started.elapsed().as_secs_f64(),
                "performance": performance,
            }));
        }
    }
    if profile {
        let measured_seconds = measured_started.elapsed().as_secs_f64();
        let input_pixels = chunks
            .iter()
            .filter_map(|chunk| chunk["performance"]["input_pixels"].as_u64())
            .sum::<u64>();
        println!(
            "{}",
            serde_json::to_string(&json!({
                "type": "profile",
                "pages": images.len(),
                "input_file_bytes": input_file_bytes,
                "input_pixels": input_pixels,
                "logical_processors": std::thread::available_parallelism().map_or(1, |value| value.get()),
                "provider_init_seconds": provider_init_seconds,
                "warmup_decode_seconds": warmup_decode_seconds,
                "warmup_seconds": warmup_seconds,
                "time_to_first_output_seconds": first_output_seconds,
                "measured_seconds": measured_seconds,
                "pages_per_second": images.len() as f64 / measured_seconds,
                "megapixels_per_second": input_pixels as f64 / 1_000_000.0 / measured_seconds,
                "chunks": chunks,
            }))?
        );
    }
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
