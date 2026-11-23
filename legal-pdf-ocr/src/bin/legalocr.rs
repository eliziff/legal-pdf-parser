#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

use legal_pdf_core::render::{open_pdf, PageRenderer};
use legal_pdf_core::{Error, Result};
use legal_pdf_ocr::{OcrOptions, OcrPageRequest, OcrProvider, PdfOcrProvider};
use serde::de::DeserializeOwned;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "legalocr INPUT.pdf|INPUT.png|IMAGES.json [--model-pack DIR] [--runtime LIBRARY] [--backend auto|cpu|directml|cuda] [--tier balanced|quality|fast|turbo|extreme] [--pages 1,3-5] [--render-pages 1,3-5] [--render-out DIR] [--output FILE] [--repeat 1..7]
Phase timings: set LEGALPDF_PHASE_TIMING.";

fn parse_pages_spec(spec: &str, total: usize) -> Result<Vec<usize>> {
    let mut selected = Vec::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if let Some((a, b)) = part.split_once('-') {
            let start: usize = a.trim().parse().map_err(|_| Error::Message(format!("invalid page range: {part}")))?;
            let end: usize = b.trim().parse().map_err(|_| Error::Message(format!("invalid page range: {part}")))?;
            if start == 0 || end < start || end > total {
                return Err(Error::Message(format!("page range {part} out of bounds 1..={total}")));
            }
            selected.extend((start - 1)..end);
        } else {
            let p: usize = part.parse().map_err(|_| Error::Message(format!("invalid page number: {part}")))?;
            if p == 0 || p > total {
                return Err(Error::Message(format!("page {p} out of bounds 1..={total}")));
            }
            selected.push(p - 1);
        }
    }
    selected.sort_unstable();
    selected.dedup();
    Ok(selected)
}

fn pdf_requests(bytes: &[u8]) -> Result<Vec<OcrPageRequest>> {
    open_pdf(bytes)?
        .pages()
        .iter()
        .enumerate()
        .map(|(page_index, page)| {
            let (width, height) = page.render_dimensions();
            let unit = page.raw().get::<f32>(b"UserUnit").unwrap_or(1.0);
            if !unit.is_finite() || unit <= 0.0 {
                return Err(Error::Message("PDF page has invalid UserUnit".into()));
            }
            Ok(OcrPageRequest {
                page_index,
                width: f64::from(width) * f64::from(unit),
                height: f64::from(height) * f64::from(unit),
                regions: Vec::new(),
                lines: Vec::new(),
                words: Vec::new(),
            })
        })
        .collect()
}

/// A command-line word read as the option value its options file would name.
fn named<T: DeserializeOwned>(text: &str, what: &str) -> Result<T> {
    serde_json::from_value(serde_json::Value::String(text.to_owned())).map_err(|_| Error::Message(format!("unknown {what}")))
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let input = arguments.next().map(PathBuf::from).ok_or_else(|| Error::Message(format!("usage: {USAGE}")))?;
    if input.as_os_str() == "--help" {
        println!("{USAGE}");
        return Ok(());
    }
    let home = std::env::current_exe().map_err(|error| Error::Message(error.to_string()))?.parent().unwrap().to_path_buf();
    let mut options = OcrOptions {
        model_pack: Some(home.join("runtime/indra")),
        runtime: Some(home.join("runtime/onnx").join(if cfg!(windows) {
            "onnxruntime.dll"
        } else if cfg!(target_os = "macos") {
            "libonnxruntime.dylib"
        } else {
            "libonnxruntime.so"
        })),
        ..Default::default()
    };
    let mut output = None;
    let mut repeat = 1;
    let mut pages_spec: Option<String> = None;
    let mut render_pages_spec: Option<String> = None;
    let mut render_out: Option<PathBuf> = None;
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or_else(|| Error::Message(format!("missing value for {}", flag.to_string_lossy())))?;
        let text = value.to_string_lossy();
        let number = || text.parse::<usize>().map_err(|_| Error::Message(format!("{} must be an integer", flag.to_string_lossy())));
        match flag.to_str() {
            Some("--model-pack") => options.model_pack = Some(value.into()),
            Some("--runtime") => options.runtime = Some(value.into()),
            Some("--output") => output = Some(PathBuf::from(value)),
            Some("--pages") => pages_spec = Some(text.into_owned()),
            Some("--render-pages") => render_pages_spec = Some(text.into_owned()),
            Some("--render-out") => render_out = Some(PathBuf::from(value)),
            Some("--tier") => options.tier = named(&text, "Indra tier")?,
            Some("--backend") => options.backend = if text == "auto" { None } else { Some(named(&text, "execution backend")?) },
            Some("--repeat") => repeat = number()?,
            Some("--workers") => options.workers = number()?,
            Some("--threads") => options.threads = number()?,
            Some("--layout-workers") => options.layout_workers = number()?,
            Some("--batch-size") => options.batch_size = number()?,
            Some("--width-bucket") => options.width_bucket = number()?,
            Some("--dpi") => options.dpi = u16::try_from(number()?).map_err(|_| Error::Message("invalid DPI".into()))?,
            _ => return Err(Error::Message(format!("unknown option {}", flag.to_string_lossy()))),
        }
    }
    if !(1..=7).contains(&repeat) {
        return Err(Error::Message("repeat must be between 1 and 7".into()));
    }
    let overwrites_input = |source: &Path| {
        output.as_ref().is_some_and(|path| {
            path == source || (path.exists() && std::fs::canonicalize(path).ok() == std::fs::canonicalize(source).ok())
        })
    };
    if overwrites_input(&input) {
        return Err(Error::Message("output must not overwrite the input".into()));
    }
    let bytes = std::fs::read(&input).map_err(|error| Error::io(&input, error))?;
    if let Some(ref spec) = render_pages_spec {
        if !bytes.starts_with(b"%PDF-") {
            return Err(Error::Message("--render-pages requires a PDF input".into()));
        }
        let pdf = open_pdf(&bytes)?;
        let indices = parse_pages_spec(spec, pdf.pages().len())?;
        let out_dir = render_out.unwrap_or_else(|| PathBuf::from("."));
        std::fs::create_dir_all(&out_dir).map_err(|error| Error::io(&out_dir, error))?;
        let renderer = PageRenderer::new(&pdf, f64::from(options.dpi));
        let mut rendered = Vec::with_capacity(indices.len());
        for idx in indices {
            let png = renderer
                .render(idx)
                .ok_or_else(|| Error::Message(format!("page {} rendered empty", idx + 1)))?
                .into_png()
                .map_err(|error| Error::Message(format!("page {} PNG encode failed: {error}", idx + 1)))?;
            let path = out_dir.join(format!("page_{:04}.png", idx + 1));
            std::fs::write(&path, &png).map_err(|error| Error::io(&path, error))?;
            rendered.push(path);
        }
        if output.is_none() {
            println!("{}", serde_json::json!({ "rendered": rendered }));
            return Ok(());
        }
    }
    enum Source {
        Pdf(Vec<OcrPageRequest>),
        Images(Vec<image::GrayImage>),
    }
    let source = if bytes.starts_with(b"%PDF-") {
        let mut reqs = pdf_requests(&bytes)?;
        if let Some(ref spec) = pages_spec {
            let keep = parse_pages_spec(spec, reqs.len())?;
            reqs = keep.into_iter().map(|i| reqs[i].clone()).collect();
        }
        Source::Pdf(reqs)
    } else if input.extension().is_some_and(|extension| extension == "json") {
        let paths: Vec<PathBuf> = serde_json::from_slice(&bytes)?;
        if paths.is_empty() || paths.len() > 512 {
            return Err(Error::Message("image list must contain 1 to 512 paths".into()));
        }
        let mut pixels = 0;
        let images = paths
            .into_iter()
            .map(|path| {
                let path = input.parent().unwrap().join(path);
                if overwrites_input(&path) {
                    return Err(Error::Message("output must not overwrite an input image".into()));
                }
                let image = image::open(&path)
                    .map_err(|error| Error::Message(format!("could not decode {}: {error}", path.display())))?
                    .into_luma8();
                pixels += image.as_raw().len();
                if pixels > 256 * 1024 * 1024 {
                    return Err(Error::Message("image list exceeds 256 MiB of decoded pixels; split it into smaller batches".into()));
                }
                Ok(image)
            })
            .collect::<Result<Vec<_>>>()?;
        Source::Images(images)
    } else {
        Source::Images(vec![image::load_from_memory(&bytes)
            .map_err(|error| Error::Message(format!("could not decode input: {error}")))?
            .into_luma8()])
    };
    let startup = Instant::now();
    let mut engine = OcrProvider::new(&options)?;
    let configuration = serde_json::to_value(engine.configuration())?;
    let identity = engine.identity().to_owned();
    let startup_seconds = startup.elapsed().as_secs_f64();
    let mut recognize = || -> Result<serde_json::Value> {
        Ok(match &source {
            Source::Pdf(requests) => serde_json::to_value(engine.extract_pages(&bytes, requests)?)?,
            Source::Images(images) => serde_json::to_value(engine.recognize_images(images)?)?,
        })
    };
    let first = Instant::now();
    let mut pages = recognize()?;
    let first_document_seconds = first.elapsed().as_secs_f64();
    legal_pdf_core::profile::tallied();
    let mut runs = vec![first_document_seconds];
    if repeat > 1 {
        runs.clear();
        for _ in 0..repeat {
            let begin = Instant::now();
            pages = recognize()?;
            runs.push(begin.elapsed().as_secs_f64());
            legal_pdf_core::profile::tallied();
        }
    }
    let mut sorted = runs.clone();
    sorted.sort_by(f64::total_cmp);
    let seconds = sorted[sorted.len() / 2];
    let count = pages.as_array().map_or(0, Vec::len);
    let document = serde_json::json!({"runtime":configuration,"identity":identity,"startup_seconds":startup_seconds,"first_document_seconds":first_document_seconds,"runs":runs,"seconds":seconds,"pages_per_second":count as f64/seconds,"pages":pages});
    let mut writer: Box<dyn Write> = match output {
        Some(path) => Box::new(std::fs::File::create(&path).map_err(|error| Error::io(path, error))?),
        None => Box::new(std::io::stdout().lock()),
    };
    serde_json::to_writer(&mut writer, &document)?;
    writer.write_all(b"\n").map_err(|error| Error::Message(error.to_string()))?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}
