use legal_pdf_core::{Error, Result};
use legal_pdf_ocr::{IndraBackend, IndraLayout, IndraOcr, IndraOptions, IndraTier, OcrPageRequest};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn pdf_requests(bytes: &[u8]) -> Result<Vec<OcrPageRequest>> {
    let pdf = hayro::hayro_syntax::Pdf::new(bytes.to_vec())
        .map_err(|error| Error::Message(format!("could not open PDF: {error:?}")))?;
    pdf.pages()
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
            })
        })
        .collect()
}

fn run() -> Result<()> {
    let mut arguments = std::env::args_os().skip(1);
    let input=arguments.next().map(PathBuf::from).ok_or_else(|| Error::Message(
        "usage: legalocr INPUT.pdf|INPUT.png|IMAGES.json [--model-pack DIR] [--runtime DLL] [--tier balanced|quality|fast|turbo|extreme] [--backend auto|cpu|directml|cuda] [--output FILE]".into()))?;
    if input.as_os_str() == "--help" {
        println!("legalocr INPUT.pdf|INPUT.png|IMAGES.json [--model-pack DIR] [--runtime LIBRARY] [--backend auto|cpu|directml|cuda] [--tier balanced|quality|fast|turbo|extreme] [--output FILE] [--repeat 1..7]");
        return Ok(());
    }
    let home = std::env::current_exe()
        .map_err(|error| Error::Message(error.to_string()))?
        .parent()
        .unwrap()
        .to_path_buf();
    let mut options = IndraOptions {
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
    while let Some(flag) = arguments.next() {
        let value = arguments.next().ok_or_else(|| {
            Error::Message(format!("missing value for {}", flag.to_string_lossy()))
        })?;
        let text = value.to_string_lossy();
        let number = || {
            text.parse::<usize>().map_err(|_| {
                Error::Message(format!("{} must be an integer", flag.to_string_lossy()))
            })
        };
        match flag.to_str() {
            Some("--model-pack") => options.model_pack = Some(value.into()),
            Some("--runtime") => options.runtime = Some(value.into()),
            Some("--output") => output = Some(PathBuf::from(value)),
            Some("--tier") => {
                options.tier = IndraTier::parse(&text)
                    .ok_or_else(|| Error::Message("unknown Indra tier".into()))?
            }
            Some("--backend") => {
                options.backend = if text == "auto" {
                    None
                } else {
                    Some(
                        IndraBackend::parse(&text)
                            .ok_or_else(|| Error::Message("unknown execution backend".into()))?,
                    )
                }
            }
            Some("--repeat") => repeat = number()?,
            Some("--layout") => {
                options.layout = IndraLayout::parse(&text)
                    .ok_or_else(|| Error::Message("unknown line detector".into()))?
            }
            Some("--tesseract-library") => options.tesseract_library = Some(value.into()),
            Some("--workers") => options.workers = number()?,
            Some("--threads") => options.threads = number()?,
            Some("--layout-workers") => options.layout_workers = number()?,
            Some("--batch-size") => options.batch_size = number()?,
            Some("--width-bucket") => options.width_bucket = number()?,
            Some("--dpi") => {
                options.dpi =
                    u16::try_from(number()?).map_err(|_| Error::Message("invalid DPI".into()))?
            }
            _ => {
                return Err(Error::Message(format!(
                    "unknown option {}",
                    flag.to_string_lossy()
                )))
            }
        }
    }
    if !(1..=7).contains(&repeat) {
        return Err(Error::Message("repeat must be between 1 and 7".into()));
    }
    let overwrites_input = |source: &Path| {
        output.as_ref().is_some_and(|path| {
            path == source
                || (path.exists()
                    && std::fs::canonicalize(path).ok() == std::fs::canonicalize(source).ok())
        })
    };
    if overwrites_input(&input) {
        return Err(Error::Message("output must not overwrite the input".into()));
    }
    let bytes = std::fs::read(&input).map_err(|error| Error::io(&input, error))?;
    enum Source {
        Pdf(Vec<OcrPageRequest>),
        Images(Vec<image::GrayImage>),
    }
    let source = if bytes.starts_with(b"%PDF-") {
        Source::Pdf(pdf_requests(&bytes)?)
    } else if input
        .extension()
        .is_some_and(|extension| extension == "json")
    {
        let paths: Vec<PathBuf> = serde_json::from_slice(&bytes)?;
        if paths.is_empty() || paths.len() > 512 {
            return Err(Error::Message(
                "image list must contain 1 to 512 paths".into(),
            ));
        }
        let mut pixels = 0;
        let images=paths.into_iter().map(|path| {
            let path=input.parent().unwrap().join(path);
            if overwrites_input(&path) { return Err(Error::Message("output must not overwrite an input image".into())); }
            let image=image::open(&path).map_err(|error|Error::Message(format!("could not decode {}: {error}",path.display())))?.into_luma8();
            pixels+=image.as_raw().len();
            if pixels>256*1024*1024 {return Err(Error::Message("image list exceeds 256 MiB of decoded pixels; split it into smaller batches".into()));}
            Ok(image)
        }).collect::<Result<Vec<_>>>()?;
        Source::Images(images)
    } else {
        Source::Images(vec![image::load_from_memory(&bytes)
            .map_err(|error| Error::Message(format!("could not decode input: {error}")))?
            .into_luma8()])
    };
    let startup = Instant::now();
    let mut engine = IndraOcr::new(&options)?;
    let configuration = serde_json::to_value(engine.configuration())?;
    let identity = engine.identity().to_owned();
    let startup_seconds = startup.elapsed().as_secs_f64();
    let mut recognize = || -> Result<(serde_json::Value, serde_json::Value)> {
        match &source {
            Source::Pdf(requests) => {
                let pages = engine.extract_pages(&bytes, requests)?;
                Ok((
                    serde_json::to_value(pages)?,
                    serde_json::to_value(engine.last_performance())?,
                ))
            }
            Source::Images(images) => {
                let result = engine.recognize_gray_images_profile(images)?;
                Ok((
                    serde_json::to_value(result.pages)?,
                    serde_json::to_value(result.performance)?,
                ))
            }
        }
    };
    let first = Instant::now();
    let (mut pages, mut profile) = recognize()?;
    let first_document_seconds = first.elapsed().as_secs_f64();
    let mut runs = vec![first_document_seconds];
    if repeat > 1 {
        runs.clear();
        for _ in 0..repeat {
            let begin = Instant::now();
            let value = recognize()?;
            runs.push(begin.elapsed().as_secs_f64());
            pages = value.0;
            profile = value.1;
        }
    }
    let mut sorted = runs.clone();
    sorted.sort_by(f64::total_cmp);
    let seconds = sorted[sorted.len() / 2];
    let count = pages.as_array().map_or(0, Vec::len);
    let document = serde_json::json!({"runtime":configuration,"identity":identity,"startup_seconds":startup_seconds,"first_document_seconds":first_document_seconds,"runs":runs,"seconds":seconds,"pages_per_second":count as f64/seconds,"profile":profile,"pages":pages});
    let mut writer: Box<dyn Write> = match output {
        Some(path) => {
            Box::new(std::fs::File::create(&path).map_err(|error| Error::io(path, error))?)
        }
        None => Box::new(std::io::stdout().lock()),
    };
    serde_json::to_writer(&mut writer, &document)?;
    writer
        .write_all(b"\n")
        .map_err(|error| Error::Message(error.to_string()))?;
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Independently invented page geometry, without document text or model assets.
    fn page(unit: u32) -> Vec<u8> {
        let objects = ["<< /Type /Catalog /Pages 2 0 R >>".to_owned(),
            "<< /Type /Pages /Count 1 /Kids [3 0 R] >>".to_owned(),
            format!("<< /Type /Page /Parent 2 0 R /Resources << >> /MediaBox [0 0 400 500] /CropBox [10 20 210 320] /Rotate 90 /UserUnit {unit} >>")];
        let mut bytes = b"%PDF-1.7\n".to_vec();
        let mut offsets = Vec::new();
        for (index, object) in objects.iter().enumerate() {
            offsets.push(bytes.len());
            bytes.extend_from_slice(format!("{} 0 obj\n{object}\nendobj\n", index + 1).as_bytes());
        }
        let xref = bytes.len();
        bytes.extend_from_slice(b"xref\n0 4\n0000000000 65535 f \n");
        for offset in offsets {
            bytes.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
        }
        bytes.extend_from_slice(
            format!("trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n").as_bytes(),
        );
        bytes
    }

    #[test]
    fn pdf_geometry_uses_visible_rotated_crop_and_physical_units() {
        let requests = pdf_requests(&page(2)).unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!((requests[0].width, requests[0].height), (600.0, 400.0));
        assert!(pdf_requests(&page(0)).is_err());
    }
}
