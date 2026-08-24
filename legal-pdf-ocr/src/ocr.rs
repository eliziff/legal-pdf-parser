use legal_pdf_core::{Error, OcrLine, OcrPageRequest, OcrPageResult, PdfOcrProvider, Result};
use serde::Deserialize;
use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
#[cfg(feature = "ocr")]
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub enum OcrOptions {
    Tesseract(TesseractOptions),
    #[cfg(feature = "kraken")]
    Kraken(crate::kraken::KrakenOptions),
}

impl From<TesseractOptions> for OcrOptions {
    fn from(options: TesseractOptions) -> Self {
        Self::Tesseract(options)
    }
}

#[cfg(feature = "kraken")]
impl From<crate::kraken::KrakenOptions> for OcrOptions {
    fn from(options: crate::kraken::KrakenOptions) -> Self {
        Self::Kraken(options)
    }
}

pub enum OcrProvider {
    Tesseract(TesseractOcr),
    #[cfg(feature = "kraken")]
    Kraken(crate::kraken::KrakenOcr),
}

pub struct PreparedOcrProvider {
    provider: PreparedProvider,
}

enum PreparedProvider {
    Tesseract(PreparedTesseract),
    #[cfg(feature = "kraken")]
    Kraken(crate::kraken::PreparedKraken),
}

impl OcrProvider {
    pub fn new(options: &OcrOptions) -> Result<Self> {
        Self::from_prepared(options, Self::prepare(options)?)
    }

    pub fn prepare(options: &OcrOptions) -> Result<PreparedOcrProvider> {
        match options {
            OcrOptions::Tesseract(options) => {
                let provider = TesseractOcr::prepare(options)?;
                Ok(PreparedOcrProvider {
                    provider: PreparedProvider::Tesseract(provider),
                })
            }
            #[cfg(feature = "kraken")]
            OcrOptions::Kraken(options) => {
                let provider = crate::kraken::KrakenOcr::prepare(options)?;
                Ok(PreparedOcrProvider {
                    provider: PreparedProvider::Kraken(provider),
                })
            }
        }
    }

    pub fn from_prepared(options: &OcrOptions, prepared: PreparedOcrProvider) -> Result<Self> {
        match (options, prepared.provider) {
            (OcrOptions::Tesseract(options), PreparedProvider::Tesseract(prepared)) => Ok(
                Self::Tesseract(TesseractOcr::from_prepared(options, prepared)),
            ),
            #[cfg(feature = "kraken")]
            (OcrOptions::Kraken(options), PreparedProvider::Kraken(prepared)) => {
                crate::kraken::KrakenOcr::from_prepared(options, prepared).map(Self::Kraken)
            }
            #[cfg(feature = "kraken")]
            _ => Err(Error::Message(
                "OCR options changed after identity preparation".to_owned(),
            )),
        }
    }

    pub fn identity(&self) -> &str {
        match self {
            Self::Tesseract(provider) => provider.identity(),
            #[cfg(feature = "kraken")]
            Self::Kraken(provider) => provider.identity(),
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Self::Tesseract(provider) => provider.name(),
            #[cfg(feature = "kraken")]
            Self::Kraken(provider) => provider.name(),
        }
    }

    fn extract_pages_inner(
        &mut self,
        pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        match self {
            Self::Tesseract(provider) => provider.extract_pages(pdf, requests),
            #[cfg(feature = "kraken")]
            Self::Kraken(provider) => provider.extract_pages(pdf, requests),
        }
    }
}

impl PreparedOcrProvider {
    pub fn identity(&self) -> &str {
        match &self.provider {
            PreparedProvider::Tesseract(provider) => &provider.identity,
            #[cfg(feature = "kraken")]
            PreparedProvider::Kraken(provider) => provider.identity(),
        }
    }

    pub fn name(&self) -> &str {
        match &self.provider {
            PreparedProvider::Tesseract(provider) => &provider.name,
            #[cfg(feature = "kraken")]
            PreparedProvider::Kraken(provider) => provider.name(),
        }
    }
}

impl PdfOcrProvider for OcrProvider {
    fn extract_pages(
        &mut self,
        pdf: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        self.extract_pages_inner(pdf, requests)
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TesseractOptions {
    pub command: Option<PathBuf>,
    pub language: String,
    pub dpi: u16,
    pub psm: u8,
    pub timeout_seconds: u64,
    pub expected_identity: Option<String>,
}

impl Default for TesseractOptions {
    fn default() -> Self {
        Self {
            command: None,
            language: "eng".to_owned(),
            dpi: 180,
            psm: 3,
            timeout_seconds: 120,
            expected_identity: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TesseractOcr {
    #[cfg_attr(not(feature = "ocr"), allow(dead_code))]
    command: PathBuf,
    #[cfg_attr(not(feature = "ocr"), allow(dead_code))]
    language: String,
    #[cfg_attr(not(feature = "ocr"), allow(dead_code))]
    dpi: u16,
    #[cfg_attr(not(feature = "ocr"), allow(dead_code))]
    psm: u8,
    #[cfg_attr(not(feature = "ocr"), allow(dead_code))]
    timeout: Duration,
    identity: String,
    name: String,
}

struct PreparedTesseract {
    command: PathBuf,
    identity: String,
    name: String,
}

impl TesseractOcr {
    pub fn new(options: &TesseractOptions) -> Result<Self> {
        let prepared = Self::prepare(options)?;
        Ok(Self::from_prepared(options, prepared))
    }

    fn prepare(options: &TesseractOptions) -> Result<PreparedTesseract> {
        validate_options(options)?;
        let command = options
            .command
            .clone()
            .or_else(|| std::env::var_os("LEGALPDF_TESSERACT_COMMAND").map(PathBuf::from))
            .unwrap_or_else(|| PathBuf::from("tesseract"));
        let output = run_checked(
            &command,
            &[OsString::from("--version")],
            Duration::from_secs(10),
        )?;
        let version = first_sanitized_line(&output).unwrap_or_else(|| "unknown".to_owned());
        let identity = format!("tesseract-cli-v1:{version}");
        if options
            .expected_identity
            .as_deref()
            .is_some_and(|expected| expected != identity)
        {
            return Err(Error::Message(format!(
                "Tesseract identity changed before OCR began: expected {}, found {identity}",
                options.expected_identity.as_deref().unwrap_or_default()
            )));
        }
        let name = format!(
            "{identity}:lang={}:dpi={}:psm={}",
            options.language, options.dpi, options.psm
        );
        Ok(PreparedTesseract {
            command,
            identity,
            name,
        })
    }

    fn from_prepared(options: &TesseractOptions, prepared: PreparedTesseract) -> Self {
        Self {
            command: prepared.command,
            language: options.language.clone(),
            dpi: options.dpi,
            psm: options.psm,
            timeout: Duration::from_secs(options.timeout_seconds),
            identity: prepared.identity,
            name: prepared.name,
        }
    }

    pub fn identity(&self) -> &str {
        &self.identity
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "ocr")]
    pub(crate) fn extract_pages(
        &self,
        bytes: &[u8],
        requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        use hayro::hayro_interpret::InterpreterSettings;
        use hayro::hayro_syntax::Pdf;
        use hayro::vello_cpu::color::palette::css::WHITE;
        use hayro::{render, RenderCache, RenderSettings};
        use std::fs;
        use std::io::Write;

        if requests.is_empty() {
            return Ok(Vec::new());
        }
        let pdf = Pdf::new(bytes.to_vec()).map_err(|error| {
            Error::Message(format!("OCR renderer could not open PDF: {error:?}"))
        })?;
        let cache = RenderCache::new();
        let interpreter = InterpreterSettings::default();
        let scale = f32::from(self.dpi) / 72.0;
        let settings = RenderSettings {
            x_scale: scale,
            y_scale: scale,
            bg_color: WHITE,
            ..Default::default()
        };
        let mut results = Vec::with_capacity(requests.len());
        for request in requests {
            let page = pdf.pages().iter().nth(request.page_index).ok_or_else(|| {
                Error::Message(format!(
                    "PDF page index is out of range: {}",
                    request.page_index
                ))
            })?;
            let pixmap = render(page, &cache, &interpreter, &settings);
            let pixel_width = pixmap.width();
            let pixel_height = pixmap.height();
            if pixel_width < 1 || pixel_height < 1 {
                return Err(Error::Message(format!(
                    "OCR renderer produced an empty page image for page {}",
                    request.page_index + 1
                )));
            }
            let separator_y = raster_separator_y(&pixmap, request.height);
            let png = pixmap.into_png().map_err(|error| {
                Error::Message(format!("OCR renderer could not encode page PNG: {error}"))
            })?;
            let temporary = temporary_png(request.page_index);
            let mut file = fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)
                .map_err(|source| Error::io(&temporary, source))?;
            if let Err(source) = file.write_all(&png) {
                let _ = fs::remove_file(&temporary);
                return Err(Error::io(&temporary, source));
            }
            drop(file);
            let arguments = [
                temporary.as_os_str().to_owned(),
                OsString::from("stdout"),
                OsString::from("-l"),
                OsString::from(&self.language),
                OsString::from("--dpi"),
                OsString::from(self.dpi.to_string()),
                OsString::from("--psm"),
                OsString::from(self.psm.to_string()),
                OsString::from("tsv"),
            ];
            let output = run_checked(&self.command, &arguments, self.timeout);
            let cleanup =
                fs::remove_file(&temporary).map_err(|source| Error::io(&temporary, source));
            let output = output?;
            cleanup?;
            let text = String::from_utf8_lossy(&output);
            let lines = tsv_lines(
                &text,
                request.width / f64::from(pixel_width),
                request.height / f64::from(pixel_height),
                request.width,
                request.height,
            );
            results.push(OcrPageResult {
                page_index: request.page_index,
                lines,
                separator_y,
            });
        }
        Ok(results)
    }

    #[cfg(not(feature = "ocr"))]
    pub(crate) fn extract_pages(
        &self,
        _pdf: &[u8],
        _requests: &[OcrPageRequest],
    ) -> Result<Vec<OcrPageResult>> {
        Err(Error::Message(
            "this legalpdf binary was built without the `ocr` feature".to_owned(),
        ))
    }
}

fn validate_options(options: &TesseractOptions) -> Result<()> {
    if options.language.is_empty()
        || !options
            .language
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || "_+-".contains(character))
    {
        return Err(Error::Message(
            "OCR language must be a Tesseract language code".to_owned(),
        ));
    }
    if !(72..=600).contains(&options.dpi) {
        return Err(Error::Message(
            "OCR DPI must be between 72 and 600".to_owned(),
        ));
    }
    if options.psm > 13 {
        return Err(Error::Message(
            "Tesseract page segmentation mode must be 0 through 13".to_owned(),
        ));
    }
    if !(1..=3600).contains(&options.timeout_seconds) {
        return Err(Error::Message(
            "OCR timeout must be between 1 and 3600 seconds".to_owned(),
        ));
    }
    Ok(())
}

fn first_sanitized_line(bytes: &[u8]) -> Option<String> {
    String::from_utf8_lossy(bytes).lines().find_map(|line| {
        let value: String = line
            .chars()
            .map(|character| {
                if character.is_control() {
                    ' '
                } else {
                    character
                }
            })
            .take(200)
            .collect();
        let value = value.trim().to_owned();
        (!value.is_empty()).then_some(value)
    })
}

fn run_checked(command: &Path, arguments: &[OsString], timeout: Duration) -> Result<Vec<u8>> {
    let mut process = Command::new(command);
    process
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        process.creation_flags(0x0800_0000);
    }
    let mut child = process.spawn().map_err(|source| {
        if source.kind() == std::io::ErrorKind::NotFound {
            Error::Message(
                "Tesseract was not found; install it or set LEGALPDF_TESSERACT_COMMAND".to_owned(),
            )
        } else {
            Error::io(command, source)
        }
    })?;
    let stdout = child.stdout.take().expect("piped stdout");
    let stderr = child.stderr.take().expect("piped stderr");
    let stdout_reader = thread::spawn(move || read_all(stdout));
    let stderr_reader = thread::spawn(move || read_all(stderr));
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|source| Error::io(command, source))?
        {
            break status;
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            let _ = stdout_reader.join();
            let _ = stderr_reader.join();
            return Err(Error::Message("Tesseract OCR timed out".to_owned()));
        }
        thread::sleep(Duration::from_millis(10));
    };
    let stdout = join_reader(stdout_reader, command)?;
    let stderr = join_reader(stderr_reader, command)?;
    if !status.success() {
        let detail = first_sanitized_line(&stderr).unwrap_or_else(|| "no error output".to_owned());
        return Err(Error::Message(format!(
            "Tesseract OCR failed with exit code {}: {detail}",
            status
                .code()
                .map_or_else(|| "unknown".to_owned(), |code| code.to_string())
        )));
    }
    Ok(stdout)
}

fn read_all(mut reader: impl Read) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn join_reader(
    handle: thread::JoinHandle<std::io::Result<Vec<u8>>>,
    command: &Path,
) -> Result<Vec<u8>> {
    handle
        .join()
        .map_err(|_| Error::Message("Tesseract output reader panicked".to_owned()))?
        .map_err(|source| Error::io(command, source))
}

#[cfg(feature = "ocr")]
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

#[cfg(feature = "ocr")]
fn temporary_png(page_index: usize) -> PathBuf {
    let sequence = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "legalpdf-ocr-{}-{sequence}-p{}.png",
        std::process::id(),
        page_index + 1
    ))
}

#[derive(Debug)]
struct TsvWord {
    order: i64,
    text: String,
    bbox: [f64; 4],
    confidence: f64,
}

pub(crate) fn tsv_lines(
    value: &str,
    x_scale: f64,
    y_scale: f64,
    page_width: f64,
    page_height: f64,
) -> Vec<OcrLine> {
    let mut lines = value.lines();
    let Some(header) = lines.next() else {
        return Vec::new();
    };
    let columns: BTreeMap<&str, usize> = header
        .trim_end_matches('\r')
        .split('\t')
        .enumerate()
        .map(|(index, name)| (name, index))
        .collect();
    let required = [
        "level",
        "page_num",
        "block_num",
        "par_num",
        "line_num",
        "word_num",
        "left",
        "top",
        "width",
        "height",
        "conf",
        "text",
    ];
    if required.iter().any(|name| !columns.contains_key(name)) {
        return Vec::new();
    }
    let mut groups: Vec<Vec<TsvWord>> = Vec::new();
    let mut group_indexes: HashMap<[String; 4], usize> = HashMap::new();
    for line in lines {
        let row = line.trim_end_matches('\r').split('\t').collect::<Vec<_>>();
        let field = |name: &str| -> Option<&str> {
            columns.get(name).and_then(|index| row.get(*index)).copied()
        };
        if field("level") != Some("5") {
            continue;
        }
        let text = field("text").unwrap_or_default().trim();
        if text.is_empty() {
            continue;
        }
        let Some(left) = field("left").and_then(|item| item.parse::<f64>().ok()) else {
            continue;
        };
        let Some(top) = field("top").and_then(|item| item.parse::<f64>().ok()) else {
            continue;
        };
        let Some(width) = field("width").and_then(|item| item.parse::<f64>().ok()) else {
            continue;
        };
        let Some(height) = field("height").and_then(|item| item.parse::<f64>().ok()) else {
            continue;
        };
        if width <= 0.0 || height <= 0.0 {
            continue;
        }
        let key = ["page_num", "block_num", "par_num", "line_num"]
            .map(|name| field(name).unwrap_or("0").to_owned());
        let group_index = if let Some(index) = group_indexes.get(&key) {
            *index
        } else {
            let index = groups.len();
            group_indexes.insert(key, index);
            groups.push(Vec::new());
            index
        };
        groups[group_index].push(TsvWord {
            order: field("word_num")
                .and_then(|item| item.parse().ok())
                .unwrap_or(0),
            text: text.to_owned(),
            bbox: [
                (left * x_scale).max(0.0),
                (top * y_scale).max(0.0),
                ((left + width) * x_scale).min(page_width),
                ((top + height) * y_scale).min(page_height),
            ],
            confidence: field("conf")
                .and_then(|item| item.parse().ok())
                .unwrap_or(0.0),
        });
    }
    groups
        .into_iter()
        .filter_map(|mut words| {
            words.sort_by_key(|word| word.order);
            let first = words.first()?;
            let mut bbox = first.bbox;
            let mut confidence = 0.0;
            let mut confidence_count = 0_u32;
            for word in &words {
                bbox[0] = bbox[0].min(word.bbox[0]);
                bbox[1] = bbox[1].min(word.bbox[1]);
                bbox[2] = bbox[2].max(word.bbox[2]);
                bbox[3] = bbox[3].max(word.bbox[3]);
                if word.confidence >= 0.0 {
                    confidence += word.confidence;
                    confidence_count += 1;
                }
            }
            Some(OcrLine {
                text: words
                    .into_iter()
                    .map(|word| word.text)
                    .collect::<Vec<_>>()
                    .join(" "),
                bbox,
                confidence: if confidence_count == 0 {
                    0.0
                } else {
                    (confidence / confidence_count as f64 / 100.0).clamp(0.0, 1.0)
                },
                baseline: vec![],
                boundary: vec![],
                words: vec![],
                region_id: String::new(),
                region_type: "unknown".to_owned(),
                block_index: 0,
            })
        })
        .collect()
}

#[cfg(feature = "ocr")]
pub(crate) fn raster_separator_y(
    pixmap: &hayro::vello_cpu::Pixmap,
    page_height: f64,
) -> Option<f64> {
    let width = usize::from(pixmap.width());
    let height = usize::from(pixmap.height());
    let gray = pixmap
        .data_as_u8_slice()
        .chunks_exact(4)
        .map(|pixel| {
            ((u32::from(pixel[0]) * 77 + u32::from(pixel[1]) * 150 + u32::from(pixel[2]) * 29)
                / 256) as u8
        })
        .collect::<Vec<_>>();
    let record = crate::separator::scan_gray_page(&gray, width, height);
    if !matches!(record.separator_status, Some("found" | "found_two_column")) {
        return None;
    }
    record.separators.and_then(|separators| {
        separators
            .first()
            .map(|rule| rule.y_center_ratio * page_height)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tesseract_options_reject_unsafe_values() {
        let mut options = TesseractOptions::default();
        options.language = "eng;rm".to_owned();
        assert!(validate_options(&options).is_err());
        options.language = "eng+fra".to_owned();
        options.dpi = 601;
        assert!(validate_options(&options).is_err());
    }

    #[test]
    fn tsv_words_become_pdf_coordinate_lines() {
        let value = concat!(
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n",
            "5\t1\t1\t1\t1\t1\t10\t20\t30\t10\t90\tHello\n",
            "5\t1\t1\t1\t1\t2\t45\t20\t35\t10\t80\tworld\n",
        );
        let lines = tsv_lines(value, 0.5, 0.25, 100.0, 100.0);
        assert_eq!(lines.len(), 1);
        assert_eq!(lines[0].text, "Hello world");
        assert_eq!(lines[0].bbox, [5.0, 5.0, 40.0, 7.5]);
        assert_eq!(lines[0].confidence, 0.85);
    }
}
