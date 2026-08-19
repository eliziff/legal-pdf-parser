use crate::error::{Error, Result};
use crate::kraken::{
    canonical_file, sha256_file, KrakenBackend, KrakenImageDiagnostics, KrakenOptions,
};
use crate::ocr::{OcrLine, OcrWord};
use image::{GrayImage, ImageFormat};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

pub(crate) struct BllaRuntime {
    child: Child,
    input: ChildStdin,
    output: mpsc::Receiver<ProcessLine>,
    stdout_thread: Option<JoinHandle<()>>,
    stderr_thread: Option<JoinHandle<()>>,
    stderr: Arc<Mutex<String>>,
    timeout: Duration,
    request_id: u64,
    identity: String,
    name: String,
}

enum ProcessLine {
    Value(String),
    Error(String),
    Eof,
}

#[derive(Deserialize)]
struct RuntimeResponse {
    id: Option<u64>,
    ok: bool,
    #[serde(default)]
    result: Value,
    #[serde(default)]
    error: Option<RuntimeError>,
}

#[derive(Deserialize)]
struct RuntimeError {
    #[serde(rename = "type")]
    kind: String,
    message: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct RuntimePage {
    width: usize,
    height: usize,
    lines: Vec<RuntimeLine>,
    #[serde(default)]
    regions: BTreeMap<String, Vec<RuntimeRegion>>,
}

#[derive(Deserialize)]
struct RuntimeLine {
    text: String,
    confidence: f64,
    #[serde(default)]
    baseline: Vec<[f64; 2]>,
    #[serde(default)]
    boundary: Vec<[f64; 2]>,
    #[serde(default)]
    characters: Vec<RuntimeCharacter>,
    #[serde(default)]
    regions: Vec<String>,
}

#[derive(Deserialize)]
struct RuntimeCharacter {
    text: String,
    #[serde(default)]
    polygon: Vec<[f64; 2]>,
}

#[derive(Deserialize)]
struct RuntimeRegion {
    id: String,
    #[serde(default)]
    boundary: Vec<[f64; 2]>,
}

impl BllaRuntime {
    pub(crate) fn new(options: &KrakenOptions) -> Result<Self> {
        let wheel = required_file(
            &options.runtime_wheel,
            "LEGALPDF_KRAKEN_RUNTIME_WHEEL",
            "Kraken runtime wheel",
        )?;
        let blla = verified_pack(
            options
                .blla_pack
                .clone()
                .or_else(|| std::env::var_os("LEGALPDF_KRAKEN_BLLA_PACK").map(PathBuf::from)),
            "blla-segmentation",
            "Kraken BLLA pack",
        )?;
        let recognizer = verified_pack(
            options
                .recognizer_pack
                .clone()
                .or_else(|| std::env::var_os("LEGALPDF_KRAKEN_RECOGNIZER_PACK").map(PathBuf::from)),
            "recognition",
            "Kraken recognizer pack",
        )?;
        let python = resolve_python(options.python.as_deref())?;
        let python_path = python_path(&wheel)?;
        let environment = python_environment(&python, &python_path)?;
        let device = runtime_device(options.backend, options.device.as_deref())?;
        let identity = format!(
            "kraken-lite-process-v1:backend={}:device={device}:fallback={}:python={}:environment={}:runtime={}:blla={}:recognizer={}",
            options.backend.name(),
            if options.cpu_fallback { "cpu" } else { "none" },
            sha256_file(&python)?,
            sha256_bytes(environment.as_bytes()),
            sha256_file(&wheel)?,
            blla.identity,
            recognizer.identity,
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

        let mut command = Command::new(&python);
        command
            .args(["-m", "kraken_lite.cli", "serve", "--blla"])
            .arg(&blla.root)
            .arg("--recognizer")
            .arg(&recognizer.root)
            .arg("--device")
            .arg(&device)
            .arg("--threads")
            .arg(options.threads.max(1).to_string())
            .arg("--batch-size")
            .arg(options.runtime_batch_size.to_string())
            .env("PYTHONPATH", python_path)
            .env("PYTHONIOENCODING", "utf-8")
            .env("PYTHONUTF8", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        if options.backend == KrakenBackend::OneDnn {
            command.args(["--providers", "DnnlExecutionProvider"]);
        }
        if !options.cpu_fallback && options.backend != KrakenBackend::Cpu {
            command.arg("--strict-device");
        }
        hide_window(&mut command);
        let mut child = command
            .spawn()
            .map_err(|source| Error::io(&python, source))?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| Error::Message("Kraken runtime stdin was not captured".to_owned()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::Message("Kraken runtime stdout was not captured".to_owned()))?;
        let stderr_pipe = child
            .stderr
            .take()
            .ok_or_else(|| Error::Message("Kraken runtime stderr was not captured".to_owned()))?;
        let (sender, output) = mpsc::channel();
        let stdout_thread = thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if sender.send(ProcessLine::Value(line)).is_err() {
                            return;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(ProcessLine::Error(error.to_string()));
                        return;
                    }
                }
            }
            let _ = sender.send(ProcessLine::Eof);
        });
        let stderr = Arc::new(Mutex::new(String::new()));
        let captured = Arc::clone(&stderr);
        let stderr_thread = thread::spawn(move || {
            for line in BufReader::new(stderr_pipe)
                .lines()
                .map_while(std::result::Result::ok)
            {
                let mut value = captured.lock().expect("Kraken stderr lock");
                if value.len() < 64 * 1024 {
                    value.push_str(&line);
                    value.push('\n');
                }
            }
        });
        let name = format!(
            "{identity}:layout=blla:batch={}:threads={}",
            options.runtime_batch_size,
            options.threads.max(1),
        );
        let mut runtime = Self {
            child,
            input,
            output,
            stdout_thread: Some(stdout_thread),
            stderr_thread: Some(stderr_thread),
            stderr,
            timeout: Duration::from_secs(options.timeout_seconds),
            request_id: 0,
            identity,
            name,
        };
        let startup = runtime.request(json!({"op": "ocr_batch", "images": []}))?;
        if !startup.as_array().is_some_and(Vec::is_empty) {
            return Err(Error::Message(
                "Kraken runtime returned an invalid startup response".to_owned(),
            ));
        }
        Ok(runtime)
    }

    pub(crate) fn identity(&self) -> &str {
        &self.identity
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn recognize(
        &mut self,
        images: &[GrayImage],
    ) -> Result<Vec<KrakenImageDiagnostics>> {
        if images.is_empty() {
            return Ok(Vec::new());
        }
        let temporary = TemporaryImages::new(images)?;
        let started = Instant::now();
        let value = self.request(json!({
            "op": "ocr_batch",
            "images": temporary.paths,
        }))?;
        let pages: Vec<RuntimePage> = serde_json::from_value(value)?;
        if pages.len() != images.len() {
            return Err(Error::Message(format!(
                "Kraken runtime returned {} pages for {} images",
                pages.len(),
                images.len()
            )));
        }
        let seconds = started.elapsed().as_secs_f64() / images.len() as f64;
        pages
            .into_iter()
            .zip(images)
            .map(|(page, image)| runtime_page(page, image, seconds))
            .collect()
    }

    fn request(&mut self, mut request: Value) -> Result<Value> {
        self.request_id += 1;
        request["id"] = self.request_id.into();
        serde_json::to_writer(&mut self.input, &request)?;
        self.input
            .write_all(b"\n")
            .and_then(|_| self.input.flush())
            .map_err(|source| Error::Message(format!("Kraken runtime input failed: {source}")))?;
        let line = match self.output.recv_timeout(self.timeout) {
            Ok(ProcessLine::Value(line)) => line,
            Ok(ProcessLine::Error(error)) => {
                return Err(Error::Message(format!(
                    "Kraken runtime output failed: {error}"
                )))
            }
            Ok(ProcessLine::Eof) | Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Error::Message(format!(
                    "Kraken runtime exited before replying{}",
                    self.stderr_detail()
                )))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = self.child.kill();
                return Err(Error::Message(format!(
                    "Kraken runtime timed out after {} seconds{}",
                    self.timeout.as_secs(),
                    self.stderr_detail()
                )));
            }
        };
        let response: RuntimeResponse = serde_json::from_str(&line).map_err(|error| {
            Error::Message(format!("Kraken runtime returned invalid JSON: {error}"))
        })?;
        if response.id != Some(self.request_id) {
            return Err(Error::Message(format!(
                "Kraken runtime response ID mismatch: expected {}, found {:?}",
                self.request_id, response.id
            )));
        }
        if !response.ok {
            let error = response.error.unwrap_or(RuntimeError {
                kind: "RuntimeError".to_owned(),
                message: "unknown error".to_owned(),
            });
            return Err(Error::Message(format!(
                "Kraken runtime {}: {}",
                error.kind, error.message
            )));
        }
        Ok(response.result)
    }

    fn stderr_detail(&self) -> String {
        let value = self.stderr.lock().expect("Kraken stderr lock");
        let value = value.trim();
        if value.is_empty() {
            String::new()
        } else {
            format!(": {}", value.lines().last().unwrap_or(value))
        }
    }
}

impl Drop for BllaRuntime {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(handle) = self.stdout_thread.take() {
            let _ = handle.join();
        }
        if let Some(handle) = self.stderr_thread.take() {
            let _ = handle.join();
        }
    }
}

fn runtime_page(
    page: RuntimePage,
    image: &GrayImage,
    seconds: f64,
) -> Result<KrakenImageDiagnostics> {
    if page.width != image.width() as usize || page.height != image.height() as usize {
        return Err(Error::Message(format!(
            "Kraken runtime changed page dimensions from {}x{} to {}x{}",
            image.width(),
            image.height(),
            page.width,
            page.height
        )));
    }
    let mut region_types = HashMap::new();
    for (kind, regions) in page.regions {
        for (index, region) in regions.into_iter().enumerate() {
            if polygon_bbox(&region.boundary).is_none()
                || region_types
                    .insert(region.id, (kind.clone(), index + 1))
                    .is_some()
            {
                return Err(Error::Message(
                    "Kraken runtime returned malformed or duplicate regions".to_owned(),
                ));
            }
        }
    }
    let mut lines = Vec::new();
    let mut layout_boxes = Vec::new();
    for (index, line) in page.lines.into_iter().enumerate() {
        let text = line.text.trim().to_owned();
        if text.is_empty() {
            continue;
        }
        let bbox = polygon_bbox(&line.boundary)
            .or_else(|| character_bbox(&line.characters))
            .or_else(|| polygon_bbox(&line.baseline))
            .ok_or_else(|| {
                Error::Message("Kraken runtime returned a text line without geometry".to_owned())
            })?;
        let bbox = clamp_bbox(bbox, page.width, page.height)?;
        let region_id = line.regions.first().cloned().unwrap_or_default();
        let (region_type, block_index) = region_types
            .get(&region_id)
            .cloned()
            .unwrap_or_else(|| ("unknown".to_owned(), index + 1));
        let words = runtime_words(&text, &line.characters);
        layout_boxes.push([
            bbox[0].floor() as usize,
            bbox[1].floor() as usize,
            bbox[2].ceil() as usize,
            bbox[3].ceil() as usize,
        ]);
        lines.push(OcrLine {
            text,
            bbox,
            confidence: if line.confidence.is_finite() {
                line.confidence.clamp(0.0, 1.0)
            } else {
                0.0
            },
            baseline: line.baseline,
            boundary: line.boundary,
            words,
            region_id,
            region_type,
            block_index,
        });
    }
    Ok(KrakenImageDiagnostics {
        lines,
        layout_boxes,
        layout_seconds: 0.0,
        recognition_seconds: seconds,
    })
}

fn runtime_words(text: &str, characters: &[RuntimeCharacter]) -> Vec<OcrWord> {
    let mut glyphs = Vec::new();
    let mut offset = 0;
    let mut reconstructed = String::new();
    for character in characters {
        let count = character.text.chars().count();
        let Some(bbox) = polygon_bbox(&character.polygon) else {
            return Vec::new();
        };
        reconstructed.push_str(&character.text);
        glyphs.push((offset, offset + count, bbox));
        offset += count;
    }
    if reconstructed != text {
        return Vec::new();
    }
    let mut output = Vec::new();
    let mut start_byte = None;
    for (byte, character) in text
        .char_indices()
        .chain(std::iter::once((text.len(), ' ')))
    {
        if character.is_whitespace() {
            if let Some(begin) = start_byte.take() {
                let start = text[..begin].chars().count();
                let end = text[..byte].chars().count();
                let boxes = glyphs
                    .iter()
                    .filter(|(glyph_start, glyph_end, _)| *glyph_start < end && start < *glyph_end)
                    .map(|(_, _, bbox)| *bbox)
                    .collect::<Vec<_>>();
                let Some(bbox) = union_bbox(&boxes) else {
                    return Vec::new();
                };
                output.push(OcrWord {
                    text: text[begin..byte].to_owned(),
                    bbox,
                    start,
                    end,
                });
            }
        } else if start_byte.is_none() {
            start_byte = Some(byte);
        }
    }
    output
}

fn character_bbox(characters: &[RuntimeCharacter]) -> Option<[f64; 4]> {
    let boxes = characters
        .iter()
        .filter_map(|character| polygon_bbox(&character.polygon))
        .collect::<Vec<_>>();
    union_bbox(&boxes)
}

fn polygon_bbox(points: &[[f64; 2]]) -> Option<[f64; 4]> {
    let first = *points.first()?;
    if points.len() < 2 || !first.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut bbox = [first[0], first[1], first[0], first[1]];
    for point in &points[1..] {
        if !point.iter().all(|value| value.is_finite()) {
            return None;
        }
        bbox[0] = bbox[0].min(point[0]);
        bbox[1] = bbox[1].min(point[1]);
        bbox[2] = bbox[2].max(point[0]);
        bbox[3] = bbox[3].max(point[1]);
    }
    (bbox[2] > bbox[0] && bbox[3] > bbox[1]).then_some(bbox)
}

fn union_bbox(boxes: &[[f64; 4]]) -> Option<[f64; 4]> {
    let first = *boxes.first()?;
    Some(boxes[1..].iter().fold(first, |mut union, bbox| {
        union[0] = union[0].min(bbox[0]);
        union[1] = union[1].min(bbox[1]);
        union[2] = union[2].max(bbox[2]);
        union[3] = union[3].max(bbox[3]);
        union
    }))
}

fn clamp_bbox(mut bbox: [f64; 4], width: usize, height: usize) -> Result<[f64; 4]> {
    bbox[0] = bbox[0].clamp(0.0, width as f64);
    bbox[1] = bbox[1].clamp(0.0, height as f64);
    bbox[2] = bbox[2].clamp(0.0, width as f64);
    bbox[3] = bbox[3].clamp(0.0, height as f64);
    if bbox[2] <= bbox[0] || bbox[3] <= bbox[1] {
        return Err(Error::Message(
            "Kraken runtime returned out-of-page line geometry".to_owned(),
        ));
    }
    Ok(bbox)
}

struct VerifiedPack {
    root: PathBuf,
    identity: String,
}

fn verified_pack(path: Option<PathBuf>, kind: &str, label: &str) -> Result<VerifiedPack> {
    let path = path.ok_or_else(|| Error::Message(format!("{label} is required")))?;
    let root = fs::canonicalize(&path).map_err(|source| Error::io(&path, source))?;
    if !root.is_dir() {
        return Err(Error::Message(format!(
            "{label} is not a directory: {}",
            path.display()
        )));
    }
    let manifest_path = root.join("manifest.json");
    let manifest: Value = serde_json::from_reader(BufReader::new(
        fs::File::open(&manifest_path).map_err(|source| Error::io(&manifest_path, source))?,
    ))?;
    if manifest.get("format").and_then(Value::as_str) != Some("kraken-lite-model/1")
        || manifest.get("kind").and_then(Value::as_str) != Some(kind)
    {
        return Err(Error::Message(format!(
            "{label} has an unsupported format or kind"
        )));
    }
    let model = verified_member(&root, &manifest, "model", "model graph")?;
    let codec = manifest
        .get("codec")
        .and_then(Value::as_object)
        .and_then(|value| value.get("file"))
        .map(|_| verified_member(&root, &manifest, "codec", "codec"))
        .transpose()?;
    let codec_identity = codec
        .as_ref()
        .map(|path| sha256_file(path))
        .transpose()?
        .map(|hash| format!(":codec={hash}"))
        .unwrap_or_default();
    let identity = format!(
        "manifest={}:model={}{}",
        sha256_file(&manifest_path)?,
        sha256_file(&model)?,
        codec_identity,
    );
    Ok(VerifiedPack { root, identity })
}

fn verified_member(root: &Path, manifest: &Value, key: &str, label: &str) -> Result<PathBuf> {
    let spec = manifest
        .get(key)
        .and_then(Value::as_object)
        .ok_or_else(|| Error::Message(format!("Kraken manifest has no {key} object")))?;
    let relative = Path::new(
        spec.get("file")
            .and_then(Value::as_str)
            .ok_or_else(|| Error::Message(format!("Kraken manifest {key}.file is required")))?,
    );
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(Error::Message(format!(
            "Kraken {label} path must stay inside its model pack"
        )));
    }
    let path = canonical_file(&root.join(relative), label)?;
    if !path.starts_with(root) {
        return Err(Error::Message(format!(
            "Kraken {label} path escapes its model pack"
        )));
    }
    if let Some(expected) = spec.get("sha256").and_then(Value::as_str) {
        let actual = sha256_file(&path)?;
        if !actual.eq_ignore_ascii_case(expected) {
            return Err(Error::Message(format!(
                "Kraken {label} hash mismatch: expected {expected}, found {actual}"
            )));
        }
    }
    Ok(path)
}

fn required_file(value: &Option<PathBuf>, variable: &str, label: &str) -> Result<PathBuf> {
    let path = value
        .clone()
        .or_else(|| std::env::var_os(variable).map(PathBuf::from))
        .ok_or_else(|| Error::Message(format!("{label} is required")))?;
    canonical_file(&path, label)
}

fn resolve_python(value: Option<&Path>) -> Result<PathBuf> {
    let command = value
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os("LEGALPDF_KRAKEN_PYTHON").map(PathBuf::from))
        .unwrap_or_else(|| PathBuf::from("python"));
    let output = small_output(
        &command,
        [
            OsString::from("-c"),
            OsString::from("import sys; print(sys.executable)"),
        ],
        None,
    )?;
    let path = PathBuf::from(output.trim());
    canonical_file(&path, "Kraken Python executable")
}

fn python_environment(python: &Path, python_path: &OsStr) -> Result<String> {
    small_output(
        python,
        [
            OsString::from("-c"),
            OsString::from(
                "import json,sys,cv2,numpy,PIL,onnxruntime; print(json.dumps({'python':sys.version,'cv2':cv2.__version__,'numpy':numpy.__version__,'pillow':PIL.__version__,'onnxruntime':onnxruntime.__version__,'providers':onnxruntime.get_available_providers()},sort_keys=True))",
            ),
        ],
        Some(python_path),
    )
}

fn small_output(
    command: &Path,
    arguments: impl IntoIterator<Item = OsString>,
    python_path: Option<&OsStr>,
) -> Result<String> {
    let mut process = Command::new(command);
    process.args(arguments).stdin(Stdio::null());
    if let Some(value) = python_path {
        process.env("PYTHONPATH", value);
    }
    hide_window(&mut process);
    let output = process
        .output()
        .map_err(|source| Error::io(command, source))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(Error::Message(format!(
            "{} failed: {}",
            command.display(),
            detail.lines().next().unwrap_or("no error output")
        )));
    }
    String::from_utf8(output.stdout).map_err(|error| {
        Error::Message(format!(
            "{} returned invalid UTF-8: {error}",
            command.display()
        ))
    })
}

fn python_path(wheel: &Path) -> Result<OsString> {
    let mut paths = vec![wheel.to_path_buf()];
    if let Some(existing) = std::env::var_os("PYTHONPATH") {
        paths.extend(std::env::split_paths(&existing));
    }
    std::env::join_paths(paths)
        .map_err(|error| Error::Message(format!("could not construct Kraken PYTHONPATH: {error}")))
}

fn runtime_device(backend: KrakenBackend, device: Option<&str>) -> Result<String> {
    let device = backend.normalized_device(device)?;
    match backend {
        KrakenBackend::Cpu => Ok("cpu".to_owned()),
        KrakenBackend::Cuda => Ok(format!("cuda:{device}")),
        KrakenBackend::TensorRt => Ok(format!("tensorrt:{device}")),
        KrakenBackend::DirectMl => Ok(format!("directml:{device}")),
        KrakenBackend::OpenVino if device == "default" => Ok("openvino".to_owned()),
        KrakenBackend::OpenVino => Err(Error::Message(
            "the established Kraken runtime does not accept an OpenVINO subdevice".to_owned(),
        )),
        KrakenBackend::OneDnn => Ok("auto".to_owned()),
    }
}

fn sha256_bytes(value: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(value))
}

fn hide_window(command: &mut Command) {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x0800_0000);
    }
}

struct TemporaryImages {
    root: PathBuf,
    paths: Vec<PathBuf>,
}

impl TemporaryImages {
    fn new(images: &[GrayImage]) -> Result<Self> {
        let sequence = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("legalpdf-kraken-{}-{sequence}", std::process::id()));
        fs::create_dir(&root).map_err(|source| Error::io(&root, source))?;
        let mut temporary = Self {
            root,
            paths: Vec::with_capacity(images.len()),
        };
        for (index, image) in images.iter().enumerate() {
            let path = temporary.root.join(format!("p{index:04}.png"));
            image
                .save_with_format(&path, ImageFormat::Png)
                .map_err(|error| {
                    Error::Message(format!("could not encode {}: {error}", path.display()))
                })?;
            temporary.paths.push(path);
        }
        Ok(temporary)
    }
}

impl Drop for TemporaryImages {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_page_preserves_reading_order_words_and_regions() {
        let page = RuntimePage {
            width: 100,
            height: 80,
            regions: BTreeMap::from([(
                "text".to_owned(),
                vec![RuntimeRegion {
                    id: "region-1".to_owned(),
                    boundary: vec![[5.0, 5.0], [95.0, 5.0], [95.0, 70.0], [5.0, 70.0]],
                }],
            )]),
            lines: vec![RuntimeLine {
                text: "A B".to_owned(),
                confidence: 0.9,
                baseline: vec![[10.0, 30.0], [80.0, 30.0]],
                boundary: vec![[8.0, 15.0], [82.0, 15.0], [82.0, 35.0], [8.0, 35.0]],
                characters: vec![
                    RuntimeCharacter {
                        text: "A".to_owned(),
                        polygon: vec![[10.0, 16.0], [20.0, 16.0], [20.0, 34.0], [10.0, 34.0]],
                    },
                    RuntimeCharacter {
                        text: " ".to_owned(),
                        polygon: vec![[21.0, 16.0], [25.0, 16.0], [25.0, 34.0], [21.0, 34.0]],
                    },
                    RuntimeCharacter {
                        text: "B".to_owned(),
                        polygon: vec![[26.0, 16.0], [36.0, 16.0], [36.0, 34.0], [26.0, 34.0]],
                    },
                ],
                regions: vec!["region-1".to_owned()],
            }],
        };
        let result = runtime_page(page, &GrayImage::new(100, 80), 1.0).unwrap();
        assert_eq!(result.lines[0].text, "A B");
        assert_eq!(result.lines[0].region_type, "text");
        assert_eq!(
            result.lines[0]
                .words
                .iter()
                .map(|word| word.text.as_str())
                .collect::<Vec<_>>(),
            ["A", "B"]
        );
        assert_eq!(result.layout_boxes, [[8, 15, 82, 35]]);
    }

    #[test]
    fn runtime_device_is_explicit_and_fail_closed() {
        assert_eq!(runtime_device(KrakenBackend::Cpu, None).unwrap(), "cpu");
        assert_eq!(
            runtime_device(KrakenBackend::Cuda, Some("2")).unwrap(),
            "cuda:2"
        );
        assert!(runtime_device(KrakenBackend::OpenVino, Some("GPU.1")).is_err());
    }
}
