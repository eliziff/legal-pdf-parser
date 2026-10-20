# Legal PDF Parser

Local PDF extraction, OCR routing and exact legal-document navigation. Digital-born
pages use native extraction; scanned pages require an explicitly enabled OCR
profile and its runtime assets. Results retain physical pages, geometry, reading
order and source witnesses, with content-addressed reuse of completed work.

## Performance

Native OCR now uses the five-tier Indra model pack shared with
[Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr). Its browser
README records browser CPU comparisons. The native command has also been measured
on the same 55-page image-only PDF and human transcripts (145,475 normalized
characters), on a Ryzen 5 2600X desktop with an RTX 3080 Ti:

| Native OCR configuration | Pages/s | Character error rate |
| --- | ---: | ---: |
| Balanced, CPU | **8.28** | **2.546%** |
| Balanced, automatic (DirectML GPU) | **13.73** | **2.548%** |
| Quality, automatic (DirectML GPU) | **12.10** | **2.258%** |

These are medians of three warm passes, including native PDF rendering, line
detection, crop preparation and recognition. CLI startup, structure analysis and
application cache writes are excluded. Automatic selection checks provider
availability and hardware capabilities. Fresh-process initialization took
**0.80 seconds on GPU** and **0.24 seconds on CPU**. The first full 55-page
document took **5.00 seconds including GPU initialization**. CPU and GPU text,
confidence and geometry match the frozen outputs exactly. Rendering uses spare
CPU capacity with a shared parsed PDF and bounded page windows.
The full Beaver preparation path also performs structure analysis and writes
application caches. Overlapping its existing separator scan with OCR raised
Balanced from 5.34 to 6.44 pages/s on DirectML and from 4.97 to 5.77 on CPU in
paired single warm passes, preserving recognized text and word geometry exactly.
The same examined test partition was used for browser experiments; this is not a
new holdout or hand-corrected geometry evaluation. Historical Kraken measurements
remain in the original experiment receipts.

Digital-born PDFs avoid OCR:

| Native benchmark | Documents | Pages | Throughput | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| Legal PDFs, fresh cache | 8 | 425 | **134.0 pages/s** | **49.3 MiB** |

This measures process launch, extraction, structure, page queries, and JSON
serialization. Three isolated runs per document produced identical output.
The fixed corpus and runner live in
[`experiments/digitalborn-benchmark`](experiments/digitalborn-benchmark).

## Capabilities

- Fast local OCR for scanned legal material.
- Native extraction for digital-born pages.
- Deterministic headings, paragraphs, footnotes, tables, references, and
  reading order.
- Exact lookup by page, paragraph, section, or footnote.
- Stable source hashes and pinpoint locators for applications and agent tools.
- A compressed content-addressed cache for immediate repeat access.

PDF cache identities follow each stage's inputs. Native extraction tracks the
Inspector, extraction code and page contracts; recognized pages track OCR code,
provider/model/settings identity, source bytes and requested page geometry.
Final documents additionally track structure/projection code and the OCR/layout
stages actually used. Edits to `legal-pdf-language`'s DOCX code preserve PDF cache
entries; structure or layout edits preserve raw extraction and recognition.
Existing entries with the old aggregate identity miss once and expire through
normal bounded cache pruning.

## Build and use

From this repository's root, with the Rust toolchain installed:

```sh
cargo build --release --locked --package legal-pdf-parser --no-default-features --features cli --bin legalpdf
./target/release/legalpdf --version
```

On Windows, the executable is `target\release\legalpdf.exe`; the configured
`rust-lld.exe` linker must be available. `cargo build` does not install the command
on PATH. Build/cache settings are in [.cargo/config.toml](.cargo/config.toml).

No features are enabled by default. [Cargo.toml](Cargo.toml) defines the profiles:

| Profile | Capability |
| --- | --- |
| `pdf` | Native PDF extraction, structure and queries; no OCR or layout models |
| `indra` | PDF plus native Indra OCR |
| `ppdoc-openvino` / `ppdoc-full` | PDF with the corresponding layout runtime |
| `full` | Language support, Indra OCR and the full layout profile |

Enabling a feature does not provide its separately licensed model/runtime pack.
A source-only build needs these assets to run OCR. Browser OCR
and its single-file HTML application live in
[Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr), not this crate.

## Native OCR runtime

The `legal-pdf-ocr` crate runs directly in Rust, shared by the PDF parser and the
standalone `legalocr` command. It requires no browser, Node.js or Python for
inference. Quality, Balanced (default), Fast, Turbo and Extreme use the same
model manifest as the browser package. Extreme trades accuracy for speed.

The Windows x64 executable, models and CPU/DirectML libraries are distributed as
`legal-ocr-runtime-windows-x64.zip` in the
[latest shared OCR release](https://github.com/eliziff/legal-browser-ocr/releases/latest).
Extract the archive and run `legalocr input.pdf --output recognized.json`.
The executable finds its packaged assets beside itself.

On Windows x64, install the released model pack and official ONNX Runtime 1.22
DirectML binaries (which also include CPU execution):

```sh
python tools/install_ocr_runtime.py
python tools/build_ocr_runtime.py
legalocr input.pdf --model-pack runtime/indra --runtime runtime/onnx/onnxruntime.dll --output recognized.json
```

The installer resolves the latest shared OCR release and verifies its archive digest
and model inventory. Pass `--models PATH` for a downloaded `legal-ocr-models.tar.gz`. Other platforms
can use `--models-only` with their own compatible CPU/CUDA ORT library. The
optional orientation classifier is installed with the existing helper:
`tools/install_model.ps1 -Model doc-orientation -OutputDir runtime/indra`.

An omitted backend uses supported GPU providers with CPU fallback. On Windows,
DirectML uses the operating system's high-performance adapter order, retaining
CPU as the default for integrated and FP16-emulating adapters. Worker limits
follow available CPU capacity. Detector and recognizer pools initialize together,
and successfully opened GPU sessions are retained.
Explicit backend/device and worker settings override these defaults.

`--backend cpu|directml|cuda`, `--workers`, `--threads`, `--layout-workers`,
`--batch-size` and `--width-bucket` override automatic choices. JSON output
records the selected plan, exact model/runtime identity, startup and processing
time. Parser JSON uses `ocr: {provider: "indra", settings: {model_pack: "...",
runtime: "..."}}`; omit `settings.backend` for automatic selection. Native
orientation, word geometry, partial-page regions and bounded page streaming are
retained. Tesseract layout remains an explicit alternative to learned detection.

## Structure and queries

[Legal Structure Parser](https://github.com/eliziff/legal-structure-parser)
provides paragraph, section, citation and text queries. PDF extraction adds
physical pages, geometry, reading order and OCR evidence.

Numbering inside quotations is distinct from the document's own paragraphs.
Automatic constituent-document detection in combined records is not complete;
physical-page and exact-text lookup remain available when boundaries are
ambiguous.

## Development

See [AGENTS.md](AGENTS.md) for build and test commands, and the
[documentation index](docs/README.md) for benchmark runners and recorded results.
The published performance figures describe their recorded corpus, hardware and
revision; they are not measurements of every release.

`python tools/check-cache-provenance.py <build-script-build executable>` checks
stage invalidation against invented source-file mutations without compiling or
running a parser. Reuse the build-script executable from the latest ordinary
build's `target/{debug,release}/build/legal-pdf-parser-*/` directory; Beaver uses
its canonical `native/legal-structure-node/target/`. Use the executable compiled
from the candidate's `build.rs`; the check never starts a build itself.

## Credits and license

Recognition and learned line detection derive from
[Indra](https://github.com/datalab-to/indra), using fine-tuned/pruned recognition
weights. The model pack carries the original modified OpenRAIL model terms,
source revision, modification notice and checksums. These weights have separate
use restrictions; the application license does not replace them.
The PDF extraction backbone is the MIT-licensed `pdf-inspector`.

[MIT](LICENSE). Third-party components and model assets retain their own licenses
and notices.
