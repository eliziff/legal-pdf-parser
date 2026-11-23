# Legal PDF Parser

Local PDF extraction, OCR routing and exact legal-document navigation. Digital-born
pages use native extraction; scanned pages require an explicitly enabled OCR profile
and its runtime assets. Outputs preserve physical pages, bounding geometry, reading
order and source witnesses, with content-addressed caching per stage.

## Performance

Dev OCR and document-structure evaluation (`wise_a075` WiSE-FT α=0.75 blend of stock
Indra and legal fine-tuning, retaining the full 1,732-token charset):

| Dev benchmark (`wise_a075`) | Metric | Value |
| --- | --- | ---: |
| OCR character error rate | CER | **2.69%** |
| OCR word error rate | WER | **5.49%** |
| Line coverage | F1 | **96.26%** |
| Paragraph joins | F1 | **84.00%** |
| Document hierarchy | F1 | **58.80%** |

CPU single-line recognition in `legal-pdf-ocr` uses the pure-Rust AVX2+FMA `IndraNative`
kernel (`width=320`, single-thread vs ONNX Runtime CPU):
- `student8cropids`: **1.85–2.47 ms/line** vs **3.48–3.54 ms/line** ORT (**1.43×–1.88×** idle).
- `wise_a075` (1,732-class 10-block model, under background worker load): **8.24 ms/line** vs **11.42 ms/line** ORT (**1.39×**).

55-page scanned PDF reference (Ryzen 5 2600X + RTX 3080 Ti, pre-`IndraNative` ORT passes):
Balanced DirectML **13.73 p/s** (`2.548%` CER), Quality DirectML **12.10 p/s** (`2.258%` CER), Balanced CPU ORT **8.28 p/s** (`2.546%` CER).

Digital-born PDFs bypass OCR ([`experiments/digitalborn-benchmark`](experiments/digitalborn-benchmark)):

| Native benchmark | Documents | Pages | Throughput | Peak memory |
| --- | ---: | ---: | ---: | ---: |
| Legal PDFs, fresh cache | 8 | 425 | **134.0 pages/s** | **49.3 MiB** |

## Capabilities

- Pure-Rust Indra OCR for scanned material and native extraction for digital-born pages.
- Deterministic headings, paragraphs, footnotes, tables, references, and reading order.
- Pinpoint lookup by page, paragraph, section, or footnote with stable source hashes.
- Content-addressed stage caches tracking Inspector, OCR identity, and structure contracts.

## Build and use

From this repository's root, with the Rust toolchain installed:

```sh
cargo build --release --locked --package legal-pdf-parser --no-default-features --features cli --bin legalpdf
./target/release/legalpdf --version
```

On Windows the binary is `target\release\legalpdf.exe` and requires `rust-lld.exe`
([.cargo/config.toml](.cargo/config.toml)). Feature profiles in [Cargo.toml](Cargo.toml):

| Profile | Capability |
| --- | --- |
| `pdf` | Native PDF extraction, structure and queries; no OCR or layout models |
| `indra` | PDF plus native Indra OCR |
| `ppdoc-openvino` / `ppdoc-full` | PDF with the corresponding layout runtime |
| `full` | Language support, Indra OCR and full layout profile |

Browser OCR lives in [Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr).

## Native OCR runtime

`legal-pdf-ocr` and `legalocr` share the five-tier model manifest with Legal Browser OCR
(`Quality`, `Balanced` default, `Fast`, `Turbo`, `Extreme`). CPU line recognition uses
`IndraNative`; ONNX Runtime handles line detection and optional CUDA/DirectML GPU
recognition.

The Windows x64 bundle (`legal-ocr-runtime-windows-x64.zip`) is published in the
[shared OCR release](https://github.com/eliziff/legal-browser-ocr/releases/latest), or
install assets locally:

```sh
python tools/install_ocr_runtime.py
python tools/build_ocr_runtime.py
legalocr input.pdf --model-pack runtime/indra --runtime runtime/onnx/onnxruntime.dll --output recognized.json
```

Optional orientation model: `tools/install_model.ps1 -Model doc-orientation -OutputDir runtime/indra`.
Override auto backend/worker selection with `--backend cpu|directml|cuda`, `--workers`,
`--threads`, `--layout-workers`, `--batch-size`, or `--width-bucket`. Parser JSON passes
`ocr: {provider: "indra", settings: {model_pack: "...", runtime: "..."}}`.

## Structure and development

[Legal Structure Parser](https://github.com/eliziff/legal-structure-parser) handles
structure and queries. See [AGENTS.md](AGENTS.md) for build/test commands and
[docs/README.md](docs/README.md) for benchmark harnesses and reports. Check stage cache
invalidation with `python tools/check-cache-provenance.py <build-script-build executable>`.

## Credits and license

Recognition and line detection derive from [Indra](https://github.com/datalab-to/indra),
using fine-tuned/pruned weights under the original modified OpenRAIL terms. The PDF
backbone is the MIT-licensed `pdf-inspector`.

[MIT](LICENSE). Third-party components and model assets retain their own licenses and
notices.
