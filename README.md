# Legal PDF Parser

Local PDF extraction, OCR, and legal document structure parsing in Rust. Digital-born pages extract natively; scanned pages run through Indra OCR. Outputs preserve pages, geometry, reading order, and content hashes.

## Performance

**Scanned legal PDFs**

Laptop (`13th Gen Intel Core i3-1315U`, CPU):

| Engine / preset | Pages/s | CER |
| --- | ---: | ---: |
| Balanced CPU, full-alphabet (12-page scan) | **6.8** | **2.7%** |
| Extreme CPU, compact 64-ch (12-page scan) | **7.2** | **3.3%** |
| Pruned Balanced CPU baseline (55-page scan) | 3.9 | 2.5% |

Desktop (`Ryzen 5 2600X + RTX 3080 Ti`, 55-page scan baselines):

| Engine / preset | Pages/s | CER |
| --- | ---: | ---: |
| Pruned Balanced GPU DirectML FP16 | **13.7** | **2.5%** |
| Pruned Quality GPU DirectML FP16 | **12.1** | **2.3%** |
| Pruned Balanced CPU baseline | **8.3** | **2.5%** |
| Stock Indra CPU (browser baseline) | 6.9 | 3.4% |
| Tesseract CPU (browser baseline, 6 workers) | 2.3 | 2.3% |

Across our broader scanned legal dev set, the full-alphabet blended model averages **2.7% CER**.

**Digital-born legal PDFs** ([`experiments/digitalborn-benchmark`](experiments/digitalborn-benchmark)): **134.0 pages/s**, **49.3 MiB** peak memory (8 PDFs, 425 pages, no OCR).

## Build & run

Requires Rust (`target\release\legalpdf.exe` on Windows uses `rust-lld.exe`, see [.cargo/config.toml](.cargo/config.toml)):

```sh
cargo build --release --locked --package legal-pdf-parser --no-default-features --features cli --bin legalpdf
```

Profiles in [Cargo.toml](Cargo.toml): `pdf` (native PDFs only), `indra` (+ Indra OCR), `ppdoc-openvino` / `ppdoc-full` (+ layout runtime), `full` (all features).

Install OCR assets from the [shared OCR release](https://github.com/eliziff/legal-browser-ocr/releases/latest) or locally and run `legalocr`:

```sh
python tools/install_ocr_runtime.py
legalocr input.pdf --model-pack runtime/indra --runtime runtime/onnx/onnxruntime.dll --output recognized.json
```

Companion repos: [Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr) and [Legal Structure Parser](https://github.com/eliziff/legal-structure-parser). See [AGENTS.md](AGENTS.md) for dev commands.

## License

Code is [MIT](LICENSE) (built on MIT `pdf-inspector`). Recognition and line detection weights derive from [Indra](https://github.com/datalab-to/indra) under its modified AI PUBS Open RAIL-M terms.
