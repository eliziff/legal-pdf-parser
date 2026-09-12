# Legal PDF Parser

Local PDF extraction, OCR routing and exact legal-document navigation. Digital-born
pages use native extraction; scanned pages require an explicitly enabled OCR
profile and its runtime assets. Results retain physical pages, geometry, reading
order and source witnesses, with content-addressed reuse of completed work.

## Performance

| OCR profile | Core i3 laptop | RTX 3080 Ti desktop | Character error rate |
| --- | ---: | ---: | ---: |
| Quality | 2.18 pages/s | **6.26 pages/s** | **2.57%** |
| Turbo | 2.50 pages/s | **6.59 pages/s** | 3.14% |
| Native Tesseract 5.4 | **2.76 pages/s** | — | 3.83% |

These are end-to-end results on a sample of 153 scanned legal pages.
The full reproducible receipt is in
[`experiments/kraken-lite/cpu-benchmark/RESULTS.md`](experiments/kraken-lite/cpu-benchmark/RESULTS.md).

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

## Build and use

From this repository's root, with the Rust toolchain installed:

```sh
cargo build --release --locked --package legal-pdf-parser --no-default-features --features pdf --bin legalpdf
./target/release/legalpdf --version
```

On Windows, the executable is `target\release\legalpdf.exe`; the configured
`rust-lld.exe` linker must be available. `cargo build` does not install the command
on PATH. Build/cache settings are in [.cargo/config.toml](.cargo/config.toml).

No features are enabled by default. [Cargo.toml](Cargo.toml) defines the profiles:

| Profile | Capability |
| --- | --- |
| `pdf` | Native PDF extraction, structure and queries; no OCR or layout models |
| `kraken` | PDF plus Kraken OCR |
| `ppdoc-openvino` / `ppdoc-full` | PDF with the corresponding layout runtime |
| `full` | Language support, Kraken OCR and the full layout profile |

Enabling a feature does not provide its separately licensed model/runtime pack.
A source-only build needs these assets to run OCR. Browser OCR
and its single-file HTML application live in
[Legal Browser OCR](https://github.com/eliziff/legal-browser-ocr), not this crate.

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

## Credits and license

Recognition uses a legal-domain fine-tune of
[CATMuS Print Small](https://zenodo.org/records/10602357), trained with
[Kraken](https://kraken.re/) and the [CATMuS project](https://huggingface.co/CATMuS).
The PDF extraction backbone is the MIT-licensed `pdf-inspector`.

[MIT](LICENSE). Third-party components and model assets retain their own licenses
and notices.
