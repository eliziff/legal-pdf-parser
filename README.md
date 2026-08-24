# Legal PDF Parser

Fast local OCR and reliable legal structure on consumer hardware.

Legal PDF Parser turns scanned, digital-born, and mixed PDFs into searchable
text with stable pages, paragraphs, sections, footnotes, tables, references,
reading order, and pinpoint locators. It runs locally and caches completed work.

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

## Use

```powershell
cargo build --release --locked --package legal-pdf-parser --no-default-features --features pdf --bin legalpdf
legalpdf --version
```

Provider-neutral detection and document queries live in the standalone
[`legal-structure`](https://github.com/eliziff/legal-structure) crate.
The `pdf` parser profile ships neither OCR nor layout models; use `kraken`,
`ppdoc-openvino`, `ppdoc-full`, or `full` only when that capability and its
separate runtime/model pack are required. The root package has no default
features, so every shipped artifact declares its capabilities explicitly.

The Node binding returns one opaque native document that owns the canonical
structure and only the extra PDF evidence required for exact lookups, without
serializing an intermediary.

## Credits

The recognition model is my legal-domain fine-tune of
[CATMuS Print Small](https://zenodo.org/records/10602357), built with the
[Kraken](https://kraken.re/) OCR ecosystem and the work of the
[CATMuS project](https://huggingface.co/CATMuS). This project adds the legal
fine-tune, optimized Rust/ONNX runtime, PDF routing, deterministic legal
structure, exact lookup, and application contract. The PDF extraction backbone
is the MIT-licensed `pdf-inspector`.

## License

Legal PDF Parser is MIT licensed. Third-party components and model assets
retain their own licenses and notices.
