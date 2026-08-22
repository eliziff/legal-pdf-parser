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

These are end-to-end results on the a sample of 153 scanned legal pages.
The full reproducible receipt is in
[`experiments/kraken-lite/cpu-benchmark/RESULTS.md`](experiments/kraken-lite/cpu-benchmark/RESULTS.md).

Digital-born PDFs avoid OCR. The native path processed a separate 156-page
legal PDF at **70.8 pages/second**.

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
cargo build --release --locked --package legal-structure --no-default-features --bin legal-structure-native
cargo build --release --locked --package legal-structure --no-default-features --features structure-inference --bin legal-structure
cargo build --release --locked --package legal-pdf-parser --no-default-features --features pdf --bin legalpdf
legalpdf contract request.json
```

The native structure host accepts provider claims without shipping raw-text
structure inference or regex. The structure-inference host adds the provider-neutral raw detector.
The `pdf` parser profile ships neither OCR nor layout models; use `kraken`,
`ppdoc-openvino`, `ppdoc-full`, or `full` only when that capability and its
separate runtime/model pack are required. The root package has no default
features, so every shipped artifact declares its capabilities explicitly.

The versioned contract supports `inspect`, `prepare`, `source_doc`, and
`structure_lookup`.

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
