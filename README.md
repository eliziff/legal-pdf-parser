# Legal PDF Parser

Fast local OCR and reliable legal structure on ordinary hardware.

Legal PDF Parser turns legal PDFs into searchable text, pages, paragraphs,
headings, footnotes, references, tables, images, and stable pinpoint locations.
It handles scanned, digital-born, and mixed documents without requiring a
network service or GPU.

The OCR path is built for inexpensive CPUs. On a budget Core i3 laptop, the
full 153-page legal-print benchmark finishes in 56 seconds at the highest
accuracy setting and 41 seconds at the fastest setting. Digital-born documents
usually run at tens of pages per second.

## OCR performance

Kraken Lite uses a legal-specific fine-tune of **CATMuS Print Small**. Its four
tiers use the same model and segmentation; they trade recognition width for
speed. Lower character error rate (CER) is better.

| OCR engine | Laptop CER | Laptop time | Laptop pages/s | Desktop CER | Desktop time | Desktop pages/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kraken Lite Quality | **2.564%** | 56.1 s | 2.728 | **2.705%** | 49.0 s | 3.122 |
| Kraken Lite Balanced | 2.748% | 46.9 s | 3.263 | 2.927% | 42.9 s | 3.569 |
| Kraken Lite Turbo | 3.129% | 44.2 s | 3.465 | 3.439% | 40.4 s | 3.783 |
| Kraken Lite Extreme | 3.773% | **40.8 s** | **3.749** | 4.246% | 38.9 s | 3.933 |
| Native Tesseract 5.4 | 3.829% | 55.5 s | 2.759 | 3.829% | **35.9 s** | **4.264** |

These are CPU-only, cold-process results on the same 153 already-rendered legal
pages and the same ground truth. Timing begins before OCR sessions are created
and ends after all text is written. There is no uncounted warm-up. Native
Tesseract uses persistent C-API sessions - eight on the laptop and twelve on the
desktop - with one thread per session. Kraken uses its automatic all-CPU schedule.

The laptop is an HP 14 with an Intel Core i3-1315U and eight logical CPUs. The
desktop uses an AMD Ryzen 5 2600X and twelve logical CPUs. Tesseract is the
fastest option on the desktop; Kraken Quality is the more accurate option on
both machines. The complete reproducible protocol is in
[`experiments/kraken-lite/cpu-benchmark/RESULTS.md`](experiments/kraken-lite/cpu-benchmark/RESULTS.md).

For digital-born legal PDFs, the native extraction path processed a separate
661-page journal set at 71.74 pages/second end to end, with 99.36% pairwise
reading-order accuracy and 0.162% median CER.

## What the parser provides

- Fast, local OCR with Quality, Balanced, Turbo, and Extreme tiers.
- Native extraction for digital-born pages.
- Deterministic pages, paragraphs, headings, footnotes, references, tables,
  images, and reading order.
- Exact lookup by page, paragraph, section, or footnote.
- Stable source hashes, line identities, page bindings, and pinpoint locators.
- A compressed content-addressed cache with atomic writes.

The extraction backbone is the MIT-licensed `pdf-inspector`, extended with
legal structure and exact lookup contracts for applications and LLM tools.

## Segmentation, layout, and routing

These stages have separate jobs:

- **Segmentation finds the text lines.** The fast path uses a compact
  Tesseract-derived line finder. The optional BLLA path uses a heavier learned
  line finder for unusually difficult scans.
- **Recognition reads the characters.** Kraken Lite applies the legal CATMuS
  model to the detected lines. Native Tesseract is also supported.
- **Layout explains the page.** The parser identifies headings, body text,
  footnotes, headers, references, and reading order. An optional local layout
  pack can add semantic regions to difficult pages.

Every document follows one simple route: read embedded text first, keep usable
pages unchanged, OCR pages that need recognition, then derive legal structure
from the resulting source lines. A scanned exhibit inside an otherwise native
PDF therefore does not slow down the rest of the document.

External layout is advisory. It may assign structure to immutable source lines,
but it cannot replace their text. Incomplete or invalid assignments are
rejected.

## Build

```powershell
cargo build --release --locked
.\target\release\legalpdf.exe --help
cargo test --locked
```

## Use

The public command accepts one versioned JSON request and returns one versioned
JSON result:

```powershell
legalpdf contract request.json
```

Inspect a PDF without parsing it or populating the cache:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "inspect",
  "source_pdf": "article.pdf"
}
```

Prepare and cache a document:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "prepare",
  "source_pdf": "article.pdf",
  "ocr": {
    "provider": "kraken-lite",
    "settings": {
      "model": "runtime/kraken/model.onnx",
      "codec": "runtime/kraken/codec.json",
      "runtime": "runtime/onnxruntime.dll",
      "tesseract_library": "runtime/legalpdf_tesseract_layout.dll",
      "tier": "quality"
    }
  }
}
```

The result includes the source hash, parser version, cache key, page count,
OCR provenance, and structural counts. Repeating the same request with the
same parser and provider identity reuses the cache.

Look up exact legal structure from the source PDF:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "structure_lookup",
  "source_pdf": "article.pdf",
  "query": {
    "locator_kind": "footnote",
    "locator": "62",
    "occurrence": 1,
    "context_blocks": 1
  }
}
```

Repeated printed labels are not guessed. The result is `ambiguous` until the
caller supplies an occurrence or page hint. Use `source_doc` for the complete
versioned text and stable-block projection.

## Security properties

- Parsing and OCR require no network access.
- Requests, lookups, process output, and cache writes are bounded.
- Cache identity includes source bytes, parser versions, and OCR/layout runtime
  identity.
- Writes are atomic and content-addressed.
- Layout providers cannot return replacement source text.
- Invalid or incomplete structural assignments fail closed.

## License

The parser and its vendored `pdf-inspector` backbone are MIT licensed.
Third-party components retain their own notices. Kraken Lite uses ONNX Runtime,
CATMuS-derived model assets, and the Tesseract/Leptonica line-layout library
under their respective licenses.
