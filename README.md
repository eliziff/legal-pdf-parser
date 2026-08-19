# Legal PDF Parser

Legal PDF Parser is a local-first Rust engine that turns legal PDFs into
searchable text and legal structure: pages, paragraphs, headings, footnotes,
references, tables, images, and stable pinpoint locations.

The extraction backbone is the vendored MIT-licensed `pdf-inspector`, extended
to return usable legal structure for other legal applications,
including use by LLMs.

It is designed for two common cases:

- Digital-born PDFs are read directly, usually at tens of pages per second.
- Scanned or weak pages are routed selectively through OCR instead of forcing
  the whole document through a slower recognition pipeline.

Parsing, OCR, footnote pairing, and structural lookup run locally. No model or
network service is required. Optional layout providers sit behind a bounded
adapter contract and cannot replace the source text.

## Performance

On a 661-page digital-born legal-journal benchmark, the native
engine processed 71.74 pages/second end to end. Pairwise reading-order accuracy
was 99.36%; median character error rate was 0.162%.

For scanned pages, the custom Kraken Lite runtime uses a legal-domain fine-tune
of **CATMuS Print Small**. The four recognition tiers trade a small amount of
accuracy for speed. Lower CER is better.

| OCR tier | Character error rate | Pages/second |
| --- | ---: | ---: |
| Kraken Lite Quality | 2.582% | 1.548 |
| Kraken Lite Balanced | 2.763% | 1.672 |
| Kraken Lite Turbo | 3.163% | 1.788 |
| Kraken Lite Extreme | 3.837% | 1.824 |
| Tesseract.js fast | 4.108% | 0.364 |

These figures use the same 153-page legal-print benchmark and already-rendered
page images. Quality is the median of three complete runs; the other Kraken
tiers are full-corpus runs. On a separate 30-page scanned-court set, Extreme
reached 2.438 pages/second at 3.819% CER, compared with Tesseract.js at 0.413
pages/second and 7.155% CER.

The native runtime has also been measured directly against Tesseract on the
same 153 pages: Kraken Lite Quality reached 1.235 pages/second at 2.820% CER,
while Tesseract reached 0.763 pages/second at 4.115% CER.

## How routing works

The default path is deliberately boring:

1. Read embedded PDF text and geometry.
2. Score each page's usable text.
3. Keep good native pages unchanged.
4. OCR only empty or weak pages.
5. Run the same deterministic legal-structure pass over the resulting lines.

This makes mixed PDFs fast: a document with one scanned exhibit does not pay
the OCR cost on every other page.

### Segmentation and layout, in plain English

Segmentation and layout solve different problems:

- **Segmentation finds text lines.** The fast Kraken path uses a compact
  Tesseract-derived line finder, then recognizes those lines with the legal
  CATMuS model. The optional BLLA path uses a heavier learned line finder for
  unusually difficult scans.
- **Layout explains what the lines mean.** The default native mode infers
  headings, body text, footnotes, headers, references, and reading order from
  the lines themselves. An optional local layout pack can add learned semantic
  regions. A caller may also provide regions through the strict layout-input
  contract for difficult pages.

Provider-supplied layout is advisory. It must assign the engine's immutable
line IDs completely and validly; otherwise it is rejected. It can change
structure, but never glyph text.

## Build

```powershell
cargo build --release --locked
.\target\release\legalpdf.exe --help
cargo test --locked
```

## Basic use

Inspect a PDF without parsing or populating the cache:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "inspect",
  "source_pdf": "article.pdf"
}
```

Prepare a document. The engine validates and reuses its content-addressed,
compressed cache; it does not publish a second document bundle.

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "prepare",
  "source_pdf": "article.pdf"
}
```

```powershell
legalpdf contract request.json
```

The result contains the source hash, parser version, deterministic cache key,
cache-hit state, page summary, and structural counts. Add `"pages": [5]` to a
`prepare` request to bound OCR work to selected physical pages; native
extraction still preserves the document's authoritative page numbering.

Look up a footnote or another exact legal structure from the source PDF:

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

Repeated printed labels are not guessed. Lookup returns `ambiguous` until the
caller supplies an occurrence or page hint. Footnote extraction and pairing
are deterministic and require no language model. Use `"operation":
"source_doc"` for the versioned full-text and stable-block projection.
For a foreground page read, a page lookup may include a bounded `"pages"`
selection covering the requested range and its context; other lookup kinds
reject partial-page preparation.

## OCR

Select OCR through the same request boundary:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "prepare",
  "source_pdf": "scan.pdf",
  "ocr": {
    "provider": "tesseract",
    "settings": { "language": "eng", "dpi": 180, "psm": 3 }
  }
}
```

Use the custom Kraken Lite runtime:

```json
{
  "schema_version": "legalpdf.document-request.v1",
  "operation": "prepare",
  "source_pdf": "scan.pdf",
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

Available tiers are `quality`, `balanced`, `turbo`, and `extreme`. Execution
backends are explicit; unsupported accelerator requests fail rather than
silently changing the route.

The compact native stack consists of the Rust executable, the CATMuS-derived
ONNX model and codec, ONNX Runtime, and a small line-layout library. The same
recognizer can be packaged into the self-contained browser runtime.

## Structure and evidence

The engine derives legal structure from the source lines rather than from
generated prose. Results retain source hashes, page bindings, line identities,
OCR provenance, and stable locators suitable for evidence receipts and deep
links.

Footnote pairing is part of the engine, not an application-side post-process.
It handles repeated note labels explicitly and produces stable pair IDs and
proposition spans.

Applications that need only a few pages can request bounded page work. Full
document structure remains available when cross-page layout or hierarchy is
required.

## Security properties

- Native parsing and OCR require no network access.
- Requests and lookups are bounded; cache writes are atomic and
  content-addressed.
- Cache identity includes source bytes, parser/schema versions, and relevant
  OCR/layout runtime identity.
- A provider cannot return replacement source text through the layout contract.
- Invalid or incomplete provider assignments fail closed.

## License

The engine and its vendored `pdf-inspector` backbone are MIT licensed.
Third-party components retain their own notices. The optional Kraken Lite pack
uses ONNX Runtime, CATMuS-derived model assets, and the Tesseract/Leptonica
layout library under their respective licenses.
