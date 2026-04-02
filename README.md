# pdf-inspector

Fast Rust library for PDF classification and text extraction. Detects whether a PDF is text-based or scanned, extracts text with position awareness, and converts to clean Markdown — all without OCR.

Built by [Firecrawl](https://firecrawl.dev) to handle text-based PDFs locally in under 200ms, skipping expensive OCR services for the ~54% of PDFs that don't need them.

## Features

- **Smart classification** — Detect TextBased, Scanned, ImageBased, or Mixed PDFs in ~10-50ms by sampling content streams. Returns a confidence score (0.0-1.0) and per-page OCR routing.
- **Text extraction** — Position-aware extraction with font info, X/Y coordinates, and automatic multi-column reading order.
- **Markdown conversion** — Headings (H1-H4 via font size ratios), bullet/numbered/letter lists, code blocks (monospace font detection), tables (rectangle-based and heuristic), bold/italic formatting, URL linking, and page breaks.
- **Table detection** — Dual-mode: rectangle-based detection from PDF drawing ops, plus heuristic detection from text alignment. Handles financial tables, footnotes, and continuation tables across pages.
- **CID font support** — ToUnicode CMap decoding for Type0/Identity-H fonts, UTF-16BE, UTF-8, and Latin-1 encodings.
- **Multi-column layout** — Automatic detection of newspaper-style columns, sequential reading order, and RTL text support.
- **Encoding issue detection** — Automatically flags broken font encodings (garbled text, replacement characters) so callers can fall back to OCR.
- **Single document load** — The document is parsed once and shared between detection and extraction, avoiding redundant I/O.
- **Lightweight** — Pure Rust, no ML models, no external services. Single dependency on `lopdf` for PDF parsing.
- **Python bindings** — Use from Python via PyO3. Install with `pip install pdf-inspector` or build from source with `maturin`.

## Quick start

### Python

Install from source (requires Rust toolchain):

```bash
pip install maturin
maturin develop --release
```

Use it:

```python
import pdf_inspector

# Full processing: detect + extract + convert to Markdown
result = pdf_inspector.process_pdf("document.pdf")
print(result.pdf_type)      # "text_based", "scanned", "image_based", "mixed"
print(result.confidence)     # 0.0 - 1.0
print(result.page_count)     # number of pages
print(result.markdown)       # Markdown string or None

# Process specific pages only
result = pdf_inspector.process_pdf("document.pdf", pages=[1, 3, 5])

# Process from bytes (no filesystem needed)
with open("document.pdf", "rb") as f:
    result = pdf_inspector.process_pdf_bytes(f.read())

# Fast detection only (no text extraction)
result = pdf_inspector.detect_pdf("document.pdf")
if result.pdf_type == "text_based":
    print("Can extract locally!")
else:
    print(f"Pages needing OCR: {result.pages_needing_ocr}")

# Plain text extraction
text = pdf_inspector.extract_text("document.pdf")

# Positioned text items with font info
items = pdf_inspector.extract_text_with_positions("document.pdf")
for item in items[:5]:
    print(f"'{item.text}' at ({item.x:.0f}, {item.y:.0f}) size={item.font_size}")
```

#### Python API reference

| Function | Description |
|---|---|
| `process_pdf(path, pages=None)` | Full processing (detect + extract + markdown) |
| `process_pdf_bytes(data, pages=None)` | Full processing from bytes |
| `detect_pdf(path)` | Fast detection only (returns PdfResult) |
| `detect_pdf_bytes(data)` | Fast detection from bytes |
| `classify_pdf(path)` | Lightweight classification (returns PdfClassification) |
| `classify_pdf_bytes(data)` | Lightweight classification from bytes |
| `extract_text(path)` | Plain text extraction |
| `extract_text_bytes(data)` | Plain text extraction from bytes |
| `extract_text_with_positions(path, pages=None)` | Text with X/Y coords and font info |
| `extract_text_with_positions_bytes(data, pages=None)` | Text with positions from bytes |
| `extract_text_in_regions(path, page_regions)` | Extract text in bounding-box regions |
| `extract_text_in_regions_bytes(data, page_regions)` | Region extraction from bytes |

**`PdfResult` fields:** `pdf_type`, `markdown`, `page_count`, `processing_time_ms`, `pages_needing_ocr`, `title`, `confidence`, `is_complex_layout`, `pages_with_tables`, `pages_with_columns`, `has_encoding_issues`

**`PdfClassification` fields:** `pdf_type`, `page_count`, `pages_needing_ocr` (0-indexed), `confidence`

**`TextItem` fields:** `text`, `x`, `y`, `width`, `height`, `font`, `font_size`, `page`, `is_bold`, `is_italic`, `item_type`

**`RegionText` fields:** `text`, `needs_ocr`

**`PageRegionTexts` fields:** `page` (0-indexed), `regions` (list of RegionText)

### Node.js (NAPI)

```bash
npm install @firecrawl/pdf-inspector-js
```

```javascript
import { readFileSync } from 'fs';
import { processPdf, classifyPdf, extractTextInRegions } from '@firecrawl/pdf-inspector-js';

const buffer = readFileSync('document.pdf');

// Full processing
const result = processPdf(buffer);
console.log(result.pdfType);   // "TextBased", "Scanned", "ImageBased", "Mixed"
console.log(result.markdown);  // Markdown string or null

// Lightweight classification
const cls = classifyPdf(buffer);
console.log(cls.pdfType, cls.pagesNeedingOcr);

// Region-based extraction (for hybrid OCR pipelines)
const regions = extractTextInRegions(buffer, [
  { page: 0, regions: [[0, 0, 600, 100]] }
]);
```

#### Node.js API reference

| Function | Description |
|---|---|
| `processPdf(buffer, pages?)` | Full processing (detect + extract + markdown) |
| `detectPdf(buffer)` | Fast detection only (returns PdfResult) |
| `classifyPdf(buffer)` | Lightweight classification (returns PdfClassification) |
| `extractText(buffer)` | Plain text extraction |
| `extractTextWithPositions(buffer, pages?)` | Text with X/Y coords and font info |
| `extractTextInRegions(buffer, pageRegions)` | Extract text in bounding-box regions |

### Rust

Add to your `Cargo.toml`:

```toml
[dependencies]
pdf-inspector = { git = "https://github.com/firecrawl/pdf-inspector" }
```

Detect and extract in one call:

```rust
use pdf_inspector::process_pdf;

let result = process_pdf("document.pdf")?;

println!("Type: {:?}", result.pdf_type);       // TextBased, Scanned, ImageBased, Mixed
println!("Confidence: {:.0}%", result.confidence * 100.0);
println!("Pages: {}", result.page_count);

if let Some(markdown) = &result.markdown {
    println!("{}", markdown);
}
```

Fast metadata-only detection (no text extraction or markdown generation):

```rust
use pdf_inspector::detect_pdf;

let info = detect_pdf("document.pdf")?;

match info.pdf_type {
    pdf_inspector::PdfType::TextBased => {
        // Extract locally — fast and free
    }
    _ => {
        // Route to OCR service
        // info.pages_needing_ocr tells you exactly which pages
    }
}
```

Customize processing with `PdfOptions`:

```rust
use pdf_inspector::{process_pdf_with_options, PdfOptions, ProcessMode, DetectionConfig, ScanStrategy};

// Analyze layout without generating markdown
let result = process_pdf_with_options(
    "document.pdf",
    PdfOptions::new().mode(ProcessMode::Analyze),
)?;

// Full extraction with custom detection strategy
let result = process_pdf_with_options(
    "large.pdf",
    PdfOptions::new().detection(DetectionConfig {
        strategy: ScanStrategy::Sample(5),
        ..Default::default()
    }),
)?;

// Process only specific pages
let result = process_pdf_with_options(
    "document.pdf",
    PdfOptions::new().pages([1, 3, 5]),
)?;
```

Process from a byte buffer (no filesystem needed):

```rust
use pdf_inspector::process_pdf_mem;

let bytes = std::fs::read("document.pdf")?;
let result = process_pdf_mem(&bytes)?;
```

### CLI

```bash
# Convert PDF to Markdown
cargo run --bin pdf2md -- document.pdf

# JSON output (for piping)
cargo run --bin pdf2md -- document.pdf --json

# Raw markdown only (no headers)
cargo run --bin pdf2md -- document.pdf --raw

# Insert page break markers (<!-- Page N -->)
cargo run --bin pdf2md -- document.pdf --pages

# Process only specific pages
cargo run --bin pdf2md -- document.pdf --select-pages 1,3,5-10

# Detection only (no extraction)
cargo run --bin detect-pdf -- document.pdf
cargo run --bin detect-pdf -- document.pdf --json

# Detection + layout analysis (tables, columns)
cargo run --bin detect-pdf -- document.pdf --analyze
cargo run --bin detect-pdf -- document.pdf --analyze --json
```

## Architecture

```
PDF bytes
  │
  ├─► detector         → PdfType (TextBased / Scanned / ImageBased / Mixed)
  │
  └─► extractor
        ├─ fonts        → font widths, encodings
        ├─ content_stream → walk PDF operators → TextItems + PdfRects
        ├─ xobjects     → Form XObject text, image placeholders
        ├─ links        → hyperlinks, AcroForm fields
        └─ layout       → column detection → line grouping → reading order
              │
              ├─► tables
              │     ├─ detect_rects      → rectangle-based tables (union-find)
              │     ├─ detect_heuristic  → alignment-based tables
              │     ├─ grid              → column/row assignment → cells
              │     └─ format            → cells → Markdown table
              │
              └─► markdown
                    ├─ analysis     → font stats, heading tiers
                    ├─ preprocess   → merge headings, drop caps
                    ├─ convert      → line loop + table/image insertion
                    ├─ classify     → captions, lists, code
                    └─ postprocess  → cleanup → final Markdown
```

The document is loaded **once** via `load_document_from_path` / `load_document_from_mem` and shared between the detection and extraction stages, so there's no redundant parsing.

### Project structure

```
src/
  lib.rs                — Public API, PdfOptions builder, convenience functions
  python.rs             — PyO3 Python bindings
  types.rs              — Shared types: TextItem, TextLine, PdfRect, ItemType
  text_utils.rs         — Character/text helpers (CJK, RTL, ligatures, bold/italic)
  process_mode.rs       — ProcessMode enum (DetectOnly, Analyze, Full)
  detector.rs           — Fast PDF type detection without full document load
  glyph_names.rs        — Adobe Glyph List → Unicode mapping
  tounicode.rs          — ToUnicode CMap parsing for CID-encoded text
  extractor/            — Text extraction pipeline
  tables/               — Table detection and formatting
  markdown/             — Markdown conversion and structure detection
  bin/                  — CLI tools (pdf2md, detect_pdf)
```

## How classification works

1. Parse the xref table and page tree (no full object load)
2. Select pages based on `ScanStrategy` (default: all pages with early exit)
3. Look for `Tj`/`TJ` (text operators) and `Do` (image operators) in content streams
4. Classify based on text operator presence across sampled pages

This detects 300+ page PDFs in milliseconds. The result includes `pages_needing_ocr` — a list of specific page numbers that lack text, enabling per-page OCR routing instead of all-or-nothing.

### Scan strategies

| Strategy | Behavior | Best for |
|---|---|---|
| `EarlyExit` (default) | Scan all pages, stop on first non-text page | Pipelines routing TextBased PDFs to fast extraction |
| `Full` | Scan all pages, no early exit | Accurate Mixed vs Scanned classification |
| `Sample(n)` | Sample `n` evenly distributed pages (first, last, middle) | Very large PDFs where speed matters more than precision |
| `Pages(vec)` | Only scan specific 1-indexed page numbers | When the caller knows which pages to check |

## Rust API

### Processing modes

| Mode | What it does | Returns |
|---|---|---|
| `ProcessMode::Full` (default) | Detect + extract + convert to Markdown | Everything populated |
| `ProcessMode::Analyze` | Detect + extract + layout analysis (no Markdown) | `markdown` is `None`, `layout` is populated |
| `ProcessMode::DetectOnly` | Classification only (fastest) | `markdown` is `None`, `layout` is default |

### Functions

| Function | Description |
|---|---|
| `process_pdf(path)` | Full processing with defaults |
| `detect_pdf(path)` | Fast metadata-only detection (no extraction) |
| `process_pdf_with_options(path, options)` | Process with custom `PdfOptions` |
| `process_pdf_mem(bytes)` | Full processing from a byte buffer |
| `detect_pdf_mem(bytes)` | Fast detection from a byte buffer |
| `process_pdf_mem_with_options(bytes, options)` | Process from bytes with custom options |
| `extract_text(path)` | Plain text extraction |
| `extract_text_with_positions(path)` | Text with X/Y coordinates and font info |
| `to_markdown(text, options)` | Convert plain text to Markdown |
| `to_markdown_from_items(items, options)` | Markdown from pre-extracted `TextItem`s |
| `to_markdown_from_items_with_rects(items, options, rects)` | Markdown with rectangle-based table detection |

Low-level detection functions are also available via the `detector` module (`detect_pdf_type`, `detect_pdf_type_with_config`, etc.) for callers who need `PdfTypeResult` instead of `PdfProcessResult`.

### Types

| Type | Description |
|---|---|
| `PdfOptions` | Builder for processing configuration (mode, detection, markdown, page filter) |
| `ProcessMode` | `DetectOnly`, `Analyze`, `Full` |
| `PdfType` | `TextBased`, `Scanned`, `ImageBased`, `Mixed` |
| `PdfProcessResult` | Full result: pdf_type, markdown, page_count, confidence, layout, has_encoding_issues, timing |
| `PdfTypeResult` | Low-level detection result: type, confidence, page count, pages needing OCR |
| `DetectionConfig` | Configuration for detection: scan strategy, thresholds |
| `ScanStrategy` | `EarlyExit`, `Full`, `Sample(n)`, `Pages(vec)` |
| `LayoutComplexity` | Layout analysis: is_complex, pages_with_tables, pages_with_columns |
| `TextItem` | Text with position, font info, and page number |
| `MarkdownOptions` | Configuration for Markdown formatting (page numbers, etc.) |
| `PdfError` | `Io`, `Parse`, `Encrypted`, `InvalidStructure`, `NotAPdf` |

## Markdown output

The converter handles:

| Element | How it's detected |
|---|---|
| Headings (H1-H4) | Font size tiers relative to body text, with 0.5pt clustering |
| Bold/italic | Font name patterns (Bold, Italic, Oblique) |
| Bullet lists | `*`, `-`, `*`, `○`, `●`, `◦` prefixes |
| Numbered lists | `1.`, `1)`, `(1)` patterns |
| Letter lists | `a.`, `a)`, `(a)` patterns |
| Code blocks | Monospace fonts (Courier, Consolas, Monaco, Menlo, Fira Code, JetBrains Mono) and keyword detection |
| Tables | Rectangle-based detection from PDF drawing ops + heuristic detection from text alignment |
| Financial tables | Token splitting for consolidated numeric values |
| Captions | "Figure", "Table", "Source:" prefix detection |
| Sub/superscript | Font size and Y-offset relative to baseline |
| URLs | Converted to Markdown links |
| Hyphenation | Rejoins words broken across lines |
| Page numbers | Filtered from output |
| Drop caps | Large initial letters merged with following text |
| Dot leaders | TOC-style dots collapsed to " ... " |

## Debugging with RUST_LOG

Structured logging via `RUST_LOG` replaces the former debug binaries. Set the environment variable to control which sections emit debug output on stderr:

```bash
# Raw PDF content stream operators (replaces dump_ops)
RUST_LOG=pdf_inspector::extractor::content_stream=trace cargo run --bin pdf2md -- file.pdf > /dev/null

# Font metadata, encodings, ligatures (replaces debug_fonts / debug_ligatures)
RUST_LOG=pdf_inspector::extractor::fonts=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# ToUnicode CMap parsing
RUST_LOG=pdf_inspector::tounicode=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# Text items per page with x/y/width (replaces debug_spaces / debug_pages)
RUST_LOG=pdf_inspector::extractor=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# Column detection and reading order (replaces debug_order)
RUST_LOG=pdf_inspector::extractor::layout=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# Y-gap analysis and paragraph thresholds (replaces debug_ygaps)
RUST_LOG=pdf_inspector::markdown::analysis=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# Table detection
RUST_LOG=pdf_inspector::tables=debug cargo run --bin pdf2md -- file.pdf > /dev/null

# Everything
RUST_LOG=pdf_inspector=debug cargo run --bin pdf2md -- file.pdf > /dev/null
```

## Use case: smart PDF routing

pdf-inspector was built for pipelines that process PDFs at scale. Instead of sending every PDF through OCR:

```
PDF arrives
  → pdf-inspector classifies it (~20ms)
  → TextBased + high confidence?
      YES → extract locally (~150ms), done
      NO  → send to OCR service (2-10s)
```

This saves cost and latency for the majority of PDFs that are already text-based (reports, papers, invoices, legal docs).

## License

MIT
