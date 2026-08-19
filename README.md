# Legal PDF Parser

A local-first native Rust engine for turning legal PDFs into stable pages,
regions, words, paragraphs, semantic sections, footnotes, references, and
paired propositions. Ordinary digital-born PDFs are processed without network
access. Codex is optional and is called only for pages carrying a concrete
structural diagnostic.

The extraction backbone is the vendored MIT-licensed `pdf-inspector`, extended
locally where the complete legal-document contract requires more source
detail. The production executable has no Python runtime, PyMuPDF fallback,
server, accounts, cloud storage, database, vector index, or application
dependency.

## Install

```powershell
cargo build --release --locked
.\target\release\legalpdf.exe --help
```

For native development:

```powershell
cargo test --locked
```

The Python implementation remains in this repository only as the frozen
behavioral oracle and benchmark tooling. To run its checks:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e '.[benchmark]'
.\.venv\Scripts\python -m pytest -q tests
```

The installed production path is the Rust executable above; there is no
compatibility shim between it and the oracle.

## Parse and look up a footnote

```powershell
legalpdf parse article.pdf --output parsed
legalpdf footnote parsed\document.json 62
legalpdf footnote parsed\document.json 1 --occurrence 2 `
  --proposition passage_since_prior_note
```

CLI parsing is cache-free by default. Add `--cache` to use the normal cache or
`--cache-dir <dir>` to use a specific one. In Codex mode, the separate bounded
repair cache remains active.

The explicitly requested `--output` projection writes `document.json` last,
after its referenced collections:

```text
document.json
pages.jsonl
paragraphs.jsonl
sections.jsonl
footnotes.jsonl
diagnostics.jsonl
repairs.jsonl
```

The internal parse cache does not contain those publication sidecars. It keeps
one compressed extraction and one compressed legal document, independently
keyed by the source SHA-256, relevant code, schema, and OCR/layout provider.
An interpretation change can therefore reuse the extraction. Cache files are
atomic, disposable, and access-evicted above 1 GiB; set
`LEGALPDF_CACHE_MAX_BYTES` to choose another byte limit.
Section rows span each heading through the paragraph before the next heading.
Pages expose a printed label only when an existing header/footer line supplies
one unambiguously; native lines carry exact word boxes and line-relative offsets.
The normal cache is under `OpenLegalProducts\LegalData` on Windows and the
normal XDG cache on other platforms; `--cache-dir` overrides it.

Thin consumers that do not retain word geometry may pass `--compact-pages`.
Parsing and every structural algorithm remain unchanged, but `pages.jsonl`
contains only stable page metadata plus each line's reading order and text.
The manifest declares `artifact_profile: compact-source`; this projection is
not a substitute for the default round-trippable engine artifacts.

If a compact consumer later needs highlighting or pinpoint boxes, add only a
compressed geometry sidecar:

```powershell
legalpdf add-geometry article.pdf --document compact\document.json `
  --output compact\geometry
```

The command verifies the source, OCR configuration, exact engine identity, and
compact text; skips all derived-evidence work; and writes complete compressed
page records with a payload hash. `load_geometry_artifacts()` verifies and
loads those exact pages. Geometry is never retained merely because a thin
consumer parsed the document.

The retained Python oracle exposes the equivalent library surface for
differential testing:

```python
from legalpdf import lookup_footnote, parse_pdf, write_artifacts

document = parse_pdf("article.pdf", cache_dir=".legalpdf-cache")
result = lookup_footnote(document, "62")
if result.status == "found":
    print(result.footnote.body)
write_artifacts(document, "parsed")
```

Repeated display labels are not guessed. Lookup returns `ambiguous` until the
caller supplies an occurrence or page hint; every note has a unique pair ID.
The persisted CLI lookup streams `footnotes.jsonl` and, for the selected
context only, `paragraphs.jsonl`; it does not open the PDF or deserialize
`pages.jsonl`.

## Canonical footnote pairing

Footnote and endnote pairing is a mandatory stage of this engine. There is no
second pairer, selector, or simpler fallback. Its implementation, pure helper
closure, and frozen 2,110-entry McGill Guide (10th) reporter and journal
inventory are content-fingerprinted in the deterministic cache key.

The implementation was ported from the author-owned Text-Fidelity-Project at
commit `d8b25257687b3b9aad644dec42cca966b45675ff` and is released here under the
repository's MIT license. The locked-corpus comparison is recorded in
[`docs/canonical-pairer-parity-2026-07-30.md`](docs/canonical-pairer-parity-2026-07-30.md).

## Optional source regions

Region-aware fidelity passes accept a complete line-level region contract from
PPDoc or an MLLM. The provider is not hardcoded: stable line IDs and non-unknown
region labels are the contract. Partial region output fails closed, leaving the
native parser's baseline decisions unchanged.

When that contract is present, the engine snapshots it before native furniture
and ordering passes, then uses the source roles to protect authors, notes, TOCs,
references, and other non-body material. Text-Fidelity body-font inference,
guarded heading demotion, display-heading promotion, outline grammar, and
same-style wrapped-heading continuation run only behind this gate. Region
passes may change structure, never source text.

The native PPDoc provider applies the production region postprocessor ported
from Text-Fidelity-Project commit
`d8b25257687b3b9aad644dec42cca966b45675ff`. It operates once per document,
after model decode and before line assignment, so repeated running heads and
sequenced page numbers have document context. The fixed production contract
also covers conservative block-quote repair, region validity, footnote bands,
and overlap priority. It adds no Python, Paddle, Docker, or runtime dependency;
the postprocessor source is fingerprinted in both engine and extraction cache
identities. Its pinned Python-to-Rust differential gate compares canonical
final regions and every line assignment byte-for-byte; the durable receipt is
in
[`experiments/ppdoc-lite/postprocess-parity/RESULTS.md`](experiments/ppdoc-lite/postprocess-parity/RESULTS.md).

The portable stock pack is installed without Python, Paddle, or Docker:

```powershell
.\tools\install_layout_model.ps1 -OutputDir runtime\layout\heron-int8
```

Its manifest pins the official Docling Heron INT8 ONNX checksum, bilinear
preprocessing, `pixel_values` input name, legal label mapping, and 0.3 score
floor. The same Rust runtime uses OpenVINO on laptop CPUs and ONNX Runtime for
CUDA, TensorRT, or DirectML; an explicitly selected accelerator fails loudly
unless `--ppdoc-cpu-fallback` is supplied. The measured stock-model gate and
selection rationale are in
[`experiments/ppdoc-lite/stock-layout/RESULTS.md`](experiments/ppdoc-lite/stock-layout/RESULTS.md).

The production speed/quality ladder is deliberately pack-based rather than
hardcoded to one model:

| Tier | Selection | Laptop result | Intended use |
|---|---|---:|---|
| Turbo | `MIKE_PDF_LAYOUT_PROVIDER=none` | 51.83 pages/s | Native PDF structure; no learned regions |
| Balanced | `MIKE_PDF_LAYOUT_PROVIDER=local` with Heron INT8 | 1.09 pages/s | Portable CPU layout without Python or Docker |
| Quality | `MIKE_PDF_LAYOUT_PROVIDER=local` with the custom PPDoc pack | Existing quality incumbent | Custom legal ontology on capable CPU/GPU hardware |

Set `LEGALPDF_PPDOC_MODEL_PACK` to select a local pack. Backend choice and
device remain independent, so the quality pack can use OpenVINO CPU or the
existing ONNX Runtime CUDA, TensorRT, and DirectML paths without changing the
Rust orchestration or postprocessor.

For a PPDoc-free vision-provider route, Beaver calls `legalpdf layout-input`,
classifies immutable line IDs plus the rendered page through its existing LLM
provider dispatch, then calls `legalpdf apply-layout`. Rust rejects unknown,
repeated, cross-page, or incomplete line assignments and performs all final
derivation itself; provider output can never replace source text. This route is
opt-in (`MIKE_PDF_LAYOUT_PROVIDER=mllm`) and defaults to `gpt-5.6-luna` when
selected. The retry-parse API also accepts `layout_provider` as `none`, `local`,
or `mllm`, with an optional `layout_model` for any registered vision-capable
provider.

## Selective Codex repair

```powershell
legalpdf parse article.pdf --output repaired --mode codex `
  --model gpt-5.6-luna --effort low
```

or repair existing local artifacts:

```powershell
legalpdf improve article.pdf --document parsed\document.json `
  --output repaired --model gpt-5.6-sol --effort max
```

Model and reasoning effort are caller values, not a hardcoded menu. The bridge
uses `codex exec --ephemeral` because each page repair is a bounded,
content-addressed operation; durable engine caching is more appropriate than an
open-ended chat session. Each request receives r=1 page images and immutable
line IDs. Its response schema contains only region types and line IDs, so the
model cannot return replacement glyph text.

A response is applied only when it:

- Matches the strict JSON schema.
- Targets the requested page.
- Includes every target line exactly once and no context-page lines.
- Uses only supported structural types.

Invalid responses are retried three times. A final failure leaves the local
parse unchanged. Valid responses and their token/latency metadata are cached by
source, context, prompt/schema, model, and effort.

## OCR

Image-only pages return `ocr_required` unless the caller supplies an
`OCRProvider`. To use the optional local Tesseract CLI without adding a Python
dependency:

```powershell
legalpdf parse scan.pdf --output parsed --ocr-provider tesseract
```

Only pages whose embedded-text quality is below the existing threshold are
rendered and sent to Tesseract. Language, render DPI, and page segmentation are
configurable with `--ocr-language`, `--ocr-dpi`, and `--ocr-psm`; those values
and the detected Tesseract version are part of the cache identity. Set
`LEGALPDF_TESSERACT_COMMAND` when the executable is not on `PATH`.

The `kraken` Cargo feature adds the native Kraken-lite path. Tesseract performs
layout only; a legal-domain fine-tune of **CATMuS Print Small** performs
recognition through ONNX Runtime. The same compact runtime can be packaged for
the browser or called natively by Beaver.
The quality default keeps the full model and selects a bounded worker/thread
schedule from available parallelism. `--kraken-low-memory` disables the CPU
arena for lower RAM use, while CUDA, TensorRT, DirectML, OpenVINO, and oneDNN
remain explicit runtime backends.

Beaver invokes the release binary directly. A local native package uses this
ignored runtime layout (paths can instead be supplied with the corresponding
`LEGALPDF_*` environment variables):

```text
runtime/
  kraken/model.onnx
  kraken/codec.json
  onnxruntime.dll
  legalpdf_tesseract_layout.dll
```

When these files are present, Beaver selects Kraken-lite for OCR-required pages
and preserves the complete runtime identity in its content-addressed parse
state. `MIKE_PDF_OCR_PROVIDER=tesseract` remains an explicit fallback;
`LEGALPDF_KRAKEN_LAYOUT=blla` selects the full stock-BLLA route.

For Windows release packaging, `tools/build_tesseract_layout.py` builds the
small layout-only DLL from the same Tesseract/Leptonica source used by the
browser runtime. The loader also accepts a normal Tesseract shared library.
The build recipe copies both upstream license files beside the DLL.

Custom providers can still return text lines and bounding boxes in PDF
coordinates:

```python
from legalpdf.ocr import OCRLine

class MyOCR:
    name = "my-ocr-v1"

    def extract_page(self, pdf_path, page_index, *, width, height):
        return [OCRLine("recognized text", [72, 72, 500, 90], 0.97)]
```

OCR pages then pass through the same region, footnote, proposition, and
artifact pipeline and retain OCR provenance. No OCR executable is required by
the base package.

## Application adapters

`legalpdf.adapters` exports plain payloads without importing consumer code:

- `to_alr_payload(document)` for ALR's `ParsedDocument` constructor fields.
- `to_toa_text_units(document)` for TableOfAuthoritiesMaker's `TextUnit`
  fields.

The consuming application owns the final one-line construction of its native
types. This keeps Mike, ALR, and TableOfAuthoritiesMaker independent of one
another.

ALR compatibility history is recorded in
`docs/alr-compatibility-notes.md`; ALR is not a runtime dependency.

## Bounded DOCX citation linking

Build citation intents without asking the main chat model to reason through
every footnote:

```powershell
legalpdf docx-link-plan brief.docx --output link-plan.json `
  --model gpt-5.6-sol --effort none --strategy auto
```

The worker returns citation text, source type, locator, and an optional exact
support quote. Its schema has no URL field, and any response containing a URL
is rejected. Mike then resolves those compact intents through its existing
A2AJ, CourtListener, UK/public-source, and local journal provider code. Those
providers construct native paragraph/section/page anchors and verified
multi-text directives.

The provider result is a plain `part_id` to URL map. Apply it deterministically:

```powershell
legalpdf docx-apply-links brief.docx --plan link-plan.json `
  --links provider-links.json --output brief-linked.docx
```

The writer verifies the source SHA-256, preserves the original citation text
and run formatting, and adds ordinary external OOXML hyperlinks. It refuses
complex Word runs instead of rewriting uncertain content.

In Mike's account-free local mode this is exposed as the single
`library_link_docx_citations` tool and saves a new Library version. The
specialized worker model is configured with `MIKE_DOCX_LINK_MODEL`,
`MIKE_DOCX_LINK_EFFORT`, and `MIKE_DOCX_LINK_STRATEGY`; it is deliberately not
another normal chat-model option. Supabase/cloud document paths remain
unchanged.

## DOCX-projected degraded-PDF candidate benchmark

Build a heuristic DOCX reference and four PDF profiles from one DOCX:

```powershell
legalpdf-benchmark docx-gold article.docx --output gold.json
legalpdf-benchmark export-docx article.docx --output export
```

Profiles are:

- `native`: normal Microsoft Word export (LibreOffice fallback).
- `print`: tags, bookmarks, form fields, and notes disabled.
- `flattened`: visible searchable PDF content re-emitted as page forms.
- `rasterized`: 144-DPI image-only pages for the OCR boundary.

The exporter's exact version, settings, and every DOCX/PDF hash are recorded.
An actual Microsoft Print to PDF or other external export can be registered
without pretending it was reproducibly automated:

```powershell
legalpdf-benchmark register-export article.docx printed.pdf `
  --output export --profile system-print-to-pdf `
  --exporter 'Microsoft Print to PDF' `
  --settings '{"tags":false,"bookmarks":false}'
```

Build the canonical ALR suite while excluding duplicate `_temp` copies and
deduplicating identical DOCX bytes:

```powershell
legalpdf-benchmark build-docx-corpus `
  --input 'C:\path\to\ALR-Quote-Verifier\data\inputs' `
  --input 'C:\path\to\ALR-Quote-Verifier\data\samples' `
  --output benchmark-output\alr-docx
```

The command creates one extractor-derived candidate reference and four exports
per unique DOCX, plus a resumable `benchmark-manifest.jsonl`. It does not create
human page, region, hierarchy, or proposition gold. Run it:

```powershell
legalpdf-benchmark run benchmark-output\alr-docx\benchmark-manifest.jsonl `
  --output benchmark-output\alr-docx\local-results.jsonl --mode local

legalpdf-benchmark run benchmark-output\alr-docx\benchmark-manifest.jsonl `
  --output benchmark-output\alr-docx\luna-low-results.jsonl --mode codex `
  --model gpt-5.6-luna --effort low
```

Run the frozen structural comparison with identical manifests and settings for:

| Arm | Model | Effort |
| --- | --- | --- |
| Local baseline | none | none |
| Luna | `gpt-5.6-luna` | `low` |
| Terra | `gpt-5.6-terra` | `low` |
| Sol | `gpt-5.6-sol` | `low` |
| Control | `gpt-5.6-luna` | `xhigh` |

These are benchmark arms, not an engine allowlist. Any model and effort
accepted by the installed Codex CLI can be supplied.

The runner flushes each result and resumes by case ID. It reports text
fidelity, order, footnote precision/recall/F1, body/proposition similarity,
citation recovery, wall time, peak process RSS, calls, tokens, repair latency,
retries, schema-valid scope rate, and source-line conservation by export
profile. Gold files may also include a `regions` array of `page_index`, `type`,
and ordered `line_ids`; when present, the report adds region-type, exact
boundary, and line-order accuracy.

Freeze a stratified manifest:

```powershell
legalpdf-benchmark freeze-manifest candidates.jsonl `
  --output frozen-80.jsonl --count 80
```

For the 661-page Text-Fidelity ordered gold scorer, export the engine result to
its existing `product_page_text.jsonl` lane:

```powershell
legalpdf-benchmark text-fidelity-product parsed\document.json `
  --output product.jsonl --dataset DATASET --article-id ARTICLE
```

Then pass `product.jsonl` to Text-Fidelity's
`score_canonical_packages_vs_ordered_gold.py --product-jsonl`. This preserves
the authoritative CER/WER, order, region-boundary, and region-type evaluator
rather than reimplementing it here.

## Known boundaries

- Embedded-text parsing is fully local. Image-only pages require the optional
  local OCR path.
- OCR quality is provider-specific and never inherits native-text confidence.

## License

The engine and its vendored `pdf-inspector` backbone are MIT licensed. The
locked default Cargo dependency closure is permissive and contains no AGPL,
GPL, or mandatory LGPL dependency; third-party crates retain their respective
license notices. Optional ONNX Runtime is MIT licensed. A packaged thin layout
DLL incorporates Apache-2.0 Tesseract and BSD-licensed Leptonica, whose license
files the build recipe emits beside that binary.
