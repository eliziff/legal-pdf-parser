# Vendored PDF fidelity components

Beaver retains the MIT-licensed `pdf-inspector` detection and fidelity
mechanics as small internal Cargo packages:

- `pdf-inspector-core`: provider-neutral scalar PDF types and text utilities
- `pdf-inspector-cid-signal`: shared CID-width Unicode fallback signal
- `pdf-inspector-loader`: PDF validation, container repair, and lopdf loading
- `pdf-inspector-detector`: native/scanned PDF classification
- `pdf-inspector-fonts`: font decoding and bundled CMap support
- `pdf-inspector-layout`: line grouping, columns, and reading order
- `pdf-inspector-table-lines` / `pdf-inspector-table-rects`: parallel geometry recovery
- `pdf-inspector-table-heuristic`: aligned-text table recovery and TOC rejection
- `pdf-inspector-tables`: the thin product-facing table detector
- `pdf-inspector-fidelity`: positioned text, glyph, line, and table extraction

The production consumer is `legal-pdf-extraction`. The old standalone
Markdown pipeline, Python extension, and `pdf2md`/`detect-pdf`/`dump_ops`
binaries had no repository callers and are not shipped. Their supported
replacement is the capability-scoped `legalpdf` CLI and library.

The former tagged-structure table reader, external table-structure-result
parser, semantic-table Markdown formatter, and full tagged-tree model also had
no Beaver product callers. PDF structure-name byte repair remains in
`pdf-inspector-loader`; product tables continue through
`pdf-inspector-fidelity::tables::detect_structured_tables`.

Legacy Markdown-only folio context, per-page OCR routing, dense-chart masking,
and TSR-only vector-grid entry points were also removed after the product
compiler proved they had no callers. Product folios and OCR routing remain in
`legalpdf`; table extraction remains on the standard rectangle, ruled-line,
and aligned-text detectors above.
