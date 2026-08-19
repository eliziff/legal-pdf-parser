# ALR compatibility notes

This engine is independent of ALR Quote Verifier. It does not import ALR,
search for an ALR checkout, or invoke ALR at runtime.

The local PDF footnote implementation tracks behavior developed in ALR Quote
Verifier's `verifier_core/pdf_adapter.py`, principally these revisions:

| Revision | Upstream change |
| --- | --- |
| `03aa05f8a0afa894c20acfe05e970299e0237d3b` | Harden PDF footnote intake |
| `e0a3d0b298a4bb78d474639977c17695f226d044` | Make PDF intake self-contained and deterministic |
| `f6a4e71756a1c24d28dc52f23b2b3c80402196ff` | Fix same-page PDF intake and corpus updates |

The corresponding neutral behavior is primarily in
`src/legalpdf/core.py`: separator detection, detached-reference association,
typographic footnote-label classification, region classification, footnote
materialization, page-spanning note continuation, and proposition pairing.
It was adapted to `LegalDocument` records and contains no ALR GUI, settings,
cache paths, or application types.

`src/legalpdf/adapters.py::to_alr_payload()` is only a plain-data export
adapter. It does not import ALR and is not used by MikeOSS or Table of
Authorities Maker.

Compatibility is checked through synthetic geometry/footnote fixtures in
`tests/test_engine.py` and the DOCX-grounded benchmark described in
`docs/real-model-benchmark-2026-07-26.md`. Later ALR changes must be reviewed
and selectively ported; they are never consumed automatically.
