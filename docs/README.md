# Parser documentation

The [root README](../README.md) owns current setup, feature profiles, PDF boundaries
and validation requirements. The standalone
[structure README](https://github.com/eliziff/legal-structure-parser) owns shared
legal-structure semantics. Do not copy either contract into another refactor plan.

## Measurements and historical receipts

| Record | Scope |
| --- | --- |
| [Native extraction benchmark](../experiments/digitalborn-benchmark) | Fixed digital-born PDF corpus and reproducible runner |
| [CPU/GPU OCR results](../experiments/kraken-lite/cpu-benchmark/RESULTS.md) | The 153-page legal-print experiment and measured hardware/profiles |
| [Corpus check](corpus-check/RESULTS.md) | The recorded corpus run, not certification of later revisions |
| [DOCX-linking benchmark, July 27](benchmarks/docx-linking-2026-07-27.md) | Historical worker/model evaluation |
| [Pairer parity, July 30](canonical-pairer-parity-2026-07-30.md) | Historical pairing comparison |
| [Real-model benchmark, July 26](real-model-benchmark-2026-07-26.md) | Historical model/corpus evidence |

Keep the original receipts, inputs and reproduction limits with their experiments.
Do not relabel them as validation of the current binary. In particular, July DOCX
and model notes describe former Python modules, tools and model-routing choices;
they are not current installation instructions or authorization to spend API
credits or send private documents to a provider.

## Historical lineage

The former Python PDF-footnote implementation selectively adapted behavior from
ALR Quote Verifier's `verifier_core/pdf_adapter.py`, including upstream revisions
`03aa05f8a0afa894c20acfe05e970299e0237d3b`,
`e0a3d0b298a4bb78d474639977c17695f226d044` and
`f6a4e71756a1c24d28dc52f23b2b3c80402196ff`. Those references explain lineage, not a
runtime dependency or automatic synchronization policy. The old Python
`src/legalpdf/core.py`, `to_alr_payload()` and `tests/test_engine.py` instructions
are not the current Rust API.

The former DOCX citation-linking plan duplicated an obsolete application-worker
implementation. Current Word mutation, linking and application integration belong
to [Beaver's document-capabilities plan](https://github.com/eliziff/Beaver/blob/main/docs/roadmap/document-capabilities.md)
and its current contracts, not the PDF parser. Git history retains the original
plans; remaining cross-project work stays in Beaver's roadmap.
