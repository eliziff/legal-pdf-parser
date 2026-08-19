# Rust port parity map

Status: release-blocking audit

The executable specification is the last clean working Python engine:
commit `11a325f` (`Verify geometry cache identity and payload`). The uncommitted
`pdf_backend.py` experiment is a candidate extractor, not the behavioral
oracle. `pdf-inspector` may replace PyMuPDF extraction, but it does not license
changes to downstream structure, pairing, artifacts, lookup, repair, or
consumer contracts.

"Equivalent" below means demonstrated on identical serialized inputs. A
similar function name, an inlined implementation, or a passing end-product
average is not parity evidence.

## Release rule

The Rust engine is not production-ready until all of these are true:

1. Every oracle file and installed public surface below remains accounted for.
2. Replaying identical extracted `Page` records through Python and Rust has no
   unexplained stage or final-artifact difference.
3. Replacing only extraction with `pdf-inspector` is non-regressing on every
   frozen qualification case against independent DOCX or journal evidence.
4. Confirmation cases remain untouched until the qualification gate is green.
5. Cache, compact artifacts, geometry upgrades, OCR routing, lookup, CLI exit
   behavior, adapters, repair, and DOCX linking pass contract differentials.
6. Only after correctness passes, every performance lane is a strict wall-time
   win with no peak-memory regression.
7. The shipped implementation has no Python fallback or compatibility shim and
   its complete dependency closure is MIT-compatible/non-AGPL.

Merged note labels are not intrinsically errors. They fail only if they change
pairing, reading order, searchable text, geometry, artifacts, or a consumer
contract. Association after pairing and consumer-specific materialization are
valid when those outcomes are preserved or improved.

Region-dependent Text-Fidelity improvements are separately fail-closed. They
activate only when every nonempty source line has a stable ID and a non-unknown
region supplied by PPDoc or an MLLM. The provider identity is irrelevant; an
incomplete contract leaves baseline parsing unchanged. Source roles are
snapshotted before native furniture and ordering mutate normalized labels.

## Frozen oracle inventory

The hashes make the map fail closed over every function, constant, regex, and
data record in the working engine, including private helpers.

| Oracle file | Bytes | SHA-256 |
| --- | ---: | --- |
| `__init__.py` | 1,182 | `1cc8f406803b8bc155335044738361e8d4ad0d38b77b874630b52d62342d136d` |
| `__main__.py` | 51 | `482b8693e2fe6dbd732e3d7961e03827d3efa187bad11d29da188be580ebbeea` |
| `adapters.py` | 4,382 | `6a1710d0bc288c294ec8322fe0ed95c85aa5932c8382c1da4f9e40a69a005aeb` |
| `anchored_scan.py` | 11,557 | `06d13c79e067c169347da02289ee9459a91334c565185fb0c5f40c243c1876e6` |
| `benchmark.py` | 46,085 | `68b460a419f2ae63b02dff7d36c6260fe3572004722e826199590b8e1e838589` |
| `cli.py` | 10,867 | `684ea4d90d0bd1579f607bbf53844fdf70df4006328ece2dba557e98586e195e` |
| `codex_repair.py` | 28,319 | `3d439a7e1e4fedd4cb77b153d2ac2317106e8e5bacadc448c71f35ec5d491f28` |
| `column_order_arbiter.py` | 17,283 | `d1e64e6911d4759819e12b95916ca812d107394df97b88a28ea1fcdbb15a68a5` |
| `core.py` | 105,292 | `790e4de14f8f43ba05303486bc48dc19bec5c765c29d8b9055fe9cb043d12a2f` |
| `deterministic_citations.py` | 25,712 | `ce8564c09992b3d2de48720edb43a1c118be76bfe593f5a24949f2e6f5ddbdd0` |
| `docx_linking.py` | 31,801 | `41285d3b3055a619c4064779274ee4dc731e3047bd5f82814367adb5bc608049` |
| `footnote_pairing.py` | 120,918 | `297ef847a0855f98ab4b2d39a419600479b69fa1651e405caba6187cde7ddeb6` |
| `footnote_pairing_support.py` | 17,232 | `ba9bc40fc4cc4d12d9f8cbc7f1620170fb4714189e5b68c417b7e3119579d76a` |
| `footnote_separator_scan.py` | 15,826 | `051f7212920c942465750fd231ba63bcbc26d9ba26b6a46be54a11ed62072af9` |
| `grammar_tables.py` | 20,908 | `d7c40f46641e4e8c612dee29320f31127d075261fe37ff793cd9e67b39e23131` |
| `model.py` | 19,738 | `da4303f896d3b67e310df02bb64ac2cfafedbc54ff12adddcc22319804838de0` |
| `note_crossrefs.py` | 4,567 | `3decebe1d56326ec417d30105d6acdf0ff6c0573c06fab32d5005461a9229399` |
| `ocr.py` | 7,013 | `8d8c2db9f172c4bfd387135b08e493294b6b53a34e8e1a8631004efb3b887977` |
| `superscript_splice.py` | 9,709 | `68817de4e652cc6337b1aad175aa7847778bcaf56252c4214f84cc4dfe1af973` |
| `data/mcgill_reporters.json` | 7,202 | `6c5aa7b0b826e0842ff0631de40ee8457cb5e948eb120e7568686489703b57d7` |
| `tools/export_docx_word.ps1` | 1,548 | `53466492360f31df55cfa77fecce4e7df5fa000dda32511cbfd5022b377eafb3` |
| `data/grammar-tables/citations.json` | 19,702 | `51675b8eb94ff067c62103c271a2a7a7d91de551df4b39bead9bcf9c14359560` |
| `data/grammar-tables/footnote-labels.json` | 5,723 | `bf283283ddaf4d5b8c9c42c94ef1d78c4b8ace93eb50f2ad2c69163dcf55ec19` |
| `data/grammar-tables/pinpoints.json` | 5,774 | `8801cef918ad82a83fdbb6bd0b95b205a9fcd49e25f5b89a0393ef1b11c5a42d` |
| `data/grammar-tables/references.json` | 22,189 | `b42f38fea7ed4c66272d0d09d33df5176744e2db92674d393cfb649a132e22d4` |

The harness also pins `pyproject.toml` and all twelve oracle test files. Those
tests define 113 behavior checks across extraction, structure, artifacts,
cache, OCR, pairing, ordering, cross-references, deterministic citations,
repair, DOCX, adapters, lookup, and CLI behavior. Rust tests do not substitute
for them; their input/output contracts must be replayed against the native
implementation.

## Installed surface

| Contract | Python oracle | Rust target | Current state |
| --- | --- | --- | --- |
| Data records and JSON field names | `model.py`: `Word`, `Span`, `Line`, `Region`, `Page`, `Paragraph`, `Section`, `Footnote`, `Diagnostic`, `RepairRecord`, `LegalDocument`, `FootnoteLookup` | `model.rs` | Mapped; differential unproven |
| Parse | `parse_pdf` | `engine::parse_pdf` | Partial; local-only behavior unproven |
| Add geometry | `add_pdf_geometry` | `engine::add_pdf_geometry` | Mapped; differential unproven |
| In-memory note lookup | `lookup_footnote` | none | Missing |
| Persisted note lookup | `lookup_artifact_footnote` | `artifact::lookup_artifact_footnote` | Mapped; differential unproven |
| Selective repair | `improve` | none | Missing |
| Artifact read/write | `load_artifacts`, `write_artifacts` | `artifact.rs` | Mapped; byte/contract differential unproven |
| Geometry read/write | `load_geometry_artifacts`, `write_geometry_artifacts` | `artifact.rs` | Mapped; byte/contract differential unproven |
| ALR and ToA adapters | `to_alr_payload`, `to_toa_text_units` | none | Missing |
| DOCX planning/apply | `assess_route`, `plan_footnotes`, `plan_docx_links`, `apply_docx_links` | none | Missing |
| OCR provider | `OCRProvider`, `TesseractOCRProvider`, `OCRLine` | `TesseractOcr`, `OcrLine` | Partial; custom-provider contract missing |

Python CLI commands are `parse`, `add-geometry`, `page-count`,
`ocr-identity`, `repair-identity`, `improve`, `footnote`, `docx-link-plan`, and
`docx-apply-links`. Rust currently lacks `repair-identity`, `improve`,
`docx-link-plan`, and `docx-apply-links`; option, JSON, error, and exit-code
parity for the remaining commands is unproven.

## Runtime stage map

| Order | Required behavior | Python oracle functions/modules | Rust target | Current state |
| ---: | --- | --- | --- | --- |
| 1 | Path/type checks, source hash, mode validation | `parse_pdf`, `_sha256_file` | `engine.rs` | Partial: Codex mode missing; errors differ |
| 2 | Engine identity and deterministic cache key | `_engine_identity`, `_cache_key`, `_stable_hash`, `_default_cache_dir` | `engine.rs` | Intentional AppData path change; identity/cache differential unproven |
| 3 | PDF open, encryption/page/metadata handling | `_extract_pdf_pages` | `pdf.rs` with `pdf-inspector`/`lopdf` | Candidate replacement; unproven |
| 4 | Native text, spans, words, offsets, blocks, quality | `_normalize_pdf_text`, `_line_words`, `_extract_native_page` | `pdf.rs` | Candidate replacement; exact consumer contract unproven |
| 5 | Vector and raster separator detection | `_separator_y`, `_raster_separator_y`, `footnote_separator_scan.py` | `pdf.rs`, `ocr.rs` | Partial translation; parity unproven |
| 6 | Low-quality routing and OCR line construction | `_ocr_page_lines`, `ocr.py`, `_extract_pdf_pages` | `ocr.rs`, `pdf.rs` | Partial; provider and diagnostic behavior differ |
| 7 | Repeated header/footer furniture | `_normalize_furniture`, `_mark_repeated_furniture`; Text-Fidelity stable-edge and alternating-folio witnesses | `structure.rs` | Baseline mapped; stable y/parity-x evidence and sequential recto/verso folios ported; replay unproven |
| 8 | Detached and orphaned superscript association | `_detached_reference_target`, `_associate_detached_references`, `_associate_spliced_markers`, `superscript_splice.py` | `structure.rs` | Partial/inlined; replay unproven |
| 9 | Typographic label and body/note/endnote/heading classification | `_label_is_typographic`, `_classify_page`; Text-Fidelity native body-font and heading grammar passes | `structure.rs`, `pairing_support.rs` | Baseline mapped; source-region-gated body-font, display, ladder, guarded demotion, and wrapped-continuation passes ported; replay with PPDoc/MLLM regions pending |
| 10 | Stateful endnote continuation between pages | `_extract_pdf_pages` continuation state | `structure::classify_pages` | Mapped shape; replay unproven |
| 11 | Conservative column/geometry ordering | `_order_page`, `column_order_arbiter.py` | `structure.rs` | Not faithful: one undocumented branch removed; full differential pending |
| 12 | Region construction and exact membership | `_build_regions` | `structure.rs` | Mapped; replay unproven |
| 13 | Printed page labels with conflict refusal | `_assign_printed_page_labels` | `structure.rs` | Mapped; replay unproven |
| 14 | Post-repair note-mode inference | `_infer_note_region_modes` | none | Missing |
| 15 | Canonical candidates, backbone, repairs, suppression, refs, symbols | `footnote_pairing.py`, `footnote_pairing_support.py`, grammar tables, reporter inventory | `pairing.rs` | Approximate/unproven; grammar tables and reporter inventory are absent |
| 16 | Detached-marker integration and canonical summary refresh | `_pair_markers`, `_merge_detached_markers`, `_refresh_pairing_summary` | folded into `pairing.rs`/`structure.rs` | Non-equivalent representation; replay unproven |
| 17 | Footnote body bounds, continuation, anchors, diagnostics | `_marker_order`, `_materialize_footnotes` | `pairing::materialize` | Partial translation; replay unproven |
| 18 | Text-flow diagnostics | `_text_flow_faults` plus arbiter primitives | `structure::text_flow_faults` | Mapped; stage order differs; replay unproven |
| 19 | Paragraph text, marker insertion, and anchors | `_join_lines`, `_build_paragraphs` | `structure.rs` | Mapped; replay unproven |
| 20 | Sentence/passage proposition attachment | `_sentence_at`, `_attach_propositions` | `structure.rs` | Uses hard-coded grammar approximation; unproven |
| 21 | Note cross-reference resolution | `note_crossrefs.py`, `resolve_note_crossrefs` | folded into `structure.rs` | Partial/inlined; replay unproven |
| 22 | Heading identity and section boundaries | `_heading_locator_kind`, `_section_identity`, `_build_sections`; Text-Fidelity heading inventory/demotion passes | `structure.rs`, `pairing_support.rs` | Document-wide counter grammar and source-gated promotion/demotion ported; exposed hierarchy and regioned-corpus differential pending |
| 23 | Unmatched-reference diagnostics | tail of `_derive` | `unmatched_reference_diagnostics` | Mapped shape; replay unproven |
| 24 | Status computation | `_status` | `structure::status` | Mapped; exact differential unproven |
| 25 | Full structural validation | `_validate_document` | `structure::validate_document` | Partial; rejection/acceptance differential unproven |
| 26 | Atomic artifact publication and compact projection | `model.py` artifact helpers | `artifact.rs` | Mapped; byte/ordering/safety differential unproven |
| 27 | Cache publication, invalid-marker recovery, cache-hit provenance | `parse_pdf` | `engine.rs`, `artifact.rs` | Partial; differential unproven |
| 28 | Geometry-only identity/text verification | `add_pdf_geometry` and model helpers | `engine.rs`, `artifact.rs` | Mapped; differential unproven |
| 29 | Note lookup ambiguity, hints, context, proposition mode | `lookup_footnote`, `lookup_artifact_footnote` | artifact lookup only | In-memory path missing; persisted path unproven |
| 30 | Bounded structural repair and replay | `codex_repair.py`, `rebuild_derived`, `improve` | none | Missing |

## Other shipped behavior

| Python module | Functions/classes | Role | Rust state |
| --- | ---: | --- | --- |
| `adapters.py` | 2 | Stable ALR/ToA plain-data projections | Missing |
| `anchored_scan.py` | 5 functions, 1 class | Exact regex acceleration used by legal grammars | Missing |
| `deterministic_citations.py` | 27 functions, 3 records | Citation field extraction and recall-first splitting | Missing |
| `grammar_tables.py` | 11 functions, 1 class | Versioned grammar loading, validation, compilation, vectors | Missing |
| `docx_linking.py` | 23 | Deterministic and bounded model-assisted DOCX link plan/apply | Missing |
| `codex_repair.py` | 16 | Bounded, validated, cached structural repair | Missing |
| `benchmark.py` | 32 functions, 1 class | Installed benchmark/DOCX gold/export command | No Rust command; harness-only code may remain Python only if it is removed from the shipped engine surface |
| `cli.py` | 2 | Installed command contract | Partial |

## Differential harness map

The revised harness has separate gates so extraction cannot hide a bad port:

1. **Oracle identity:** verify commit and every file hash above before running.
2. **Common-input replay:** serialize raw oracle `Page` records and separators,
   then run the exact production stage sequence in both implementations.
   Compare per-stage page annotations, ordering, regions, markers/pairs,
   diagnostics, paragraphs, propositions, sections, status, and validation.
3. **Pure contracts:** replay existing oracle vectors for column ordering,
   separator scan, superscript splice, grammar tables, citation splitting,
   cross-references, adapters, lookup, artifacts, cache, OCR, repair, and DOCX.
4. **Extractor qualification:** compare final products from PyMuPDF and
   `pdf-inspector` to independent DOCX/journal truth per document. No average
   may conceal a losing case.
5. **Held-out confirmation:** run only after all mapped stages and
   qualification documents are green.
6. **Performance:** alternate fresh processes, require Rust to win each frozen
   case and lane on median wall time, and prohibit peak-RSS regression.
7. **Packaging/license:** build from a clean checkout, inspect the full Cargo
   license tree, and prove the released binary works with Python absent.

The current end-product corpus gate remains useful, but by itself it does not
prove a port. Until common-input replay and the pure-contract lanes exist and
pass, the Rust implementation remains an experiment.
