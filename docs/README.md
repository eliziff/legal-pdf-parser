# Parser documentation

See the [root README](../README.md) for installation and feature profiles.

## Paragraph and note ownership

Paragraph-shaped margin labels remain candidates through document-wide note pairing.
Table values, citation fragments and other structural exclusions retain their separate
exclusion state. Pairing uses reference form, protected citations, geometry, reading
order, note regions, numbering sequences and restarts before the structure resolver
adjudicates competing paragraph and note roles.

The final resolver compares actual marker occurrences against resolved paragraphs,
provisions and explicit exclusions; it does not infer ownership from the proportion
of numbers shared by two sequences. Each note claim carries its numbering restart, the complete selected label-run length,
the number of observed label/reference pairs, whether its own pair was observed,
reference typography, and its note-region witness. Repaired matches do not count
as observed support. An observed pair with typography or a note region can defeat
a competing structural reading even in a short document;
a repaired pair needs both a note region and a corroborated run. Unreferenced labels
can borrow support only from their own run. Counts are taken before unsupported labels
are omitted, so omission cannot inflate the apparent pairing rate. These are explicit
evidence rules, not calibrated probabilities.

Layout paragraph candidates pass through the same quotation, citation, furniture,
table and contents exclusions as grammar candidates. The citation grammar runs over
the whole document; citation core spans project back to source lines and protect
both labels and references, including citations wrapping over line boundaries.
Surrounding case names and introductory signals do not absorb preceding note labels.
A citation-shaped label can still compete when a reference and note region independently
support it. These spans supplement the cheaper layout cues. Consumers query the resulting
paragraph graph IDs and source ranges.

The pairer still chooses its endnote mode from the first selected numeric segment;
mixed footnote/endnote apparatuses in compound documents need separate validation.
The focused tests and private source comparisons do not certify that broader case.

## Measurements and historical receipts

| Record | Scope |
| --- | --- |
| [Native extraction benchmark](../experiments/digitalborn-benchmark) | Fixed digital-born PDF corpus and reproducible runner |
| [CPU/GPU OCR results](../experiments/kraken-lite/cpu-benchmark/RESULTS.md) | The 153-page legal-print experiment and measured hardware/profiles |
| [Corpus check](corpus-check/RESULTS.md) | The recorded corpus run, not certification of later revisions |
| [DOCX-linking benchmark, July 27](benchmarks/docx-linking-2026-07-27.md) | Historical worker/model evaluation |
| [Pairer parity, July 30](canonical-pairer-parity-2026-07-30.md) | Historical pairing comparison |
| [Real-model benchmark, July 26](real-model-benchmark-2026-07-26.md) | Historical model/corpus evidence |

These records describe the measured revisions. July DOCX and model benchmarks
use the former Python implementation and are not current installation guides.

## Historical lineage

The former Python PDF-footnote implementation selectively adapted behavior from
ALR Quote Verifier's `verifier_core/pdf_adapter.py`, including upstream revisions
`03aa05f8a0afa894c20acfe05e970299e0237d3b`,
`e0a3d0b298a4bb78d474639977c17695f226d044` and
`f6a4e71756a1c24d28dc52f23b2b3c80402196ff`. Those references explain lineage, not a
runtime dependency or automatic synchronization policy. The old Python
`src/legalpdf/core.py`, `to_alr_payload()` and `tests/test_engine.py` instructions
are not the current Rust API.

## OCR marker source evidence

Native Kraken retains confidently recognized isolated characters and symbols from
uncovered ink, including narrow fragments beside a body line. Recognition does
not assign their legal role. OCR source lines retain the supplied confidence,
baseline, boundary and recovered-fragment flag; absent polygons/baselines stay
absent. Original marker/body text, word boxes and source line identities remain
separate.

After OCR pages receive source order, the core owner uses the shared numbering
grammar, same-row geometry, reading order and document numbering neighbours to
record marker/body associations. Uncertain bodies remain null, with marker source
evidence retained. These are physical associations, not paragraph/note decisions.
The application supplies the existing structure-analysis implementation.

The BLLA producer preserves recognized fragments and receives the same association
logic. Recovering ink that BLLA omitted still requires a recognition-only operation
for bounded source fragments in its maintained runtime: the available runtime
protocol only supports segmentation and page OCR. Re-segmenting a crop is not
implemented as a substitute. This change has not received behavioral validation.


## Outline

The structure stage gives each unit its parent once the units are settled. A unit's outline
lives in its context: the constituent document holding it (an exhibit or schedule), and the
quotation and note or table cell holding it. In reading order, a heading nests under the nearest
open heading of a more prominent type, a numbered unit under the unit its numbering descends from
(a sibling closes itself and what it opened), and other units under the nearest open heading or
lead-in; printed contents and bookmarks place a heading where they name its parent. Types rank by
size, weight and slant within their context. A parent link between contexts is dropped, except a
row's to its table and a cell's to its row.

## Composition for the structure gold

`structureCompositionBlocks`, `applyStructureComposition` and `resolveCompositionAnnotations`
read a parser structure, or a gold, as blocks and annotations over source lines: the blocks'
owning documents, their logical blocks across continuations, and continued notes and tables as
one object. `experiments/legal-structure-gold/evaluate.py` scores through them.
