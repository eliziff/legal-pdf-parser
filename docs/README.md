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


### Scoped outline inference candidate

PDF candidate materialization now invokes one outline assembler before graph
projection and literal derived-view assembly. Units retain their node IDs and exact
source spans; each constituent, quotation and reading stream has its own active
ancestor stack. The stack exposes an explicit null root and every open ancestor,
without a recent-unit window. Heading/section duplicates and later learned
reparenting are no longer on this PDF path. Document-relative styles and shared
grammar ordinal/prefix facts supply attachment features. Source OCR marker/body
associations adjust candidate body anchors without deciding paragraph versus note.
Printed locators attach to the existing exact role unit; accepted starts split
native layout groups without duplicating that unit as a second numbered node.

All learned heads consume the same prepared source witnesses in training and
inference, including contents, quotation, table, translation and source-layout
facts. Their document statistics include every physical line, even without a
reader classification. This evidence preparation does not read gold roles or run
the learned structure heads. Earlier feature contracts are incompatible with
these contextual, complete-source values. Membership uses
`source-fragment-membership-v4`.

Aligned source spans carry their own document-wide style rank, physical-line
frequency and body-style membership. Inline headings and body text on one line
therefore use their own typography aggregates. Physical candidates also carry
shared-grammar marker ordinals and compound prefixes, whole-source sequence
frequency, and preceding/following succession or restart evidence. These source
neighbours do not declare shared ownership. Scope candidates compare the current
source marker against each identity's last numbered marker, retaining that marker
through intervening unnumbered text; outline attachment uses the same relation
helper. Gold role labels and scope boundaries do not supply these features.

The new attachment feature contract is `scoped-span-root-v6`. Historical parent
weights did not learn explicit roots and are incompatible with this contract.
Until revised gold is frozen and new weights are trained, production uses the
scoped deterministic stack; this is an unfinished candidate, not a demonstrated
quality improvement. The former line-level unit predictor is replaced by the native
`source-span-role-v7` boundary/eight-role predictor, supplied through
`LEGALPDF_SPAN_MODEL`. Its candidate starts include physical lines, typography
changes, shared grammar markers and actual quotation punctuation within lines.
Punctuation creates possible split positions, not quotation ownership. Explicit
primary profiles and their absence are features in the predictors. Actual PDF
structure tags supply additional witnesses only where retained span offsets match
the printed source. Tags do not dictate the legal role. Missing weights leave source
heuristic roles in use. Exact bookmark titles confirmed in body text supply
attachment witnesses, while mere contents-text matches remain weaker evidence.
Training derives those bookmark/body associations from the same source readers;
authored navigation targets are not accepted as feature inputs.
Detected display quotation identity survives page-furniture interruptions.
The scope stage uses `LEGALPDF_SCOPE_OPENING_MODEL` (`source-scope-opening-v3`)
and `LEGALPDF_SCOPE_MODEL` (`source-scope-identity-v8`) together. The opening
regressor predicts how many constituent or quotation identities begin at each
exact physical-source candidate anchor, including word, typography, numbering and
quotation boundaries inside existing paragraphs. Scope candidates do not depend
on the role predictor splitting those paragraphs. Only selected ownership changes
split materialized units. Training scores the same physical intervals; missing
ownership supervision remains unavailable. Finite predictions round to nonnegative integer counts;
there is no fixed nesting vocabulary. Coincident declarations receive distinct
identities before direct ownership is selected, so an enclosing scope can exist
without directly owning a unit. The ranker chooses each declaration's parent,
with an explicit null root distinct from the named enclosing document. Quotation
declarations separately select their owning document from already declared
constituents. Gold identities and parents supply targets, never input features. With neither
compatible model, the existing heuristic path remains in use. Configuring only
one compatible member of the pair is an explicit error.

Unit ownership then selects declared or resumed document/quotation identities;
reading streams retain their new-or-resumed decision. Source opening ordinals
and selected ancestry distinguish coincident declarations. Unmatched gold opening
anchors and unavailable parent targets are reported explicitly rather than
inventing root labels or negative openings. Parent scopes retain positive source
opening membership and accumulate descendant ranges separately from directly
owned ranges. Titles, attribution and relationships are not fabricated.
Quotation and reading-stream histories belong independently to their constituent.
A stream may resume across quotation transitions but cannot cross a constituent
boundary. Explicit table/note reading-stream facts remain authoritative. Inline
quotation ownership still depends on source-span starts and compatible learned
weights, which have not been trained for this candidate.
Typed scope annotations retain exact membership across resumptions. Titles, layout,
attribution and attachment relationships remain absent unless supplied. Container
ownership propagates only when contained units agree; table/row/cell containment
stays together, and mixed scopes retain the established container owner.

`tools/train_structure.py outline|spans|scope_openings|scopes|membership FAMILY_MANIFEST OUTPUT_MODEL` generates features through
the corresponding native training-row operation, including `pdfScopeOpeningTrainingRows`, using
canonical corrected source and literal gold facts. Training reuses the same source
reader preparation and quotation/table/contents/layout witnesses as production,
with preserved character typography, aligned tags and available physical OCR data.
It requires the family manifest's `evaluation_frozen` status,
no blockers, unchanged selected gold artifact hashes, and only reads training
members. Gold identities never collapse to line IDs; null roots are positive
training targets. Gold relationships outside the fast producer's available stack
are reported as unavailable queries, never relabelled as negative roots. The span
trainer reports missing candidate starts and excludes unannotated source from
negative boundary examples. Scope-opening training uses count regression; scope training follows literal
identities and ranks declared/resumed ownership without using identity strings
as features. Training
uses the existing offline LightGBM environment; runtime tree scoring stays in Rust.
No training, model selection or performance/quality evaluation has been run for
this replacement.

## Model-assisted composition

`experiments/structure-composer/composer.py` exposes bounded, provider-neutral
composition over source evidence. The embedding caller supplies model dispatch
and the shared preparation, application and projection operations. Complete
requests return a `DocumentStructure` through `projectStructureComposition`;
partial chunks return composition state for continuation. The shared Rust owner
constructs source/rendered coordinates, annotation layers and retained native
witness mappings. Gold export uses the same projection and adds its artifact
provenance. Provider selection and image transport remain with the embedding
caller. No model-run accuracy or latency acceptance is claimed for this path.

Native composition preparation also supplies annotation seeds and exact retained-node
selection. Annotation continuation and quotation containment use shared source-view
resolution during authoring and final projection. Original producer extras retain
their coordinate basis as native witnesses. Cross-reference edges and contents
offsets are remapped through typed source correspondence; unconfirmed facts remain
explicit evidence rather than active links with stale coordinates.
