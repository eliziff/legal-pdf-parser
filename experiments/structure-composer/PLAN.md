# Source-anchored multimodal structure producer

Status: implementation candidate, not an accepted corpus result. The composer uses
the shared source-span composition contract. Complete runtime calls are wired to
the shared Rust document projector. The consolidated native build and Python/JavaScript
syntax checks passed on 2026-10-13; behavioral and performance acceptance remain
separate gates.

## Objective and ownership

Provide one provider-neutral structural producer for runtime quality mode and
independent corpus authoring. Both callers use the same literal annotation
operations. The slower producer owns its resulting structure; it may start from
fast-parser proposals but can reconsider every source block. It is not restricted
to repairing an already selected heading tree.

The shared Rust structure owner supplies semantic types, literal edit application,
validation and derived structure. The PDF parser supplies extraction/OCR evidence,
geometry and page images. The embedding caller supplies provider/model selection,
image transport, metering, caching and receipts. Gold scheduling, source correction
and publication remain with the existing gold runner. No additional service or UI
is required by this contract.

The owning design is the workspace's [PDF structure gold roadmap](../../../docs/roadmap/pdf-structure-gold.md).
The gold runner's [plan](../../../experiments/legal-structure-gold/PLAN.md) owns the
corpus workflow. These workspace links address sibling first-party owners.
Court-record enrichment and the form/authority catalogue retain their separate
owners and taxonomies.

## Current implementation

The provider-neutral operations in composer.py are:

- prepare: align native source fragments with page text, geometry and typography,
  and collect every target block plus addressable enclosing context;
- prompt and schema: describe sparse source-anchored decisions and enabled modules;
- apply: decode response aliases, check baseline identity and pass literal edits
  to the shared Rust applicator;
- compose: call the embedding dispatch function, retry invalid responses or
  requested absent modules, and stop after three bounded attempts.

The embedding caller passes the existing native preparation and application
operations and a required project_structure callback to compose. Complete calls
project the literal request through the shared Rust owner and return its typed
structure, evidence, trace and composition state. Baseline node identities identify
which existing structural nodes are superseded. Native preparation supplies the
annotation facts and the exact retained-node selection as well as editable blocks;
unresolved grids remain native evidence. Source annotation continuation and
quotation-scope views use the shared resolver without requiring a final manuscript.
Original coordinate-tagged witnesses stay separate from current annotation spans.
Applied composition also carries surviving native inventory across partial calls.
Explicit removal follows native parent edges; ambiguous correction spans lose
addressability while their original evidence remains.
Python prepares evidence and
decodes the author response; it does not supply a competing hierarchy classifier.
The lower-level apply operation accepts the projector optionally for existing gold
callers, whose final publication remains a separate assembly step.

### Source and structural facts

Block references use Unicode-scalar spans inside identified physical source
fragments. Multiple structural starts may share a line. Native UTF-16 source
coordinates are converted explicitly at this boundary. Replacement membership
must cover the affected source characters exactly once; unchanged source remains
available with its extraction witnesses.

The eight primary roles are text, heading, caption, instruction, headnote,
marginal_note, furniture and page_label. Lists retain marker, locator and nesting
attributes rather than becoming a primary role. A document title is an anchored
document annotation, not automatically a section heading. Tables, fields, notes,
quotations and navigation remain independent, potentially overlapping annotation
layers. Blank and merged cells and exact field/note components survive composition.

Every block supplies a parent source start or explicit null for its scope root.
Levels and exact extents are derived. Document, quotation and independent reading
stream ownership keep local locators and numbering separate. Constituent document
identity is distinct from its source opening: existing annotation aliases or fresh
response-local document IDs address parents and resumptions, including declarations
sharing an opening. Do not infer containment from a mandatory single packet root.

Physical source order, logical joins and object continuations are distinct facts.
Literal separators and manuscript normalization preserve printed source text;
normalization does not rewrite the extraction witness. A non-text continuation
need not invent a textual separator. Text seams must be resolved before a complete
manuscript can be exported.

### Bounded work and complete coverage

Targets are contiguous physical pages with full-page evidence. The producer
receives all target blocks, including unmarked text, plus document-wide heading,
navigation, style and scope context. Enclosing parents, constituent identities,
reading-stream roots and continuation endpoints remain addressable outside the
target chunk. Sparse omission preserves existing facts; it does not exclude a block
from structural reconsideration.

The gold runner separates source correction from document structure. Source
correction handles transcription, missing fragments, physical grouping and reading
order. Structural authoring works on the corrected source; discovered source
defects return to the source-correction responsibility. Short documents are supplied
whole; long documents use contiguous chunks with carried enclosing context. A
separate filtered hierarchy author is retired.

An explicit through boundary supports partial assembly. Inherited text seams
whose target lies beyond that boundary may remain pending. When their target is
authored, the response must resolve the separator or remove the asserted link.
Partial products contain composition state rather than a final typed document;
final assembly rejects unresolved text seams. The default runtime composition
requests complete assembly through the shared projector.

Modules can be omitted for a bounded runtime request when the caller chooses.
Only absent modules may be requested in a retry, and a module request carries no
edits. Gold structural authoring retains complete layer coverage. Source-derived
gold remains independent of parser scores.

## Historical measurements

The earlier repair experiment used a three-page affidavit/form packet with
Luna/high, one Below Normal worker, schema enforcement and full-page images. These
measurements describe the old candidate and response contract. They do not validate
the current eight-role, source-span or shared-projection implementation.

The target-page result preserved all 71 source lines and source text, demoted a
false section heading, recovered a five-row by four-column table and eight blank
cells, retained field annotations inside populated cells, and survived the recorded
no-op replay. Its first response copied source text into correction records; local
validation rejected that response and a second response obeyed the metadata-only
contract.

The full-document result preserved those 71 lines, recovered the same table and
blank cells, identified three constituent starts, and proposed 18 block corrections
and 34 annotations. It was not accepted as reference gold. Review found an incorrect
three-root decomposition, form instructions misclassified as an author note and
linked to an unrelated heading, and field annotations more exhaustive than the
repair use case required.

That historical experiment added a one-root restriction. The current representation
instead records explicit constituent identity, supplied parents and scope ownership;
the old restriction is not evidence that every legal packet has one authored root.
The observed note and constituent errors still need independent current-candidate
proof. No historical figure here establishes their resolution.

## Remaining work and evidence gates

1. Establish behavior of the shared Rust projector and its runtime/gold callers
   when behavioral runs are authorized. Compilation and source review have passed;
   they do not establish preservation of source/rendered membership, normalization,
   ownership and unrelated retained layers under real document inputs.
2. Preserve retained native facts and structure-only metadata through canonical
   gold packing and reopening. The current package omits that residue even though
   the projector retains it; carried removal controls are not its factual payload.
   Then complete independently justified structural authoring under the settled policy,
   including the selected complete long documents. Keep source defects and
   unsupported source appearances explicit. Preserve original gold and receipts
   until an authorized revision is ready; do not regenerate answers from parser
   output.
3. Freeze source/template families and duplicate-page membership before final model
   selection. Keep historical exposure and unknown provenance visible. The root
   roadmap owns corpus counts, split status and the 48-document expansion.
4. When authorized, measure native and slower producers against the same independent
   answers, including digitalborn and OCR modes, rare roles, document/quotation
   ownership, exact continuations, tables, fields, notes and consumer source lookup.
   Record model calls, tokens, latency and memory separately from fast-path costs.
5. Demonstrate end-to-end source fidelity and consumer behavior with the actual
   candidate implementation. Compilation alone does not establish structural
   accuracy, browser behavior or performance acceptance.

The active user instruction prohibits writing or running tests, evaluations,
benchmarks and acceptance runs. No new model dispatch or private-document
transmission is authorized by this document. Consequently the behavioral and
performance gates remain unproven; the historical runs above are not substitutes.
Do not introduce a release/pin gate, mandatory model cascade or separate hierarchy
repair stage to satisfy them.
