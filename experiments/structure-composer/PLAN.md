# Multimodal structure composer candidate

## Objective

Build one provider-neutral structure-composition operation for two callers:

1. the PDF parser's multimodal repair/salvage pathway; and
2. a corpus worker that produces corrected manuscripts and structural gold for
   parser and witness-stack evaluation.

The composer improves the parser's existing structure. It does not ask a model to
recreate source text, geometry, IDs or every unchanged decision.

```text
native PDF/OCR structure + witnesses + full-page images
                         |
                         v
          cheap module selection and bounded request
                         |
                         v
              provider-owned model dispatch
                         |
                         v
       local validation + sparse materialized corrections
                         |
              +----------+-----------+
              |                      |
              v                      v
      parser repair candidate   manuscript + structure gold
```

The engine owns evidence preparation, request grammar, source-identity validation,
composition and replay. The embedding caller owns provider/model choice, image
transport, metering, caching and receipts.

## Current candidate

`composer.py` currently provides four provider-neutral operations:

- `prepare`: aligns immutable native line IDs to text, geometry and typography;
- `prompt` and `schema`: produce a compact bounded correction request;
- `apply`: validate and materialize sparse corrections without altering the baseline;
- `compose`: repeat a request when validation fails or an absent module is requested.

Page repair receives the target plus r=1 full-page images. The gold runner extends
the correction grammar with precise text spans, transcription edits, continuations
and resumptions. One serial gold pass edits a single target page with its r=1
neighbours and all five annotation contracts. Compact inherited parent IDs carry
enclosing structure. Code assembles and validates the document before export.
Runtime parser integration remains separate work.

The layers are deliberately independent:

```text
source line
  +-- block role: prose / heading / list item / instruction / furniture
  +-- table membership
  +-- form-field role
  +-- note or quotation relationship
  `-- constituent-document membership
```

This permits, for example, a form field inside a table cell without forcing either
classification to replace the other.

## Measured results so far

The current experiment used the three-page High Court of Australia affidavit form
packet with Luna/high, one Below Normal worker, schema enforcement and complete page
images.

### Target-page composition

The index page result:

- preserved all 71 document source lines and source text;
- demoted `DESCRIPTION` from a false section heading;
- recovered the visible 5-row by 4-column table;
- represented all 8 visibly blank cells;
- retained form-field annotations inside populated table cells; and
- survived a no-op replay without changing the materialized composition.

The first response copied source text into correction records. Local validation
rejected it, and the second response obeyed the metadata-only contract.

### Full-document composition

The full-document result:

- preserved all 71 source lines and source text;
- recovered the same 5-by-4 table and 8 blank cells;
- identified starts for the affidavit, index of exhibits and exhibit certificate;
- proposed 18 block corrections and 34 annotations.

It is not an accepted reference result. Semantic review found:

- all three constituents were returned as roots rather than one packet hierarchy;
- the printed form instructions headed `Notes` were incorrectly classified as an
  author note and linked to the `INDEX OF EXHIBITS` heading; and
- form-field annotation remains more exhaustive than a sparse parser-repair call
  usually needs.

The document validator now requires exactly one root and earlier enclosing parents
for later constituent documents. The note error remains an observed corpus failure.

## Remaining implementation

### 1. Finish the bounded correction contract

- Preserve sparse semantics: omitted structure remains unchanged.
- Keep basic block bodies empty; allow JSON bodies only for headings and enabled
  annotation modules.
- Reject duplicate annotations of the same kind at one anchor.
- Validate note references against plausible marker-bearing source spans and reject
  instruction blocks masquerading as notes.
- Preserve heading promotion, demotion, level and parent corrections globally.
- Retain exact source-ID partition and source-text identity after every replay.

### 2. Compose witnesses before model dispatch

Feed the existing parser result and its independent witnesses, including:

- current blocks, heading grammar and diagnostics;
- text, geometry, typography and page canvas;
- reading order, columns, tables, forms, quotations and notes;
- printed and physical page-number changes;
- lexical and typographic discontinuities; and
- page-size, orientation and margin changes.

Parser repair can use cheap evidence to omit irrelevant response modules. A loaded module is used
directly; only absent modules may be requested, and a request repeats the same
evidence with the expanded grammar.

### 3. Complete constituent-document composition

Represent one root plus nested/sibling constituent documents, each with:

- type and source-anchored start;
- enclosing parent;
- mechanically derived extent;
- evidence-backed facets such as title, court, parties, author, date and procedural
  role; and
- typed relationships such as attachment, exhibit, schedule or continuation.

Document boundaries must be supported by a chorus of witnesses. Printed-page-number
restart alone is insufficient. Template placeholders must never create hypothetical
parties, dates or attachments.

### 4. Connect the two callers

Parser repair:

- route only diagnosed or witness-conflicted scopes by default;
- apply validated sparse corrections to the parser's structure operation;
- preserve provider selection at the application boundary; and
- retain the existing cache, budget and provenance requirements.

Gold generation:

- inspect every page once in an r=1 window and assemble document relationships;
- constrain correction records and their payloads through Structured Outputs;
- retain the baseline, final gold with replayable corrections, evidence hashes,
  prompt/schema hashes and provider receipt;
- assemble the corrected manuscript and all source-anchored annotation layers;
- retain digitalborn/OCR settings and simple run provenance; and
- export deterministic replay and comparison inputs for the witness stack.

### 5. Corpus iteration gates

Do not self-regenerate an expected baseline. Measure against observed corpus failures
and independently reviewed references.

1. Re-run the affidavit packet after the one-root and note fixes.
2. Exercise several monodocument profiles: judgments, motions, orders, affidavits,
   journal articles without paragraph numbering, and form-heavy filings.
3. Exercise composite records: motion records, exhibit packets, appendices and
   multi-order bundles.
4. Compare native baseline, sparse patch and materialized candidate for exact source
   preservation and reviewed structural correctness.
5. Expand to the 750 born-digital corpus with one worker only after focused failures
   stop changing the contract.
6. Add OCR PDFs when their extracted outputs are available, keeping OCR quality and
   structural correctness as separate measurements.
7. Gate any parser-path replacement on the exact parser/structure/Inspector/Beaver
   combination that produced the accepted corpus result.

## Acceptance criteria

The candidate may replace the current multimodal salvage path only when:

- source identity, text and exact line coverage are mechanically guaranteed;
- unchanged native structure survives replay byte-for-byte at the composition layer;
- reviewed table, form, note, quotation, heading and reading-order cases improve over
  the native baseline without profile regressions;
- full-document composition correctly decomposes reviewed composite records;
- the same operation produces reusable structure gold without a parallel
  annotation schema; and
- provider/runtime configuration remains outside the structure engine.

