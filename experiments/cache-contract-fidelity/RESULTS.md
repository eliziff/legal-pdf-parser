# Source-PDF cache contract fidelity

This gate exercises the public `legalpdf contract` boundary only. It uses the
complete eight-PDF legal-generalization corpus: the seven PDFs under `raw/`
and the MAUD corpus README PDF. Those are the eight documents and 425 physical
pages used by the earlier projection gate; no document or page is sampled.

The harness checks:

- exact cold/warm `source_doc` equality except `source.cache_hit`;
- exact direct-parse versus `prepare`-then-`source_doc` equality;
- exact full-cache versus selected-page lookup results for every physical page
  with zero, one, and two context blocks;
- every paragraph locator, every published section locator/alias, and every
  footnote locator with its derived occurrence;
- source hash, profile cache-key separation, stable unit IDs, exact unit text,
  page bindings, ordering, and aggregate canonical digests;
- recovery from corrupt document caches and from simultaneous corrupt document
  and extraction caches; and
- source-byte cache-key separation without changing parsed content.

Raw request/response records and caches are ignored under `raw/`. Records are
content-addressed and resumable; progress is printed at least every ten pages.
Parser child processes run at below-normal priority on Windows.

```powershell
python legal-pdf-parser/experiments/cache-contract-fidelity/harness.py --self-test
python legal-pdf-parser/experiments/cache-contract-fidelity/harness.py
```

## Complete result

**PASS** on 2026-08-19 against parser `0.3.0`.

| Identity | Value |
| --- | --- |
| Release binary SHA-256 | `ea9e0a1f7c0ca6585074256200dca42b97ec62709c566d62c1613a26644d7321` |
| Run ID | `0b82b3418d433a7b488e` |
| Corpus SHA-256 | `23a2adfea15aebcc6ac1b0a1a6f3c4fd2e517edf9449e97eb0401a1f1d06d321` |
| SourceDoc aggregate SHA-256 | `ec28a3c821fb7914a2a6ddacfe90cf4178d046d77c775c9fc207d709eef788ee` |
| Lookup aggregate SHA-256 | `1b6cd2b89adc19ef6c595aa9c7f06f3c0cb2aca22864cb0202fd2516e57fbbf7` |

The fresh run executed 14,264 contract calls with no resumed calls. It covered
all 8 documents and 425 pages, including 1,275 page/context comparisons,
7,567 paragraph blocks, 45 section blocks, 677 footnote blocks, and 10,365
structural queries. The run took 4,127.5 seconds with every parser child at
below-normal priority.

| Document | Pages | Page/context comparisons | Paragraphs | Sections | Footnotes | Structural queries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MAUD README | 7 | 21 | 80 | 7 | 3 | 106 |
| Bhasin v Hrynew | 67 | 201 | 1,344 | 0 | 2 | 1,352 |
| Wastech Services | 67 | 201 | 624 | 0 | 3 | 636 |
| SEC v Honig | 106 | 318 | 2,279 | 0 | 304 | 3,495 |
| SEC comp25666 | 9 | 27 | 127 | 0 | 34 | 263 |
| ILND 05-cv-03198 | 10 | 30 | 172 | 0 | 4 | 188 |
| TXSD 10-cv-04728 | 75 | 225 | 1,520 | 0 | 34 | 1,656 |
| TXSD 16-cv-01947 Westport | 84 | 252 | 1,421 | 38 | 293 | 2,669 |

All cold/warm and direct/prepare SourceDoc envelopes were canonically identical
after removing only `source.cache_hit`. Every full-cache page lookup was value
identical to both the cold and warm selected-page lookup. Every published
paragraph, section, footnote anchor, and occurrence-qualified footnote alias
resolved with stable IDs, exact UTF-16 text slices, ordered page bindings, and
ordered units.

The gate also passed 16 corrupt-cache rebuilds: one document-cache-only and one
document-plus-extraction-cache recovery for every PDF. Each rebuilt value was
identical to its cold value and became a value-identical warm hit. Full-document
and selected-page profiles had distinct, stable cache keys. Appending a benign
PDF trailing comment changed both source SHA-256 and cache key while preserving
the SourceDoc exactly:

- original key: `a43c14dc34a22cd7ea0a0eab0a4ab48ca30ab547a47da169b106bb4745cfb8bc`;
- changed-source key: `57b0bac8cfec3a7e3604ef79a04fcdc858f745a74dd944d3a56244971b8692ca`.

## Defects exposed before the passing run

The exhaustive gate found four contract defects that smaller checks had missed:

1. selected-page `structure_lookup` requests were rejected because `pages` was
   absent from the strict request allowlist;
2. emitted `section:<section.id>` locators did not resolve as exact section IDs;
3. paragraph numbering drifted after empty or marker-only paragraphs because
   SourceDoc and lookup used different paragraph projections; and
4. an oversized final heuristic section was published even though its exact
   locator could only return the 60,000-character response-limit error.

The passing binary uses the same paragraph projection for SourceDoc and lookup,
supports exact bounded section IDs, and publishes only section blocks that the
public lookup contract can return. Westport retains all 136,068 UTF-16 source
characters and all 1,421 paragraph blocks while publishing 38 bounded section
blocks instead of the unusable oversized block.
