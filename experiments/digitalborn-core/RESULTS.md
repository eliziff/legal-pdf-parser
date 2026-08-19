# Digital-born core experiment

The frozen inputs are `external_sources.jsonl`, the independent DOCX fixtures,
and the ignored `_temp/journals-heldout-1024.jsonl` journal manifest. Downloaded
PDFs and raw benchmark output stay under ignored `_temp/` paths.

Release figures are CER, footnote pairing precision/recall/F1, reading-order
accuracy, table fidelity, image detection and vision-routing recall, wall time,
peak memory, and production source lines. Results are recorded here only after
the corresponding held-out gates have completed.

## 2026-08-14 Rust digital-born qualification

The frozen journal qualification set contains 1,024 articles from 40 datasets
(27,391 pages); 929 articles contain paired-note evidence. The accepted Rust
run completed every article. Against the shipped evidence it matched the
Python product exactly for CER/WER, reading order, footnotes, images, and
vision routing. Table precision improved on 66 articles and recall improved
on nine, with no evidence-backed recall regressions. The only apparent table
precision regression is a real 19-by-3 police-code table in
`journal:UBC-L-REV:7317` that the frozen evidence omitted.

Aggregate journal results:

- source CER: 0.015202 mean, 0.001620 median
- source WER: 0.021591 mean, 0.003977 median
- reading order: 0.948766 anchor recall, 0.993639 pairwise accuracy,
  0.996395 adjacent accuracy
- note labels: 0.955722 precision, 0.975931 recall, 0.954846 F1
- note reference pages: 0.944823 F1
- tables: 0.767259 precision, 0.969543 recall, 0.766381 F1, 0.971665
  shape accuracy
- images: 0.688460 precision, 0.995907 recall, 0.691367 F1; vision-routing
  recall was 1.0 on image-positive articles
- throughput: 72.42 pages/s by summed child time (71.74 pages/s end to end)
- peak RSS: 38.2 MiB median, 55.0 MiB p95, 348.6 MiB maximum

The final memory-optimized run had zero case-metric differences and zero
product-count differences from the accepted 1,024-article run. Its summed
child time fell from 381.19 to 378.21 seconds and p95 RSS fell from 62.60 to
54.99 MiB; the 0.25 MiB median-RSS increase and 0.03 MiB maximum-RSS decrease
are immaterial.

The separate external set deliberately excludes fillable IRCC forms. It has
29 substantive Canadian legal PDFs (1,445 pages): four factums, three tribunal
reasons, six securities filings, eleven regulatory submissions, two commission
submissions, and one each of legislation, an administrative manual, and a
government briefing binder. Three clean release runs completed all 29 with a
95.17 pages/s sum-of-case-medians throughput and about 29.2 MiB median RSS.
That is 38.4% faster than the earlier 68.78 pages/s external baseline. All 232
product sidecars remained byte-identical after the page-range and memory
optimizations.

The 179-page National Bank circular exposed lopdf retaining 209 MiB of sparse
tagged-PDF ParentTree arrays that extraction never reads. Dropping only arrays
with at least 4,096 entries, at least 95% `Null`, and no values other than
`Null` or references reduced controlled median Rust peak RSS from 269.10 to
134.89 MiB, below the Python oracle's 177.35 MiB. Median wall time remained
1.63 seconds versus the oracle's 4.93 seconds.

Two qualifications remain. The frozen journal output is generally strong but
is not uniformly gold, so disagreements are adjudicated from source evidence
rather than forced to match. Also, the National Bank base prospectus still
produces three empty, unreferenced label-only notes (`30`, `32`, and `39`);
broad suppression rules were rejected because they removed genuine journal
notes. This remains an open release gate, not hidden as parity.

## 2026-08-14 content-operation ownership pass

A 156-page APPEAL issue exposed the remaining memory outlier. Its map page has
a 5.65 MiB decoded content stream containing 319,056 operations and 78,847
vector segments. The semantic output was already correct; memory was lost by
cloning lopdf's entire operation tree before Form expansion and by reaching
the largest stream only after retaining 147 pages of results.

The accepted fix moves decoded operations into the Form expander, borrows
comment-free stream bytes instead of copying them, processes the largest
encoded page streams before accumulated results, and restores page order from
page buckets. It discards no text, glyph, or vector evidence. A single-path
segment cap and mimalloc as the default allocator were both rejected: the cap
did not address the document's many small paths, while mimalloc reduced warm
wall time but raised median peak RSS from 195.35 to 268.71 MiB on the map-page
ablation. Mimalloc remains opt-in as `fast-allocator`, never the release
default.

Final exact-binary gates:

- 1,024/1,024 journals and 8,192/8,192 product sidecars were byte-identical to
  the accepted run.
- Journal throughput increased from 72.422 to 81.702 pages/s (+12.81%).
- Journal median RSS fell from 38.168 to 37.699 MiB, p95 from 54.633 to
  54.152 MiB, and maximum from 348.555 to 246.020 MiB (-29.42%).
- The APPEAL outlier's controlled three-run median was 2.204 seconds and
  210.01 MiB versus the Python oracle's 7.441 seconds and 232.96 MiB.
- Three external-corpus runs reached 99.31 pages/s by sum of case medians,
  +4.35% over the prior 95.17 pages/s accepted run. Median case RSS was
  28.91 MiB, p95 was 112.24 MiB, and maximum case-median RSS was 142.98 MiB.
  All 232 product sidecars remained byte-identical.

The frozen structural ledger now maps all 25 Python source/data files (594
symbols) to their Rust counterparts with zero identity errors and zero
blocking gaps. The former benchmark module is superseded by the differential
harness, and the Microsoft Word export script remains only a DOCX gold-fixture
generator; neither is linked into the production engine. The release library
also passes Clippy with warnings denied. Deprecated pdf-inspector compatibility
entry points were removed from the vendored backbone rather than retained as
shims.

## 2026-08-14 interpretation profile

A three-run, per-document-median profile over seven retained legal PDFs (418
pages) excluded sidecar serialization and separated the raw
`pdf_inspector::extract_fidelity_from_doc` call from downstream product
interpretation. Raw extraction was 42.3% of parse CPU and downstream work was
56.4%. Within legal interpretation, footnote pairing consumed 54.3%, repeated
furniture detection 23.3%, and page classification 14.9%.

The accepted allocation pass replaced expanded weighted-font samples with an
exact weighted selection, normalized furniture text once, cached per-line font
sizes, computed grouped-line medians once, removed one line-text clone, and
constructed word boxes without temporary vectors. In the paired warm run,
downstream interpretation fell from 1.5457 to 1.3985 seconds (-9.5%), legal
interpretation from 0.6170 to 0.5278 seconds (-14.5%), and total parse
throughput rose from 158.0 to 167.94 pages/s (+6.3%). All seven extraction and
replay products remained byte-identical. Raw outputs were discarded.

## 2026-08-14 compact parse cache

The parser's former cache was the full publication directory. It is replaced
by independently keyed extraction and legal-document records, each one atomic
gzip-compressed JSON file. CLI parses are cache-free unless `--cache` or
`--cache-dir` is explicit; enabled parse caches are access-evicted above 1 GiB.

A bounded debug-binary check on the retained 67-page Bhasin SCC PDF produced
two cache files totalling 1,469,082 bytes. The cold explicit compact
publication took 9.4300 seconds and the warm publication 0.9163 seconds
(10.29x); all eight structural collection payloads were byte-identical. These
figures include requested publication serialization and verify cache behavior,
not release-parser throughput. The temporary cache and both publications were
deleted immediately.

## 2026-08-14 byte-identical structure fast path

The pairing hot path now decodes each line into characters once, uses linear
regexes for protected citations and one `RegexSet` for boolean citation
signals, and performs punctuation tests without temporary formatted strings.
The matched phase profiles fell from 2.0088 to 0.7747 seconds for protected
citation scanning (-61.4%) and from 0.8896 to 0.3518 seconds for reference
candidate extraction (-60.5%). The profiled pairing build fell from 4.8531 to
2.0910 seconds (-56.9%).

A three-run, per-document-median release replay of the seven frozen common
inputs (418 pages) measured the production normalized-page structure stage at
0.7030 seconds: 0.3077 seconds to prepare pages and 0.3953 seconds to derive
legal structure, or 594.6 pages/second. The replay-only page clone cost 0.0444
seconds and is excluded. All seven result files remained byte-identical; the
single reused output file was deleted.

The build graph now keeps release incremental objects while pinning 16 codegen
units for runtime quality, uses `rust-lld`, omits pdf-inspector's unused cdylib
and CLI logger graph, and compiles the native Tesseract layout shim only with
the `kraken` feature. The first cache re-key was intentionally bounded and did
not complete within 180 seconds; no warm final-build figure is recorded yet.

## 2026-08-14 deep interpretation allocation pass

A counting-allocator profile of the same seven retained common inputs (418
pages) found two startup costs disguised as page work. The aligned-furniture
search repeatedly allocated page clusters, and the first uncommon citation
compiled a single alternation containing the bundled McGill reporter names. The
accepted implementation reuses furniture scratch storage, lazily compiles only
the reporter initial actually witnessed by a citation-shaped prefix, and keeps
the five protected-citation grammars independently lazy behind necessary
witnesses. The underlying citation regexes and their match order are unchanged.

Measured with the allocation profiler, aligned-furniture allocation fell from
144.27 MiB to 0.27 MiB (-99.8%). Furniture citation-note allocation fell from
93.11 MiB to 18.29 MiB (-80.4%), and the enclosing furniture-application phase
fell from 102.83 MiB to 28.01 MiB (-72.8%). A one-line probe had shown that a
generic `Text 1` line previously initialized all five protected-citation
automata (9.96 MiB and 49,998 allocations); the new necessary-witness routing
does not initialize inapplicable grammars.

The same pass caches line font sizes, uses stack storage for the weighted font
median observed on every retained line, replaces two quadratic pairing scans
with direct indexes, defers rejected outline JSON construction, materializes
accepted digit strings only once, and performs citation and furniture
normalization in one pass. A combined RegexSet prefilter was rejected after it
added 53.8 MiB of process-start allocation and slowed line construction.

All seven final release results remained SHA-256 byte-identical. The bucketed
McGill matcher was also checked call-for-call against the former full regex on
the retained inputs before the oracle was removed. All 124 Rust library tests
passed. Three independent release benchmark batches varied with laptop
contention from 0.4148 to 0.6958 seconds for the product stage; the median was
0.5350 seconds, or 781.3 pages/second. Against the prior 0.7030-second record,
that is 23.9% less product time and 31.4% more throughput. The replay-only page
clone and output serialization remain excluded, matching the prior protocol.
Generated replay output was deleted.

## 2026-08-14 McGill 10th reporter and journal correction

The bundled 779-entry reporter list was incomplete. The `Reporters & Journals`
sheet in the McGill Guide (10th) appendices workbook contains 2,144 titled rows
and 2,110 ordinal-distinct abbreviations. Only 777 were present in the old
bundle, leaving 1,333 real reporter and journal abbreviations absent. The two
old bundle-only values were mojibake forms of `CPJI (Sér A)` and `CPJI (Sér B)`.

The replacement inventory excludes navigation and section-header rows, folds
duplicate abbreviations, and retains the matcher-required decreasing-length
order. Its extraction rule and fingerprint are recorded beside the data. The
inherited abbreviation compiler also discarded apostrophes, colons, commas,
and accented letters; 153 entries therefore could not match their own canonical
spelling. The Rust and Python compilers now preserve those characters while
retaining flexible acronym punctuation and whitespace. All 2,110 abbreviations
self-match in the exhaustive Python check, and Rust vectors cover journal,
apostrophe, colon, comma, and accented-name cases.

All seven retained release products (418 pages) remained byte-identical after
the correction. All 125 Rust library tests and the focused Python inventory
tests passed. A fresh three-run, per-document-median release replay measured
0.5838 seconds for the product stage, or 716.0 pages/second: 17.0% less time and
20.4% more throughput than the 0.7030-second / 594.6-pages-per-second baseline.
