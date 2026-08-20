# Structure-engine extraction and parity plan

This plan governs the factor-out of a shared structure engine from Legal PDF
Parser and SourceDocs. The first promotion step is deliberately behavioural:
move the settled derivation behind one boundary without changing one output
byte. Quality changes follow as separately measured commits. A cached replay is
not an end-to-end PDF proof, and a silver or candidate set is not gold.

## Acceptance sequence

1. Freeze the current public outputs and the exact input denominator.
2. Factor the structure engine while the 748-document native replay remains
   byte-identical and all existing SourceDocs provider tests remain green.
   Provider-native structure stays an input/oracle; it is not displaced.
3. Remove the process/raw-output bottleneck. A fresh full replay must cover all
   24,707 pages in at most 30 seconds and at least 1,000 pages/second on the
   same machine. Resumed receipts do not count as fresh throughput.
4. Exercise the public PDF/cache contract from source PDFs. This proves the
   extraction-to-common-input boundary that the replay intentionally bypasses.
5. Improve numbered sequences, footnotes/endnotes, pages, headings, citations,
   and geometry-derived unnumbered paragraphs. Each quality change gets a
   before/after corpus receipt and an explicit baseline update only after the
   unchanged-output refactor has landed.
6. Run provider breadth in SourceDocs. For every provider, compare native
   structure, the shared engine's derived structure, and the integrated product
   result; require monotonic sequence coverage without regressing a non-native
   path.

Rust is justified for the deterministic, data-parallel hot path only after a
profile identifies it. Provider adapters, orchestration, and grammar authoring
stay outside that core. `cargo quick` remains the iteration gate; no full build
belongs in the edit loop.

## Frozen native parity gate

`harness.py --all-cache` consumes exactly 748 extraction and 748 document gzip
records under `.tmp/digital-native-structure-audit`, comprising 24,707 pages
and 1,221,262 extracted lines. It reconstructs only the common input consumed
by `_parity-replay`; it never reparses a PDF. The denominator SHA-256 is
`87d3ca92f568b5077b45df44382bea09cac89bfa59d6424c7a21aab51026b4df`.
The aggregate common-input, output, and source hashes are respectively:

- `f3c528077cbc4a65a2ba75329e2c03165cb19892dfcd28dc2e93f675ba2e9b24`
- `eca681b34db5186d7689f1532f401c9e311c4aa85a11a73a151b0dd29c50d9d3`
- `37cd41fb3201ca9e41bd041a4195fcab757cc48c3d82c9ae4992000eaa544785`

The frozen binary SHA-256 is
`85be89d29d6cfde928eaac66a7834d26568bbca25b467fbccf6945b1bd3075b4`.
An independent check matched 748/748 exact output byte streams. The fresh run
used no retained raw outputs, observed at most 659,092,968 temporary bytes for
one document, bounded uncompressed inputs in flight to 201,415,295 bytes, and
left about 1.35 MB of resumable receipts. It processed 24,707 pages in 168.628
seconds (147.3 pages/second), correctly failing the former 250-pages/second
speed gate. The refactor must make this lane persistent, batched, hash-only, or
in-process; lowering the gate is not an acceptable fix.

The first digest-only candidate (`f626562b...2232b5`) proved 748/748 exact
outputs and aggregate hash `eca681b3...0d9d3` without retaining raw output,
but still took 143.0 seconds (172.7 pages/second). The gate failed unchanged.
Eleven heavy inputs had been placed in one sequential batch, so the harness now
isolates heavy inputs and reports each completion; this correction is not a
license to claim the 30-second target without another explicitly fresh run.
The candidate's post-edit `cargo quick` took about 10 seconds and its cached
production-feature release link took 62 seconds. The authoritative warm-build
budgets are median at most 2 seconds and p95 at most 4 seconds; the release-link
budget is 30 seconds. These results fail without a retry or relaxed threshold.

The next optimized candidate (`61ba9ee8...6f46b`) preserved all 748 exact byte
streams and aggregate hash `eca681b3...0d9d3`, rejected misaligned evidence,
and retained no raw outputs. Its fresh replay took 89.588 seconds (275.8
pages/second), still failing the unchanged 30-second and 1,000-pages/second
gates. The heaviest 1,401-page replay improved from 11.592 to 4.074 seconds;
preparation fell from 5.282 to 0.325 seconds after exact x-window count
memoization. A more aggressive anchor skip produced 747/748 parity and was
discarded rather than incorporated or normalized into the baseline.

The following direct-cache candidate (`ae4c6a7c...145e5`) removed the Python
compact-common serialization, temporary file, and Rust reparse from the batch
path. Receipts now bind the gzip extraction bytes that are actually consumed.
The 1,401-page worst case remained exact at `ad7ebccc...ec7d` but took 7.984
seconds, only 1.24x faster than the former combined 9.874-second path. The
eleven-document smoke remained exact at 409.0 pages/second. Because the prior
heavy lane was about 60 seconds, the measured ratio projects it near 48 seconds
even while the light lane overlaps; the required evidence for another fresh
748-document run was therefore absent, and no such run was made.

The heavy profile measured 2.821 seconds for gzip decode/typed parse, 2.382
seconds for replay and recursively sorted value construction, and 1.427 seconds
for exact pretty serialization plus SHA-256. The sink already sustained about
388 MB/second. A page-at-a-time sorted wrapper retained the exact
553,317,754-byte hash but took 14.80 and 13.16 seconds in two observed runs,
because it moved recursive conversion into serialization rather than removing
it. Its 29.85-second check and 2m39s link also failed the build gates, so the
prototype was discarded. A future attempt must serialize model fields directly
in the frozen sorted order with compact source and demonstrate reduced peak
memory; another wrapper around `serde_json::to_value` is ruled out.

The present overlapping scheduler bounds uncompressed evidence to 896 MiB
(two 128 MiB heavy batches plus four 160 MiB light batches) and retains no raw
output. Four-heavy concurrency remains prohibited until measured total peak
memory, not merely input bytes, fits below 2 GiB.

A true no-edit warm `cargo quick` took 3.224 seconds; the one post-edit check
took 2.221 seconds, and the production-feature link took 44.24 seconds. The
single quick samples are below the 4-second p95 ceiling but do not establish the
2-second median; the link remains over its 30-second ceiling. The source-budget
counter reported whole-project production at 125,883 lines, below the 125,896
hard ceiling, and LPP production at 30,803 lines. Its separate test/authored and
subrepo-pin checks remain red and are not relabelled as production-budget
successes.

The eleven-document smoke gate remains the edit-loop check: 401 pages across
all five jurisdictions and manually audited document shapes, with startup,
replay, and warm `cargo quick` budgets. It does not replace the 748-document
acceptance gate.

## Local evidence inventory

Hashes below are raw file SHA-256 unless marked **canonical**. A canonical hash
is SHA-256 over a sorted, newline-delimited inventory of the named fields; the
inventory recipe and fields must be preserved in the eventual gate receipt.

| Surface | Exact local denominator and identity | What a complete run proves |
| --- | --- | --- |
| Materialized legal PDF corpus | 1,500 PDFs, 111,542 pages, about 6.53 GB; 750 digital-born and 750 non-digital. Canonical `{relative_path,sha256,page_count,generation,jurisdiction,kind,source}` hash `30b303c37299f671d22d3e25e73ee528e2b0ddaf10002325f425ccdfe4243450`. | The product-scale end-to-end denominator. Native extraction must report 750 successes/failures; current replay covers 748 successful caches. The non-digital half is the settled Kraken-lite OCR lane. |
| Acquisition ledger | 4,693 rows; 1,735 rows marked accepted; file hash `2b6e1bcf44cd0c34db94e7e5a677b07139b3f231bb5a84ce1a484305ba0076fa`. Exactly one accepted row maps to each of the 1,500 materialized PDFs. | Provenance/source identity only. It is not the run denominator because 235 accepted rows are not materialized. |
| Native extraction cache | 748 extraction plus 748 document records, 24,707 pages; the full frozen hashes are in the preceding section. | Exact structure derivation from settled common input. It cannot prove current PDF extraction or the two native parser failures. |
| Deterministic cross-lane sample | 120 PDFs, 10,435 pages, 60 native and 60 non-digital; manifest hash `72c8ad4cff174b15c732d1cb40bf4963febf954033e48303a82c5675aeb00e9c`. | A bounded end-to-end cross-lane gate only when all 120 are executed. The present cache has 36 extraction/document pairs (1,627 pages; canonical cache hash `6c227c6afc55a9ad92e29eb96f3156b8c8f7fe7d2bb0be1ae0eb480bc6bb1ab5`) and the historical summary completed only two, so neither is a 120-document result. |
| Legal generalization corpus | 31 multi-format sources; manifest hash `24b4bcb2572d4b0f2a86b901c2a49e81bf1664bf1e9828c69b230e9bd21fc0f0`. Seven are PDFs; the cache-contract set adds the MAUD README PDF. | Provider breadth across PDF and SourceDocs formats. Results must be reported per provider and source format, not only as an aggregate. |
| Public cache-contract set | 8 PDFs, 425 pages; manifest hash `aab25b794d7d47e543019d57f044f2e83bf58775e972e904d39875c9ead6a9f9`; all eight files are present. | Cold/warm/prepare/selected-page/lookups/corrupt-cache/source-identity behavior through the public `legalpdf contract` boundary. Its old 14,264-call, 4,127.5-second method is too slow and must become a batched/persistent acceptance gate before reuse. |
| Canadian structure gold | 18 JSON gold records paired with 18 texts: 8 decisions and 10 statutes; canonical file manifest hash `0a22f15ba6c033303ef5a6e7420fef64a949824f2940d50e72e00fdd7b591a5a`. | Shared-engine sequence/heading quality for the paired legal texts. It does not prove PDF extraction. |
| Ordered journal gold | `gold_lines_slim.jsonl.gz`: 661 pages, 350 articles, 32,553 lines; hash `121acce1069540de0f597c7d067acdce8312e74ff7e735a6c2d9fec700df7c41`. | Full ordering differential with perfect region labels through `dev/bench_order_gold.py`. This is journal geometry evidence, not universal structure parity. The local set is revision `_01`; OCR planning treats `_05` as accepted, so the revision must be reconciled before any OCR-quality claim. |
| Legal25 layout holdout | 87 PNG pages, 975 annotations, 25 categories; COCO annotation hash `e7242d2418c8a6e3078385aeda2707cbe3b8a9c0745184ff203decdff5a520a5`. | Region detector layout accuracy. The region-consumer ablation joins only 48 pages/20 articles/2,650 manual lines and cannot be reported as an 87-page line-role, ordering, or footnote-pairing result. |
| Real-document layout surface | 1,500 images and 1,500 annotations; CSV manifest hash `b58d11b29cc13a8e7d2428c9596a01386bc2887cdadd1a67e7b03e038d57f957`. | Broad region-detector scoring only. Image count is not a semantic-structure denominator. |
| Harvested grammar vectors | 271 rows: 11 guard-negative, 31 pure-reference, 162 raw-string, 31 splitter-I/O, 36 TOA-I/O; hash `13c4fdf44c47624550039fa3470e2574abc18430f7c185cd585ae013a6494d5f`. | Parser/grammar regression only when the command receipt proves all 271 rows were asserted. The existing oracle test models only 18 of 31 splitter rows and must not be described as full coverage. |
| Authored grammar corpus | 64 entries/252 vectors; hash `8e6da9011c1cf78c609d54d53abb67b7a3e50f9a67cbf48cd72ab8136b16606f`. | Exact grammar-vector spans and status across runtimes, with a frozen row count and hash. |
| Kraken-lite validation pointers | 6 images; manifest hash `f2809c957ecb9e8ece7603673406c4968834089b0e30d5016983593cf7b742bf`; all pointed-to images presently exist. One repository-local parity image has hash `3b89b2ab9b5857f2b3e40a343db90cd80c41b3e9f68d3a4630fbbadd3a937ca4`. | Six-image runtime validation, provided absolute-path inputs are hashed at run time. Absolute paths make the manifest non-portable and vulnerable to stale substitution. |
| Kraken benchmark splits | Benchmark 153, heldout/manual gold 55, validation/manual gold 68, silver 30, probe 12. List hashes: `833c3ad3cb3b89504d574398a0ff78dec5233238a68c00726f2107704139f1b0`, `88319cd7f9c5ddfb4627d27d7dfc1567e1b4b5936ddf7a79e427a4993045a98b`, `4100c1dfb6bd9c5975a93b83b9139b6f3409de50c4ab259d589c4fb587453512`, `7a27e0f7434d3d932ef7aa420ffee9926e3be566e1f99080025f32432df426ba`, `d268f7adc032701df25795860e7123afbe6a591703d2117b13e9cd9d68d6342a`. | OCR runtime comparisons on explicitly named splits. Silver is not gold, and aliased files with identical contents do not double the denominator. |
| Scan-silver | 42 PNG/XML candidates; candidate-list hash `dee741c484a52fa36cec703f573d3815e2e94c12b31e8cd3b082580a469f1770`; zero verified pages. | Mechanics/silver generation only, never quality acceptance. |
| Court scan corpus | 115 PDFs, 650 PNGs, 290 XMLs; only 4 verified pages, list hash `3cccefd6442f314db819a061f2b7069f06dbcb00d9c689d16687f69d5f54b866`. | Corpus mechanics on all artifacts; OCR quality on exactly four verified pages. |
| CourtListener scan silver | 33 PDFs and 270 candidate page/image/XML/OCR groups; candidate hash `3f7a8fa480bf0c58224a06b44191b1c94aca5bbce8404b9618f44c12617ce7bc`; 26 verified pages, hash `084642088389b24f2b0a172e20993c0ca6570dfd33121995aa867d1b84379e47`. | OCR quality on 26 verified pages; silver generation/mechanics on 270 candidates. |
| External digital-born URLs | 29 URL rows; hash `d74c4a6eda7636763558f1d17926992c997f9a3f75ebfdbdbe8d195f9c511537`. No source PDFs are locally materialized. | Nothing can be rerun locally from URLs alone. The historical 29-document/1,445-page result is evidence, not a current acceptance run. |
| Small parse caches | E2E smoke: 1 doc/3 pages (TSV inventory hash `f818228af8298e9674b4c27cf428079880e886e9c7884192e97d9884c02d43ab`). Three Kraken OCR smokes contain the same 1 doc/2 pages (`a2e5d634bbe4068ace82b94fb07bc696b0f24f05fb94c7d7aba4608c9af75a6d` each). Other Kraken smokes are 2 docs/2 pages (`e11b4008536b9321340f85109bc0d1461b62696e1280bfd81484d564c8c7b778`), 1/1 (`7c6afdf69c1b5d55fb8943d5de5ebbd241abedd64a21dbc7fa93d6ac247f7328`), and 2/2 (`943c39bfa1a02ca8a68c474b6c7a5072849254d7d1986824c7ac23b4cbe77891`). Release app projection cache: 7 profile pairs/163 counted pages but only 5 unique sources (`6dd6ba7f65b6fea00128470ee53486296c2d43fe99b6f70794f82e50913bdc9f`). These hashes cover sorted UTF-8 TSV rows `{lane,name,compressed_bytes,file_sha256}` with a final newline. | Developer smoke only. Duplicate runs and provider/profile variants must not inflate a corpus/document count. |
| Synthesized unit fixtures | 103 Python `test_` functions and 156 first-party Rust `#[test]` declarations found in source; DOCX/PDF fixtures are synthesized and no durable DOCX corpus is retained. | Public behavior claimed only from an executed command receipt. Source declaration counts vary with features/configuration and are not pass counts. |

The historical 1,024-page-list journal holdout and its 27,391-page result are
not locally materialized; they cannot be rerun as a current gate. No local
`parse-v1` cache was found in the inspected application-data surface.

## Anti-cheat and staleness requirements

- Every acceptance receipt binds the exact ID/path set, source bytes, page
  count, provider/generation, extraction bytes, binary bytes, feature/profile,
  and schema version. Path or modification time is never an identity.
- Expected and actual sets must be equal in both directions. Missing inputs are
  failures; no acceptance loop may use `continue`, `skipUnless`, a sample flag,
  or an environment filter to shrink the denominator. Require
  `successes + failures == expected` and `failures == 0`.
- A cached-common-input run proves derivation only. Current end-to-end parity
  additionally requires source-PDF extraction and public product/contract
  gates. Conversely, a public product smoke does not replace exact internal
  structure parity.
- Exact parity hashes raw output bytes without parsing, sorting, whitespace
  normalization, or dropping nondeterministic fields. Semantic quality metrics
  are a different lane and may intentionally change bytes.
- Baselines change only under an explicit freeze command. Freeze reports no
  exact-match count; a separate independent check must compare the new
  baseline. A mismatch never auto-updates expected output.
- Resume receipts bind binary SHA, extraction/input SHA, source/path, page
  count, and receipt schema. Report fresh and resumed documents separately.
  Resumed work is never included in fresh throughput.
- Provider/profile caches are counted by declared `(source, provider, profile)`
  identity; duplicate physical inputs or aliased split lists cannot inflate
  document or page totals.
- Candidate and silver counts measure generation coverage. Quality denominators
  are only manually verified/gold records, with their revision and model
  identity recorded. Absolute-path inputs are rehashed when the gate starts.
- Long runs write bounded, resumable receipts and retain failures, but do not
  duplicate raw outputs. Progress includes completed, failed, pages, and
  elapsed time.
- Build/test claims record the exact command, binary/features, executed test
  count, duration, and exit code. Counts obtained by searching source do not
  count as execution evidence.
