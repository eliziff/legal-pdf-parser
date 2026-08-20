# Structure-engine parity gate

The corpus-wide acceptance order and anti-skip/staleness rules are specified in
[`PARITY_PLAN.md`](PARITY_PLAN.md).

This is the fast first gate for factoring structure derivation out of the PDF
parser. It reconstructs the structure engine's consumed common-input fields
directly from the settled digital-native extraction cache, so it does not parse
PDFs, run OCR, or call a model. The frozen sample covers all five corpus jurisdictions and eleven
document shapes, including the manually audited submission, court form, rules,
presentation, zoning-map, research-report, agreement, and transcript cases.

Each run performs two exact-byte digests of all 401 pages, compares their byte
counts and hashes to `baseline.json`, measures executable startup, and runs the
repository's warm `cargo quick` gate. The Rust process serializes the unchanged
pretty JSON directly into SHA-256, so no raw replay output reaches disk.

```powershell
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --self-test
python legal-pdf-parser\experiments\structure-engine-parity\harness.py
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --all-cache --fresh
```

Windows build and focused-test commands in this program run through
`run_bounded.py`. It sets the runner and child BelowNormal, places the process
tree in a kill-on-close job, enforces the hard timeout, and writes a compact
receipt containing priority observations plus hashes of the runner, command,
and captured output. It never records environment values or raw command
arguments.

Only an intentional product-output change may replace the baseline:

```powershell
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --freeze
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --all-cache --fresh --freeze
```

The all-cache acceptance lane covers all 748 successful native extraction
caches. It reads their aligned page/line evidence directly into warmed Rust
processes, digests exact pretty bytes, and keeps compact per-document receipts
under `.tmp`. Heavy and light lanes overlap with three 128 MiB heavy batches
and up to six 160 MiB light batches. `--fresh` ignores receipts and
fails closed below 1,000 pages/second or above 30 seconds. Without `--fresh`,
valid receipts resume interrupted work but never count as fresh speed proof.

The frozen all-cache denominator is exactly 748 extraction records and 748
document records: 24,707 pages and 1,221,262 lines. An independent baseline
check matched 748/748 outputs exactly with aggregate output SHA-256
`eca681b3...0d9d3`. The fresh one-process-per-document run took 168.628
seconds (147.3 pages/second), so it failed the unchanged 250-pages/second
performance gate. That failure is retained in the baseline receipt: the
refactor had to remove the per-document process and raw-output boundary rather
than normalizing this runtime.

The baseline locks exact common-input and replay-output byte counts and SHA-256
hashes per document. Binary identity and timing are receipts, not parity keys,
so a refactored binary passes only when it emits the frozen bytes while staying
inside the latency budgets. Quality improvements must first pass this gate on
the unchanged-output lane, then update the baseline explicitly alongside their
corpus-quality evidence.

The first fresh digest-only candidate (`f626562b...2232b5`) matched 748/748
documents and all 24,707 pages with aggregate output hash
`eca681b3...0d9d3`, but took 143.0 seconds (172.7 pages/second). The fixed
1,000-pages/second and 30-second gates rejected it. That run exposed eleven
heavy inputs incorrectly sharing one sequential batch; heavy inputs are now
isolated for progress and load balancing. No second fresh corpus run was used
to erase the failed measurement.

The same candidate's one post-edit `cargo quick` passed in about 10 seconds and
its cached production-feature release link passed in 62 seconds. Both remain
build-performance failures. The authoritative warm-build contract is a median
at most 2 seconds and p95 at most 4 seconds; this local harness's single-run
ceiling is the 4-second p95 bound. The release-link ceiling is 30 seconds.

The optimized candidate (`61ba9ee8...6f46b`) matched all 748 documents and
24,707 pages exactly, including aggregate output hash `eca681b3...0d9d3`, with
misalignment rejection and zero retained raw outputs. It completed the fresh
lane in 89.588 seconds (275.8 pages/second), so the unchanged 30-second and
1,000-pages/second gates still rejected it. On the 1,401-page worst case,
memoizing repeated x-window page counts reduced preparation from 5.282 to
0.325 seconds and candidate replay from 11.592 to 4.074 seconds without a byte
change. A broader repeated-anchor skip changed one of 748 outputs and was
rejected before this final run; it is not in production.

The next batch path consumes the frozen gzip extraction cache directly and
binds receipts to those exact compressed bytes; it no longer writes or reparses
the 1.478 GB synthetic common-input JSON. The bounded candidate
(`ae4c6a7c...145e5`) kept the worst 1,401-page output byte-identical at
`ad7ebccc...ec7d` and passed the eleven-document smoke at 409.0 pages/second
with aggregate output `ea4531f8...b109d`. The heavy case took 7.984 seconds,
versus 9.874 seconds for the former Python materialization plus Rust replay.
That 1.24x gain projects the heavy lane at roughly 48 seconds even with the
light lane overlapped, so no fresh 748-document run was spent or claimed.

Profiling split the exact heavy path into 2.821 seconds of gzip decode/typed
parse, 2.382 seconds of replay and sorted-value construction, and 1.427 seconds
of pretty serialization plus SHA-256. A rejected page-at-a-time wrapper moved
conversion into serialization and took 13.16--14.80 seconds. The promoted
dependency-free serializer instead walks typed fields in the frozen key order,
while `serde_json` still owns escaping and float formatting. It kept the exact
553,317,754-byte heavy hash and reduced that case from 7.984 to 5.528 seconds
with a 466,767,872-byte peak working set.

The direct scheduler caps each heavy uncompressed batch at 128 MiB and each
light batch at 160 MiB. Three heavy and six light workers may overlap, giving a
1,409,286,144-byte explicit uncompressed-evidence bound; raw outputs remain
forbidden. A fixed qualification of all eleven heavy documents plus the six
largest light documents matched 17/17 outputs across 11,495 pages in 18.379
seconds. Its measured aggregate controller-and-child peak was 1,258,016,768
bytes; the frozen 25% evidence bound is 1,572,520,960 bytes, below the 2 GiB
hard limit.

The typed candidate passed the eleven-document smoke at 474.6 pages/second with
the frozen `ea4531f8...b109d` aggregate output and rejected misaligned evidence.
The qualifier's short light burst projected a 22.459-second full replay at
1,100.1 pages/second, authorizing exactly one fresh full run. Candidate
`76808ab0...6f2245` then matched 748/748 exact outputs (24,707 pages and
7,803,025,160 output bytes), the frozen `eca681b3...0d9d3` aggregate hash, and
misalignment rejection with no raw outputs. Replay took 27.044 seconds at
913.6 pages/second (27.794 seconds total): the 30-second gate passed, but the
unchanged 1,000-pages/second gate failed. The brief qualification understated
sustained heavy/light CPU contention, so no retry was used; future projections
must keep the light lane busy for the entire measured heavy-lane wall.

An attempted 3-heavy/8-light qualification was rejected without changing the
harness. Its arithmetic input bound was 1,744,830,464 bytes, and adding two
160 MiB slots to the prior conservative evidence bound yielded 1,908,065,280
bytes, nominally below 2 GiB. In practice, all eleven heavy documents plus the
160 largest light documents remained oversubscribed after four minutes, so the
run was stopped. The qualifier buffered progress until completion; therefore
zero of 171 documents have an admissible durable exact receipt from that run,
and its actual completed count and aggregate peak are not recoverable. No raw
output or temporary manifest survived, and no retry was made. Future targeted
qualifiers must persist a bounded digest/peak receipt after each completed task
and enforce a wall timeout before testing additional concurrency.

Its one post-edit `cargo quick` took 2.759 seconds, fitting the 4-second p95
ceiling but not proving the 2-second median. The reported 50.121-second release
command was not a valid same-feature incremental link: it changed from the
existing `fast-allocator,kraken,ocr,ppdoc,ppdoc-openvino` fingerprint to
`kraken,ppdoc`, forcing about 42 seconds of library codegen in a cold feature
namespace; executable object/link work was about 4 seconds. No build was
rerun to manufacture a warmer result. Future build receipts must freeze the
exact feature set before measuring the 30-second incremental-link contract.

The frozen `0.3.0` binary (`85be89d2...075b4`) produced aggregate output SHA-256
`ea4531f8...109d`. The baseline run replayed 802 page-passes in 2.737 seconds
(293.1 pages/second), started in a 7.376 ms median, and completed warm
`cargo quick` in 0.799 seconds.
