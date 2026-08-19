# Structure-engine parity gate

The corpus-wide acceptance order and anti-skip/staleness rules are specified in
[`PARITY_PLAN.md`](PARITY_PLAN.md).

This is the fast first gate for factoring structure derivation out of the PDF
parser. It reconstructs the structure engine's consumed common-input fields
directly from the settled digital-native extraction cache, so it does not parse
PDFs, run OCR, or call a model. The frozen sample covers all five corpus jurisdictions and eleven
document shapes, including the manually audited submission, court form, rules,
presentation, zoning-map, research-report, agreement, and transcript cases.

Each run performs two exact-byte replays of all 401 pages, compares their bytes,
checks their hashes against `baseline.json`, measures executable startup, and
runs the repository's warm `cargo quick` gate. Temporary inputs and outputs are
deleted as soon as each document's receipt is assembled and are limited to 128
MiB per concurrent document. Default budgets fail closed at a
100 ms median startup, 250 replay pages/second, and a 5 second warm Rust check.

```powershell
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --self-test
python legal-pdf-parser\experiments\structure-engine-parity\harness.py
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --all-cache
```

Only an intentional product-output change may replace the baseline:

```powershell
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --freeze
python legal-pdf-parser\experiments\structure-engine-parity\harness.py --all-cache --freeze
```

The all-cache acceptance lane covers all 748 successful native extraction
caches. It writes only one replay output at a time, hashes and deletes it, and
keeps compact resumable receipts under `.tmp`. Inputs over 20 MiB run at no
more than two at once with a 1 GiB per-document temporary ceiling; the rest use
the requested job count with a 160 MiB per-document ceiling. No raw replay
output is retained.

The frozen all-cache denominator is exactly 748 extraction records and 748
document records: 24,707 pages and 1,221,262 lines. An independent baseline
check matched 748/748 outputs exactly with aggregate output SHA-256
`eca681b3...0d9d3`. The fresh one-process-per-document run took 168.628
seconds (147.3 pages/second), so it failed the unchanged 250-pages/second
performance gate. That failure is retained in the baseline receipt: the
refactor must remove the per-document process/pretty-JSON boundary rather than
normalizing this runtime.

The baseline locks exact common-input and replay-output byte counts and SHA-256
hashes per document. Binary identity and timing are receipts, not parity keys,
so a refactored binary passes only when it emits the frozen bytes while staying
inside the latency budgets. Quality improvements must first pass this gate on
the unchanged-output lane, then update the baseline explicitly alongside their
corpus-quality evidence.

The frozen `0.3.0` binary (`85be89d2...075b4`) produced aggregate output SHA-256
`ea4531f8...109d`. The baseline run replayed 802 page-passes in 2.737 seconds
(293.1 pages/second), started in a 7.376 ms median, and completed warm
`cargo quick` in 0.799 seconds.
