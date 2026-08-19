# CPU OCR benchmark

Measured 2026-08-19 on two ordinary Windows machines. The purpose is one fair,
reproducible comparison of the shipped Kraken Lite tiers with native Tesseract.

## Result

| OCR engine | i3-1315U CER | i3 time | i3 pages/s | Ryzen 2600X CER | Ryzen time | Ryzen pages/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Kraken Lite Quality | 2.5642% | 56.089 s | 2.7278 | 2.7051% | 49.008 s | 3.1219 |
| Kraken Lite Balanced | 2.7482% | 46.887 s | 3.2631 | 2.9272% | 42.864 s | 3.5695 |
| Kraken Lite Turbo | 3.1294% | 44.150 s | 3.4654 | 3.4391% | 40.442 s | 3.7832 |
| Kraken Lite Extreme | 3.7733% | 40.810 s | 3.7491 | 4.2465% | 38.905 s | 3.9326 |
| Native Tesseract 5.4 | 3.8285% | 55.450 s | 2.7592 | 3.8285% | 35.883 s | 4.2638 |

Kraken's output is scored independently on each CPU because its automatic
session schedule and CPU execution differ: four recognition sessions with two
threads each on the laptop and six by two on the desktop. Tesseract produced
the same CER on both machines.

## Protocol

- Corpus: 153 fixed legal-print pages: 123 manual-gold pages and 30 manually
  vetted silver pages.
- Corpus fingerprint:
  `1ad486e8692fdd67693c92ee4f8d58f7e6e33c81bf79619e0d8a1135daf028d2`.
- Input: identical already-rendered PNG pixels on both machines.
- Metric: character-weighted CER after
  `nfkc-collapse-not-soft-hyphen-v1` normalization.
- Timing: cold process wall from launch through written OCR output. No warm-up
  page is removed.
- Kraken: production Rust provider and automatic CPU schedule; model/session
  initialization, image decoding, OCR, and output are timed.
- Tesseract: native Tesseract 5.4.0 C API, one persistent session per logical
  CPU, one OpenMP thread per session. Session initialization, image decoding,
  OCR, and output are timed.
- Priority: both engines ran BelowNormal. The same binary, model, runtime,
  Tesseract distribution, and corpus archive ran on both machines.

Hardware:

- Laptop: HP Laptop 14-ee0xxx, Intel Core i3-1315U, 8 logical CPUs.
- Desktop: AMD Ryzen 5 2600X, 12 logical CPUs.

The benchmark runner is
[`../benchmark_cpu_ocr.py`](../benchmark_cpu_ocr.py). Raw OCR transcripts are
ignored and disposable; the table above is the durable result.

## Frozen inputs

| Input | SHA-256 |
| --- | --- |
| Kraken benchmark runner | `c47650f353f1e7161a9318063dd0f4605873841bc922d953778fb2db1e49a951` |
| Kraken legal model | `c57efb79d08bc3f56568c0b3fbb076eccfc7f0f823c57d688cd7047827e45573` |
| Kraken codec | `5eb42e63e250d812b454f1c655af1dade5c2f1fdc267f4ccfc13ea62f360dd90` |
| ONNX Runtime | `b7dfcb4dea88f8488812c99e2c9016b9a30a374c83b888d39664df3238bcb48b` |
| Line-layout library | `21ef7215c66668473ed8264d76f30944c215108f08d7b6c2aa5a00a8d94db34e` |
| Native Tesseract executable | `babb405f4366b480d02cd8ff2bac8d497170f6c1711ce6f3d5d8bf0fb7fa6ed9` |
| Tesseract English model | `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2` |

Final scored result receipts:

- Laptop: `f074eec86492b6eef49d30e0dbd48815813871011ac75a04dd2c3049a60118cb`
- Desktop: `b499db144f66a1c4da757b50a462879d53aac326d3e915b5b80e471c52e6eab3`
