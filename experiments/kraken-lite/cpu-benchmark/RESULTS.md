# Legal OCR benchmark results

Measured 2026-08-19 on the same 153 rendered legal-print pages and 438,577
ground-truth characters. The corpus fingerprint is
`1ad486e8692fdd67693c92ee4f8d58f7e6e33c81bf79619e0d8a1135daf028d2`.

## End-to-end result

| OCR profile | Ryzen 5 2600X CPU | RTX 3080 Ti CUDA | CER |
| --- | ---: | ---: | ---: |
| Legal OCR — Quality | 2.951 pages/s | 6.263 pages/s | 2.571% |
| Legal OCR — Balanced | 3.390 pages/s | 6.396 pages/s | 2.745% |
| Legal OCR — Turbo | 3.939 pages/s | 6.591 pages/s | 3.135% |
| Legal OCR — Extreme | 4.095 pages/s | 6.874 pages/s | 3.759% |
| Native Tesseract 5.4 | 3.899 pages/s | not supported | 3.829% |

Timing is cold process wall from launch through flushed OCR output. It includes
runtime/session initialization, page decoding, line finding, recognition, and
output. Every figure is the median of three complete runs. Tesseract produced
identical text in all three trials. CPU and CUDA legal-OCR output differed by at
most nine characters in 438,577 for the two narrowest profiles; the table shows
the CUDA CER.

Tesseract uses twelve persistent C-API sessions on the desktop and eight on the
desktop, with `OMP_THREAD_LIMIT=1`. This follows Tesseract's production guidance
to process multiple pages with independent single-threaded instances. Its
OpenCL path is experimental and not a supported GPU comparison.

## Runtime efficiency

Warmed CUDA recognition isolates the model from page layout and I/O:

| Recognition model | Pure recognition | Runtime recognition | CER |
| --- | ---: | ---: | ---: |
| Legal fine-tune, CATMuS Print Small | 932 lines/s | 879 lines/s | 2.575% |
| Stock CATMuS Print Small | 1,066 lines/s | 994 lines/s | 3.901% |
| Stock CATMuS Print Large | 521 lines/s | 418 lines/s | not scored |

The legal fine-tune trades some raw recognition throughput for substantially
lower legal-print error. The full production benchmark is slower than pure
recognition because it also decodes pages, finds lines, prepares variable-width
line tensors, and writes results. On the RTX 3080 Ti, overlapping those CPU
stages raised Balanced end-to-end throughput to 6.40 pages/s while warmed
recognition sustained about 900–1,300 lines/s depending on batch packing.

## Protocol

- Inputs: identical 200-dpi PNG pages for every engine and machine.
- Accuracy: character-weighted CER after
  `nfkc-collapse-not-soft-hyphen-v1` normalization.
- Scheduling: benchmark processes run at BelowNormal priority.
- Legal OCR CUDA: 64-line batches, 128-pixel width buckets, twelve concurrent
  CPU layout workers, one CUDA recognition stream.
- Native Tesseract: 5.4.0 with integer `tessdata_fast` English data, one OpenMP
  thread per persistent worker.
- Hardware: Ryzen 5 2600X/RTX 3080 Ti 12 GB desktop. A separate Core i3-1315U
  laptop run reached 2.180 pages/s Quality and 2.503 pages/s Turbo.

The benchmark runner is
[`../benchmark_cpu_ocr.py`](../benchmark_cpu_ocr.py). Raw transcripts, models,
runtimes, and generated receipts are ignored.
