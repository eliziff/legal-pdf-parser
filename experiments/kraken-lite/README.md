# Kraken Lite

Kraken Lite is the parser's fast CPU OCR path. It combines a compact
Tesseract-derived line finder with a legal-specific fine-tune of CATMuS Print
Small. The same model supports four recognition-width tiers:

- Quality prioritizes the lowest character error rate.
- Balanced keeps most of Quality's accuracy with higher throughput.
- Turbo favors faster document turnaround.
- Extreme uses the narrowest recognition width.

The current fair CPU benchmark is recorded in
[`cpu-benchmark/RESULTS.md`](cpu-benchmark/RESULTS.md). It compares every tier
with native Tesseract 5.4 on the same 153 legal pages, on both a Core i3 laptop
and a Ryzen desktop. Both engines include cold session/worker setup in the
timed wall; no browser-worker startup is hidden.

## Runtime

The native pack contains the Rust parser, ONNX Runtime, the compact legal OCR
model and codec, and the line-layout library. CPU scheduling adapts to the
available logical processors. The low-memory option disables the ONNX CPU arena
to reduce peak memory at a small throughput cost without changing OCR output.

The browser experiment packages the same recognizer into a self-contained HTML
application. It remains useful for portability testing, but public performance
claims use the native CPU comparison above.

## Benchmark set

`kraken-lite-native/benchmark-splits/benchmark-153.lst` defines the fixed set:
123 manual-gold pages and 30 manually vetted silver pages. Accuracy and speed
use that same set. Normalization collapses whitespace and removes soft hyphen
and the model's mathematical-NOT line-break substitute before CER scoring.

Run the benchmark with `benchmark_cpu_ocr.py`; raw transcripts belong under an
ignored `_temp/` directory. Record durable findings in the benchmark
`RESULTS.md`, not in generated output files.
