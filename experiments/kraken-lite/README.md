# Legal OCR experiments

This directory contains the reproducible OCR benchmarks behind Legal PDF
Parser's Quality, Balanced, Turbo, and Extreme profiles.

The recognition model is a legal-domain fine-tune of
[CATMuS Print Small](https://zenodo.org/records/10602357), trained with the
[Kraken](https://kraken.re/) OCR ecosystem. The production runtime is a compact
Rust/ONNX pipeline that overlaps page decoding, line finding, line preparation,
and CUDA recognition.

The fixed benchmark contains 153 legal-print pages: 123 manual-gold pages and
30 manually reviewed silver pages. Accuracy and speed use the same pages,
pixels, ground truth, and normalization. Generated transcripts and model files
belong in ignored `_temp/` directories; durable measurements belong in
[`cpu-benchmark/RESULTS.md`](cpu-benchmark/RESULTS.md).
