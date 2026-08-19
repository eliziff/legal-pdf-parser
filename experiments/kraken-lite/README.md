# Kraken Lite

Lightweight local OCR experiments for the fixed legal-print Kraken recognizer. The browser build is a self-contained HTML application; the native build keeps full stock BLLA segmentation available.

## Browser

Build and test:

```powershell
cd kraken-lite-browser
npm test
npm run single
```

Open `kraken-lite-browser/dist/kraken-lite.html` directly. It accepts PNG and PDF files and contains the model, PDF renderer, Tesseract layout core, recognition runtime, and workers. `kraken-lite-lean.html` omits mature layout and uses projection segmentation.

The shipping full build reserves one browser thread and uses the remaining browser-reported parallelism for persistent single-thread recognition workers (seven on the eight-thread benchmark laptop), plus two layout workers, 32-line batches, 24-pixel width buckets, a 48-pixel-height recognizer, relaxed-SIMD ONNX Runtime, and per-channel INT8 LSTM weights. All four modes use the same model and layout; only recognition width changes.

| Browser mode | Width | CER | pages/s |
| --- | ---: | ---: | ---: |
| Quality | 1.00 | 2.582% | 1.548 |
| Balanced | 0.85 | 2.763% | 1.672 |
| Turbo | 0.76 | 3.163% | 1.788 |
| Extreme | 0.70 | 3.837% | 1.824 |
| Tesseract.js fast | — | 4.108% | 0.364 |

Results use the same 153-page benchmark and already-rendered pixels for CER and timing. Quality is the median of three complete runs; the other tiers are full-corpus single-run confirmations. On the 30 diversified scanned-court pages, Extreme reaches 2.438 pages/s at 3.819% CER versus Tesseract.js at 0.413 pages/s and 7.155% CER.

## Native

```powershell
cd kraken-lite-native
python ocr.py --tier quality page.png
```

The default quality tier reuses the browser's proven dynamic-batch INT8 model, two layout workers, two page-level recognition workers, 32-line batches, and 24-pixel buckets. It reaches 2.820% CER and 1.235 pages/s on the same 153 pages, versus the established in-process Tesseract result of 4.115% CER and 0.763 pages/s. Use `--tier fidelity` for the original full-size recognizer plus stock BLLA geometry.

## Benchmark set

`kraken-lite-native/benchmark-splits/benchmark-153.lst` contains 123 manual-gold pages and 30 manually vetted silver pages. Both accuracy and throughput are measured on this one set; normalization removes both soft hyphen and the model's mathematical-NOT substitute.

Rejected defaults include WebGPU LSTM partitioning, static Conv quantization, binary layout thresholding, layout downscaling, whitespace scans, Sauvola/Otsu/morphology preprocessing, a third layout worker, and the 0.62 width tier. Each either regressed CER, throughput, or both on the mixed benchmark.
