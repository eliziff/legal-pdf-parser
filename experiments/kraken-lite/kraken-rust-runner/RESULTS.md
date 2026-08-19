# Native Kraken promotion results

The production Kraken modules are compiled directly by this small runner, so
normal OCR edits use the same code as the application without rebuilding the
unrelated PDF, DOCX, citation, and repair graph:

```powershell
cargo kraken-check
cargo kraken-build
```

On the 2026-08-13 Core i3-1315U evaluation, cached edits built in 5-11 seconds.
The production release binary also built successfully. Raw receipts remain in
ignored `_temp/`; the fixed list files are the durable corpus definitions.

## Promotion result

The comparison uses the full-quality recognizer and identical page images. The
native process ran at Normal priority for the 12- and 30-page timing gates and
BelowNormal priority for the corpus promotion run.

| Gate | Browser pages/s | Native pages/s | Native gain |
| --- | ---: | ---: | ---: |
| 12 pages, median of 3 | 1.9605 | 2.4880 | 26.9% |
| 30 pages | 1.7825 | 2.4105 | 35.2% |
| 153 pages | 1.5477 | 1.8051 | 16.6% |

The 12-page native and browser outputs have identical CER (2.086%). The
153-page browser receipt and the current native receipt use different revisions
of the XML truth, so their printed CER values are not directly comparable.
Rescoring both outputs against the browser receipt's embedded frozen truth gives
2.5818% browser CER and 2.5642% native CER: native makes 77 fewer character
edits. It is better on 51 pages, worse on 33, and tied on 17 of the 101 pages
whose outputs differ.

Against the prior native engine on the current 153-page truth, the promoted
path improves speed from 1.5115 to 1.8051 pages/s and improves every product
quality metric below:

| Metric | Prior native | Promoted native |
| --- | ---: | ---: |
| CER | 2.7065% | 2.6400% |
| WER | 6.8689% | 6.8276% |
| Layout recall | 96.7690% | 96.8228% |
| Mean line IoU | 0.78292 | 0.78309 |
| Reading-order agreement | 99.2235% | 99.2243% |
| Matched-line CER | 1.5909% | 1.5772% |

Raw layout precision changes from 99.1175% to 99.0634%, but layout F1 improves
from 0.9792916 to 0.9793028. Only five pages change layout boxes; two gain
matched gold lines, none lose one, and the extra unmatched boxes on one page
are blank and filtered before product output. That counter is therefore not a
product regression.

## Runtime shape

- The one-call layout core built from browser-matching Tesseract 5.3 and
  Leptonica 1.83 is 3.57 MB, down from the installed 101.47 MB Tesseract DLL.
  Tesseract 5.4 produced three additional character errors on the 12-page gate
  and was rejected.
- The release runner is 1.61 MB, ONNX Runtime 1.26 is 16.78 MB, the full-quality
  model is 0.62 MB, and the codec is 5 KB. The measured native OCR stack is
  about 22.6 MB before ordinary OS libraries.
- ONNX Runtime 1.26 improves sustained 30-page throughput by 6.2% over the
  previous 1.22 runtime with identical output.
- `--kraken-low-memory` disables the CPU arena. It lowers peak working set from
  about 1,428 MB to 806 MB (43.5%) for roughly a 6% throughput cost, with
  identical OCR output.
- A direct four-channel RGBA layout input is output-equivalent but about 10%
  slower. Production instead shares one browser-equivalent one-byte RGB-average
  page buffer between layout and recognition. On an 11-page scanned legal PDF,
  this reduced release wall time from 15.7 to 13.6 seconds and produced
  byte-identical pages and footnotes.
- Keeping the layout API alive between pages, as the browser core does, produced
  identical text and layout and made the layout stage 1.3% faster in an adjacent
  30-page A/B. Both the compact wrapper and stock-library fallback now retain it.
- A warmed four-session ONNX trace over the 30-page quality gate attributes
  66.55% of operator time to `DynamicQuantizeLSTM`, 14.84% to transposes, 6.96%
  to convolution, 3.09% to max-pooling, and 2.96% to the final matrix multiply.
  ONNX host overhead outside operators is only 0.91% of `Run` time. This rules
  out allocator, serialization, and generic host-overhead work as useful next
  levers; the retained short gate is thread scheduling inside the hot kernels.
- ONNX Runtime's 1 ms bounded spin with exponential backoff preserved every
  text and layout hash but won only two of three adjacent 30-page pairs
  (+6.03%, +17.60%, -40.15%). Despite a +12.99% median, it was rejected for
  inconsistent sustained-load behavior and is not present in production.
- Extending the existing preparation producer upstream through PDF rendering
  preserves its established OCR windows while bounding raster retention. On a
  28-page scanned legal PDF, all shipped page, footnote, paragraph, section,
  repair, and diagnostic artifacts were byte-identical across every run. Three
  warmed adjacent pairs reduced median release wall time from 14.94 to 13.45
  seconds (10.0%) and median peak working set from 1,555 to 1,404 MiB (9.7%).
  The unchanged short-document path also reproduced all six artifacts exactly
  on the prior 11-page scan. An earlier outer eight-page chunk prototype was
  removed because it fragmented established recognition batches: it used less
  memory but was slower and changed page and paragraph bytes.
- The operator profile does not justify another accelerator experiment. Its
  bottleneck is the quantized recurrent kernel, while the already-captured
  OpenVINO screen required the unquantized model and reached only 59.77
  samples/s on the laptop GPU after a 15.7-second compile (the best screened
  CPU INT8 configuration reached 95.59 samples/s). No alternate execution
  provider had evidence strong enough to replace the exact current CPU path.
- A same-document stage profile explains the apparent 4.15 versus 2.08 page/s
  gap. On the dense 28-page scan (2,264 prepared lines), median full-product
  wall was 13.46 seconds and the OCR path was 12.77 seconds (94.9%). Additive
  work was 3.74 seconds rendering, 3.69 layout, 0.50 line preparation, and 8.97
  recognition; pipelining overlaps those stages. Provider construction was
  only 0.134 seconds and the complete no-OCR product path was 0.211 seconds.
  The matching 28 source PNGs measured 2.47 page/s after excluding a 1.31-second
  warmup, while their cold process wall was 12.90 seconds. The PDF-to-product
  job therefore cost only about 0.56 seconds (4.3%) beyond the comparable cold
  image job; 4.15 page/s came from a different, warmer 30-page image mix, not
  from avoiding a large PDF-engine tax.

The native schedule uses bounded page-window preparation overlapped with
recognition, preserves page order, and chooses balanced windows instead of a
machine-specific fixed page count. Promotion requires exact short-gate output
for scheduler/runtime changes and corpus-scale CER/WER/layout/order checks for
anything that can change numerical output.
