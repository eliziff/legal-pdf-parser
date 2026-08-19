# PPdoc Lite results

This file contains durable findings only. Raw models, corpora, logs, Python
environments, and generated benchmark output remain outside Git.

## Laptop runtime

### PP-YOLOE bbox architecture gate

The standard PP-YOLOE+ S/M bbox family passed the runtime gate before any
legal-data training was spent on it. PaddleDetection's documented
`exclude_nms=True` export produces static 640-by-640 image inputs and raw
`boxes[1,8400,4]` plus `scores[1,C,8400]` outputs. The Rust provider now applies
the released recipe's exact class-wise NMS contract: 0.01 score threshold, 0.7
IoU threshold, top 1,000 candidates per class, and at most 300 global results.
This removes Paddle, Python, and graph-level NMS from the shipped path without
changing the detector family or ontology.

Normal-priority paired runs used the same twelve laptop images, Rust
`ppdoc-images` command, OpenVINO CPU provider, eight threads, preprocessing,
decode, and native NMS:

| Architecture proof | FP16 pack bytes | Median seconds/page | Pages/second | Relative to incumbent |
| --- | ---: | ---: | ---: | ---: |
| legal PP-DocLayoutV3-L q75, two decoder layers | 113,275,737 | 0.559400 | 1.79 | control |
| stock PP-YOLOE+-M 640 | 47,332,536 | 0.277588 | 3.60 | 2.02x throughput |
| stock PP-YOLOE+-S 640 | 16,256,050 | 0.134486 | 7.44 | 4.16x throughput |

The stock COCO checkpoints are architecture-only speed controls and are not
legal-layout quality candidates. They prove that legal25 fine-tuning can add a
material balanced and fast tier if quality survives. A graph with embedded
ONNX NMS was also tried and rejected: OpenVINO could not compile its
dynamic-rank `Squeeze`. The official raw-output export plus small native Rust
NMS is both the documented and the faster portable route. These figures do not
yet include PDF parsing and artifact writes; trained candidates must also pass
the real end-to-end product benchmark before promotion.

The reproducible legal25 recipe is a bbox-only override on PaddleDetection's
released 80-epoch PP-YOLOE+ configuration. It keeps the frozen 25 labels and
existing 499/75/87 split, COCO pretrained S/M checkpoints, official
augmentation, multiscale 320--768 training, EMA, AMP, five-epoch linear
warmup, cosine decay, and the released static-to-task-aligned assigner switch
at epoch 30. Learning rate is scaled linearly from the documented 64-image
global-batch recipe. No mask or learned reading-order target enters this model.

Training from the F: hard drive made image loading the hot path. The immutable
dataset was checksum-verified after copying to the desktop's C:-backed WSL ext4
filesystem. The desktop WSL VM is capped at 8 GB system RAM: batch sizes 8--12
with multiple loader workers exhausted that limit even though the 3080 Ti had
VRAM available. Batch 4, two workers, and `prefetch_factor: 1` is the stable M
operating point. PaddleDetection's loader wrapper did not expose Paddle's
documented prefetch parameter, so the minimal pinned-source patch is preserved
in `training/paddledetection_dataloader_prefetch_factor.patch`.

Measured on the i3-1315U laptop at the teacher's native 800-by-800 input:

| Path | Warm seconds/page | Finding |
| --- | ---: | --- |
| initial Paddle/Python baseline | about 2.78 | dependency-heavy control |
| Rust preprocessing plus ONNX Runtime | about 1.678 | large improvement from a persistent native process |
| direct Rust/OpenVINO CPU, 8 threads | 1.010955 | fastest exact-FP32 CPU path tested |
| direct Rust/OpenVINO FP32, 75-page validation | 1.186 median; 1.160 mean after startup contention | complete scored end-to-end run |
| PP-DocLayoutV3-L 640, optimized ONNX/OpenVINO, 8 threads | 0.859-0.896 across three 12-page medians | 1.06-1.13 pages/second; quality/speed candidate |
| direct Rust/OpenVINO FP16, 75-page validation | 1.238 median | smaller storage artifact, not a CPU speed tier |
| direct OpenVINO Intel iGPU | 0.935–0.981 | small device-dependent win; CPU remains the portable floor |

The production build now has two features. `ppdoc-openvino` resolves no `ort`
crate and dynamically loads only the OpenVINO C ABI; its stripped Windows
binary was 12,851,200 bytes. `ppdoc` preserves ONNX Runtime and CUDA/TensorRT/
DirectML/oneDNN compatibility; its stripped binary was 13,033,984 bytes. Both
exclude Python, Paddle/PaddleX, Docker, training, and calibration dependencies.
The model pack is also native OpenVINO IR: a hash-checked `model.xml` plus
`model.bin`, converted once with the documented `ovc` path. It does not require
ONNX Runtime or model conversion on the destination laptop.

A clean-directory CPU smoke passed with only `openvino_c.dll`, `openvino.dll`,
`openvino_intel_cpu_plugin.dll`, `openvino_ir_frontend.dll`, and `tbb12.dll`.
Those five runtime files total 61,721,048 bytes; no Python environment or other
OpenVINO frontend/plugin was present. Adding only the 36,969,464-byte Intel GPU
plugin enabled the same binary and model pack on the laptop iGPU.

The optional OpenVINO compiled-model cache reduced iGPU cold-process wall time
from about 22 seconds to 2.02 seconds on the next launch; warm-cache inference
was 0.746 seconds/page. The generated cache was 252 MB and is specific to the
device and OpenVINO version. CPU caching did not reduce startup and is not a
recommended default.

OpenVINO's official `benchmark_app` rejected asynchronous request pooling as a
material laptop tier for this graph. FP32 CPU throughput was 0.82 pages/second
with one synchronous request and 0.80 with the automatically selected four
asynchronous requests. Intel GPU execution, using the required FP32 precision,
rose from 1.12 to 1.31 pages/second with four requests. Cumulative CPU+iGPU
execution reached only 1.06 pages/second because both devices share the same
memory and power budget. Native GPU FP16 compilation failed inside OpenVINO's
dynamic-rank graph rewrite. These results do not justify a Rust async pool.

The document-safe PP-DocLayoutV3-L 640 checkpoint selected at epoch 15 scored
0.663100 AP, 0.738119 AP50, and 0.723130 AP75 after OpenVINO conversion on all
75 validation pages. Three Normal-priority laptop runs over a fixed 12-page
sample measured 0.8963, 0.8757, and 0.8591 median seconds/page. Mean throughput
was 1.06-1.13 pages/second; the same OpenVINO 2026.3 runtime, Rust binary,
eight threads, images, and threshold measured the optimized 800 teacher at
1.6711 median seconds/page. The prior complete teacher validation median was
1.186 seconds/page, so the cross-corpus directional comparison is a 26%
latency reduction while the controlled sample is a 47% reduction. The 640
model remains roughly the teacher's size and loses substantial detector AP;
it is a measured Pareto candidate, not yet a promoted default.

Export route is a first-order performance variable. Direct conversion of the
legacy Paddle graph took 5.36 seconds for the fixed comparison page at 640 and
6.26 seconds at 800. Passing the same 640 Paddle export through pinned
Paddle2ONNX 1.0.5 at opset 16, retaining only decoded boxes/counts as graph
results, and then running `ovc` reduced that page to 1.17-1.25 seconds. The
direct and optimized 640 graphs emitted 144 detections with zero label, order,
score, or coordinate difference. `build_paddle_openvino_pack.py` now performs
that complete route, verifies the OpenVINO inputs/outputs and 640 input shape,
hash-locks the pack, and records tool versions. Future L/M/S or distilled
checkpoints must use this command rather than direct Paddle-to-OpenVINO
conversion.

The reproducible 640 pack was then staged with the current Rust executable and
five CPU runtime DLLs. Its recursive clean-bundle manifest contains 11 files
totalling 212,973,443 bytes; Python, Paddle, ONNX Runtime, Docker, exporter,
training, and calibration dependencies are absent. Three complete 9-page SEC
complaint parses took 10.779, 9.019, and 9.413 seconds, for a median 1.046
seconds/page including process startup, PDF parsing, 72-DPI rendering,
inference, role attachment, and artifact writes. This is 37% faster than the
previous clean teacher bundle's 1.657 seconds/page. All runs covered the same
275 lines, emitted no `PPDOC_LAYOUT_INCOMPLETE` diagnostic, and produced
byte-identical nine-file output trees. The existing degraded status came only
from 28 `FOOTNOTE_UNMATCHED_LABEL` diagnostics and was not a layout-provider
failure.

Decoder/query ablations used that same epoch-15 640 checkpoint and frozen
legal-25 ontology; they did not retrain or alter labels. The supported
`eval_idx` export selected an earlier decoder result, while the optional export
flag omitted the unused learned reading-order output. Complete 75-page
article-disjoint OpenVINO results were:

| Variant | IR bytes | AP | AP50 | AP75 | AR100 | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| q75, 6 decoder layers | 130,082,200 | 0.665011 | 0.743607 | 0.728120 | 0.829221 | quality control |
| q75, 3 layers, order-free | 117,366,547 | 0.664292 | 0.734918 | 0.720951 | not recorded | dominated by two layers |
| q75, 2 layers, order-free | 113,275,737 | 0.658115 | 0.731048 | 0.719274 | 0.835013 | balanced candidate |
| q50, 2 layers, order-free | 113,275,737 | 0.655867 | 0.729126 | 0.718989 | 0.797942 | rejected: no speed gain and lower recall |

The q50 and q75 two-layer graphs measured 0.4359 and 0.4365 median seconds per
page respectively on the same desktop OpenVINO benchmark, so reducing to 50
queries bought no throughput. On the laptop, four counter-ordered 12-page Rust
comparisons put the q75 two-layer candidate about 11% below the full decoder's
median latency despite substantial machine-state noise. Two counter-ordered
real-product parses of the 9-page SEC complaint had median latencies of 0.6323
seconds/page for two layers and 0.7102 for full depth, also an 11% reduction.
Every output file was byte-identical between tiers except `document.json`, whose
model identity and derived cache key correctly differed. Repeated outputs
within each tier were byte-identical. The two-layer export therefore advances
as the balanced 640 tier; neither q50 nor three layers advances.

`export_ppdocv3_inference_ablation.sh` and
`build_and_benchmark_openvino_ablation.sh` preserve the exact successful export,
conversion, pack verification, and 75-page scoring route for future checkpoints.

The Rust provider is also wired into the real PDF parse path: Hayro renders a
page in memory, PPdoc labels the existing extracted lines, and the structural
engine consumes those source roles without replacing PDF text. Assignment is
fail-closed across the document; if one nonempty line is not covered, all model
roles are discarded and a diagnostic is emitted. A release-mode product smoke
on a 9-page, 275-line SEC complaint completed with full line coverage in 14.309
seconds (1.590 seconds/page) on CPU. The native parse without layout took 0.117
seconds total, so inference remains the product hot path.

The Text-Fidelity PPDoc region postprocessor is now native production Rust,
pinned to clean Text-Fidelity-Project commit
`d8b25257687b3b9aad644dec42cca966b45675ff`. The provider first collects the
whole document's decoded detections, then applies the source production
defaults before assigning lines: the exact smallest-containing-region/10%
overlap rule, conservative inset block quotes, hard label validity, byline
windowing, repeated headers and footers, edge and sequenced page numbers,
Roman headings, footnote sandwich/top-band repair, full-width quote demotion,
overlap priority, and source reading order. Disabled source experiments were
not promoted. Seven focused Rust parity fixtures pass. A current release smoke
with the q75 two-decoder-layer large pack processed the same 9-page, 275-line
SEC complaint in 7.789 seconds (0.865 seconds/page), covered every line, and
emitted no `PPDOC_LAYOUT_INCOMPLETE`. Compared with the pre-port q75 run, the
pages, paragraphs, sections, footnotes, diagnostics, and repairs files were
all byte-identical, showing that the port did not perturb durable output on
this document. The Rust source participates in both engine and extraction
cache identities. The optimized binary's phase timer measured the
postprocessor itself at 3.120 milliseconds for all nine pages, or 0.347
milliseconds/page (about 2,885 pages/second); it consumed about 0.04% of the
8.342-second product run and is not an inference hot path.

The actual pinned Python-versus-Rust differential gate is also complete. Both
implementations independently processed the same raw region order and emitted
1,410,350 identical canonical UTF-8 bytes across 206 documents, 726 pages,
9,179 regions, and 14,275 line assignments. The shared output SHA-256 is
`453f4050098f83906499b0ec5b5a2e0692677dfb089c27812dbf128f69c846dc`.
The compared contract includes every final label, score, rectangle, reading
order, raw region identity, and line-to-region selection. The reproducible
gate and full receipt are in
[`postprocess-parity/RESULTS.md`](postprocess-parity/RESULTS.md).

Rendering at 72 DPI instead of 96 DPI reduced that end-to-end run from 16.851
to 14.309 seconds (1.872 to 1.590 seconds/page). The produced pages,
paragraphs, sections, footnotes, and diagnostics were byte-identical, so 72 DPI
is the default while the CLI retains a DPI calibration option. This is a
directional integration/throughput result, not a replacement for the sealed
held-out quality evaluation.

The production bundle script then staged the combined Rust PDF/OCR/layout
binary, teacher model pack, and five CPU runtime DLLs into a clean directory.
Its recursive manifest contained 10 files totalling 210,787,875 bytes; no
Python, Docker, ONNX Runtime, exporter, calibration data, or training package
was present. That copied bundle parsed the same 9-page PDF successfully at
1.657 seconds/page with no `PPDOC_LAYOUT_INCOMPLETE` diagnostic.

The selected 640-by-640 D-FINE-S student proves the laptop runtime path, but is
not a promotable quality tier. Its final standalone CPU bundle contains the
combined Rust binary, FP32 model pack, and the same five OpenVINO DLLs: 10 files
and 122,194,453 bytes, with no Python, Docker, ONNX Runtime, training code, or
calibration data. Three clean bundled parses of the 9-page complaint had a
median wall time of 3.246 seconds, or 0.361 seconds/page (2.77 pages/second),
including startup, PDF parsing, rendering, inference, role attachment, and
artifact writing. All 275 lines were covered, no `PPDOC_LAYOUT_INCOMPLETE`
diagnostic was emitted, and repeated outputs were byte-identical. This is 4.4x
the teacher product baseline throughput, although native parsing alone remains
much faster. Epoch 50 changed only 5 of the complaint's 275 line roles relative
to the earlier epoch-45 student probe. The final FP16 pack also passed the Rust
product path with full coverage, but took 0.568 seconds/page and changed 2 of
275 roles relative to FP32, confirming that it is a storage tier rather than a
CPU speed tier. The sealed held-out result below rejects this artifact for
product selection despite that runtime result.

## Precision and quantization ablations

The complete 75-page article-disjoint validation comparison at the production
0.10 score threshold is:

| Artifact | IR bytes | COCO AP | AP50 | CPU median seconds/page | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| teacher OpenVINO FP32 | 129,805,514 | 0.820605 | 0.892021 | 1.186 | quality incumbent |
| teacher OpenVINO FP16 constants | 65,806,635 | 0.820402 | 0.892021 | 1.238 | storage tier; effectively identical quality but no CPU speed win |

The earlier 40-page development surface was used for the directional PTQ
comparison:

| Artifact | Bytes | COCO AP | AP50 | CPU seconds/page | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| teacher FP32 | 129.4 MB | 0.755790 | 0.839790 | 1.011 | incumbent |
| selective backbone INT8/FP32 | 92.4 MB | 0.743068 | 0.828266 | 1.092 | footprint-only candidate; slower and less accurate |

Broader static INT8 variants lost more quality and were slower on this laptop.
Dynamic quantization, alternate ORT thread/provider settings, and the attempted
NNCF accuracy-controlled path did not earn a throughput tier. Quantization is
therefore not assumed to be faster: only a QAT/QDQ artifact that passes both
quality and laptop timing can be promoted.

The final student used its validation-selected 0.01 score threshold. On all 75
article-disjoint validation pages, using four OpenVINO CPU threads, its precision
ladder was:

| Student artifact | IR bytes | COCO AP | AP50 | AP75 | Median seconds/page | Pages/second | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| OpenVINO FP32 | 42,149,344 | 0.536008 | 0.619356 | 0.589402 | 0.2367 | 4.16 | runtime candidate; rejected by held-out quality |
| OpenVINO FP16 constants | 21,693,494 | 0.537793 | 0.621640 | 0.590829 | 0.2579 | 3.59 | compact runtime candidate; rejected with the source model |
| INT8 mixed preset | 12,614,685 | 0.348725 | 0.445815 | 0.385230 | 0.2060 | 4.64 | rejected: 35% AP loss for 11% throughput gain |
| INT8 performance preset | 12,614,875 | 0.291432 | 0.406358 | 0.308649 | 0.2033 | 4.81 | rejected: 46% AP loss for 16% throughput gain |

NNCF accuracy-controlled PTQ was also rejected rather than left as an assumed
win. It measured the FP32 reference at 0.536008 AP and full INT8 at 0.351861 AP,
then ranked 265 quantized operations. Restoring its top-ranked operation made
AP worse (0.311662); subsequent candidates reached only 0.345972 and 0.345084.
The raw detector outputs repeatedly produced non-finite ranking warnings, and a
bounded run was still re-ranking after 30 minutes. The full INT8 packs above are
kept only as negative-control evidence, not shipped product tiers.

The validation-selected epoch-50 FP32 student was then evaluated exactly once
on the sealed 87-page journal-held-out split. The annotation receipt was
`e7242d2418c8a6e3078385aeda2707cbe3b8a9c0745184ff203decdff5a520a5`; the
26,100-prediction receipt was
`e0535b42c500118c706ede61f734a7ff28a5871608334a11e7dc48a7fb802c0c`.
Paddle/COCO measured AP 0.332860, AP50 0.400926, AP75 0.362189, and AR100
0.632800. This large validation-to-test drop rejects D-FINE-S as a shipped fast
tier. No held-out result was used to tune that model.

Changing the fixed 800-by-800 exported graph to a lower input declaration is
not valid. Its learned graph contains 800-specific positional and feature-map
constants (including a 25-by-25 positional grid), and both ONNX Runtime's shape
tool and OpenVINO conversion reject a 640 override. A lower-resolution tier
requires a genuine model re-export or training run, not graph metadata surgery.

## OCR-dependent product A/B

The 2026-08-14 end-to-end port check used the genuine 11-page full-page raster
scan `SCC-1970-SCR-638/source.pdf`. A native-text-only control completed in
0.083 seconds with `ocr_required`, proving that the timed path actually used
OCR. The corrected comparison is the old Tesseract recognition path versus the
new native Kraken-lite path, not two PPDoc binaries. Both used the current Rust
product, Normal priority, and, for the complete run, the same
q75/two-decoder-layer/640-pixel PPDoc pack at threshold 0.10, 72-DPI PPDoc
rendering, and OpenVINO CPU. No Docker or network service was involved.

| Path | OCR-only wall | OCR-only pages/s | OCR + PPDoc wall | End-to-end pages/s |
| --- | ---: | ---: | ---: | ---: |
| Tesseract 5.4 CLI (old OCR) | 30.889 s | 0.356 | 38.269 s | 0.287 |
| native Kraken-lite quality (new OCR) | 9.765 s | 1.126 | 19.765 s | 0.557 |

On this dense bilingual scan, Kraken was 3.16x faster for OCR and 1.94x faster
for the complete OCR-plus-layout product. Kraken emitted 960 lines and seven
footnotes; Tesseract emitted 956 lines and two footnotes. This document has no
trustworthy exact transcription, so those counts are not presented as an
accuracy score. The separate 153-page labelled OCR benchmark remains the
quality evidence: native Kraken quality recorded 2.820% CER versus Tesseract's
4.115%, alongside 1.235 versus 0.763 pages/second on that image surface.

The earlier roughly 4.15 pages/second Kraken figure came from a different warm
30-page image mix, not a cold PDF-to-product run. The same-document PDF test in
the Kraken runtime work had previously measured 13.6 seconds for this 11-page
scan; the current 9.765-second OCR-only result is faster, not a lost runtime
optimization.

PPDoc did not cover every OCR line on either path, so both complete runs emitted
`PPDOC_LAYOUT_INCOMPLETE` and discarded PPDoc roles under the existing safety
gate. The Rust postprocessor itself is not the bottleneck: an instrumented
Kraken run measured it at 0.002469 seconds total for all 11 pages. The separate
Python-oracle differential remains its semantic proof (206 documents, 726
pages, 9,179 regions, 14,275 line assignments, and 1,410,350 byte-identical
output bytes).

Raw outputs are disposable and remain under ignored
`.tmp/ocr-tesseract-vs-kraken-20260814/`.

## Model-training findings

- A compatible PP-DocLayoutV3 student construction now passes before training.
  It retains the V3 transformer, legal-25 classifier, masks, and reading-order
  head while applying PaddleDetection's released Mask RT-DETR HGNetV2-M/S
  backbone and neck size settings. M has 19,191,573 trainable parameters
  (76.8 MB estimated FP32 parameters), 42% fewer than the roughly 33.3M L
  teacher. S has 15,035,637 (60.1 MB), 55% fewer. The composite M checkpoint
  has complete initialization coverage: 9,752,989 elements come from the legal
  teacher and 9,438,584 architecture-specific elements from Paddle's official
  Mask RT-DETR-M checkpoint. S initializes 99.993% of parameters from those two
  sources; its only new tensor is the explicitly recorded 1,024-weight final
  mask adapter caused by the 64-to-32 internal mask-channel change.
- All M/S 300- and 100-query graphs loaded the same prepared checkpoint and
  produced finite 640-pixel CPU outputs. Controlled cold first-forward times on
  the desktop CPU were 4.95 seconds for L/300, 3.73 for L/100, 3.59 for M/300,
  2.62 for M/100, 3.21 for S/300, and 2.29 for S/100. These are architecture
  preflight timings, not laptop/OpenVINO throughput claims. They show that the
  released smaller architecture and supported query reduction stack; M/100 was
  1.89x and S/100 2.16x faster than the L/300 control on this surface.
- PaddleDetection's shipped feature-distillation wrappers do not support DETR.
  A minimal experiment patch now applies its existing normalized CWD loss only
  to V3's shape-aligned three 256-channel neck levels and 32-channel mask
  feature, while the ordinary legal detection, mask, and order losses remain
  authoritative. A real one-page, 320-pixel CPU train/eval smoke completed:
  supervised total before distillation was about 167.01, encoder CWD was 2.946,
  mask CWD was 1.041, and the combined loss was finite at 170.994. The teacher
  remained frozen; no invalid query-index loss was introduced.

- The corrected D-FINE-S 640 run used the unchanged legal25 ontology, official
  COCO pretraining, AMP, batch size 8, and the 499-page article-disjoint train
  split. Validation AP rose from 0.162 at epoch 5 to 0.307 at 10, 0.425 at 20,
  0.482 at 30, 0.520 at 40, and a best 0.540 at epoch 50 (AP50 0.618, AP75
  0.595). Later evaluations plateaued, so epoch 50 is the selected checkpoint.
  The exported OpenVINO FP32 graph reproduced AP50 and was within 0.004 AP of
  Paddle on the same validation split.
- The run's patience state correctly reached five stale validation checks, but
  the wrapper initially waited for a checkpoint message that is absent after a
  non-improving evaluation. The harness now stops at the first following epoch
  log line, which proves evaluation and checkpoint callbacks completed while
  preserving the previous completed checkpoint.
- The PicoDet-M run reached about 0.273 COCO AP versus about 0.254 before legal
  fine-tuning. This is not a viable tier and the run is stopped.
- Paddle's own family comparison reports substantially lower mAP@0.5 for the
  S/M PicoDet variants than for PP-DocLayoutV3. A future small-model attempt
  must stay compatible with the RT-DETR task/head family and use the incumbent
  as teacher; another PicoDet knob search is not planned.
- The current compression experiment uses PaddleSlim 2.6's official RT-DETR
  200-step INT8 QAT plus soft-label self-distillation recipe. The legacy
  `.pdmodel/.pdiparams` graph is required; PaddleX 3 PIR `.json` is not accepted
  by PaddleSlim's legacy graph wrapper.
- The verbatim official 200-step QAT recipe was rejected: its loss was
  `313.4459` at step 0 and non-finite by the step-10 log point. PaddleSlim still
  returned success and exported a model, so the local runner now treats any
  NaN/Inf loss as failure. A one-variable retry using PaddleSlim's native
  `ClipGradByGlobalNorm(1.0)` optimizer hook also became NaN at step 10 and was
  rejected. QAT is parked; the incumbent is unchanged.

## Evaluation boundary

The original split is retained only for regression. Model selection uses the
499-page train and 75-page article-disjoint validation split. The 87-page
journal-held-out test was opened once for the validation-selected D-FINE-S
epoch-50 candidate; its failed result is recorded above and is not used for
further tuning. Future candidates remain selected on validation without
repeated test-set probing.

Primary recipe sources:

- <https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection>
- <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml>
- <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md>
