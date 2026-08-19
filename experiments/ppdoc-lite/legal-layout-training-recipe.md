# Fast generalizable legal layout recipe

## 2026-08 PP-YOLOE decision update

This section supersedes the earlier assumption below that every useful student
must remain in the Mask RT-DETR/PP-DocLayoutV3 family. The prior PicoDet,
D-FINE, and hybrid PP-DocLayoutV3 results remain valid negative evidence; they
do not apply to the standard PP-YOLOE+ bbox family.

The target capability is fast inference for the frozen legal25 ontology using
the existing 661 pages. Masks and learned reading order are not required by the
engine's layout-provider contract, so the normal, documented answer is to
fine-tune a released bbox detector on the same COCO boxes and labels rather
than invent a hybrid Mask RT-DETR architecture. PP-YOLOE+ S and M accept that
data directly and have official 80-epoch pretrained recipes, AMP training,
export, pruning, and distillation support in PaddleDetection.

The model ladder under test is:

| Tier | Candidate | Role |
| --- | --- | --- |
| quality | incumbent legal PP-DocLayoutV3-L q75 | accuracy ceiling and heavier CPU/GPU option |
| balanced | legal25 PP-YOLOE+-M 640 | materially faster default candidate |
| fast | legal25 PP-YOLOE+-S 640 | laptop-throughput candidate |

This choice passed an architecture-only gate in the actual Rust/OpenVINO
laptop runtime before training: stock M measured 0.277588 seconds/page and
stock S 0.134486, versus 0.559400 for the incumbent q75 two-layer graph on the
same twelve images. Stock weights say nothing about legal quality; they prove
the family has enough inference headroom to justify the bounded training work.

Fine-tune both released checkpoints at 640 evaluation resolution with the
official PP-YOLOE+ recipe: 80 epochs, official augmentation, multiscale
320--768 batches, EMA, AMP, five warmup epochs, cosine decay, and static
assigner through epoch 30. Scale the released 0.001 learning rate linearly from
its global batch of 64. On the 8 GB WSL/3080 Ti desktop surface, the measured
stable M point is batch 4, two workers, and loader prefetch 1, giving
0.0000625 base LR; S should use the largest separately proven stable batch and
the same scaling rule. Select checkpoints on the fixed 75-page validation set,
then open the 87-page sealed test only for the final validation-selected
candidates.

Export with `exclude_nms=True` to static 640-by-640 OpenVINO. The Rust provider
implements the released class-wise NMS parameters (score 0.01, IoU 0.7,
top-k 1,000, keep 300), eliminating graph-level dynamic NMS and all Python from
the product. Evaluate FP32 and FP16 first, then standard static INT8 with
representative training images. Keep quantized variants only when their real
laptop throughput/size gain is non-dominated at acceptable validation and
sealed-test quality. Preserve the full runtime's GPU providers and the thin
OpenVINO CPU build.

Primary recipe source:

- <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.8.1/configs/ppyoloe>

## Decision

The exact fine-tuned PP-DocLayoutV3 RT-DETR teacher is the incumbent. Optimize
and compress it before considering another detector:

| Candidate | Model | Purpose |
| --- | --- | --- |
| Quality | teacher FP32, 800 px | accuracy ceiling and portable CPU baseline |
| Compact | selectively quantized teacher | smaller download when its measured quality loss is acceptable |
| Balanced | official RT-DETR INT8 QAT plus soft-label self-distillation | seek the standard INT8 speed/quality point |
| Fast | a compatible smaller RT-DETR/HGNet student, only if it earns promotion | optional later tier, not an assumed result |

The production implementation is direct Rust. The minimal laptop build uses
OpenVINO's C ABI; the full build retains ONNX Runtime providers for CUDA,
TensorRT, DirectML, and other capable hosts. Paddle, PaddleX, Python, Docker,
calibration data, and training code do not ship.

PicoDet-S/M is not the default plan. Paddle's own comparison reports 70.9 and
75.2 mAP@0.5 versus 90.4 for PP-DocLayoutV3, and the completed legal M screen
did not approach the incumbent. That is an architecture/family ceiling before
it is a learning-rate problem. Do not spend another long run on PicoDet unless
new evidence changes that conclusion.

Sources:

- <https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PP-StructureV3.en.md>
- <https://arxiv.org/abs/2111.00902>
- <https://github.com/PaddlePaddle/PaddleOCR/blob/main/ppstructure/layout/README.md>

## Why the first small-model work is rejected

The S/XS and completed M results are not deployable candidates.

- The early S/XS screens used incoherent shortened schedules and therefore do
  not isolate capacity.
- Changing 17 classes to 26 caused every classifier-output tensor to be skipped.
  Backbone, neck, and regression weights transferred; all class channels were
  random.
- S was trained and evaluated at 480 while the exact teacher runs at 800.
  In the 661 pages, the median 480-space height is about 8.2 px for footnotes,
  8.0 px for headers, 9.1 px for bylines, and 7.1 px for page numbers. PicoDet's
  finest prediction stride is 8, so these targets are roughly one feature cell
  high.

The later official M run corrected the obvious schedule issue but still failed
the quality gate. It confirms that a superficially better recipe is not enough.
Any future student must preserve the RT-DETR task/head family, use explicit
teacher distillation, and compare identical metrics: COCO AP@0.5:0.95 is not
interchangeable with the official mAP@0.5 table.

## Honest data surface

The original split leaks layout identity. Thirty-nine of 40 validation pages
and 37 of 40 benchmark pages share an article with training; every held-out
journal also occurs in training. Only 18 of 26 declared classes have any gold
training instance, and the original validation set exercises nine classes.

`training/make_generalization_splits.py` merges all source annotations and
creates this immutable v1 surface:

- train: 499 pages, 259 articles, 23 journals;
- validation: 75 pages, 43 articles, article-disjoint from train;
- test: 87 pages, 48 articles, with APPEAL, CAN-US-LJ, and
  MCGILL-LJ-BACKCAT completely absent from development;
- all article overlaps are zero and development/test journal overlap is zero;
- source and output SHA-256 values are recorded in `split_manifest.json`.

The old 40/40 sets remain useful only for regression against the old teacher.
Its historical score is not an unbiased generalization estimate, but the
existing model remains the incumbent quality tier. Do not retrain it merely to
move its evaluation numbers.

The trained legal ontology has 25 classes. It deliberately omits
`inline_formula`: the 661 pages contain no instance, legal layout does not need
the distinction, and a random unsupported class can only add false positives.
The engine's provider-neutral wire type may still accept `inline_formula` from
other models. Report performance separately for:

1. exact supported classes;
2. engine roles after a fixed collapse (body, heading, footnote, running
   material, reference, table/image/formula);
3. unsupported classes with zero gold test evidence.

Never turn a class with zero test instances into a claimed success.

## Data boundary

The retained 661 pages are the complete training-data surface for this effort.
Do not fetch, generate, pseudo-label, or mix additional training data.

## Compatible V3 student initialization

Do not initialize another PicoDet or D-FINE student. Use the same
PP-DocLayoutV3 task architecture and fixed legal-25 ontology, changing only
components for which PaddleDetection already publishes Mask RT-DETR size
recipes:

| Tier | Backbone/neck recipe | Trainable parameters | Initialization |
| --- | --- | ---: | --- |
| quality | HGNetV2-L, encoder index 3, expansion 1.0 | about 33.3M | legal V3 teacher |
| balanced | HGNetV2-M, encoder index 2, expansion 0.5 | 19,191,573 | complete checkpoint coverage |
| fast candidate | HGNetV2-S, encoder index 2, expansion 0.5 | 15,035,637 | 99.993% coverage; one 1,024-weight mask adapter is new |

The transformer, 25-way classifier, boxes, masks, and reading-order branch are
unchanged. Build the initial student by loading every exact-name/exact-shape
tensor from Paddle's matching Mask RT-DETR S/M checkpoint, then overlay every
compatible tensor from the legal V3-L teacher. The legal teacher therefore
wins for the classifier and all task-specific state, while Paddle's released
student checkpoint supplies the resized backbone, neck, and projections. Fail
if any trainable tensor lacks provenance, except S's explicitly named
`neck.enc_mask_output.1.weight`; never use a blanket partial-load allowance.

Train with 300 queries. Separately evaluate 100 queries at inference time using
the same checkpoint: Paddle documents loading a 300-query Mask RT-DETR
checkpoint with 100 queries as a supported speed optimization with little
accuracy loss. Because `learnt_init_query` is false, this changes graph work,
not checkpoint tensor shapes. It still requires our own validation gate.

Sources:

- <https://github.com/PaddlePaddle/PaddleDetection/tree/release/2.9/configs/mask_rtdetr>
- <https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/configs/mask_rtdetr/README.md>

## Compression sequence on the 12 GB 3080 Ti

### 1. Freeze the incumbent baseline

Use the exact epoch-14 fine-tune and its 26-label order. Prove Paddle-to-export
tensor parity, then record COCO AP/AP50 on the article-disjoint 75-page
validation set and end-to-end laptop latency. Do not retrain the incumbent just
to obtain a cleaner evaluation number.

### 2. Exhaust deployment-only compression first

Test standard static INT8 QDQ/PTQ, selective backbone INT8, provider-native
FP16/BF16 where supported, and thread/device settings. Reject any candidate
that is slower at equal concurrency. Keep a size-only candidate only when the
footprint improvement is explicitly useful and its quality loss is measured.

### 3. Run the official RT-DETR QAT recipe

Run PaddleSlim AutoCompression QAT with the original FP32 graph as teacher.
Use the shipped RT-DETR configuration: per-channel weight INT8,
moving-average activation INT8, `conv2d`, `depthwise_conv2d`, and `matmul_v2`,
QDQ-compatible `onnx_format`, 200 steps, `3e-5` cosine learning rate, and
soft-label self-distillation. This is a short compression run, not a fresh
detector training run.

The legacy `.pdmodel/.pdiparams` graph is required by PaddleSlim 2.6;
PaddleX 3's PIR `.json` export is not compatible with that graph wrapper. Keep
the 26-output teacher ontology intact during QAT. A future smaller model may
omit unsupported `inline_formula`, but compression must not silently alter the
incumbent's class head.

Sources:

- <https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression/detection>
- <https://github.com/PaddlePaddle/PaddleSlim/blob/develop/example/auto_compression/detection/configs/rtdetr_hgnetv2_l_qat_dis.yaml>

### 4. Train a smaller compatible model only if it can add a tier

Use document-safe augmentation: mild photometric/noise/blur changes and small
scan skew; no vertical flip, mosaic, destructive crop, or large perspective
transform. Keep the legal-25 ontology fixed.

PaddleDetection's generic `DistillModel` cannot be used as-is: its FGD/CWD
wrappers are explicitly limited to PicoDet, RetinaNet, GFL, or PP-YOLOE, and
raw DETR queries are not safely aligned teacher-to-student. The bounded V3
distiller therefore retains all ordinary supervised PP-DocLayoutV3 losses and
adds Paddle's existing channel-wise distillation loss only at shape-aligned
neck outputs: three 256-channel encoder levels and the 32-channel mask feature.
The L teacher stays frozen and in evaluation mode. No query-index KL/MSE loss is
used; DETR-specific research methods require assignment-aware query priors and
would be a separate, substantially larger implementation.

Use a controlled ablation from the identical initialized M checkpoint:

1. supervised-only M and M+CWD use the same seed, pages, augmentations, and
   schedule;
2. compare them on validation after the same early epochs and stop the losing
   branch quickly;
3. train S only after M establishes the achievable accuracy/latency curve;
4. score 300- and 100-query exports from each surviving checkpoint;
5. do not open the journal-held-out test until one validation-selected student
   is genuinely promotable.

The 320-pixel one-page CPU smoke must pass first with finite supervised,
encoder-distillation, and mask-distillation losses. A full 640 GPU memory smoke
must then fit the 12 GB 3080 Ti before a long run.

Source for why naïve query-to-query loss is not a safe DETR distiller:

- <https://openaccess.thecvf.com/content/ICCV2023/html/Chang_DETRDistill_A_Universal_Knowledge_Distillation_Framework_for_DETR-families_ICCV_2023_paper.html>

### PP-DocLayoutV3 export contract

Export a surviving L/M/S checkpoint once with the pinned, portable-ops
PaddleDetection checkout. Then run
`ppdoc-lite-runtime/tools/build_paddle_openvino_pack.py` on that inference
directory. The builder pins Paddle2ONNX 1.0.5 and opset 16, retains only the
decoded boxes/counts graph results, runs `ovc`, verifies the OpenVINO output and
input-shape contract, and writes the hash-locked pack receipts. This is the
required product route: direct Paddle-to-OpenVINO conversion was output-exact
but about four times slower on the laptop. The complete command and dependency
boundary are in `ppdoc-lite-runtime/README.md`.

### D-FINE export contract

Export each surviving D-FINE checkpoint twice from the pinned PaddleDetection
2.9 checkout. The ordinary decoded export is the Paddle parity oracle. The
product/quantization export sets the architecture's supported
`exclude_post_process=True` flag, retaining `[1, queries, 4]` boxes and
`[1, queries, classes]` logits so the shared Rust top-k decoder remains outside
the graph:

```shell
python tools/export_model.py -c LEGAL25_CONFIG.yml \
  -o weights=CHECKPOINT trt=True \
  --output_dir=export-decoded
python tools/export_model.py -c LEGAL25_CONFIG.yml \
  -o weights=CHECKPOINT trt=True exclude_post_process=True \
  --output_dir=export-raw

paddle2onnx --model_dir=export-raw/DEIM-D-FINE-S-640 \
  --model_filename=model.pdmodel --params_filename=model.pdiparams \
  --opset_version=16 --save_file=legal25-dfine-s-raw.onnx
```

Use the exact vendor-pinned `onnx==1.13.0` and `paddle2onnx==1.0.5` first. Do
not upgrade an exporter in place to work around a conversion failure; record a
separate candidate. Package the raw graph with the `rtdetr_raw` contract, then
prove decoded Paddle/ONNX/OpenVINO predictions agree before benchmarking or
quantizing it.

## Evaluation and promotion

Model selection uses validation only. Open the journal-held-out test once per
finalist family.

Measure:

- COCO AP@0.5:0.95, AP50, AP75, recall, size buckets, and every supported class;
- the fixed engine-role collapse and critical macro recall;
- native-line assignment accuracy after snapping region boxes to parsed/OCR
  lines;
- downstream heading, footnote, header/footer, reference, and reading-order
  contracts;
- born-digital, degraded-scan, journal, and page-type slices;
- bootstrap confidence intervals, because 87 test pages are not infinite data.

Raw detector IoU is not the only product metric. The Rust parser and Kraken
already supply accurate line geometry, so the provider should snap semantic
regions to those lines. This can recover boundary accuracy more cheaply than a
larger raster model.

Initial promotion targets are:

- balanced: within two points of the incumbent teacher on engine-role macro
  quality, with no critical-class collapse;
- fast: within four points of balanced and materially faster on the laptop;
- a compressed artifact: no more than 0.5 AP or one point critical macro-recall
  loss, and at least a 10% latency win unless footprint alone is required.

## Quantization and runtime ablation

Export fixed-shape rectangle-only ONNX graphs for every surviving resolution.
Fuse Conv/BN, remove unused mask/order outputs, hold one session open, reuse
buffers, and benchmark batch 1 plus a small multi-page batch. Measure complete
PDF render, preprocessing, inference, postprocessing, cold start, warm p50/p95,
peak RSS, model/runtime bytes, and thread count on the actual laptop.

For each S/M finalist, record this matrix:

1. FP32 baseline and ORT optimization levels;
2. FP16/mixed FP16 for CUDA, DirectML/WinML, and supported OpenVINO devices;
3. BF16 where the provider and CPU support it;
4. dynamic INT8 as a negative-control baseline;
5. static QDQ S8S8 per-channel with MinMax, Entropy, and Percentile calibration;
6. U8S8 and U8U8, reduce-range, per-tensor, and QOperator variants where the
   target provider implements them efficiently;
7. selective mixed INT8/FP32 using quantization-loss debugging;
8. Paddle QAT only if the best PTQ artifact misses its accuracy gate;
9. weight-only INT4 recorded as inapplicable or rejected for the convolutional
   student unless a provider gains real kernels. ORT's INT4 path targets
   constant MatMul weights, not PicoDet's dominant convolutions.

Calibration is a class/journal-balanced subset of the retained training split.
Quantization is hardware-specific: the existing teacher screen already showed
FP32 beating every INT8/INT4 variant on the older Ryzen CPU. Ship FP32 when it
wins; do not treat smaller bytes as faster inference.

ONNX Runtime recommends static rather than dynamic quantization for CNNs and
S8S8 QDQ as the first CPU choice. Its provider, threading, graph optimization,
ORT-format, and reduced-operator-build facilities supply the remaining standard
deployment work:

- <https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
- <https://onnxruntime.ai/docs/performance/tune-performance/threading.html>
- <https://onnxruntime.ai/docs/execution-providers/>
- <https://onnxruntime.ai/docs/build/custom.html>

## Product shape

The Legal PDF Parser owns the provider and model manifests. Reuse the
Kraken implementation's `ort`, `hayro`, and `image` boundary, pass rendered RGB
tensors in memory, and write source region labels before the existing structure
passes. A reduced ORT build may be produced after the winning model set freezes;
do not build one per experiment.

The only honest way to approach the pure Rust parser's average throughput is a
hybrid route:

- confident born-digital pages use native geometry and can skip the neural
  detector or request only the fast tier;
- ambiguous born-digital pages use S/M and snap results to native lines;
- scanned pages share the render/line geometry already required by Kraken, so
  layout work is batched alongside the slower OCR path;
- quality mode explicitly selects M or the heavy teacher.

Expose `auto`, `fast`, `balanced`, and `quality`; let measured capability and
page evidence choose, while always allowing an explicit override. This yields a
fast default without deleting the heavy accuracy ceiling.
