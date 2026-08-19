# PPdoc Lite Runtime

This package is the no-Docker export, parity, quantization, and benchmark
harness owned by the Legal PDF Parser. The shipped inference path is
the engine's direct Rust `ppdoc-openvino` or `ppdoc` feature and
`legalpdf ppdoc-images` command. The thin laptop build dynamically loads only
OpenVINO's C runtime; the full build retains ONNX Runtime execution providers
for CUDA, TensorRT, DirectML, oneDNN, and generic ONNX models. Both paths
exclude Python, NumPy, Paddle/PaddleX wheels, Docker, and this harness.
The harness is deliberately split into three surfaces:

- Core: NumPy plus exactly one ONNX Runtime distribution. Callers that provide
  the three preprocessed model tensors use `infer_tensors` and do not need an
  image library.
- Image CLI: opt into either Pillow or OpenCV. OpenCV reproduces the promoted
  PaddleX resize exactly; Pillow is a smaller candidate that must pass the
  measured fidelity gate before promotion.
- Build/benchmark harness: ONNX and pycocotools are standalone scripts under
  `tools/`. They are not modules, dependencies, or entry points in the runtime
  wheel.

The experiment's pinned teacher is
`ppdocv3_train1-12_project_full_valrc_20260619_01`. Its source hashes, retained
split counts, and exact 26-label order live in `../teacher_source.json`, outside
the generic runtime wheel. Model binaries are hash-checked external packs and
are not committed.

Every graph, student, numeric format, execution provider, calibration recipe,
and tuning combination gets a unique `variant_id`. The required product ladder
has at least three independently measured tiers: the incumbent
PP-DocLayoutV3-L quality tier, a provisional PP-YOLOE-M balanced tier, and a
provisional PP-YOLOE-S fast tier. The S/M names become user-facing only if the
trained legal25 models pass held-out quality and real laptop product
benchmarks. FP16 or INT8 variants are retained only when they are non-dominated
in measured speed, size, and quality; numeric format is not itself a tier.

## Direct Rust runtime

```powershell
cargo build --release --locked --no-default-features --features ppdoc-openvino
.\target\release\legalpdf.exe ppdoc-images C:\pages\page-1.png `
  --model-pack C:\models\ppdoc-teacher-openvino-fp32 `
  --runtime C:\runtimes\openvino_c.dll `
  --backend openvino --device CPU --threads 8 `
  --cache-dir C:\models\openvino-cache
```

The `ppdoc-openvino` feature adds only PNG image decoding and dynamic loading
of the OpenVINO C ABI. It does not resolve the `ort` crate. Build with
`--features ppdoc` when ONNX or GPU execution-provider compatibility is
required. Neither feature enables Hayro/OCR, Tesseract, Python, or any training
dependency. The Python command below exists for differential testing and
provider/quantization experiments, not as a production shim.
The optional cache directory uses OpenVINO's compiled-model cache. It mainly
removes repeated device compilation cost, especially for Intel GPU; cached
blobs are device- and OpenVINO-version-specific and are never model artifacts.

## Repeatable legal25 PP-YOLOE export

The S/M training wrapper writes a self-contained config and run manifest. Once
a validation-selected checkpoint exists, export both OpenVINO FP32 and FP16
packs with one command from the pinned desktop training environment:

```shell
bash training/export_ppyoloe_legal25.sh \
  M \
  "$LEGALPDF_TRAINING_ROOT/runs/legal25-ppyoloe-m-640-e80-official-seed20260813-r3" \
  output/best_model.pdparams \
  "$LEGALPDF_TRAINING_ROOT/exports/legal25-ppyoloe-m-best"
```

The script hash-locks the checkpoint, generated training config, run manifest,
dataset contract, and label order. It invokes PaddleDetection's official
`exclude_nms=True` inference export, requires the pinned 0.10 Paddle display
threshold, and builds static 640-by-640 FP32/FP16 OpenVINO packs with the raw
PP-YOLOE output contract. Production uses exact native Rust class-wise NMS;
Paddle, Paddle2ONNX, OpenVINO conversion tools, Python, and the training tree
remain build-time dependencies only and never enter the portable bundle.

Use `S` and the S run root for the fast candidate. The same command and manifest
contract applies to future checkpoints, so a new S/M training run does not
require rediscovering export names, thresholds, shapes, or postprocessing.

## Repeatable PP-DocLayoutV3 export

Do not convert a legacy PP-DocLayoutV3 Paddle graph directly with `ovc`. That
route is valid but measured about four times slower on the laptop. From the
standard PaddleDetection inference export, build the verified runtime pack in
one command:

```shell
python tools/build_paddle_openvino_pack.py \
  --model-dir export/PP-DocLayoutV3-640 \
  --inference-yml export/PP-DocLayoutV3-640/infer_cfg.yml \
  --source-manifest ../ppdocv3_640_e15_source.json \
  --source-dir /path/to/hash-locked-training-run \
  --variant-id legal25-ppdocv3-l640-openvino-fp32 \
  --output-dir packs/legal25-ppdocv3-l640-openvino-fp32 \
  --paddle2onnx /path/to/pinned/paddle2onnx \
  --ovc /path/to/ovc
```

The build environment pins Paddle2ONNX 1.0.5, ONNX 1.13, opset 16, and the
selected OpenVINO release. The command removes only the unused mask graph
result without invoking ONNX's stale-shape inference, converts the decoded
graph, verifies exactly two output ports plus the configured image shape, and
writes hash-locked `manifest.json`, `conversion.json`, and `export.json`
receipts. None of these build dependencies ship with the Rust runtime.

The proven Windows CPU payload contains only these OpenVINO files beside the
Rust executable: `openvino_c.dll`, `openvino.dll`,
`openvino_intel_cpu_plugin.dll`, `openvino_ir_frontend.dll`, and `tbb12.dll`.
Add `openvino_intel_gpu_plugin.dll` to expose Intel GPU devices. ONNX, Paddle,
PyTorch, TensorFlow, AUTO, HETERO, NPU, and training libraries are not shipped.
Build a clean, recursively hash-listed product bundle with:

```powershell
.\tools\prepare_ppdoc_windows_bundle.ps1 `
  -LegalPdfBinary .\target\release\legalpdf.exe `
  -ModelPack C:\models\ppdoc-openvino `
  -OpenVinoLibDir C:\openvino\libs `
  -OutputDir C:\legalpdf-portable
```

The output directory must be empty. Add `-Gpu` only when the Intel GPU plugin
is wanted; compiled-model caches and all training/export dependencies remain
outside the bundle.

## Python parity harness

```powershell
python -m pip install ".[runtime-cpu,images-opencv]"
ppdoc-lite infer `
  --model-pack C:\models\ppdoc-fp32 `
  --image-dir C:\pages `
  --output-dir C:\output\ppdoc_raw_layout_json
```

Use `serve` for a persistent JSONL stdio process so model startup is amortized.
DPI metadata inspection is optional (`--check-dpi`) and is the only CLI feature
that requires Pillow independently of the selected image backend.

## Audited FP32 baseline

The first build phase does no quantization. It records the graph dependency
surface, extracts the rectangle-only PP-DocLayout tail, and packages each graph
with source hashes:

```powershell
python tools\graph.py audit-graph --model model.onnx --output graph_audit.json
python tools\graph.py extract-ppdoc-rect --model model.onnx --output model.rect.onnx
python tools\graph.py prepare-pack `
  --onnx model.onnx `
  --inference-yml paddle_source\inference.yml `
  --source-manifest ..\teacher_source.json `
  --source-dir paddle_source `
  --variant-id teacher-fp32-full `
  --output-dir packs\teacher-fp32-full
```

The rectangle-only pack uses `--output-contract ppdoc_rect_parts`. This removes
mask and learned-order-only tails, but graph audit shows whether the change is
material; it is not described as pruning merely because outputs were hidden.

For the thin Rust build, convert a raw-output ONNX pack with OpenVINO's standard
`ovc` tool and hash-lock the two-file IR without copying Python into the product:

```powershell
ovc packs\teacher-raw-fp32\model.onnx `
  --output_model work\teacher-openvino-fp32\model.xml `
  --compress_to_fp16=False
python tools\graph.py prepare-openvino-pack `
  --xml work\teacher-openvino-fp32\model.xml `
  --bin work\teacher-openvino-fp32\model.bin `
  --source-pack packs\teacher-raw-fp32\manifest.json `
  --variant-id teacher-openvino-fp32 `
  --precision fp32 `
  --output-dir packs\teacher-openvino-fp32
```

Omit `--compress_to_fp16=False` only when intentionally producing the separate
FP16 storage tier; OpenVINO's `ovc` defaults to compressing constants to FP16.

## PP-YOLOE INT8 ablation

Use OpenVINO/NNCF PTQ on the exported raw FP32 pack; do not insert NMS into the
graph. The quantization harness accepts both PP-YOLOE and RT-DETR raw contracts,
uses the same preprocessing and decoder as the parity evaluator, selects a
class/journal-balanced calibration subset from training only, and preserves
progress plus provenance receipts:

```shell
python tools/quantize_nncf.py \
  --source-pack packs/legal25-ppyoloe-m-raw-openvino-fp32 \
  --output-pack packs/legal25-ppyoloe-m-raw-openvino-int8-mixed \
  --variant-id legal25-ppyoloe-m-raw-openvino-int8-mixed \
  --backend openvino \
  --calibration-annotations dataset/annotations_generalization_v1/instance_train.json \
  --validation-annotations dataset/annotations_generalization_v1/instance_val.json \
  --image-root dataset/images \
  --calibration-pages 300 \
  --preset mixed \
  --target-device cpu \
  --max-drop 0.01
```

This follows OpenVINO's basic flow (`nncf.Dataset`, representative calibration,
and `nncf.quantize`) and its documented
`nncf.quantize_with_accuracy_control` fallback. Run full `performance` and
`mixed` PTQ as separate variant IDs first; use accuracy control only if the
full-INT8 quality loss needs selective precision restoration. Save restored
models without FP16-compressing their remaining floating-point weights. The
calibrator and validation code are experiment dependencies and do not ship.

## Benchmark contract

```powershell
python tools\benchmark.py run `
  --model-pack packs\teacher-fp32-full `
  --annotations dataset\annotations\instance_benchmark.json `
  --image-root dataset\images `
  --output results\teacher-fp32-full.json
```

The command reports progress and preserves partial page results. It measures
model bytes, session load, cold first page, warm latency/throughput, RSS, and
held-out COCO bounding-box metrics. Quantized or smaller models are promoted
only against the same split and output contract.
