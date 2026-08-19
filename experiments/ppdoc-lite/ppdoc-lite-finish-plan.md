# PPdoc-lite experiment finish plan

## Fixed inputs and gates

- Teacher: `ppdocv3_train1-12_project_full_valrc_20260619_01`; exact Paddle
  inference-file hashes and label order are in `teacher_source.json`.
- Corpus: 661 legal-journal pages already retained on the desktop. The leaky
  original 581/40/40 files remain legacy regression inputs. The promotion
  surface is 499 train, 75 article-disjoint validation, and 87 test pages from
  three wholly held-out journals. Calibration uses the new train split only.
- Runtime contract: direct Rust/ORT rectangle class, confidence, coordinates, and deterministic
  order sufficient for the Legal PDF Parser's source-region adapter.
  Masks and learned order are retained only if an ablation proves downstream
  value.
- Promotion metrics: held-out COCO bbox AP/AP50/recall and per-class failures;
  Paddle/ONNX differential; model bytes; dependency bytes; cold start; peak RSS;
  warm milliseconds/page and pages/second on the target laptop.

## 1. Freeze the thinnest faithful FP32 runtime

1. Hash-lock the source, exported inference configuration, and converted graph.
2. Inventory output ancestry, weights, operators, and installed runtime files.
3. Compare full, mask-free, and rectangle-only graphs on all 80 held-out pages.
4. Compare caller-owned RGB tensors, Pillow, and exact OpenCV preprocessing.
5. Keep a persistent process API and batch path only where measurements win.
6. Prove the `--features ppdoc` Rust binary and external model/runtime pack on
   a clean surface with no Python, NumPy, Paddle/PaddleX, Docker, Hayro/OCR,
   Tesseract, or training packages.

## 2. Train purpose-built smaller students

Keep the existing heavy teacher unchanged as the incumbent quality tier. Train
PP-DocLayout-M from its official adapted checkpoint using PaddleX's shipped
custom-dataset fine-tuning schedule: 100 epochs, batch 1, LR `1e-4`, 100 warmup
steps, evaluation every epoch, and PicoDet's static assigner transition at epoch
10. Preserve the 25-class legal output contract through the explicit 23-to-25
classifier-head transplant. If M succeeds, use Paddle's supported FGD path for
M-to-S distillation. Do not add training data or retrain the incumbent merely to
move its evaluation numbers. Preserve checkpoints, progress, logs, and usable
partial results.

## 3. Applicable optimization and quantization ablation

For teacher and surviving students, measure rather than assume:

- ORT graph optimization levels, optimized ONNX/ORT format, preprocessing
  backend, thread counts, batch sizes, and persistent versus cold sessions;
- dynamic MatMul/Gemm INT8 where applicable;
- static QDQ and QOperator for supported Conv/MatMul subsets;
- S8S8, U8S8, and U8U8 CPU formats, reduce-range, per-tensor/per-channel;
- MinMax, Entropy, and Percentile calibration with frozen train-only sample
  sizes;
- supported weight-only INT8/INT4, mixed exclusions, FP16 GPU, and BF16/INT8
  OpenVINO paths where the graph/provider actually supports them;
- CPU as the portable floor, plus OpenVINO, DirectML, CUDA/TensorRT only as
  optional accelerators.

Unsupported or inapplicable cells are recorded with the exact reason. Screen
on article-disjoint validation, then open the journal-held-out test only for
the surviving finalist families.

## Done when

- A reproducible model pack and clean runtime install work without Docker,
  Paddle, PaddleX, a compiler, or training code.
- At least the standard small students received comparable fresh-training
  screens, with full training/distillation applied according to the fixed gate.
- Every materially applicable standard quantization/optimization family has a
  receipt, including failures and rejected regressions.
- The final speed/quality ladder is benchmarked end to end on the laptop and
  the fastest acceptable tier is wired to the Legal PDF Parser.
- Any gap from the Rust parser's throughput is reported as the measured neural
  layout floor, not hidden by excluding startup, preprocessing, or bad pages.
