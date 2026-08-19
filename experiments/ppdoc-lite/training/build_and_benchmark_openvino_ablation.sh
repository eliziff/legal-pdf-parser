#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 ROOT VARIANT_ID [DETECTIONS_PER_IMAGE=75]" >&2
  exit 2
fi

root=$1
variant_id=$2
detections_per_image=${3:-75}
[[ $detections_per_image =~ ^[1-9][0-9]*$ ]] || {
  echo "DETECTIONS_PER_IMAGE must be a positive integer" >&2
  exit 2
}
model_dir="$root/paddle-legacy26/PP-DocLayoutV3-640"
pack_dir="$root/pack/openvino-fp32"
work_dir="$root/intermediate/p2o105-opset16"
benchmark_dir="$root/benchmarks"

python=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/export-venv/bin/python
builder=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/ppdoc-lite-q100-builder/build_paddle_openvino_pack.py
benchmark=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/ppdoc-lite-q100-builder/benchmark.py
paddle2onnx=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/paddle2onnx105-venv/bin/paddle2onnx
ovc=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/export-venv/bin/ovc
source_manifest=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/ppdoc-lite-q100-builder/ppdocv3_640_e15_source.json
source_dir=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/runs/legal25-ppdocv3-640-document-safe-e30-seed20260813-r5
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
annotations="$dataset/annotations_generalization_v1/instance_val.json"

echo "phase=preflight root=$root variant=$variant_id"
for path in \
  "$model_dir/model.pdmodel" \
  "$model_dir/model.pdiparams" \
  "$model_dir/infer_cfg.yml" \
  "$python" \
  "$builder" \
  "$benchmark" \
  "$paddle2onnx" \
  "$ovc" \
  "$source_manifest" \
  "$annotations"; do
  [[ -f "$path" ]] || { echo "missing=$path" >&2; exit 1; }
done

if [[ ! -f "$pack_dir/manifest.json" ]]; then
  echo "phase=convert"
  "$python" "$builder" \
    --model-dir "$model_dir" \
    --inference-yml "$model_dir/infer_cfg.yml" \
    --source-manifest "$source_manifest" \
    --source-dir "$source_dir" \
    --variant-id "$variant_id" \
    --output-dir "$pack_dir" \
    --paddle2onnx "$paddle2onnx" \
    --ovc "$ovc" \
    --work-dir "$work_dir" \
    --detections-per-image "$detections_per_image" \
    --output-width 6
else
  echo "phase=convert status=already-complete"
fi

mkdir -p "$benchmark_dir"
echo "phase=benchmark pages=75"
"$python" "$benchmark" run \
  --model-pack "$pack_dir" \
  --annotations "$annotations" \
  --image-root "$dataset/images" \
  --output "$benchmark_dir/openvino-fp32-val-thr0001.json" \
  --device openvino-native \
  --openvino-device CPU \
  --inference-precision f32 \
  --performance-hint latency \
  --openvino-threads 8 \
  --warmup-runs 2 \
  --threshold 0.001 \
  --image-backend opencv \
  --input-mode paths \
  --score

echo "phase=complete pack=$pack_dir benchmark=$benchmark_dir/openvino-fp32-val-thr0001.json"
