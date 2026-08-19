#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  printf 'usage: %s S|M RUN_ROOT CHECKPOINT EXPORT_ROOT [VARIANT_STEM]\n' "$0" >&2
  exit 2
fi

size=$1
run=$2
checkpoint=$3
export_root=$4
case "$size" in
  S) size_id=s ;;
  M) size_id=m ;;
  *)
    printf 'size must be S or M, got %s\n' "$size" >&2
    exit 2
    ;;
esac
variant_stem=${5:-legal25-ppyoloe-${size_id}-640}
if [[ -e "$export_root" ]]; then
  printf 'export target already exists: %s\n' "$export_root" >&2
  exit 2
fi

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
training_repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
export_repo="$root/PaddleDetection-paddle26-export"
export_python="$root/paddle26-export-venv/bin/python"
build_python="$root/export-venv/bin/python"
paddle2onnx="$root/paddle2onnx105-venv/bin/paddle2onnx"
ovc="$root/export-venv/bin/ovc"
builder=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/ppdoc-lite-q100-builder/build_paddle_openvino_pack.py
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
manifest_tool="$script_dir/make_ppyoloe_source_manifest.py"
config="$run/config/PP-YOLOE-${size}-640.yml"
if [[ "$checkpoint" != /* ]]; then
  checkpoint="$run/$checkpoint"
fi
for required in \
  "$config" "$checkpoint" "$manifest_tool" "$builder" \
  "$export_python" "$build_python" "$paddle2onnx" "$ovc"; do
  if [[ ! -f "$required" ]]; then
    printf 'required file is missing: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ ! -d "$training_repo" || ! -d "$export_repo" ]]; then
  printf 'pinned PaddleDetection checkout is missing\n' >&2
  exit 2
fi

mkdir -p "$export_root"
source_manifest="$export_root/source.json"
printf 'phase=source-manifest\n'
"$root/venv/bin/python" "$manifest_tool" \
  --run-root "$run" \
  --checkpoint "$checkpoint" \
  --source-id "$variant_stem" \
  --model-name "PP-YOLOE-${size} legal25 640" \
  --output "$source_manifest"

printf 'phase=paddle-export\n'
(
  cd "$export_repo"
  "$export_python" tools/export_model.py \
    -c "$config" \
    -o "weights=$checkpoint" draw_threshold=0.1 exclude_nms=True \
    --output_dir "$export_root/paddle"
)
mapfile -t inference_configs < <(find "$export_root/paddle" -type f -name infer_cfg.yml)
if [[ ${#inference_configs[@]} -ne 1 ]]; then
  printf 'expected one Paddle inference export, found %d\n' "${#inference_configs[@]}" >&2
  exit 2
fi
inference_yml=${inference_configs[0]}
model_dir=$(dirname -- "$inference_yml")
if ! grep -Eq '^draw_threshold: 0\.1(0*)?$' "$inference_yml"; then
  printf 'Paddle export did not preserve draw_threshold=0.10: %s\n' "$inference_yml" >&2
  exit 2
fi

for precision in fp32 fp16; do
  printf 'phase=openvino-pack precision=%s\n' "$precision"
  "$build_python" "$builder" \
    --model-dir "$model_dir" \
    --inference-yml "$inference_yml" \
    --source-manifest "$source_manifest" \
    --source-dir "$run" \
    --variant-id "$variant_stem-raw-openvino-$precision" \
    --output-dir "$export_root/packs/$precision" \
    --work-dir "$export_root/work/$precision" \
    --paddle2onnx "$paddle2onnx" \
    --ovc "$ovc" \
    --precision "$precision" \
    --inputs image scale_factor \
    --output-contract ppyoloe_raw \
    --boxes-output save_infer_model/scale_0.tmp_0 \
    --scores-output save_infer_model/scale_1.tmp_0 \
    --nms-score-threshold 0.01 \
    --nms-threshold 0.7 \
    --nms-top-k 1000 \
    --ovc-input 'image[1,3,640,640],scale_factor[1,2]'
done
printf 'phase=complete export_root=%s\n' "$export_root"
