#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 OUTPUT_ROOT QUERIES EVAL_IDX" >&2
  exit 2
fi

output_root=$1
queries=$2
eval_idx=$3
training_root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$training_root/paddle26-export-venv/bin/python"
repo="$training_root/PaddleDetection-paddle26-export"
source_run="$training_root/runs/legal25-ppdocv3-640-document-safe-e30-seed20260813-r5"
config="$source_run/config/PP-DocLayoutV3-640.yml"
checkpoint="$source_run/output/14.pdparams"
export_root="$output_root/paddle-legacy26"
expected="$export_root/PP-DocLayoutV3-640"

[[ $queries =~ ^[1-9][0-9]*$ ]] || { echo "QUERIES must be a positive integer" >&2; exit 2; }
[[ $eval_idx =~ ^[0-5]$ ]] || { echo "EVAL_IDX must be in 0..5" >&2; exit 2; }
for path in "$python" "$repo/tools/export_model.py" "$config" "$checkpoint"; do
  [[ -f "$path" ]] || { echo "missing=$path" >&2; exit 1; }
done
if [[ -e "$expected" ]]; then
  echo "refusing to overwrite existing export=$expected" >&2
  exit 1
fi

mkdir -p "$export_root"
echo "phase=export queries=$queries eval_idx=$eval_idx order=false resolution=640"
cd "$repo"
"$python" tools/export_model.py \
  -c "$config" \
  --output_dir "$export_root" \
  -o \
  weights="$checkpoint" \
  use_gpu=False \
  trt=True \
  export_with_pir=False \
  DocLayoutV3Transformer.num_queries="$queries" \
  DocLayoutV3Transformer.eval_idx="$eval_idx" \
  DocLayoutV3PostProcess.num_top_queries="$queries" \
  DocLayoutV3PostProcess.with_order=False

for path in "$expected/model.pdmodel" "$expected/model.pdiparams" "$expected/infer_cfg.yml"; do
  [[ -f "$path" ]] || { echo "missing-export=$path" >&2; exit 1; }
done
echo "phase=complete export=$expected"
