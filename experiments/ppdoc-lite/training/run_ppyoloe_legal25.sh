#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  printf 'usage: %s S|M check|smoke|train RUN_ROOT [TRAIN_STUDENT_ARGS...]\n' "$0" >&2
  exit 2
fi

size=$1
mode=$2
run=$3
shift 3
case "$size" in
  S)
    model=PP-YOLOE-S
    pretrain=ppyoloe_plus_crn_s_80e_coco.pdparams
    ;;
  M)
    model=PP-YOLOE-M
    pretrain=ppyoloe_plus_crn_m_80e_coco.pdparams
    ;;
  *)
    printf 'size must be S or M, got %s\n' "$size" >&2
    exit 2
    ;;
esac
case "$mode" in
  check|smoke|train) ;;
  *)
    printf 'mode must be check, smoke, or train, got %s\n' "$mode" >&2
    exit 2
    ;;
esac
if [[ -e "$run" ]]; then
  printf 'run target already exists: %s\n' "$run" >&2
  exit 2
fi

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset="$root/datasets/legal25-generalization-v1"
models=/mnt/f/oajd_compute_storage/ppdoc_lite_models/official
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)
early_stop=()
if [[ "$mode" == train ]]; then
  early_stop=(
    --early-stop-patience 6
    --early-stop-min-epoch 40
    --early-stop-min-delta 0.002
  )
fi

mkdir -p "$run"
printf '%s\n' "$$" > "$run/launcher.pid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$root/venv/bin/python" "$script_dir/train_student.py" \
  --dataset "$dataset" \
  --annotations-dir annotations_generalization_v1 \
  --run-root "$run" \
  --model "$model" \
  --pretrain "$models/$pretrain" \
  --mode "$mode" \
  --workers 2 \
  --augmentation official \
  --seed 20260813 \
  --paddledetection-root "$repo" \
  "${early_stop[@]}" \
  "$@" \
  >> "$run/wrapper.log" 2>&1
