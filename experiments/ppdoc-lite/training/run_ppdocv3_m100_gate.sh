#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'usage: %s RUN_ROOT supervised|cwd [EPOCHS]\n' "$0" >&2
  exit 2
fi

run=$1
branch=$2
epochs=${3:-6}
case "$branch" in
  supervised) distill=() ;;
  cwd)
    distill=(
      --distill-teacher /mnt/f/oajd_compute_storage/ppdoc_lite_models/official/PP-DocLayoutV3_legal25_pretrained.pdparams
      --distill-encoder-weight 1.0
      --distill-mask-weight 1.0
      --distill-tau 1.0
    )
    ;;
  *)
    printf 'branch must be supervised or cwd, got %s\n' "$branch" >&2
    exit 2
    ;;
esac
if [[ -e "$run" ]]; then
  printf 'run target already exists: %s\n' "$run" >&2
  exit 2
fi

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
pretrain=/mnt/f/oajd_compute_storage/ppdoc_lite_models/student_preflight/PP-DocLayoutV3-M_legal25_initialized.pdparams
script_dir=$(cd -- "$(dirname -- "$0")" && pwd)

mkdir -p "$run"
printf '%s\n' "$$" > "$run/launcher.pid"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
exec "$root/venv/bin/python" "$script_dir/train_student.py" \
  --dataset "$dataset" \
  --annotations-dir annotations_generalization_v1 \
  --run-root "$run" \
  --model PP-DocLayoutV3 \
  --pretrain "$pretrain" \
  --mode train \
  --resolution 640 \
  --epochs "$epochs" \
  --batch-size 1 \
  --workers 2 \
  --learning-rate 0.00005 \
  --warmup-steps 20 \
  --eval-interval 1 \
  --log-interval 10 \
  --augmentation document-safe \
  --backbone-arch M \
  --num-queries 100 \
  --no-amp \
  --paddledetection-root "$repo" \
  "${distill[@]}" \
  >> "$run/wrapper.log" 2>&1
