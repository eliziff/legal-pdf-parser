#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$root/venv/bin/python"
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
models=/mnt/f/oajd_compute_storage/ppdoc_lite_models
teacher="$models/official/PP-DocLayoutV3_legal25_pretrained.pdparams"
preflight="$models/student_preflight"
recipe=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/train_student.py
forward=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/preflight_ppdocv3_forward.py

q300_config="$root/runs/legal25-ppdocv3-640-document-safe-e30-seed20260813-r5/config/PP-DocLayoutV3-640.yml"
q100_run="$preflight/ppdocv3-l-q100-v1"
if [[ ! -e "$q100_run/run_manifest.json" ]]; then
  "$python" "$recipe" \
    --dataset "$dataset" \
    --annotations-dir annotations_generalization_v1 \
    --run-root "$q100_run" \
    --model PP-DocLayoutV3 \
    --pretrain "$teacher" \
    --mode check \
    --no-amp \
    --resolution 640 \
    --epochs 30 \
    --batch-size 1 \
    --workers 2 \
    --learning-rate 0.00005 \
    --warmup-steps 20 \
    --eval-interval 1 \
    --augmentation document-safe \
    --backbone-arch L \
    --num-queries 100 \
    --paddledetection-root "$repo"
fi

for queries in 300 100; do
  config="$q300_config"
  if [[ "$queries" == 100 ]]; then
    config="$q100_run/config/PP-DocLayoutV3-640.yml"
  fi
  receipt="$preflight/PP-DocLayoutV3-L-q${queries}-cpu-forward.json"
  if [[ -e "$receipt" ]]; then
    printf 'Forward receipt already exists: %s\n' "$receipt" >&2
    exit 1
  fi
  "$python" "$forward" \
    --paddledetection-root "$repo" \
    --config "$config" \
    --weights "$teacher" \
    --receipt "$receipt"
done
