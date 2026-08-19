#!/usr/bin/env bash
set -euo pipefail

root=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}
python="$root/venv/bin/python"
repo="$root/PaddleDetection-2.9-dfine-amp-20260814"
dataset=/mnt/f/oajd_compute_storage/ppdoclayoutv3_training_runs/ppdocv3_train1-12_project_full_20260619_01/dataset
models=/mnt/f/oajd_compute_storage/ppdoc_lite_models/official
preflight=/mnt/f/oajd_compute_storage/ppdoc_lite_models/student_preflight
teacher="$models/PP-DocLayoutV3_legal25_pretrained.pdparams"
recipe=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/train_student.py
initializer=${LEGALPDF_TRAINING_TOOLS:?set LEGALPDF_TRAINING_TOOLS}/prepare_ppdocv3_student.py

for size in M S; do
  lower=${size,,}
  run="$preflight/ppdocv3-$lower-v1"
  checkpoint="$models/mask_rtdetr_hgnetv2_${lower}_6x_coco.pdparams"
  output="$preflight/PP-DocLayoutV3-${size}_legal25_initialized.pdparams"
  receipt="$preflight/PP-DocLayoutV3-${size}_legal25_initialized.receipt.json"
  if [[ -e "$run/run_manifest.json" || -e "$output" || -e "$receipt" ]]; then
    printf 'Preflight target already exists for %s; refusing to overwrite it.\n' "$size" >&2
    exit 1
  fi

  "$python" "$recipe" \
    --dataset "$dataset" \
    --annotations-dir annotations_generalization_v1 \
    --run-root "$run" \
    --model PP-DocLayoutV3 \
    --pretrain "$teacher" \
    --mode check \
    --resolution 640 \
    --epochs 30 \
    --batch-size 1 \
    --workers 2 \
    --learning-rate 0.00005 \
    --warmup-steps 20 \
    --eval-interval 1 \
    --augmentation document-safe \
    --backbone-arch "$size" \
    --num-queries 300 \
    --paddledetection-root "$repo"

  random_allowance=()
  if [[ "$size" == S ]]; then
    # The released S neck uses 32 internal mask channels and 128 prototypes;
    # legal PP-DocLayoutV3 keeps 32 prototypes. This is the sole new adapter.
    random_allowance=(--allow-random-parameter neck.enc_mask_output.1.weight)
  fi
  "$python" "$initializer" \
    --paddledetection-root "$repo" \
    --config "$run/config/PP-DocLayoutV3-640.yml" \
    --student-pretrain "$checkpoint" \
    --teacher "$teacher" \
    --output "$output" \
    --receipt "$receipt" \
    "${random_allowance[@]}"
done
