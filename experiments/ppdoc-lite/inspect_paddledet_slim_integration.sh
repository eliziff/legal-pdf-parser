#!/usr/bin/env bash
set -u

repo=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/PaddleDetection-2.9-dfine-amp-20260814
cd "$repo"

printf '%s\n' '=== slim package ==='
sed -n '1,280p' ppdet/slim/__init__.py

printf '%s\n' '=== trainer slim integration ==='
grep -n -A130 -B40 -E 'slim|DistillModel' ppdet/engine/trainer.py | head -n 520

printf '%s\n' '=== train CLI slim integration ==='
grep -n -A100 -B30 -E 'slim|Trainer' tools/train.py | head -n 360

printf '%s\n' '=== CWD loss implementation ==='
grep -n -A130 -B20 'class CWDFeatureLoss' ppdet/slim/distill_loss.py

printf '%s\n' '=== example generic distill configs ==='
find configs/slim -type f -iname '*distill*.yml' | sort | head -n 8
for file in $(find configs/slim -type f -iname '*distill*.yml' | sort | head -n 3); do
  printf '\n--- %s ---\n' "$file"
  sed -n '1,260p' "$file"
done
