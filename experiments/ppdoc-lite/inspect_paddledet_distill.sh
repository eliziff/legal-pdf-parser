#!/usr/bin/env bash
set -euo pipefail

repo=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/PaddleDetection-2.9-dfine-amp-20260814
cd "$repo"

printf '%s\n' '=== candidate configs ==='
find configs -type f \
  | grep -Ei '(^|/)(layout_analysis|rtdetr|slim)/|distill|hgnet' \
  | sort \
  | head -n 400

printf '%s\n' '=== slim implementation ==='
find ppdet/slim -maxdepth 2 -type f | sort
grep -R -n -E 'class Distill|DistillModel|Distill.*Loss|KnowledgeDistillation|FGD' \
  ppdet/slim configs/slim 2>/dev/null \
  | head -n 300

printf '%s\n' '=== layout source config ==='
sed -n '1,240p' configs/layout_analysis/PP-DocLayoutV3.yaml

printf '%s\n' '=== backbone variants ==='
grep -n -A180 -B10 -E 'class PPHGNetV2|arch_configs|stage_config' \
  ppdet/modeling/backbones/hgnet_v2.py \
  | head -n 300

printf '%s\n' '=== architecture hooks ==='
grep -R -n -E 'class PPDocLayoutV3|class RTDETR|return.*loss|order_loss|mask' \
  ppdet/modeling/architectures ppdet/modeling/heads \
  | head -n 300
