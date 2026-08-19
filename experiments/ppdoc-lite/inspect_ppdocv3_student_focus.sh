#!/usr/bin/env bash
set -u

repo=${LEGALPDF_TRAINING_ROOT:?set LEGALPDF_TRAINING_ROOT}/PaddleDetection-2.9-dfine-amp-20260814
cd "$repo"

printf '%s\n' '=== DocLayoutV3 transformer subclass ==='
grep -n -A360 -B15 'class DocLayoutV3Transformer' \
  ppdet/modeling/transformers/mask_rtdetr_transformer.py

printf '%s\n' '=== DocLayoutV3 head ==='
file=$(grep -R -l 'class DocLayoutV3Head' ppdet/modeling | head -n 1)
grep -n -A360 -B15 'class DocLayoutV3Head' "$file"

printf '%s\n' '=== MaskHybridEncoder shape contract ==='
file=$(grep -R -l 'class MaskHybridEncoder' ppdet/modeling/necks | head -n 1)
grep -n -A320 -B15 'class MaskHybridEncoder' "$file"
