#!/usr/bin/env bash
set -euo pipefail

destination=${1:?usage: fetch_mask_rtdetr_student_pretrains.sh DESTINATION}
mkdir -p "$destination"

base=https://paddledet.bj.bcebos.com/models
for size in s m; do
  name="mask_rtdetr_hgnetv2_${size}_6x_coco.pdparams"
  output="$destination/$name"
  partial="$output.part"
  if [[ ! -f "$output" ]]; then
    printf 'Downloading %s\n' "$name"
    curl --fail --location --retry 3 --continue-at - \
      --output "$partial" "$base/$name"
    mv "$partial" "$output"
  fi
  sha256sum "$output"
  stat --printf='%s bytes %n\n' "$output"
done
