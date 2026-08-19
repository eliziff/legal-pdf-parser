#!/usr/bin/env python3
"""Transplant the official PP-DocLayout 23-class head into the legal 25 classes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import paddle


SOURCE_LABELS = (
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "table_title", "reference",
    "doc_title", "footnote", "header", "algorithm", "footer", "seal",
    "chart_title", "chart", "formula_number", "header_image", "footer_image",
    "aside_text",
)
TARGET_LABELS = (
    "abstract", "algorithm", "chart", "content", "display_formula", "doc_title",
    "figure_title", "footer", "footer_image", "footnote", "formula_number",
    "header", "header_image", "image", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote", "block_quote", "byline",
)
RENAMES = {"formula": "display_formula", "aside_text": "vertical_text"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    state = paddle.load(str(args.source))
    source_index = {label: index for index, label in enumerate(SOURCE_LABELS)}
    target_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    mapping = {
        source: RENAMES.get(source, source)
        for source in SOURCE_LABELS
        if RENAMES.get(source, source) in target_index
    }
    initialized = sorted(set(TARGET_LABELS) - set(mapping.values()))
    if initialized != ["block_quote", "byline", "reference_content", "vision_footnote"]:
        raise AssertionError(initialized)

    rng = np.random.default_rng(args.seed)
    bias_prior = -math.log(99.0)
    transplanted: list[dict[str, object]] = []
    for level in range(4):
        aliases = (
            f"head.head_cls_list.{level}",
            f"head.head_cls{level}",
        )
        source_weight = state[f"{aliases[0]}.weight"].numpy()
        source_bias = state[f"{aliases[0]}.bias"].numpy()
        for alias in aliases[1:]:
            if not np.array_equal(source_weight, state[f"{alias}.weight"].numpy()):
                raise ValueError(f"Classifier weight aliases differ at level {level}")
            if not np.array_equal(source_bias, state[f"{alias}.bias"].numpy()):
                raise ValueError(f"Classifier bias aliases differ at level {level}")
        if source_weight.shape[0] != len(SOURCE_LABELS) or source_bias.shape != (len(SOURCE_LABELS),):
            raise ValueError(f"Unexpected classifier shape at level {level}: {source_weight.shape}")

        target_weight = rng.normal(0.0, 0.01, (len(TARGET_LABELS), *source_weight.shape[1:])).astype(source_weight.dtype)
        target_bias = np.full((len(TARGET_LABELS),), bias_prior, dtype=source_bias.dtype)
        for source_label, target_label in mapping.items():
            target_weight[target_index[target_label]] = source_weight[source_index[source_label]]
            target_bias[target_index[target_label]] = source_bias[source_index[source_label]]
        for alias in aliases:
            state[f"{alias}.weight"] = paddle.to_tensor(target_weight)
            state[f"{alias}.bias"] = paddle.to_tensor(target_bias)
        transplanted.append({"level": level, "source_shape": list(source_weight.shape), "target_shape": list(target_weight.shape)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    paddle.save(state, str(temporary))
    os.replace(temporary, args.output)

    check = paddle.load(str(args.output))
    for level in range(4):
        listed = check[f"head.head_cls_list.{level}.weight"].numpy()
        named = check[f"head.head_cls{level}.weight"].numpy()
        if listed.shape[0] != len(TARGET_LABELS) or not np.array_equal(listed, named):
            raise AssertionError(f"Invalid saved classifier at level {level}")

    receipt = {
        "schema_version": "legalpdf.ppdoc_head_transplant.v1",
        "source": {"path": str(args.source), "sha256": sha256(args.source), "labels": list(SOURCE_LABELS)},
        "output": {"path": str(args.output), "sha256": sha256(args.output), "labels": list(TARGET_LABELS)},
        "mapping": mapping,
        "dropped_source_labels": sorted(set(SOURCE_LABELS) - set(mapping)),
        "initialized_target_labels": initialized,
        "initializer": {"seed": args.seed, "weight": "normal(0, 0.01)", "bias": bias_prior},
        "classifier_tensors": transplanted,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".part")
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_tmp.replace(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
