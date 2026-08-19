#!/usr/bin/env python3
"""Transplant the official PP-DocLayoutV3 classifier into the legal-25 ontology."""

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
    "abstract", "algorithm", "aside_text", "chart", "content",
    "display_formula", "doc_title", "figure_title", "footer", "footer_image",
    "footnote", "formula_number", "header", "header_image", "image",
    "inline_formula", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote",
)
TARGET_LABELS = (
    "abstract", "algorithm", "chart", "content", "display_formula", "doc_title",
    "figure_title", "footer", "footer_image", "footnote", "formula_number",
    "header", "header_image", "image", "number", "paragraph_title", "reference",
    "reference_content", "seal", "table", "text", "vertical_text",
    "vision_footnote", "block_quote", "byline",
)


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

    paddle.set_device("cpu")
    state = paddle.load(str(args.source))
    source_index = {label: index for index, label in enumerate(SOURCE_LABELS)}
    target_index = {label: index for index, label in enumerate(TARGET_LABELS)}
    common = tuple(label for label in SOURCE_LABELS if label in target_index)
    initialized = sorted(set(TARGET_LABELS) - set(common))
    dropped = sorted(set(SOURCE_LABELS) - set(common))
    if initialized != ["block_quote", "byline"] or dropped != ["aside_text", "inline_formula"]:
        raise AssertionError((initialized, dropped))

    rng = np.random.default_rng(args.seed)
    embedding = state["transformer.denoising_class_embed.weight"].numpy()
    score_weight = state["transformer.score_head.weight"].numpy()
    score_bias = state["transformer.score_head.bias"].numpy()
    if embedding.shape != (25, 256) or score_weight.shape != (256, 25) or score_bias.shape != (25,):
        raise ValueError(
            f"Unexpected classifier shapes: {embedding.shape}, {score_weight.shape}, {score_bias.shape}"
        )

    target_embedding = rng.normal(0.0, 1.0, embedding.shape).astype(embedding.dtype)
    xavier_bound = math.sqrt(6.0 / (score_weight.shape[0] + score_weight.shape[1]))
    target_score_weight = rng.uniform(
        -xavier_bound, xavier_bound, score_weight.shape
    ).astype(score_weight.dtype)
    target_score_bias = np.full(score_bias.shape, -math.log(99.0), dtype=score_bias.dtype)
    for label in common:
        source_channel = source_index[label]
        target_channel = target_index[label]
        target_embedding[target_channel] = embedding[source_channel]
        target_score_weight[:, target_channel] = score_weight[:, source_channel]
        target_score_bias[target_channel] = score_bias[source_channel]

    state["transformer.denoising_class_embed.weight"] = paddle.to_tensor(target_embedding)
    state["transformer.score_head.weight"] = paddle.to_tensor(target_score_weight)
    state["transformer.score_head.bias"] = paddle.to_tensor(target_score_bias)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    paddle.save(state, str(temporary))
    os.replace(temporary, args.output)

    check = paddle.load(str(args.output))
    for label in common:
        source_channel = source_index[label]
        target_channel = target_index[label]
        if not np.array_equal(
            embedding[source_channel],
            check["transformer.denoising_class_embed.weight"][target_channel].numpy(),
        ):
            raise AssertionError(f"Embedding transplant failed for {label}")
        if not np.array_equal(
            score_weight[:, source_channel],
            check["transformer.score_head.weight"][:, target_channel].numpy(),
        ):
            raise AssertionError(f"Score transplant failed for {label}")

    receipt = {
        "schema_version": "legalpdf.ppdocv3_head_transplant.v1",
        "source": {
            "path": str(args.source),
            "sha256": sha256(args.source),
            "labels": list(SOURCE_LABELS),
        },
        "output": {
            "path": str(args.output),
            "sha256": sha256(args.output),
            "labels": list(TARGET_LABELS),
        },
        "copied_labels": list(common),
        "dropped_source_labels": dropped,
        "initialized_target_labels": initialized,
        "initializer": {
            "seed": args.seed,
            "denoising_embedding": "normal(0, 1)",
            "score_weight": f"xavier_uniform(bound={xavier_bound})",
            "score_bias": -math.log(99.0),
        },
        "classifier_tensors": {
            "transformer.denoising_class_embed.weight": list(embedding.shape),
            "transformer.score_head.weight": list(score_weight.shape),
            "transformer.score_head.bias": list(score_bias.shape),
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".part")
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_tmp.replace(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
