#!/usr/bin/env python3
"""Run a bounded CPU forward pass for a prepared PP-DocLayoutV3 student."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


def describe(value: Any) -> Any:
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {"shape": [int(item) for item in value.shape], "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(key): describe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [describe(item) for item in value]
    return type(value).__name__


def tensors(value: Any):
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddledetection-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()

    root = args.paddledetection_root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    import paddle
    from ppdet.core.workspace import create, load_config

    paddle.set_device("cpu")
    print("building model", flush=True)
    config = load_config(str(args.config.resolve()))
    model = create(config.architecture)
    state = paddle.load(str(args.weights.resolve()))
    model.set_state_dict(state)
    model.eval()
    height, width = (int(item) for item in config.eval_size)
    inputs = {
        "image": paddle.zeros([1, 3, height, width], dtype="float32"),
        "im_shape": paddle.to_tensor([[height, width]], dtype="float32"),
        "scale_factor": paddle.ones([1, 2], dtype="float32"),
    }
    print(f"forward {height}x{width}", flush=True)
    started = time.perf_counter()
    with paddle.no_grad():
        output = model(inputs)
    elapsed = time.perf_counter() - started
    non_finite = [
        [int(item) for item in value.shape]
        for value in tensors(output)
        if value.dtype in (paddle.float16, paddle.float32, paddle.float64)
        and not bool(paddle.isfinite(value).all().item())
    ]
    if non_finite:
        raise RuntimeError(f"Non-finite output tensors: {non_finite}")
    receipt = {
        "schema_version": "legalpdf.ppdocv3_student_forward_preflight.v1",
        "device": "cpu",
        "input_shape": [1, 3, height, width],
        "elapsed_seconds": elapsed,
        "outputs": describe(output),
        "finite": True,
        "config": str(args.config.resolve()),
        "weights": str(args.weights.resolve()),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_suffix(args.receipt.suffix + ".part")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
