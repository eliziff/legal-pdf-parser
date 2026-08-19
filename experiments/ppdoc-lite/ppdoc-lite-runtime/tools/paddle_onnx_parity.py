from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ppdoc_lite.runtime import prepare_image


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Paddle and ONNX inference tensors")
    parser.add_argument("--paddle-model", type=Path, required=True)
    parser.add_argument("--paddle-params", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    import onnxruntime as ort
    import paddle.inference as paddle_infer

    feeds, _ = prepare_image(
        args.image,
        (args.target_size, args.target_size),
        backend="opencv",
        scale=1.0 / 255.0,
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
    )
    feeds = {name: np.expand_dims(value, axis=0) for name, value in feeds.items()}

    config = paddle_infer.Config(str(args.paddle_model), str(args.paddle_params))
    config.disable_gpu()
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(feeds[name])
    predictor.run()
    paddle_names = predictor.get_output_names()
    paddle_outputs = {
        name: predictor.get_output_handle(name).copy_to_cpu() for name in paddle_names
    }

    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    onnx_names = [value.name for value in session.get_outputs()]
    onnx_values = session.run(onnx_names, {name: feeds[name] for name in feeds})
    onnx_outputs = dict(zip(onnx_names, onnx_values, strict=True))

    rows = []
    for index, (paddle_name, onnx_name) in enumerate(
        zip(paddle_names, onnx_names, strict=True)
    ):
        paddle_value = np.asarray(paddle_outputs[paddle_name])
        onnx_value = np.asarray(onnx_outputs[onnx_name])
        same_shape = paddle_value.shape == onnx_value.shape
        difference = (
            np.abs(paddle_value.astype(np.float64) - onnx_value.astype(np.float64))
            if same_shape
            else None
        )
        rows.append(
            {
                "index": index,
                "paddle_name": paddle_name,
                "onnx_name": onnx_name,
                "paddle_shape": list(paddle_value.shape),
                "onnx_shape": list(onnx_value.shape),
                "paddle_sha256": sha256_array(paddle_value),
                "onnx_sha256": sha256_array(onnx_value),
                "exact": bool(same_shape and np.array_equal(paddle_value, onnx_value)),
                "max_abs_error": float(difference.max()) if difference is not None else None,
                "mean_abs_error": float(difference.mean()) if difference is not None else None,
            }
        )
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_paddle_onnx_parity.v1",
        "image": str(args.image),
        "outputs": rows,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
