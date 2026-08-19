from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ppdoc_lite.runtime import prepare_image


def run(model: Path, params: Path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    import paddle.inference as paddle_infer

    config = paddle_infer.Config(str(model), str(params))
    config.disable_gpu()
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(feeds[name])
    predictor.run()
    return [predictor.get_output_handle(name).copy_to_cpu()
            for name in predictor.get_output_names()]


def digest(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two Paddle inference exports")
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-params", type=Path, required=True)
    parser.add_argument("--candidate-model", type=Path, required=True)
    parser.add_argument("--candidate-params", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-output-prefix", action="store_true")
    args = parser.parse_args()

    feeds, _ = prepare_image(
        args.image, (args.target_size, args.target_size), backend="opencv",
        scale=1.0 / 255.0, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0))
    feeds = {name: np.expand_dims(value, axis=0) for name, value in feeds.items()}
    reference = run(args.reference_model, args.reference_params, feeds)
    candidate = run(args.candidate_model, args.candidate_params, feeds)
    if len(reference) != len(candidate) and not args.allow_output_prefix:
        raise RuntimeError(f"output count differs: {len(reference)} != {len(candidate)}")

    outputs = []
    for index, (left, right) in enumerate(zip(reference, candidate)):
        left, right = np.asarray(left), np.asarray(right)
        same_shape = left.shape == right.shape
        difference = (np.abs(left.astype(np.float64) - right.astype(np.float64))
                      if same_shape else None)
        outputs.append({
            "index": index,
            "reference_shape": list(left.shape),
            "candidate_shape": list(right.shape),
            "reference_sha256": digest(left),
            "candidate_sha256": digest(right),
            "exact": bool(same_shape and np.array_equal(left, right)),
            "max_abs_error": float(difference.max()) if difference is not None else None,
            "mean_abs_error": float(difference.mean()) if difference is not None else None,
            "column_max_abs_error": (
                difference.reshape(-1, difference.shape[-1]).max(axis=0).tolist()
                if difference is not None and difference.ndim == 2 else None),
            "column_mean_abs_error": (
                difference.reshape(-1, difference.shape[-1]).mean(axis=0).tolist()
                if difference is not None and difference.ndim == 2 else None),
        })
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_paddle_paddle_parity.v1",
        "image": str(args.image),
        "reference_output_count": len(reference),
        "candidate_output_count": len(candidate),
        "outputs": outputs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
