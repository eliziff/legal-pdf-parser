from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _feeds(path: Path, target_size: int, width: int, height: int) -> dict[str, np.ndarray]:
    image = np.fromfile(path, dtype=np.float32)
    expected = 3 * target_size * target_size
    if image.size != expected:
        raise ValueError(f"{path} has {image.size} floats; expected {expected}")
    return {
        "image": image.reshape(1, 3, target_size, target_size),
        "im_shape": np.asarray([[target_size, target_size]], dtype=np.float32),
        "scale_factor": np.asarray(
            [[target_size / height, target_size / width]], dtype=np.float32
        ),
    }


def _paddle_predict(model: Path, params: Path, feeds: dict[str, np.ndarray], threads: int) -> list[np.ndarray]:
    import paddle.inference as paddle_infer

    config = paddle_infer.Config(str(model), str(params))
    config.disable_gpu()
    config.set_cpu_math_library_num_threads(threads)
    config.switch_ir_optim(True)
    config.disable_glog_info()
    predictor = paddle_infer.create_predictor(config)
    for name in predictor.get_input_names():
        predictor.get_input_handle(name).copy_from_cpu(feeds[name])
    predictor.run()
    return [
        np.asarray(predictor.get_output_handle(name).copy_to_cpu())
        for name in predictor.get_output_names()
    ]


def _onnx_predict(model: Path, feeds: dict[str, np.ndarray], threads: int) -> list[np.ndarray]:
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.log_severity_level = 3
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    names = [output.name for output in session.get_outputs()]
    return [np.asarray(value) for value in session.run(names, feeds)]


def _decode(
    boxes: np.ndarray,
    logits: np.ndarray,
    feeds: dict[str, np.ndarray],
    top_k: int,
) -> np.ndarray:
    centers = boxes[..., :2]
    half_size = boxes[..., 2:] * np.float32(0.5)
    xyxy = np.concatenate((centers - half_size, centers + half_size), axis=-1)
    origin_shape = np.floor(
        feeds["im_shape"] / feeds["scale_factor"] + np.float32(0.5)
    )
    xyxy *= np.tile(origin_shape[:, ::-1], (1, 2))[:, None, :]

    scores = np.float32(1.0) / (np.float32(1.0) + np.exp(-logits))
    flat = scores.reshape(scores.shape[0], -1)
    flat_indices = np.argsort(-flat, axis=1, kind="stable")[:, :top_k]
    selected_scores = np.take_along_axis(flat, flat_indices, axis=1)
    labels = flat_indices % logits.shape[-1]
    queries = flat_indices // logits.shape[-1]
    selected_boxes = xyxy[np.arange(xyxy.shape[0])[:, None], queries]
    order = np.full(selected_scores.shape, -1, dtype=np.float32)
    rows = np.concatenate(
        (
            labels[..., None].astype(np.float32),
            selected_scores[..., None],
            selected_boxes.astype(np.float32),
            order[..., None],
        ),
        axis=-1,
    )
    return rows.reshape(-1, 7)


def _tensor_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    same_shape = left.shape == right.shape
    difference = (
        np.abs(left.astype(np.float64) - right.astype(np.float64)) if same_shape else None
    )
    return {
        "left_shape": list(left.shape),
        "right_shape": list(right.shape),
        "left_sha256": _sha256_array(left),
        "right_sha256": _sha256_array(right),
        "exact": bool(same_shape and np.array_equal(left, right)),
        "max_abs_error": float(difference.max()) if difference is not None else None,
        "mean_abs_error": float(difference.mean()) if difference is not None else None,
    }


def _decoded_comparison(reference: np.ndarray, candidate: np.ndarray, threshold: float) -> dict[str, Any]:
    comparison = _tensor_comparison(reference, candidate)
    same_shape = reference.shape == candidate.shape
    if not same_shape:
        return comparison
    kept_reference = reference[reference[:, 1] >= threshold]
    kept_candidate = candidate[candidate[:, 1] >= threshold]
    relevant = (reference[:, 1] >= threshold) | (candidate[:, 1] >= threshold)
    relevant_reference = reference[relevant]
    relevant_candidate = candidate[relevant]
    comparison.update(
        {
            "label_mismatches": int(np.count_nonzero(reference[:, 0] != candidate[:, 0])),
            "order_mismatches": int(np.count_nonzero(reference[:, 6] != candidate[:, 6])),
            "max_score_error": float(np.max(np.abs(reference[:, 1] - candidate[:, 1]))),
            "max_box_error_pixels": float(
                np.max(np.abs(reference[:, 2:6] - candidate[:, 2:6]))
            ),
            "threshold": threshold,
            "reference_kept": int(kept_reference.shape[0]),
            "candidate_kept": int(kept_candidate.shape[0]),
            "relevant_rows": int(np.count_nonzero(relevant)),
            "relevant_label_mismatches": int(
                np.count_nonzero(relevant_reference[:, 0] != relevant_candidate[:, 0])
            ),
            "relevant_max_score_error": float(
                np.max(np.abs(relevant_reference[:, 1] - relevant_candidate[:, 1]))
            ),
            "relevant_max_box_error_pixels": float(
                np.max(np.abs(relevant_reference[:, 2:6] - relevant_candidate[:, 2:6]))
            ),
        }
    )
    return comparison


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prove raw Paddle/ONNX outputs and external RT-DETR decoding against decoded Paddle."
    )
    parser.add_argument("--reference-model", type=Path, required=True)
    parser.add_argument("--reference-params", type=Path, required=True)
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--raw-params", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-f32", type=Path, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--target-size", type=int, default=800)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    feeds = _feeds(args.image_f32, args.target_size, args.width, args.height)
    reference = _paddle_predict(
        args.reference_model, args.reference_params, feeds, args.threads
    )[0]
    paddle_raw = _paddle_predict(args.raw_model, args.raw_params, feeds, args.threads)
    onnx_raw = _onnx_predict(args.onnx, feeds, args.threads)
    if len(paddle_raw) != 2 or len(onnx_raw) != 2:
        raise ValueError("Raw models must return boxes and logits")

    paddle_decoded = _decode(*paddle_raw, feeds, args.top_k)
    onnx_decoded = _decode(*onnx_raw, feeds, args.top_k)
    payload = {
        "schema_version": "legalpdf.ppdoc_raw_export_parity.v1",
        "raw_backend": {
            "boxes": _tensor_comparison(paddle_raw[0], onnx_raw[0]),
            "logits": _tensor_comparison(paddle_raw[1], onnx_raw[1]),
        },
        "decoded_against_reference": {
            "paddle_raw": _decoded_comparison(reference, paddle_decoded, args.threshold),
            "onnx_raw": _decoded_comparison(reference, onnx_decoded, args.threshold),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
