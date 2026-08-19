from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ppdoc_lite.runtime import ModelPack, load_rgb, prepare_rgb


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def create_session(pack: ModelPack, threads: int, *, profile_prefix: Path | None = None):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(0, threads)
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if profile_prefix is not None:
        options.enable_profiling = True
        options.profile_file_prefix = str(profile_prefix)
    return ort.InferenceSession(
        str(pack.model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def prepare(pack: ModelPack, image_path: Path) -> dict[str, np.ndarray]:
    config = pack.manifest.get("input", {})
    feeds, _ = prepare_rgb(
        load_rgb(image_path, "opencv"),
        config.get("target_size", [800, 800]),
        backend="opencv",
        scale=float(config.get("scale", 1.0 / 255.0)),
        mean=config.get("mean", [0.0, 0.0, 0.0]),
        std=config.get("std", [1.0, 1.0, 1.0]),
    )
    return {name: np.expand_dims(value, 0) for name, value in feeds.items()}


def tensor_stats(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    finite = array[np.isfinite(array)]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_values": int(finite.size),
        "nan_values": int(np.isnan(array).sum()),
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "mean": float(finite.mean()) if finite.size else None,
        "std": float(finite.std()) if finite.size else None,
    }


def tensor_drift(candidate: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        return {"candidate_shape": list(left.shape), "reference_shape": list(right.shape)}
    delta = np.abs(left - right)
    denominator = np.maximum(np.abs(right), 1e-12)
    return {
        "shape": list(left.shape),
        "max_abs": float(delta.max()),
        "mean_abs": float(delta.mean()),
        "p99_abs": float(np.percentile(delta, 99)),
        "max_relative": float((delta / denominator).max()),
    }


def summarize_profile(path: Path) -> dict[str, Any]:
    events = json.loads(path.read_text(encoding="utf-8"))
    by_op: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    by_provider: dict[str, float] = defaultdict(float)
    by_node: dict[tuple[str, str, str], float] = defaultdict(float)
    for event in events:
        if event.get("cat") != "Node" or not event.get("dur"):
            continue
        args = event.get("args") or {}
        duration = float(event["dur"])
        op_name = str(args.get("op_name") or "unknown")
        provider = str(args.get("provider") or "unknown")
        node_name = str(event.get("name") or "unknown")
        by_op[op_name][0] += duration
        by_op[op_name][1] += 1
        by_provider[provider] += duration
        by_node[(node_name, op_name, provider)] += duration
    total = sum(value[0] for value in by_op.values())
    return {
        "node_time_us": total,
        "operators": [
            {
                "op": name,
                "total_us": values[0],
                "calls": int(values[1]),
                "share": values[0] / total if total else 0.0,
            }
            for name, values in sorted(by_op.items(), key=lambda row: row[1][0], reverse=True)
        ],
        "providers": [
            {"provider": name, "total_us": duration, "share": duration / total if total else 0.0}
            for name, duration in sorted(by_provider.items(), key=lambda row: row[1], reverse=True)
        ],
        "top_nodes": [
            {"node": key[0], "op": key[1], "provider": key[2], "total_us": duration}
            for key, duration in sorted(by_node.items(), key=lambda row: row[1], reverse=True)[:30]
        ],
    }


def run(args: argparse.Namespace) -> int:
    pack = ModelPack.load(args.model_pack)
    output_names = [pack.output_names[key] for key in ("boxes", "logits")]
    feeds = prepare(pack, args.image)
    write_json(
        args.output,
        {
            "schema_version": "legalpdf.ppdoc_lite_onnx_profile.v1",
            "phase": "running",
            "variant_id": pack.variant_id,
        },
    )
    load_started = time.perf_counter()
    session = create_session(pack, args.threads, profile_prefix=args.output.with_suffix(""))
    load_seconds = time.perf_counter() - load_started
    for index in range(args.warmup_runs):
        session.run(output_names, feeds)
        print(f"[ppdoc-lite profile] warmup={index + 1}/{args.warmup_runs}", flush=True)
    times = []
    values = None
    for index in range(args.runs):
        started = time.perf_counter()
        values = session.run(output_names, feeds)
        times.append(time.perf_counter() - started)
        print(f"[ppdoc-lite profile] run={index + 1}/{args.runs} seconds={times[-1]:.4f}", flush=True)
    assert values is not None
    profile_path = Path(session.end_profiling())
    reference_drift = None
    if args.reference_pack:
        reference_pack = ModelPack.load(args.reference_pack)
        reference_names = [reference_pack.output_names[key] for key in ("boxes", "logits")]
        reference_session = create_session(reference_pack, args.threads)
        reference_values = reference_session.run(reference_names, feeds)
        reference_drift = {
            name: tensor_drift(value, reference)
            for name, value, reference in zip(("boxes", "logits"), values, reference_values, strict=True)
        }
    payload = {
        "schema_version": "legalpdf.ppdoc_lite_onnx_profile.v1",
        "phase": "complete",
        "variant_id": pack.variant_id,
        "model_bytes": pack.model_path.stat().st_size,
        "providers": session.get_providers(),
        "threads": args.threads,
        "warmup_runs": args.warmup_runs,
        "runs": args.runs,
        "session_load_seconds": load_seconds,
        "median_seconds": statistics.median(times),
        "p95_seconds": float(np.percentile(times, 95)),
        "outputs": {
            name: tensor_stats(value)
            for name, value in zip(("boxes", "logits"), values, strict=True)
        },
        "reference_drift": reference_drift,
        "profile": summarize_profile(profile_path),
    }
    write_json(args.output, payload)
    profile_path.unlink()
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile PPdoc ONNX operators and raw-output drift")
    parser.add_argument("--model-pack", type=Path, required=True)
    parser.add_argument("--reference-pack", type=Path)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
