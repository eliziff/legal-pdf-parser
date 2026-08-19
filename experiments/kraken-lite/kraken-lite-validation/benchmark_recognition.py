from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import statistics
import time
from pathlib import Path

import onnxruntime as ort
from PIL import Image

from kraken_lite.geometry import rectify_line
from kraken_lite.model_pack import ModelPack
from kraken_lite.recognition import prepare_line


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("segmentations", type=Path)
    parser.add_argument("pack", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-base", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-lines", type=int, default=0)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--provider", default="CPUExecutionProvider")
    args = parser.parse_args()

    pack = ModelPack.load(args.pack)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(0, args.threads)
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if args.provider == "DmlExecutionProvider":
        opts.enable_mem_pattern = False
    if args.block_base > 0:
        opts.add_session_config_entry(
            "session.dynamic_block_base", str(args.block_base)
        )
    session = ort.InferenceSession(
        str(pack.model_path),
        sess_options=opts,
        providers=[args.provider, "CPUExecutionProvider"],
    )
    input_name = str(pack.manifest["model"].get("input", "image"))
    output_name = str(pack.manifest["model"].get("output", "logits"))

    prepared = []
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    for index, item in enumerate(corpus, 1):
        segmentation = json.loads(
            (args.segmentations / f"{item['id']}.candidate.json").read_text(
                encoding="utf-8"
            )
        )
        with Image.open(item["image"]) as source:
            image = source.convert("RGB")
        lines = segmentation["lines"]
        if args.max_lines > 0:
            per_page = max(1, math.ceil(args.max_lines / len(corpus)))
            if len(lines) > per_page:
                stride = len(lines) / per_page
                lines = [lines[int(line_index * stride)] for line_index in range(per_page)]
        for line in lines:
            rectification = rectify_line(
                image,
                [tuple(point) for point in line["baseline"]],
                [tuple(point) for point in line["boundary"]],
            )
            prepared.append(prepare_line(rectification.image, pack.manifest).tensor)
        print(
            f"[{index}/{len(corpus)}] {item['id']} tensors={len(prepared)}",
            flush=True,
        )

    if args.max_lines > 0 and len(prepared) > args.max_lines:
        stride = len(prepared) / args.max_lines
        prepared = [prepared[int(index * stride)] for index in range(args.max_lines)]
        print(f"sampled tensors={len(prepared)}", flush=True)

    session.run([output_name], {input_name: prepared[0]})
    measurements = []
    for repeat in range(1, args.repeats + 1):
        started = time.perf_counter()
        if args.workers == 1:
            for tensor in prepared:
                session.run([output_name], {input_name: tensor})
        else:
            def infer(tensor):
                return session.run([output_name], {input_name: tensor})

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                list(pool.map(infer, prepared))
        elapsed = time.perf_counter() - started
        measurements.append(elapsed)
        print(
            f"repeat={repeat}/{args.repeats} lines={len(prepared)} seconds={elapsed:.3f}",
            flush=True,
        )

    result = {
        "pack": pack.id,
        "modelBytes": pack.model_path.stat().st_size,
        "lines": len(prepared),
        "blockBase": args.block_base,
        "threads": args.threads,
        "workers": args.workers,
        "provider": args.provider,
        "activeProviders": session.get_providers(),
        "repeats": measurements,
        "medianSeconds": statistics.median(measurements),
        "linesPerSecond": len(prepared) / statistics.median(measurements),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
