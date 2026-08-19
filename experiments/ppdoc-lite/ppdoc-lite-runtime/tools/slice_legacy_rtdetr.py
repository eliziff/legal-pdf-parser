from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import paddle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save a legacy Paddle RT-DETR graph with raw boxes/logits fetches."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--classes", type=int, default=26)
    parser.add_argument("--boxes-name")
    parser.add_argument("--logits-name")
    parser.add_argument("--target-size", type=int, default=800)
    args = parser.parse_args()

    source_model = args.model_dir / "model.pdmodel"
    source_params = args.model_dir / "model.pdiparams"
    paddle.enable_static()
    executor = paddle.static.Executor(paddle.CPUPlace())
    program, feed_names, _ = paddle.static.load_inference_model(
        str(args.model_dir),
        executor,
        model_filename=source_model.name,
        params_filename=source_params.name,
    )
    block = program.global_block()
    source_operation_count = len(block.ops)
    producers = {
        name: (index, operation.type)
        for index, operation in enumerate(block.ops)
        for name in operation.output_arg_names
    }

    def candidates(shape: list[int], operation_type: str) -> list[tuple[int, str]]:
        found = []
        for variable in block.vars.values():
            try:
                variable_shape = list(variable.shape)
            except RuntimeError:
                continue
            producer = producers.get(variable.name)
            if variable_shape == shape and producer and producer[1] == operation_type:
                found.append((producer[0], variable.name))
        return sorted(found)

    logits = candidates([-1, 300, args.classes], "elementwise_add")
    if args.logits_name:
        logits_name = args.logits_name
        logits_index = producers[logits_name][0]
    elif logits:
        logits_index, logits_name = logits[-1]
    else:
        raise ValueError("could not find final RT-DETR class logits")

    boxes = [item for item in candidates([-1, 300, 4], "sigmoid") if item[0] < logits_index]
    if args.boxes_name:
        boxes_name = args.boxes_name
        boxes_index = producers[boxes_name][0]
    elif boxes:
        boxes_index, boxes_name = boxes[-1]
    else:
        raise ValueError("could not find final normalized RT-DETR boxes")

    if "image" not in feed_names:
        raise ValueError("legacy RT-DETR graph has no image feed")
    feed_names = ["image"]

    wrapper_vars = set()
    for index in reversed(range(len(block.ops))):
        operation = block.ops[index]
        outputs = operation.output_arg_names
        if operation.type == "fetch":
            block._remove_op(index)
        elif operation.type == "scale" and any(
            name.startswith("save_infer_model/scale_") for name in outputs
        ):
            wrapper_vars.update(outputs)
            block._remove_op(index)
    for name in wrapper_vars:
        block._remove_var(name)
    program._sync_with_cpp()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "model"
    paddle.static.save_inference_model(
        str(prefix),
        [block.var(name) for name in feed_names],
        [block.var(boxes_name), block.var(logits_name)],
        executor,
        program=program,
        clip_extra=True,
    )
    output_model = prefix.with_suffix(".pdmodel")
    output_params = prefix.with_suffix(".pdiparams")
    config = paddle.inference.Config(str(output_model), str(output_params))
    config.disable_gpu()
    config.disable_glog_info()
    predictor = paddle.inference.create_predictor(config)
    predictor.get_input_handle("image").copy_from_cpu(
        np.zeros((1, 3, args.target_size, args.target_size), dtype=np.float32)
    )
    predictor.run()
    output_values = [
        np.asarray(predictor.get_output_handle(name).copy_to_cpu())
        for name in predictor.get_output_names()
    ]
    output_shapes = [list(values.shape) for values in output_values]
    expected_shapes = [[1, 300, 4], [1, 300, args.classes]]
    output_stats = []
    for values in output_values:
        finite = np.isfinite(values)
        finite_values = values[finite]
        output_stats.append(
            {
                "shape": list(values.shape),
                "nonfinite": int(values.size - finite_values.size),
                "minimum": float(finite_values.min()) if finite_values.size else None,
                "maximum": float(finite_values.max()) if finite_values.size else None,
            }
        )
    receipt = {
        "schema_version": "legalpdf.ppdoc_raw_legacy_slice.v1",
        "source": {
            "model": str(source_model.resolve()),
            "model_sha256": sha256(source_model),
            "params": str(source_params.resolve()),
            "params_sha256": sha256(source_params),
            "operations": source_operation_count,
        },
        "output": {
            "model": str(output_model.resolve()),
            "model_sha256": sha256(output_model),
            "params": str(output_params.resolve()),
            "params_sha256": sha256(output_params),
            "validated_output_shapes": output_shapes,
            "zero_input_stats": output_stats,
        },
        "feeds": feed_names,
        "fetches": {
            "boxes": {"name": boxes_name, "producer_index": boxes_index},
            "logits": {"name": logits_name, "producer_index": logits_index},
        },
    }
    (args.output_dir / "raw_slice.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    if output_shapes != expected_shapes:
        raise ValueError(f"raw output shapes differ: {output_shapes} != {expected_shapes}")
    if any(stats["nonfinite"] for stats in output_stats):
        raise ValueError(f"raw outputs contain non-finite values: {output_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
