from __future__ import annotations

import argparse
import json
from pathlib import Path

import paddle


def main() -> int:
    parser = argparse.ArgumentParser(
        description="List raw RT-DETR tensor candidates in a legacy Paddle inference graph."
    )
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("--classes", type=int, default=26)
    parser.add_argument("--tail-ops", type=int, default=0)
    args = parser.parse_args()

    paddle.enable_static()
    executor = paddle.static.Executor(paddle.CPUPlace())
    program, feed_names, fetch_targets = paddle.static.load_inference_model(
        str(args.model_dir),
        executor,
        model_filename="model.pdmodel",
        params_filename="model.pdiparams",
    )
    block = program.global_block()
    producers = {
        name: {"index": index, "type": operation.type}
        for index, operation in enumerate(block.ops)
        for name in operation.output_arg_names
    }
    candidates = []
    for variable in block.vars.values():
        try:
            shape = list(variable.shape)
        except RuntimeError:
            continue
        if shape in ([-1, 300, 4], [-1, 300, args.classes]):
            candidates.append(
                {
                    "name": variable.name,
                    "shape": shape,
                    "persistable": variable.persistable,
                    "producer": producers.get(variable.name),
                }
            )
    print(
        json.dumps(
            {
                "feeds": feed_names,
                "fetches": [target.name for target in fetch_targets],
                "operations": len(block.ops),
                "tail_operations": [
                    {
                        "index": len(block.ops) - min(args.tail_ops, len(block.ops)) + offset,
                        "type": operation.type,
                        "inputs": operation.input_arg_names,
                        "outputs": operation.output_arg_names,
                    }
                    for offset, operation in enumerate(block.ops[-args.tail_ops :])
                ]
                if args.tail_ops
                else [],
                "candidates": candidates,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
