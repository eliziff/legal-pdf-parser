#!/usr/bin/env python3
"""Build a PP-DocLayoutV3 student checkpoint from released compatible weights.

The student keeps PP-DocLayoutV3's transformer, mask, and reading-order task.
Only the released HGNetV2/MaskHybridEncoder size changes. Exact-shape weights
from the corresponding Mask RT-DETR checkpoint initialize the small backbone
and neck; the legal PP-DocLayoutV3 teacher then takes precedence for every
compatible task-specific tensor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_numel(value: Any) -> int:
    total = 1
    for dimension in value.shape:
        total *= int(dimension)
    return total


def unwrap_state(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is {type(payload).__name__}, expected a state dictionary")
    for key in ("state_dict", "model"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return payload


def exact_matches(
    target: dict[str, Any],
    source: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    copied: dict[str, Any] = {}
    mismatched: list[str] = []
    for name, value in source.items():
        if name not in target:
            continue
        if tuple(target[name].shape) == tuple(value.shape):
            copied[name] = value
        else:
            mismatched.append(name)
    return copied, sorted(mismatched)


def component_counts(names: list[str], state: dict[str, Any]) -> dict[str, dict[str, int]]:
    tensors: Counter[str] = Counter()
    elements: Counter[str] = Counter()
    for name in names:
        component = name.split(".", 1)[0]
        tensors[component] += 1
        elements[component] += tensor_numel(state[name])
    return {
        component: {"tensors": tensors[component], "elements": elements[component]}
        for component in sorted(tensors)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paddledetection-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--student-pretrain", type=Path, required=True)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument(
        "--allow-random-parameter",
        action="append",
        default=[],
        help="Exact trainable tensor allowed to retain its initializer; repeat as needed.",
    )
    args = parser.parse_args()

    root = args.paddledetection_root.resolve()
    for path in (root, args.config, args.student_pretrain, args.teacher):
        if not path.exists():
            raise FileNotFoundError(path)
    sys.path.insert(0, str(root))
    os.chdir(root)

    import paddle
    from ppdet.core.workspace import create, load_config

    paddle.set_device("cpu")
    config = load_config(str(args.config.resolve()))
    model = create(config.architecture)
    initial = model.state_dict()
    student_source = unwrap_state(paddle.load(str(args.student_pretrain.resolve())))
    teacher_source = unwrap_state(paddle.load(str(args.teacher.resolve())))

    student_matches, student_shape_mismatches = exact_matches(initial, student_source)
    teacher_matches, teacher_shape_mismatches = exact_matches(initial, teacher_source)

    final = dict(initial)
    provenance: dict[str, str] = {}
    for name, value in student_matches.items():
        final[name] = value
        provenance[name] = "student_pretrain"
    for name, value in teacher_matches.items():
        final[name] = value
        provenance[name] = "legal_teacher"

    parameter_names = {name for name, _ in model.named_parameters()}
    uncovered_parameters = sorted(parameter_names - provenance.keys())
    allowed_random = sorted(set(args.allow_random_parameter))
    unexpected_random = sorted(set(uncovered_parameters) - set(allowed_random))
    stale_allowance = sorted(set(allowed_random) - set(uncovered_parameters))
    if unexpected_random or stale_allowance:
        raise RuntimeError(
            "Checkpoint initialization coverage differs from the exact allowance; "
            f"unexpected={unexpected_random[:20]}, stale_allowance={stale_allowance[:20]}"
        )

    model.set_state_dict(final)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    paddle.save(model.state_dict(), str(temporary))
    os.replace(temporary, args.output)

    final_state = model.state_dict()
    teacher_names = sorted(name for name, source in provenance.items() if source == "legal_teacher")
    student_names = sorted(name for name, source in provenance.items() if source == "student_pretrain")
    all_elements = sum(tensor_numel(value) for value in final_state.values())
    trainable_elements = sum(tensor_numel(value) for _, value in model.named_parameters())
    teacher_elements = sum(tensor_numel(final_state[name]) for name in teacher_names)
    student_elements = sum(tensor_numel(final_state[name]) for name in student_names)
    uncovered_elements = sum(tensor_numel(final_state[name]) for name in uncovered_parameters)
    receipt = {
        "schema_version": "legalpdf.ppdocv3_student_init.v1",
        "architecture": {
            "config": str(args.config.resolve()),
            "config_sha256": sha256(args.config),
            "state_tensors": len(final_state),
            "state_elements": all_elements,
            "trainable_elements": trainable_elements,
            "estimated_fp32_parameter_bytes": trainable_elements * 4,
        },
        "student_pretrain": {
            "path": str(args.student_pretrain.resolve()),
            "sha256": sha256(args.student_pretrain),
            "selected_tensors": len(student_names),
            "selected_elements": student_elements,
            "shape_mismatch_count": len(student_shape_mismatches),
            "shape_mismatches": student_shape_mismatches,
            "by_component": component_counts(student_names, final_state),
        },
        "legal_teacher": {
            "path": str(args.teacher.resolve()),
            "sha256": sha256(args.teacher),
            "selected_tensors": len(teacher_names),
            "selected_elements": teacher_elements,
            "shape_mismatch_count": len(teacher_shape_mismatches),
            "shape_mismatches": teacher_shape_mismatches,
            "by_component": component_counts(teacher_names, final_state),
        },
        "coverage": {
            "initialized_parameter_fraction": (
                1.0 - uncovered_elements / trainable_elements if trainable_elements else 1.0
            ),
            "uncovered_parameter_count": len(uncovered_parameters),
            "uncovered_parameter_elements": uncovered_elements,
            "uncovered_parameters": uncovered_parameters,
            "explicit_random_allowance": allowed_random,
        },
        "precedence": ["student_pretrain", "legal_teacher"],
        "output": {
            "path": str(args.output.resolve()),
            "sha256": sha256(args.output),
            "bytes": args.output.stat().st_size,
        },
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_tmp = args.receipt.with_suffix(args.receipt.suffix + ".part")
    receipt_tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt_tmp.replace(args.receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
