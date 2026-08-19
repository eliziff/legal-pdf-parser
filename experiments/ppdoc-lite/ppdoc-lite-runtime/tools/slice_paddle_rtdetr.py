from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Sequence


def _items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _shape(value: dict[str, Any]) -> list[int] | None:
    descriptor = value.get("TT", {}).get("D", [])
    if len(descriptor) > 1 and isinstance(descriptor[1], list):
        return descriptor[1]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_outputs(ops: list[dict[str, Any]], classes: int) -> tuple[int, dict[str, Any], int, dict[str, Any]]:
    logits = [
        (index, output)
        for index, op in enumerate(ops)
        if op.get("#") == "1.add"
        for output in _items(op.get("O"))
        if _shape(output) == [-1, 300, classes]
    ]
    if not logits:
        raise ValueError(f"Could not find final [-1, 300, {classes}] class logits")
    logits_index, logits_output = logits[-1]

    boxes = [
        (index, output)
        for index, op in enumerate(ops[:logits_index])
        if op.get("#") == "1.sigmoid"
        for output in _items(op.get("O"))
        if _shape(output) == [-1, 300, 4]
    ]
    if not boxes:
        raise ValueError("Could not find final normalized [-1, 300, 4] boxes")
    boxes_index, boxes_output = boxes[-1]
    return boxes_index, boxes_output, logits_index, logits_output


def slice_model(source: Path, destination: Path, classes: int) -> dict[str, Any]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    try:
        ops = payload["program"]["regions"][0]["blocks"][0]["ops"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError(f"{source} is not a supported Paddle PIR inference program") from error
    if not isinstance(ops, list):
        raise ValueError(f"{source} has no top-level Paddle PIR operation list")

    fetches = [op for op in ops if op.get("#") == "1.fetch"]
    if len(fetches) < 2:
        raise ValueError(f"{source} must contain at least two fetch operations")
    boxes_index, boxes, logits_index, logits = _raw_outputs(ops, classes)

    value_ids = [
        int(output["%"])
        for op in ops
        for output in _items(op.get("O"))
        if isinstance(output.get("%"), int)
    ]
    next_id = max(value_ids) + 1
    raw_values = (boxes, logits)
    new_fetches: list[dict[str, Any]] = []
    for column, (template, value) in enumerate(zip(fetches[:2], raw_values, strict=True)):
        fetch = copy.deepcopy(template)
        fetch["I"] = [{"%": value["%"]}]
        output = _items(fetch.get("O"))[0]
        output["%"] = next_id + column
        output["TT"] = copy.deepcopy(value["TT"])
        for attribute in fetch.get("A", []):
            if attribute.get("N") == "name":
                attribute["AT"]["D"] = f"fetch_name_{column}"
            elif attribute.get("N") == "col":
                attribute["AT"]["D"] = column
        new_fetches.append(fetch)

    retained = ops[: logits_index + 1]
    payload["program"]["regions"][0]["blocks"][0]["ops"] = retained + new_fetches
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return {
        "schema_version": "legalpdf.ppdoc_raw_pir_slice.v1",
        "source_model": str(source.resolve()),
        "source_sha256": _sha256(source),
        "output_model": str(destination.resolve()),
        "output_sha256": _sha256(destination),
        "source_operations": len(ops),
        "output_operations": len(retained) + len(new_fetches),
        "removed_operations": len(ops) - len(retained),
        "outputs": {
            "boxes": {"value_id": boxes["%"], "shape": _shape(boxes), "producer_index": boxes_index},
            "logits": {"value_id": logits["%"], "shape": _shape(logits), "producer_index": logits_index},
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Slice a PP-DocLayoutV3 Paddle PIR model at raw RT-DETR boxes and logits."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, default=26)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    receipt = slice_model(args.model, args.output_dir / "model.json", args.classes)
    params_target = args.output_dir / "model.pdiparams"
    if args.params.resolve() != params_target.resolve():
        shutil.copy2(args.params, params_target)
    receipt["params"] = {
        "file": str(params_target.resolve()),
        "sha256": _sha256(params_target),
        "bytes": params_target.stat().st_size,
    }
    receipt_path = args.output_dir / "raw_slice.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
