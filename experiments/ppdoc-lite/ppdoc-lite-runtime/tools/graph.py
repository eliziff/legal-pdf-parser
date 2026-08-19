from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from ppdoc_lite.runtime import PACK_FORMAT, sha256_file


PPDOC_RECT_OUTPUTS = {
    "contract": "ppdoc_rect_parts",
    "classes": "auto.cast.702",
    "scores": "auto.cast.703",
    "coordinates": "auto.cast.704",
}


def load_source_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = ("source_id", "model_name", "source_files", "labels")
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"Source manifest is missing: {missing}")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def verify_source(path: Path, source: dict[str, Any]) -> None:
    for name, expected in source["source_files"].items():
        item = path / name
        if not item.is_file():
            raise FileNotFoundError(item)
        actual = sha256_file(item)
        if actual != expected:
            raise ValueError(f"Source hash mismatch for {item}: expected {expected}, got {actual}")


def load_inference_config(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    operations = {
        str(row.get("type")): row
        for row in payload.get("Preprocess", [])
        if isinstance(row, dict) and row.get("type")
    }
    resize = operations.get("Resize")
    normalize = operations.get("NormalizeImage")
    if not resize or not normalize or "Permute" not in operations:
        raise ValueError(f"{path} lacks the standard Resize/NormalizeImage/Permute contract")
    if bool(resize.get("keep_ratio")):
        raise ValueError("The thin runtime currently requires keep_ratio=false")
    target_size = [int(value) for value in resize.get("target_size", [])]
    mean = [float(value) for value in normalize.get("mean", [])]
    std = [float(value) for value in normalize.get("std", [])]
    labels = [str(value) for value in payload.get("label_list", [])]
    if len(target_size) != 2 or len(mean) != 3 or len(std) != 3 or not labels:
        raise ValueError(f"{path} has incomplete target size, normalization, or labels")
    interpolation = {0: "nearest", 1: "linear", 2: "cubic"}.get(int(resize.get("interp", 2)))
    if interpolation is None:
        raise ValueError(f"Unsupported Paddle resize interpolation: {resize.get('interp')}")
    return {
        "target_size": target_size,
        "color": "RGB",
        "resize": interpolation,
        "scale": 1.0 / 255.0 if bool(normalize.get("is_scale", True)) else 1.0,
        "mean": mean,
        "std": std,
        "layout": "NCHW",
        "labels": labels,
        "source": path.name,
        "sha256": sha256_file(path),
        "detections_per_image": int((payload.get("NMS") or {}).get("keep_top_k", 300)),
        "model_nms": payload.get("NMS") or {},
        "draw_threshold": float(payload.get("draw_threshold", 0.10)),
    }


def _tensor_bytes(tensor: Any) -> int:
    if tensor.raw_data:
        return len(tensor.raw_data)
    widths = {
        1: 4,  # float32
        2: 1,  # uint8
        3: 1,  # int8
        4: 2,  # uint16
        5: 2,  # int16
        6: 4,  # int32
        7: 8,  # int64
        9: 1,  # bool
        10: 2,  # float16
        11: 8,  # float64
        12: 4,  # uint32
        13: 8,  # uint64
        16: 2,  # bfloat16
    }
    return math.prod(tensor.dims) * widths.get(int(tensor.data_type), 0)


def graph_audit(path: Path) -> dict[str, Any]:
    import onnx

    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    producer = {value: node for node in graph.node for value in node.output}
    initializers = {value.name: value for value in graph.initializer}

    def ancestors(outputs: Sequence[str]) -> tuple[set[int], set[str]]:
        nodes: set[int] = set()
        values = set(outputs)
        pending = list(outputs)
        while pending:
            value = pending.pop()
            node = producer.get(value)
            if node is None or id(node) in nodes:
                continue
            nodes.add(id(node))
            for item in node.input:
                if item and item not in values:
                    values.add(item)
                    pending.append(item)
        return nodes, values

    def shape(value: Any) -> list[int | str | None]:
        return [dimension.dim_value or dimension.dim_param or None for dimension in value.type.tensor_type.shape.dim]

    output_rows: dict[str, Any] = {}
    for output in graph.output:
        nodes, values = ancestors([output.name])
        used = set(initializers) & values
        output_rows[output.name] = {
            "shape": shape(output),
            "node_count": len(nodes),
            "initializer_count": len(used),
            "initializer_bytes": sum(_tensor_bytes(initializers[name]) for name in used),
            "operator_counts": dict(sorted(Counter(node.op_type for node in graph.node if id(node) in nodes).items())),
        }

    rect_nodes: set[int] = set()
    rect_values: set[str] = set()
    if all(name in producer for name in PPDOC_RECT_OUTPUTS.values() if name != "ppdoc_rect_parts"):
        rect_nodes, rect_values = ancestors(
            [PPDOC_RECT_OUTPUTS[key] for key in ("classes", "scores", "coordinates")]
        )
    rect_initializers = set(initializers) & rect_values
    return {
        "schema_version": "legalpdf.ppdoc_lite_graph_audit.v1",
        "model": str(path.resolve()),
        "sha256": sha256_file(path),
        "file_bytes": path.stat().st_size,
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": {value.name: shape(value) for value in graph.input},
        "outputs": output_rows,
        "graph": {
            "node_count": len(graph.node),
            "initializer_count": len(graph.initializer),
            "initializer_bytes": sum(_tensor_bytes(value) for value in graph.initializer),
            "operator_counts": dict(sorted(Counter(node.op_type for node in graph.node).items())),
        },
        "ppdoc_rect_path": {
            "available": bool(rect_nodes),
            "outputs": PPDOC_RECT_OUTPUTS,
            "node_count": len(rect_nodes),
            "initializer_count": len(rect_initializers),
            "initializer_bytes": sum(_tensor_bytes(initializers[name]) for name in rect_initializers),
            "removed_node_count": len(graph.node) - len(rect_nodes) if rect_nodes else None,
            "removed_initializer_bytes": (
                sum(_tensor_bytes(value) for value in graph.initializer)
                - sum(_tensor_bytes(initializers[name]) for name in rect_initializers)
                if rect_nodes
                else None
            ),
        },
    }


def run_audit(args: argparse.Namespace) -> int:
    payload = graph_audit(args.model)
    if args.output:
        write_json(args.output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def run_extract_rect(args: argparse.Namespace) -> int:
    import onnx
    from onnx.utils import extract_model

    model = onnx.load(args.model, load_external_data=False)
    input_names = [value.name for value in model.graph.input]
    available = {value for node in model.graph.node for value in node.output}
    output_names = [PPDOC_RECT_OUTPUTS[key] for key in ("classes", "scores", "coordinates")]
    missing = [value for value in output_names if value not in available]
    if missing:
        raise ValueError(f"The graph is not the pinned PP-DocLayoutV3 export; missing: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extract_model(args.model, args.output, input_names, output_names, check_model=True)
    payload = graph_audit(args.output)
    if args.audit_output:
        write_json(args.audit_output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def run_extract_decoded(args: argparse.Namespace) -> int:
    from onnx.utils import extract_model

    audit = graph_audit(args.model)
    input_names = list(audit["inputs"])
    output_names = [args.boxes_output, args.counts_output]
    missing = [name for name in output_names if name not in audit["outputs"]]
    if missing:
        raise ValueError(f"The decoded output graph is missing: {missing}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    extract_model(args.model, args.output, input_names, output_names, check_model=True)
    payload = graph_audit(args.output)
    if args.audit_output:
        write_json(args.audit_output, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def manifest_for(
    model: Path,
    *,
    variant_id: str,
    source: dict[str, Any],
    outputs: dict[str, str],
    inference: dict[str, Any],
    inputs: list[str],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": PACK_FORMAT,
        "variant_id": variant_id,
        "source": {
            "source_id": source["source_id"],
            "model_name": source["model_name"],
            "source_files": source["source_files"],
            "dataset": source.get("dataset", {}),
        },
        "model": {
            "file": model.name,
            "sha256": sha256_file(model),
            "inputs": inputs,
            "outputs": outputs,
            "detections_per_image": inference["detections_per_image"],
        },
        "input": {
            key: value
            for key, value in inference.items()
            if key
            not in {"labels", "detections_per_image", "model_nms", "draw_threshold"}
        },
        "labels": inference["labels"],
        "postprocess": {
            "geometry": "rect",
            "default_threshold": inference["draw_threshold"],
            "default_layout_nms": False,
            "reference": "PaddleX 3.6.1",
            "model_nms": inference["model_nms"],
        },
        "provenance": provenance or {"precision": "fp32", "transform": "none"},
    }


def run_prepare_pack(args: argparse.Namespace) -> int:
    source = load_source_manifest(args.source_manifest)
    verify_source(args.source_dir, source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.output_dir / "model.onnx"
    if target.resolve() != args.onnx.resolve():
        shutil.copy2(args.onnx, target)
    inference = load_inference_config(args.inference_yml)
    audit = graph_audit(target)
    inputs = list(audit["inputs"])
    if args.output_contract == "decoded_boxes":
        outputs = {
            "contract": "decoded_boxes",
            "boxes": args.boxes_output,
            "counts": args.counts_output,
        }
    elif args.output_contract == "rtdetr_raw":
        outputs = {
            "contract": "rtdetr_raw",
            "boxes": args.boxes_output,
            "logits": args.logits_output,
        }
    elif args.output_contract == "ppyoloe_raw":
        outputs = {
            "contract": "ppyoloe_raw",
            "boxes": args.boxes_output,
            "scores": args.scores_output,
        }
    else:
        outputs = dict(PPDOC_RECT_OUTPUTS)
    missing = [
        name
        for key, name in outputs.items()
        if key != "contract" and name not in audit["outputs"]
    ]
    if missing:
        raise ValueError(f"The ONNX graph is missing declared outputs: {missing}")
    manifest = manifest_for(
        target,
        variant_id=args.variant_id,
        source=source,
        outputs=outputs,
        inference=inference,
        inputs=inputs,
        provenance={"precision": "fp32", "transform": args.transform},
    )
    if args.output_contract == "ppyoloe_raw":
        manifest["postprocess"]["model_nms"] = {
            "score_threshold": args.nms_score_threshold,
            "nms_threshold": args.nms_threshold,
            "nms_top_k": args.nms_top_k,
            "keep_top_k": inference["detections_per_image"],
        }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "graph_audit.json", audit)
    print(json.dumps({"variant_id": args.variant_id, "pack": str(args.output_dir), "model_sha256": sha256_file(target)}, indent=2))
    return 0


def run_prepare_openvino_pack(args: argparse.Namespace) -> int:
    manifest = json.loads(args.source_pack.read_text(encoding="utf-8"))
    if manifest.get("format") != PACK_FORMAT:
        raise ValueError(f"Unsupported source pack format: {manifest.get('format')!r}")
    source_model = manifest.get("model") or {}
    source_file = args.source_pack.parent / str(source_model.get("file", ""))
    if not source_file.is_file():
        raise FileNotFoundError(source_file)
    source_sha256 = sha256_file(source_file)
    if source_sha256 != source_model.get("sha256"):
        raise ValueError(
            f"Source model hash mismatch for {source_file}: "
            f"expected {source_model.get('sha256')}, got {source_sha256}"
        )
    if not args.xml.is_file() or not args.bin.is_file():
        raise FileNotFoundError(f"OpenVINO IR is incomplete: {args.xml}, {args.bin}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_xml = args.output_dir / "model.xml"
    target_bin = args.output_dir / "model.bin"
    if target_xml.resolve() != args.xml.resolve():
        shutil.copy2(args.xml, target_xml)
    if target_bin.resolve() != args.bin.resolve():
        shutil.copy2(args.bin, target_bin)

    source_variant_id = manifest.get("variant_id")
    manifest["variant_id"] = args.variant_id
    manifest["model"] = {
        **source_model,
        "backend": "openvino",
        "file": target_xml.name,
        "bytes": target_xml.stat().st_size + target_bin.stat().st_size,
        "sha256": sha256_file(target_xml),
        "files": [{"file": target_bin.name, "sha256": sha256_file(target_bin)}],
    }
    provenance = dict(manifest.get("provenance") or {})
    provenance.update(
        {
            "precision": args.precision,
            "transform": "openvino_ovc",
            "derived_from": {
                "variant_id": source_variant_id,
                "model_sha256": source_sha256,
            },
        }
    )
    manifest["provenance"] = provenance
    receipt = {
        "schema_version": "legalpdf.ppdoc_lite_openvino_conversion.v1",
        "variant_id": args.variant_id,
        "precision": args.precision,
        "source_pack": str(args.source_pack.resolve()),
        "source_model_sha256": source_sha256,
        "model_xml_sha256": manifest["model"]["sha256"],
        "model_bin_sha256": manifest["model"]["files"][0]["sha256"],
        "model_bytes": manifest["model"]["bytes"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "conversion.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def run_prepare_paddle_openvino_pack(args: argparse.Namespace) -> int:
    source = load_source_manifest(args.source_manifest)
    verify_source(args.source_dir, source)
    inference = load_inference_config(args.inference_yml)
    detections_per_image = getattr(args, "detections_per_image", None)
    if detections_per_image is not None:
        if detections_per_image <= 0:
            raise ValueError("detections_per_image must be positive")
        inference["detections_per_image"] = detections_per_image
    output_width = getattr(args, "output_width", None)
    if output_width is not None and output_width < 6:
        raise ValueError("output_width must be at least 6")
    if inference["labels"] != source["labels"]:
        raise ValueError("Exported labels do not match the hash-locked source ontology")
    if not args.xml.is_file() or not args.bin.is_file():
        raise FileNotFoundError(f"OpenVINO IR is incomplete: {args.xml}, {args.bin}")

    output_contract = getattr(args, "output_contract", "decoded_boxes")
    if output_contract == "decoded_boxes":
        outputs = {
            "contract": "decoded_boxes",
            "boxes": args.boxes_output,
            "counts": args.counts_output,
        }
    elif output_contract == "ppyoloe_raw":
        outputs = {
            "contract": "ppyoloe_raw",
            "boxes": args.boxes_output,
            "scores": args.scores_output,
        }
    else:
        raise ValueError(f"Unsupported direct Paddle/OpenVINO contract: {output_contract}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_xml = args.output_dir / "model.xml"
    target_bin = args.output_dir / "model.bin"
    if target_xml.resolve() != args.xml.resolve():
        shutil.copy2(args.xml, target_xml)
    if target_bin.resolve() != args.bin.resolve():
        shutil.copy2(args.bin, target_bin)
    manifest = manifest_for(
        target_xml,
        variant_id=args.variant_id,
        source=source,
        outputs=outputs,
        inference=inference,
        inputs=args.inputs,
        provenance={
            "precision": args.precision,
            "transform": getattr(args, "transform", "paddle_legacy26_openvino_ovc"),
        },
    )
    manifest["model"].update(
        {
            "backend": "openvino",
            "bytes": target_xml.stat().st_size + target_bin.stat().st_size,
            "files": [{"file": target_bin.name, "sha256": sha256_file(target_bin)}],
        }
    )
    if output_width is not None:
        manifest["model"]["output_width"] = output_width
    if output_contract == "ppyoloe_raw":
        manifest["postprocess"]["model_nms"] = {
            "score_threshold": args.nms_score_threshold,
            "nms_threshold": args.nms_threshold,
            "nms_top_k": args.nms_top_k,
            "keep_top_k": inference["detections_per_image"],
        }
    receipt = {
        "schema_version": "legalpdf.ppdoc_lite_openvino_conversion.v1",
        "variant_id": args.variant_id,
        "precision": args.precision,
        "source_id": source["source_id"],
        "model_xml_sha256": manifest["model"]["sha256"],
        "model_bin_sha256": manifest["model"]["files"][0]["sha256"],
        "model_bytes": manifest["model"]["bytes"],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    write_json(args.output_dir / "conversion.json", receipt)
    print(json.dumps(receipt, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ppdoc-lite-graph", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-graph", help="Record operators and output dependency paths")
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=run_audit)

    extract = subparsers.add_parser("extract-ppdoc-rect", help="Remove mask and learned-order-only tails")
    extract.add_argument("--model", type=Path, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--audit-output", type=Path)
    extract.set_defaults(handler=run_extract_rect)

    decoded = subparsers.add_parser("extract-decoded", help="Remove outputs not needed for decoded boxes")
    decoded.add_argument("--model", type=Path, required=True)
    decoded.add_argument("--output", type=Path, required=True)
    decoded.add_argument("--boxes-output", default="fetch_name_0")
    decoded.add_argument("--counts-output", default="fetch_name_1")
    decoded.add_argument("--audit-output", type=Path)
    decoded.set_defaults(handler=run_extract_decoded)

    pack = subparsers.add_parser("prepare-pack", help="Hash-lock an ONNX graph into a runtime model pack")
    pack.add_argument("--onnx", type=Path, required=True)
    pack.add_argument("--source-manifest", type=Path, required=True)
    pack.add_argument("--source-dir", type=Path, required=True)
    pack.add_argument("--inference-yml", type=Path, required=True)
    pack.add_argument("--output-dir", type=Path, required=True)
    pack.add_argument("--variant-id", required=True)
    pack.add_argument(
        "--output-contract",
        choices=("decoded_boxes", "rtdetr_raw", "ppyoloe_raw", "ppdoc_rect_parts"),
        default="decoded_boxes",
    )
    pack.add_argument("--boxes-output", default="fetch_name_0")
    pack.add_argument("--counts-output", default="fetch_name_1")
    pack.add_argument("--logits-output", default="fetch_name_1")
    pack.add_argument("--scores-output", default="fetch_name_1")
    pack.add_argument("--nms-score-threshold", type=float, default=0.01)
    pack.add_argument("--nms-threshold", type=float, default=0.7)
    pack.add_argument("--nms-top-k", type=int, default=1_000)
    pack.add_argument("--transform", default="none")
    pack.set_defaults(handler=run_prepare_pack)

    openvino_pack = subparsers.add_parser(
        "prepare-openvino-pack",
        help="Hash-lock an OpenVINO IR converted from an existing runtime pack",
    )
    openvino_pack.add_argument("--xml", type=Path, required=True)
    openvino_pack.add_argument("--bin", type=Path, required=True)
    openvino_pack.add_argument("--source-pack", type=Path, required=True)
    openvino_pack.add_argument("--output-dir", type=Path, required=True)
    openvino_pack.add_argument("--variant-id", required=True)
    openvino_pack.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    openvino_pack.set_defaults(handler=run_prepare_openvino_pack)

    paddle_openvino_pack = subparsers.add_parser(
        "prepare-paddle-openvino-pack",
        help="Hash-lock an OpenVINO IR converted directly from a Paddle export",
    )
    paddle_openvino_pack.add_argument("--xml", type=Path, required=True)
    paddle_openvino_pack.add_argument("--bin", type=Path, required=True)
    paddle_openvino_pack.add_argument("--inference-yml", type=Path, required=True)
    paddle_openvino_pack.add_argument("--source-manifest", type=Path, required=True)
    paddle_openvino_pack.add_argument("--source-dir", type=Path, required=True)
    paddle_openvino_pack.add_argument("--output-dir", type=Path, required=True)
    paddle_openvino_pack.add_argument("--variant-id", required=True)
    paddle_openvino_pack.add_argument(
        "--precision", choices=("fp32", "fp16", "int8"), required=True
    )
    paddle_openvino_pack.add_argument(
        "--inputs", nargs="+", default=["im_shape", "image", "scale_factor"]
    )
    paddle_openvino_pack.add_argument(
        "--boxes-output", default="save_infer_model/scale_0.tmp_0"
    )
    paddle_openvino_pack.add_argument(
        "--counts-output", default="save_infer_model/scale_1.tmp_0"
    )
    paddle_openvino_pack.add_argument(
        "--output-contract", choices=("decoded_boxes", "ppyoloe_raw"), default="decoded_boxes"
    )
    paddle_openvino_pack.add_argument(
        "--scores-output", default="save_infer_model/scale_1.tmp_0"
    )
    paddle_openvino_pack.add_argument("--nms-score-threshold", type=float, default=0.01)
    paddle_openvino_pack.add_argument("--nms-threshold", type=float, default=0.7)
    paddle_openvino_pack.add_argument("--nms-top-k", type=int, default=1_000)
    paddle_openvino_pack.add_argument(
        "--transform", default="paddle_legacy26_openvino_ovc"
    )
    paddle_openvino_pack.add_argument("--detections-per-image", type=int)
    paddle_openvino_pack.add_argument("--output-width", type=int)
    paddle_openvino_pack.set_defaults(handler=run_prepare_paddle_openvino_pack)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
