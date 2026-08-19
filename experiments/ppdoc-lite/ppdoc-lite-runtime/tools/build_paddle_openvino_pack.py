from __future__ import annotations

import argparse
import contextlib
import copy
import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from graph import (
    load_inference_config,
    run_prepare_paddle_openvino_pack,
    sha256_file,
    write_json,
)


DECODED_TRANSFORM = "paddle2onnx105_opset16_decoded_openvino_ovc"
RAW_TRANSFORM = "paddle2onnx105_opset16_ppyoloe_raw_openvino_ovc"
TRANSFORM = DECODED_TRANSFORM


def select_onnx_outputs(model: Any, names: Sequence[str]) -> None:
    available = {value.name: value for value in model.graph.output}
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Paddle2ONNX graph is missing decoded outputs: {missing}")
    selected = [copy.deepcopy(available[name]) for name in names]
    del model.graph.output[:]
    model.graph.output.extend(selected)


def run(command: Sequence[str]) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    subprocess.run(command, check=True)


def version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout.strip()


def verify_openvino_model(
    path: Path,
    *,
    output_contract: str,
    boxes_output: str,
    second_output: str,
    target_size: Sequence[int],
    class_count: int,
    output_width: int | None,
) -> dict[str, Any]:
    import openvino as ov

    model = ov.Core().read_model(path)
    outputs = [sorted(port.names) for port in model.outputs]
    expected = (boxes_output, second_output)
    if len(outputs) != 2 or any(name not in names for name, names in zip(expected, outputs)):
        raise ValueError(f"OpenVINO output contract is {outputs}, expected {list(expected)}")
    image = next((port for port in model.inputs if "image" in port.names), None)
    if image is None:
        raise ValueError("OpenVINO graph has no image input")
    shape = [dimension.get_length() if dimension.is_static else None for dimension in image.partial_shape]
    if shape[-2:] != list(target_size):
        raise ValueError(f"OpenVINO image shape is {shape}, expected *x{list(target_size)}")
    boxes_shape = model.output(boxes_output).partial_shape
    if output_contract == "ppyoloe_raw":
        scores_shape = model.output(second_output).partial_shape
        if (
            len(boxes_shape) != 3
            or len(scores_shape) != 3
            or not all(
                dimension.is_static for dimension in list(boxes_shape) + list(scores_shape)
            )
            or boxes_shape[0].get_length() != 1
            or boxes_shape[2].get_length() != 4
            or scores_shape[0].get_length() != 1
            or scores_shape[1].get_length() != class_count
            or boxes_shape[1].get_length() != scores_shape[2].get_length()
        ):
            raise ValueError(
                f"OpenVINO PP-YOLOE output shapes are boxes={boxes_shape}, scores={scores_shape}"
            )
    elif output_width is not None:
        actual_width = boxes_shape[-1]
        if not actual_width.is_static or actual_width.get_length() != output_width:
            raise ValueError(
                f"OpenVINO decoded-box shape is {boxes_shape}, expected width {output_width}"
            )
    return {
        "openvino": ov.__version__,
        "inputs": [{"names": sorted(port.names), "shape": str(port.partial_shape)} for port in model.inputs],
        "outputs": [{"names": names, "shape": str(port.partial_shape)} for names, port in zip(outputs, model.outputs)],
        "ordered_ops": len(model.get_ordered_ops()),
    }


def build(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {args.output_dir}")
    inference = load_inference_config(args.inference_yml)
    transform = RAW_TRANSFORM if args.output_contract == "ppyoloe_raw" else DECODED_TRANSFORM
    second_output = (
        args.scores_output if args.output_contract == "ppyoloe_raw" else args.counts_output
    )
    model_file = args.model_dir / args.model_filename
    params_file = args.model_dir / args.params_filename
    for path in (model_file, params_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.work_dir is not None:
        if args.work_dir.exists() and any(args.work_dir.iterdir()):
            raise ValueError(f"Work directory must be empty: {args.work_dir}")
        args.work_dir.mkdir(parents=True, exist_ok=True)
        work_context = contextlib.nullcontext(args.work_dir)
    else:
        work_context = tempfile.TemporaryDirectory(prefix="ppdoc-openvino-export-")

    with work_context as temporary:
        work = Path(temporary)
        raw_onnx = work / "model.raw.onnx"
        decoded_onnx = work / "model.decoded.onnx"
        converted_xml = work / "model.xml"
        print("phase=paddle2onnx", flush=True)
        run(
            [
                str(args.paddle2onnx),
                "--model_dir",
                str(args.model_dir),
                "--model_filename",
                args.model_filename,
                "--params_filename",
                args.params_filename,
                "--opset_version",
                str(args.opset_version),
                "--save_file",
                str(raw_onnx),
                "--enable_onnx_checker",
                "True",
            ]
        )

        print("phase=select-decoded-outputs", flush=True)
        import onnx

        model = onnx.load(raw_onnx, load_external_data=False)
        select_onnx_outputs(model, (args.boxes_output, second_output))
        onnx.checker.check_model(model)
        onnx.save(model, decoded_onnx)

        print("phase=openvino", flush=True)
        ovc_command = [
            str(args.ovc),
            str(decoded_onnx),
            "--output_model",
            str(converted_xml),
            f"--compress_to_fp16={'True' if args.precision == 'fp16' else 'False'}",
        ]
        if args.ovc_input:
            ovc_command.extend(("--input", args.ovc_input))
        run(ovc_command)
        verification = verify_openvino_model(
            converted_xml,
            output_contract=args.output_contract,
            boxes_output=args.boxes_output,
            second_output=second_output,
            target_size=inference["target_size"],
            class_count=len(inference["labels"]),
            output_width=args.output_width,
        )

        print("phase=package", flush=True)
        run_prepare_paddle_openvino_pack(
            argparse.Namespace(
                xml=converted_xml,
                bin=converted_xml.with_suffix(".bin"),
                inference_yml=args.inference_yml,
                source_manifest=args.source_manifest,
                source_dir=args.source_dir,
                output_dir=args.output_dir,
                variant_id=args.variant_id,
                precision=args.precision,
                inputs=args.inputs,
                boxes_output=args.boxes_output,
                counts_output=args.counts_output,
                scores_output=args.scores_output,
                output_contract=args.output_contract,
                nms_score_threshold=args.nms_score_threshold,
                nms_threshold=args.nms_threshold,
                nms_top_k=args.nms_top_k,
                transform=transform,
                detections_per_image=args.detections_per_image,
                output_width=args.output_width,
            )
        )
        write_json(
            args.output_dir / "export.json",
            {
                "schema_version": "legalpdf.ppdoc_lite_paddle_openvino_export.v1",
                "variant_id": args.variant_id,
                "transform": transform,
                "versions": {
                    "paddle2onnx": version([str(args.paddle2onnx), "--version"]),
                    "onnx": onnx.__version__,
                    "openvino": verification["openvino"],
                },
                "opset_version": args.opset_version,
                "precision": args.precision,
                "source": {
                    "model": str(model_file.resolve()),
                    "model_sha256": sha256_file(model_file),
                    "params": str(params_file.resolve()),
                    "params_sha256": sha256_file(params_file),
                },
                "intermediate": {
                    "raw_onnx_sha256": sha256_file(raw_onnx),
                    "decoded_onnx_sha256": sha256_file(decoded_onnx),
                },
                "openvino": verification,
            },
        )
    print("phase=complete", flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a verified thin OpenVINO pack from a legacy Paddle inference export."
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-filename", default="model.pdmodel")
    parser.add_argument("--params-filename", default="model.pdiparams")
    parser.add_argument("--inference-yml", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--variant-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--paddle2onnx", type=Path, default=Path("paddle2onnx"))
    parser.add_argument("--ovc", type=Path, default=Path("ovc"))
    parser.add_argument("--opset-version", type=int, default=16, choices=(16,))
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--inputs", nargs="+", default=["im_shape", "image", "scale_factor"])
    parser.add_argument("--boxes-output", default="save_infer_model/scale_0.tmp_0")
    parser.add_argument("--counts-output", default="save_infer_model/scale_1.tmp_0")
    parser.add_argument(
        "--output-contract", choices=("decoded_boxes", "ppyoloe_raw"), default="decoded_boxes"
    )
    parser.add_argument("--scores-output", default="save_infer_model/scale_1.tmp_0")
    parser.add_argument("--nms-score-threshold", type=float, default=0.01)
    parser.add_argument("--nms-threshold", type=float, default=0.7)
    parser.add_argument("--nms-top-k", type=int, default=1_000)
    parser.add_argument("--detections-per-image", type=int)
    parser.add_argument("--output-width", type=int)
    parser.add_argument(
        "--ovc-input",
        help="Optional static OpenVINO input declaration, for example image[1,3,640,640],scale_factor[1,2]",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Preserve intermediate ONNX and OpenVINO files in an empty directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return build(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
