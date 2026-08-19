from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from kraken_lite.blla import prepare_blla_page


DEVELOPMENT_PAGES = {
    "wrongful-opening",
    "ubc-interior",
    "comparative-opening",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PageReader(CalibrationDataReader):
    def __init__(self, paths: list[Path], input_name: str) -> None:
        self.paths = iter(paths)
        self.input_name = input_name

    def get_next(self) -> dict[str, np.ndarray] | None:
        try:
            path = next(self.paths)
        except StopIteration:
            return None
        print(f"calibrate {path.stem}", flush=True)
        with np.load(path) as values:
            return {self.input_name: values["image"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    parser.add_argument("source_pack", type=Path)
    parser.add_argument("output_pack", type=Path)
    parser.add_argument("--height", type=int, default=1800)
    args = parser.parse_args()

    source_manifest = json.loads(
        (args.source_pack / "manifest.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(json.dumps(source_manifest))
    manifest["input"]["height"] = args.height
    input_name = str(manifest["model"].get("input", "image"))

    cache = args.output_pack / "calibration"
    cache.mkdir(parents=True, exist_ok=True)
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    calibration_paths: list[Path] = []
    selected = [item for item in corpus if item["id"] in DEVELOPMENT_PAGES]
    for index, item in enumerate(selected, 1):
        target = cache / f"{item['id']}.npz"
        if target.is_file():
            state = "cached"
        else:
            with Image.open(item["image"]) as source:
                tensor = prepare_blla_page(source.convert("RGB"), manifest).tensor
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, image=tensor)
            os.replace(temporary, target)
            state = "created"
        calibration_paths.append(target)
        print(f"[{index}/{len(selected)}] {item['id']} tensor={state}", flush=True)

    args.output_pack.mkdir(parents=True, exist_ok=True)
    source_model = args.source_pack / source_manifest["model"]["file"]
    preprocessed = args.output_pack / "model.preprocessed.onnx"
    output_model = args.output_pack / "model.onnx"
    print("preprocess graph", flush=True)
    quant_pre_process(
        source_model,
        preprocessed,
        skip_symbolic_shape=True,
        skip_optimization=False,
        skip_onnx_shape=False,
    )
    print("quantize S8S8 QDQ MinMax", flush=True)
    quantize_static(
        preprocessed,
        output_model,
        PageReader(calibration_paths, input_name),
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    preprocessed.unlink(missing_ok=True)

    manifest["id"] = f"kraken-stock-blla-int8-static-h{args.height}"
    manifest["model"]["sha256"] = sha256(output_model)
    manifest["model"]["quantization"] = {
        "method": "static",
        "format": "QDQ",
        "activationType": "QInt8",
        "weightType": "QInt8",
        "calibration": "MinMax",
        "pages": [item["id"] for item in selected],
    }
    manifest["verification"] = {
        "requested": False,
        "onnxRuntimeParity": False,
        "shapes": [],
    }
    temporary_manifest = args.output_pack / "manifest.json.tmp"
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary_manifest, args.output_pack / "manifest.json")
    print(
        json.dumps(
            {
                "model": str(output_model),
                "bytes": output_model.stat().st_size,
                "sha256": manifest["model"]["sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
