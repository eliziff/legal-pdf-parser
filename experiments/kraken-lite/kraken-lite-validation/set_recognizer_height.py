from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import onnx


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_pack", type=Path)
    parser.add_argument("output_pack", type=Path)
    parser.add_argument("height", type=int)
    args = parser.parse_args()

    manifest = json.loads(
        (args.source_pack / "manifest.json").read_text(encoding="utf-8")
    )
    args.output_pack.mkdir(parents=True, exist_ok=True)
    model = onnx.load(args.source_pack / manifest["model"]["file"])
    image_input = next(
        value
        for value in model.graph.input
        if value.name == manifest["model"].get("input", "image")
    )
    height_dimension = image_input.type.tensor_type.shape.dim[2]
    height_dimension.ClearField("dim_value")
    height_dimension.dim_param = "height"
    output_model = args.output_pack / "model.onnx"
    onnx.save(model, output_model)

    codec = manifest.get("codec", {})
    if codec.get("file"):
        shutil.copy2(args.source_pack / codec["file"], args.output_pack / codec["file"])
    manifest["id"] = f"{manifest.get('id', args.source_pack.name)}-h{args.height}"
    manifest["input"]["height"] = args.height
    manifest["model"]["sha256"] = sha256(output_model)
    manifest["model"]["dynamic_height"] = True
    manifest.setdefault("runtime", {})["dynamicBlockBase"] = 4
    temporary = args.output_pack / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_pack / "manifest.json")
    print(
        json.dumps(
            {
                "height": args.height,
                "modelBytes": output_model.stat().st_size,
                "sha256": manifest["model"]["sha256"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
