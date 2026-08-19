from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


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
    parser.add_argument("--ops", default="LSTM")
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    parser.add_argument("--per-channel", action="store_true")
    parser.add_argument("--reduce-range", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(
        (args.source_pack / "manifest.json").read_text(encoding="utf-8")
    )
    args.output_pack.mkdir(parents=True, exist_ok=True)
    source_model = args.source_pack / manifest["model"]["file"]
    output_model = args.output_pack / "model.onnx"
    ops = [value.strip() for value in args.ops.split(",") if value.strip()]
    weight_type = QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8
    print(
        f"quantize ops={ops} weight={args.weight_type} "
        f"per_channel={args.per_channel} reduce_range={args.reduce_range}",
        flush=True,
    )
    quantize_dynamic(
        source_model,
        output_model,
        op_types_to_quantize=ops,
        weight_type=weight_type,
        per_channel=args.per_channel,
        reduce_range=args.reduce_range,
    )
    codec = manifest.get("codec", {})
    if codec.get("file"):
        shutil.copy2(args.source_pack / codec["file"], args.output_pack / codec["file"])

    suffix = f"dynamic-{args.weight_type}-{'-'.join(value.lower() for value in ops)}"
    if args.per_channel:
        suffix += "-per-channel"
    if args.reduce_range:
        suffix += "-reduce-range"
    manifest["id"] = f"{manifest.get('id', args.source_pack.name)}-{suffix}"
    manifest["model"]["sha256"] = sha256(output_model)
    manifest["model"]["quantization"] = {
        "method": "dynamic",
        "weightType": args.weight_type,
        "ops": ops,
        "perChannel": args.per_channel,
        "reduceRange": args.reduce_range,
    }
    manifest["verification"] = {
        "requested": False,
        "onnxRuntimeParity": False,
        "shapes": [],
    }
    temporary = args.output_pack / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_pack / "manifest.json")
    print(
        json.dumps(
            {
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
