from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = (
    "PicoDet-XS",
    "PicoDet-S_layout_17cls",
    "PicoDet-L_layout_17cls",
    "PP-DocLayoutV3",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _annotation_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    categories = sorted(payload.get("categories", []), key=lambda row: int(row["id"]))
    return {
        "path": str(path),
        "sha256": _sha256(path),
        "images": len(payload.get("images", [])),
        "annotations": len(payload.get("annotations", [])),
        "categories": [
            {"id": int(row["id"]), "name": str(row["name"])} for row in categories
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import paddle
    import paddlex
    from paddlex.repo_apis.base.register import get_registered_model_info

    annotations = args.dataset / "annotations"
    splits = {
        name: _annotation_summary(annotations / f"instance_{name}.json")
        for name in ("train", "val", "benchmark")
    }
    category_lists = [split["categories"] for split in splits.values()]
    if any(categories != category_lists[0] for categories in category_lists[1:]):
        raise ValueError("Dataset category IDs/names differ across retained splits")

    models: dict[str, Any] = {}
    for name in MODELS:
        try:
            info = dict(get_registered_model_info(name))
            config_path = Path(str(info.get("config_path", "")))
            info["config_exists"] = config_path.is_file()
            info["config_sha256"] = _sha256(config_path) if config_path.is_file() else None
            models[name] = info
        except Exception as exc:  # diagnostic receipt must preserve every candidate
            models[name] = {"error": f"{type(exc).__name__}: {exc}"}

    report = {
        "schema_version": "legalpdf.ppdoc_lite_training_environment.v1",
        "paddle": paddle.__version__,
        "paddlex": paddlex.__version__,
        "paddle_device": paddle.get_device(),
        "cuda_devices": paddle.device.cuda.device_count(),
        "paddlex_root": str(Path(paddlex.__file__).resolve().parent),
        "models": models,
        "dataset": {
            "root": str(args.dataset.resolve()),
            "image_files": len([path for path in (args.dataset / "images").iterdir() if path.is_file()]),
            "splits": splits,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
