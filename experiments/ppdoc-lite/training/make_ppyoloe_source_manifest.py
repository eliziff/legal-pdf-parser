#!/usr/bin/env python3
"""Hash-lock a completed legal25 PP-YOLOE checkpoint for runtime export."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    run_root: Path,
    checkpoint: Path,
    source_id: str,
    model_name: str,
) -> dict[str, object]:
    run_root = run_root.resolve()
    checkpoint = checkpoint.resolve()
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("model") not in {"PP-YOLOE-S", "PP-YOLOE-M"}:
        raise ValueError("source run is not a PP-YOLOE S/M legal-layout run")
    if run_manifest.get("test_used_for_training_or_selection") is not False:
        raise ValueError("source run does not preserve the sealed-test contract")
    dataset_contract = run_manifest.get("dataset_contract") or {}
    labels = dataset_contract.get("labels") or []
    if len(labels) != 25 or len(set(labels)) != len(labels):
        raise ValueError("source run does not declare the frozen 25-class ontology")
    config = Path(str((run_manifest.get("config") or {}).get("path", ""))).resolve()
    if not checkpoint.is_file() or not config.is_file():
        raise FileNotFoundError(checkpoint if not checkpoint.is_file() else config)
    source_files: dict[str, str] = {}
    for path in (checkpoint, config, run_manifest_path):
        try:
            relative = path.relative_to(run_root).as_posix()
        except ValueError as error:
            raise ValueError(f"source file is outside run root: {path}") from error
        source_files[relative] = sha256(path)
    return {
        "source_id": source_id,
        "model_name": model_name,
        "source_files": source_files,
        "labels": labels,
        "dataset": dataset_contract,
        "training": {
            key: run_manifest.get(key)
            for key in (
                "model",
                "resolution",
                "epochs",
                "batch_size",
                "learning_rate",
                "warmup_steps",
                "static_assigner_epoch",
                "augmentation",
                "seed",
            )
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_manifest(args.run_root, args.checkpoint, args.source_id, args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
