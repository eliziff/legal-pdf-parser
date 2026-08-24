#!/usr/bin/env python3
"""Benchmark the production digital-born parser through its public gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
PARSER_ROOT = HERE.parents[1]
CORPUS_ROOT = PARSER_ROOT.parent
MANIFEST = PARSER_ROOT / "experiments" / "cache-contract-fidelity" / "manifest.json"
MANIFEST_SHA256 = "aab25b794d7d47e543019d57f044f2e83bf58775e972e904d39875c9ead6a9f9"
BOUNDED_RUNNER = PARSER_ROOT / "experiments" / "structure-engine-parity" / "run_bounded.py"
REPORT_SCHEMA = "legalpdf.digitalborn-benchmark.v1"
REPETITIONS = 3


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def median(values: list[float | int]) -> float:
    return round(float(statistics.median(values)), 6)


def receipt(
    documents: list[dict[str, Any]], identities: dict[str, str], complete: bool
) -> dict[str, Any]:
    runs = [run for document in documents for run in document["runs"]]
    pages = sum(document["pages"] for document in documents)
    measured_pages = sum(document["pages"] * len(document["runs"]) for document in documents)
    measured_wall = sum(run["elapsed_seconds"] for run in runs)
    median_wall = sum(document["median_seconds"] for document in documents)
    outputs = [
        {
            "path": document["path"],
            "product_sha256": document["product_sha256"],
            "serialized_output_sha256": document["serialized_output_sha256"],
        }
        for document in documents
    ]
    return {
        "schema_version": REPORT_SCHEMA,
        "complete": complete,
        "protocol": {
            "command": "legalpdf _pdf-inspector-gate <one-document-manifest> <corpus-root>",
            "production_core": "the same legalpdf::derive_pdf_document call used by Node derivePdfDocument",
            "cache": "a new temporary parser cache for every process run",
            "execution": "three sequential process runs per document",
            "timing": "process launch through page queries and serialized JSON stdout",
            "memory": "Windows Job Object peak commit charge for the legalpdf process tree; excludes harness stdout buffering",
        },
        **identities,
        "documents": len(documents),
        "completed_documents": sum(len(document["runs"]) == REPETITIONS for document in documents),
        "pages": pages,
        "repetitions": REPETITIONS,
        "median_corpus_wall_seconds": round(median_wall, 6),
        "median_corpus_pages_per_second": round(pages / median_wall, 3) if median_wall else 0,
        "measured_wall_seconds": round(measured_wall, 6),
        "measured_pages": measured_pages,
        "measured_pages_per_second": round(measured_pages / measured_wall, 3) if measured_wall else 0,
        "peak_process_memory_bytes": max(
            (run["peak_process_memory_bytes"] for run in runs), default=0
        ),
        "peak_job_memory_bytes": max((run["peak_job_memory_bytes"] for run in runs), default=0),
        "outputs_sha256": hashlib.sha256(
            json.dumps(outputs, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "documents_detail": documents,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--binary", type=Path, default=PARSER_ROOT / "target" / "release" / "legalpdf.exe"
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--corpus-root", type=Path, default=CORPUS_ROOT)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    binary = args.binary.resolve()
    manifest_path = args.manifest.resolve()
    corpus_root = args.corpus_root.resolve()
    if not binary.is_file():
        raise SystemExit(f"binary not found: {binary}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("documents")
    if (
        manifest.get("schema_version") != "legalpdf.cache-contract-corpus.v1"
        or file_sha256(manifest_path) != MANIFEST_SHA256
        or not isinstance(rows, list)
        or len(rows) != 8
        or not all(isinstance(row, dict) for row in rows)
        or sum(row.get("pages", 0) for row in rows) != 425
        or len({row.get("path") for row in rows}) != 8
    ):
        raise SystemExit(f"invalid manifest: {manifest_path}")

    identities = {
        "binary": str(binary),
        "binary_sha256": file_sha256(binary),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "harness_sha256": file_sha256(Path(__file__)),
        "bounded_runner_sha256": file_sha256(BOUNDED_RUNNER),
    }
    documents: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="digitalborn-benchmark-", dir=output.parent) as raw:
        workspace = Path(raw)
        for position, row in enumerate(rows, 1):
            pdf = corpus_root / row["path"]
            if not pdf.is_file() or file_sha256(pdf) != row["sha256"]:
                raise SystemExit(f"missing or changed corpus input: {pdf}")
            document = {
                "path": row["path"],
                "sha256": row["sha256"],
                "pages": row["pages"],
                "product_sha256": None,
                "serialized_output_sha256": None,
                "runs": [],
            }
            documents.append(document)
            for repetition in range(1, REPETITIONS + 1):
                print(
                    f"DOCUMENT {position}/{len(rows)} RUN {repetition}/{REPETITIONS} "
                    f"{row['path']}",
                    flush=True,
                )
                run_dir = workspace / f"{position:02d}-{repetition}"
                run_dir.mkdir()
                one_document_manifest = run_dir / "manifest.json"
                one_document_manifest.write_text(
                    json.dumps(
                        {"schema_version": manifest["schema_version"], "documents": [row]},
                        separators=(",", ":"),
                    ),
                    encoding="utf-8",
                )
                bounded_receipt = run_dir / "receipt.json"
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(BOUNDED_RUNNER),
                        "--cwd",
                        str(PARSER_ROOT),
                        "--receipt",
                        str(bounded_receipt),
                        "--timeout",
                        str(args.timeout),
                        "--",
                        str(binary),
                        "_pdf-inspector-gate",
                        str(one_document_manifest),
                        str(corpus_root),
                    ],
                    capture_output=True,
                )
                if completed.returncode:
                    raise SystemExit(
                        f"legalpdf failed for {row['path']} ({completed.returncode}): "
                        f"{completed.stderr.decode(errors='replace').strip()}"
                    )
                bounded = json.loads(bounded_receipt.read_text(encoding="utf-8"))
                lines = completed.stdout.splitlines()
                if len(lines) != 1:
                    raise SystemExit(f"expected one product row for {row['path']}")
                product = json.loads(lines[0])
                product_sha256 = product.get("product_sha256")
                if (
                    product.get("path") != row["path"]
                    or len(product.get("pages", [])) != row["pages"]
                    or not isinstance(product.get("structure"), dict)
                    or not isinstance(product_sha256, str)
                    or len(product_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in product_sha256)
                ):
                    raise SystemExit(f"invalid product output for {row['path']}")
                if document["product_sha256"] not in (None, product_sha256):
                    raise SystemExit(f"non-deterministic product output for {row['path']}")
                stdout_sha256 = hashlib.sha256(completed.stdout).hexdigest()
                if document["serialized_output_sha256"] not in (None, stdout_sha256):
                    raise SystemExit(f"non-deterministic serialized output for {row['path']}")
                if (
                    bounded.get("schema_version") != "legalpdf.bounded-command.v1"
                    or bounded["runner_sha256"] != identities["bounded_runner_sha256"]
                    or bounded["stdout_sha256"] != stdout_sha256
                    or bounded["peak_process_memory_bytes"] <= 0
                    or bounded["peak_job_memory_bytes"] <= 0
                ):
                    raise SystemExit(f"invalid bounded receipt for {row['path']}")
                document["product_sha256"] = product_sha256
                document["serialized_output_sha256"] = stdout_sha256
                document["runs"].append(
                    {
                        "elapsed_seconds": bounded["elapsed_seconds"],
                        "stdout_sha256": bounded["stdout_sha256"],
                        "peak_process_memory_bytes": bounded["peak_process_memory_bytes"],
                        "peak_job_memory_bytes": bounded["peak_job_memory_bytes"],
                    }
                )
                elapsed = [run["elapsed_seconds"] for run in document["runs"]]
                document["median_seconds"] = median(elapsed)
                document["median_pages_per_second"] = round(
                    row["pages"] / document["median_seconds"], 3
                )
                document["median_peak_process_memory_bytes"] = int(
                    statistics.median(
                        run["peak_process_memory_bytes"] for run in document["runs"]
                    )
                )
                document["median_peak_job_memory_bytes"] = int(
                    statistics.median(run["peak_job_memory_bytes"] for run in document["runs"])
                )
                atomic_json(
                    output,
                    receipt(documents, identities, False),
                )

    current_identities = {
        "binary_sha256": file_sha256(binary),
        "manifest_sha256": file_sha256(manifest_path),
        "harness_sha256": file_sha256(Path(__file__)),
        "bounded_runner_sha256": file_sha256(BOUNDED_RUNNER),
    }
    if any(identities[key] != value for key, value in current_identities.items()):
        raise SystemExit("binary, manifest, or harness changed during benchmark")
    final = receipt(documents, identities, True)
    atomic_json(output, final)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
