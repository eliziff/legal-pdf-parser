"""Diagnostic product comparison against a frozen run; app throughput is measured by pdf-corpus.ts."""

import argparse
import concurrent.futures
from collections import Counter
import hashlib
import json
import subprocess
import time
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1, choices=range(1, 5))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    documents = manifest["documents"]
    if not documents or len({d["sha256"] for d in documents}) != len(documents):
        raise ValueError("Corpus must contain distinct source identities")
    baseline = args.baseline_dir.resolve(strict=True)
    frozen = json.loads((baseline / "summary.json").read_text(encoding="utf-8"))
    if frozen["manifest_sha256"] != digest(args.manifest):
        raise ValueError("Frozen baseline manifest differs")
    # One reusable output directory; never remove a caller-supplied directory tree.
    args.out.mkdir(parents=True, exist_ok=True)
    marker = args.out / "owner.json"
    owner = {"baseline": str(baseline), "manifest_sha256": digest(args.manifest)}
    if marker.exists():
        if json.loads(marker.read_text(encoding="utf-8")) != owner:
            raise ValueError("Candidate output belongs to a different comparison")
    elif any(args.out.iterdir()):
        raise ValueError("Candidate output is not owned by this runner")
    marker.write_text(json.dumps(owner), encoding="utf-8")
    root = args.root.resolve(strict=True)
    binaries = {"candidate": args.candidate.resolve(strict=True)}
    started = time.monotonic()

    def compare(entry):
        source = (root / entry["path"]).resolve(strict=True)
        if not source.is_relative_to(root) or digest(source) != entry["sha256"]:
            raise ValueError(f"Source identity mismatch: {entry['path']}")
        directory = args.out / entry["sha256"]
        directory.mkdir(exist_ok=True)
        one = directory / "manifest.json"
        one.write_text(json.dumps({"schema_version": manifest["schema_version"], "documents": [entry]}), encoding="utf-8")
        prior = json.loads((baseline / entry["sha256"] / "receipt.json").read_text(encoding="utf-8"))
        if any(prior[key] != entry[key] for key in ("path", "sha256", "pages")):
            raise ValueError("Frozen source identity differs")
        receipt = {**entry, "runs": {"base": prior["runs"]["base"]}}
        for label, binary in binaries.items():
            output = directory / f"{label}.jsonl"
            errors = directory / f"{label}.stderr"
            begin = time.monotonic()
            with output.open("wb") as stdout, errors.open("wb") as stderr:
                try:
                    code = subprocess.run([str(binary), "_pdf-inspector-gate", str(one.resolve()), str(root)], stdout=stdout, stderr=stderr, timeout=600).returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
            receipt["runs"][label] = {"exit_code": code, "seconds": round(time.monotonic() - begin, 3), "output_sha256": digest(output)}
        base, candidate = receipt["runs"].values()
        if base["exit_code"] == candidate["exit_code"] == 0:
            receipt["outcome"] = "identical" if base["output_sha256"] == candidate["output_sha256"] else "changed"
        elif base["exit_code"] != 0 and candidate["exit_code"] == 0:
            receipt["outcome"] = "recovered"
        elif base["exit_code"] == 0:
            receipt["outcome"] = "regressed"
        else:
            receipt["outcome"] = "both_failed"
        if receipt["outcome"] == "changed":
            products = {"candidate": json.loads((directory / "candidate.jsonl").read_text(encoding="utf-8"))}
            base_output = baseline / entry["sha256"] / "base.jsonl"
            if base_output.exists():
                products["base"] = json.loads(base_output.read_text(encoding="utf-8"))
            receipt["differences"] = {
                "text_equal": products["base"]["structure"]["text"] == products["candidate"]["structure"]["text"] if "base" in products else None,
                "notes_equal": products["base"]["structure"].get("notes", []) == products["candidate"]["structure"].get("notes", []) if "base" in products else None,
                "products": {label: {
                    "text_characters": len(product["structure"]["text"]),
                    "node_kinds": dict(Counter(node["kind"] for node in product["structure"]["nodes"])),
                    "notes": len(product["structure"].get("notes", [])),
                } for label, product in products.items()},
            }
        (directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        if receipt["outcome"] == "identical":
            for label in binaries:
                (directory / f"{label}.jsonl").unlink()
        one.unlink()
        return receipt

    receipts = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(compare, entry) for entry in documents]
        for future in concurrent.futures.as_completed(futures):
            receipt = future.result()
            receipts.append(receipt)
            print(f"{len(receipts)}/{len(documents)} {receipt['outcome']} {receipt['path']}", flush=True)
    counts = {name: sum(r["outcome"] == name for r in receipts) for name in ("identical", "changed", "recovered", "regressed", "both_failed")}
    summary = {"manifest_sha256": digest(args.manifest), "binary_sha256": {"base": frozen["binary_sha256"]["base"], **{k: digest(v) for k, v in binaries.items()}}, "documents": len(receipts), "pages": sum(r["pages"] for r in receipts), "seconds": round(time.monotonic() - started, 3), "outcomes": counts}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return int(counts["changed"] + counts["regressed"] + counts["both_failed"] != 0)


if __name__ == "__main__":
    raise SystemExit(main())
