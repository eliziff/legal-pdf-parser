import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STUDENT = ROOT / "kraken-lite-student" / "known-good-input"
OUTPUT = HERE / "benchmark-splits"
VERIFICATION = ROOT / "kraken-lite-training-data" / "ordered-export-verification.json"


def validate_manual_silver(path):
    audit_path = path.parent / "manual-audit.json"
    if not audit_path.exists():
        raise ValueError(f"unvetted silver (no manual audit): {path}")
    page = json.loads(audit_path.read_text(encoding="utf-8"))["pages"].get(path.stem, {})
    if page.get("status") != "verified" or page.get("label") != path.name:
        raise ValueError(f"unvetted silver: {path}")


def validate_true_gold(path, exports):
    run_id = path.parent.name
    transcription = exports.get(run_id)
    if transcription not in {"manual", "clean-hard-examples"}:
        raise ValueError(f"not verified manual gold: {path}")


def validate_benchmark_paths(paths):
    verification = json.loads(VERIFICATION.read_text(encoding="utf-8"))
    exports = {row["runId"]: row["transcription"] for row in verification["exports"]}
    counts = {"true_gold": 0, "manually_vetted_silver": 0}
    for path in paths:
        path = Path(path).resolve()
        if path.is_relative_to(STUDENT.resolve()):
            validate_true_gold(path, exports)
            counts["true_gold"] += 1
        elif path.is_relative_to(HERE.resolve()):
            validate_manual_silver(path)
            counts["manually_vetted_silver"] += 1
        else:
            raise ValueError(f"benchmark truth has no accepted provenance: {path}")
    return counts


def relocated(name):
    paths = []
    for line in (STUDENT / name).read_text(encoding="utf-8-sig").splitlines():
        if line:
            parts = Path(line).parts
            marker = parts.index("known-good-input")
            paths.append((STUDENT / Path(*parts[marker + 1 :])).resolve())
    return paths


def main():
    OUTPUT.mkdir(exist_ok=True)
    diversified = []
    for corpus in (HERE / "courtlistener-scan-silver", HERE / "court-scan-corpus"):
        diversified.extend(Path(line).with_suffix(".xml").resolve() for line in (corpus / "verified-pages.lst").read_text(encoding="utf-8-sig").splitlines() if line)
    splits = {
        "diversified-manually-vetted-silver-30.lst": diversified,
        "validation-manual-gold-68.lst": relocated("host_eval.lst"),
        "benchmark-manual-gold-55.lst": relocated("host_test.lst"),
    }
    combined = []
    truth = {}
    for name, paths in splits.items():
        missing = [str(path) for path in paths if not path.exists() or not path.with_suffix(".png").exists()]
        if missing:
            raise FileNotFoundError(f"{name}: {missing[:3]}")
        truth[name] = validate_benchmark_paths(paths)
        (OUTPUT / name).write_text("\n".join(map(str, paths)) + "\n", encoding="utf-8")
        combined.extend(paths)
    if len(set(map(str, combined))) != len(combined):
        raise ValueError("benchmark strata overlap")
    (OUTPUT / "benchmark-153.lst").write_text("\n".join(map(str, combined)) + "\n", encoding="utf-8")
    receipt = {
        "benchmark": "benchmark-153.lst",
        "pages": len(combined),
        "composition": {name: len(paths) for name, paths in splits.items()},
        "overlap": 0,
        "truth": {
            "true_gold": sum(row["true_gold"] for row in truth.values()),
            "manually_vetted_silver": sum(row["manually_vetted_silver"] for row in truth.values()),
            "by_source": truth,
        },
        "normalization": "nfkc-collapse-not-soft-hyphen-v1",
        "policy": "one benchmark measures CER and throughput on the same 153 pages and pixels; truth is limited to manual gold or silver with a completed per-page manual audit",
    }
    (OUTPUT / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
