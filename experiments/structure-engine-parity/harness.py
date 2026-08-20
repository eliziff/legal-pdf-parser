"""Fast byte-parity and build-latency gate for the Rust structure engine."""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import tempfile
import time
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
AUDIT = ROOT / ".tmp" / "digital-native-structure-audit"
DEFAULT_BINARY = ROOT / ".tmp" / "release-live" / "bin" / "legalpdf.exe"
BASELINE = HERE / "baseline.json"
ALL_BASELINE = HERE / "all-cache-baseline.json"
ALL_RECEIPTS = ROOT / ".tmp" / "structure-engine-parity-receipts"
HEAVY_INPUT_BYTES = 20 * 1024 * 1024
LIGHT_BATCH_BYTES = 160 * 1024 * 1024
HEAVY_BATCH_BYTES = 128 * 1024 * 1024
HEAVY_JOBS, LIGHT_JOBS = 3, 6
QUALIFIED_PEAK_BYTES, PEAK_EVIDENCE_BOUND = 1_258_016_768, 1_572_520_960
PEAK_HARD_LIMIT = 2 * 1024 * 1024 * 1024
SAMPLE_PREFIXES = (
    "1887",  # audited Canadian closing submission; 159 pages and dense notes
    "52c",  # audited US court form
    "9e1",  # audited US rules
    "0b1",  # audited Canadian presentation/TOC
    "d451",  # audited zoning maps and repeated headers
    "ea4",  # audited research report with tables/notes
    "49bc",  # audited agreement with a long table
    "057c",  # audited US Supreme Court transcript/index
    "0002",  # Australian form
    "0f93",  # New Zealand order
    "0051",  # United Kingdom order
)
SOURCE_SHA = re.compile(br'"source_sha256"\s*:\s*"([0-9a-f]{64})"')


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def load_sample() -> list[dict]:
    records = AUDIT / "records"
    rows = []
    for prefix in SAMPLE_PREFIXES:
        matches = sorted(records.glob(f"{prefix}*.json"))
        if len(matches) != 1:
            raise AssertionError(f"expected one audit record for {prefix}, found {len(matches)}")
        record = json.loads(matches[0].read_text(encoding="utf-8"))
        if record.get("outcome") != "passed":
            raise AssertionError(f"audit record did not pass: {matches[0]}")
        candidate = record["candidate"]
        rows.append(
            {
                "candidate_id": matches[0].stem,
                "document_type": candidate["document_type"],
                "jurisdiction": candidate["jurisdiction"],
                "lines": record["document"]["lines"],
                "pages": candidate["page_count"],
                "relative_path": candidate["relative_path"],
                "source_sha256": candidate["sha256"],
            }
        )
    return rows


def load_all_sample() -> list[dict]:
    rows = []
    record_paths = sorted((AUDIT / "records").glob("*.json"))
    outcomes: dict[str, int] = {}
    for path in record_paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        outcome = record.get("outcome", "missing")
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if record.get("outcome") != "passed":
            continue
        candidate = record["candidate"]
        rows.append(
            {
                "candidate_id": path.stem,
                "document_type": candidate["document_type"],
                "jurisdiction": candidate["jurisdiction"],
                "lines": record.get("document", {}).get("lines", 0),
                "pages": candidate["page_count"],
                "relative_path": candidate["relative_path"],
                "source_sha256": candidate["sha256"],
            }
        )
    if len(record_paths) != 750 or outcomes != {"failed": 2, "passed": 748}:
        raise AssertionError(
            f"audit denominator drift: records={len(record_paths)}, outcomes={outcomes}"
        )
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise AssertionError("duplicate candidate ID in audit records")
    if len({row["source_sha256"] for row in rows}) != len(rows):
        raise AssertionError("duplicate source SHA-256 in passed audit records")
    return rows


def extraction_index(wanted: set[str]) -> tuple[dict[str, Path], float]:
    started = time.perf_counter()
    found: dict[str, Path] = {}
    extraction_root = AUDIT / "cache" / "parse-v1" / "extractions"
    for path in extraction_root.glob("*.json.gz"):
        with gzip.open(path, "rb") as stream:
            match = SOURCE_SHA.search(stream.read(4096))
        if match:
            source_sha = match.group(1).decode("ascii")
            if source_sha in wanted:
                if source_sha in found:
                    raise AssertionError(f"duplicate cached extraction for source: {source_sha}")
                found[source_sha] = path
    missing = sorted(wanted - found.keys())
    if missing:
        raise AssertionError(f"missing cached extractions: {missing}")
    return found, time.perf_counter() - started


def validate_document_cache(wanted: set[str]) -> float:
    started = time.perf_counter()
    found: dict[str, Path] = {}
    document_root = AUDIT / "cache" / "parse-v1" / "documents"
    for path in document_root.glob("*.json.gz"):
        with gzip.open(path, "rb") as stream:
            match = SOURCE_SHA.search(stream.read(4096))
        if not match:
            raise AssertionError(f"document cache has no source SHA-256 header: {path}")
        source_sha = match.group(1).decode("ascii")
        if source_sha in found:
            raise AssertionError(f"duplicate cached document for source: {source_sha}")
        found[source_sha] = path
    missing = sorted(wanted - found.keys())
    unexpected = sorted(found.keys() - wanted)
    if missing or unexpected:
        raise AssertionError(
            f"document cache/source drift: missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    return time.perf_counter() - started


def gzip_size(path: Path) -> int:
    with path.open("rb") as stream:
        stream.seek(-4, os.SEEK_END)
        return struct.unpack("<I", stream.read(4))[0]


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str], *, cwd: Path | None = None, timeout: float = 15) -> tuple[float, str]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        creationflags=getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0),
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise AssertionError(f"command failed ({completed.returncode}): {' '.join(command)}\n{completed.stderr}")
    return elapsed, completed.stdout


def startup_metrics(binary: Path, repetitions: int) -> dict:
    run([str(binary), "--version"])
    timings = [run([str(binary), "--version"])[0] * 1000 for _ in range(repetitions)]
    return {
        "median_ms": round(median(timings), 3),
        "p95_ms": round(percentile(timings, 0.95), 3),
        "runs": repetitions,
        "timings_ms": [round(value, 3) for value in timings],
    }


def cargo_is_idle() -> bool:
    if os.name != "nt":
        return True
    for executable in ("cargo.exe", "rustc.exe"):
        completed = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {executable}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if executable.lower() in completed.stdout.lower():
            return False
    return True


def cargo_quick_metrics(max_seconds: float) -> dict:
    if not cargo_is_idle():
        raise AssertionError("cargo or rustc is already running; refusing to start cargo quick")
    elapsed, _ = run(["cargo", "quick"], cwd=ROOT / "legal-pdf-parser", timeout=max_seconds + 2)
    return {"command": "cargo quick", "elapsed_seconds": round(elapsed, 3), "budget_seconds": max_seconds}


def replay(
    binary: Path,
    sample: list[dict],
    index: dict[str, Path],
    work: Path,
    repetitions: int,
    jobs: int,
    max_temp_bytes: int,
) -> tuple[list[dict], float, float]:
    def replay_one(item: tuple[int, dict]) -> dict:
        position, row = item
        print(f"[{position}/{len(sample)}] {row['candidate_id']} ({row['pages']} pages)", flush=True)
        extraction = index[row["source_sha256"]]
        manifest = work / f"{row['candidate_id']}.manifest.json"
        job = [str(extraction), Path(row["relative_path"]).name]
        manifest.write_text(json.dumps([job] * repetitions), encoding="utf-8")
        elapsed, stdout = run([str(binary), "_parity-replay-batch", str(manifest)], timeout=30)
        outputs = list(map(json.loads, stdout.splitlines()))
        if len(outputs) != repetitions or any(value != outputs[0] for value in outputs[1:]):
            raise AssertionError(f"repeat replay bytes differ: {row['candidate_id']}")
        temporary_bytes = manifest.stat().st_size
        if temporary_bytes > max_temp_bytes:
            raise AssertionError(
                f"temporary input for {row['candidate_id']} is {temporary_bytes} bytes; "
                f"budget is {max_temp_bytes}"
            )
        result = {
            **row,
            **outputs[0],
            "extraction_bytes": extraction.stat().st_size,
            "extraction_sha256": sha256_file(extraction),
            "replay_seconds": [round(elapsed / repetitions, 6)] * repetitions,
        }
        manifest.unlink()
        return result

    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(replay_one, enumerate(sample, 1)))
    return results, time.perf_counter() - started, sum(sum(row["replay_seconds"]) for row in results)


def comparable(report: dict) -> dict:
    return {
        "sample_sha256": report["sample_sha256"],
        "documents": [
            {
                key: row[key]
                for key in ("candidate_id", "output_bytes", "output_sha256", "pages")
            }
            for row in report["documents"]
        ],
    }


def comparable_document(row: dict) -> dict:
    return {
        key: row[key]
        for key in ("candidate_id", "extraction_bytes", "extraction_sha256", "output_bytes", "output_sha256", "pages")
    }


def packed(rows: list[dict], index: dict[str, Path], maximum: int) -> list[list[dict]]:
    batches, current, size = [], [], 0
    for row in rows:
        estimated = gzip_size(index[row["source_sha256"]])
        if current and (len(current) == 25 or size + estimated > maximum):
            batches.append(current)
            current, size = [], 0
        current.append(row)
        size += estimated
    if current:
        batches.append(current)
    return batches


def digest_batch(binary: Path, binary_sha: str, rows: list[dict], index: dict[str, Path], work: Path, fresh: bool, maximum: int) -> tuple[list[dict], int]:
    results, pending, extraction_shas = [], [], {}
    for row in rows:
        extraction = index[row["source_sha256"]]
        extraction_sha = sha256_file(extraction)
        extraction_shas[row["candidate_id"]] = extraction_sha
        receipt_path = ALL_RECEIPTS / binary_sha / f"{row['candidate_id']}.json"
        if not fresh and receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            if all((
                receipt.get("schema_version") == "legalpdf.structure-engine-parity-receipt.v3",
                receipt.get("binary_sha256") == binary_sha,
                receipt.get("extraction_sha256") == extraction_sha,
                receipt.get("source_sha256") == row["source_sha256"],
                receipt.get("pages") == row["pages"],
                receipt.get("relative_path") == row["relative_path"],
            )):
                results.append({**receipt, "resumed": True})
                continue
        pending.append(row)
    if not pending:
        return results, 0
    manifest = work / f"{pending[0]['candidate_id']}.manifest.json"
    jobs = [[str(index[row["source_sha256"]]), Path(row["relative_path"]).name] for row in pending]
    manifest.write_text(json.dumps(jobs, separators=(",", ":")), encoding="utf-8")
    uncompressed_bytes = sum(gzip_size(index[row["source_sha256"]]) for row in pending)
    if uncompressed_bytes > maximum:
        raise AssertionError(f"batch uncompressed inputs are {uncompressed_bytes} bytes")
    _, stdout = run([str(binary), "_parity-replay-batch", str(manifest)], timeout=180)
    values = list(map(json.loads, stdout.splitlines()))
    digests = {value["source_sha256"]: value for value in values}
    if len(values) != len(pending) or set(digests) != {row["source_sha256"] for row in pending}:
        raise AssertionError("digest batch output did not cover every source exactly once")
    for row in pending:
        extraction = index[row["source_sha256"]]
        digest = digests[row["source_sha256"]]
        receipt = {
            "schema_version": "legalpdf.structure-engine-parity-receipt.v3",
            "binary_sha256": binary_sha,
            "candidate_id": row["candidate_id"],
            "document_type": row["document_type"],
            "extraction_bytes": extraction.stat().st_size,
            "extraction_sha256": extraction_shas[row["candidate_id"]],
            "input_bytes_uncompressed": gzip_size(extraction),
            "jurisdiction": row["jurisdiction"],
            "lines": digest["input_lines"],
            "max_temp_bytes": manifest.stat().st_size,
            "pages": row["pages"],
            "relative_path": row["relative_path"],
            "source_sha256": row["source_sha256"],
            **{key: value for key, value in digest.items() if key != "input_lines"},
        }
        atomic_json(ALL_RECEIPTS / binary_sha / f"{row['candidate_id']}.json", receipt)
        results.append({**receipt, "resumed": False})
    manifest.unlink()
    return results, uncompressed_bytes


def run_batches(binary: Path, binary_sha: str, batches: list[list[dict]], index: dict[str, Path], work: Path, jobs: int, fresh: bool, maximum: int, completed: int, total: int) -> tuple[list[dict], list[dict], list[int], int]:
    results, failures, temporary = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {pool.submit(digest_batch, binary, binary_sha, rows, index, work, fresh, maximum): rows for rows in batches}
        for future in concurrent.futures.as_completed(futures):
            rows = futures[future]
            completed += len(rows)
            try:
                batch, used = future.result()
                results.extend(batch)
                temporary.append(used)
                state = "resume" if all(row["resumed"] for row in batch) else "digest"
                print(f"[{completed}/{total}] {state} {len(rows)} documents", flush=True)
            except Exception as error:
                failures.append({"candidate_id": rows[0]["candidate_id"], "error": str(error)})
                print(f"[{completed}/{total}] FAIL {rows[0]['candidate_id']}: {error}", flush=True)
    return results, failures, temporary, completed


def misalignment_rejection(binary: Path, work: Path, extraction: Path) -> None:
    source, manifest = work / "misaligned.json.gz", work / "misaligned-manifest.json"
    with gzip.open(extraction, "rt", encoding="utf-8") as stream:
        cached = json.load(stream)
    cached["extraction"]["separators"].pop()
    with gzip.open(source, "wt", encoding="utf-8", compresslevel=1) as stream:
        json.dump(cached, stream, ensure_ascii=False, separators=(",", ":"))
    manifest.write_text(json.dumps([[str(source), "bad.pdf"]]), encoding="utf-8")
    result = subprocess.run([str(binary), "_parity-replay-batch", str(manifest)], capture_output=True, text=True, encoding="utf-8")
    if result.returncode != 1 or "invalid extraction cache" not in result.stderr or result.stdout:
        raise AssertionError("aligned-input rejection contract failed")
    source.unlink()
    manifest.unlink()


def all_cache_execute(args: argparse.Namespace) -> dict:
    started = time.perf_counter()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise AssertionError(f"binary not found: {binary}")
    binary_sha = sha256_file(binary)
    sample = load_all_sample()
    extraction_files = list((AUDIT / "cache" / "parse-v1" / "extractions").glob("*.json.gz"))
    document_files = list((AUDIT / "cache" / "parse-v1" / "documents").glob("*.json.gz"))
    if len(extraction_files) != len(sample) or len(document_files) != len(sample):
        raise AssertionError(
            f"cache denominator drift: records={len(sample)}, extractions={len(extraction_files)}, "
            f"documents={len(document_files)}"
        )
    wanted_sources = {row["source_sha256"] for row in sample}
    index, scan_seconds = extraction_index(wanted_sources)
    document_scan_seconds = validate_document_cache(wanted_sources)
    heavy = [row for row in sample if gzip_size(index[row["source_sha256"]]) > HEAVY_INPUT_BYTES]
    light = [row for row in sample if row not in heavy]
    heavy_jobs, light_jobs = min(HEAVY_JOBS, args.jobs), min(LIGHT_JOBS, args.jobs)
    results = []
    failures = []
    batch_temporary = []
    replay_started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="structure-engine-parity-all-", dir=ROOT / ".tmp") as temporary:
        work = Path(temporary)
        misalignment_rejection(binary, work, min(index.values(), key=lambda path: path.stat().st_size))
        lanes = (
            ([[row] for row in heavy], heavy_jobs, HEAVY_BATCH_BYTES),
            (packed(light, index, LIGHT_BATCH_BYTES), light_jobs, LIGHT_BATCH_BYTES),
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    run_batches, binary, binary_sha, batches, index, work,
                    jobs, args.fresh, maximum, 0, sum(map(len, batches))
                )
                for batches, jobs, maximum in lanes
            ]
            for future in concurrent.futures.as_completed(futures):
                batch, errors, used, _ = future.result()
                results.extend(batch)
                failures.extend(errors)
                batch_temporary.extend(used)
    replay_wall = time.perf_counter() - replay_started
    results.sort(key=lambda row: row["candidate_id"])
    expected_by_id = {}
    if not args.freeze:
        expected = json.loads(ALL_BASELINE.read_text(encoding="utf-8"))
        expected_by_id = {row["candidate_id"]: row for row in expected["documents"]}
        if set(expected_by_id) != {row["candidate_id"] for row in sample}:
            failures.append({"candidate_id": "all-cache", "error": "baseline/sample ID set mismatch"})
    exact_matches = sum(
        comparable_document(row) == comparable_document(expected_by_id[row["candidate_id"]])
        for row in results
        if row["candidate_id"] in expected_by_id
    )
    if not args.freeze:
        for row in results:
            expected_row = expected_by_id.get(row["candidate_id"])
            if expected_row is None or comparable_document(row) != comparable_document(expected_row):
                failures.append({"candidate_id": row["candidate_id"], "error": "byte/hash mismatch"})
    executed = [row for row in results if not row["resumed"]]
    executed_pages = sum(row["pages"] for row in executed)
    executed_rate = executed_pages / replay_wall if executed_pages else None
    if args.fresh and executed_rate is not None and executed_rate < args.min_all_cache_pages_per_second:
        failures.append(
            {
                "candidate_id": "all-cache",
                "error": (
                    f"replay {executed_rate:.1f} pages/s < "
                    f"{args.min_all_cache_pages_per_second} pages/s"
                ),
            }
        )
    if args.fresh and replay_wall > args.max_all_cache_seconds:
        failures.append({"candidate_id": "all-cache", "error": f"replay {replay_wall:.1f}s > {args.max_all_cache_seconds}s"})
    report = {
        "schema_version": "legalpdf.structure-engine-parity-all-cache.v1",
        "binary_sha256": binary_sha,
        "documents": [{key: value for key, value in row.items() if key not in {"resumed"}} for row in results],
        "failures": failures,
        "metrics": {
            "cache_index_seconds": round(scan_seconds, 3),
            "document_cache_index_seconds": round(document_scan_seconds, 3),
            "documents": len(results),
            "exact_matches": None if args.freeze else exact_matches,
            "executed_documents": len(executed),
            "executed_pages": executed_pages,
            "executed_pages_per_second": round(executed_rate, 1) if executed_rate else None,
            "heavy_documents": len(heavy),
            "jobs_heavy": heavy_jobs,
            "jobs_light": light_jobs,
            "lines": sum(row["lines"] for row in results),
            "pages": sum(row["pages"] for row in results),
            "replay_wall_seconds": round(replay_wall, 3),
            "resumed_documents": len(results) - len(executed),
            "fresh": args.fresh,
            "misalignment_rejection": "pass",
            "total_seconds": round(time.perf_counter() - started, 3),
        },
        "bounds": {
            "heavy_input_threshold_mib": HEAVY_INPUT_BYTES / 1024 / 1024,
            "heavy_uncompressed_mib_per_batch": HEAVY_BATCH_BYTES / 1024 / 1024,
            "light_uncompressed_mib_per_batch": LIGHT_BATCH_BYTES / 1024 / 1024,
            "max_uncompressed_batch_bytes_observed": max(batch_temporary, default=0),
            "raw_outputs_retained": 0,
            "uncompressed_input_bytes_in_flight_bound": (
                heavy_jobs * HEAVY_BATCH_BYTES + light_jobs * LIGHT_BATCH_BYTES
            ),
            "qualified_peak_working_set_bytes": QUALIFIED_PEAK_BYTES,
            "peak_working_set_evidence_bound_bytes": PEAK_EVIDENCE_BOUND,
            "peak_working_set_hard_limit_bytes": PEAK_HARD_LIMIT,
        },
        "extraction_sha256": canonical_sha(
            [{"candidate_id": row["candidate_id"], "extraction_sha256": row["extraction_sha256"]} for row in results]
        ),
        "output_sha256": canonical_sha(
            [{"candidate_id": row["candidate_id"], "output_sha256": row["output_sha256"]} for row in results]
        ),
        "source_sha256": canonical_sha(
            [{"candidate_id": row["candidate_id"], "source_sha256": row["source_sha256"]} for row in results]
        ),
        "sample_sha256": canonical_sha(
            [
                {
                    "candidate_id": row["candidate_id"],
                    "pages": row["pages"],
                    "relative_path": row["relative_path"],
                    "source_sha256": row["source_sha256"],
                }
                for row in sample
            ]
        ),
    }
    failed_run_path = ALL_RECEIPTS / binary_sha / "failed-run.json"
    if failed_run_path.is_file():
        failed_run = json.loads(failed_run_path.read_text(encoding="utf-8"))
        if failed_run.get("binary_sha256") == binary_sha and len(failed_run.get("documents", [])) == len(sample):
            report["fresh_run_receipt"] = {
                "bounds": failed_run.get("bounds"),
                "failures": failed_run.get("failures"),
                "metrics": failed_run.get("metrics"),
            }
    failed = bool(failures or len(results) != len(sample))
    if failed:
        atomic_json(ALL_RECEIPTS / binary_sha / "failed-run.json", report)
    receipt_files = list((ALL_RECEIPTS / binary_sha).glob("*.json"))
    summary = {
        "binary_sha256": binary_sha,
        "bounds": report["bounds"],
        "failures": failures,
        "extraction_sha256": report["extraction_sha256"],
        "metrics": report["metrics"],
        "output_sha256": report["output_sha256"],
        "receipt_bytes": sum(path.stat().st_size for path in receipt_files),
        "receipt_files": len(receipt_files),
        "source_sha256": report["source_sha256"],
        "sample_sha256": report["sample_sha256"],
        "status": "FAIL" if failed else "PASS",
    }
    if ALL_BASELINE.is_file():
        summary["baseline_bytes"] = ALL_BASELINE.stat().st_size
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise AssertionError(f"all-cache parity failed: {len(results)}/{len(sample)} complete; {failures[:3]}")
    if args.freeze:
        ALL_BASELINE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def execute(args: argparse.Namespace) -> dict:
    binary = args.binary.resolve()
    if not binary.is_file():
        raise AssertionError(f"binary not found: {binary}")
    sample = load_sample()
    sample_sha = canonical_sha(
        [{"candidate_id": row["candidate_id"], "source_sha256": row["source_sha256"]} for row in sample]
    )
    index, scan_seconds = extraction_index({row["source_sha256"] for row in sample})
    startup = startup_metrics(binary, args.startup_runs)
    with tempfile.TemporaryDirectory(prefix="structure-engine-parity-", dir=ROOT / ".tmp") as temporary:
        documents, replay_seconds, process_seconds = replay(
            binary,
            sample,
            index,
            Path(temporary),
            args.repetitions,
            args.jobs,
            int(args.max_temp_mib_per_document * 1024 * 1024),
        )
    pages = sum(row["pages"] for row in sample) * args.repetitions
    replay_pages_per_second = pages / replay_seconds
    cargo_quick = None if args.skip_cargo_quick else cargo_quick_metrics(args.max_cargo_quick_seconds)
    report = {
        "schema_version": "legalpdf.structure-engine-parity.v1",
        "binary": str(binary),
        "binary_sha256": sha256_file(binary),
        "cargo_quick": cargo_quick,
        "documents": documents,
        "extraction_sha256": canonical_sha(
            [{"candidate_id": row["candidate_id"], "extraction_sha256": row["extraction_sha256"]} for row in documents]
        ),
        "metrics": {
            "cache_index_seconds": round(scan_seconds, 3),
            "documents": len(sample),
            "lines_per_repetition": sum(row["lines"] for row in sample),
            "max_temp_mib_per_document": args.max_temp_mib_per_document,
            "parallel_jobs": args.jobs,
            "pages_per_repetition": sum(row["pages"] for row in sample),
            "repetitions": args.repetitions,
            "replay_pages_per_second": round(replay_pages_per_second, 1),
            "replay_process_seconds": round(process_seconds, 3),
            "replay_seconds": round(replay_seconds, 3),
            "startup": startup,
        },
        "output_sha256": canonical_sha(
            [{"candidate_id": row["candidate_id"], "output_sha256": row["output_sha256"]} for row in documents]
        ),
        "sample_sha256": sample_sha,
    }
    failures = []
    if startup["median_ms"] > args.max_startup_ms:
        failures.append(f"startup median {startup['median_ms']} ms > {args.max_startup_ms} ms")
    if replay_pages_per_second < args.min_replay_pages_per_second:
        failures.append(
            f"replay {replay_pages_per_second:.1f} pages/s < {args.min_replay_pages_per_second} pages/s"
        )
    if cargo_quick and cargo_quick["elapsed_seconds"] > args.max_cargo_quick_seconds:
        failures.append(
            f"cargo quick {cargo_quick['elapsed_seconds']} s > {args.max_cargo_quick_seconds} s"
        )
    if not args.freeze:
        expected = json.loads(BASELINE.read_text(encoding="utf-8"))
        if comparable(report) != comparable(expected):
            failures.append("replay output is not byte-identical to baseline hashes")
    if failures:
        raise AssertionError("; ".join(failures))
    if args.freeze:
        BASELINE.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "binary_sha256": report["binary_sha256"],
                "cargo_quick": report["cargo_quick"],
                "metrics": report["metrics"],
                "output_sha256": report["output_sha256"],
                "sample_sha256": report["sample_sha256"],
                "status": "PASS",
            },
            indent=2,
        ),
        flush=True,
    )
    return report


def self_test() -> None:
    assert QUALIFIED_PEAK_BYTES < PEAK_EVIDENCE_BOUND < PEAK_HARD_LIMIT
    assert percentile([3.0, 1.0, 2.0], 0.95) == 3.0
    assert canonical_sha({"b": 2, "a": 1}) == canonical_sha({"a": 1, "b": 2})
    assert comparable({"sample_sha256": "s", "documents": [{
        "candidate_id": "x", "output_bytes": 2,
        "output_sha256": "o", "pages": 3, "replay_seconds": [1.0]
    }]})["documents"][0]["output_sha256"] == "o"
    print("self-test PASS")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--all-cache", action="store_true", help="gate every cached native document")
    parser.add_argument("--freeze", action="store_true", help="replace the frozen byte-hash baseline")
    parser.add_argument("--repetitions", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=min(LIGHT_JOBS, max(1, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--startup-runs", type=int, default=7)
    parser.add_argument("--max-startup-ms", type=float, default=100.0)
    parser.add_argument("--max-temp-mib-per-document", type=float, default=128.0)
    parser.add_argument("--min-replay-pages-per-second", type=float, default=250.0)
    parser.add_argument("--min-all-cache-pages-per-second", type=float, default=1000.0)
    parser.add_argument("--max-all-cache-seconds", type=float, default=30.0)
    parser.add_argument("--max-cargo-quick-seconds", type=float, default=4.0)
    parser.add_argument("--skip-cargo-quick", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore resumable receipts and enforce full-corpus speed")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if args.jobs < 1:
        parser.error("--jobs must be at least 1")
    if args.all_cache:
        if args.freeze and not args.fresh:
            parser.error("--all-cache --freeze requires --fresh")
        if not args.freeze and not ALL_BASELINE.is_file():
            parser.error("all-cache-baseline.json is missing; run --all-cache --freeze")
        all_cache_execute(args)
        return 0
    if args.repetitions < 2:
        parser.error("--repetitions must be at least 2 to prove byte determinism")
    if not args.freeze and not BASELINE.is_file():
        parser.error("baseline.json is missing; run once with --freeze")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
