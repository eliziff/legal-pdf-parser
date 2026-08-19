from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence


MANIFEST_SCHEMA = "legalpdf.fidelity-manifest.v1"
REPORT_SCHEMA = "legalpdf.fidelity-report.v1"
EXTRACTION_FIELDS = (
    "schema_version",
    "source_name",
    "source_sha256",
    "pages",
    "separators",
    "metadata",
)
REPLAY_FIELDS = (
    "schema_version",
    "source_sha256",
    "prepared_pages",
    "derived_pages",
    "markers",
    "marker_summary",
    "pairing_summary",
    "paragraphs",
    "sections",
    "footnotes",
    "diagnostics",
    "status",
    "validation",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != MANIFEST_SCHEMA:
        raise ValueError("unsupported fidelity manifest")
    if set(value) != {"schema_version", "cases"} or not isinstance(value["cases"], list):
        raise ValueError("fidelity manifest must contain only schema_version and cases")
    cases = []
    seen = set()
    for raw in value["cases"]:
        if not isinstance(raw, dict) or set(raw) != {"id", "pdf", "sha256"}:
            raise ValueError("each fidelity case requires exactly id, pdf, and sha256")
        case_id = raw["id"]
        digest = raw["sha256"]
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen
            or not isinstance(raw["pdf"], str)
            or not raw["pdf"]
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("invalid fidelity case")
        seen.add(case_id)
        pdf = Path(raw["pdf"])
        if pdf.is_absolute():
            raise ValueError("fidelity PDF paths must be relative to the manifest")
        cases.append({**raw, "pdf": str((path.parent / pdf).resolve())})
    return cases


def differences(left: Any, right: Any, *, limit: int = 100) -> list[str]:
    found: list[str] = []

    def visit(old: Any, new: Any, path: str) -> None:
        if len(found) >= limit:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                if key not in old:
                    found.append(f"{path}/{key}: missing from reference")
                elif key not in new:
                    found.append(f"{path}/{key}: missing from Rust")
                else:
                    visit(old[key], new[key], f"{path}/{key}")
            return
        if isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                found.append(f"{path}/length: {len(old)} -> {len(new)}")
            for index, (old_item, new_item) in enumerate(zip(old, new)):
                visit(old_item, new_item, f"{path}/{index}")
            return
        if type(old) is not type(new) or old != new:
            found.append(f"{path}: {old!r} -> {new!r}")

    visit(left, right, "")
    return found[:limit]


def selected(value: dict[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    return {field: value.get(field) for field in fields}


def run(command: list[str], *, cwd: Path, env: dict[str, str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout,
        check=False,
        shell=False,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    return {
        "returncode": completed.returncode,
        "seconds": round(time.perf_counter() - started, 6),
        "error": (completed.stderr or completed.stdout)[-2000:] if completed.returncode else "",
    }


def git_revision(root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        encoding="utf-8",
        capture_output=True,
        check=True,
        shell=False,
    )
    return completed.stdout.strip()


def check_case(
    case: dict[str, Any],
    *,
    oracle_root: Path,
    rust_binary: Path,
    helper: Path,
    timeout: float,
) -> dict[str, Any]:
    pdf = Path(case["pdf"])
    if not pdf.is_file() or sha256(pdf) != case["sha256"]:
        raise ValueError(f"{case['id']}: missing or changed PDF")
    oracle_env = os.environ.copy()
    oracle_env["PYTHONPATH"] = os.pathsep.join(
        [str(oracle_root / "src"), oracle_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    with tempfile.TemporaryDirectory(prefix="legalpdf-fidelity-") as temporary:
        root = Path(temporary)
        paths = {
            "oracle_input": root / "oracle-input.json",
            "rust_input": root / "rust-input.json",
            "oracle_result": root / "oracle-result.json",
            "rust_result": root / "rust-result.json",
        }
        commands = {
            "oracle_extract": (
                [sys.executable, str(helper), "extract", str(pdf), "--output", str(paths["oracle_input"])],
                oracle_root,
                oracle_env,
            ),
            "rust_extract": (
                [str(rust_binary), "_parity-extract", str(pdf), "--output", str(paths["rust_input"])],
                rust_binary.parent,
                os.environ.copy(),
            ),
            "oracle_replay": (
                [sys.executable, str(helper), "replay", str(paths["oracle_input"]), "--output", str(paths["oracle_result"])],
                oracle_root,
                oracle_env,
            ),
            "rust_replay": (
                [str(rust_binary), "_parity-replay", str(paths["oracle_input"]), "--output", str(paths["rust_result"])],
                rust_binary.parent,
                os.environ.copy(),
            ),
        }
        timings = {}
        failures = []
        for name, (command, cwd, env) in commands.items():
            measured = run(command, cwd=cwd, env=env, timeout=timeout)
            timings[name] = measured["seconds"]
            if measured["returncode"]:
                failures.append(f"{name}: {measured['error']}")
                break
        extraction_differences = []
        replay_differences = []
        if not failures:
            oracle_input = json.loads(paths["oracle_input"].read_text(encoding="utf-8"))
            rust_input = json.loads(paths["rust_input"].read_text(encoding="utf-8"))
            oracle_result = json.loads(paths["oracle_result"].read_text(encoding="utf-8"))
            rust_result = json.loads(paths["rust_result"].read_text(encoding="utf-8"))
            extraction_differences = differences(
                selected(oracle_input, EXTRACTION_FIELDS),
                selected(rust_input, EXTRACTION_FIELDS),
            )
            replay_differences = differences(
                selected(oracle_result, REPLAY_FIELDS),
                selected(rust_result, REPLAY_FIELDS),
            )
    return {
        "id": case["id"],
        "sha256": case["sha256"],
        "passed": not failures and not extraction_differences and not replay_differences,
        "failures": failures,
        "extraction_differences": extraction_differences,
        "replay_differences": replay_differences,
        "timings": timings,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare frozen Python and Rust PDF fidelity")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--rust-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--case", action="append", default=[])
    arguments = parser.parse_args(argv)
    manifest = arguments.manifest.resolve()
    oracle_root = arguments.oracle_root.resolve()
    rust_binary = arguments.rust_binary.resolve()
    output = arguments.output.resolve()
    if not rust_binary.is_file():
        raise FileNotFoundError(rust_binary)
    cases = load_manifest(manifest)
    if arguments.case:
        requested = set(arguments.case)
        cases = [case for case in cases if case["id"] in requested]
        if {case["id"] for case in cases} != requested:
            raise ValueError("unknown fidelity case")
    report = {
        "schema_version": REPORT_SCHEMA,
        "oracle_revision": git_revision(oracle_root),
        "rust_sha256": sha256(rust_binary),
        "passed": True,
        "cases": [],
    }
    helper = Path(__file__).with_name("fidelity_oracle.py").resolve()
    for index, case in enumerate(cases, start=1):
        result = check_case(
            case,
            oracle_root=oracle_root,
            rust_binary=rust_binary,
            helper=helper,
            timeout=arguments.timeout,
        )
        report["cases"].append(result)
        report["passed"] = report["passed"] and result["passed"]
        atomic_json(output, report)
        print(f"{index}/{len(cases)} {'PASS' if result['passed'] else 'FAIL'} {case['id']}", flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
