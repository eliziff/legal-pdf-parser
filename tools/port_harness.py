from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path, PureWindowsPath
from typing import Any, Iterable, Sequence


MANIFEST_SCHEMA = "legalpdf.port-corpus.v1"
RUN_SCHEMA = "legalpdf.port-run.v1"
REPORT_SCHEMA = "legalpdf.port-gate.v1"
CHECKPOINT_EVERY = 25
MARKER_RE = re.compile(r"\u27e6FN:(?P<pair>[^\u27e7]+)\u27e7")
SUPERSCRIPTS = str.maketrans("\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079", "0123456789")
STATUS_RANK = {
    "failed": 0,
    "empty": 0,
    "ocr_required": 1,
    "degraded": 2,
    "ready": 3,
}

ORACLE_COMMIT = "11a325fe7676f799fd7fb9d119826687dbb68dcf"
ORACLE_FILES = {
    "__init__.py": "1cc8f406803b8bc155335044738361e8d4ad0d38b77b874630b52d62342d136d",
    "__main__.py": "482b8693e2fe6dbd732e3d7961e03827d3efa187bad11d29da188be580ebbeea",
    "adapters.py": "6a1710d0bc288c294ec8322fe0ed95c85aa5932c8382c1da4f9e40a69a005aeb",
    "anchored_scan.py": "06d13c79e067c169347da02289ee9459a91334c565185fb0c5f40c243c1876e6",
    "benchmark.py": "68b460a419f2ae63b02dff7d36c6260fe3572004722e826199590b8e1e838589",
    "cli.py": "684ea4d90d0bd1579f607bbf53844fdf70df4006328ece2dba557e98586e195e",
    "codex_repair.py": "3d439a7e1e4fedd4cb77b153d2ac2317106e8e5bacadc448c71f35ec5d491f28",
    "column_order_arbiter.py": "d1e64e6911d4759819e12b95916ca812d107394df97b88a28ea1fcdbb15a68a5",
    "core.py": "790e4de14f8f43ba05303486bc48dc19bec5c765c29d8b9055fe9cb043d12a2f",
    "deterministic_citations.py": "ce8564c09992b3d2de48720edb43a1c118be76bfe593f5a24949f2e6f5ddbdd0",
    "docx_linking.py": "41285d3b3055a619c4064779274ee4dc731e3047bd5f82814367adb5bc608049",
    "footnote_pairing.py": "297ef847a0855f98ab4b2d39a419600479b69fa1651e405caba6187cde7ddeb6",
    "footnote_pairing_support.py": "ba9bc40fc4cc4d12d9f8cbc7f1620170fb4714189e5b68c417b7e3119579d76a",
    "footnote_separator_scan.py": "051f7212920c942465750fd231ba63bcbc26d9ba26b6a46be54a11ed62072af9",
    "grammar_tables.py": "d7c40f46641e4e8c612dee29320f31127d075261fe37ff793cd9e67b39e23131",
    "model.py": "da4303f896d3b67e310df02bb64ac2cfafedbc54ff12adddcc22319804838de0",
    "note_crossrefs.py": "3decebe1d56326ec417d30105d6acdf0ff6c0573c06fab32d5005461a9229399",
    "ocr.py": "8d8c2db9f172c4bfd387135b08e493294b6b53a34e8e1a8631004efb3b887977",
    "superscript_splice.py": "68817de4e652cc6337b1aad175aa7847778bcaf56252c4214f84cc4dfe1af973",
    "data/mcgill_reporters.json": "6c5aa7b0b826e0842ff0631de40ee8457cb5e948eb120e7568686489703b57d7",
    "tools/export_docx_word.ps1": "53466492360f31df55cfa77fecce4e7df5fa000dda32511cbfd5022b377eafb3",
}
ORACLE_ROOT_FILES = {
    "pyproject.toml": "6198c6ab137ba714411ba60fcb05c800f4bce696da0d87d77ad7352a5005f5c2",
    "data/grammar-tables/citations.json": "51675b8eb94ff067c62103c271a2a7a7d91de551df4b39bead9bcf9c14359560",
    "data/grammar-tables/footnote-labels.json": "bf283283ddaf4d5b8c9c42c94ef1d78c4b8ace93eb50f2ad2c69163dcf55ec19",
    "data/grammar-tables/pinpoints.json": "8801cef918ad82a83fdbb6bd0b95b205a9fcd49e25f5b89a0393ef1b11c5a42d",
    "data/grammar-tables/references.json": "b42f38fea7ed4c66272d0d09d33df5176744e2db92674d393cfb649a132e22d4",
    "tests/test_anchored_scan.py": "2a1e925ac9b35f8bee7e9e512a9a0fadfefd1b0e1b45f83c1e6bf7028383156f",
    "tests/test_benchmark_contract.py": "a9829492afa35924bcb442927fe4efd066d01ee074210d41a2d4bab7fc856846",
    "tests/test_column_order_arbiter.py": "22c4367b4cd4fcd0ce9d295154f558c636a5e86bcdf826fa0deafea2d2f41b2a",
    "tests/test_deterministic_citations.py": "1aae26a5a9da2f11673e4e27f7d693b30b4f5d5b799c69991d88855423bf9f1c",
    "tests/test_docx_linking.py": "8d2e12d7bfcca74c6e439fb433b9f0e14138b802baef17be2f7a16e85ea8f78e",
    "tests/test_engine.py": "fcc7898983ead8d492051737f25d62fd985a53f7555eac319edc8f009b2ae0fd",
    "tests/test_footnote_pairing.py": "2f5b88696b556d21659588e69815ecea88b2beba88c55c06c54d2832cc630f3d",
    "tests/test_footnote_separator_scan.py": "7f0ab871b95003089a4dd9fdf3b999e9f594c7045e52895566bdcee226a0be2d",
    "tests/test_note_crossrefs.py": "38a0a4ac3c739ad347cd2b35d91a11f86b37aae1ac4db9de5ebfaf81e007399c",
    "tests/test_oracle_vectors.py": "466274a77c4ae95e485bbe96cd77a2f6c7c913672612565c41237bf2edb41b44",
    "tests/test_superscript_splice.py": "a3d575106d19b4b9d9f66469f5830c6007fa1fa1c9906096ea0125f4a9722469",
    "tests/test_text_flow_faults.py": "1229a66e433483e7afe238331825ba99d73616b26a6d1af287a681f8666da5ca",
}

# This is a scope ledger, not parity evidence. A mapped production module stays
# blocking until its differential lane proves the implementation. The two tool
# dispositions are non-runtime by design: the corpus harness replaces the old
# benchmark CLI, while Word automation only creates DOCX gold fixtures.
COMPLETE_PORT_STATES = {"proven", "harness-only", "gold-fixture-only"}
PORT_MODULES = {
    "__init__.py": ("proven", ["lib.rs"]),
    "__main__.py": ("proven", ["main.rs"]),
    "adapters.py": ("proven", ["adapters.rs"]),
    "anchored_scan.py": ("proven", ["grammar_tables.rs"]),
    "benchmark.py": ("harness-only", []),
    "cli.py": ("proven", ["main.rs"]),
    "codex_repair.py": ("harness-only", []),
    "column_order_arbiter.py": ("proven", ["structure.rs"]),
    "core.py": (
        "proven",
        [
            "engine.rs",
            "pdf.rs",
            "structure.rs",
            "pairing.rs",
            "storage.rs",
            "lookup.rs",
        ],
    ),
    "deterministic_citations.py": ("proven", ["deterministic_citations.rs"]),
    "docx_linking.py": ("proven", ["codex.rs", "docx.rs"]),
    "footnote_pairing.py": ("proven", ["pairing.rs"]),
    "footnote_pairing_support.py": ("proven", ["pairing_support.rs"]),
    "footnote_separator_scan.py": ("proven", ["separator.rs"]),
    "grammar_tables.py": ("proven", ["grammar_tables.rs", "grammar_word.rs"]),
    "model.py": ("proven", ["model.rs", "storage.rs"]),
    "note_crossrefs.py": ("proven", ["structure.rs"]),
    "ocr.py": ("proven", ["ocr.rs"]),
    "superscript_splice.py": ("proven", ["structure.rs"]),
    "data/mcgill_reporters.json": ("proven", ["pairing_support.rs"]),
    "tools/export_docx_word.ps1": ("gold-fixture-only", []),
    "data/grammar-tables/citations.json": ("proven", ["grammar_tables.rs"]),
    "data/grammar-tables/footnote-labels.json": ("proven", ["grammar_tables.rs"]),
    "data/grammar-tables/pinpoints.json": ("proven", ["grammar_tables.rs"]),
    "data/grammar-tables/references.json": ("proven", ["grammar_tables.rs"]),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str:
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={path.as_posix()}",
            "-C",
            str(path),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "could not identify oracle")
    return completed.stdout.strip()


def _python_symbols(path: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: list[dict[str, Any]] = []

    def visit(body: Sequence[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{node.name}"
                result.append(
                    {
                        "kind": "function",
                        "name": name,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    }
                )
                visit(node.body, f"{name}.<local>.")
            elif isinstance(node, ast.ClassDef):
                name = f"{prefix}{node.name}"
                result.append(
                    {
                        "kind": "class",
                        "name": name,
                        "line": node.lineno,
                        "end_line": getattr(node, "end_lineno", node.lineno),
                    }
                )
                visit(node.body, f"{name}.")
            elif not prefix and isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        result.append(
                            {
                                "kind": "constant",
                                "name": target.id,
                                "line": node.lineno,
                                "end_line": getattr(node, "end_lineno", node.lineno),
                            }
                        )

    visit(tree.body)
    return result


def audit_port_map(arguments: argparse.Namespace) -> int:
    oracle_root = arguments.oracle_root.resolve()
    rust_root = arguments.rust_root.resolve()
    package_root = oracle_root / "src" / "legalpdf"
    errors: list[str] = []
    revision = _git_head(oracle_root)
    if revision != ORACLE_COMMIT:
        errors.append(f"oracle commit changed: {ORACLE_COMMIT} -> {revision}")

    actual_package_files = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    expected_package_files = set(ORACLE_FILES)
    for value in sorted(expected_package_files - actual_package_files):
        errors.append(f"missing oracle package file: {value}")
    for value in sorted(actual_package_files - expected_package_files):
        errors.append(f"unmapped oracle package file: {value}")

    modules: dict[str, Any] = {}
    for relative, expected_hash in sorted(ORACLE_FILES.items()):
        path = package_root / relative
        observed_hash = sha256(path) if path.is_file() else None
        if observed_hash != expected_hash:
            errors.append(
                f"oracle file changed: {relative} ({expected_hash} -> {observed_hash})"
            )
        state, targets = PORT_MODULES[relative]
        modules[relative] = {
            "sha256": observed_hash,
            "symbols": _python_symbols(path) if path.suffix == ".py" and path.is_file() else [],
            "rust_targets": targets,
            "state": state,
        }

    for relative, expected_hash in sorted(ORACLE_ROOT_FILES.items()):
        path = oracle_root / relative
        observed_hash = sha256(path) if path.is_file() else None
        if observed_hash != expected_hash:
            errors.append(
                f"oracle file changed: {relative} ({expected_hash} -> {observed_hash})"
            )
        if relative in PORT_MODULES:
            state, targets = PORT_MODULES[relative]
            modules[relative] = {
                "sha256": observed_hash,
                "symbols": [],
                "rust_targets": targets,
                "state": state,
            }

    mapped_files = set(PORT_MODULES)
    required_mapped_files = set(ORACLE_FILES) | set(ORACLE_ROOT_FILES) & {
        value for value in ORACLE_ROOT_FILES if value.startswith("data/grammar-tables/")
    }
    for value in sorted(required_mapped_files - mapped_files):
        errors.append(f"oracle production file has no port-map entry: {value}")
    for value in sorted(mapped_files - required_mapped_files):
        errors.append(f"port-map entry has no frozen oracle file: {value}")

    rust_files = {path.name for path in rust_root.glob("*.rs") if path.is_file()}
    for module, (_state, targets) in PORT_MODULES.items():
        for target in targets:
            if target not in rust_files:
                errors.append(f"{module}: missing Rust target {target}")

    blockers = [
        f"{module}: {state}"
        for module, (state, _targets) in sorted(PORT_MODULES.items())
        if state not in COMPLETE_PORT_STATES
    ]
    report = {
        "schema_version": "legalpdf.port-map.v1",
        "oracle_commit": revision,
        "expected_oracle_commit": ORACLE_COMMIT,
        "identity_valid": not errors,
        "implementation_complete": not errors and not blockers,
        "errors": errors,
        "blockers": blockers,
        "modules": modules,
    }
    atomic_json(arguments.output.resolve(), report)
    print(
        f"port map: {len(modules)} production files, "
        f"{sum(len(value['symbols']) for value in modules.values())} symbols, "
        f"{len(errors)} identity errors, {len(blockers)} blockers; "
        f"{arguments.output.resolve()}",
        flush=True,
    )
    return 0 if report["implementation_complete"] else 1


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
    )


def normalize_text(value: Any) -> str:
    text = MARKER_RE.sub("", str(value or ""))
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def normalize_label(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "").strip()).translate(
        SUPERSCRIPTS
    )
    try:
        return str(int(text))
    except ValueError:
        return text


def sequence_error(left: Sequence[Any], right: Sequence[Any]) -> int:
    try:
        from rapidfuzz.distance import Levenshtein
    except ImportError as error:
        raise RuntimeError("CER/WER scoring requires the benchmark extra") from error
    return int(Levenshtein.distance(left, right))


def similarity(left: Any, right: Any) -> float:
    return difflib.SequenceMatcher(
        a=normalize_text(left), b=normalize_text(right), autojunk=False
    ).ratio()


def text_metrics(reference: str, candidate: str, prefix: str) -> dict[str, float]:
    expected = normalize_text(reference)
    actual = normalize_text(candidate)
    expected_words = expected.split()
    actual_words = actual.split()
    return {
        f"{prefix}.cer": sequence_error(expected, actual) / max(1, len(expected)),
        f"{prefix}.wer": sequence_error(expected_words, actual_words)
        / max(1, len(expected_words)),
    }


def page_aligned_text_metrics(
    expected_pages: dict[int, str], candidate_pages: Sequence[dict[str, Any]], prefix: str
) -> dict[str, float]:
    character_errors = word_errors = expected_characters = expected_words = 0
    actual_pages = {
        index: "\n".join(
            str(line.get("text") or "")
            for line in sorted(
                page.get("lines") or [],
                key=lambda value: int(value.get("reading_order") or 0),
            )
        )
        for index, page in enumerate(candidate_pages, start=1)
    }
    for page_number, reference in expected_pages.items():
        expected = normalize_text(reference)
        actual = normalize_text(actual_pages.get(page_number, ""))
        expected_tokens = expected.split()
        character_errors += sequence_error(expected, actual)
        word_errors += sequence_error(expected_tokens, actual.split())
        expected_characters += len(expected)
        expected_words += len(expected_tokens)
    return {
        f"{prefix}.cer": character_errors / max(1, expected_characters),
        f"{prefix}.wer": word_errors / max(1, expected_words),
    }


def f1(expected: set[Any], actual: set[Any], prefix: str) -> dict[str, float]:
    common = len(expected & actual)
    precision = common / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = common / len(expected) if expected else 1.0
    value = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        f"{prefix}.precision": precision,
        f"{prefix}.recall": recall,
        f"{prefix}.f1": value,
    }


def deterministic_split(identity: str, salt: str) -> str:
    bucket = int(hashlib.sha256(f"{salt}:{identity}".encode()).hexdigest()[:8], 16)
    return "qualification" if bucket % 5 == 0 else "confirmation"


def manifest_case_token(case_id: str) -> str:
    return hashlib.sha256(case_id.encode()).hexdigest()[:24]


def _resolved_pdf_path(raw_path: str, pdf_root: Path) -> Path:
    parts = PureWindowsPath(raw_path).parts
    lowered = [part.casefold() for part in parts]
    if "pdfs" in lowered:
        parts = parts[lowered.index("pdfs") + 1 :]
    return (pdf_root / Path(*parts)).resolve()


def _resolved_contract_path(raw_path: str, contract_root: Path) -> Path:
    parts = PureWindowsPath(raw_path).parts
    lowered = [part.casefold() for part in parts]
    if "final_contracts" in lowered:
        parts = parts[lowered.index("final_contracts") + 1 :]
    return (contract_root / Path(*parts) / "pages.jsonl").resolve()


def _journal_rows(
    database: Path,
    pdf_root: Path,
    contract_root: Path,
    datasets: set[str],
) -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        query = """select a.article_id,a.dataset,a.pdf_path,a.pdf_page_count,
            a.document_date_en,a.language,a.doc_type,a.pdf_type,
            f.source_dir from articles a join article_final_contracts f using(article_id)"""
        parameters: tuple[Any, ...] = ()
        if datasets:
            placeholders = ",".join("?" for _ in datasets)
            query += f" where a.dataset in ({placeholders})"
            parameters = tuple(sorted(datasets))
        rows = []
        for row in connection.execute(query, parameters):
            article_id = int(row["article_id"])
            if not row["pdf_path"]:
                continue
            dataset = str(row["dataset"])
            rows.append(
                {
                    "article_id": article_id,
                    "dataset": dataset,
                    "page_count": int(row["pdf_page_count"] or 0),
                    "date": str(row["document_date_en"] or ""),
                    "language": str(row["language"] or "unknown"),
                    "doc_type": str(row["doc_type"] or "unknown"),
                    "pdf_type": str(row["pdf_type"] or "unknown"),
                    "pdf": _resolved_pdf_path(str(row["pdf_path"]), pdf_root),
                    "reference": _resolved_contract_path(
                        str(row["source_dir"]), contract_root
                    ),
                }
            )
        return sorted(rows, key=lambda item: (item["dataset"], item["article_id"]))
    finally:
        connection.close()


def _sample_journal_rows(
    rows: list[dict[str, Any]],
    per_dataset: int | None,
    minimums: dict[str, int],
    salt: str,
    diverse: bool = False,
    total: int | None = None,
) -> list[dict[str, Any]]:
    if per_dataset is None and total is None:
        return rows
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    selected = []
    for dataset, values in sorted(by_dataset.items()):
        target = (
            minimums.get(dataset, 0)
            if per_dataset is None
            else max(per_dataset, minimums.get(dataset, 0))
        )
        rank = lambda row: hashlib.sha256(
            f"{salt}:journal:{dataset}:{row['article_id']}".encode()
        ).hexdigest()
        if diverse:
            buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in values:
                year = re.search(r"(?:17|18|19|20)\d{2}", str(row.get("date") or ""))
                decade = f"{int(year.group()) // 10 * 10}s" if year else "unknown"
                pages = int(row.get("page_count") or 0)
                page_band = next(
                    name
                    for ceiling, name in ((10, "short"), (25, "medium"), (50, "long"), (sys.maxsize, "very-long"))
                    if pages <= ceiling
                )
                buckets[
                    (
                        decade,
                        page_band,
                        str(row.get("language") or "unknown").casefold(),
                        str(row.get("doc_type") or "unknown").casefold(),
                        str(row.get("pdf_type") or "unknown").casefold(),
                    )
                ].append(row)
            for bucket in buckets.values():
                bucket.sort(key=rank)
            bucket_order = sorted(
                buckets,
                key=lambda key: hashlib.sha256(f"{salt}:{dataset}:{key}".encode()).hexdigest(),
            )
            values = [
                row
                for offset in range(max(map(len, buckets.values()), default=0))
                for key in bucket_order
                for row in buckets[key][offset : offset + 1]
            ]
        else:
            values.sort(key=rank)
        selected.extend(values[:target])
    if total is not None:
        if len(selected) > total:
            raise ValueError(f"dataset quotas selected {len(selected)} rows, above total {total}")
        selected_ids = {row["article_id"] for row in selected}
        remaining = [row for row in rows if row["article_id"] not in selected_ids]
        counts = Counter(str(row["dataset"]) for row in selected)
        while len(selected) < total and remaining:
            row = min(
                remaining,
                key=lambda value: (
                    counts[str(value["dataset"])],
                    hashlib.sha256(
                        f"{salt}:fill:{value['dataset']}:{value['article_id']}".encode()
                    ).hexdigest(),
                ),
            )
            remaining.remove(row)
            selected.append(row)
            counts[str(row["dataset"])] += 1
        if len(selected) != total:
            raise ValueError(f"only {len(selected)} journal rows are available for total {total}")
    return sorted(selected, key=lambda item: (item["dataset"], item["article_id"]))


def _dataset_minimums(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        dataset, separator, raw_count = value.partition("=")
        if not separator or not dataset or not raw_count.isdigit():
            raise ValueError(f"invalid dataset minimum: {value!r}; expected DATASET=COUNT")
        result[dataset] = int(raw_count)
    return result


def _mark_performance_cases(rows: list[dict[str, Any]], per_lane: int) -> None:
    by_lane: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        row["performance_case"] = False
        by_lane[str(row["lane"])].append(row)
    for values in by_lane.values():
        values.sort(
            key=lambda row: hashlib.sha256(str(row["case_id"]).encode()).hexdigest()
        )
        for row in values[:per_lane]:
            row["performance_case"] = True


def freeze_journals(arguments: argparse.Namespace) -> int:
    output = arguments.output.resolve()
    existing = {
        str(row["case_id"]): row
        for row in (read_jsonl(output) if output.is_file() else [])
    }
    source_rows = _journal_rows(
        arguments.database.resolve(),
        arguments.pdf_root.resolve(),
        arguments.contract_root.resolve(),
        set(arguments.dataset or []),
    )
    excluded = {
        str(row["case_id"])
        for manifest in arguments.exclude_manifest
        for row in read_jsonl(manifest.resolve())
    }
    source_rows = [
        row
        for row in source_rows
        if f"journal:{row['dataset']}:{row['article_id']}" not in excluded
    ]
    source_rows = _sample_journal_rows(
        source_rows,
        arguments.per_dataset,
        _dataset_minimums(arguments.minimum_dataset),
        arguments.salt,
        arguments.diverse,
        arguments.total,
    )
    frozen: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        case_id = f"journal:{source['dataset']}:{source['article_id']}"
        old = existing.get(case_id)
        pdf = source["pdf"]
        reference = source["reference"]
        if not pdf.is_file():
            raise FileNotFoundError(f"missing PDF for {case_id}: {pdf}")
        if not reference.is_file():
            raise FileNotFoundError(f"missing pages.jsonl for {case_id}: {reference}")
        pdf_stat = pdf.stat()
        reference_stat = reference.stat()
        reusable = bool(
            old
            and old.get("pdf") == str(pdf)
            and old.get("pdf_bytes") == pdf_stat.st_size
            and old.get("evidence", {}).get("path") == str(reference)
            and old.get("evidence", {}).get("bytes") == reference_stat.st_size
        )
        if reusable:
            row = old
        else:
            reference_rows = read_jsonl(reference)
            reference_pdf_pages = [
                int(page.get("pdf_page") or page_index + 1)
                for page_index, page in enumerate(reference_rows)
            ]
            if len(reference_pdf_pages) != len(set(reference_pdf_pages)):
                raise ValueError(
                    f"duplicate reference PDF page for {case_id}: {reference_pdf_pages}"
                )
            if source["page_count"] and any(
                page < 1 or page > source["page_count"] for page in reference_pdf_pages
            ):
                raise ValueError(
                    f"reference PDF page is outside {case_id}: "
                    f"database={source['page_count']} reference={reference_pdf_pages}"
                )
            taxonomy_counts: Counter[str] = Counter()
            labels: set[str] = set()
            refs: set[str] = set()
            for page in reference_rows:
                for annotation in page.get("annotations") or []:
                    taxonomy = str(annotation.get("taxonomy_name") or "")
                    if taxonomy:
                        taxonomy_counts[taxonomy] += 1
                    if annotation.get("pair_status") != "paired" or not annotation.get("pair_id"):
                        continue
                    if taxonomy == "fn_label":
                        labels.add(str(annotation["pair_id"]))
                    elif taxonomy == "fn_ref":
                        refs.add(str(annotation["pair_id"]))
            pair_count = len(labels & refs)
            stratum = {
                "APPEAL": "multicolumn_true-footnotes",
                "CONST-FORUM": "multicolumn_endnotes",
            }.get(
                source["dataset"],
                "journal-paired-notes" if pair_count else "journal-no-paired-note",
            )
            row = {
                "schema_version": MANIFEST_SCHEMA,
                "case_id": case_id,
                "lane": f"journal:{source['dataset']}",
                "stratum": stratum,
                "split": deterministic_split(case_id, arguments.salt),
                "pdf": str(pdf),
                "pdf_bytes": pdf_stat.st_size,
                "pdf_sha256": sha256(pdf),
                "page_count": source["page_count"] or max(reference_pdf_pages, default=0),
                "evidence": {
                    "kind": "canonical-derived",
                    "path": str(reference),
                    "bytes": reference_stat.st_size,
                    "sha256": sha256(reference),
                    "page_count": len(reference_rows),
                    "pdf_pages": reference_pdf_pages,
                    "paired_note_count": pair_count,
                    "annotation_counts": dict(sorted(taxonomy_counts.items())),
                },
            }
        row["selection"] = {
            key: source[key]
            for key in ("date", "language", "doc_type", "pdf_type", "page_count")
        }
        frozen.append(row)
        if index % CHECKPOINT_EVERY == 0 or index == len(source_rows):
            _mark_performance_cases(frozen, arguments.performance_per_lane)
            atomic_jsonl(output, sorted(frozen, key=lambda value: value["case_id"]))
        print(f"{index}/{len(source_rows)} frozen {case_id}", flush=True)
    _mark_performance_cases(frozen, arguments.performance_per_lane)
    atomic_jsonl(output, sorted(frozen, key=lambda value: value["case_id"]))
    print(f"frozen {len(frozen)} journal cases at {output}", flush=True)
    return 0


def freeze_docx(arguments: argparse.Namespace) -> int:
    source_rows = read_jsonl(arguments.input.resolve())
    output = arguments.output.resolve()
    frozen: list[dict[str, Any]] = []
    for index, source in enumerate(source_rows, start=1):
        pdf = Path(source["pdf"]).resolve()
        gold_path = Path(source["gold"]).resolve()
        docx = Path(source["docx"]).resolve()
        for kind, path in (("PDF", pdf), ("DOCX", docx), ("gold", gold_path)):
            if not path.is_file():
                raise FileNotFoundError(f"missing {kind}: {path}")
        pdf_hash = sha256(pdf)
        docx_hash = sha256(docx)
        if source.get("pdf_sha256") not in (None, pdf_hash):
            raise ValueError(f"PDF hash changed: {pdf}")
        if source.get("docx_sha256") not in (None, docx_hash):
            raise ValueError(f"DOCX hash changed: {docx}")
        gold = json.loads(gold_path.read_text(encoding="utf-8"))
        note_counts = gold.get("note_counts") or {}
        kinds = "+".join(sorted(key for key, count in note_counts.items() if count))
        profile = str(source.get("profile") or "registered")
        case_id = str(source.get("case_id") or f"docx:{docx_hash[:20]}:{profile}")
        frozen.append(
            {
                "schema_version": MANIFEST_SCHEMA,
                "case_id": case_id,
                "lane": f"docx:{profile}",
                "stratum": f"true-{kinds or 'no-notes'}",
                "split": deterministic_split(docx_hash, arguments.salt),
                "pdf": str(pdf),
                "pdf_bytes": pdf.stat().st_size,
                "pdf_sha256": pdf_hash,
                "page_count": source.get("page_count"),
                "source_docx": str(docx),
                "source_docx_sha256": docx_hash,
                "exporter": source.get("exporter"),
                "settings": source.get("settings"),
                "evidence": {
                    "kind": "independent-docx",
                    "path": str(gold_path),
                    "bytes": gold_path.stat().st_size,
                    "sha256": sha256(gold_path),
                },
            }
        )
        if index % CHECKPOINT_EVERY == 0 or index == len(source_rows):
            _mark_performance_cases(frozen, arguments.performance_per_lane)
            atomic_jsonl(output, sorted(frozen, key=lambda value: value["case_id"]))
        print(f"{index}/{len(source_rows)} frozen {case_id}", flush=True)
    _mark_performance_cases(frozen, arguments.performance_per_lane)
    atomic_jsonl(output, sorted(frozen, key=lambda value: value["case_id"]))
    print(f"frozen {len(frozen)} DOCX cases at {output}", flush=True)
    return 0


def merge_manifests(arguments: argparse.Namespace) -> int:
    rows: dict[str, dict[str, Any]] = {}
    for path in arguments.input:
        for row in read_jsonl(path.resolve()):
            case_id = str(row["case_id"])
            if case_id in rows and rows[case_id] != row:
                raise ValueError(f"conflicting frozen case: {case_id}")
            rows[case_id] = row
    atomic_jsonl(arguments.output.resolve(), [rows[key] for key in sorted(rows)])
    print(f"merged {len(rows)} cases at {arguments.output.resolve()}", flush=True)
    return 0


def engine_identity(arm: str, oracle_root: Path | None, rust_binary: Path | None) -> str:
    if arm == "rust":
        if rust_binary is None or not rust_binary.is_file():
            raise FileNotFoundError(f"Rust binary not found: {rust_binary}")
        return f"rust-sha256:{sha256(rust_binary)}"
    if oracle_root is None or not (oracle_root / "src" / "legalpdf").is_dir():
        raise FileNotFoundError(f"oracle checkout not found: {oracle_root}")
    completed = subprocess.run(
        [
            "git",
            "-c",
            f"safe.directory={oracle_root.as_posix()}",
            "-C",
            str(oracle_root),
            "rev-parse",
            "HEAD",
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        shell=False,
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip() or "could not identify oracle")
    return f"oracle-git:{completed.stdout.strip()}"


def _child_peak_sampler(process: subprocess.Popen[str], stop: threading.Event) -> int:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return 0
    peak = 0
    try:
        root = psutil.Process(process.pid)
        while not stop.wait(0.02):
            processes = [root, *root.children(recursive=True)]
            peak = max(
                peak,
                sum(
                    child.memory_info().rss
                    for child in processes
                    if child.is_running()
                ),
            )
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return peak


def run_child(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> dict[str, Any]:
    creationflags = (
        subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        if os.name == "nt"
        else 0
    )
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
        creationflags=creationflags,
    )
    stop = threading.Event()
    peak_holder = {"value": 0}

    def sample() -> None:
        peak_holder["value"] = _child_peak_sampler(process, stop)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        process.kill()
        stdout, stderr = process.communicate()
    finally:
        stop.set()
        sampler.join()
    return {
        "returncode": process.returncode,
        "timed_out": timed_out,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_bytes": peak_holder["value"] or None,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def engine_command(
    arm: str,
    pdf: Path,
    artifact: Path,
    oracle_root: Path | None,
    rust_binary: Path | None,
) -> tuple[list[str], Path, dict[str, str]]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if arm == "oracle":
        assert oracle_root is not None
        env["PYTHONPATH"] = os.pathsep.join(
            [str(oracle_root / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return (
            [
                sys.executable,
                "-X",
                "utf8",
                "-m",
                "legalpdf",
                "parse",
                str(pdf),
                "--output",
                str(artifact),
                "--no-cache",
            ],
            oracle_root,
            env,
        )
    assert rust_binary is not None
    return (
        [
            str(rust_binary),
            "parse",
            str(pdf),
            "--output",
            str(artifact),
            "--no-cache",
        ],
        rust_binary.parent,
        env,
    )


def parity_command(
    phase: str,
    arm: str,
    source: Path,
    artifact: Path,
    oracle_root: Path,
    rust_binary: Path,
) -> tuple[list[str], Path, dict[str, str]]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    if arm == "oracle":
        env["PYTHONPATH"] = os.pathsep.join(
            [str(oracle_root / "src"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        return (
            [
                sys.executable,
                str(Path(__file__).with_name("oracle_replay.py").resolve()),
                phase,
                str(source),
                "--output",
                str(artifact),
            ],
            oracle_root,
            env,
        )
    return (
        [
            str(rust_binary),
            f"_parity-{phase}",
            str(source),
            "--output",
            str(artifact),
        ],
        rust_binary.parent,
        env,
    )


def selected_cases(manifest: Path, split: str) -> list[dict[str, Any]]:
    rows = read_jsonl(manifest)
    if split == "all":
        return rows
    return [row for row in rows if row.get("split") == split]


def selected_case_ids(
    cases: list[dict[str, Any]], requested: Sequence[str]
) -> list[dict[str, Any]]:
    if not requested:
        return cases
    wanted = set(requested)
    available = {str(case["case_id"]) for case in cases}
    if missing := wanted - available:
        raise ValueError(f"unknown case IDs: {', '.join(sorted(missing))}")
    return [case for case in cases if str(case["case_id"]) in wanted]


def case_pdf(case: dict[str, Any]) -> Path:
    for key in ("pdf", "pdf_path"):
        if value := case.get(key):
            return Path(str(value))
    raise ValueError(f"{case.get('case_id', 'case')} has no PDF path")


def run_arm(arguments: argparse.Namespace) -> int:
    manifest = arguments.manifest.resolve()
    progress_every = max(1, arguments.progress_every)
    cases = selected_case_ids(
        selected_cases(manifest, arguments.split), getattr(arguments, "case", [])
    )
    output = arguments.output.resolve()
    oracle_root = arguments.oracle_root.resolve() if arguments.oracle_root else None
    rust_binary = arguments.rust_binary.resolve() if arguments.rust_binary else None
    identity = engine_identity(arguments.arm, oracle_root, rust_binary)
    engine_token = hashlib.sha256(identity.encode()).hexdigest()[:16]
    result_root = output / "results" / arguments.arm / engine_token
    artifact_root = output / "artifacts" / arguments.arm / engine_token
    completed_rows: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_id = str(case["case_id"])
        token = manifest_case_token(case_id)
        result_path = result_root / f"{token}.json"
        artifact = artifact_root / token
        if result_path.is_file():
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if (
                prior.get("engine_id") == identity
                and prior.get("pdf_sha256") == case.get("pdf_sha256")
                and prior.get("returncode") == 0
                and (artifact / "document.json").is_file()
            ):
                completed_rows.append(prior)
                if index == 1 or index == len(cases) or index % progress_every == 0:
                    print(f"{index}/{len(cases)} skip {case_id}", flush=True)
                continue
        command, cwd, env = engine_command(
            arguments.arm,
            case_pdf(case),
            artifact,
            oracle_root,
            rust_binary,
        )
        measured = run_child(command, cwd=cwd, env=env, timeout=arguments.timeout)
        result = {
            "schema_version": RUN_SCHEMA,
            "case_id": case_id,
            "lane": case["lane"],
            "split": case["split"],
            "arm": arguments.arm,
            "engine_id": identity,
            "pdf_sha256": case["pdf_sha256"],
            "artifact": str(artifact / "document.json"),
            **measured,
        }
        atomic_json(result_path, result)
        completed_rows.append(result)
        state = "complete" if measured["returncode"] == 0 else "failed"
        if measured["returncode"] != 0 or index == len(cases) or index % progress_every == 0:
            print(
                f"{index}/{len(cases)} {state} {case_id} "
                f"wall={measured['wall_seconds']:.4f}s",
                flush=True,
            )
    index_path = output / f"{arguments.arm}-{engine_token}-{arguments.split}.jsonl"
    atomic_jsonl(index_path, sorted(completed_rows, key=lambda row: row["case_id"]))
    print(f"run index: {index_path}", flush=True)
    return 0 if all(row["returncode"] == 0 for row in completed_rows) else 1


COMMON_REPLAY_FIELDS = (
    "prepared_pages",
    "derived_pages",
    "paragraphs",
    "sections",
    "footnotes",
    "diagnostics",
    "status",
    "validation",
)


def _canonical_common(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, list):
        return [_canonical_common(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _canonical_common(item)
            for key, item in sorted(value.items())
            if key not in {"created_at", "elapsed_seconds", "engine"}
        }
    return value


def _value_differences(
    left: Any,
    right: Any,
    path: str = "",
    *,
    limit: int = 100,
) -> list[str]:
    failures: list[str] = []

    def visit(old: Any, new: Any, current: str) -> None:
        if len(failures) >= limit:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                child = f"{current}/{key}"
                if key not in old:
                    failures.append(f"{child}: missing from oracle")
                elif key not in new:
                    failures.append(f"{child}: missing from Rust")
                else:
                    visit(old[key], new[key], child)
            return
        if isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                failures.append(f"{current}/length: {len(old)} -> {len(new)}")
            for index, (old_item, new_item) in enumerate(zip(old, new)):
                visit(old_item, new_item, f"{current}/{index}")
            return
        if old != new:
            failures.append(f"{current}: {old!r} -> {new!r}")

    visit(_canonical_common(left), _canonical_common(right), path)
    return failures


def _exact_value_differences(
    left: Any,
    right: Any,
    path: str = "",
    *,
    limit: int = 100,
) -> list[str]:
    failures: list[str] = []

    def visit(old: Any, new: Any, current: str) -> None:
        if len(failures) >= limit:
            return
        if isinstance(old, dict) and isinstance(new, dict):
            for key in sorted(set(old) | set(new)):
                child = f"{current}/{key}"
                if key not in old:
                    failures.append(f"{child}: missing from oracle")
                elif key not in new:
                    failures.append(f"{child}: missing from Rust")
                else:
                    visit(old[key], new[key], child)
            return
        if isinstance(old, list) and isinstance(new, list):
            if len(old) != len(new):
                failures.append(f"{current}/length: {len(old)} -> {len(new)}")
            for index, (old_item, new_item) in enumerate(zip(old, new)):
                visit(old_item, new_item, f"{current}/{index}")
            return
        if type(old) is not type(new) or old != new:
            failures.append(f"{current}: {old!r} -> {new!r}")

    visit(left, right, path)
    return failures


def run_contract_diff(arguments: argparse.Namespace) -> int:
    oracle_root = arguments.oracle_root.resolve()
    rust_binary = arguments.rust_binary.resolve()
    contract_input = arguments.input.resolve()
    if _git_head(oracle_root) != ORACLE_COMMIT:
        raise ValueError("contract replay requires the frozen oracle commit")
    if not rust_binary.is_file():
        raise FileNotFoundError(rust_binary)
    helper = Path(__file__).with_name("oracle_replay.py").resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(oracle_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    commands = {
        "oracle": (
            [sys.executable, str(helper), "contract", str(contract_input)],
            oracle_root,
            env,
        ),
        "rust": (
            [str(rust_binary), "contract", str(contract_input)],
            rust_binary.parent,
            os.environ.copy(),
        ),
    }
    results: dict[str, Any] = {}
    failures: list[str] = []
    timings: dict[str, float] = {}
    for name, (command, cwd, command_env) in commands.items():
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=command_env,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=arguments.timeout,
                check=False,
                shell=False,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    | subprocess.BELOW_NORMAL_PRIORITY_CLASS
                    if os.name == "nt"
                    else 0
                ),
            )
        except subprocess.TimeoutExpired:
            failures.append(f"{name} timed out")
            continue
        timings[name] = time.perf_counter() - started
        if completed.returncode:
            failures.append(
                f"{name} failed ({completed.returncode}): "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )
            continue
        try:
            results[name] = json.loads(completed.stdout)
        except ValueError as error:
            failures.append(f"{name} returned invalid JSON: {error}")
    if not failures:
        failures.extend(
            _exact_value_differences(results["oracle"], results["rust"], limit=100)
        )
    report = {
        "schema_version": "legalpdf.contract-diff.v1",
        "oracle_commit": ORACLE_COMMIT,
        "rust_sha256": sha256(rust_binary),
        "input": str(contract_input),
        "operation": json.loads(contract_input.read_text(encoding="utf-8")).get(
            "operation"
        ),
        "passed": not failures,
        "failure_count": len(failures),
        "failures": failures,
        "timings": timings,
    }
    atomic_json(arguments.output.resolve(), report)
    print(
        f"contract {'PASS' if not failures else 'FAIL'} "
        f"{report['operation']} ({len(failures)} differences); "
        f"{arguments.output.resolve()}",
        flush=True,
    )
    return 0 if not failures else 1


def common_input_regressions(
    oracle: dict[str, Any], candidate: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    for field in COMMON_REPLAY_FIELDS:
        failures.extend(
            _value_differences(
                oracle.get(field), candidate.get(field), f"/{field}", limit=25
            )
        )
    if candidate.get("markers") is None:
        failures.append("/markers: Rust marker-stage trace is missing")
    else:
        failures.extend(
            _value_differences(
                oracle.get("markers"), candidate.get("markers"), "/markers", limit=25
            )
        )
    if candidate.get("marker_summary") is None:
        failures.append("/marker_summary: Rust marker-stage summary is missing")
    else:
        failures.extend(
            _value_differences(
                oracle.get("marker_summary"),
                candidate.get("marker_summary"),
                "/marker_summary",
                limit=25,
            )
        )
    failures.extend(
        _value_differences(
            oracle.get("pairing_summary"),
            candidate.get("pairing_summary"),
            "/pairing_summary",
            limit=25,
        )
    )
    return failures


_CANDIDATE_COUNT_DIFFERENCE = re.compile(
    r"^/(?:marker_summary|pairing_summary)/(?:label_candidate_count|"
    r"article_footnote_pair_materialization/label_backbone/segments/\d+/candidate_count)"
    r": (?P<oracle>\d+) -> (?P<candidate>\d+)$"
)
_NATIVE_SUPERSCRIPT_EVIDENCE = re.compile(
    r"^/markers/\d+/candidate_reason: 'attached_symbol_marker' -> "
    r"'native_superscript_span'$"
)
_MALFORMED_LABEL_SUPPRESSION = re.compile(
    r"^/(?:prepared_pages|derived_pages)/\d+/lines/\d+/suppress_footnote_label: "
    r"True -> False$"
)
_MALFORMED_LABEL_PREFIX = re.compile(r"^\s*(?:\d{1,4}|[*\u2020\u2021\u00a7\u00b6#]{1,8})[.)\],:;-]\S")
_COMPACT_NOTE_LINE = re.compile(
    r"^\s*(?:\d{1,4}|[*\u2020\u2021\u00a7\u00b6#])"
    r"(?:\s*[.)\],:]\s*|\s+)(?P<body>.*)$"
)


def _unused_candidate_pruning(failures: Sequence[str]) -> bool:
    """Accept stricter rejected-candidate accounting when the product is exact."""
    if not failures:
        return False
    matches = [_CANDIDATE_COUNT_DIFFERENCE.fullmatch(value) for value in failures]
    return all(matches) and all(
        int(match.group("candidate")) <= int(match.group("oracle"))
        for match in matches
        if match is not None
    )


def _unused_candidate_growth(failures: Sequence[str]) -> bool:
    """Ignore changed candidate telemetry when all durable stages are exact."""
    if not failures:
        return False
    matches = [_CANDIDATE_COUNT_DIFFERENCE.fullmatch(value) for value in failures]
    return all(matches) and all(
        int(match.group("candidate")) >= int(match.group("oracle"))
        for match in matches
        if match is not None
    )


def _richer_native_superscript_evidence(failures: Sequence[str]) -> bool:
    return bool(failures) and all(_NATIVE_SUPERSCRIPT_EVIDENCE.fullmatch(value) for value in failures)


def _rejected_malformed_label_prefixes(
    failures: Sequence[str], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    if not failures or not all(_MALFORMED_LABEL_SUPPRESSION.fullmatch(value) for value in failures):
        return False
    changed = 0
    for field in ("prepared_pages", "derived_pages"):
        for old_page, new_page in zip(oracle.get(field) or [], candidate.get(field) or []):
            old_lines = old_page.get("lines") or []
            new_lines = new_page.get("lines") or []
            for old_line, new_line in zip(old_lines, new_lines):
                if old_line.get("suppress_footnote_label") == new_line.get("suppress_footnote_label"):
                    continue
                if (
                    old_line.get("suppress_footnote_label") is not True
                    or new_line.get("suppress_footnote_label") is not False
                    or not _MALFORMED_LABEL_PREFIX.match(str(new_line.get("text") or ""))
                ):
                    return False
                changed += 1
    return changed == len(failures)


def _common_document(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "pages": value.get("derived_pages") or [],
        "paragraphs": value.get("paragraphs") or [],
        "sections": value.get("sections") or [],
        "footnotes": value.get("footnotes") or [],
        "diagnostics": value.get("diagnostics") or [],
        "status": value.get("status"),
    }


def _line_map(value: dict[str, Any], field: str) -> dict[str, dict[str, Any]]:
    return {
        str(line.get("id")): line
        for page in value.get(field) or []
        for line in page.get("lines") or []
    }


def _reference_annotation_labels(path: Path, taxonomy: str) -> set[str]:
    return {
        normalize_label(annotation.get("selected_text"))
        for page in read_jsonl(path)
        for annotation in page.get("annotations") or []
        if annotation.get("taxonomy_name") == taxonomy
        and annotation.get("pair_status") == "paired"
    }


def _preserved_compact_note_furniture(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    """Prove that a semantic difference only restores note rows lost as furniture.

    This is deliberately evidence- and shape-based: it cannot bless a case ID,
    page number, deleted line, new footer classification, or unsupported note.
    """
    evidence = case.get("evidence") or {}
    if evidence.get("kind") != "canonical-derived":
        return False
    evidence_path = Path(str(evidence.get("path") or ""))
    if not evidence_path.is_file() or sha256(evidence_path) != evidence.get("sha256"):
        return False
    if candidate.get("validation") != "ok":
        return False

    oracle_lines = _line_map(oracle, "prepared_pages")
    candidate_lines = _line_map(candidate, "prepared_pages")
    if set(oracle_lines) != set(candidate_lines):
        return False
    if any(
        oracle_lines[line_id].get("text") != candidate_lines[line_id].get("text")
        for line_id in oracle_lines
    ):
        return False

    restored: list[str] = []
    for line_id, old in oracle_lines.items():
        new = candidate_lines[line_id]
        old_region = old.get("region_type")
        new_region = new.get("region_type")
        if old_region == new_region:
            continue
        if old_region != "footer" or new_region in {"header", "footer"}:
            return False
        text = str(new.get("text") or "")
        match = _COMPACT_NOTE_LINE.match(text)
        if match is None or sum(character.isalpha() for character in match.group("body")) < 4:
            return False
        restored.append(text)
    if not restored:
        return False

    reference = journal_reference(evidence_path)
    reference_text = normalize_text(reference["text"])
    if any(normalize_text(text) not in reference_text for text in restored):
        return False
    old_metrics = journal_metrics(reference, _common_document(oracle))
    new_metrics = journal_metrics(reference, _common_document(candidate))
    if (
        float(new_metrics["source.cer"]) > float(old_metrics["source.cer"])
        or float(new_metrics["source.wer"]) > float(old_metrics["source.wer"])
    ):
        return False

    old_notes = keyed_notes(oracle.get("footnotes") or [])
    new_notes = keyed_notes(candidate.get("footnotes") or [])
    if not set(old_notes) <= set(new_notes):
        return False
    reference_labels = _reference_annotation_labels(evidence_path, "fn_ref")
    if any(label not in reference_labels for label, _ in set(new_notes) - set(old_notes)):
        return False
    added_supported_notes = bool(set(new_notes) - set(old_notes))
    for key, old_note in old_notes.items():
        old_body = normalize_text(old_note.get("body"))
        new_body = normalize_text(new_notes[key].get("body"))
        if old_body == new_body:
            continue
        if len(new_body) <= len(old_body) or new_body not in reference_text:
            return False
        added_supported_notes = True
    source_improved = (
        float(new_metrics["source.cer"]) < float(old_metrics["source.cer"])
        and float(new_metrics["source.wer"]) < float(old_metrics["source.wer"])
    )
    return source_improved or added_supported_notes


_STANDALONE_NOTE_LABEL = re.compile(
    r"^(?:\d{1,4}|[*\u2020\u2021\u00a7\u00b6#]{1,8})$"
)


def _same_visual_row(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_box = left.get("bbox") or [0.0] * 4
    right_box = right.get("bbox") or [0.0] * 4
    overlap = min(left_box[3], right_box[3]) - max(left_box[1], right_box[1])
    minimum_height = min(left_box[3] - left_box[1], right_box[3] - right_box[1])
    return overlap > 0 and minimum_height > 0 and overlap / minimum_height >= 0.5


def _standalone_note_label(line: dict[str, Any]) -> bool:
    return bool(_STANDALONE_NOTE_LABEL.fullmatch(str(line.get("text") or "").strip()))


def _nearby_note_body(label: dict[str, Any], body: dict[str, Any]) -> bool:
    label_box = label.get("bbox") or [0.0] * 4
    body_box = body.get("bbox") or [0.0] * 4
    return (
        label.get("region_type") == body.get("region_type") == "footnote"
        and not _standalone_note_label(body)
        and body_box[0] >= label_box[0]
        and _same_visual_row(label, body)
    )


def _only_attaches_detached_note_labels(
    oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    old_pages = oracle.get("prepared_pages") or []
    new_pages = candidate.get("prepared_pages") or []
    if len(old_pages) != len(new_pages):
        return False
    changed = False
    for old_page, new_page in zip(old_pages, new_pages):
        old_shell = {
            key: value for key, value in old_page.items() if key not in {"lines", "regions"}
        }
        new_shell = {
            key: value for key, value in new_page.items() if key not in {"lines", "regions"}
        }
        if _canonical_common(old_shell) != _canonical_common(new_shell):
            return False
        old_lines = old_page.get("lines") or []
        new_lines = new_page.get("lines") or []
        old_by_id = {str(line.get("id")): line for line in old_lines}
        new_by_id = {str(line.get("id")): line for line in new_lines}
        if set(old_by_id) != set(new_by_id):
            return False
        for line_id, old_line in old_by_id.items():
            old_value = {key: value for key, value in old_line.items() if key != "reading_order"}
            new_value = {
                key: value
                for key, value in new_by_id[line_id].items()
                if key != "reading_order"
            }
            if _canonical_common(old_value) != _canonical_common(new_value):
                return False
        old_order = [str(line.get("id")) for line in old_lines]
        new_order = [str(line.get("id")) for line in new_lines]
        old_regions = {
            str(region.get("id")): region for region in old_page.get("regions") or []
        }
        new_regions = {
            str(region.get("id")): region for region in new_page.get("regions") or []
        }
        if set(old_regions) != set(new_regions):
            return False
        for region_id, old_region in old_regions.items():
            new_region = new_regions[region_id]
            old_value = {
                key: value for key, value in old_region.items() if key != "line_ids"
            }
            new_value = {
                key: value for key, value in new_region.items() if key != "line_ids"
            }
            old_ids = [str(value) for value in old_region.get("line_ids") or []]
            new_ids = [str(value) for value in new_region.get("line_ids") or []]
            if (
                _canonical_common(old_value) != _canonical_common(new_value)
                or set(old_ids) != set(new_ids)
                or new_ids != [line_id for line_id in new_order if line_id in set(new_ids)]
            ):
                return False
        if old_order == new_order:
            continue
        changed = True
        new_position = {line_id: index for index, line_id in enumerate(new_order)}
        proven_labels = {
            line_id
            for line_id, line in old_by_id.items()
            if _standalone_note_label(line)
            and any(_nearby_note_body(line, body) for body in old_lines)
        }
        for index, old_earlier_id in enumerate(old_order):
            for old_later_id in old_order[index + 1 :]:
                if new_position[old_earlier_id] < new_position[old_later_id]:
                    continue
                body = old_by_id[old_earlier_id]
                label = old_by_id[old_later_id]
                if not (
                    old_later_id in proven_labels
                    and body.get("region_type") == "footnote"
                    and float((body.get("bbox") or [0.0])[0])
                    > float((label.get("bbox") or [0.0])[0])
                    and _same_visual_row(label, body)
                ):
                    return False
    return changed


def _attached_detached_note_labels(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    evidence = case.get("evidence") or {}
    evidence_path = Path(str(evidence.get("path") or ""))
    if (
        evidence.get("kind") != "canonical-derived"
        or not evidence_path.is_file()
        or sha256(evidence_path) != evidence.get("sha256")
        or candidate.get("validation") != "ok"
        or not _only_attaches_detached_note_labels(oracle, candidate)
    ):
        return False
    old_notes = keyed_notes(oracle.get("footnotes") or [])
    new_notes = keyed_notes(candidate.get("footnotes") or [])
    old_body_lines = {
        str(line_id)
        for note in old_notes.values()
        for line_id in note.get("body_line_ids") or []
    }
    new_body_lines = {
        str(line_id)
        for note in new_notes.values()
        for line_id in note.get("body_line_ids") or []
    }
    if set(old_notes) != set(new_notes) or not old_body_lines <= new_body_lines:
        return False
    reference = journal_reference(evidence_path)
    old_metrics = journal_metrics(reference, _common_document(oracle))
    new_metrics = journal_metrics(reference, _common_document(candidate))
    nondecreasing = ("notes.labels.f1", "notes.reference_pages.f1")
    nonincreasing = ("source.cer", "source.wer")
    return (
        all(float(new_metrics[key]) >= float(old_metrics[key]) for key in nondecreasing)
        and all(float(new_metrics[key]) <= float(old_metrics[key]) for key in nonincreasing)
        and any(float(new_metrics[key]) < float(old_metrics[key]) for key in nonincreasing)
    )


_LAYOUT_LINE_FIELDS = {
    "reading_order",
    "region_id",
    "region_type",
    "note_region_mode",
    "suppress_footnote_label",
}
_LAYOUT_PAGE_FIELDS = {"printed_label", "printed_label_line_id", "printed_label_source"}


def _only_layout_changes(oracle: dict[str, Any], candidate: dict[str, Any]) -> bool:
    changed = False
    for field in ("prepared_pages", "derived_pages"):
        old_pages = oracle.get(field) or []
        new_pages = candidate.get(field) or []
        if len(old_pages) != len(new_pages):
            return False
        for old_page, new_page in zip(old_pages, new_pages):
            if _canonical_common(
                {
                    key: value
                    for key, value in old_page.items()
                    if key not in {"lines", "regions", *_LAYOUT_PAGE_FIELDS}
                }
            ) != _canonical_common(
                {
                    key: value
                    for key, value in new_page.items()
                    if key not in {"lines", "regions", *_LAYOUT_PAGE_FIELDS}
                }
            ):
                return False
            old_lines = {str(line.get("id")): line for line in old_page.get("lines") or []}
            new_lines = {str(line.get("id")): line for line in new_page.get("lines") or []}
            if set(old_lines) != set(new_lines):
                return False
            old_printed = tuple(old_page.get(key) for key in sorted(_LAYOUT_PAGE_FIELDS))
            new_printed = tuple(new_page.get(key) for key in sorted(_LAYOUT_PAGE_FIELDS))
            if old_printed != new_printed:
                label = new_page.get("printed_label")
                label_line = new_lines.get(str(new_page.get("printed_label_line_id")))
                if (
                    old_page.get("printed_label") is not None
                    or not label
                    or label_line is None
                    or normalize_label(label_line.get("text")) != normalize_label(label)
                    or new_page.get("printed_label_source") not in {"header", "footer"}
                ):
                    return False
                changed = True
            for line_id, old_line in old_lines.items():
                new_line = new_lines[line_id]
                old_source = {
                    key: value for key, value in old_line.items() if key not in _LAYOUT_LINE_FIELDS
                }
                new_source = {
                    key: value for key, value in new_line.items() if key not in _LAYOUT_LINE_FIELDS
                }
                if _canonical_common(old_source) != _canonical_common(new_source):
                    return False
                changed |= any(old_line.get(key) != new_line.get(key) for key in _LAYOUT_LINE_FIELDS)
            changed |= [str(line.get("id")) for line in old_page.get("lines") or []] != [
                str(line.get("id")) for line in new_page.get("lines") or []
            ]
    return changed


def _identical_source_lines(oracle: dict[str, Any], candidate: dict[str, Any]) -> bool:
    old_pages = oracle.get("prepared_pages") or []
    new_pages = candidate.get("prepared_pages") or []
    if len(old_pages) != len(new_pages):
        return False
    for old_page, new_page in zip(old_pages, new_pages):
        old_lines = {str(line.get("id")): line for line in old_page.get("lines") or []}
        new_lines = {str(line.get("id")): line for line in new_page.get("lines") or []}
        if set(old_lines) != set(new_lines):
            return False
        for line_id, old_line in old_lines.items():
            new_line = new_lines[line_id]
            for field in ("text", "bbox", "spans", "words", "source", "source_index"):
                if _canonical_common(old_line.get(field)) != _canonical_common(new_line.get(field)):
                    return False
    return True


def _source_supported_product_change(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    evidence = case.get("evidence") or {}
    evidence_path = Path(str(evidence.get("path") or ""))
    if (
        evidence.get("kind") != "canonical-derived"
        or not evidence_path.is_file()
        or sha256(evidence_path) != evidence.get("sha256")
        or candidate.get("validation") != "ok"
        or not _identical_source_lines(oracle, candidate)
    ):
        return False
    old_notes = keyed_notes(oracle.get("footnotes") or [])
    new_notes = keyed_notes(candidate.get("footnotes") or [])
    if not set(old_notes) <= set(new_notes):
        return False
    reference = journal_reference(evidence_path)
    old_metrics = journal_metrics(reference, _common_document(oracle))
    new_metrics = journal_metrics(reference, _common_document(candidate))
    relevant = ("source.cer", "source.wer", "notes.labels.f1", "notes.reference_pages.f1")
    if not any(float(new_metrics[key]) != float(old_metrics[key]) for key in relevant):
        return False
    if not (
        all(
            float(new_metrics[key]) <= float(old_metrics[key])
            for key in ("source.cer", "source.wer")
        )
        and all(
            float(new_metrics[key]) >= float(old_metrics[key])
            for key in ("notes.labels.f1", "notes.reference_pages.f1")
        )
    ):
        return False
    if any(
        float(new_metrics[key]) < float(old_metrics[key])
        for key in ("source.cer", "source.wer")
    ):
        return True
    if set(old_notes) != set(new_notes) or not (
        float(new_metrics["notes.reference_pages.f1"])
        > float(old_metrics["notes.reference_pages.f1"])
    ):
        return False
    reference_fields = {
        "reference_line_id",
        "reference_page",
        "sentence_proposition",
        "passage_since_prior_note",
        "warnings",
    }
    added_reference = False
    for key, old_note in old_notes.items():
        new_note = new_notes[key]
        if _canonical_common(
            {field: value for field, value in old_note.items() if field not in reference_fields}
        ) != _canonical_common(
            {field: value for field, value in new_note.items() if field not in reference_fields}
        ):
            return False
        if old_note.get("reference_page") != new_note.get("reference_page"):
            if old_note.get("reference_page") is not None or new_note.get("reference_page") is None:
                return False
            added_reference = True
    return added_reference


_TABLE_CAPTION = re.compile(r"(?i)^(?:table|tableau)\s+(?:\d+|[ivxlcdm]+)\b")
_SOURCE_NOTE_PREFIX = re.compile(
    r"^\s*(?P<label>\d{1,4}|[*\u2020\u2021\u00a7\u00b6#]{1,8})"
    r"(?:(?i:endnote)\s+(?P=label)|(?=$|[\s.)\],:;-]))"
)
_SOURCE_ATTACHED_SYMBOL = re.compile(r"([*\u2020\u2021\u00a7\u00b6#\uf02a]{1,8})\s*$")


def _table_band_line_ids(page: dict[str, Any]) -> set[str]:
    lines = page.get("lines") or []
    valid = [
        (index, line)
        for index, line in enumerate(lines)
        if len(line.get("bbox") or []) == 4
        and float(line["bbox"][2]) > float(line["bbox"][0])
        and float(line["bbox"][3]) > float(line["bbox"][1])
        and not line.get("exclude_from_body")
    ]
    heights = sorted(float(line["bbox"][3]) - float(line["bbox"][1]) for _, line in valid)
    if not heights:
        return set()
    height = heights[len(heights) // 2]
    rows: defaultdict[int, list[int]] = defaultdict(list)
    for index, line in valid:
        center = (float(line["bbox"][1]) + float(line["bbox"][3])) / 2
        rows[round(center / (height * 0.75))].append(index)
    dense = [
        row
        for row in rows.values()
        if len(row) >= 2
        and statistics.median(len(str(lines[index].get("text") or "").strip()) for index in row)
        <= 24
    ]
    dense.sort(
        key=lambda row: min(
            (float(lines[index]["bbox"][1]) + float(lines[index]["bbox"][3])) / 2
            for index in row
        )
    )
    captions = [line for _, line in valid if _TABLE_CAPTION.match(str(line.get("text") or "").strip())]
    if captions:
        caption_bottom = min(float(line["bbox"][3]) for line in captions)
        dense = [
            row
            for row in dense
            if min(
                (float(lines[index]["bbox"][1]) + float(lines[index]["bbox"][3])) / 2
                for index in row
            )
            >= caption_bottom - height
        ]
        connected: list[list[int]] = []
        prior = None
        for row in dense:
            center = min(
                (float(lines[index]["bbox"][1]) + float(lines[index]["bbox"][3])) / 2
                for index in row
            )
            if prior is not None and center - prior > height * 15:
                break
            connected.append(row)
            prior = center
        dense = connected
    if len(dense) < 3:
        return set()
    columns: Counter[int] = Counter()
    for row in dense:
        columns.update({round(float(lines[index]["bbox"][0]) / (height * 2)) for index in row})
    dense_indexes = {index for row in dense for index in row}
    texts = [str(lines[index].get("text") or "").strip() for index in dense_indexes]
    numeric = sum(
        any(character.isdigit() for character in text)
        and all(
            character.isdigit()
            or character.isspace()
            or character in ".,%()/$-\u2013\u2014"
            for character in text
        )
        for text in texts
    )
    strong = (
        len(dense) >= 6
        and sum(count >= 3 for count in columns.values()) >= 3
        and numeric * 5 >= len(texts)
    )
    if not captions and not strong:
        return set()
    top = min(float(lines[index]["bbox"][1]) for index in dense_indexes)
    bottom = max(float(lines[index]["bbox"][3]) for index in dense_indexes)
    left = min(float(lines[index]["bbox"][0]) for index in dense_indexes)
    right = max(float(lines[index]["bbox"][2]) for index in dense_indexes)
    return {
        str(line.get("id"))
        for _, line in valid
        if float(line["bbox"][3]) >= top - height
        and float(line["bbox"][1]) <= bottom
        and float(line["bbox"][2]) >= left
        and float(line["bbox"][0]) <= right
    }


def _source_note_label(text: Any) -> str | None:
    match = _SOURCE_NOTE_PREFIX.match(str(text or ""))
    return normalize_label(match.group("label")) if match else None


def _note_partition_quality(value: dict[str, Any]) -> tuple[float, int]:
    lines = _line_map(value, "prepared_pages")
    notes = keyed_notes(value.get("footnotes") or [])
    labels = {key[0] for key in notes}
    bounded_labels = {
        key[0]
        for key, note in notes.items()
        if (body_ids := [str(line_id) for line_id in note.get("body_line_ids") or []])
        and _source_note_label(lines.get(body_ids[0], {}).get("text")) == key[0]
    }
    bounded = intrusions = 0
    for key, note in notes.items():
        body_ids = [str(line_id) for line_id in note.get("body_line_ids") or []]
        if body_ids and _source_note_label(lines.get(body_ids[0], {}).get("text")) == key[0]:
            bounded += 1
        for line_id in body_ids:
            label = _source_note_label(lines.get(line_id, {}).get("text"))
            intrusions += int(
                label in labels and label != key[0] and label not in bounded_labels
            )
    return bounded / max(1, len(notes)), intrusions


def _source_note_supported(
    key: tuple[str, int], note: dict[str, Any], lines: dict[str, dict[str, Any]]
) -> bool:
    body_ids = [str(line_id) for line_id in note.get("body_line_ids") or []]
    reference = lines.get(str(note.get("reference_line_id") or ""), {})
    if (
        not body_ids
        or _source_note_label(lines.get(body_ids[0], {}).get("text")) != key[0]
        or sum(character.isalpha() for character in str(note.get("body") or "")) < 4
    ):
        return False
    native = any(
        span.get("superscript") and normalize_label(span.get("text")) == key[0]
        for span in reference.get("spans") or []
    )
    detached = any(
        normalize_label(marker.get("note_id")) == key[0]
        for marker in reference.get("detached_references") or []
    )
    attached = _SOURCE_ATTACHED_SYMBOL.search(str(reference.get("text") or ""))
    return native or detached or (
        attached is not None
        and normalize_label(attached.group(1).replace("\uf02a", "*")) == key[0]
    )


def _credible_endnote_line_ids(value: dict[str, Any]) -> set[str]:
    pages = value.get("prepared_pages") or []
    heading_page = next(
        (
            index
            for index, page in enumerate(pages)
            if any(
                re.fullmatch(r"(?i)(?:end)?notes?", str(line.get("text") or "").strip())
                for line in page.get("lines") or []
            )
        ),
        None,
    )
    if heading_page is None:
        return set()
    lines = _line_map(value, "prepared_pages")
    boundaries = []
    for note in value.get("footnotes") or []:
        body_ids = [str(line_id) for line_id in note.get("body_line_ids") or []]
        if not body_ids:
            continue
        line = lines.get(body_ids[0], {})
        if line.get("note_region_mode") == "endnote" and _source_note_label(line.get("text")):
            boundaries.append((int(line.get("page_index") or 0), int(line.get("reading_order") or 0)))
    if len(boundaries) < 3 or boundaries != sorted(boundaries):
        return set()
    return {
        str(line.get("id"))
        for page in pages[heading_page:]
        for line in page.get("lines") or []
        if line.get("region_type") == "footnote" and line.get("note_region_mode") == "endnote"
    }


def _source_inversions(page: dict[str, Any], line_ids: set[str]) -> int:
    indexes = [
        int(line.get("source_index") or 0)
        for line in sorted(page.get("lines") or [], key=lambda line: int(line.get("reading_order") or 0))
        if str(line.get("id")) in line_ids
    ]
    return sum(left > right for index, left in enumerate(indexes) for right in indexes[index + 1 :])


def _margin_geometry_inversions(page: dict[str, Any], line_ids: set[str]) -> int | None:
    width = float(page.get("width") or 0)
    lines = [
        line
        for line in sorted(
            page.get("lines") or [], key=lambda value: int(value.get("reading_order") or 0)
        )
        if str(line.get("id")) in line_ids
    ]
    if not lines or width <= 0:
        return None
    boxes = [line.get("bbox") or [] for line in lines]
    if any(
        len(box) != 4
        or float(box[2]) <= float(box[0])
        or float(box[3]) <= float(box[1])
        or float(box[2]) - float(box[0]) > width * 0.30
        for box in boxes
    ):
        return None
    centers = [(float(box[0]) + float(box[2])) / 2 for box in boxes]
    if not (all(center <= width * 0.35 for center in centers) or all(center >= width * 0.65 for center in centers)):
        return None
    geometry = [
        ((float(box[1]) + float(box[3])) / 2, (float(box[0]) + float(box[2])) / 2)
        for box in boxes
    ]
    return sum(left > right for index, left in enumerate(geometry) for right in geometry[index + 1 :])


def _source_line_font_size(line: dict[str, Any]) -> float:
    sizes = [
        float(span.get("size") or 0)
        for span in line.get("spans") or []
        if float(span.get("size") or 0) > 0 and not span.get("superscript")
    ]
    return statistics.median(sizes) if sizes else 0.0


def _credible_furniture_line_ids(value: dict[str, Any]) -> set[str]:
    pages_by_key: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
    lines_by_key: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for page in value.get("prepared_pages") or []:
        height = float(page.get("height") or 0)
        for line in page.get("lines") or []:
            region = str(line.get("region_type") or "")
            box = line.get("bbox") or []
            if len(box) != 4 or height <= 0 or region not in {"header", "footer"}:
                continue
            at_edge = (
                float(box[1]) <= height * 0.12
                if region == "header"
                else float(box[3]) >= height * 0.90
            )
            text = re.sub(r"\d+", "#", normalize_text(line.get("text")))
            if not at_edge or not text:
                continue
            key = (region, text)
            pages_by_key[key].add(int(page.get("index") or 0))
            lines_by_key[key].add(str(line.get("id")))
    return {
        line_id
        for key, pages in pages_by_key.items()
        if len(pages) >= 2
        for line_id in lines_by_key[key]
    }


def _looks_like_body_prose(line: dict[str, Any], page: dict[str, Any]) -> bool:
    text = str(line.get("text") or "")
    box = line.get("bbox") or [0.0] * 4
    size = _source_line_font_size(line)
    body_sizes = [
        _source_line_font_size(candidate)
        for candidate in page.get("lines") or []
        if candidate.get("region_type") == "body"
        and sum(character.isalpha() for character in str(candidate.get("text") or "")) >= 4
        and float((candidate.get("bbox") or [0.0] * 4)[2])
        - float((candidate.get("bbox") or [0.0] * 4)[0])
        >= float(page.get("width") or 0) * 0.45
        and _source_line_font_size(candidate) > 0
    ]
    wide_peer = any(
        candidate.get("region_id") == line.get("region_id")
        and float((candidate.get("bbox") or [0.0] * 4)[2])
        - float((candidate.get("bbox") or [0.0] * 4)[0])
        >= float(page.get("width") or 0) * 0.45
        for candidate in page.get("lines") or []
    )
    structured_heading = bool(
        re.match(r"(?i)^(?:[ivxlcdm]+|\d+)[.)]\s+\w", text.strip())
    )
    letters = sum(character.isalpha() for character in text)
    return (
        (letters >= 4 or (letters >= 2 and wide_peer))
        and len(box) == 4
        and body_sizes
        and size >= statistics.median(body_sizes) * 0.90
        and (
            float(box[2]) - float(box[0]) >= float(page.get("width") or 0) * 0.35
            or wide_peer
            or structured_heading
        )
    )


def _source_supported_layout_and_note_partition(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    evidence = case.get("evidence") or {}
    evidence_path = Path(str(evidence.get("path") or ""))
    if (
        evidence.get("kind") != "canonical-derived"
        or not evidence_path.is_file()
        or sha256(evidence_path) != evidence.get("sha256")
        or candidate.get("validation") != "ok"
        or not _identical_source_lines(oracle, candidate)
        or not _only_layout_changes(oracle, candidate)
    ):
        return False
    old_notes = keyed_notes(oracle.get("footnotes") or [])
    new_notes = keyed_notes(candidate.get("footnotes") or [])
    reference = journal_reference(evidence_path)
    expected = {(item["label"], int(item["occurrence"])) for item in reference["pairs"]}
    if expected & set(old_notes) - set(new_notes):
        return False
    candidate_lines = _line_map(candidate, "prepared_pages")
    for key in set(old_notes) - set(new_notes):
        body = normalize_text(old_notes[key].get("body"))
        if key in expected or any(character.isalpha() for character in body):
            return False
    for key in set(new_notes) - set(old_notes):
        if key not in expected and not _source_note_supported(key, new_notes[key], candidate_lines):
            return False
    old_quality = _note_partition_quality(oracle)
    new_quality = _note_partition_quality(candidate)
    if not (
        new_quality[0] >= old_quality[0]
        and new_quality[1] <= old_quality[1]
    ):
        return False
    expected_pages = {
        ((item["label"], int(item["occurrence"])), int(page))
        for item in reference["pairs"]
        for page in item["reference_pages"]
    }
    actual_pages = lambda notes: {
        (key, int(note["reference_page"]))
        for key, note in notes.items()
        if note.get("reference_page") is not None
    }
    if expected_pages & actual_pages(old_notes) - actual_pages(new_notes):
        return False
    endnote_ids = _credible_endnote_line_ids(candidate)
    candidate_note_lines = {
        str(line_id)
        for note in new_notes.values()
        for line_id in note.get("body_line_ids") or []
    }
    source_supported_note_lines = {
        str(line_id)
        for key, note in new_notes.items()
        if _source_note_supported(key, note, candidate_lines)
        for line_id in note.get("body_line_ids") or []
    }
    furniture_ids = _credible_furniture_line_ids(candidate)
    ordinary_note_pages = {
        int(page.get("index") or 0)
        for page in candidate.get("prepared_pages") or []
        if any(
            line.get("region_type") == "footnote"
            and line.get("note_region_mode") == "footnote"
            for line in page.get("lines") or []
        )
    }
    table_gain = prose_gain = endnote_gain = footnote_gain = False
    furniture_gain = False
    for old_page, new_page in zip(
        oracle.get("prepared_pages") or [], candidate.get("prepared_pages") or []
    ):
        old_lines = {str(line.get("id")): line for line in old_page.get("lines") or []}
        new_lines = {str(line.get("id")): line for line in new_page.get("lines") or []}
        table_ids = _table_band_line_ids(new_page)
        old_order = [str(line.get("id")) for line in old_page.get("lines") or []]
        new_order = [str(line.get("id")) for line in new_page.get("lines") or []]
        if old_order != new_order:
            positions = {line_id: index for index, line_id in enumerate(new_order)}
            flipped = {
                line_id
                for index, left in enumerate(old_order)
                for right in old_order[index + 1 :]
                if positions[left] > positions[right]
                for line_id in (left, right)
            }
            if (
                flipped
                and flipped <= table_ids
                and _source_inversions(new_page, table_ids)
                < _source_inversions(old_page, table_ids)
            ):
                table_gain = True
            elif (
                flipped
                and flipped <= source_supported_note_lines
                and _source_inversions(new_page, source_supported_note_lines)
                < _source_inversions(old_page, source_supported_note_lines)
            ):
                footnote_gain = True
            elif (
                flipped & source_supported_note_lines
                and (new_margin := _margin_geometry_inversions(new_page, flipped))
                is not None
                and (old_margin := _margin_geometry_inversions(old_page, flipped))
                is not None
                and new_margin < old_margin
            ):
                footnote_gain = True
            else:
                return False
        for line_id, old_line in old_lines.items():
            new_line = new_lines[line_id]
            old_layout = tuple(old_line.get(field) for field in _LAYOUT_LINE_FIELDS - {"reading_order", "region_id"})
            new_layout = tuple(new_line.get(field) for field in _LAYOUT_LINE_FIELDS - {"reading_order", "region_id"})
            if old_layout == new_layout:
                continue
            if line_id in table_ids and (
                (new_line.get("region_type") == "body" and not new_line.get("note_region_mode"))
                or (
                    new_line.get("region_type") == "footnote"
                    and _source_note_label(new_line.get("text"))
                    and not str(_source_note_label(new_line.get("text"))).isdigit()
                )
            ):
                table_gain = True
            elif table_ids and _TABLE_CAPTION.match(str(new_line.get("text") or "").strip()):
                table_gain = True
            elif endnote_ids and re.fullmatch(
                r"(?i)(?:end)?notes?", str(new_line.get("text") or "").strip()
            ) and new_line.get("region_type") == "heading":
                endnote_gain = True
            elif line_id in endnote_ids:
                endnote_gain = True
            elif (
                line_id in candidate_note_lines
                and old_line.get("note_region_mode") == "endnote"
                and new_line.get("region_type") == "footnote"
                and new_line.get("note_region_mode") == "footnote"
            ):
                footnote_gain = True
            elif (
                line_id in source_supported_note_lines
                and old_line.get("region_type") in {"body", "footer"}
                and new_line.get("region_type") == "footnote"
                and new_line.get("note_region_mode") == "footnote"
            ):
                footnote_gain = True
            elif (
                line_id in furniture_ids
                and old_line.get("region_type") not in {"header", "footer"}
                and new_line.get("region_type") in {"header", "footer"}
            ):
                furniture_gain = True
            elif (
                old_line.get("note_region_mode") == "endnote"
                and line_id not in candidate_note_lines
                and int(new_page.get("index") or 0) in ordinary_note_pages
                and (
                    new_line.get("region_type") in {"body", "heading"}
                    or line_id in furniture_ids
                )
            ):
                prose_gain = True
            elif (
                old_line.get("region_type") in {"footnote", "heading"}
                and new_line.get("region_type") == "body"
                and not new_line.get("note_region_mode")
                and line_id not in candidate_note_lines
                and _looks_like_body_prose(new_line, new_page)
            ):
                prose_gain = True
            else:
                return False
    old_metrics = journal_metrics(reference, _common_document(oracle))
    new_metrics = journal_metrics(reference, _common_document(candidate))
    text_worse = any(
        float(new_metrics[key]) > float(old_metrics[key]) for key in ("source.cer", "source.wer")
    )
    return (
        table_gain
        or prose_gain
        or endnote_gain
        or footnote_gain
        or furniture_gain
    ) and (not text_worse or table_gain)


def _page_column_switches(page: dict[str, Any]) -> int | None:
    width = float(page.get("width") or 0)
    if width <= 0:
        return None
    split = width / 2
    lines = []
    for line in sorted(page.get("lines") or [], key=lambda value: int(value.get("reading_order") or 0)):
        box = line.get("bbox") or [0.0] * 4
        if (
            len(box) < 4
            or box[2] <= box[0]
            or box[3] <= box[1]
            or line.get("exclude_from_body")
            or line.get("region_type") in {"header", "footer"}
            or (box[2] - box[0]) / width > 0.55
            or box[0] < split < box[2]
        ):
            continue
        lines.append((int((box[0] + box[2]) / 2 >= split), float(box[1]), float(box[3])))
    sides = [[line for line in lines if line[0] == side] for side in (0, 1)]
    if any(len(side) < 3 for side in sides):
        return None
    span = max(max(line[2] for line in side) for side in sides) - min(
        min(line[1] for line in side) for side in sides
    )
    overlap = min(max(line[2] for line in side) for side in sides) - max(
        min(line[1] for line in side) for side in sides
    )
    if span <= 0 or max(0.0, overlap) / span < 0.20:
        return None
    return sum(left[0] != right[0] for left, right in zip(lines, lines[1:]))


def _credible_column_layout_change(
    oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    for old_page, new_page in zip(
        oracle.get("prepared_pages") or [], candidate.get("prepared_pages") or []
    ):
        old_order = [str(line.get("id")) for line in old_page.get("lines") or []]
        new_order = [str(line.get("id")) for line in new_page.get("lines") or []]
        if old_order == new_order:
            continue
        old_switches = _page_column_switches(old_page)
        new_switches = _page_column_switches(new_page)
        if new_switches is not None and (
            new_switches <= 2
            or (old_switches is not None and new_switches < old_switches)
        ):
            return True
    return False


def _source_supported_column_layout(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> bool:
    evidence = case.get("evidence") or {}
    evidence_path = Path(str(evidence.get("path") or ""))
    if (
        evidence.get("kind") != "canonical-derived"
        or not evidence_path.is_file()
        or sha256(evidence_path) != evidence.get("sha256")
        or candidate.get("validation") != "ok"
        or not _only_layout_changes(oracle, candidate)
        or not _credible_column_layout_change(oracle, candidate)
    ):
        return False
    old_notes = keyed_notes(oracle.get("footnotes") or [])
    new_notes = keyed_notes(candidate.get("footnotes") or [])
    if not set(old_notes) <= set(new_notes):
        return False
    reference = journal_reference(evidence_path)
    old_metrics = journal_metrics(reference, _common_document(oracle))
    new_metrics = journal_metrics(reference, _common_document(candidate))
    return (
        all(float(new_metrics[key]) < float(old_metrics[key]) for key in ("source.cer", "source.wer"))
        and all(
            float(new_metrics[key]) >= float(old_metrics[key])
            for key in ("notes.labels.f1", "notes.reference_pages.f1")
        )
    )


def common_input_qualification(
    case: dict[str, Any], oracle: dict[str, Any], candidate: dict[str, Any]
) -> tuple[list[str], list[str]]:
    """Return unresolved regressions and separately named proven improvements."""
    failures = common_input_regressions(oracle, candidate)
    if not failures:
        return [], []
    remaining = list(failures)
    improvements: list[str] = []
    trace_improvements = (
        (
            _CANDIDATE_COUNT_DIFFERENCE,
            _unused_candidate_pruning,
            "stricter-unused-label-candidate-pruning",
        ),
        (
            _CANDIDATE_COUNT_DIFFERENCE,
            _unused_candidate_growth,
            "non-product-unused-label-candidate-accounting",
        ),
        (
            _NATIVE_SUPERSCRIPT_EVIDENCE,
            _richer_native_superscript_evidence,
            "native-superscript-evidence-over-geometric-fallback",
        ),
    )
    for pattern, qualifies, name in trace_improvements:
        matching = [failure for failure in remaining if pattern.fullmatch(failure)]
        if matching and qualifies(matching):
            remaining = [failure for failure in remaining if failure not in matching]
            improvements.append(name)
    malformed = [
        failure for failure in remaining if _MALFORMED_LABEL_SUPPRESSION.fullmatch(failure)
    ]
    if malformed and _rejected_malformed_label_prefixes(malformed, oracle, candidate):
        remaining = [failure for failure in remaining if failure not in malformed]
        improvements.append("rejected-malformed-label-prefixes")
    if not remaining:
        return [], improvements
    if _preserved_compact_note_furniture(case, oracle, candidate):
        return [], improvements + ["preserved-compact-note-lines-from-footer-furniture"]
    if _attached_detached_note_labels(case, oracle, candidate):
        return [], improvements + ["attached-detached-note-labels-to-their-lines"]
    if _source_supported_column_layout(case, oracle, candidate):
        return [], improvements + ["source-supported-column-reading-order-and-notes"]
    if _source_supported_layout_and_note_partition(case, oracle, candidate):
        return [], improvements + ["source-supported-layout-and-note-partition"]
    if _source_supported_product_change(case, oracle, candidate):
        return [], improvements + ["source-supported-structure-and-note-recovery"]
    return remaining, improvements


def _valid_common_file(path: Path, source_hash: str, schema: str) -> bool:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    valid = (
        value.get("schema_version") == schema
        and value.get("source_sha256") == source_hash
    )
    if schema == "legalpdf.common-input.v1":
        valid = valid and all(
            field in value
            for field in ("source_name", "pages", "separators", "metadata")
        )
    return valid


def _extraction_text(value: Any) -> str:
    return str(value or "")


def _squash_whitespace(value: Any) -> str:
    return "".join(_extraction_text(value).split())


def _oracle_projection_differences(
    oracle: Any,
    candidate: Any,
    path: str = "",
    *,
    example_limit: int = 100,
) -> tuple[int, list[str]]:
    """Compare every oracle value while permitting candidate-only object fields."""
    count = 0
    examples: list[str] = []

    def mismatch(message: str) -> None:
        nonlocal count
        count += 1
        if len(examples) < example_limit:
            examples.append(message)

    def visit(left: Any, right: Any, current: str) -> None:
        if isinstance(left, dict):
            if not isinstance(right, dict):
                mismatch(f"{current}: expected object, got {type(right).__name__}")
                return
            for key in sorted(left):
                child = f"{current}/{key}"
                if key not in right:
                    mismatch(f"{child}: missing from Rust")
                else:
                    visit(left[key], right[key], child)
            return
        if isinstance(left, list):
            if not isinstance(right, list):
                mismatch(f"{current}: expected array, got {type(right).__name__}")
                return
            if len(left) != len(right):
                mismatch(f"{current}/length: {len(left)} -> {len(right)}")
            for index, (left_item, right_item) in enumerate(zip(left, right)):
                visit(left_item, right_item, f"{current}/{index}")
            return
        if type(left) is not type(right) or left != right:
            mismatch(f"{current}: {left!r} -> {right!r}")

    visit(oracle, candidate, path)
    return count, examples


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": percentile(ordered, 0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _line_evidence(line: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": line.get("id"),
        "text": line.get("text"),
        "bbox": line.get("bbox"),
        "block_index": line.get("block_index"),
        "source_index": line.get("source_index"),
        "reading_order": line.get("reading_order"),
        "spans": [
            {
                key: span.get(key)
                for key in (
                    "text",
                    "start",
                    "end",
                    "bbox",
                    "font",
                    "size",
                    "flags",
                    "superscript",
                )
            }
            for span in (line.get("spans") or [])[:12]
        ],
        "words": [
            {key: word.get(key) for key in ("text", "start", "end", "bbox")}
            for word in (line.get("words") or [])[:12]
        ],
    }


def _line_alignment(
    oracle_lines: Sequence[dict[str, Any]], candidate_lines: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    left = [_squash_whitespace(line.get("text")) for line in oracle_lines]
    right = [_squash_whitespace(line.get("text")) for line in candidate_lines]
    result: list[dict[str, Any]] = []
    for operation, a0, a1, b0, b1 in difflib.SequenceMatcher(
        a=left, b=right, autojunk=False
    ).get_opcodes():
        if operation == "equal":
            for left_index, right_index in zip(range(a0, a1), range(b0, b1)):
                raw_equal = _extraction_text(oracle_lines[left_index].get("text")) == _extraction_text(
                    candidate_lines[right_index].get("text")
                )
                result.append(
                    {
                        "kind": "exact" if raw_equal else "whitespace_only",
                        "oracle": [left_index],
                        "candidate": [right_index],
                    }
                )
            continue
        if operation == "delete":
            result.append(
                {"kind": "missing", "oracle": list(range(a0, a1)), "candidate": []}
            )
            continue
        if operation == "insert":
            result.append(
                {"kind": "extra", "oracle": [], "candidate": list(range(b0, b1))}
            )
            continue
        left_joined = "".join(left[a0:a1])
        right_joined = "".join(right[b0:b1])
        if left_joined == right_joined:
            kind = (
                "merge"
                if a1 - a0 > b1 - b0
                else "split"
                if a1 - a0 < b1 - b0
                else "resegmented"
            )
            result.append(
                {
                    "kind": kind,
                    "oracle": list(range(a0, a1)),
                    "candidate": list(range(b0, b1)),
                }
            )
        elif a1 - a0 == b1 - b0:
            for left_index, right_index in zip(range(a0, a1), range(b0, b1)):
                result.append(
                    {
                        "kind": "text_change",
                        "oracle": [left_index],
                        "candidate": [right_index],
                    }
                )
        else:
            result.append(
                {
                    "kind": "content_and_segmentation_change",
                    "oracle": list(range(a0, a1)),
                    "candidate": list(range(b0, b1)),
                }
            )
    return result


def _unit_violations(line: dict[str, Any], field: str) -> int:
    text = _extraction_text(line.get("text"))
    prior_end = 0
    violations = 0
    for unit in line.get(field) or []:
        start = unit.get("start")
        end = unit.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            violations += 1
            continue
        if start < prior_end or start < 0 or end < start or end > len(text):
            violations += 1
        elif text[start:end] != _extraction_text(unit.get("text")):
            violations += 1
        prior_end = max(prior_end, end)
    return violations


def extraction_contract_diagnostics(
    oracle: dict[str, Any], candidate: dict[str, Any], *, case_id: str = "case"
) -> dict[str, Any]:
    issues: Counter[str] = Counter()
    observations: Counter[str] = Counter()
    measurements: defaultdict[str, list[float]] = defaultdict(list)
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    evidence_cache: dict[int, dict[str, Any]] = {}

    def line_evidence(line: dict[str, Any]) -> dict[str, Any]:
        key = id(line)
        if key not in evidence_cache:
            evidence_cache[key] = _line_evidence(line)
        return evidence_cache[key]

    def issue(name: str, count: int = 1, evidence: dict[str, Any] | None = None) -> None:
        issues[name] += count
        if evidence is not None and len(examples[name]) < 5:
            examples[name].append({"case_id": case_id, **evidence})

    projection_difference_count, projection_differences = (
        _oracle_projection_differences(oracle, candidate)
    )
    observations["contract.oracle_projection_differences"] = (
        projection_difference_count
    )
    if projection_difference_count:
        issue(
            "contract.oracle_projection_mismatch",
            evidence={
                "difference_count": projection_difference_count,
                "differences": projection_differences,
            },
        )

    oracle_pages = oracle.get("pages") or []
    candidate_pages = candidate.get("pages") or []
    observations["documents"] = 1
    for field in ("schema_version", "source_name", "source_sha256"):
        if oracle.get(field) != candidate.get(field):
            issue(
                f"document.{field}_mismatch",
                evidence={"oracle": oracle.get(field), "candidate": candidate.get(field)},
            )
    observations["oracle.pages"] = len(oracle_pages)
    observations["candidate.pages"] = len(candidate_pages)
    if len(oracle_pages) != len(candidate_pages):
        issue(
            "page.count_mismatch",
            abs(len(oracle_pages) - len(candidate_pages)),
            {"oracle": len(oracle_pages), "candidate": len(candidate_pages)},
        )

    oracle_separators = oracle.get("separators") or []
    candidate_separators = candidate.get("separators") or []
    for page_index in range(max(len(oracle_pages), len(candidate_pages))):
        if page_index >= len(oracle_pages) or page_index >= len(candidate_pages):
            continue
        oracle_page = oracle_pages[page_index]
        candidate_page = candidate_pages[page_index]
        page_number = page_index + 1
        observations["pages.compared"] += 1
        for field in (
            "id",
            "index",
            "number",
            "regions",
            "source",
            "printed_label",
            "printed_label_source",
            "printed_label_line_id",
        ):
            if oracle_page.get(field) != candidate_page.get(field):
                issue(
                    f"page.{field}_mismatch",
                    evidence={
                        "page": page_number,
                        "oracle": oracle_page.get(field),
                        "candidate": candidate_page.get(field),
                    },
                )
        for field in ("width", "height"):
            left = float(oracle_page.get(field) or 0.0)
            right = float(candidate_page.get(field) or 0.0)
            delta = abs(left - right)
            measurements[f"page.{field}.absolute_error"].append(delta)
            if delta > 1e-9:
                issue(
                    f"page.{field}_mismatch",
                    evidence={"page": page_number, "oracle": left, "candidate": right},
                )
        quality_delta = abs(
            float(oracle_page.get("text_quality") or 0.0)
            - float(candidate_page.get("text_quality") or 0.0)
        )
        measurements["page.text_quality.absolute_error"].append(quality_delta)
        if quality_delta > 1e-9:
            issue("page.text_quality_mismatch", evidence={"page": page_number})

        oracle_lines = oracle_page.get("lines") or []
        candidate_lines = candidate_page.get("lines") or []
        observations["oracle.lines"] += len(oracle_lines)
        observations["candidate.lines"] += len(candidate_lines)
        oracle_text = "\n".join(_extraction_text(line.get("text")) for line in oracle_lines)
        candidate_text = "\n".join(
            _extraction_text(line.get("text")) for line in candidate_lines
        )
        if oracle_text == candidate_text:
            observations["page_text.exact"] += 1
        elif _squash_whitespace(oracle_text) == _squash_whitespace(candidate_text):
            issue("page_text.whitespace_only", evidence={"page": page_number})
        else:
            issue(
                "page_text.content_mismatch",
                evidence={
                    "page": page_number,
                    "oracle_chars": len(oracle_text),
                    "candidate_chars": len(candidate_text),
                },
            )
        observations["page_text.oracle_chars"] += len(oracle_text)
        observations["page_text.candidate_chars"] += len(candidate_text)
        observations["page_text.character_edits"] += sequence_error(
            oracle_text, candidate_text
        )
        observations["page_text.word_edits"] += sequence_error(
            oracle_text.split(), candidate_text.split()
        )
        if Counter(map(_squash_whitespace, (line.get("text") for line in oracle_lines))) == Counter(
            map(_squash_whitespace, (line.get("text") for line in candidate_lines))
        ) and [
            _squash_whitespace(line.get("text")) for line in oracle_lines
        ] != [
            _squash_whitespace(line.get("text")) for line in candidate_lines
        ]:
            issue("line.order_mismatch", evidence={"page": page_number})

        alignment = _line_alignment(oracle_lines, candidate_lines)
        comparable_pairs: list[tuple[int, int]] = []
        for group in alignment:
            kind = str(group["kind"])
            observations[f"line_alignment.{kind}"] += 1
            if kind not in {"exact"}:
                issue(
                    f"line.{kind}",
                    evidence={
                        "page": page_number,
                        "oracle": [
                            line_evidence(oracle_lines[index])
                            for index in group["oracle"][:4]
                        ],
                        "candidate": [
                            line_evidence(candidate_lines[index])
                            for index in group["candidate"][:4]
                        ],
                    },
                )
            if len(group["oracle"]) == len(group["candidate"]) == 1 and kind in {
                "exact",
                "whitespace_only",
            }:
                comparable_pairs.append((group["oracle"][0], group["candidate"][0]))

        previous_pair: tuple[int, int] | None = None
        for oracle_index, candidate_index in comparable_pairs:
            left = oracle_lines[oracle_index]
            right = candidate_lines[candidate_index]
            observations["lines.field_comparable"] += 1
            for field in (
                "id",
                "page_index",
                "page_number",
                "source_index",
                "reading_order",
                "block_index",
                "detached_references",
                "exclude_from_body",
                "suppress_footnote_label",
                "note_region_mode",
                "region_id",
                "region_type",
                "source",
            ):
                if left.get(field) != right.get(field):
                    issue(
                        f"line.{field}_mismatch",
                        evidence={
                            "page": page_number,
                            "oracle": line_evidence(left),
                            "candidate": line_evidence(right),
                        },
                    )
            if previous_pair is not None:
                prior_oracle, prior_candidate = previous_pair
                if oracle_index == prior_oracle + 1 and candidate_index == prior_candidate + 1:
                    oracle_boundary = left.get("block_index") != oracle_lines[prior_oracle].get(
                        "block_index"
                    )
                    candidate_boundary = right.get("block_index") != candidate_lines[
                        prior_candidate
                    ].get("block_index")
                    observations["block.boundaries_compared"] += 1
                    if oracle_boundary and not candidate_boundary:
                        issue(
                            "block.boundary_missing",
                            evidence={
                                "page": page_number,
                                "oracle": line_evidence(left),
                                "candidate": line_evidence(right),
                            },
                        )
                    elif candidate_boundary and not oracle_boundary:
                        issue(
                            "block.boundary_extra",
                            evidence={
                                "page": page_number,
                                "oracle": line_evidence(left),
                                "candidate": line_evidence(right),
                            },
                        )
                    else:
                        observations["block.boundary_exact"] += 1
            previous_pair = (oracle_index, candidate_index)

            left_bbox = left.get("bbox") or []
            right_bbox = right.get("bbox") or []
            if len(left_bbox) != 4 or len(right_bbox) != 4:
                issue(
                    "line.bbox_contract_invalid",
                    evidence={
                        "page": page_number,
                        "oracle": line_evidence(left),
                        "candidate": line_evidence(right),
                    },
                )
            for coordinate, left_value, right_value in zip(
                ("x0", "y0", "x1", "y1"), left_bbox, right_bbox
            ):
                delta = abs(float(left_value) - float(right_value))
                measurements[f"line.bbox.{coordinate}.absolute_error"].append(delta)
                if delta > 1e-9:
                    issue(
                        f"line.bbox.{coordinate}_mismatch",
                        evidence={
                            "page": page_number,
                            "absolute_error": delta,
                            "oracle": line_evidence(left),
                            "candidate": line_evidence(right),
                        },
                    )

            for field in ("spans", "words"):
                left_units = left.get(field) or []
                right_units = right.get(field) or []
                left_violations = _unit_violations(left, field)
                right_violations = _unit_violations(right, field)
                if left_violations:
                    issue(f"oracle.{field}.offset_contract_invalid", left_violations)
                if right_violations:
                    issue(
                        f"candidate.{field}.offset_contract_invalid",
                        right_violations,
                        {"page": page_number, "candidate": line_evidence(right)},
                    )
                left_signature = [
                    (unit.get("text"), unit.get("start"), unit.get("end"))
                    for unit in left_units
                ]
                right_signature = [
                    (unit.get("text"), unit.get("start"), unit.get("end"))
                    for unit in right_units
                ]
                observations[f"{field}.lines_compared"] += 1
                if left_signature != right_signature:
                    issue(
                        f"{field}.segmentation_mismatch",
                        evidence={
                            "page": page_number,
                            "oracle": line_evidence(left),
                            "candidate": line_evidence(right),
                        },
                    )
                else:
                    observations[f"{field}.segmentation_exact"] += 1
                for left_unit, right_unit in zip(left_units, right_units):
                    if (
                        left_unit.get("text"),
                        left_unit.get("start"),
                        left_unit.get("end"),
                    ) != (
                        right_unit.get("text"),
                        right_unit.get("start"),
                        right_unit.get("end"),
                    ):
                        continue
                    observations[f"{field}.units_comparable"] += 1
                    if left_unit.get("id") != right_unit.get("id"):
                        issue(
                            f"{field}.id_mismatch",
                            evidence={
                                "page": page_number,
                                "oracle": left_unit.get("id"),
                                "candidate": right_unit.get("id"),
                            },
                        )
                    left_unit_bbox = left_unit.get("bbox") or []
                    right_unit_bbox = right_unit.get("bbox") or []
                    if len(left_unit_bbox) != 4 or len(right_unit_bbox) != 4:
                        issue(
                            f"{field}.bbox_contract_invalid",
                            evidence={
                                "page": page_number,
                                "oracle": left_unit_bbox,
                                "candidate": right_unit_bbox,
                                "oracle_line": line_evidence(left),
                                "candidate_line": line_evidence(right),
                            },
                        )
                    for coordinate, left_value, right_value in zip(
                        ("x0", "y0", "x1", "y1"),
                        left_unit_bbox,
                        right_unit_bbox,
                    ):
                        delta = abs(float(left_value) - float(right_value))
                        measurements[f"{field}.bbox.{coordinate}.absolute_error"].append(
                            delta
                        )
                        if delta > 1e-9:
                            issue(
                                f"{field}.bbox.{coordinate}_mismatch",
                                evidence={
                                    "page": page_number,
                                    "absolute_error": delta,
                                    "oracle_line": line_evidence(left),
                                    "candidate_line": line_evidence(right),
                                },
                            )
                    if field == "spans":
                        for attribute in ("font", "flags", "superscript"):
                            if left_unit.get(attribute) != right_unit.get(attribute):
                                issue(
                                    f"spans.{attribute}_mismatch",
                                    evidence={
                                        "page": page_number,
                                        "oracle": left_unit.get(attribute),
                                        "candidate": right_unit.get(attribute),
                                        "oracle_line": line_evidence(left),
                                        "candidate_line": line_evidence(right),
                                    },
                                )
                        size_delta = abs(
                            float(left_unit.get("size") or 0.0)
                            - float(right_unit.get("size") or 0.0)
                        )
                        measurements["spans.size.absolute_error"].append(size_delta)
                        if size_delta > 1e-9:
                            issue(
                                "spans.size_mismatch",
                                evidence={
                                    "page": page_number,
                                    "absolute_error": size_delta,
                                    "oracle": left_unit.get("size"),
                                    "candidate": right_unit.get("size"),
                                    "oracle_line": line_evidence(left),
                                    "candidate_line": line_evidence(right),
                                },
                            )

        left_separator = (
            oracle_separators[page_index]
            if page_index < len(oracle_separators)
            else None
        )
        right_separator = (
            candidate_separators[page_index]
            if page_index < len(candidate_separators)
            else None
        )
        if left_separator is None and right_separator is None:
            observations["separator.both_missing"] += 1
        elif left_separator is None:
            issue(
                "separator.extra",
                evidence={"page": page_number, "candidate": right_separator},
            )
        elif right_separator is None:
            issue(
                "separator.missing",
                evidence={"page": page_number, "oracle": left_separator},
            )
        else:
            delta = abs(float(left_separator) - float(right_separator))
            measurements["separator.absolute_error"].append(delta)
            if delta > 1e-9:
                issue(
                    "separator.position_mismatch",
                    evidence={
                        "page": page_number,
                        "oracle": left_separator,
                        "candidate": right_separator,
                        "absolute_error": delta,
                    },
                )
            else:
                observations["separator.exact"] += 1

    oracle_metadata = oracle.get("metadata") or {}
    candidate_metadata = candidate.get("metadata") or {}
    for key, value in oracle_metadata.items():
        if key not in candidate_metadata:
            issue("metadata.source_key_missing", evidence={"key": key, "oracle": value})
        elif candidate_metadata[key] != value:
            issue(
                "metadata.source_value_mismatch",
                evidence={"key": key, "oracle": value, "candidate": candidate_metadata[key]},
            )
        else:
            observations["metadata.source_value_exact"] += 1
    observations["metadata.candidate_extra_keys"] += len(
        set(candidate_metadata) - set(oracle_metadata)
    )

    return {
        "case_id": case_id,
        "passed": not issues,
        "issue_count": sum(issues.values()),
        "issues": dict(sorted(issues.items())),
        "observations": dict(sorted(observations.items())),
        "measurements": {
            key: _distribution(values) for key, values in sorted(measurements.items())
        },
        "_measurement_values": dict(measurements),
        "examples": dict(sorted(examples.items())),
    }


def run_extraction_autopsy(arguments: argparse.Namespace) -> int:
    oracle_root = arguments.oracle_root.resolve()
    rust_binary = arguments.rust_binary.resolve()
    if _git_head(oracle_root) != ORACLE_COMMIT:
        raise ValueError("extraction autopsy requires the frozen oracle commit")
    if not rust_binary.is_file():
        raise FileNotFoundError(rust_binary)
    cases = selected_cases(arguments.manifest.resolve(), arguments.split)
    output = arguments.output.resolve()
    rust_identity = sha256(rust_binary)
    oracle_output = output / "oracle" / ORACLE_COMMIT[:12]
    rust_output = output / "rust" / rust_identity[:12]
    result_output = output / "cases"
    helper = Path(__file__).with_name("oracle_replay.py").resolve()
    oracle_env = os.environ.copy()
    oracle_env["PYTHONIOENCODING"] = "utf-8"
    oracle_env["PYTHONPATH"] = os.pathsep.join(
        [str(oracle_root / "src"), oracle_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    rows_by_index: dict[int, dict[str, Any]] = {}
    all_issues: Counter[str] = Counter()
    all_observations: Counter[str] = Counter()
    all_measurements: defaultdict[str, list[float]] = defaultdict(list)
    all_examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    def inspect_case(
        index: int, case: dict[str, Any]
    ) -> tuple[int, dict[str, Any], dict[str, list[float]]]:
        case_id = str(case["case_id"])
        source_hash = str(case["pdf_sha256"])
        token = manifest_case_token(case_id)
        oracle_path = oracle_output / f"{token}.json"
        rust_path = rust_output / f"{token}.json"
        failures: list[str] = []
        commands = (
            (
                oracle_path,
                [
                    sys.executable,
                    str(helper),
                    "extract",
                    str(Path(case["pdf"])),
                    "--output",
                    str(oracle_path),
                ],
                oracle_root,
                oracle_env,
            ),
            (
                rust_path,
                [
                    str(rust_binary),
                    "_parity-extract",
                    str(Path(case["pdf"])),
                    "--output",
                    str(rust_path),
                ],
                rust_binary.parent,
                os.environ.copy(),
            ),
        )
        try:
            for name, (path, command, cwd, environment) in zip(
                ("oracle", "rust"), commands
            ):
                if _valid_common_file(path, source_hash, "legalpdf.common-input.v1"):
                    continue
                measured = run_child(
                    command, cwd=cwd, env=environment, timeout=arguments.timeout
                )
                if measured["returncode"] != 0 or not _valid_common_file(
                    path, source_hash, "legalpdf.common-input.v1"
                ):
                    failures.append(
                        f"{name} extraction failed: "
                        f"{measured['stderr'] or measured['stdout']}"
                    )
                    break
            if failures:
                row = {
                    "case_id": case_id,
                    "passed": False,
                    "issue_count": len(failures),
                    "issues": {"extraction.failed": len(failures)},
                    "failures": failures,
                }
            else:
                row = extraction_contract_diagnostics(
                    json.loads(oracle_path.read_text(encoding="utf-8")),
                    json.loads(rust_path.read_text(encoding="utf-8")),
                    case_id=case_id,
                )
        except Exception as error:  # preserve the rest of a corpus run
            row = {
                "case_id": case_id,
                "passed": False,
                "issue_count": 1,
                "issues": {"extraction.failed": 1},
                "failures": [f"{type(error).__name__}: {error}"],
            }
        measurement_values = row.pop("_measurement_values", {})
        atomic_json(result_output / f"{token}.json", row)
        return index, row, measurement_values

    jobs = min(max(1, arguments.jobs), max(1, len(cases)))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(inspect_case, index, case)
            for index, case in enumerate(cases)
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            index, row, measurement_values = future.result()
            rows_by_index[index] = row
            case_id = str(row["case_id"])
            for key, value in (row.get("issues") or {}).items():
                all_issues[key] += int(value)
            for key, value in (row.get("observations") or {}).items():
                all_observations[key] += int(value)
            for key, values in measurement_values.items():
                all_measurements[key].extend(float(value) for value in values)
            for key, values in (row.get("examples") or {}).items():
                all_examples[key].extend(
                    values[: max(0, 10 - len(all_examples[key]))]
                )
            print(
                f"{completed_count}/{len(cases)} {case_id}: "
                f"{row['issue_count']} extraction-contract issues",
                flush=True,
            )
    rows = [rows_by_index[index] for index in range(len(cases))]
    report = {
        "schema_version": "legalpdf.extraction-autopsy.v1",
        "oracle_commit": ORACLE_COMMIT,
        "rust_sha256": rust_identity,
        "case_count": len(rows),
        "passed": bool(rows) and not all_issues,
        "issue_count": sum(all_issues.values()),
        "issues": dict(all_issues.most_common()),
        "observations": dict(sorted(all_observations.items())),
        "measurements": {
            key: _distribution(values) for key, values in sorted(all_measurements.items())
        },
        "examples": dict(sorted(all_examples.items())),
        "case_results": str((output / "cases.jsonl").resolve()),
    }
    atomic_jsonl(output / "cases.jsonl", rows)
    atomic_json(output / "report.json", report)
    print(f"extraction autopsy: {output / 'report.json'}", flush=True)
    return 0 if report["passed"] else 1


def run_common_input(arguments: argparse.Namespace) -> int:
    oracle_root = arguments.oracle_root.resolve()
    rust_binary = arguments.rust_binary.resolve()
    if _git_head(oracle_root) != ORACLE_COMMIT:
        raise ValueError("common-input replay requires the frozen oracle commit")
    if not rust_binary.is_file():
        raise FileNotFoundError(rust_binary)
    cases = selected_case_ids(
        selected_cases(arguments.manifest.resolve(), arguments.split), arguments.case
    )
    output = arguments.output.resolve()
    rust_identity = sha256(rust_binary)
    run_root = output / f"{ORACLE_COMMIT[:12]}-{rust_identity[:12]}"
    input_root = output / "inputs" / ORACLE_COMMIT[:12]
    oracle_root_out = output / "oracle" / ORACLE_COMMIT[:12]
    rust_root_out = run_root / "rust"
    result_root = run_root / "results"
    helper = Path(__file__).with_name("oracle_replay.py").resolve()
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join(
        [str(oracle_root / "src"), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    rows_by_index: dict[int, dict[str, Any]] = {}
    index_path = run_root / "common-input.jsonl"

    def replay_case(index: int, case: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        case_id = str(case["case_id"])
        source_hash = str(case["pdf_sha256"])
        token = manifest_case_token(case_id)
        common_path = input_root / f"{token}.json"
        oracle_path = oracle_root_out / f"{token}.json"
        rust_path = rust_root_out / f"{token}.json"
        result_path = result_root / f"{token}.json"
        failures: list[str] = []
        commands = (
            (
                common_path,
                "legalpdf.common-input.v1",
                [
                    sys.executable,
                    str(helper),
                    "extract",
                    str(Path(case["pdf"])),
                    "--output",
                    str(common_path),
                ],
                oracle_root,
                env,
            ),
            (
                oracle_path,
                "legalpdf.common-input-result.v1",
                [
                    sys.executable,
                    str(helper),
                    "replay",
                    str(common_path),
                    "--output",
                    str(oracle_path),
                ],
                oracle_root,
                env,
            ),
            (
                rust_path,
                "legalpdf.common-input-result.v1",
                [
                    str(rust_binary),
                    "_parity-replay",
                    str(common_path),
                    "--output",
                    str(rust_path),
                ],
                rust_binary.parent,
                os.environ.copy(),
            ),
        )
        timings: dict[str, float] = {}
        try:
            for position, (path, schema, command, cwd, command_env) in enumerate(commands):
                name = ("extract", "oracle", "rust")[position]
                if _valid_common_file(path, source_hash, schema):
                    continue
                measured = run_child(
                    command, cwd=cwd, env=command_env, timeout=arguments.timeout
                )
                timings[name] = float(measured["wall_seconds"])
                if measured["returncode"] != 0 or not _valid_common_file(
                    path, source_hash, schema
                ):
                    failures.append(
                        f"{name} failed: {measured['stderr'] or measured['stdout']}"
                    )
                    break
        except Exception as error:
            failures.append(f"replay failed: {type(error).__name__}: {error}")
        try:
            if not failures:
                oracle = json.loads(oracle_path.read_text(encoding="utf-8"))
                rust = json.loads(rust_path.read_text(encoding="utf-8"))
                failures, improvements = common_input_qualification(case, oracle, rust)
            else:
                improvements = []
        except Exception as error:  # preserve the rest of a corpus run
            failures = [f"qualification failed: {type(error).__name__}: {error}"]
            improvements = []
        row = {
            "schema_version": "legalpdf.common-input-run.v1",
            "case_id": case_id,
            "pdf_sha256": source_hash,
            "oracle_commit": ORACLE_COMMIT,
            "rust_sha256": rust_identity,
            "common_input": str(common_path),
            "oracle_result": str(oracle_path),
            "rust_result": str(rust_path),
            "timings": timings,
            "passed": not failures,
            "failure_count": len(failures),
            "failures": failures,
            "improvement_count": len(improvements),
            "improvements": improvements,
        }
        atomic_json(result_path, row)
        return index, row

    jobs = min(max(1, arguments.jobs), max(1, len(cases)))
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(replay_case, index, case)
            for index, case in enumerate(cases)
        ]
        for completed_count, future in enumerate(as_completed(futures), start=1):
            index, row = future.result()
            rows_by_index[index] = row
            atomic_jsonl(
                index_path,
                [rows_by_index[position] for position in sorted(rows_by_index)],
            )
            print(
                f"{completed_count}/{len(cases)} "
                f"{'PASS' if row['passed'] else 'FAIL'} {row['case_id']} "
                f"({row['failure_count']} regressions, "
                f"{row['improvement_count']} improvements)",
                flush=True,
            )
    rows = [rows_by_index[index] for index in range(len(cases))]
    atomic_jsonl(index_path, rows)
    print(f"common-input index: {index_path}", flush=True)
    return 0 if rows and all(row["passed"] for row in rows) else 1


def _safe_artifact(root: Path, name: Any) -> Path:
    candidate = (root / str(name)).resolve()
    if not candidate.is_relative_to(root.resolve()) or candidate == root.resolve():
        raise ValueError(f"unsafe artifact path: {name}")
    return candidate


def load_document(path: Path, expected_hash: str) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    root = path.parent.resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("document manifest has no artifact map")
    document = dict(manifest)
    for key in ("pages", "paragraphs", "sections", "footnotes", "diagnostics", "repairs"):
        document[key] = read_jsonl(_safe_artifact(root, artifacts.get(key)))
    for key in ("tables", "images"):
        document[key] = (
            read_jsonl(_safe_artifact(root, artifacts[key])) if key in artifacts else []
        )
    violations: list[str] = []
    if manifest.get("source_sha256") != expected_hash:
        violations.append("source_sha256")
    counts = manifest.get("counts") or {}
    actual_counts = {
        "pages": len(document["pages"]),
        "lines": sum(len(page.get("lines") or []) for page in document["pages"]),
        "paragraphs": len(document["paragraphs"]),
        "sections": len(document["sections"]),
        "footnotes": len(document["footnotes"]),
        "tables": len(document["tables"]),
        "images": len(document["images"]),
        "diagnostics": len(document["diagnostics"]),
        "repairs": len(document["repairs"]),
    }
    if any(counts.get(key) != value for key, value in actual_counts.items() if key in counts):
        violations.append("manifest_counts")
    if manifest.get("page_count") != len(document["pages"]):
        violations.append("page_count")
    line_ids: list[str] = []
    region_ids: set[str] = set()
    for page_index, page in enumerate(document["pages"]):
        if page.get("index") != page_index:
            violations.append("page_indexes")
        lines = page.get("lines") or []
        orders = [line.get("reading_order") for line in lines]
        if len(orders) != len(set(orders)):
            violations.append("duplicate_reading_order")
        line_ids.extend(str(line.get("id")) for line in lines)
        regions = page.get("regions") or []
        region_ids.update(str(region.get("id")) for region in regions)
        membership = Counter(
            str(line_id)
            for region in regions
            for line_id in (region.get("line_ids") or [])
        )
        for line in lines:
            line_id = str(line.get("id"))
            if membership[line_id] != 1:
                violations.append("region_line_coverage")
            if line.get("region_id") and str(line.get("region_id")) not in region_ids:
                violations.append("missing_region")
    if len(line_ids) != len(set(line_ids)):
        violations.append("duplicate_line_id")
    line_id_set = set(line_ids)
    pair_ids = [str(note.get("pair_id")) for note in document["footnotes"]]
    if len(pair_ids) != len(set(pair_ids)):
        violations.append("duplicate_pair_id")
    for note in document["footnotes"]:
        if note.get("reference_line_id") not in (None, "") and str(
            note.get("reference_line_id")
        ) not in line_id_set:
            violations.append("missing_reference_line")
        if any(str(line_id) not in line_id_set for line_id in note.get("body_line_ids") or []):
            violations.append("missing_body_line")
    paragraph_ids = {str(paragraph.get("id")) for paragraph in document["paragraphs"]}
    for section in document["sections"]:
        if any(str(value) not in paragraph_ids for value in section.get("paragraph_ids") or []):
            violations.append("missing_section_paragraph")
    markers = {
        match.group("pair")
        for paragraph in document["paragraphs"]
        for match in MARKER_RE.finditer(str(paragraph.get("text") or ""))
    }
    if markers - set(pair_ids):
        violations.append("orphan_footnote_marker")
    document["contract_violations"] = sorted(set(violations))
    return document


def source_text(document: dict[str, Any], pdf_pages: set[int] | None = None) -> str:
    return "\n".join(
        str(line.get("text") or "")
        for page_number, page in enumerate(document["pages"], start=1)
        if pdf_pages is None or page_number in pdf_pages
        for line in sorted(
            page.get("lines") or [], key=lambda value: int(value.get("reading_order") or 0)
        )
    )


def body_text(document: dict[str, Any]) -> str:
    return "\n\n".join(str(value.get("text") or "") for value in document["paragraphs"])


def keyed_notes(notes: Sequence[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    counters: Counter[str] = Counter()
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for note in notes:
        label = normalize_label(note.get("label"))
        counters[label] += 1
        occurrence = int(note.get("occurrence") or counters[label])
        key = (label, occurrence)
        if key in result:
            key = (label, counters[label])
        result[key] = note
    return result


def journal_reference(path: Path) -> dict[str, Any]:
    pages = read_jsonl(path)
    pair_rows: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"labels": [], "refs": []}
    )
    tables: list[dict[str, Any]] = []
    images: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        pdf_page = int(page.get("pdf_page") or page_index + 1)
        payload = page.get("region_payload") or {}
        width, height = (payload.get("page_size") or [1, 1])[:2]
        table_detection = payload.get("native_pdf_table_detection") or {}
        for table in table_detection.get("tables") or []:
            tables.append(
                {
                    "page": pdf_page,
                    "bbox": normalized_bbox(table.get("bbox"), width, height),
                    "rows": int(table.get("row_count") or 0),
                    "columns": int(table.get("col_count") or 0),
                }
            )
        for raster in payload.get("native_embedded_rasters") or []:
            images.append(
                {
                    "page": pdf_page,
                    "bbox": normalized_bbox(raster.get("bbox"), width, height),
                }
            )
        for annotation in page.get("annotations") or []:
            if annotation.get("pair_status") != "paired" or not annotation.get("pair_id"):
                continue
            taxonomy = annotation.get("taxonomy_name")
            if taxonomy not in {"fn_label", "fn_ref"}:
                continue
            item = {
                "value": normalize_label(annotation.get("selected_text")),
                "page": pdf_page,
                "order": (
                    page_index,
                    int(annotation.get("start_line_order") or 0),
                    int(annotation.get("start_offset") or 0),
                ),
            }
            pair_rows[str(annotation["pair_id"])][
                "labels" if taxonomy == "fn_label" else "refs"
            ].append(item)
    complete = []
    for pair_id, values in pair_rows.items():
        if not values["labels"] or not values["refs"]:
            continue
        label = min(values["labels"], key=lambda item: item["order"])
        complete.append(
            {
                "pair_id": pair_id,
                "label": label["value"],
                "order": label["order"],
                "reference_pages": sorted({item["page"] for item in values["refs"]}),
            }
        )
    complete.sort(key=lambda item: item["order"])
    counters: Counter[str] = Counter()
    for item in complete:
        counters[item["label"]] += 1
        item["occurrence"] = counters[item["label"]]
    return {
        "text": "\n".join(str(page.get("text") or "") for page in pages),
        "page_texts": {
            int(page.get("pdf_page") or page_index + 1): str(page.get("text") or "")
            for page_index, page in enumerate(pages)
        },
        "page_lines": {
            int(page.get("pdf_page") or page_index + 1): sorted(
                (
                    {
                        "order": int(line.get("codex_text_order") or 0),
                        "text": str(line.get("text") or ""),
                    }
                    for region in page.get("regions") or []
                    for line in region.get("lines") or []
                    if line.get("codex_text_order") is not None
                ),
                key=lambda line: line["order"],
            )
            for page_index, page in enumerate(pages)
        },
        "pairs": complete,
        "tables": tables,
        "images": images,
        "page_count": len(pages),
        "pdf_pages": [
            int(page.get("pdf_page") or page_index + 1)
            for page_index, page in enumerate(pages)
        ],
    }


def normalized_bbox(value: Any, width: Any, height: Any) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        bbox = [value.get(key) for key in ("x0", "y0", "x1", "y1")]
    else:
        bbox = list(value or [])
    if len(bbox) != 4 or not all(isinstance(item, (int, float)) for item in bbox):
        return (0.0, 0.0, 0.0, 0.0)
    width = max(1.0, float(width or 1))
    height = max(1.0, float(height or 1))
    return (
        float(bbox[0]) / width,
        float(bbox[1]) / height,
        float(bbox[2]) / width,
        float(bbox[3]) / height,
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return overlap / max(1e-12, left_area + right_area - overlap)


def visual_metrics(
    expected: Sequence[dict[str, Any]],
    actual: Sequence[dict[str, Any]],
    pages: Sequence[dict[str, Any]],
    prefix: str,
) -> tuple[dict[str, float], list[tuple[dict[str, Any], dict[str, Any]]]]:
    page_sizes = {
        int(page.get("number") or index + 1): (
            page.get("width") or 1,
            page.get("height") or 1,
        )
        for index, page in enumerate(pages)
    }
    candidates = []
    for item in actual:
        page = int(item.get("page_number") or 0)
        width, height = page_sizes.get(page, (1, 1))
        candidates.append(
            {
                **item,
                "page": page,
                "normalized_bbox": normalized_bbox(item.get("bbox"), width, height),
            }
        )
    unmatched = set(range(len(candidates)))
    matches = []
    for reference in expected:
        choices = [
            (bbox_iou(reference["bbox"], candidates[index]["normalized_bbox"]), index)
            for index in unmatched
            if candidates[index]["page"] == reference["page"]
        ]
        if choices and max(choices)[0] >= 0.5:
            _, index = max(choices)
            unmatched.remove(index)
            matches.append((reference, candidates[index]))
    precision = len(matches) / len(actual) if actual else (1.0 if not expected else 0.0)
    recall = len(matches) / len(expected) if expected else 1.0
    return (
        {
            f"{prefix}.precision": precision,
            f"{prefix}.recall": recall,
            f"{prefix}.f1": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        },
        matches,
    )


def page_reading_order_metrics(
    expected_pages: dict[int, Sequence[dict[str, Any]]],
    candidate_pages: Sequence[dict[str, Any]],
) -> dict[str, float | None]:
    candidates = {
        index: sorted(
            page.get("lines") or [],
            key=lambda line: int(line.get("reading_order") or 0),
        )
        for index, page in enumerate(candidate_pages, start=1)
    }
    eligible = matched = comparable = concordant = adjacent = adjacent_correct = 0
    for page_number, expected_lines in expected_pages.items():
        expected = [normalize_text(line.get("text")) for line in expected_lines]
        actual = [normalize_text(line.get("text")) for line in candidates.get(page_number, [])]
        expected_counts = Counter(text for text in expected if len(text) >= 4)
        actual_counts = Counter(text for text in actual if len(text) >= 4)
        positions = {
            text: index
            for index, text in enumerate(actual)
            if len(text) >= 4 and actual_counts[text] == 1
        }
        anchors = [
            (index, positions[text])
            for index, text in enumerate(expected)
            if len(text) >= 4 and expected_counts[text] == 1 and text in positions
        ]
        eligible += sum(count == 1 for count in expected_counts.values())
        matched += len(anchors)
        comparable += len(anchors) * (len(anchors) - 1) // 2
        tree = [0] * (len(actual) + 1)
        for _, position in anchors:
            cursor = position
            while cursor > 0:
                concordant += tree[cursor]
                cursor -= cursor & -cursor
            cursor = position + 1
            while cursor < len(tree):
                tree[cursor] += 1
                cursor += cursor & -cursor
        adjacent += max(0, len(anchors) - 1)
        adjacent_correct += sum(
            left[1] < right[1] for left, right in zip(anchors, anchors[1:])
        )
    return {
        "reading_order.anchor_recall": matched / max(1, eligible),
        "reading_order.pairwise": concordant / comparable if comparable else None,
        "reading_order.adjacent": adjacent_correct / adjacent if adjacent else None,
    }


def paragraph_order_metrics(
    expected: Sequence[dict[str, Any]], candidate: Sequence[dict[str, Any]]
) -> dict[str, float | None]:
    def signature(value: Any) -> str:
        return " ".join(normalize_text(value).split()[:12])

    expected_values = [signature(item.get("text")) for item in expected]
    candidate_positions = {
        signature(item.get("text")): index
        for index, item in enumerate(candidate)
        if signature(item.get("text"))
    }
    matches = [
        (index, candidate_positions[value])
        for index, value in enumerate(expected_values)
        if value and value in candidate_positions
    ]
    comparable = 0
    correct = 0
    for left in range(len(matches)):
        for right in range(left + 1, len(matches)):
            comparable += 1
            correct += matches[left][1] < matches[right][1]
    adjacent_total = max(0, len(matches) - 1)
    adjacent_correct = sum(
        left[1] < right[1] for left, right in zip(matches, matches[1:])
    )
    exact = sum(left == right for left, right in matches)
    return {
        "paragraphs.pairwise_order": correct / comparable if comparable else None,
        "paragraphs.adjacent_order": (
            adjacent_correct / adjacent_total if adjacent_total else None
        ),
        "paragraphs.exact_position": exact / len(matches) if matches else None,
    }


def docx_metrics(gold: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    expected_body = "\n\n".join(
        str(paragraph.get("text") or "") for paragraph in gold.get("paragraphs") or []
    )
    result: dict[str, Any] = text_metrics(expected_body, body_text(document), "body")
    result.update(paragraph_order_metrics(gold.get("paragraphs") or [], document["paragraphs"]))
    expected_paragraphs = Counter(
        normalize_text(value.get("text")) for value in gold.get("paragraphs") or []
    )
    actual_paragraphs = Counter(
        normalize_text(value.get("text")) for value in document["paragraphs"]
    )
    expected_units = {
        (text, index)
        for text, count in expected_paragraphs.items()
        for index in range(1, count + 1)
        if text
    }
    actual_units = {
        (text, index)
        for text, count in actual_paragraphs.items()
        for index in range(1, count + 1)
        if text
    }
    result.update(f1(expected_units, actual_units, "paragraphs.boundary"))
    expected_notes = keyed_notes(gold.get("footnotes") or [])
    actual_notes = keyed_notes(document["footnotes"])
    result.update(f1(set(expected_notes), set(actual_notes), "notes.labels"))
    common = sorted(set(expected_notes) & set(actual_notes))
    for field, metric in (
        ("body", "notes.body_similarity"),
        ("sentence_proposition", "notes.sentence_similarity"),
        ("passage_since_prior_note", "notes.passage_similarity"),
    ):
        values = [
            similarity(expected_notes[key].get(field), actual_notes[key].get(field))
            for key in common
        ]
        result[metric] = sum(values) / len(values) if values else (1.0 if not expected_notes else 0.0)
    return result


def journal_metrics(reference: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = page_aligned_text_metrics(
        reference["page_texts"], document["pages"], "source"
    )
    result.update(page_reading_order_metrics(reference.get("page_lines") or {}, document["pages"]))
    expected = {
        (item["label"], int(item["occurrence"])): item for item in reference["pairs"]
    }
    actual = keyed_notes(document["footnotes"])
    result.update(f1(set(expected), set(actual), "notes.labels"))
    expected_pages = {
        (key, page)
        for key, item in expected.items()
        for page in item["reference_pages"]
    }
    actual_pages = {
        (key, int(item["reference_page"]))
        for key, item in actual.items()
        if item.get("reference_page") is not None
    }
    result.update(f1(expected_pages, actual_pages, "notes.reference_pages"))
    table_scores, table_matches = visual_metrics(
        reference.get("tables") or [], document.get("tables") or [], document["pages"], "tables.detection"
    )
    result.update(table_scores)
    shape_scores = []
    for expected_table, actual_table in table_matches:
        cells = actual_table.get("cells") or []
        rows = len(cells)
        columns = max((len(row) for row in cells), default=0)
        denominator = max(1, expected_table["rows"] + expected_table["columns"])
        shape_scores.append(
            max(
                0.0,
                1.0
                - (
                    abs(rows - expected_table["rows"])
                    + abs(columns - expected_table["columns"])
                )
                / denominator,
            )
        )
    result["tables.shape"] = (
        sum(shape_scores) / len(shape_scores)
        if shape_scores
        else (1.0 if not reference.get("tables") else 0.0)
    )
    image_scores, image_matches = visual_metrics(
        reference.get("images") or [], document.get("images") or [], document["pages"], "images.detection"
    )
    result.update(image_scores)
    expected_vision = [
        (expected_image, actual_image)
        for expected_image, actual_image in image_matches
        if 0.01
        <= (expected_image["bbox"][2] - expected_image["bbox"][0])
        * (expected_image["bbox"][3] - expected_image["bbox"][1])
        < 0.75
        and expected_image["bbox"][3] > 0.12
        and expected_image["bbox"][1] < 0.88
    ]
    result["images.vision_recall"] = (
        sum(actual.get("route") == "vision" for _, actual in expected_vision)
        / len(expected_vision)
        if expected_vision
        else 1.0
    )
    return result


def evaluate(case: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "contract.invalid_count": len(document["contract_violations"]),
        "document.status_rank": STATUS_RANK.get(str(document.get("status")), -1),
        "document.page_count": len(document["pages"]),
    }
    evidence = case["evidence"]
    evidence_path = Path(evidence["path"])
    if sha256(evidence_path) != evidence["sha256"]:
        raise ValueError(f"evidence hash changed for {case['case_id']}")
    if evidence["kind"] == "independent-docx":
        metrics.update(
            docx_metrics(json.loads(evidence_path.read_text(encoding="utf-8")), document)
        )
    elif evidence["kind"] == "canonical-derived":
        metrics.update(journal_metrics(journal_reference(evidence_path), document))
    else:
        raise ValueError(f"unsupported evidence kind: {evidence['kind']}")
    return metrics


def semantic_contract(document: dict[str, Any]) -> dict[str, Any]:
    line_by_id = {
        str(line.get("id")): line
        for page in document["pages"]
        for line in page.get("lines") or []
    }
    return {
        "printed_pages": [page.get("printed_label") for page in document["pages"]],
        "paragraphs": [
            (paragraph.get("region_type"), normalize_text(paragraph.get("text")))
            for paragraph in document["paragraphs"]
        ],
        "sections": [
            (
                section.get("locator_kind"),
                normalize_text(section.get("locator")),
                normalize_text(section.get("heading")),
                normalize_text(section.get("text")),
            )
            for section in document["sections"]
        ],
        "regions": [
            [
                (
                    region.get("type"),
                    normalize_text(
                        " ".join(
                            str(line_by_id.get(str(line_id), {}).get("text") or "")
                            for line_id in region.get("line_ids") or []
                        )
                    ),
                )
                for region in sorted(
                    page.get("regions") or [],
                    key=lambda value: int(value.get("reading_order") or 0),
                )
            ]
            for page in document["pages"]
        ],
        "notes": [
            (
                normalize_label(note.get("label")),
                int(note.get("occurrence") or 0),
                note.get("reference_page"),
                tuple(note.get("body_pages") or []),
                normalize_text(note.get("body")),
                normalize_text(note.get("sentence_proposition")),
                normalize_text(note.get("passage_since_prior_note")),
                json.dumps(note.get("crossrefs") or [], sort_keys=True, ensure_ascii=False),
            )
            for note in document["footnotes"]
        ],
        "tables": [
            (
                table.get("page_number"),
                table.get("bbox"),
                table.get("cells"),
                table.get("provenance"),
            )
            for table in document["tables"]
        ],
        "images": [
            (
                image.get("page_number"),
                image.get("bbox"),
                image.get("route"),
            )
            for image in document["images"]
        ],
        "diagnostics": sorted(
            (value.get("severity"), value.get("code"))
            for value in document["diagnostics"]
        ),
    }


LOWER_IS_BETTER = {
    "contract.invalid_count",
    "body.cer",
    "body.wer",
    "source.cer",
    "source.wer",
}
EXACT_METRICS = {"document.page_count"}


def metric_regressions(
    oracle: dict[str, Any], candidate: dict[str, Any], tolerance: float = 1e-12
) -> list[str]:
    failures: list[str] = []
    for key in sorted(set(oracle) | set(candidate)):
        left = oracle.get(key)
        right = candidate.get(key)
        if left is None and right is None:
            continue
        if left is None or right is None:
            failures.append(f"{key}: missing comparable value ({left!r} -> {right!r})")
            continue
        if key in EXACT_METRICS:
            if right != left:
                failures.append(f"{key}: {left!r} -> {right!r}")
        elif key in LOWER_IS_BETTER:
            if float(right) > float(left) + tolerance:
                failures.append(f"{key}: {left:.9g} -> {right:.9g}")
        elif float(right) + tolerance < float(left):
            failures.append(f"{key}: {left:.9g} -> {right:.9g}")
    if candidate.get("contract.invalid_count") != 0:
        failures.append(
            f"contract.invalid_count must be zero, got {candidate.get('contract.invalid_count')}"
        )
    return failures


def compatibility_regressions(
    evidence_kind: str,
    oracle: dict[str, Any],
    candidate: dict[str, Any],
) -> list[str]:
    old = semantic_contract(oracle)
    new = semantic_contract(candidate)
    if evidence_kind == "independent-docx":
        exact_fields = ("printed_pages", "sections", "regions", "diagnostics")
    else:
        exact_fields = tuple(old)
    return [
        f"semantic contract changed: {field}"
        for field in exact_fields
        if old[field] != new[field]
    ]


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of an empty sequence")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1))
    return ordered[index]


def load_run_index(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        case_id = str(row["case_id"])
        if case_id in result:
            raise ValueError(f"duplicate run result: {case_id}")
        result[case_id] = row
    return result


def load_perf_index(path: Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def summarize_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
    return {
        key: {
            "count": len(numbers),
            "mean": statistics.mean(numbers),
            "median": statistics.median(numbers),
            "p05": percentile(numbers, 0.05),
            "p95": percentile(numbers, 0.95),
            "min": min(numbers),
            "max": max(numbers),
        }
        for key, numbers in sorted(values.items())
    }


def score_case(case: dict[str, Any], document: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    metrics: dict[str, Any] = {
        "contract.invalid_count": len(document["contract_violations"]),
        "document.status_rank": STATUS_RANK.get(str(document.get("status")), -1),
        "document.page_count": len(document["pages"]),
    }
    evidence = case["evidence"]
    evidence_path = Path(evidence["path"])
    if sha256(evidence_path) != evidence["sha256"]:
        raise ValueError(f"evidence hash changed for {case['case_id']}")
    if evidence["kind"] == "canonical-derived":
        reference = journal_reference(evidence_path)
        metrics.update(journal_metrics(reference, document))
        counts = {
            "notes": len(reference["pairs"]),
            "tables": len(reference["tables"]),
            "images": len(reference["images"]),
        }
    elif evidence["kind"] == "independent-docx":
        gold = json.loads(evidence_path.read_text(encoding="utf-8"))
        metrics.update(docx_metrics(gold, document))
        counts = {
            "notes": len(gold.get("footnotes") or []),
            "tables": len(gold.get("tables") or []),
            "images": len(gold.get("images") or []),
        }
    else:
        raise ValueError(f"unsupported evidence kind: {evidence['kind']}")
    return metrics, counts


def score_run(arguments: argparse.Namespace) -> int:
    output_path = arguments.output.resolve()
    cases_path = output_path.with_suffix(".cases.jsonl")
    cases = {
        str(case["case_id"]): case
        for case in selected_case_ids(
            selected_cases(arguments.manifest.resolve(), arguments.split),
            getattr(arguments, "case", []),
        )
    }
    runs = load_run_index(arguments.results.resolve())

    def score_one(case_id: str, case: dict[str, Any]) -> dict[str, Any]:
        run = runs.get(case_id)
        row: dict[str, Any] = {
            "case_id": case_id,
            "lane": case["lane"],
            "page_count": int(case["page_count"]),
            "wall_seconds": run.get("wall_seconds") if run else None,
            "peak_rss_bytes": run.get("peak_rss_bytes") if run else None,
            "metrics": {},
            "evidence_counts": {},
            "failure": None,
        }
        if run is None:
            row["failure"] = "missing run"
        elif run.get("returncode") != 0:
            row["failure"] = str(run.get("stderr") or "run failed").strip()
        else:
            try:
                document = load_document(Path(run["artifact"]), str(case["pdf_sha256"]))
                row["metrics"], row["evidence_counts"] = score_case(case, document)
                row["product_counts"] = {
                    "notes": len(document["footnotes"]),
                    "tables": len(document["tables"]),
                    "images": len(document["images"]),
                }
            except Exception as error:  # preserve every other usable corpus result
                row["failure"] = str(error)
        return row

    report_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, arguments.jobs)) as executor:
        futures = [
            executor.submit(score_one, case_id, case)
            for case_id, case in sorted(cases.items())
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            report_rows.append(future.result())
            if index % arguments.progress_every == 0 or index == len(cases):
                atomic_jsonl(cases_path, sorted(report_rows, key=lambda row: row["case_id"]))
                print(f"score {index}/{len(cases)}", flush=True)

    report_rows.sort(key=lambda row: row["case_id"])

    successful = [row for row in report_rows if not row["failure"]]
    subsets = {
        "notes_positive": [row for row in successful if row["evidence_counts"].get("notes", 0) > 0],
        "tables_positive": [row for row in successful if row["evidence_counts"].get("tables", 0) > 0],
        "tables_negative": [row for row in successful if row["evidence_counts"].get("tables", 0) == 0],
        "images_positive": [row for row in successful if row["evidence_counts"].get("images", 0) > 0],
        "images_negative": [row for row in successful if row["evidence_counts"].get("images", 0) == 0],
    }
    lane_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in successful:
        lane_rows[row["lane"]].append(row)

    def group_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        seconds = [float(row["wall_seconds"]) for row in rows if row.get("wall_seconds") is not None]
        pages = sum(int(row["page_count"]) for row in rows)
        memory = [float(row["peak_rss_bytes"]) for row in rows if row.get("peak_rss_bytes") is not None]
        total_seconds = sum(seconds)
        return {
            "case_count": len(rows),
            "page_count": pages,
            "wall_seconds": total_seconds,
            "pages_per_second": pages / total_seconds if total_seconds else None,
            "median_peak_rss_bytes": statistics.median(memory) if memory else None,
            "p95_peak_rss_bytes": percentile(memory, 0.95) if memory else None,
            "metrics": summarize_metric_rows(rows),
        }

    report = {
        "schema_version": "legalpdf.port-score.v1",
        "case_count": len(cases),
        "success_count": len(successful),
        "failure_count": len(report_rows) - len(successful),
        "failures": [
            {"case_id": row["case_id"], "failure": row["failure"]}
            for row in report_rows
            if row["failure"]
        ],
        "all": group_summary(successful),
        "subsets": {name: group_summary(rows) for name, rows in subsets.items()},
        "lanes": {name: group_summary(rows) for name, rows in sorted(lane_rows.items())},
    }
    atomic_json(output_path, report)
    print(
        f"score: {len(successful)}/{len(cases)} successful; {output_path}",
        flush=True,
    )
    return 0 if len(successful) == len(cases) else 1


def performance_regressions(
    cases: dict[str, dict[str, Any]],
    rows: Sequence[dict[str, Any]],
    repeats: int,
) -> tuple[list[str], dict[str, Any]]:
    failures: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["case_id"]), str(row["arm"]))].append(row)
    lane_times: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"oracle": [], "rust": []}
    )
    lane_memory: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"oracle": [], "rust": []}
    )
    per_case: dict[str, Any] = {}
    for case_id, case in cases.items():
        if not case.get("performance_case"):
            continue
        values: dict[str, Any] = {}
        for arm in ("oracle", "rust"):
            successful = [
                row
                for row in grouped.get((case_id, arm), [])
                if row.get("returncode") == 0
            ]
            if len(successful) < repeats:
                failures.append(
                    f"{case_id}: {arm} has {len(successful)}/{repeats} performance runs"
                )
                continue
            successful = sorted(successful, key=lambda row: int(row["repeat"]))[:repeats]
            times = [float(row["wall_seconds"]) for row in successful]
            memories = [
                float(row["peak_rss_bytes"])
                for row in successful
                if row.get("peak_rss_bytes") is not None
            ]
            if len(memories) != repeats:
                failures.append(f"{case_id}: {arm} peak RSS was not measured")
            values[arm] = {
                "median_seconds": statistics.median(times),
                "p95_seconds": percentile(times, 0.95),
                "median_peak_rss": statistics.median(memories) if memories else None,
            }
            lane_times[str(case["lane"])][arm].append(statistics.median(times))
            if memories:
                lane_memory[str(case["lane"])][arm].append(statistics.median(memories))
        if set(values) == {"oracle", "rust"}:
            if values["rust"]["median_seconds"] >= values["oracle"]["median_seconds"]:
                failures.append(f"{case_id}: Rust median wall time was not faster")
            old_memory = values["oracle"]["median_peak_rss"]
            new_memory = values["rust"]["median_peak_rss"]
            if old_memory is not None and new_memory is not None and new_memory > old_memory:
                failures.append(f"{case_id}: Rust median peak RSS increased")
        per_case[case_id] = values
    lanes: dict[str, Any] = {}
    for lane, arms in lane_times.items():
        if not arms["oracle"] or not arms["rust"]:
            continue
        lanes[lane] = {
            arm: {
                "median_seconds": statistics.median(arms[arm]),
                "p95_seconds": percentile(arms[arm], 0.95),
                "max_seconds": max(arms[arm]),
                "p95_peak_rss": (
                    percentile(lane_memory[lane][arm], 0.95)
                    if lane_memory[lane][arm]
                    else None
                ),
            }
            for arm in ("oracle", "rust")
        }
        for metric in ("median_seconds", "p95_seconds", "max_seconds"):
            if lanes[lane]["rust"][metric] >= lanes[lane]["oracle"][metric]:
                failures.append(f"{lane}: Rust {metric} was not faster")
        old_memory = lanes[lane]["oracle"]["p95_peak_rss"]
        new_memory = lanes[lane]["rust"]["p95_peak_rss"]
        if old_memory is not None and new_memory is not None and new_memory > old_memory:
            failures.append(f"{lane}: Rust p95 peak RSS increased")
    return failures, {"per_case": per_case, "lanes": lanes}


def gate(arguments: argparse.Namespace) -> int:
    output_path = arguments.output.resolve()
    case_results_path = output_path.with_suffix(".cases.jsonl")
    manifest_rows = selected_cases(arguments.manifest.resolve(), arguments.split)
    cases = {str(row["case_id"]): row for row in manifest_rows}
    oracle_runs = load_run_index(arguments.oracle_results.resolve())
    rust_runs = load_run_index(arguments.rust_results.resolve())
    report_rows = []
    failures: list[str] = []
    for index, (case_id, case) in enumerate(sorted(cases.items()), start=1):
        old_run = oracle_runs.get(case_id)
        new_run = rust_runs.get(case_id)
        case_failures: list[str] = []
        if old_run is None or new_run is None:
            case_failures.append("missing oracle or Rust run")
            old_metrics: dict[str, Any] = {}
            new_metrics: dict[str, Any] = {}
        elif old_run.get("returncode") != 0 or new_run.get("returncode") != 0:
            case_failures.append("oracle or Rust run failed")
            old_metrics = {}
            new_metrics = {}
        else:
            old_document = load_document(
                Path(old_run["artifact"]), str(case["pdf_sha256"])
            )
            new_document = load_document(
                Path(new_run["artifact"]), str(case["pdf_sha256"])
            )
            old_metrics = evaluate(case, old_document)
            new_metrics = evaluate(case, new_document)
            case_failures.extend(metric_regressions(old_metrics, new_metrics))
            case_failures.extend(
                compatibility_regressions(
                    str(case["evidence"]["kind"]), old_document, new_document
                )
            )
        if case_failures:
            failures.extend(f"{case_id}: {value}" for value in case_failures)
        report_rows.append(
            {
                "case_id": case_id,
                "lane": case["lane"],
                "evidence_kind": case["evidence"]["kind"],
                "oracle": old_metrics,
                "rust": new_metrics,
                "failures": case_failures,
            }
        )
        atomic_jsonl(case_results_path, report_rows)
        print(
            f"{index}/{len(cases)} {'PASS' if not case_failures else 'FAIL'} {case_id}",
            flush=True,
        )
    performance: dict[str, Any] | None = None
    if arguments.performance_results:
        perf_failures, performance = performance_regressions(
            cases,
            load_perf_index(arguments.performance_results.resolve()),
            arguments.performance_repeats,
        )
        failures.extend(perf_failures)
    elif any(case.get("performance_case") for case in cases.values()):
        failures.append("performance results are required for the frozen performance cases")
    report = {
        "schema_version": REPORT_SCHEMA,
        "passed": not failures,
        "case_count": len(cases),
        "failure_count": len(failures),
        "failures": failures,
        "cases": report_rows,
        "performance": performance,
    }
    atomic_json(output_path, report)
    print(
        f"gate {'PASSED' if not failures else 'FAILED'}: "
        f"{len(cases)} cases, {len(failures)} failures; {output_path}",
        flush=True,
    )
    return 0 if not failures else 1


def run_performance(arguments: argparse.Namespace) -> int:
    cases = selected_case_ids(
        [
            case
            for case in selected_cases(arguments.manifest.resolve(), arguments.split)
            if case.get("performance_case")
        ],
        arguments.case,
    )
    output = arguments.output.resolve()
    records_root = output / "records"
    scratch_root = output / "scratch"
    scratch_root.mkdir(parents=True, exist_ok=True)
    oracle_root = arguments.oracle_root.resolve()
    rust_binary = arguments.rust_binary.resolve()
    identities = {
        "oracle": engine_identity("oracle", oracle_root, None),
        "rust": engine_identity("rust", None, rust_binary),
    }
    common_inputs: dict[str, Path] = {}
    if arguments.phase == "replay":
        common_root = output / "common"
        common_root.mkdir(parents=True, exist_ok=True)
        for case in cases:
            case_id = str(case["case_id"])
            common_path = common_root / f"{manifest_case_token(case_id)}.json"
            common_inputs[case_id] = common_path
            if _valid_common_file(
                common_path, str(case["pdf_sha256"]), "legalpdf.common-input.v1"
            ):
                continue
            command, cwd, env = parity_command(
                "extract",
                "oracle",
                case_pdf(case),
                common_path,
                oracle_root,
                rust_binary,
            )
            measured = run_child(command, cwd=cwd, env=env, timeout=arguments.timeout)
            if measured["returncode"] != 0 or not _valid_common_file(
                common_path, str(case["pdf_sha256"]), "legalpdf.common-input.v1"
            ):
                raise RuntimeError(
                    f"{case_id}: common-input setup failed: "
                    f"{measured['stderr'] or measured['stdout']}"
                )
    total = len(cases) * arguments.repeats * 2
    progress = 0
    rows: list[dict[str, Any]] = []
    for repeat in range(arguments.repeats):
        for case in cases:
            case_id = str(case["case_id"])
            order = ["oracle", "rust"]
            if (repeat + int(manifest_case_token(case_id), 16)) % 2:
                order.reverse()
            for arm in order:
                progress += 1
                key = hashlib.sha256(
                    f"{arguments.phase}:{case_id}:{arm}:{identities[arm]}:{repeat}".encode()
                ).hexdigest()[:32]
                result_path = records_root / f"{key}.json"
                if result_path.is_file():
                    prior = json.loads(result_path.read_text(encoding="utf-8"))
                    if prior.get("returncode") == 0:
                        rows.append(prior)
                        print(f"{progress}/{total} skip {case_id} {arm} r{repeat + 1}", flush=True)
                        continue
                with tempfile.TemporaryDirectory(dir=scratch_root) as temporary:
                    artifact = Path(temporary) / "artifact"
                    if arguments.phase == "full":
                        command, cwd, env = engine_command(
                            arm,
                            case_pdf(case),
                            artifact,
                            oracle_root,
                            rust_binary,
                        )
                    else:
                        source = (
                            common_inputs[case_id]
                            if arguments.phase == "replay"
                            else case_pdf(case)
                        )
                        command, cwd, env = parity_command(
                            arguments.phase,
                            arm,
                            source,
                            artifact,
                            oracle_root,
                            rust_binary,
                        )
                    measured = run_child(
                        command, cwd=cwd, env=env, timeout=arguments.timeout
                    )
                row = {
                    "schema_version": RUN_SCHEMA,
                    "case_id": case_id,
                    "lane": case["lane"],
                    "arm": arm,
                    "engine_id": identities[arm],
                    "phase": arguments.phase,
                    "repeat": repeat,
                    "pdf_sha256": case["pdf_sha256"],
                    **measured,
                }
                atomic_json(result_path, row)
                rows.append(row)
                print(
                    f"{progress}/{total} {'complete' if measured['returncode'] == 0 else 'failed'} "
                    f"{case_id} {arm} r{repeat + 1} wall={measured['wall_seconds']:.4f}s",
                    flush=True,
                )
    index_path = output / "performance.jsonl"
    atomic_jsonl(
        index_path,
        sorted(rows, key=lambda row: (row["case_id"], row["arm"], row["repeat"])),
    )
    print(f"performance index: {index_path}", flush=True)
    return 0 if all(row["returncode"] == 0 for row in rows) else 1


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        description="Frozen, fail-closed differential harness for the native Rust port."
    )
    commands = root.add_subparsers(dest="command", required=True)

    audit = commands.add_parser(
        "audit-map",
        help="Verify the frozen Python oracle and report every mapped symbol/blocker",
    )
    audit.add_argument("--oracle-root", type=Path, required=True)
    audit.add_argument("--rust-root", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.set_defaults(handler=audit_port_map)

    common = commands.add_parser(
        "common-input",
        help="Replay identical oracle-extracted pages through Python and Rust",
    )
    common.add_argument("manifest", type=Path)
    common.add_argument("--oracle-root", type=Path, required=True)
    common.add_argument("--rust-binary", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)
    common.add_argument(
        "--split",
        choices=("qualification", "confirmation", "all"),
        default="qualification",
    )
    common.add_argument("--timeout", type=float, default=300.0)
    common.add_argument("--case", action="append", default=[])
    common.add_argument("--jobs", type=int, default=4)
    common.set_defaults(handler=run_common_input)

    extraction = commands.add_parser(
        "extraction-autopsy",
        help="Diff the complete pre-structure extraction contract and rank root gaps",
    )
    extraction.add_argument("manifest", type=Path)
    extraction.add_argument("--oracle-root", type=Path, required=True)
    extraction.add_argument("--rust-binary", type=Path, required=True)
    extraction.add_argument("--output", type=Path, required=True)
    extraction.add_argument(
        "--split",
        choices=("qualification", "confirmation", "all"),
        default="qualification",
    )
    extraction.add_argument("--timeout", type=float, default=300.0)
    extraction.add_argument("--jobs", type=int, default=4)
    extraction.set_defaults(handler=run_extraction_autopsy)

    contracts = commands.add_parser(
        "contract-diff",
        help="Compare one pure public contract against the frozen Python oracle",
    )
    contracts.add_argument("input", type=Path)
    contracts.add_argument("--oracle-root", type=Path, required=True)
    contracts.add_argument("--rust-binary", type=Path, required=True)
    contracts.add_argument("--output", type=Path, required=True)
    contracts.add_argument("--timeout", type=float, default=60.0)
    contracts.set_defaults(handler=run_contract_diff)

    journals = commands.add_parser("freeze-journals")
    journals.add_argument("--database", type=Path, required=True)
    journals.add_argument("--pdf-root", type=Path, required=True)
    journals.add_argument("--contract-root", type=Path, required=True)
    journals.add_argument("--dataset", action="append")
    journals.add_argument("--per-dataset", type=int)
    journals.add_argument(
        "--minimum-dataset",
        action="append",
        default=[],
        metavar="DATASET=COUNT",
    )
    journals.add_argument("--output", type=Path, required=True)
    journals.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    journals.add_argument("--diverse", action="store_true")
    journals.add_argument("--total", type=int)
    journals.add_argument("--salt", default="legalpdf-port-v1")
    journals.add_argument("--performance-per-lane", type=int, default=16)
    journals.set_defaults(handler=freeze_journals)

    docx = commands.add_parser("freeze-docx")
    docx.add_argument("--input", type=Path, required=True)
    docx.add_argument("--output", type=Path, required=True)
    docx.add_argument("--salt", default="legalpdf-port-v1")
    docx.add_argument("--performance-per-lane", type=int, default=8)
    docx.set_defaults(handler=freeze_docx)

    merge = commands.add_parser("merge")
    merge.add_argument("--input", type=Path, action="append", required=True)
    merge.add_argument("--output", type=Path, required=True)
    merge.set_defaults(handler=merge_manifests)

    run = commands.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--arm", choices=("oracle", "rust"), required=True)
    run.add_argument("--oracle-root", type=Path)
    run.add_argument("--rust-binary", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", choices=("qualification", "confirmation", "all"), default="all")
    run.add_argument("--timeout", type=float, default=300.0)
    run.add_argument("--progress-every", type=int, default=25)
    run.add_argument("--case", action="append", default=[])
    run.set_defaults(handler=run_arm)

    score = commands.add_parser("score", help="Score one corpus run against frozen evidence")
    score.add_argument("manifest", type=Path)
    score.add_argument("--results", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--split", choices=("qualification", "confirmation", "all"), default="all")
    score.add_argument("--progress-every", type=int, default=25)
    score.add_argument("--jobs", type=int, default=4)
    score.add_argument("--case", action="append", default=[])
    score.set_defaults(handler=score_run)

    performance = commands.add_parser("performance")
    performance.add_argument("manifest", type=Path)
    performance.add_argument("--oracle-root", type=Path, required=True)
    performance.add_argument("--rust-binary", type=Path, required=True)
    performance.add_argument("--output", type=Path, required=True)
    performance.add_argument("--split", choices=("qualification", "confirmation", "all"), default="all")
    performance.add_argument("--repeats", type=int, default=7)
    performance.add_argument("--timeout", type=float, default=300.0)
    performance.add_argument("--case", action="append", default=[])
    performance.add_argument(
        "--phase", choices=("full", "extract", "replay"), default="full"
    )
    performance.set_defaults(handler=run_performance)

    release_gate = commands.add_parser("gate")
    release_gate.add_argument("manifest", type=Path)
    release_gate.add_argument("--oracle-results", type=Path, required=True)
    release_gate.add_argument("--rust-results", type=Path, required=True)
    release_gate.add_argument("--performance-results", type=Path)
    release_gate.add_argument("--performance-repeats", type=int, default=7)
    release_gate.add_argument("--split", choices=("qualification", "confirmation", "all"), default="all")
    release_gate.add_argument("--output", type=Path, required=True)
    release_gate.set_defaults(handler=gate)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
