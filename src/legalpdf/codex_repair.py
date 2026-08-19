from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .model import Diagnostic, LegalDocument, Region, RepairRecord

PROMPT_VERSION = "legalpdf.codex.structure.r1.v2"
CONTEXT_RADIUS = 1
MAX_ATTEMPTS = 3
MAX_LIVE_CALLS = 6
MAX_SCOPE_PAGES = 2
_REPAIRABLE = {
    "COLUMN_ORDER_UNCERTAIN",
    "FOOTNOTE_UNMATCHED_LABEL",
    "FOOTNOTE_UNMATCHED_REFERENCE",
    "FOOTNOTE_REGION_UNCERTAIN",
    "TEXT_QUALITY_LOW",
}
_REGION_TYPES = ["body", "heading", "footnote", "header", "footer", "unknown"]


def _stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_strict_json(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
    )


def _schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["pages"],
        "properties": {
            "pages": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["page_index", "regions"],
                    "properties": {
                        "page_index": {"type": "integer", "minimum": 0},
                        "regions": {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": ["region_type", "line_ids"],
                                "properties": {
                                    "region_type": {
                                        "type": "string",
                                        "enum": _REGION_TYPES,
                                    },
                                    "line_ids": {
                                        "type": "array",
                                        "minItems": 1,
                                        "items": {
                                            "type": "string",
                                            "minLength": 1,
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def repair_identity() -> dict[str, Any]:
    repairable = sorted(_REPAIRABLE)
    return {
        "schema_version": "legalpdf.codex.repair-identity.v1",
        "prompt_version": PROMPT_VERSION,
        "response_schema_sha256": _stable_hash(_schema()),
        "context_radius": CONTEXT_RADIUS,
        "max_attempts": MAX_ATTEMPTS,
        "max_live_calls": MAX_LIVE_CALLS,
        "max_scope_pages": MAX_SCOPE_PAGES,
        "repairable_diagnostics": repairable,
        "repairable_diagnostics_sha256": _stable_hash(repairable),
    }


def _context(document: LegalDocument, target_pages: Sequence[int]) -> dict[str, Any]:
    targets = set(target_pages)
    pages = []
    for index in range(
        max(0, min(target_pages) - CONTEXT_RADIUS),
        min(document.page_count, max(target_pages) + CONTEXT_RADIUS + 1),
    ):
        page = document.pages[index]
        pages.append(
            {
                "page_index": page.index,
                "width": page.width,
                "height": page.height,
                "target": page.index in targets,
                "lines": [
                    {
                        "id": line.id,
                        "text": line.text,
                        "bbox": line.bbox,
                        "current_region_type": line.region_type,
                        "current_reading_order": line.reading_order,
                    }
                    for line in page.lines
                ],
            }
        )
    diagnostics = [
        asdict(diagnostic)
        for diagnostic in document.diagnostics
        if diagnostic.page_index in targets and diagnostic.code in _REPAIRABLE
    ]
    return {
        "schema_version": "legalpdf.codex.input.v1",
        "target_pages": list(target_pages),
        "pages": pages,
        "diagnostics": diagnostics,
    }


def _prompt(context: dict[str, Any], previous_error: str = "") -> str:
    retry = (
        "\nThe previous response was rejected for this reason: "
        f"{previous_error}\nCorrect that exact contract failure."
        if previous_error
        else ""
    )
    return (
        "You are repairing structure in a legal PDF. The input contains immutable "
        "line IDs and immutable text for one or more adjacent target pages with r=1 "
        "context. Return one output page for EVERY TARGET PAGE and no context pages. "
        "Region order and line order inside each region define reading order. Include "
        "every target-page line ID exactly once. You cannot edit glyph text "
        "because the output contract contains IDs only. Classify page furniture, "
        "headings, body text, and footnote/endnote material conservatively. "
        "Do not abstain and do not add commentary."
        f"{retry}\n\nINPUT:\n"
        + json.dumps(context, ensure_ascii=False, sort_keys=True)
    )


def _render_context(
    pdf_path: Path, target_pages: Sequence[int], output_dir: Path
) -> list[Path]:
    import fitz

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with fitz.open(pdf_path) as pdf:
        for index in range(
            max(0, min(target_pages) - CONTEXT_RADIUS),
            min(pdf.page_count, max(target_pages) + CONTEXT_RADIUS + 1),
        ):
            output = output_dir / f"p{index + 1:04d}.png"
            if not output.is_file():
                page = pdf.load_page(index)
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                pixmap.save(output)
            paths.append(output)
    return paths


def _validate(
    response: Any,
    *,
    target_pages: Sequence[int],
    expected_line_ids: Mapping[int, Iterable[str]],
) -> tuple[bool, str]:
    if not isinstance(response, dict):
        return False, "response is not an object"
    if set(response) != {"pages"}:
        return False, "response has missing or additional top-level properties"
    pages = response.get("pages")
    if not isinstance(pages, list) or not pages:
        return False, "pages must be a non-empty list"
    actual_pages = [
        page.get("page_index") for page in pages if isinstance(page, dict)
    ]
    if len(actual_pages) != len(set(actual_pages)) or set(actual_pages) != set(
        target_pages
    ):
        return False, "output pages do not exactly match the requested targets"
    for page in pages:
        if not isinstance(page, dict) or set(page) != {"page_index", "regions"}:
            return False, "an output page has missing or additional properties"
        page_index = page["page_index"]
        if type(page_index) is not int or page_index < 0:
            return False, "a page_index is not a non-negative integer"
        regions = page.get("regions")
        if not isinstance(regions, list) or not regions:
            return False, f"page {page_index} regions must be a non-empty list"
        actual: list[str] = []
        for region in regions:
            if not isinstance(region, dict) or set(region) != {
                "region_type",
                "line_ids",
            }:
                return False, "a region has missing or additional properties"
            if region.get("region_type") not in _REGION_TYPES:
                return False, "a region_type is unsupported"
            line_ids = region.get("line_ids")
            if not isinstance(line_ids, list) or not line_ids:
                return False, "a region has no line IDs"
            if not all(isinstance(line_id, str) and line_id for line_id in line_ids):
                return False, "a line ID is not a non-empty string"
            actual.extend(line_ids)
        expected = list(expected_line_ids[page_index])
        if len(actual) != len(set(actual)):
            return False, f"page {page_index} contains a duplicate line ID"
        if set(actual) != set(expected) or len(actual) != len(expected):
            missing = sorted(set(expected) - set(actual))
            unknown = sorted(set(actual) - set(expected))
            return (
                False,
                f"page {page_index} line coverage mismatch; "
                f"missing={missing}, unknown={unknown}",
            )
    return True, ""


def _usage_from_events(stdout: str) -> dict[str, int]:
    totals: dict[str, int] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in {
                    "input_tokens",
                    "output_tokens",
                    "cached_input_tokens",
                    "total_tokens",
                } and isinstance(child, int):
                    totals[key] = max(totals.get(key, 0), child)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    for line in stdout.splitlines():
        try:
            visit(json.loads(line))
        except json.JSONDecodeError:
            continue
    return totals


def _codex_command() -> str | None:
    return os.environ.get("CODEX_EXEC_COMMAND", "").strip() or shutil.which(
        "codex"
    )


def _invoke(
    *,
    prompt: str,
    schema_path: Path,
    image_paths: list[Path],
    model: str,
    effort: str,
    work_dir: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    executable = _codex_command()
    if not executable:
        raise RuntimeError("codex executable was not found on PATH")
    output_path = work_dir / "last-message.json"
    output_path.unlink(missing_ok=True)
    arguments = [
        executable,
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--model",
        model,
        "-c",
        f"model_reasoning_effort={json.dumps(effort)}",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "--color",
        "never",
        "--json",
    ]
    for image_path in image_paths:
        arguments.extend(["--image", str(image_path)])
    arguments.append("-")
    started = time.perf_counter()
    completed = subprocess.run(
        arguments,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        cwd=work_dir,
        timeout=timeout_seconds,
        check=False,
        shell=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"codex exec exited with {completed.returncode}: {message[-2000:]}"
        )
    if not output_path.is_file():
        raise RuntimeError("codex exec did not write its final response")
    try:
        response = json.loads(output_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"codex response is not valid JSON: {exc}") from exc
    return response, _usage_from_events(completed.stdout), elapsed


def _replay_page(
    document: LegalDocument, page_index: int, regions_response: list[dict[str, Any]]
) -> None:
    page = document.pages[page_index]
    line_by_id = {line.id: line for line in page.lines}
    ordered = []
    regions: list[Region] = []
    for region_index, item in enumerate(regions_response, start=1):
        region_id = f"p{page.number:04d}-r{region_index:04d}"
        lines = [line_by_id[line_id] for line_id in item["line_ids"]]
        for line in lines:
            line.region_id = region_id
            line.region_type = item["region_type"]
            line.reading_order = len(ordered) + 1
            ordered.append(line)
        boxes = [line.bbox for line in lines]
        regions.append(
            Region(
                id=region_id,
                page_index=page_index,
                type=item["region_type"],
                line_ids=[line.id for line in lines],
                bbox=[
                    min(box[0] for box in boxes),
                    min(box[1] for box in boxes),
                    max(box[2] for box in boxes),
                    max(box[3] for box in boxes),
                ],
                reading_order=lines[0].reading_order,
            )
        )
    page.lines = ordered
    page.regions = regions


def _replay(document: LegalDocument, response: dict[str, Any]) -> None:
    for page in response["pages"]:
        _replay_page(document, int(page["page_index"]), page["regions"])


def _repair_scopes(document: LegalDocument) -> list[list[int]]:
    codes_by_page: dict[int, set[str]] = {}
    for diagnostic in document.diagnostics:
        if (
            diagnostic.page_index is not None
            and diagnostic.code in _REPAIRABLE
            and 0 <= diagnostic.page_index < document.page_count
        ):
            codes_by_page.setdefault(diagnostic.page_index, set()).add(
                diagnostic.code
            )
    scopes: list[list[int]] = []
    for page_index in sorted(codes_by_page):
        if (
            scopes
            and len(scopes[-1]) < MAX_SCOPE_PAGES
            and page_index == scopes[-1][-1] + 1
            and codes_by_page[page_index] & codes_by_page[scopes[-1][-1]]
        ):
            scopes[-1].append(page_index)
        else:
            scopes.append([page_index])
    return scopes


def improve_document(
    document: LegalDocument,
    pdf_path: Path,
    *,
    model: str,
    effort: str,
    cache_dir: Path,
    timeout_seconds: int = 600,
) -> LegalDocument:
    if not model.strip() or not effort.strip():
        raise ValueError("model and effort must be non-empty")
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)
    scopes = _repair_scopes(document)
    targets = [page for scope in scopes for page in scope]
    identity = repair_identity()
    cache_dir.mkdir(parents=True, exist_ok=True)
    expected_schema = _schema()
    schema_path = (
        cache_dir
        / f"{PROMPT_VERSION}.{identity['response_schema_sha256'][:16]}.schema.json"
    )
    try:
        cached_schema = _read_strict_json(schema_path)
        if (
            cached_schema != expected_schema
            or _stable_hash(cached_schema) != identity["response_schema_sha256"]
        ):
            raise ValueError("response schema mismatch")
    except (OSError, ValueError):
        _atomic_json(schema_path, expected_schema)
    total_calls = 0
    skipped_pages: list[int] = []
    for target_pages in scopes:
        context = _context(document, target_pages)
        input_hash = _stable_hash(context)
        cache_key = _stable_hash(
            {
                "source_sha256": document.source_sha256,
                "context_hash": input_hash,
                "prompt_version": PROMPT_VERSION,
                "response_schema_sha256": identity["response_schema_sha256"],
                "repairable_diagnostics_sha256": identity[
                    "repairable_diagnostics_sha256"
                ],
                "max_live_calls": MAX_LIVE_CALLS,
                "max_scope_pages": MAX_SCOPE_PAGES,
                "model": model,
                "effort": effort,
            }
        )
        cache_contract = {
            "schema_version": "legalpdf.codex.cache.v1",
            "cache_key": cache_key,
            "model": model,
            "effort": effort,
            "prompt_version": PROMPT_VERSION,
            "response_schema_sha256": identity["response_schema_sha256"],
            "repairable_diagnostics_sha256": identity[
                "repairable_diagnostics_sha256"
            ],
            "repairable_diagnostics": identity["repairable_diagnostics"],
            "context_radius": CONTEXT_RADIUS,
            "max_attempts": MAX_ATTEMPTS,
            "max_live_calls": MAX_LIVE_CALLS,
            "max_scope_pages": MAX_SCOPE_PAGES,
        }
        entry = cache_dir / cache_key
        response_path = entry / "response.json"
        metadata_path = entry / "metadata.json"
        response: dict[str, Any] | None = None
        usage: dict[str, int] = {}
        elapsed = 0.0
        attempts = 0
        error = ""
        if response_path.is_file() or metadata_path.is_file():
            try:
                if not response_path.is_file() or not metadata_path.is_file():
                    raise ValueError("cache publication is incomplete")
                candidate = _read_strict_json(response_path)
                metadata = _read_strict_json(metadata_path)
                if not isinstance(metadata, dict) or any(
                    metadata.get(key) != value
                    for key, value in cache_contract.items()
                ):
                    raise ValueError("cache metadata contract mismatch")
                if metadata.get("response_sha256") != _stable_hash(candidate):
                    raise ValueError("cached response hash mismatch")
                valid, validation_error = _validate(
                    candidate,
                    target_pages=target_pages,
                    expected_line_ids={
                        page: [line.id for line in document.pages[page].lines]
                        for page in target_pages
                    },
                )
                if not valid:
                    raise ValueError(validation_error)
                raw_usage = metadata.get("token_usage")
                if not isinstance(raw_usage, dict) or any(
                    not isinstance(key, str)
                    or not isinstance(value, int)
                    or isinstance(value, bool)
                    or value < 0
                    for key, value in raw_usage.items()
                ):
                    raise ValueError("cache token usage is invalid")
                cached_attempts = metadata.get("attempts")
                cached_elapsed = metadata.get("elapsed_seconds")
                if (
                    not isinstance(cached_attempts, int)
                    or isinstance(cached_attempts, bool)
                    or not 1 <= cached_attempts <= MAX_ATTEMPTS
                    or not isinstance(cached_elapsed, (int, float))
                    or isinstance(cached_elapsed, bool)
                    or not math.isfinite(cached_elapsed)
                    or cached_elapsed < 0
                ):
                    raise ValueError("cache attempt metadata is invalid")
                response = candidate
                usage = dict(raw_usage)
                elapsed = float(cached_elapsed)
                attempts = cached_attempts
            except Exception:
                response = None
                shutil.rmtree(entry, ignore_errors=True)
        if response is None and total_calls >= MAX_LIVE_CALLS:
            skipped_pages.extend(target_pages)
            error = "document live-call budget exhausted"
            document.repairs.append(
                RepairRecord(
                    page_index=target_pages[0],
                    status="skipped",
                    model=model,
                    effort=effort,
                    prompt_version=PROMPT_VERSION,
                    cache_key=cache_key,
                    attempts=0,
                    elapsed_seconds=0.0,
                    input_line_hash=input_hash,
                    error=error,
                    scope_pages=list(target_pages),
                )
            )
            document.diagnostics.append(
                Diagnostic(
                    code="CODEX_REPAIR_BUDGET_EXHAUSTED",
                    severity="warning",
                    message="Codex structural repair skipped because the document live-call budget was exhausted.",
                    page_index=target_pages[0],
                    details={
                        "scope_pages": list(target_pages),
                        "live_calls": total_calls,
                        "max_live_calls": MAX_LIVE_CALLS,
                    },
                )
            )
            continue
        if response is None:
            entry.mkdir(parents=True, exist_ok=True)
            images = _render_context(
                pdf_path,
                target_pages,
                cache_dir / "renders" / document.source_sha256,
            )
            previous_error = error
            allowed_attempts = min(
                MAX_ATTEMPTS, MAX_LIVE_CALLS - total_calls
            )
            for attempts in range(1, allowed_attempts + 1):
                total_calls += 1
                try:
                    candidate, attempt_usage, attempt_elapsed = _invoke(
                        prompt=_prompt(context, previous_error),
                        schema_path=schema_path,
                        image_paths=images,
                        model=model,
                        effort=effort,
                        work_dir=entry,
                        timeout_seconds=timeout_seconds,
                    )
                    elapsed += attempt_elapsed
                    usage = attempt_usage
                    valid, validation_error = _validate(
                        candidate,
                        target_pages=target_pages,
                        expected_line_ids={
                            page: [line.id for line in document.pages[page].lines]
                            for page in target_pages
                        },
                    )
                    if valid:
                        response = candidate
                        error = ""
                        break
                    previous_error = validation_error
                    error = validation_error
                except Exception as exc:
                    previous_error = str(exc)
                    error = str(exc)
            if response is not None:
                _atomic_json(response_path, response)
                _atomic_json(
                    metadata_path,
                    {
                        **cache_contract,
                        "response_sha256": _stable_hash(response),
                        "attempts": attempts,
                        "elapsed_seconds": elapsed,
                        "token_usage": usage,
                    },
                )
        if response is None:
            document.repairs.append(
                RepairRecord(
                    page_index=target_pages[0],
                    status="failed",
                    model=model,
                    effort=effort,
                    prompt_version=PROMPT_VERSION,
                    cache_key=cache_key,
                    attempts=attempts,
                    elapsed_seconds=round(elapsed, 4),
                    input_line_hash=input_hash,
                    token_usage=usage,
                    error=error,
                    scope_pages=list(target_pages),
                )
            )
            document.diagnostics.append(
                Diagnostic(
                    code="CODEX_REPAIR_FAILED",
                    severity="warning",
                    message=f"Codex structural repair failed after {attempts} attempts: {error}",
                    page_index=target_pages[0],
                    details={"scope_pages": list(target_pages)},
                )
            )
            continue
        _replay(document, response)
        output_hash = _stable_hash(response)
        document.repairs.append(
            RepairRecord(
                page_index=target_pages[0],
                status="applied",
                model=model,
                effort=effort,
                prompt_version=PROMPT_VERSION,
                cache_key=cache_key,
                attempts=attempts,
                elapsed_seconds=round(elapsed, 4),
                input_line_hash=input_hash,
                output_hash=output_hash,
                token_usage=usage,
                scope_pages=list(target_pages),
            )
        )
        for diagnostic in document.diagnostics:
            if (
                diagnostic.page_index in target_pages
                and diagnostic.code in _REPAIRABLE
            ):
                diagnostic.severity = "info"
                diagnostic.details = {
                    **diagnostic.details,
                    "codex_repair_applied": True,
                    "repair_output_hash": output_hash,
                }
        for target_page in target_pages:
            document.diagnostics.append(
                Diagnostic(
                    code="CODEX_REPAIR_APPLIED",
                    severity="info",
                    message="Validated Codex structural repair applied.",
                    page_index=target_page,
                    details={
                        "model": model,
                        "effort": effort,
                        "cache_key": cache_key,
                        "scope_pages": list(target_pages),
                    },
                )
            )
    if any(repair.status == "applied" for repair in document.repairs):
        from .core import rebuild_derived

        rebuild_derived(document)
    document.provenance = {
        **document.provenance,
        "codex": {
            "model": model,
            "effort": effort,
            "prompt_version": PROMPT_VERSION,
            "response_schema_sha256": identity["response_schema_sha256"],
            "repairable_diagnostics_sha256": identity[
                "repairable_diagnostics_sha256"
            ],
            "repairable_diagnostics": identity["repairable_diagnostics"],
            "context_radius": CONTEXT_RADIUS,
            "max_attempts": MAX_ATTEMPTS,
            "max_live_calls": MAX_LIVE_CALLS,
            "max_scope_pages": MAX_SCOPE_PAGES,
            "target_pages": targets,
            "skipped_pages": skipped_pages,
            "live_calls": total_calls,
        },
    }
    return document
