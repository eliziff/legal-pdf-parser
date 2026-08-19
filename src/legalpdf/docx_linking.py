from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence
from xml.etree import ElementTree as ET

from .benchmark import extract_docx_gold
from .codex_repair import _atomic_json, _invoke, _stable_hash
from .deterministic_citations import extract_fields, split_footnote
from .grammar_tables import lazy_table_entry as _table

PROMPT_VERSION = "legalpdf.docx.citation-intents.v1"
DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_EFFORT = "none"
MAX_FOOTNOTES = 400
MAX_BATCH_FOOTNOTES = 32
MAX_BATCH_CHARS = 45_000
MAX_BATCHES = 13
# Measured on this workstation with Codex CLI 0.145.0. It is deliberately
# configurable because the installed CLI/model context can change.
FIXED_CODEX_TOKENS = int(os.environ.get("LEGALPDF_CODEX_FIXED_TOKENS", "14500"))
MIN_ROUTE_SAVINGS = int(os.environ.get("LEGALPDF_ROUTE_MIN_TOKEN_SAVINGS", "512"))

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"
HYPERLINK_REL = (
    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
)
KINDS = {
    "statute",
    "gazette",
    "case",
    "unreported",
    "parliamentary_paper",
    "non_parliamentary",
    "journal",
    "book",
    "essay_collection",
    "report",
    "other",
}
# Table binds; names unchanged. SUPRA_NOTE_RE's capturing group is named
# "note" in the table and stays group(1) for existing callers.
REFERENCE_RE = _table("ref.token")
SUPRA_NOTE_RE = _table("ref.supra-note.linking")
URL_RE = _table("cite.url.prefix")

ET.register_namespace("w", W_NS)
ET.register_namespace("r", R_NS)
ET.register_namespace("", PKG_REL_NS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _part_id(footnote_id: str, part_index: int) -> str:
    return f"{footnote_id}:{part_index}"


def _intent(
    *,
    part_id: str,
    verbatim: str,
    corrected: str,
    kind: str,
    pinpoint_fragments: Sequence[str],
    page_pinpoints: Sequence[int],
    short_form: str,
    bare_citation: str,
    citation_with_style: str,
    support_quote: str = "",
    route: str,
) -> dict[str, Any]:
    fragments = [str(value).strip() for value in pinpoint_fragments if str(value).strip()]
    pages = [
        int(value)
        for value in page_pinpoints
        if isinstance(value, int) and value > 0
    ]
    locator_kind = "none"
    locator = ""
    first = fragments[0] if fragments else ""
    if first.lower().startswith("par"):
        locator_kind, locator = "paragraph", first[3:]
    elif first.lower().startswith("sec"):
        locator_kind, locator = "section", first[3:]
    elif pages:
        locator_kind, locator = "page", str(pages[0])
    return {
        "part_id": part_id,
        "verbatim": verbatim,
        "corrected": corrected or verbatim,
        "kind": kind if kind in KINDS else "other",
        "pinpoint_fragments": fragments,
        "page_pinpoints": pages,
        "short_form": short_form,
        "bare_citation": bare_citation or verbatim,
        "citation_with_style": citation_with_style or verbatim,
        "support_quote": support_quote,
        "locator_kind": locator_kind,
        "locator": locator,
        "route": route,
    }


def deterministic_intents(footnote_id: str, text: str) -> list[dict[str, Any]] | None:
    """Mirror ALR Ultra Economy's safe replacement gate, without emitting URLs."""
    result = split_footnote(text)
    if result.status != "deterministic_complete" or not result.parts:
        return None
    fields = [extract_fields(part) for part in result.parts]
    if any(field.status != "complete" for field in fields):
        return None
    if any(REFERENCE_RE.search(part.text) for part in result.parts):
        return None
    # Unlike ALR's in-process replacement gate, this plan intentionally has
    # no URL yet: Mike's provider layer verifies every complete identity next.
    if any(
        field.kind in {"case", "unreported", "statute"}
        and not field.bare_citation.strip()
        for field in fields
    ):
        return None
    return [
        _intent(
            part_id=_part_id(footnote_id, index),
            verbatim=part.text,
            corrected=field.corrected,
            kind=field.kind,
            pinpoint_fragments=field.pinpoint_fragments,
            page_pinpoints=field.page_pinpoints,
            short_form=field.short_form,
            bare_citation=field.bare_citation,
            citation_with_style=field.citation_with_style,
            route="deterministic",
        )
        for index, (part, field) in enumerate(zip(result.parts, fields), start=1)
    ]


def _schema() -> dict[str, Any]:
    part = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "verbatim",
            "corrected",
            "kind",
            "pinpoint_fragments",
            "page_pinpoints",
            "short_form",
            "bare_citation",
            "citation_with_style",
            "support_quote",
        ],
        "properties": {
            "verbatim": {"type": "string", "minLength": 1},
            "corrected": {"type": "string"},
            "kind": {"type": "string", "enum": sorted(KINDS)},
            "pinpoint_fragments": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "string", "maxLength": 80},
            },
            "page_pinpoints": {
                "type": "array",
                "maxItems": 20,
                "items": {"type": "integer", "minimum": 1},
            },
            "short_form": {"type": "string", "maxLength": 240},
            "bare_citation": {"type": "string", "maxLength": 1000},
            "citation_with_style": {"type": "string", "maxLength": 1600},
            "support_quote": {"type": "string", "maxLength": 1200},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": ["results"],
        "properties": {
            "results": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "parts"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "parts": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 20,
                            "items": part,
                        },
                    },
                },
            }
        },
    }


def _prompt(records: Sequence[Mapping[str, Any]]) -> str:
    return (
        "You are a bounded citation-intent worker for legal DOCX footnotes. "
        "For every record, split its footnote into source-level parts using the "
        "McGill-style rule: top-level semicolons normally split sources, missing "
        "semicolons between distinct authorities still split, and semicolons "
        "inside one citation do not. Isolate every supra or ibid reference. "
        "Preserve each verbatim part as an exact, non-overlapping substring; do "
        "not invent or drop characters. Classify the source and extract only "
        "compact deterministic lookup fields. pinpoint_fragments use parN for "
        "case paragraphs and secN for legislation sections/rules/articles; keep "
        "all separate pinpoints but only the first endpoint of a range. "
        "page_pinpoints contains integer reporter/PDF pages, never paragraph "
        "numbers. support_quote is either an exact quotation copied from the "
        "record's proposition/footnote that the cited source is said to support, "
        "or an empty string. NEVER output or construct a URL. Mike resolves every "
        "identity and locator through verified provider tools after this call. "
        "Return each requested id exactly once and no commentary.\n\nINPUT:\n"
        + json.dumps({"records": list(records)}, ensure_ascii=False, sort_keys=True)
    )


def _norm_with_map(value: str) -> tuple[str, list[int]]:
    out: list[str] = []
    positions: list[int] = []
    previous_space = True
    translate = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
    for index, character in enumerate(value):
        if character.isspace():
            if previous_space:
                continue
            out.append(" ")
            positions.append(index)
            previous_space = True
        else:
            out.append(translate.get(character, character).lower())
            positions.append(index)
            previous_space = False
    while out and out[-1] == " ":
        out.pop()
        positions.pop()
    return "".join(out), positions


def _snap_parts(text: str, parts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized, positions = _norm_with_map(text)
    spans: list[list[int]] = []
    cursor = 0
    for part in parts:
        wanted, _ = _norm_with_map(str(part.get("verbatim") or ""))
        wanted = wanted.strip()
        if not wanted:
            raise ValueError("worker returned an empty citation part")
        start = normalized.find(wanted, cursor)
        if start < 0:
            start = normalized.find(wanted)
        if start < 0:
            raise ValueError("worker part is not an exact footnote substring")
        spans.append([start, start + len(wanted)])
        cursor = start + len(wanted)
    for left, right in zip(spans, spans[1:]):
        if left[1] > right[0]:
            left[1] = right[0]
        if left[1] <= left[0]:
            raise ValueError("worker returned overlapping citation parts")
    if spans:
        trailing = normalized[spans[-1][1] :]
        if trailing and re.fullmatch(r"""[.,:!?'"\]\)}â€™â€]+""", trailing):
            spans[-1][1] = len(normalized)
    snapped: list[dict[str, Any]] = []
    for part, (start, end) in zip(parts, spans):
        if end <= start:
            raise ValueError("worker returned an empty citation span")
        item = dict(part)
        item["verbatim"] = text[positions[start] : positions[end - 1] + 1].strip()
        snapped.append(item)
    source_core = re.sub(r"[\s;]+", "", text)
    actual_core = re.sub(
        r"[\s;]+", "", "".join(str(part["verbatim"]) for part in snapped)
    )
    if source_core != actual_core:
        raise ValueError("worker split lost, gained, or reordered footnote characters")
    return snapped


def _validate_response(
    response: Any,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if URL_RE.search(json.dumps(response, ensure_ascii=False)):
        raise ValueError("worker output contains a URL")
    if not isinstance(response, dict) or set(response) != {"results"}:
        raise ValueError("worker response has the wrong top-level shape")
    results = response["results"]
    if not isinstance(results, list):
        raise ValueError("worker results is not an array")
    record_by_id = {str(record["id"]): record for record in records}
    result_ids = [
        str(item.get("id") or "") for item in results if isinstance(item, dict)
    ]
    if len(result_ids) != len(set(result_ids)) or set(result_ids) != set(record_by_id):
        raise ValueError("worker result ids do not exactly match the request")
    validated: dict[str, list[dict[str, Any]]] = {}
    for raw in results:
        if not isinstance(raw, dict) or set(raw) != {"id", "parts"}:
            raise ValueError("worker result has an unsupported property")
        record = record_by_id[str(raw["id"])]
        parts = raw["parts"]
        if not isinstance(parts, list) or not 1 <= len(parts) <= 20:
            raise ValueError("worker returned an invalid part count")
        snapped = _snap_parts(str(record["text"]), parts)
        allowed_quote_text = " ".join(
            [str(record["text"]), str(record.get("proposition") or "")]
        )
        for part in snapped:
            if set(part) != {
                "verbatim",
                "corrected",
                "kind",
                "pinpoint_fragments",
                "page_pinpoints",
                "short_form",
                "bare_citation",
                "citation_with_style",
                "support_quote",
            }:
                raise ValueError("worker part has an unsupported property")
            if part["kind"] not in KINDS:
                raise ValueError("worker returned an unsupported citation kind")
            quote = str(part["support_quote"] or "").strip()
            if quote and quote not in allowed_quote_text:
                raise ValueError("worker support_quote is not copied from the input")
        validated[str(raw["id"])] = snapped
    return validated


def _batch(records: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    batches: list[list[Mapping[str, Any]]] = []
    current: list[Mapping[str, Any]] = []
    chars = 0
    for record in records:
        size = len(str(record.get("text") or "")) + len(
            str(record.get("proposition") or "")
        )
        if current and (
            len(current) >= MAX_BATCH_FOOTNOTES or chars + size > MAX_BATCH_CHARS
        ):
            batches.append(current)
            current, chars = [], 0
        current.append(record)
        chars += size
    if current:
        batches.append(current)
    if len(batches) > MAX_BATCHES:
        raise ValueError(
            f"citation linking requires {len(batches)} Codex batches; limit is {MAX_BATCHES}"
        )
    return batches


def _token_estimate(records: Sequence[Mapping[str, Any]]) -> int:
    batches = _batch(records)
    chars = sum(len(_prompt(batch)) for batch in batches)
    return len(batches) * FIXED_CODEX_TOKENS + math.ceil(chars / 4)


def assess_route(footnotes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    deterministic: dict[str, list[dict[str, Any]]] = {}
    fallback: list[Mapping[str, Any]] = []
    for note in footnotes:
        note_id = str(note["id"])
        intents = deterministic_intents(note_id, str(note["text"]))
        if intents is None:
            fallback.append(note)
        else:
            deterministic[note_id] = intents
    direct_tokens = _token_estimate(footnotes)
    hybrid_tokens = _token_estimate(fallback) if fallback else 0
    savings = direct_tokens - hybrid_tokens
    return {
        "recommended_strategy": (
            "hybrid"
            if deterministic and savings >= MIN_ROUTE_SAVINGS
            else "direct"
        ),
        "footnote_count": len(footnotes),
        "deterministic_count": len(deterministic),
        "fallback_count": len(fallback),
        "estimated_direct_tokens": direct_tokens,
        "estimated_hybrid_tokens": hybrid_tokens,
        "estimated_token_savings": savings,
        "fixed_codex_tokens_per_batch": FIXED_CODEX_TOKENS,
        "minimum_route_savings": MIN_ROUTE_SAVINGS,
        "_deterministic": deterministic,
        "_fallback": fallback,
    }


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir:
        return Path(cache_dir).expanduser().resolve()
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    return base / "OpenLegalProducts" / "LegalData" / "cache" / "docx-linking"


def _invoke_batch(
    records: Sequence[Mapping[str, Any]],
    *,
    model: str,
    effort: str,
    cache_dir: Path,
    timeout_seconds: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    prompt = _prompt(records)
    key = _stable_hash(
        {
            "prompt_version": PROMPT_VERSION,
            "prompt": prompt,
            "model": model,
            "effort": effort,
        }
    )
    entry = cache_dir / key
    response_path = entry / "last-message.json"
    metadata_path = entry / "metadata.json"
    if response_path.is_file() and metadata_path.is_file():
        response = json.loads(response_path.read_text(encoding="utf-8"))
        validated = _validate_response(response, records)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return validated, {**metadata, "cache_hit": True}

    entry.mkdir(parents=True, exist_ok=True)
    schema_path = cache_dir / f"{PROMPT_VERSION}.schema.json"
    if not schema_path.is_file():
        _atomic_json(schema_path, _schema())
    response, usage, elapsed = _invoke(
        prompt=prompt,
        schema_path=schema_path,
        image_paths=[],
        model=model,
        effort=effort,
        work_dir=entry,
        timeout_seconds=timeout_seconds,
    )
    validated = _validate_response(response, records)
    metadata = {
        "schema_version": "legalpdf.docx_link_batch.v1",
        "prompt_version": PROMPT_VERSION,
        "cache_key": key,
        "model": model,
        "effort": effort,
        "elapsed_seconds": round(elapsed, 4),
        "token_usage": usage,
        "record_count": len(records),
        "input_chars": len(prompt),
        "cache_hit": False,
    }
    _atomic_json(metadata_path, metadata)
    return validated, metadata


def _resolve_references(footnotes: list[dict[str, Any]]) -> None:
    by_label = {str(note.get("label") or ""): note for note in footnotes}
    previous: dict[str, Any] | None = None
    for note in footnotes:
        for part in note["parts"]:
            text = str(part["verbatim"])
            origin: dict[str, Any] | None = None
            if re.search(r"\bibid\b", text, re.I):
                origin = previous
            else:
                match = SUPRA_NOTE_RE.search(text)
                if match and match.group(1):
                    candidates = by_label.get(match.group(1), {}).get("parts", [])
                    hint = text[: match.start()].strip(" ,.;[]()")
                    matching = [
                        candidate
                        for candidate in candidates
                        if hint
                        and hint.casefold()
                        in " ".join(
                            [
                                str(candidate.get("short_form") or ""),
                                str(candidate.get("citation_with_style") or ""),
                            ]
                        ).casefold()
                    ]
                    origin = (
                        matching[0]
                        if len(matching) == 1
                        else candidates[0]
                        if len(candidates) == 1
                        else None
                    )
            if origin:
                for key in ("kind", "bare_citation", "citation_with_style"):
                    part[key] = origin[key]
                part["origin_part_id"] = origin["part_id"]
            else:
                part["origin_part_id"] = ""
            if part["kind"] != "other" or not REFERENCE_RE.search(text):
                previous = part


def plan_footnotes(
    notes: Sequence[Mapping[str, Any]],
    *,
    strategy: Literal["auto", "direct", "hybrid"] = "auto",
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    cache_dir: str | Path | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    normalized_notes = [
        {
            "id": str(note["id"]),
            "label": str(note.get("label") or note["id"]),
            "text": str(note["text"]),
            "proposition": str(note.get("proposition") or ""),
        }
        for note in notes[: MAX_FOOTNOTES + 1]
    ]
    if len(normalized_notes) > MAX_FOOTNOTES:
        raise ValueError(f"DOCX has more than {MAX_FOOTNOTES} linkable footnotes")
    assessment = assess_route(normalized_notes)
    selected = (
        assessment["recommended_strategy"] if strategy == "auto" else strategy
    )
    deterministic = (
        assessment["_deterministic"] if selected == "hybrid" else {}
    )
    model_records = (
        assessment["_fallback"] if selected == "hybrid" else normalized_notes
    )
    model_results: dict[str, list[dict[str, Any]]] = {}
    telemetry: list[dict[str, Any]] = []
    started = time.perf_counter()
    for batch in _batch(model_records):
        results, metadata = _invoke_batch(
            batch,
            model=model,
            effort=effort,
            cache_dir=_cache_root(cache_dir),
            timeout_seconds=timeout_seconds,
        )
        model_results.update(results)
        telemetry.append(metadata)
    planned: list[dict[str, Any]] = []
    for note in normalized_notes:
        note_id = str(note["id"])
        raw_parts = deterministic.get(note_id) or [
            _intent(
                part_id=_part_id(note_id, index),
                verbatim=str(part["verbatim"]),
                corrected=str(part["corrected"]),
                kind=str(part["kind"]),
                pinpoint_fragments=part["pinpoint_fragments"],
                page_pinpoints=part["page_pinpoints"],
                short_form=str(part["short_form"]),
                bare_citation=str(part["bare_citation"]),
                citation_with_style=str(part["citation_with_style"]),
                support_quote=str(part["support_quote"]),
                route="codex",
            )
            for index, part in enumerate(model_results[note_id], start=1)
        ]
        planned.append({**note, "parts": raw_parts})
    _resolve_references(planned)
    token_usage: dict[str, int] = {}
    for batch in telemetry:
        for key, value in batch.get("token_usage", {}).items():
            token_usage[key] = token_usage.get(key, 0) + int(value)
    return {
        "schema_version": "legalpdf.footnote_link_plan.v1",
        "model": model,
        "effort": effort,
        "strategy_requested": strategy,
        "strategy_used": selected,
        "assessment": {
            key: value
            for key, value in assessment.items()
            if not key.startswith("_")
        },
        "footnotes": planned,
        "telemetry": {
            "elapsed_seconds": round(time.perf_counter() - started, 4),
            "codex_batches": len(telemetry),
            "live_codex_batches": sum(
                not bool(item.get("cache_hit")) for item in telemetry
            ),
            "token_usage": token_usage,
            "batches": telemetry,
        },
    }


def plan_docx_links(
    docx_path: str | Path,
    *,
    strategy: Literal["auto", "direct", "hybrid"] = "auto",
    model: str = DEFAULT_MODEL,
    effort: str = DEFAULT_EFFORT,
    cache_dir: str | Path | None = None,
    timeout_seconds: int = 600,
) -> dict[str, Any]:
    source = Path(docx_path).expanduser().resolve()
    gold = extract_docx_gold(source)
    plan = plan_footnotes(
        [
            {
                "id": note["ooxml_id"],
                "label": note["label"],
                "text": note["body"],
                "proposition": note["passage_since_prior_note"],
            }
            for note in gold["footnotes"]
        ],
        strategy=strategy,
        model=model,
        effort=effort,
        cache_dir=cache_dir,
        timeout_seconds=timeout_seconds,
    )
    return {
        **plan,
        "schema_version": "legalpdf.docx_link_plan.v1",
        "source": str(source),
        "source_sha256": _sha256(source),
    }


def _simple_run(run: ET.Element) -> bool:
    return all(
        child.tag in {f"{{{W_NS}}}rPr", f"{{{W_NS}}}t"} for child in run
    )


def _run_text(run: ET.Element) -> str:
    return "".join(
        node.text or "" for node in run.iter(f"{{{W_NS}}}t")
    )


def _new_run(source: ET.Element, text: str) -> ET.Element:
    run = ET.Element(f"{{{W_NS}}}r")
    properties = source.find(f"{{{W_NS}}}rPr")
    if properties is not None:
        run.append(deepcopy(properties))
    value = ET.SubElement(run, f"{{{W_NS}}}t")
    if text[:1].isspace() or text[-1:].isspace():
        value.set(f"{{{XML_NS}}}space", "preserve")
    value.text = text
    return run


def _link_paragraph(
    paragraph: ET.Element,
    spans: Sequence[tuple[int, int, str]],
    relationship_ids: Mapping[str, str],
) -> int:
    children = list(paragraph)
    cursor = 0
    linked = 0
    rebuilt: list[ET.Element] = []
    for child in children:
        if child.tag != f"{{{W_NS}}}r":
            rebuilt.append(child)
            continue
        text = _run_text(child)
        start, end = cursor, cursor + len(text)
        cursor = end
        intersections = [
            (max(start, left), min(end, right), url)
            for left, right, url in spans
            if max(start, left) < min(end, right)
        ]
        if not intersections:
            rebuilt.append(child)
            continue
        if not _simple_run(child):
            raise ValueError("citation crosses a complex Word run")
        local = 0
        for left, right, url in sorted(intersections):
            left -= start
            right -= start
            if left > local:
                rebuilt.append(_new_run(child, text[local:left]))
            hyperlink = ET.Element(
                f"{{{W_NS}}}hyperlink",
                {f"{{{R_NS}}}id": relationship_ids[url]},
            )
            hyperlink.append(_new_run(child, text[left:right]))
            rebuilt.append(hyperlink)
            linked += 1
            local = right
        if local < len(text):
            rebuilt.append(_new_run(child, text[local:]))
    paragraph[:] = rebuilt
    return linked


def _relationship_document(raw: bytes | None) -> ET.Element:
    return (
        ET.fromstring(raw)
        if raw
        else ET.Element(f"{{{PKG_REL_NS}}}Relationships")
    )


def apply_docx_links(
    docx_path: str | Path,
    plan: Mapping[str, Any] | str | Path,
    resolved_links: Mapping[str, str],
    output_path: str | Path,
) -> dict[str, Any]:
    source = Path(docx_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    payload = (
        json.loads(Path(plan).read_text(encoding="utf-8"))
        if isinstance(plan, (str, Path))
        else dict(plan)
    )
    if payload.get("source_sha256") != _sha256(source):
        raise ValueError("link plan does not match the DOCX bytes")
    links = {
        str(key): str(url)
        for key, url in resolved_links.items()
        if isinstance(url, str) and url.startswith(("https://", "http://"))
    }
    if any(len(url) > 8000 for url in links.values()):
        raise ValueError("resolved provider URL is too long")

    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        files = {name: archive.read(name) for name in names}
    if "word/footnotes.xml" not in files:
        raise ValueError("DOCX has no footnotes.xml")
    footnotes = ET.fromstring(files["word/footnotes.xml"])
    rel_path = "word/_rels/footnotes.xml.rels"
    relationships = _relationship_document(files.get(rel_path))
    existing = {
        rel.get("Target", ""): rel.get("Id", "")
        for rel in relationships
        if rel.get("Type") == HYPERLINK_REL
    }
    used_ids = {rel.get("Id", "") for rel in relationships}
    relationship_ids: dict[str, str] = {}
    for url in sorted(set(links.values())):
        if url in existing:
            relationship_ids[url] = existing[url]
            continue
        number = 1
        while f"rId{number}" in used_ids:
            number += 1
        rel_id = f"rId{number}"
        used_ids.add(rel_id)
        relationship_ids[url] = rel_id
        ET.SubElement(
            relationships,
            f"{{{PKG_REL_NS}}}Relationship",
            {
                "Id": rel_id,
                "Type": HYPERLINK_REL,
                "Target": url,
                "TargetMode": "External",
            },
        )

    notes_by_id = {
        str(note.get(f"{{{W_NS}}}id") or ""): note
        for note in footnotes.findall(f"{{{W_NS}}}footnote")
    }
    linked_parts = 0
    skipped_parts = 0
    for note in payload.get("footnotes", []):
        node = notes_by_id.get(str(note.get("id") or ""))
        if node is None:
            skipped_parts += len(note.get("parts", []))
            continue
        paragraphs = node.findall(f".//{{{W_NS}}}p")
        found_part_ids: set[str] = set()
        for paragraph in paragraphs:
            text = "".join(_run_text(run) for run in paragraph.findall(f"{{{W_NS}}}r"))
            spans: list[tuple[int, int, str]] = []
            cursor = 0
            for part in note.get("parts", []):
                part_id = str(part.get("part_id") or "")
                url = links.get(part_id)
                verbatim = str(part.get("verbatim") or "")
                start = text.find(verbatim, cursor)
                if start < 0:
                    start = text.find(verbatim)
                if start >= 0:
                    cursor = start + len(verbatim)
                    if url:
                        spans.append((start, cursor, url))
                        found_part_ids.add(part_id)
            if spans:
                _link_paragraph(paragraph, spans, relationship_ids)
                linked_parts += len(spans)
        skipped_parts += sum(
            str(part.get("part_id") or "") in links
            and str(part.get("part_id") or "") not in found_part_ids
            for part in note.get("parts", [])
        )

    files["word/footnotes.xml"] = ET.tostring(
        footnotes, encoding="utf-8", xml_declaration=True
    )
    files[rel_path] = ET.tostring(
        relationships, encoding="utf-8", xml_declaration=True
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".docx", dir=target.parent
    )
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        os.replace(temporary, target)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return {
        "output": str(target),
        "linked_parts": linked_parts,
        "skipped_parts": skipped_parts,
        "resolved_link_count": len(links),
    }
