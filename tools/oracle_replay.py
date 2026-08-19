from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "legalpdf.common-input.v1"
RESULT_SCHEMA = "legalpdf.common-input-result.v1"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract(arguments: argparse.Namespace) -> int:
    import fitz

    from legalpdf import core
    from legalpdf.model import Page

    source = arguments.pdf.resolve()
    pages = []
    separators = []
    metadata = {}
    global_offset = 0
    with fitz.open(source) as pdf:
        if pdf.needs_pass:
            raise ValueError("Encrypted PDFs requiring a password are not supported.")
        metadata = dict(pdf.metadata or {})
        for page_index in range(pdf.page_count):
            raw_page = pdf.load_page(page_index)
            lines, quality = core._extract_native_page(
                raw_page,
                page_index=page_index,
                global_line_offset=global_offset,
            )
            separators.append(core._separator_y(raw_page, lines))
            pages.append(
                Page(
                    id=f"p{page_index + 1:04d}",
                    index=page_index,
                    number=page_index + 1,
                    width=round(float(raw_page.rect.width), 3),
                    height=round(float(raw_page.rect.height), 3),
                    lines=lines,
                    regions=[],
                    source="native",
                    text_quality=quality,
                )
            )
            global_offset += len(lines)
    _atomic_json(
        arguments.output.resolve(),
        {
            "schema_version": SCHEMA,
            "source_name": source.name,
            "source_sha256": _sha256(source),
            "pages": [asdict(page) for page in pages],
            "separators": separators,
            "metadata": metadata,
        },
    )
    return 0


def _prepare(pages: list[Any], separators: Sequence[float | None]) -> list[Any]:
    from legalpdf import core

    diagnostics = []
    core._mark_repeated_furniture(pages)
    for page in pages:
        core._associate_detached_references(page, separators[page.index])
    expected_endnote = None
    continuing_endnote_size = None
    for page in pages:
        diagnostics.extend(
            core._classify_page(
                page,
                separators[page.index],
                continuing_endnotes=expected_endnote is not None,
                expected_endnote=expected_endnote,
                continuing_endnote_size=continuing_endnote_size,
            )
        )
        endnote_lines = [
            line for line in page.lines if line.note_region_mode == "endnote"
        ]
        endnote_numbers = [
            int(match.group("label"))
            for line in endnote_lines
            for match in [core._LABEL_RE.match(line.text)]
            if match and match.group("label").isdigit()
        ]
        if endnote_lines:
            if endnote_numbers:
                expected_endnote = endnote_numbers[-1] + 1
            sizes = [
                core._line_font_size(line)
                for line in endnote_lines
                if core._line_font_size(line) > 0
            ]
            if sizes:
                continuing_endnote_size = statistics.median(sizes)
        else:
            expected_endnote = None
            continuing_endnote_size = None
        diagnostics.extend(core._order_page(page))
        core._build_regions(page)
    diagnostics.extend(core._assign_printed_page_labels(pages))
    return diagnostics


def replay(arguments: argparse.Namespace) -> int:
    from legalpdf import core
    from legalpdf.model import LegalDocument, _page_from_dict

    value = json.loads(arguments.input.read_text(encoding="utf-8"))
    if value.get("schema_version") != SCHEMA:
        raise ValueError(f"unsupported common-input schema: {value.get('schema_version')!r}")
    pages = [_page_from_dict(page) for page in value["pages"]]
    separators = [None if item is None else float(item) for item in value["separators"]]
    if len(separators) != len(pages):
        raise ValueError("common input must contain one separator value per page")
    diagnostics = _prepare(pages, separators)
    prepared_pages = [asdict(page) for page in pages]

    # This call is observation-only. The production _derive call below still
    # executes the canonical pairer and materializer on the same page records.
    core._infer_note_region_modes(pages)
    markers, marker_summary = core._pair_markers(pages)
    paragraphs, footnotes, derived_diagnostics, pairing_summary = core._derive(pages)
    diagnostics.extend(derived_diagnostics)
    sections = core._build_sections(paragraphs)
    document = LegalDocument(
        document_id=f"doc-{value['source_sha256'][:20]}",
        source_name=value["source_name"],
        source_sha256=value["source_sha256"],
        page_count=len(pages),
        status=core._status(diagnostics, pages),
        pages=pages,
        paragraphs=paragraphs,
        sections=sections,
        footnotes=footnotes,
        diagnostics=diagnostics,
        metadata={"pairing": pairing_summary},
        provenance={"common_input_replay": True},
    )
    core._validate_document(document)
    _atomic_json(
        arguments.output.resolve(),
        {
            "schema_version": RESULT_SCHEMA,
            "source_sha256": value["source_sha256"],
            "prepared_pages": prepared_pages,
            "derived_pages": [asdict(page) for page in pages],
            "markers": markers,
            "marker_summary": marker_summary,
            "pairing_summary": pairing_summary,
            "paragraphs": [asdict(item) for item in paragraphs],
            "sections": [asdict(item) for item in sections],
            "footnotes": [asdict(item) for item in footnotes],
            "diagnostics": [asdict(item) for item in diagnostics],
            "status": document.status,
            "validation": "ok",
        },
    )
    return 0


def contract(arguments: argparse.Namespace) -> int:
    from dataclasses import asdict

    from legalpdf.adapters import to_alr_payload, to_toa_text_units
    from legalpdf.core import lookup_artifact_footnote, lookup_footnote
    from legalpdf.deterministic_citations import (
        extract_fields,
        split_footnote,
        split_footnote_recall_first,
    )
    from legalpdf.model import load_artifacts
    from legalpdf.benchmark import extract_docx_gold
    from legalpdf.docx_linking import (
        _validate_response,
        apply_docx_links,
        assess_route,
        deterministic_intents,
        plan_docx_links,
    )
    from legalpdf.codex_repair import (
        _context as repair_context,
        _repair_scopes as repair_scopes,
        _validate as validate_repair_response,
        repair_identity,
    )

    def docx_notes(gold: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "id": note["ooxml_id"],
                "label": note["label"],
                "text": note["body"],
                "proposition": note["passage_since_prior_note"],
            }
            for note in gold["footnotes"]
        ]

    def stable_plan(plan: dict[str, Any]) -> dict[str, Any]:
        plan = json.loads(json.dumps(plan, ensure_ascii=False))
        plan["telemetry"].pop("elapsed_seconds", None)
        for batch_metadata in plan["telemetry"].get("batches", []):
            batch_metadata.pop("elapsed_seconds", None)
        return plan

    def stable_gold(gold: dict[str, Any]) -> dict[str, Any]:
        gold = dict(gold)
        gold.pop("source_sha256", None)
        return gold

    def link_targets(path: Path) -> list[str]:
        import zipfile
        from xml.etree import ElementTree as ET

        relationship_type = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
        )
        with zipfile.ZipFile(path) as archive:
            raw = archive.read("word/_rels/footnotes.xml.rels")
        root = ET.fromstring(raw)
        return sorted(
            {
                relationship.get("Target", "")
                for relationship in root
                if relationship.get("Type") == relationship_type
            }
        )

    value = json.loads(arguments.input.read_text(encoding="utf-8"))
    if value.get("schema_version") != "legalpdf.contract-input.v1":
        raise ValueError("unsupported contract input schema")
    operation = value.get("operation")
    artifact = value.get("artifact")
    if artifact and not Path(artifact).is_absolute():
        artifact = arguments.input.resolve().parent / artifact
    if operation == "separator_contract":
        import numpy as np

        from legalpdf.footnote_separator_scan import classify_separator, scan_gray_page

        scans = []
        for image in value["images"]:
            width = image["width"]
            height = image["height"]
            gray = np.full(
                (height, width), image.get("background", 255), dtype=np.uint8
            )
            for x0, y0, x1, y1, shade in image.get("fills", []):
                gray[y0:y1, x0:x1] = shade
            for (
                y0,
                y1,
                y_step,
                row_height,
                x0,
                x1,
                x_step,
                shade,
                dot_width,
            ) in image.get("dotted_rows", []):
                for base_y in range(y0, y1, y_step):
                    for x in range(x0, x1, x_step):
                        gray[base_y : base_y + row_height, x : x + dot_width] = shade
            scans.append({"id": image.get("id"), "record": scan_gray_page(gray)})
        classifications = []
        for case in value.get("classifications", []):
            separators, status = classify_separator(
                case["rules"],
                case.get("vertical_rules", []),
                min_y_ratio=case.get("min_y_ratio", 0.30),
            )
            classifications.append(
                {
                    "id": case.get("id"),
                    "separators": separators,
                    "status": status,
                }
            )
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "scans": scans,
            "classifications": classifications,
        }
    elif operation == "pairing_support":
        from legalpdf.footnote_pairing_support import (
            LEGAL_CITATION_CUE_RE,
            LEGAL_LABEL_CITATION_CONTINUATION_RE,
            PROTECTED_CITATION_SPAN_RES,
            _has_citation_signal,
            enumerator_interpretations,
            heading_text_plausible,
            parse_heading_ladder,
        )

        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "headings": [
                {"text": text, "plausible": heading_text_plausible(text)}
                for text in value["headings"]
            ],
            "texts": [
                {
                    "text": text,
                    "cue": LEGAL_CITATION_CUE_RE.search(text) is not None,
                    "continuation": LEGAL_LABEL_CITATION_CONTINUATION_RE.match(text)
                    is not None,
                    "signal": _has_citation_signal(text),
                    "protected_spans": [
                        match.span()
                        for _name, pattern in PROTECTED_CITATION_SPAN_RES
                        for match in pattern.finditer(text)
                    ],
                }
                for text in value["texts"]
            ],
            "enumerators": [
                {
                    **item,
                    "interpretations": enumerator_interpretations(
                        item["value"], item["punct"]
                    ),
                }
                for item in value["enumerators"]
            ],
            "ladders": [parse_heading_ladder(ladder) for ladder in value["ladders"]],
        }
    elif operation == "ocr_tsv":
        from legalpdf.ocr import _tsv_lines

        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "lines": [
                asdict(line)
                for line in _tsv_lines(
                    value["tsv"],
                    x_scale=value["x_scale"],
                    y_scale=value["y_scale"],
                    page_width=value["page_width"],
                    page_height=value["page_height"],
                )
            ],
        }
    elif operation == "repair_identity":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "identity": repair_identity(),
        }
    elif operation == "repair_contract":
        document = load_artifacts(artifact)
        targets = value["target_pages"]
        validation = None
        if "response" in value:
            valid, error = validate_repair_response(
                value["response"],
                target_pages=targets,
                expected_line_ids={
                    page: [line.id for line in document.pages[page].lines]
                    for page in targets
                },
            )
            validation = {"valid": valid, "error": error}
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "identity": repair_identity(),
            "scopes": repair_scopes(document),
            "context": repair_context(document, targets),
            "validation": validation,
        }
    elif operation == "docx_intents":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "intents": deterministic_intents(value["footnote_id"], value["text"]),
        }
    elif operation == "docx_route":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "assessment": assess_route(value["footnotes"]),
        }
    elif operation == "docx_validate":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "validated": _validate_response(value["response"], value["records"]),
        }
    elif operation == "docx_extract":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "gold": extract_docx_gold(value["docx"]),
        }
    elif operation == "docx_batch":
        results = []
        for case in value["cases"]:
            gold = extract_docx_gold(case["docx"])
            notes = docx_notes(gold)
            results.append(
                {
                    "docx": case["docx"],
                    "gold": gold,
                    "assessment": assess_route(notes),
                    "intents": [
                        {
                            "id": note["id"],
                            "intents": deterministic_intents(note["id"], note["text"]),
                        }
                        for note in notes
                    ],
                }
            )
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "results": results,
        }
    elif operation == "docx_plan_hybrid":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "plan": stable_plan(plan_docx_links(value["docx"], strategy="hybrid")),
        }
    elif operation == "docx_apply":
        source = Path(value["docx"])
        output = Path(value["output"])
        plan = plan_docx_links(source, strategy="hybrid")
        applied = apply_docx_links(source, plan, value["links"], output)
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "plan": stable_plan(plan),
            "applied": applied,
            "gold": stable_gold(extract_docx_gold(output)),
            "targets": link_targets(output),
        }
    elif operation == "citation_batch":
        results = []
        for case in value["cases"]:
            mode = case.get("mode", "recall_first")
            function = (
                split_footnote
                if mode == "conservative"
                else split_footnote_recall_first
            )
            split = function(case["text"])
            results.append(
                {
                    "mode": mode,
                    "text": case["text"],
                    "split": asdict(split),
                    "fields": [asdict(extract_fields(part)) for part in split.parts],
                }
            )
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "results": results,
        }
    elif operation in {"citation_split", "citation_split_recall_first"}:
        function = (
            split_footnote
            if operation == "citation_split"
            else split_footnote_recall_first
        )
        split = function(value["text"])
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "split": asdict(split),
            "fields": [asdict(extract_fields(part)) for part in split.parts],
        }
    elif operation == "load_document":
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "document": asdict(load_artifacts(artifact)),
        }
    elif operation == "artifact_bytes":
        import tempfile

        from legalpdf.model import write_artifacts

        compact = bool(value.get("compact", False))
        with tempfile.TemporaryDirectory(prefix="legalpdf-contract-") as temporary:
            root = Path(temporary)
            write_artifacts(load_artifacts(artifact), root, compact_pages=compact)
            hashes = {
                path.name: {
                    "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
                for path in sorted(root.iterdir())
                if path.is_file()
            }
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "compact": compact,
            "artifacts": hashes,
        }
    elif operation == "adapters":
        document = load_artifacts(artifact)
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "alr": to_alr_payload(document),
            "toa": to_toa_text_units(document),
        }
    elif operation in {"lookup", "artifact_lookup"}:
        function = lookup_footnote if operation == "lookup" else lookup_artifact_footnote
        target = load_artifacts(artifact) if operation == "lookup" else artifact
        found = function(
            target,
            value["query"],
            page=value.get("page"),
            occurrence=value.get("occurrence"),
            proposition_mode=value.get("proposition_mode", "sentence"),
        )
        result = {
            "schema_version": "legalpdf.contract-result.v1",
            "operation": operation,
            "result": asdict(found),
        }
    else:
        raise ValueError(f"unsupported contract operation: {operation}")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    extract_command = commands.add_parser("extract")
    extract_command.add_argument("pdf", type=Path)
    extract_command.add_argument("--output", type=Path, required=True)
    extract_command.set_defaults(handler=extract)
    replay_command = commands.add_parser("replay")
    replay_command.add_argument("input", type=Path)
    replay_command.add_argument("--output", type=Path, required=True)
    replay_command.set_defaults(handler=replay)
    contract_command = commands.add_parser("contract")
    contract_command.add_argument("input", type=Path)
    contract_command.set_defaults(handler=contract)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parser().parse_args(argv)
    return int(arguments.handler(arguments))


if __name__ == "__main__":
    raise SystemExit(main())
