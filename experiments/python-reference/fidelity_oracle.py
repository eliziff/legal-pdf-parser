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


INPUT_SCHEMA = "legalpdf.common-input.v1"
RESULT_SCHEMA = "legalpdf.common-input-result.v1"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path: Path) -> str:
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
    offset = 0
    with fitz.open(source) as pdf:
        if pdf.needs_pass:
            raise ValueError("password-protected PDFs are unsupported")
        metadata = dict(pdf.metadata or {})
        for page_index in range(pdf.page_count):
            raw_page = pdf.load_page(page_index)
            lines, quality = core._extract_native_page(
                raw_page, page_index=page_index, global_line_offset=offset
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
            offset += len(lines)
    atomic_json(
        arguments.output.resolve(),
        {
            "schema_version": INPUT_SCHEMA,
            "source_name": source.name,
            "source_sha256": sha256(source),
            "pages": [asdict(page) for page in pages],
            "separators": separators,
            "metadata": metadata,
        },
    )
    return 0


def prepare(pages: list[Any], separators: Sequence[float | None]) -> list[Any]:
    from legalpdf import core

    diagnostics = []
    core._mark_repeated_furniture(pages)
    for page in pages:
        core._associate_detached_references(page, separators[page.index])
    expected_endnote = None
    continuing_size = None
    for page in pages:
        diagnostics.extend(
            core._classify_page(
                page,
                separators[page.index],
                continuing_endnotes=expected_endnote is not None,
                expected_endnote=expected_endnote,
                continuing_endnote_size=continuing_size,
            )
        )
        endnote_lines = [line for line in page.lines if line.note_region_mode == "endnote"]
        numbers = [
            int(match.group("label"))
            for line in endnote_lines
            for match in [core._LABEL_RE.match(line.text)]
            if match and match.group("label").isdigit()
        ]
        if endnote_lines:
            if numbers:
                expected_endnote = numbers[-1] + 1
            sizes = [core._line_font_size(line) for line in endnote_lines]
            sizes = [size for size in sizes if size > 0]
            if sizes:
                continuing_size = statistics.median(sizes)
        else:
            expected_endnote = None
            continuing_size = None
        diagnostics.extend(core._order_page(page))
        core._build_regions(page)
    diagnostics.extend(core._assign_printed_page_labels(pages))
    return diagnostics


def replay(arguments: argparse.Namespace) -> int:
    from legalpdf import core
    from legalpdf.model import LegalDocument, _page_from_dict

    value = json.loads(arguments.input.read_text(encoding="utf-8"))
    if value.get("schema_version") != INPUT_SCHEMA:
        raise ValueError("unsupported common-input schema")
    pages = [_page_from_dict(page) for page in value["pages"]]
    separators = [None if item is None else float(item) for item in value["separators"]]
    if len(separators) != len(pages):
        raise ValueError("common input must contain one separator per page")
    diagnostics = prepare(pages, separators)
    prepared_pages = [asdict(page) for page in pages]
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
    )
    core._validate_document(document)
    atomic_json(
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


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("pdf", type=Path)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.set_defaults(handler=extract)
    replay_parser = commands.add_parser("replay")
    replay_parser.add_argument("input", type=Path)
    replay_parser.add_argument("--output", type=Path, required=True)
    replay_parser.set_defaults(handler=replay)
    return root


if __name__ == "__main__":
    arguments = parser().parse_args()
    raise SystemExit(arguments.handler(arguments))
