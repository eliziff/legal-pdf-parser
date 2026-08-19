from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

from .adapters import to_toa_text_units
from .anchored_scan import AnchoredPattern
from .core import parse_pdf
from .model import SCHEMA_VERSION, LegalDocument, load_artifacts

# The neutral-citation pattern's court codes hide behind a 2-char-minimum
# branch the AST walk refuses, so they are supplied as hand anchors: every
# match mandatorily contains one code (lowercased for the anchor search).
# The reporter pattern has no mandatory literal at all and stays a full
# scan inside its AnchoredPattern. Differential in test_anchored_scan.py.
_NEUTRAL_COURTS = (
    "SCC|FC|FCA|ABCA|ABKB|ONCA|ONSC|BCCA|BCSC|QCCA|QCCS|NSCA|NBCA|MBCA|"
    "SKCA|NLCA|PECA|YKCA|NWTCA|NUCA"
)
_CITATION_RES = (
    AnchoredPattern(
        re.compile(
            r"\b(?:18|19|20)\d{2}\s+(?:" + _NEUTRAL_COURTS + r")\s+\d+\b"
        ),
        hand_literals=[code.lower() for code in _NEUTRAL_COURTS.split("|")],
    ),
    AnchoredPattern(re.compile(r"\b\d+\s+[A-Z][A-Za-z.'’& -]{1,50}\s+\d+\b")),
)
_MARKER_RE = re.compile(r"⟦FN:(?P<id>[^⟧]+)⟧")


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore[import-not-found]

        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_memory = kernel32.K32GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        process = kernel32.GetCurrentProcess()
        if get_memory(process, ctypes.byref(counters), counters.cb):
            return int(counters.WorkingSetSize)
        return 0
    try:
        import resource

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError):
        return 0


class _PeakRSS:
    def __init__(self) -> None:
        self.peak = _rss_bytes()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.wait(0.05):
            self.peak = max(self.peak, _rss_bytes())

    def __enter__(self) -> "_PeakRSS":
        self._thread.start()
        return self

    def __exit__(self, *_error: Any) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, _rss_bytes())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _attribute(node: ET.Element, local_name: str) -> str:
    for key, value in node.attrib.items():
        if _local(key) == local_name:
            return value
    return ""


def _paragraph_text(node: ET.Element) -> tuple[str, list[str]]:
    values: list[str] = []
    references: list[str] = []
    for child in node.iter():
        name = _local(child.tag)
        if name == "t":
            values.append(child.text or "")
        elif name == "tab":
            values.append("\t")
        elif name in {"br", "cr"}:
            values.append("\n")
        elif name in {"footnoteReference", "endnoteReference"}:
            note_id = _attribute(child, "id")
            if note_id:
                kind = "footnote" if name == "footnoteReference" else "endnote"
                key = f"{kind}:{note_id}"
                values.append(f"⟦FN:{key}⟧")
                references.append(key)
    return "".join(values), references


def _paragraph_style(node: ET.Element) -> str:
    for child in node.iter():
        if _local(child.tag) == "pStyle":
            return _attribute(child, "val")
    return ""


def _sentence_at(text: str, offset: int) -> str:
    boundary = re.compile(r"[.!?](?:[\"'”’)\]]*)(?=\s|⟦FN:|$)")
    boundaries = list(boundary.finditer(text))
    previous = [match for match in boundaries if match.end() <= offset]
    if previous and not text[previous[-1].end() : offset].strip():
        end = previous[-1].end()
        start = previous[-2].end() if len(previous) > 1 else 0
    else:
        start = previous[-1].end() if previous else 0
        following = next(
            (match for match in boundaries if match.start() >= offset), None
        )
        end = following.end() if following else len(text)
    return _MARKER_RE.sub("", text[start:end]).strip()


def _citations(text: str) -> list[str]:
    found = {
        " ".join(match.group().split())
        for pattern in _CITATION_RES
        for match in pattern.finditer(text)
    }
    return sorted(found)


def extract_docx_gold(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.suffix.casefold() != ".docx" or not source.is_file():
        raise ValueError(f"DOCX not found: {source}")
    with zipfile.ZipFile(source) as archive:
        document_xml = ET.fromstring(archive.read("word/document.xml"))
        paragraphs: list[dict[str, Any]] = []
        reference_order: list[str] = []
        for paragraph in (
            node for node in document_xml.iter() if _local(node.tag) == "p"
        ):
            text, references = _paragraph_text(paragraph)
            if not text.strip():
                continue
            reference_order.extend(references)
            paragraphs.append(
                {
                    "text": text,
                    "style": _paragraph_style(paragraph),
                    "footnote_ids": [
                        key.split(":", 1)[1]
                        for key in references
                        if key.startswith("footnote:")
                    ],
                    "endnote_ids": [
                        key.split(":", 1)[1]
                        for key in references
                        if key.startswith("endnote:")
                    ],
                }
            )
        note_text: dict[str, str] = {}
        for kind, member_name in (
            ("footnote", "word/footnotes.xml"),
            ("endnote", "word/endnotes.xml"),
        ):
            if member_name not in archive.namelist():
                continue
            root = ET.fromstring(archive.read(member_name))
            for note in (
                node for node in root.iter() if _local(node.tag) == kind
            ):
                note_id = _attribute(note, "id")
                try:
                    usable = bool(note_id) and int(note_id) > 0
                except ValueError:
                    usable = False
                if not usable:
                    continue
                text = " ".join(
                    value.strip()
                    for paragraph in (
                        node for node in note.iter() if _local(node.tag) == "p"
                    )
                    for value, _references in [_paragraph_text(paragraph)]
                    if value.strip()
                )
                if text:
                    note_text[f"{kind}:{note_id}"] = text
    unique_order = list(dict.fromkeys(reference_order))
    unique_order.extend(
        note_key for note_key in note_text if note_key not in unique_order
    )
    counters: defaultdict[str, int] = defaultdict(int)
    display_by_key: dict[str, str] = {}
    for note_key in unique_order:
        kind, _note_id = note_key.split(":", 1)
        counters[kind] += 1
        display_by_key[note_key] = str(counters[kind])
    propositions: dict[str, dict[str, str]] = {}
    passage_parts: list[str] = []
    for paragraph in paragraphs:
        previous_offset = 0
        for match in _MARKER_RE.finditer(paragraph["text"]):
            note_key = match.group("id")
            segment = _MARKER_RE.sub(
                "", paragraph["text"][previous_offset : match.start()]
            ).strip()
            if segment:
                passage_parts.append(segment)
            propositions[note_key] = {
                "sentence": _sentence_at(paragraph["text"], match.start()),
                "passage_since_prior_note": "\n\n".join(passage_parts),
            }
            passage_parts = []
            previous_offset = match.end()
        tail = _MARKER_RE.sub("", paragraph["text"][previous_offset:]).strip()
        if tail:
            passage_parts.append(tail)
    notes = [
        {
            "ooxml_id": note_key.split(":", 1)[1],
            "kind": note_key.split(":", 1)[0],
            "label": display_by_key[note_key],
            "occurrence": 1,
            "body": note_text[note_key],
            "sentence_proposition": propositions.get(note_key, {}).get(
                "sentence", ""
            ),
            "passage_since_prior_note": propositions.get(note_key, {}).get(
                "passage_since_prior_note", ""
            ),
        }
        for note_key in unique_order
        if note_key in note_text
    ]
    body_text = "\n\n".join(paragraph["text"] for paragraph in paragraphs)
    return {
        "schema_version": "legalpdf.docx_gold.v2",
        "source_name": source.name,
        "source_sha256": _sha256(source),
        "paragraphs": paragraphs,
        # Kept under the established key because the engine's public model
        # represents footnotes and endnotes as one paired-note collection.
        "footnotes": notes,
        "note_counts": dict(sorted(Counter(note["kind"] for note in notes).items())),
        "citations": _citations(body_text + "\n" + "\n".join(note_text.values())),
    }


def _normalize(text: str) -> str:
    return " ".join(_MARKER_RE.sub("", text).casefold().split())


def _sequence_error(
    gold: Sequence[Any], candidate: Sequence[Any]
) -> tuple[int, str]:
    try:
        from rapidfuzz.distance import Levenshtein as RapidLevenshtein

        return int(RapidLevenshtein.distance(gold, candidate)), "rapidfuzz"
    except ImportError:
        pass
    try:
        import Levenshtein  # type: ignore[import-not-found]

        if isinstance(gold, str) and isinstance(candidate, str):
            return int(Levenshtein.distance(gold, candidate)), "python-Levenshtein"
    except ImportError:
        pass
    matcher = difflib.SequenceMatcher(a=gold, b=candidate, autojunk=True)
    edits = 0
    for operation, a0, a1, b0, b1 in matcher.get_opcodes():
        if operation == "replace":
            edits += max(a1 - a0, b1 - b0)
        elif operation == "delete":
            edits += a1 - a0
        elif operation == "insert":
            edits += b1 - b0
    return edits, "difflib-opcode-approximate"


def _similarity(gold: str, candidate: str) -> float:
    return round(
        difflib.SequenceMatcher(
            a=_normalize(gold), b=_normalize(candidate), autojunk=False
        ).ratio(),
        6,
    )


def _order_metrics(
    gold_paragraphs: Sequence[dict[str, Any]],
    candidate_paragraphs: Sequence[str],
) -> dict[str, Any]:
    def signature(text: str) -> str:
        return " ".join(_normalize(text).split()[:12])

    gold_signatures = [
        signature(str(paragraph.get("text") or "")) for paragraph in gold_paragraphs
    ]
    candidate_positions = {
        signature(text): index
        for index, text in enumerate(candidate_paragraphs)
        if signature(text)
    }
    matches = [
        (gold_index, candidate_positions[value])
        for gold_index, value in enumerate(gold_signatures)
        if value and value in candidate_positions
    ]
    comparable = 0
    correct = 0
    for left in range(len(matches)):
        for right in range(left + 1, len(matches)):
            comparable += 1
            if matches[left][1] < matches[right][1]:
                correct += 1
    adjacent_total = max(0, len(matches) - 1)
    adjacent_correct = sum(
        1
        for left, right in zip(matches, matches[1:])
        if left[1] < right[1]
    )
    exact = sum(1 for gold_index, candidate_index in matches if gold_index == candidate_index)
    return {
        "matched_paragraphs": len(matches),
        "pairwise_order_accuracy": correct / comparable if comparable else None,
        "adjacent_order_recall": (
            adjacent_correct / adjacent_total if adjacent_total else None
        ),
        "exact_position_accuracy": exact / len(matches) if matches else None,
    }


def _structure_metrics(
    gold: dict[str, Any], document: LegalDocument
) -> dict[str, Any] | None:
    gold_regions = gold.get("regions")
    if not isinstance(gold_regions, list) or not gold_regions:
        return None
    expected_type: dict[str, str] = {}
    expected_order: list[str] = []
    expected_boundaries: set[tuple[int, str, tuple[str, ...]]] = set()
    for region in gold_regions:
        if not isinstance(region, dict):
            continue
        page_index = int(region.get("page_index", 0))
        region_type = str(region.get("type") or region.get("region_type") or "unknown")
        line_ids = tuple(str(value) for value in region.get("line_ids", []))
        expected_boundaries.add((page_index, region_type, line_ids))
        for line_id in line_ids:
            expected_type[line_id] = region_type
            expected_order.append(line_id)

    actual_type = {line.id: line.region_type for line in document.lines}
    actual_order = [
        line.id
        for page in document.pages
        for line in sorted(page.lines, key=lambda value: value.reading_order)
    ]
    actual_boundaries = {
        (region.page_index, region.type, tuple(region.line_ids))
        for page in document.pages
        for region in page.regions
    }
    common = set(expected_type) & set(actual_type)
    correct_types = sum(
        expected_type[line_id] == actual_type[line_id] for line_id in common
    )
    boundary_common = expected_boundaries & actual_boundaries
    boundary_precision = (
        len(boundary_common) / len(actual_boundaries) if actual_boundaries else 0.0
    )
    boundary_recall = (
        len(boundary_common) / len(expected_boundaries) if expected_boundaries else 1.0
    )
    boundary_f1 = (
        2
        * boundary_precision
        * boundary_recall
        / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )

    expected_rank = {line_id: index for index, line_id in enumerate(expected_order)}
    observed = [line_id for line_id in actual_order if line_id in expected_rank]
    comparable = 0
    correctly_ordered = 0
    for left_index, left in enumerate(observed):
        for right in observed[left_index + 1 :]:
            comparable += 1
            correctly_ordered += expected_rank[left] < expected_rank[right]
    exact_positions = sum(
        left == right for left, right in zip(expected_order, observed)
    )
    return {
        "gold_lines": len(expected_type),
        "candidate_lines": len(actual_type),
        "covered_lines": len(common),
        "line_coverage": len(common) / len(expected_type) if expected_type else 1.0,
        "region_type_accuracy": (
            correct_types / len(expected_type) if expected_type else 1.0
        ),
        "region_boundary_precision": boundary_precision,
        "region_boundary_recall": boundary_recall,
        "region_boundary_f1": boundary_f1,
        "pairwise_line_order_accuracy": (
            correctly_ordered / comparable if comparable else None
        ),
        "exact_line_position_accuracy": (
            exact_positions / len(expected_order) if expected_order else None
        ),
    }


def _repair_metrics(
    document: LegalDocument, baseline_document: LegalDocument | None
) -> dict[str, Any]:
    repairs = document.repairs
    applied = sum(repair.status == "applied" for repair in repairs)
    before = (
        {line.id: line.text for line in baseline_document.lines}
        if baseline_document is not None
        else None
    )
    after = {line.id: line.text for line in document.lines}
    conserved = (
        sum(before.get(line_id) == text for line_id, text in after.items())
        if before is not None
        else None
    )
    denominator = max(len(before or {}), len(after), 1)
    return {
        "scopes": len(repairs),
        "applied": applied,
        "failed": sum(repair.status != "applied" for repair in repairs),
        "attempts": sum(repair.attempts for repair in repairs),
        "retries": sum(max(0, repair.attempts - 1) for repair in repairs),
        "schema_valid_scope_rate": applied / len(repairs) if repairs else None,
        "model_latency_seconds": round(
            sum(repair.elapsed_seconds for repair in repairs), 4
        ),
        "source_line_conservation": (
            conserved / denominator if conserved is not None else None
        ),
        "source_line_count_before": len(before) if before is not None else None,
        "source_line_count_after": len(after),
    }


def score_docx_gold(
    gold: dict[str, Any],
    document: LegalDocument,
    *,
    baseline_document: LegalDocument | None = None,
) -> dict[str, Any]:
    gold_text = "\n\n".join(
        str(paragraph.get("text") or "") for paragraph in gold["paragraphs"]
    )
    candidate_text = "\n\n".join(paragraph.text for paragraph in document.paragraphs)
    normalized_gold = _normalize(gold_text)
    normalized_candidate = _normalize(candidate_text)
    char_edits, char_backend = _sequence_error(normalized_gold, normalized_candidate)
    gold_words = normalized_gold.split()
    candidate_words = normalized_candidate.split()
    word_edits, word_backend = _sequence_error(gold_words, candidate_words)

    candidate_by_label = {
        (note.label, note.occurrence): note for note in document.footnotes
    }
    gold_keys = {
        (str(note["label"]), int(note.get("occurrence", 1)))
        for note in gold["footnotes"]
    }
    candidate_keys = set(candidate_by_label)
    common = gold_keys & candidate_keys
    body_scores = []
    sentence_scores = []
    passage_scores = []
    for note in gold["footnotes"]:
        key = (str(note["label"]), int(note.get("occurrence", 1)))
        candidate = candidate_by_label.get(key)
        if candidate is None:
            continue
        body_scores.append(_similarity(str(note["body"]), candidate.body))
        sentence_scores.append(
            _similarity(
                str(note.get("sentence_proposition") or ""),
                candidate.sentence_proposition,
            )
        )
        passage_scores.append(
            _similarity(
                str(note.get("passage_since_prior_note") or ""),
                candidate.passage_since_prior_note,
            )
        )
    precision = len(common) / len(candidate_keys) if candidate_keys else 0.0
    recall = len(common) / len(gold_keys) if gold_keys else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    candidate_citations = set(
        _citations(candidate_text + "\n" + "\n".join(note.body for note in document.footnotes))
    )
    gold_citations = set(gold.get("citations", []))
    citation_common = gold_citations & candidate_citations
    result = {
        "schema_version": "legalpdf.benchmark.result.v1",
        "source_sha256": document.source_sha256,
        "status": document.status,
        "text": {
            "characters": len(normalized_gold),
            "character_edits": char_edits,
            "cer": char_edits / max(1, len(normalized_gold)),
            "words": len(gold_words),
            "word_edits": word_edits,
            "wer": word_edits / max(1, len(gold_words)),
            "backend": f"{char_backend}/{word_backend}",
        },
        "order": _order_metrics(
            gold["paragraphs"], [paragraph.text for paragraph in document.paragraphs]
        ),
        "footnotes": {
            "gold": len(gold_keys),
            "candidate": len(candidate_keys),
            "matched": len(common),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_body_similarity": (
                sum(body_scores) / len(body_scores) if body_scores else None
            ),
        },
        "propositions": {
            "mean_sentence_similarity": (
                sum(sentence_scores) / len(sentence_scores)
                if sentence_scores
                else None
            ),
            "mean_passage_similarity": (
                sum(passage_scores) / len(passage_scores)
                if passage_scores
                else None
            ),
        },
        "citations": {
            "gold": len(gold_citations),
            "candidate": len(candidate_citations),
            "matched": len(citation_common),
            "recall": len(citation_common) / len(gold_citations)
            if gold_citations
            else 1.0,
        },
        "application_equivalence": {
            "toa_text_units": len(to_toa_text_units(document)),
        },
        "codex": {
            "calls": int(
                document.provenance.get("codex", {}).get("live_calls", 0)
            ),
            "model": document.provenance.get("codex", {}).get("model"),
            "effort": document.provenance.get("codex", {}).get("effort"),
            "tokens": {
                key: sum(repair.token_usage.get(key, 0) for repair in document.repairs)
                for key in {
                    usage_key
                    for repair in document.repairs
                    for usage_key in repair.token_usage
                }
            },
            "repair": _repair_metrics(document, baseline_document),
        },
    }
    structure = _structure_metrics(gold, document)
    if structure is not None:
        result["structure"] = structure
    return result


def _soffice() -> str:
    executable = (
        shutil.which("soffice.com")
        or shutil.which("soffice")
        or shutil.which("soffice.exe")
    )
    if not executable:
        for windows_default in (
            Path(r"C:\Program Files\LibreOffice\program\soffice.com"),
            Path(r"C:\Program Files\LibreOffice\program\soffice.exe"),
        ):
            if windows_default.is_file():
                executable = str(windows_default)
                break
    if not executable:
        raise RuntimeError("LibreOffice soffice was not found")
    return executable


def _soffice_version(executable: str) -> str:
    completed = subprocess.run(
        [executable, "--version"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        shell=False,
    )
    return (completed.stdout or completed.stderr).strip()


def _libreoffice_export(
    docx: Path, output_dir: Path, *, stripped: bool
) -> Path:
    executable = _soffice()
    filter_name = "pdf:writer_pdf_Export"
    if stripped:
        options = {
            "UseTaggedPDF": {"type": "boolean", "value": "false"},
            "ExportBookmarks": {"type": "boolean", "value": "false"},
            "ExportFormFields": {"type": "boolean", "value": "false"},
            "ExportNotes": {"type": "boolean", "value": "false"},
        }
        filter_name += ":" + json.dumps(options, separators=(",", ":"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="legalpdf-lo-profile-") as profile:
        profile_uri = Path(profile).resolve().as_uri()
        completed = subprocess.run(
            [
                executable,
                f"-env:UserInstallation={profile_uri}",
                "--headless",
                "--convert-to",
                filter_name,
                "--outdir",
                str(output_dir),
                str(docx),
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
            shell=False,
            timeout=180,
        )
    output = output_dir / f"{docx.stem}.pdf"
    if completed.returncode != 0 or not output.is_file():
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"LibreOffice PDF export failed: {message}")
    return output


def _word_path() -> Path | None:
    candidates = (
        Path(r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"),
        Path(r"C:\Program Files (x86)\Microsoft Office\root\Office16\WINWORD.EXE"),
    )
    return next((path for path in candidates if path.is_file()), None)


def _word_export(docx: Path, output: Path, *, profile: str) -> str:
    if os.name != "nt" or _word_path() is None:
        raise RuntimeError("Microsoft Word is not installed")
    script = Path(__file__).resolve().parent / "tools" / "export_docx_word.ps1"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
            "-InputPath",
            str(docx),
            "-OutputPath",
            str(output),
            "-Profile",
            profile,
        ],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        shell=False,
        timeout=180,
    )
    if completed.returncode != 0 or not output.is_file():
        message = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Microsoft Word PDF export failed: {message}")
    return completed.stdout.strip() or "Microsoft Word"


def _flatten_pdf(source: Path, output: Path, *, rasterize: bool) -> None:
    import fitz

    with fitz.open(source) as original, fitz.open() as target:
        for source_page in original:
            page = target.new_page(width=source_page.rect.width, height=source_page.rect.height)
            if rasterize:
                pixmap = source_page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                page.insert_image(page.rect, stream=pixmap.tobytes("png"))
            else:
                page.show_pdf_page(page.rect, original, source_page.number)
        target.save(output, garbage=4, deflate=True)


def _write_export_manifest(output_dir: Path, records: Sequence[dict[str, Any]]) -> Path:
    path = output_dir / "export-manifest.jsonl"
    text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    )
    temporary = path.with_suffix(".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def export_docx_matrix(docx_path: str | Path, output_dir: str | Path) -> Path:
    docx = Path(docx_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(docx)
    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="legalpdf-export-", dir=output) as temporary:
        temporary_root = Path(temporary)
        destinations = {
            "native": output / f"{docx.stem}.native.pdf",
            "print": output / f"{docx.stem}.print.pdf",
        }
        word_error = ""
        if _word_path() is not None:
            try:
                exporter_version = _word_export(
                    docx, destinations["native"], profile="native"
                )
                _word_export(docx, destinations["print"], profile="print")
            except Exception as exc:
                word_error = str(exc)
        if _word_path() is None or word_error:
            executable = _soffice()
            exporter_version = _soffice_version(executable)
            native_source = _libreoffice_export(
                docx, temporary_root / "native", stripped=False
            )
            print_source = _libreoffice_export(
                docx, temporary_root / "print", stripped=True
            )
            shutil.copy2(native_source, destinations["native"])
            shutil.copy2(print_source, destinations["print"])
        destinations["flattened"] = output / f"{docx.stem}.flattened.pdf"
        destinations["rasterized"] = output / f"{docx.stem}.rasterized.pdf"
        _flatten_pdf(
            destinations["print"], destinations["flattened"], rasterize=False
        )
        _flatten_pdf(
            destinations["print"], destinations["rasterized"], rasterize=True
        )
    settings = {
        "native": {"tagged": "exporter default"},
        "print": {
            "UseTaggedPDF": False,
            "ExportBookmarks": False,
            "ExportFormFields": False,
            "ExportNotes": False,
        },
        "flattened": {"source": "print", "method": "PyMuPDF show_pdf_page"},
        "rasterized": {"source": "print", "dpi": 144},
    }
    for profile, pdf in destinations.items():
        records.append(
            {
                "schema_version": "legalpdf.export_manifest.v1",
                "docx": str(docx),
                "docx_sha256": source_hash,
                "pdf": str(pdf),
                "pdf_sha256": _sha256(pdf),
                "profile": profile,
                "exporter": exporter_version,
                "settings": settings[profile],
            }
        )
    return _write_export_manifest(output, records)


def build_docx_corpus(
    input_roots: Sequence[str | Path], output_dir: str | Path
) -> Path:
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidates = sorted(
        {
            path.resolve()
            for raw_root in input_roots
            for path in Path(raw_root).expanduser().resolve().rglob("*.docx")
            if "_temp" not in {part.casefold() for part in path.parts}
        }
    )
    unique: list[tuple[Path, str]] = []
    seen_hashes: set[str] = set()
    for path in candidates:
        fingerprint = _sha256(path)
        if fingerprint not in seen_hashes:
            seen_hashes.add(fingerprint)
            unique.append((path, fingerprint))
    benchmark_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, (docx, fingerprint) in enumerate(unique, start=1):
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "-", docx.stem).strip("-")[:80]
        case_root = output / f"{safe_stem}-{fingerprint[:12]}"
        case_root.mkdir(parents=True, exist_ok=True)
        gold_path = case_root / "gold.json"
        try:
            gold = extract_docx_gold(docx)
            gold_path.write_text(
                json.dumps(gold, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            export_manifest = export_docx_matrix(docx, case_root / "pdf")
            for export in _read_jsonl(export_manifest):
                benchmark_rows.append(
                    {
                        "case_id": f"{fingerprint[:20]}-{export['profile']}",
                        "docx": str(docx),
                        "docx_sha256": fingerprint,
                        "pdf": export["pdf"],
                        "pdf_sha256": export["pdf_sha256"],
                        "gold": str(gold_path),
                        "profile": export["profile"],
                        "exporter": export["exporter"],
                        "settings": export["settings"],
                    }
                )
            print(
                f"{index}/{len(unique)} built {docx.name} "
                f"profiles={len(_read_jsonl(export_manifest))}",
                flush=True,
            )
        except Exception as exc:
            errors.append(
                {
                    "docx": str(docx),
                    "docx_sha256": fingerprint,
                    "error": str(exc),
                }
            )
            print(
                f"{index}/{len(unique)} failed {docx.name}: {exc}",
                flush=True,
            )
    manifest = output / "benchmark-manifest.jsonl"
    manifest.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in benchmark_rows
        ),
        encoding="utf-8",
    )
    (output / "corpus-errors.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in errors
        ),
        encoding="utf-8",
    )
    return manifest


def register_export(
    docx_path: str | Path,
    pdf_path: str | Path,
    output_dir: str | Path,
    *,
    profile: str,
    exporter: str,
    settings: str,
) -> Path:
    docx = Path(docx_path).expanduser().resolve()
    pdf = Path(pdf_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "legalpdf.export_manifest.v1",
        "docx": str(docx),
        "docx_sha256": _sha256(docx),
        "pdf": str(pdf),
        "pdf_sha256": _sha256(pdf),
        "profile": profile,
        "exporter": exporter,
        "settings": json.loads(settings) if settings else {},
    }
    existing_path = output / "export-manifest.jsonl"
    existing = (
        [
            json.loads(line)
            for line in existing_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if existing_path.is_file()
        else []
    )
    return _write_export_manifest(output, [*existing, record])


def write_text_fidelity_product(
    document: LegalDocument,
    output: str | Path,
    *,
    dataset: str,
    article_id: str,
) -> Path:
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for page in document.pages:
        line_by_id = {line.id: line for line in page.lines}
        regions = []
        for region in sorted(page.regions, key=lambda item: item.reading_order):
            regions.append(
                {
                    "type": region.type,
                    "bbox": dict(zip(("x0", "y0", "x1", "y1"), region.bbox)),
                    "lines": [
                        {
                            "source_line_id": line_id,
                            "text": line_by_id[line_id].text,
                            "bbox": dict(
                                zip(
                                    ("x0", "y0", "x1", "y1"),
                                    line_by_id[line_id].bbox,
                                )
                            ),
                        }
                        for line_id in region.line_ids
                        if line_id in line_by_id
                    ],
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "article_id": article_id,
                "pdf_page": page.number,
                "model_tag": "legalpdf",
                "selected_output": "legalpdf",
                "final_surface": SCHEMA_VERSION,
                "final_regions": {
                    "status": "ok",
                    "page_size": [page.width, page.height],
                    "regions": regions,
                },
            }
        )
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return target


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def freeze_manifest(input_path: str | Path, output_path: str | Path, count: int) -> Path:
    source = Path(input_path).expanduser().resolve()
    target = Path(output_path).expanduser().resolve()
    rows = _read_jsonl(source)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        group = "|".join(
            [
                str(row.get("journal") or row.get("dataset") or ""),
                str(row.get("failure_type") or row.get("stratum") or "ordinary"),
            ]
        )
        identity = json.dumps(row, sort_keys=True, ensure_ascii=False)
        row = {**row, "_freeze_hash": hashlib.sha256(identity.encode()).hexdigest()}
        groups[group].append(row)
    for values in groups.values():
        values.sort(key=lambda value: value["_freeze_hash"])
    selected: list[dict[str, Any]] = []
    group_names = sorted(groups)
    while len(selected) < min(count, len(rows)):
        progressed = False
        for group in group_names:
            if groups[group] and len(selected) < count:
                selected.append(groups[group].pop(0))
                progressed = True
        if not progressed:
            break
    selected = [
        {key: value for key, value in row.items() if key != "_freeze_hash"}
        for row in selected
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in selected
        ),
        encoding="utf-8",
    )
    return target


def run_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    mode: str,
    model: str | None,
    effort: str | None,
    cache_dir: str | Path | None,
) -> Path:
    manifest = Path(manifest_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    rows = _read_jsonl(manifest)
    completed: set[str] = set()
    if output.is_file():
        for result in _read_jsonl(output):
            completed.add(str(result.get("case_id") or ""))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows, start=1):
            case_id = str(
                row.get("case_id")
                or hashlib.sha256(
                    json.dumps(row, sort_keys=True).encode()
                ).hexdigest()[:20]
            )
            if case_id in completed:
                print(f"{index}/{len(rows)} skip {case_id}", flush=True)
                continue
            pdf = Path(row["pdf"])
            gold_path = Path(row["gold"])
            if not pdf.is_absolute():
                pdf = manifest.parent / pdf
            if not gold_path.is_absolute():
                gold_path = manifest.parent / gold_path
            started = time.perf_counter()
            with _PeakRSS() as memory:
                document = parse_pdf(
                    pdf,
                    mode=mode,  # type: ignore[arg-type]
                    cache_dir=cache_dir,
                    model=model,
                    effort=effort,
                )
            baseline_document = (
                parse_pdf(
                    pdf,
                    mode="local",
                    cache_dir=cache_dir,
                )
                if mode == "codex"
                else None
            )
            gold = json.loads(gold_path.read_text(encoding="utf-8"))
            result = {
                "case_id": case_id,
                "arm": {"mode": mode, "model": model, "effort": effort},
                "profile": row.get("profile", ""),
                "pdf": str(pdf.resolve()),
                "gold": str(gold_path.resolve()),
                "wall_seconds": round(time.perf_counter() - started, 4),
                "peak_rss_bytes": memory.peak,
                "metrics": score_docx_gold(
                    gold, document, baseline_document=baseline_document
                ),
            }
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            print(
                f"{index}/{len(rows)} complete {case_id} "
                f"profile={row.get('profile', '')} "
                f"live_codex_calls={document.provenance.get('codex', {}).get('live_calls', 0)}",
                flush=True,
            )
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="legalpdf-benchmark",
        description="Build and score reproducible legal-PDF benchmark cases.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    gold = commands.add_parser("docx-gold")
    gold.add_argument("docx", type=Path)
    gold.add_argument("--output", type=Path, required=True)

    export = commands.add_parser("export-docx")
    export.add_argument("docx", type=Path)
    export.add_argument("--output", type=Path, required=True)

    corpus = commands.add_parser("build-docx-corpus")
    corpus.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Directory containing canonical DOCX files; repeatable",
    )
    corpus.add_argument("--output", type=Path, required=True)

    register = commands.add_parser("register-export")
    register.add_argument("docx", type=Path)
    register.add_argument("pdf", type=Path)
    register.add_argument("--output", type=Path, required=True)
    register.add_argument("--profile", required=True)
    register.add_argument("--exporter", required=True)
    register.add_argument("--settings", default="{}")

    score = commands.add_parser("score")
    score.add_argument("document", type=Path)
    score.add_argument("gold", type=Path)

    product = commands.add_parser("text-fidelity-product")
    product.add_argument("document", type=Path)
    product.add_argument("--output", type=Path, required=True)
    product.add_argument("--dataset", required=True)
    product.add_argument("--article-id", required=True)

    freeze = commands.add_parser("freeze-manifest")
    freeze.add_argument("input", type=Path)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--count", type=int, default=80)

    run = commands.add_parser("run")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--mode", choices=("local", "codex"), default="local")
    run.add_argument("--model")
    run.add_argument("--effort")
    run.add_argument("--cache-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    arguments = _parser().parse_args(argv)
    if arguments.command == "docx-gold":
        payload = extract_docx_gold(arguments.docx)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(arguments.output.resolve())
    elif arguments.command == "export-docx":
        print(export_docx_matrix(arguments.docx, arguments.output))
    elif arguments.command == "build-docx-corpus":
        print(build_docx_corpus(arguments.input, arguments.output))
    elif arguments.command == "register-export":
        print(
            register_export(
                arguments.docx,
                arguments.pdf,
                arguments.output,
                profile=arguments.profile,
                exporter=arguments.exporter,
                settings=arguments.settings,
            )
        )
    elif arguments.command == "score":
        document = load_artifacts(arguments.document)
        gold = json.loads(arguments.gold.read_text(encoding="utf-8"))
        print(json.dumps(score_docx_gold(gold, document), indent=2, sort_keys=True))
    elif arguments.command == "text-fidelity-product":
        document = load_artifacts(arguments.document)
        print(
            write_text_fidelity_product(
                document,
                arguments.output,
                dataset=arguments.dataset,
                article_id=arguments.article_id,
            )
        )
    elif arguments.command == "freeze-manifest":
        print(freeze_manifest(arguments.input, arguments.output, arguments.count))
    else:
        print(
            run_manifest(
                arguments.manifest,
                arguments.output,
                mode=arguments.mode,
                model=arguments.model,
                effort=arguments.effort,
                cache_dir=arguments.cache_dir,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
