from __future__ import annotations

import difflib
import hashlib
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence
from xml.etree import ElementTree as ET

from .adapters import to_toa_text_units
from .anchored_scan import AnchoredPattern
from .model import LegalDocument

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
