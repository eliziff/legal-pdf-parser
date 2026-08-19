#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.deterministic_citations import (
    extract_fields,
    split_footnote,
    split_footnote_recall_first,
)


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(";").strip()


def accepted(row: dict, actual: list[str]) -> bool:
    partitions = [row.get("expected_verbatim_parts") or []]
    partitions.extend(
        item
        for item in row.get("acceptable_partitions") or []
        if isinstance(item, list)
    )
    return [normalized(value) for value in actual] in [
        [normalized(value) for value in partition] for partition in partitions
    ]


def outcome(row: dict, actual: list[str]) -> str:
    if accepted(row, actual):
        return "exact"
    partitions = [row.get("expected_verbatim_parts") or []]
    partitions.extend(
        item
        for item in row.get("acceptable_partitions") or []
        if isinstance(item, list)
    )
    expected_counts = {len(partition) for partition in partitions if partition}
    if len(actual) < min(expected_counts):
        return "under_split"
    if len(actual) > max(expected_counts):
        return "over_split"
    return "boundary_mismatch"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for line in args.gold.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("status") or "").casefold() == "accepted":
            rows.append(row)

    fingerprints = [
        normalized(str(row.get("footnote_text") or "")).casefold()
        for row in rows
    ]
    duplicate_fingerprints = len(fingerprints) - len(set(fingerprints))

    conservative = []
    recall = []
    replacement_eligible = 0
    character_failures = 0
    for row in rows:
        text = str(row.get("footnote_text") or "")
        result = split_footnote(text)
        parts = [part.text for part in result.parts]
        conservative.append(
            outcome(row, parts)
            if result.status == "deterministic_complete"
            else "abstain"
        )
        free = split_footnote_recall_first(text)
        free_parts = [part.text for part in free.parts]
        recall.append(outcome(row, free_parts))
        if (
            result.status == "deterministic_complete"
            and result.parts
            and all(extract_fields(part).status == "complete" for part in result.parts)
            and not re.search(r"\b(?:supra|ibid)\b", " ".join(parts), re.I)
        ):
            replacement_eligible += 1
        if re.sub(r"[\s;]+", "", text) != re.sub(
            r"[\s;]+", "", "".join(free_parts)
        ):
            character_failures += 1

    summary = {
        "schema_version": "legalpdf.alr_splitter_benchmark.v1",
        "claim": "descriptive",
        "aggregate_label": "NOT SCOREABLE",
        "candidate_reference_rows": len(rows),
        "duplicate_fingerprint_count": duplicate_fingerprints,
        "limitations": [
            "accepted status does not establish human adjudicator provenance",
            "the source is failure-selected rather than representative",
            "duplicate normalized texts are reported rather than weighted as independent gold",
        ],
        "conservative": {
            key: conservative.count(key)
            for key in ("exact", "under_split", "over_split", "boundary_mismatch", "abstain")
        },
        "recall_first": {
            key: recall.count(key)
            for key in ("exact", "under_split", "over_split", "boundary_mismatch")
        },
        "replacement_eligible": replacement_eligible,
        "recall_first_character_failures": character_failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
