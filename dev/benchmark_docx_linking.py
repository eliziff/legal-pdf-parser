#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.benchmark import extract_docx_gold
from legalpdf.docx_linking import deterministic_intents, plan_footnotes


def normalized(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().rstrip(";").strip()


def core(value: str) -> str:
    return re.sub(r"[\s;]+", "", normalized(value))


def accepted_partitions(row: dict[str, Any]) -> list[list[str]]:
    values = [row.get("expected_verbatim_parts") or []]
    values.extend(
        item
        for item in row.get("acceptable_partitions") or []
        if isinstance(item, list)
    )
    return [[normalized(str(part)) for part in parts] for parts in values]


def score(expected: dict[str, dict[str, Any]], plan: dict[str, Any]) -> dict[str, Any]:
    notes = plan.get("footnotes")
    if not isinstance(notes, list):
        raise ValueError("plan must contain a footnotes array")
    actual_ids = [str(note.get("id") or "") for note in notes]
    if (
        not all(actual_ids)
        or len(actual_ids) != len(set(actual_ids))
        or set(actual_ids) != set(expected)
    ):
        raise ValueError("prediction IDs must form an exact bijection with gold IDs")
    details = []
    for note in notes:
        row = expected[str(note["id"])]
        actual = [normalized(part["verbatim"]) for part in note["parts"]]
        partitions = accepted_partitions(row)
        expected_counts = {len(partition) for partition in partitions if partition}
        if not expected_counts:
            raise ValueError(f"reference row has no accepted partition: {note['id']}")
        details.append(
            {
                "id": note["id"],
                "gold_id": row.get("id"),
                "expected": partitions[0],
                "actual": actual,
                "exact": actual == partitions[0],
                "tolerant_exact": actual in partitions,
                "character_neutral": core(row["footnote_text"]) == core("".join(actual)),
                "over_split": len(actual) > max(expected_counts),
                "under_split": len(actual) < min(expected_counts),
            }
        )
    total = len(details)
    return {
        "cases": total,
        "exact": sum(item["exact"] for item in details),
        "exact_rate": round(sum(item["exact"] for item in details) / total, 4),
        "tolerant_exact": sum(item["tolerant_exact"] for item in details),
        "tolerant_exact_rate": round(
            sum(item["tolerant_exact"] for item in details) / total, 4
        ),
        "character_neutral": sum(item["character_neutral"] for item in details),
        "over_splits": sum(item["over_split"] for item in details),
        "under_splits": sum(item["under_split"] for item in details),
        "details": details,
    }


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    ambiguous: set[str] = set()
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("status") or "").casefold() != "accepted":
            continue
        text = normalized(
            row.get("verbatim_footnote_text") or row.get("footnote_text") or ""
        ).casefold()
        if text in rows:
            ambiguous.add(text)
        elif text:
            rows[text] = row
    for text in ambiguous:
        rows.pop(text, None)
    return rows


def select_cases(
    docx: Path, gold_path: Path, sample_size: int, seed: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    gold = load_gold(gold_path)
    candidates = []
    for note in extract_docx_gold(docx)["footnotes"]:
        row = gold.get(normalized(note["body"]).casefold())
        if not row:
            continue
        record = {
            "id": str(note["ooxml_id"]),
            "label": str(note["label"]),
            "text": str(note["body"]),
            "proposition": str(note["passage_since_prior_note"] or ""),
            "gold": row,
        }
        record["deterministic"] = bool(
            deterministic_intents(record["id"], record["text"])
        )
        record["order"] = hashlib.sha256(
            f"{seed}|{row.get('id')}|{record['id']}".encode()
        ).hexdigest()
        candidates.append(record)
    safe = sorted(
        (record for record in candidates if record["deterministic"]),
        key=lambda record: record["order"],
    )
    model = sorted(
        (record for record in candidates if not record["deterministic"]),
        key=lambda record: record["order"],
    )
    selected = [*safe[:1], *model[: max(0, sample_size - min(1, len(safe)))]]
    selected = selected[:sample_size]
    expected = {record["id"]: record.pop("gold") for record in selected}
    for record in selected:
        record.pop("order", None)
        record.pop("deterministic", None)
    return selected, expected


def select_fixture_cases(
    fixture: Path, sample_size: int
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("fixture must contain a non-empty cases array")
    selected = rows[:sample_size]
    records: list[dict[str, Any]] = []
    expected: dict[str, dict[str, Any]] = {}
    for raw in selected:
        if not isinstance(raw, dict):
            raise ValueError("fixture cases must be objects")
        note_id = str(raw["id"])
        text = str(raw["text"])
        records.append(
            {
                "id": note_id,
                "label": str(raw.get("label") or note_id),
                "text": text,
                "proposition": str(raw.get("proposition") or ""),
            }
        )
        expected[note_id] = {
            "id": note_id,
            "footnote_text": text,
            "expected_verbatim_parts": raw["expected_verbatim_parts"],
            "acceptable_partitions": raw.get("acceptable_partitions") or [],
        }
    return records, expected


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--docx", type=Path)
    source.add_argument("--fixture", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", default="docx-linking-v1")
    parser.add_argument(
        "--arm",
        action="append",
        default=[],
        help="MODEL:STRATEGY; repeatable",
    )
    parser.add_argument("--effort", default="none")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    arms = args.arm or [
        "gpt-5.2:direct",
        "gpt-5.6-sol:direct",
        "gpt-5.6-terra:direct",
        "gpt-5.6-sol:hybrid",
        "gpt-5.6-terra:hybrid",
    ]
    sample_size = max(2, min(args.sample_size, 32))
    if args.fixture:
        records, expected = select_fixture_cases(
            args.fixture.resolve(), sample_size
        )
        source_label = str(args.fixture.resolve())
        gold_label = source_label
        claim = "synthetic_invariant"
        aggregate_label = "SYNTHETIC CONTRACT INVARIANT"
    else:
        if not args.gold:
            parser.error("--gold is required with --docx")
        records, expected = select_cases(
            args.docx.resolve(),
            args.gold.resolve(),
            sample_size,
            args.seed,
        )
        source_label = str(args.docx.resolve())
        gold_label = str(args.gold.resolve())
        claim = "descriptive"
        aggregate_label = "NOT SCOREABLE"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.unlink(missing_ok=True)
    print(
        f"selected={len(records)} deterministic="
        f"{sum(bool(deterministic_intents(r['id'], r['text'])) for r in records)} "
        f"source={source_label}",
        flush=True,
    )
    with args.output.open("a", encoding="utf-8", newline="\n") as stream:
        for index, arm in enumerate(arms, start=1):
            model, strategy = arm.rsplit(":", 1)
            print(
                f"{index}/{len(arms)} start model={model} strategy={strategy}",
                flush=True,
            )
            started = time.perf_counter()
            result: dict[str, Any] = {
                "arm": {
                    "model": model,
                    "effort": args.effort,
                    "strategy": strategy,
                },
                "source": source_label,
                "gold": gold_label,
                "claim": claim,
                "aggregate_label": aggregate_label,
                "sample_ids": [record["id"] for record in records],
            }
            try:
                plan = plan_footnotes(
                    records,
                    strategy=strategy,  # type: ignore[arg-type]
                    model=model,
                    effort=args.effort,
                    cache_dir=args.output.parent / "cache",
                    timeout_seconds=args.timeout_seconds,
                )
                result["plan_summary"] = {
                    "strategy_used": plan["strategy_used"],
                    "assessment": plan["assessment"],
                    "telemetry": plan["telemetry"],
                }
                result["score"] = score(expected, plan)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["wall_seconds"] = round(time.perf_counter() - started, 4)
            stream.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"{index}/{len(arms)} done model={model} strategy={strategy} "
                f"wall={result['wall_seconds']} error={result.get('error', '')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
