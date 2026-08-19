#!/usr/bin/env python
"""Recall survey: do the grammar tables see the citations providers attest?

For every A2AJ court parquet (the provider bulk corpus the ALR verifier
already mirrors locally), sample decisions and treat `cases_cited_<lang>`
as gold: each entry is a neutral-or-canonical citation string the
provider says this decision cites. Locate each gold citation in the
decision text (whitespace/NBSP-tolerant), then check whether any span
from the table-compiled citation grammars (CITE_ENTRY_IDS below)
covers that occurrence.

This is a survey instrument, not a gate: it reports per-court/lang
recall so jurisdiction gaps are numbers, not vibes. Exit 0 unless the
setup is unusable. The differential harness stays the parity gate.

Requires duckdb (the corpus is parquet); run with a python that has it,
e.g. `python -X utf8 tools/grammar_recall.py`.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grammar_differential as gd  # reuses loader bootstrap + WORKSPACE

CITE_ENTRY_IDS = (
    "cite.neutral",
    "cite.neutral.tribunal",
    "cite.canlii",
    "cite.reporter.splitter",
    "cite.reporter.toa",
)
DEFAULT_CORPUS = (
    Path.home() / "AppData" / "Local" / "ALR Quote Verifier" / "a2aj_corpus" / "cases"
)


def load_cite_patterns(corpus_path: Path) -> dict[str, re.Pattern[str]]:
    patterns: dict[str, re.Pattern[str]] = {}
    for table in gd.gt.read_corpus(corpus_path).values():
        defs = table.get("defs") or {}
        for entry in table.get("entries", []):
            if entry.get("id") in CITE_ENTRY_IDS:
                patterns[entry["id"]] = gd.gt.compile_entry(entry, defs)
    return patterns


def occurrence_re(citation: str) -> re.Pattern[str] | None:
    tokens = citation.split()
    if not tokens:
        return None
    # Unicode \s (no re.ASCII here) already covers NBSP variants.
    return re.compile(r"\s+".join(re.escape(tok) for tok in tokens))


def sample_rows(parquet: Path, lang: str, n: int) -> list[tuple[str, list[str]]]:
    import duckdb

    text_col, cited_col = f"unofficial_text_{lang}", f"cases_cited_{lang}"
    query = (
        f"SELECT {text_col}, {cited_col} FROM ("
        f"  SELECT {text_col}, {cited_col} FROM read_parquet(?)"
        f"  WHERE {text_col} IS NOT NULL AND len({cited_col}) > 0"
        f") USING SAMPLE {int(n)} ROWS (reservoir, 42)"
    )
    try:
        rows = duckdb.connect().execute(query, [str(parquet)]).fetchall()
    except duckdb.Error as error:
        print(f"  {parquet.parent.name}/{lang}: query failed, skipping ({error})")
        return []
    return [(t, list(c)) for t, c in rows if isinstance(t, str)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--tables",
        type=Path,
        default=gd.WORKSPACE / "packages" / "legal-grammar-tables" / "grammar-corpus.json",
    )
    parser.add_argument("--sample", type=int, default=50, help="decisions per court per language")
    parser.add_argument("--langs", default="en,fr")
    parser.add_argument("--courts", default="", help="comma list; default all")
    parser.add_argument("--misses", type=int, default=12, help="missed examples to print")
    args = parser.parse_args()

    patterns = load_cite_patterns(args.tables)
    if len(patterns) < len(CITE_ENTRY_IDS):
        missing = set(CITE_ENTRY_IDS) - set(patterns)
        print(f"missing table entries {sorted(missing)} under {args.tables}")
        return 2
    court_dirs = sorted(
        d for d in args.corpus.iterdir() if d.is_dir() and (d / "train.parquet").is_file()
    ) if args.corpus.is_dir() else []
    if args.courts:
        wanted = {c.strip().upper() for c in args.courts.split(",") if c.strip()}
        court_dirs = [d for d in court_dirs if d.name.upper() in wanted]
    if not court_dirs:
        print(f"no court parquets under {args.corpus}")
        return 2

    header = ("court", "lang", "docs", "gold", "absent", "found", "recall")
    rows: list[tuple] = []
    missed: list[str] = []
    totals = {"docs": 0, "gold": 0, "absent": 0, "found": 0}
    for court in court_dirs:
        for lang in [l.strip() for l in args.langs.split(",") if l.strip()]:
            sampled = sample_rows(court / "train.parquet", lang, args.sample)
            if not sampled:
                continue
            gold = absent = found = 0
            for text, cited in sampled:
                spans: list[tuple[int, int]] | None = None  # computed lazily per doc
                for citation in dict.fromkeys(cited):
                    finder = occurrence_re(citation)
                    if finder is None:
                        continue
                    occurrences = [m.span() for m in finder.finditer(text)]
                    gold += 1
                    if not occurrences:
                        absent += 1
                        continue
                    if spans is None:
                        spans = sorted(
                            s for p in patterns.values() for s in gd.spans_of(p, text)
                        )
                    if any(
                        s <= a and e >= b for a, b in occurrences for s, e in spans
                    ):
                        found += 1
                    elif len(missed) < args.misses:
                        a, b = occurrences[0]
                        ctx = text[max(0, a - 40):b + 25].replace("\n", " ")
                        missed.append(f"{court.name}/{lang} |{citation}| in ...{ctx}...")
            rows.append((court.name, lang, len(sampled), gold, absent, found,
                         f"{found / (gold - absent):6.1%}" if gold > absent else "   n/a"))
            totals["docs"] += len(sampled)
            totals["gold"] += gold
            totals["absent"] += absent
            totals["found"] += found

    widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0)) for i, h in enumerate(header)]
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-" * (sum(widths) + 3 * (len(header) - 1)))
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))
    denom = totals["gold"] - totals["absent"]
    overall = totals["found"] / denom if denom else 0.0
    print(
        f"\nTOTAL: {totals['docs']} docs, {totals['gold']} gold citations, "
        f"{totals['absent']} not literally present in text (excluded), "
        f"{totals['found']}/{denom} detected = {overall:.1%} recall"
    )
    if missed:
        print(f"\nmissed examples (first {len(missed)}):")
        for line in missed:
            print(f"  - {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
