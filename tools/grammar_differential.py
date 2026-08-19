#!/usr/bin/env python
"""Differential parity gate: grammar-table entries vs runtime bindings.

For every grammar-table entry that maps to a runtime binding, compile BOTH
sides (the table entry via legalpdf.grammar_tables.compile_entry with
the table's defs; the consumer pattern from its module) and run finditer
over real-world inputs, comparing full-match span lists (start, end).

Unicode caveat, by design: table entries compile with re.ASCII while
the source modules compile with Unicode semantics, so \\w \\b \\d \\s can
genuinely diverge on accented text (French case names). Any mismatch
that disappears when the SOURCE is recompiled with re.ASCII is counted
as "unicode-semantics" — an expected, reportable finding, not a
failure. Everything else is pattern-drift and fails the gate.

Exit codes: 0 parity (unicode findings allowed), 1 pattern drift or a
table entry that will not compile, 2 setup problems (no tables, no
inputs, nothing compared).
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = REPO_ROOT.parent  # the "MikeOSS Fork" checkout that hosts shared/

try:
    import legalpdf.grammar_tables as gt
except ModuleNotFoundError:  # running outside the venv: use the src layout
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import legalpdf.grammar_tables as gt
import legalpdf.core as _core
import legalpdf.deterministic_citations as _det
import legalpdf.docx_linking as _docx

# entry id -> (consumer kind, consumer attribute). "det"/"core"/"docx" are
# imported engine modules; "toa" bindings are read from toa_maker.py without
# importing its UI.
MAPPING: dict[str, tuple[str, str]] = {
    "cite.neutral": ("det", "_NEUTRAL_RE"),
    "cite.reporter.splitter": ("det", "_REPORTER_RE"),
    "cite.statute.splitter": ("det", "_STATUTE_RE"),
    "cite.journal.splitter": ("det", "_JOURNAL_RE"),
    "title.legal.splitter": ("det", "_LEGAL_TITLE_RE"),
    "title.named-code": ("det", "_NAMED_CODE_RE"),
    "cite.url": ("det", "_URL_RE"),
    "frame.book": ("det", "_BOOK_FRAME_RE"),
    "cite.quoted": ("det", "_QUOTED_CITATION_RE"),
    "cite.secondary": ("det", "_SECONDARY_CITATION_RE"),
    "pinpoint.para.splitter": ("det", "_PAR_PIN_RE"),
    "pinpoint.section.splitter": ("det", "_SEC_PIN_RE"),
    "pinpoint.page.splitter": ("det", "_PAGE_PIN_RE"),
    "ref.token": ("det", "_REF_TOKEN_RE"),
    "ref.pure.splitter": ("det", "_PURE_REF_RE"),
    "signal.prefix.splitter": ("det", "_SIGNAL_PREFIX_RE"),
    "signal.source": ("det", "_SOURCE_SIGNAL_RE"),
    "signal.aggressive": ("det", "_AGGRESSIVE_SIGNAL_RE"),
    "bracket.editorial": ("det", "_EDITORIAL_BRACKET_RE"),
    "attach.link": ("det", "_LINK_ATTACHMENT_RE"),
    "shortform.splitter": ("det", "_TRAILING_SHORT_FORM_RE"),
    "boundary.sentence.splitter": ("det", "_SENTENCE_BOUNDARY_RE"),
    "boundary.conjunction": ("det", "_CONJUNCTION_RE"),
    "ref.cross-reference": ("det", "_CROSS_REFERENCE_RE"),
    "ref.note-reference": ("det", "_NOTE_REFERENCE_START_RE"),
    "ref.quoted-work-author": ("det", "_QUOTED_WORK_AUTHOR_RE"),
    "cite.canlii": ("toa", "_CANLII_RE"),
    "cite.reporter.toa": ("toa", "_REPORTER_RE"),
    "cite.statute.toa": ("toa", "_STATUTE_RE"),
    "cite.journal.toa": ("toa", "_JOURNAL_RE"),
    "title.legal.toa": ("toa", "_LEGAL_TITLE_RE"),
    "ref.pure.toa": ("toa", "_REFERENCE_RE"),
    "ref.inline.toa": ("toa", "_INLINE_REFERENCE_RE"),
    "ref.history.toa": ("toa", "_HISTORY_RE"),
    "signal.prefix.toa": ("toa", "_SIGNAL_PREFIX_RE"),
    "signal.citation.toa": ("toa", "_CITATION_SIGNAL_RE"),
    "pinpoint.para.toa": ("toa", "_PAR_RE"),
    "pinpoint.section.toa": ("toa", "_SECTION_RE"),
    "pinpoint.page.toa": ("toa", "_PAGE_RE"),
    "shortform.toa": ("toa", "_SHORT_FORM_SUFFIX_RE"),
    "label.line-start": ("core", "_LABEL_RE"),
    "label.pure": ("core", "_PURE_LABEL_RE"),
    "label.superscript": ("core", "_SUPER_RE"),
    "boundary.sentence.engine": ("core", "_SENTENCE_EDGE_RE"),
    "marker.inline-fn": ("core", "_INLINE_FN_RE"),
    "label.standalone": ("core", "_STANDALONE_REF_RE"),
    "trap.double-zero-width": ("core", "_DOUBLE_ZERO_WIDTH_RE"),
    "cite.url.prefix": ("docx", "URL_RE"),
    "ref.supra-note.linking": ("docx", "SUPRA_NOTE_RE"),
}


def load_toa_patterns(path: Path) -> tuple[dict[str, re.Pattern[str]], str | None]:
    """Load only the corpus entries bound by toa_maker, without importing its UI."""
    if not path.is_file():
        return {}, f"toa source not found: {path}"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bindings: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "_table"
            and len(call.args) == 1
            and isinstance(call.args[0], ast.Constant)
            and isinstance(call.args[0].value, str)
        ):
            bindings[node.targets[0].id] = call.args[0].value
    loader_path = path.with_name("grammar_tables.py")
    spec = importlib.util.spec_from_file_location("toa_grammar_tables", loader_path)
    if spec is None or spec.loader is None:
        return {}, f"toa grammar loader not found: {loader_path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {name: module.table_entry(entry_id) for name, entry_id in bindings.items()}, None


def resolve_source(entry_id: str, toa: dict[str, re.Pattern[str]], toa_note: str | None):
    kind, attr = MAPPING[entry_id]
    if kind == "toa":
        if toa_note:
            return None, toa_note
        pattern = toa.get(attr)
        return (pattern, None) if pattern else (None, f"{attr} not extracted from toa_maker")
    module = {"det": _det, "core": _core, "docx": _docx}[kind]
    pattern = getattr(module, attr, None)
    if isinstance(pattern, re.Pattern):
        return pattern, None
    return None, f"{module.__name__}.{attr} missing"


def read_jsonl_texts(path: Path, field: str) -> list[str]:
    texts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            value = json.loads(line).get(field)
            if isinstance(value, str) and value.strip():
                texts.append(value)
    return texts


def load_a2aj_paragraphs(path: Path, sample: int, column: str) -> list[str]:
    texts: list[str] = []
    try:
        import duckdb

        rows = duckdb.connect().execute(
            f"SELECT {column} FROM read_parquet(?) "
            f"USING SAMPLE {int(sample)} ROWS (reservoir, 42)",
            [str(path)],
        ).fetchall()
        texts = [row[0] for row in rows if isinstance(row[0], str)]
    except ModuleNotFoundError:
        try:
            import pandas as pd

            frame = pd.read_parquet(path, columns=[column])
            frame = frame.sample(min(int(sample), len(frame)), random_state=42)
            texts = [t for t in frame[column].tolist() if isinstance(t, str)]
        except ModuleNotFoundError:
            print(
                "a2aj skipped: neither duckdb nor pandas is installed "
                "(pip install duckdb  # or: pip install pandas pyarrow)"
            )
            return []
    paragraphs: list[str] = []
    for text in texts:
        for block in re.split(r"(?:\r?\n){2,}", text):
            block = block.strip()
            if len(block) >= 20:
                paragraphs.append(block)
            if len(paragraphs) >= sample:
                return paragraphs
    return paragraphs


def spans_of(pattern: re.Pattern[str], text: str) -> tuple[tuple[int, int], ...]:
    return tuple(match.span() for match in pattern.finditer(text))


def ascii_recompile(pattern: re.Pattern[str]) -> re.Pattern[str]:
    return re.compile(pattern.pattern, (pattern.flags | re.ASCII) & ~re.UNICODE)


def clip(text: str, width: int = 110) -> str:
    text = text.replace("\n", "\\n")
    return text if len(text) <= width else text[: width - 1] + "…"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tables",
        type=Path,
        default=WORKSPACE / "packages" / "legal-grammar-tables" / "grammar-corpus.json",
    )
    parser.add_argument(
        "--jsonl",
        type=Path,
        default=WORKSPACE / "benchmarks" / "docx_corpus" / "private_results" / "local"
        / "cases.private.jsonl",
        help="corpus jsonl; the footnote_text field of every row (deduped)",
    )
    parser.add_argument(
        "--harvest",
        type=Path,
        default=WORKSPACE / "benchmarks" / "grammar_vectors" / "harvested.jsonl",
        help="harvested vectors jsonl; the input field of every row",
    )
    parser.add_argument("--a2aj", type=Path, default=None, help="optional parquet of case text")
    parser.add_argument("--a2aj-sample", type=int, default=500)
    parser.add_argument("--a2aj-column", default="text")
    parser.add_argument(
        "--toa",
        type=Path,
        default=WORKSPACE / "TableOfAuthoritiesMaker" / "toa_maker.py",
        help="toa_maker.py to ast-extract the toa dialect sources from",
    )
    args = parser.parse_args()

    inputs: dict[str, None] = {}  # ordered dedupe
    if args.jsonl.is_file():
        for text in read_jsonl_texts(args.jsonl, "footnote_text"):
            inputs.setdefault(text)
        print(f"jsonl: {args.jsonl} -> {len(inputs)} unique footnote_text inputs")
    else:
        print(f"warning: --jsonl not found, skipping: {args.jsonl}")
    if args.harvest.is_file():
        before = len(inputs)
        for text in read_jsonl_texts(args.harvest, "input"):
            inputs.setdefault(text)
        print(f"harvest: {args.harvest} -> +{len(inputs) - before} inputs")
    else:
        print(f"warning: --harvest not found, skipping: {args.harvest}")
    if args.a2aj:
        before = len(inputs)
        for text in load_a2aj_paragraphs(args.a2aj, args.a2aj_sample, args.a2aj_column):
            inputs.setdefault(text)
        print(f"a2aj: {args.a2aj} -> +{len(inputs) - before} paragraphs")
    texts = list(inputs)
    if not texts:
        print("no inputs available; nothing to compare")
        return 2

    try:
        grammar_tables = gt.read_corpus(args.tables)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot load grammar corpus: {error}")
        return 2
    toa, toa_note = load_toa_patterns(args.toa)

    rows: list[tuple[str, int, int, int, int]] = []
    unmapped: list[str] = []
    skipped: list[str] = []
    compile_errors: list[str] = []
    unicode_examples: dict[str, list[str]] = {}
    drift_examples: dict[str, list[str]] = {}
    seen_ids: set[str] = set()

    for table_name, table in grammar_tables.items():
        defs = table.get("defs") or {}
        for entry in table.get("entries", []):
            entry_id = entry.get("id", "?")
            seen_ids.add(entry_id)
            if entry_id not in MAPPING:
                unmapped.append(f"{entry_id} ({table_name})")
                continue
            source, note = resolve_source(entry_id, toa, toa_note)
            if source is None:
                skipped.append(f"{entry_id}: {note}")
                continue
            try:
                table_re = gt.compile_entry(entry, defs)
            except (re.error, KeyError, ValueError) as error:
                compile_errors.append(f"{entry_id}: table entry does not compile: {error}")
                continue
            ascii_source: re.Pattern[str] | None = None
            n_spans = drift = unicode_findings = 0
            for text in texts:
                table_spans = spans_of(table_re, text)
                source_spans = spans_of(source, text)
                n_spans += max(len(table_spans), len(source_spans))
                if table_spans == source_spans:
                    continue
                if ascii_source is None:
                    ascii_source = ascii_recompile(source)
                if spans_of(ascii_source, text) == table_spans:
                    unicode_findings += 1
                    bucket = unicode_examples.setdefault(entry_id, [])
                    if len(bucket) < 10:
                        bucket.append(clip(text))
                else:
                    drift += 1
                    bucket = drift_examples.setdefault(entry_id, [])
                    if len(bucket) < 10:
                        bucket.append(
                            f"input: {clip(text)}\n"
                            f"      table  spans: {list(table_spans)[:8]}\n"
                            f"      source spans: {list(source_spans)[:8]}"
                        )
            rows.append((entry_id, len(texts), n_spans, drift, unicode_findings))

    absent = sorted(entry_id for entry_id in MAPPING if entry_id not in seen_ids)

    print()
    header = ("entry", "inputs", "spans", "drift", "unicode-findings")
    widths = [max(len(header[0]), *(len(r[0]) for r in rows)) if rows else len(header[0])]
    widths += [max(len(h), 6) for h in header[1:]]
    line = " | ".join(h.ljust(w) for h, w in zip(header, widths))
    print(line)
    print("-" * len(line))
    for row in sorted(rows):
        cells = [row[0].ljust(widths[0])]
        cells += [str(v).rjust(w) for v, w in zip(row[1:], widths[1:])]
        print(" | ".join(cells))

    for entry_id, examples in sorted(unicode_examples.items()):
        print(f"\nunicode-semantics findings for {entry_id} (up to 10 examples):")
        for example in examples:
            print(f"  - {example}")
    for entry_id, examples in sorted(drift_examples.items()):
        print(f"\nPATTERN DRIFT for {entry_id}:")
        for example in examples:
            print(f"  - {example}")
    for message in compile_errors:
        print(f"\nCOMPILE ERROR: {message}")
    if unmapped:
        print(f"\nunmapped table entries (no source regex; not an error): {', '.join(unmapped)}")
    if skipped:
        print("\nskipped (source unavailable):")
        for message in skipped:
            print(f"  - {message}")
    if absent:
        print(f"\nmapped sources with no table entry yet (table absent): {', '.join(absent)}")

    total_drift = sum(row[3] for row in rows)
    total_unicode = sum(row[4] for row in rows)
    print(
        f"\n{len(rows)} entries compared over {len(texts)} inputs: "
        f"{total_drift} pattern-drift, {total_unicode} unicode-semantics findings"
    )
    if not rows:
        return 2
    return 1 if total_drift or compile_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
