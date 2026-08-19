"""anchored_scan: differential equivalence + contract-boundary vectors.

The load-bearing test is the differential: over synthesized large
documents built from every grammar table's own vectors, the windowed
scanner must return byte-identical spans to the bare pattern for every
entry. The unit vectors pin the escape hatches (clip-guard, coverage
bailout, lower-length mismatch, small-text cutoff) and the documented
limitation of heuristic pads.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from legalpdf import anchored_scan
from legalpdf.anchored_scan import (
    AnchoredPattern,
    anchored_matches,
    derive_anchor,
    derive_gate,
)
from legalpdf.grammar_tables import compile_entry, load_tables

FILLER = (
    " The tribunal weighed the record before it and reserved judgment on "
    "the remaining issues, having heard the parties at length on costs. "
)


def _all_entries() -> list[tuple[str, re.Pattern[str], dict]]:
    return [
        (eid, compile_entry(entry, defs), entry)
        for eid, (entry, defs) in sorted(load_tables().items())
    ]


def _synth_doc() -> str:
    pieces = []
    for _eid, _rx, entry in _all_entries():
        for vector in entry.get("vectors", []):
            pieces.append(vector["input"])
            pieces.append(FILLER)
    doc = "".join(pieces * 3)
    assert len(doc) > anchored_scan.MIN_WINDOW_TEXT
    return doc


def test_differential_all_entries_on_synth_doc():
    doc = _synth_doc()
    anchored_count = 0
    for eid, rx, _entry in _all_entries():
        ap = AnchoredPattern(rx)
        if ap.anchors is not None:
            anchored_count += 1
        expected = [m.span() for m in rx.finditer(doc)]
        got = ap.findall_spans(doc)
        assert got == expected, (
            f"{eid}: windowed spans diverge from full scan "
            f"(anchors={ap.anchors}, pad={ap.pad})"
        )
    # The differential proves nothing if nobody takes the windowed path.
    assert anchored_count >= 10, f"only {anchored_count} entries anchored"


def test_differential_on_each_vector_small_text():
    for eid, rx, entry in _all_entries():
        ap = AnchoredPattern(rx)
        for vector in entry.get("vectors", []):
            text = vector["input"]
            assert ap.findall_spans(text) == [
                m.span() for m in rx.finditer(text)
            ], f"{eid} on vector {text!r}"


def test_gate_passes_every_matching_vector():
    # Gates may only over-pass: any vector that matches must pass the gate.
    for eid, rx, entry in _all_entries():
        gate = derive_gate(rx)
        if gate is None:
            continue
        for vector in entry.get("vectors", []):
            if not rx.search(vector["input"]):
                continue
            lower = vector["input"].lower()
            assert any(lit in lower for lit in gate), (
                f"{eid}: gate {gate} rejects matching vector "
                f"{vector['input']!r}"
            )


def _big(text_core: str) -> str:
    # Embed the interesting core deep inside > MIN_WINDOW_TEXT of filler.
    return FILLER * 40 + text_core + FILLER * 40


def test_clip_guard_recovers_edge_touching_match():
    # A long match runs past the window edge as a clipped-but-present
    # match; the guard must fall back and the result stays exact.
    rx = re.compile(r"marker\d+")
    doc = _big("marker" + "7" * 500 + " end")
    anchor = derive_anchor(rx)
    assert anchor is not None
    anchors, pad = anchor
    assert pad < 500  # the match genuinely exceeds the pad
    expected = [m.span() for m in rx.finditer(doc)]
    assert anchored_matches(rx, doc, anchors, pad) and [
        m.span() for m in anchored_matches(rx, doc, anchors, pad)
    ] == expected


def test_heuristic_pad_limitation_is_the_documented_one():
    # CONTRACT BOUNDARY, not a soundness proof: when an unbounded repeat
    # exceeds the assumed span AND the clipped regex fails entirely inside
    # the window, the miss is invisible to the clip-guard. This is why
    # consumers must keep a corpus differential. If this test ever fails
    # because the miss disappeared, the pads got safer — update the
    # module docstring before deleting it.
    rx = re.compile(r"marker\s+\d+")
    run = " " * (8 * anchored_scan._ASSUMED_REPEAT_SPAN)
    doc = _big("marker" + run + "42")
    anchor = derive_anchor(rx)
    assert anchor is not None
    anchors, pad = anchor
    full = [m.span() for m in rx.finditer(doc)]
    windowed = [m.span() for m in anchored_matches(rx, doc, anchors, pad)]
    assert len(full) == 1
    assert windowed == [] or windowed == full


def test_lower_length_mismatch_falls_back():
    rx = re.compile(r"marker\d+")
    doc = _big("İstanbul marker42")  # 'İ'.lower() is two chars
    assert len(doc.lower()) != len(doc)
    anchor = derive_anchor(rx)
    assert anchor is not None
    anchors, pad = anchor
    assert [m.span() for m in anchored_matches(rx, doc, anchors, pad)] == [
        m.span() for m in rx.finditer(doc)
    ]


def test_coverage_bailout_stays_exact():
    rx = re.compile(r"the\s+tribunal")
    doc = FILLER * 200  # anchor 'the' saturates the text
    anchor = derive_anchor(rx)
    assert anchor is not None
    anchors, pad = anchor
    assert [m.span() for m in anchored_matches(rx, doc, anchors, pad)] == [
        m.span() for m in rx.finditer(doc)
    ]


def test_benchmark_citation_patterns_windowed_equals_full():
    # benchmark._citations scans body+footnotes with these; the neutral
    # pattern's court codes are hand anchors (the AST walk refuses its
    # 2-char branch members), so their per-match-mandatory contract needs
    # its own differential here.
    from legalpdf import benchmark

    core = (
        "In 2019 SCC 65 and 2003 FC 296 the Court held; see also "
        "125 DLR 456, R. v. X, 1998 NUCA 4, and 2020 ONSC 1234. "
    )
    doc = _big(core * 5)
    for ap in benchmark._CITATION_RES:
        assert isinstance(ap, AnchoredPattern)
        expected = [m.span() for m in ap.rx.finditer(doc)]
        assert expected  # the differential proves nothing on zero matches
        assert ap.findall_spans(doc) == expected
    neutral = benchmark._CITATION_RES[0]
    assert neutral.anchors is not None
    for match in neutral.rx.finditer(doc):
        lowered = match.group().lower()
        assert any(lit in lowered for lit in neutral.anchors)


def test_deterministic_citations_anchored_handles_are_memoized():
    from legalpdf import deterministic_citations as dc

    first = dc._anchored(dc._NEUTRAL_RE)
    assert dc._anchored(dc._NEUTRAL_RE) is first
    assert first.rx is dc._NEUTRAL_RE


def test_no_anchor_hits_means_no_matches():
    rx = re.compile(r"marker\d+")
    doc = FILLER * 60
    anchor = derive_anchor(rx)
    assert anchor is not None
    anchors, pad = anchor
    assert anchored_matches(rx, doc, anchors, pad) == []
    assert list(rx.finditer(doc)) == []
