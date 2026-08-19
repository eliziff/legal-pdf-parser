"""The ALR no-regression envelope, executable.

benchmarks/grammar_vectors/harvested.jsonl carries vectors AST-extracted
from the read-only oracle's own test suite, expectations included. Every
splitter-io row runs end-to-end against this engine's (table-driven)
splitter; any failure is a real regression against what the oracle
already had down pat. Skips cleanly when the workspace checkout that
hosts the harvest is absent.
"""

import json
import unittest
from pathlib import Path

from legalpdf.deterministic_citations import (
    split_footnote,
    split_footnote_recall_first,
)

HARVEST = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "grammar_vectors"
    / "harvested.jsonl"
)

_SPLITTERS = {
    "split_footnote": split_footnote,
    "split_footnote_recall_first": split_footnote_recall_first,
}


def _splitter_rows():
    if not HARVEST.exists():
        return []
    rows = []
    with open(HARVEST, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "splitter-io" and isinstance(row.get("expect"), dict):
                rows.append(row)
    return rows


@unittest.skipUnless(HARVEST.exists(), "oracle harvest not present in this checkout")
class OracleSplitterVectors(unittest.TestCase):
    def test_every_harvested_splitter_vector(self):
        rows = _splitter_rows()
        self.assertGreater(len(rows), 20, "harvest looks truncated")
        checked = 0
        unchecked = []
        for row in rows:
            expect = row["expect"]
            names = expect.get("splitter") or ["split_footnote_recall_first"]
            for name in names:
                splitter = _SPLITTERS.get(name)
                if splitter is None:
                    continue
                with self.subTest(source=row["source"], splitter=name):
                    result = splitter(row["input"])
                    asserted = False
                    if "parts_text" in expect:
                        self.assertEqual(
                            [part.text for part in result.parts],
                            expect["parts_text"],
                        )
                        asserted = True
                    if "parts_len" in expect:
                        self.assertEqual(len(result.parts), expect["parts_len"])
                        asserted = True
                    if "status" in expect:
                        self.assertEqual(result.status, expect["status"])
                        asserted = True
                    if asserted:
                        checked += 1
                    else:
                        unchecked.append(row["source"])
        # 18 of the 31 harvested rows carry parts/status expectations this
        # harness models; the rest use finer field probes (pinpoint lists,
        # short-form fields) that can join the net later.
        self.assertGreaterEqual(checked, 15, f"too few assertable vectors: {checked}")
        # Rows whose folded expectations use field probes this harness
        # does not model are fine — but they must stay a minority.
        self.assertLess(len(unchecked), checked, f"unchecked outgrew checked: {unchecked[:5]}")


if __name__ == "__main__":
    unittest.main()
