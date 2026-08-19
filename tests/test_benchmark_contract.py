from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "dev" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


alr = load_script("benchmark_alr_splitter")
linking = load_script("benchmark_docx_linking")


class BenchmarkContractTests(unittest.TestCase):
    def test_split_direction_uses_all_accepted_partition_counts(self) -> None:
        row = {
            "expected_verbatim_parts": ["A", "B", "C"],
            "acceptable_partitions": [["A; B", "C"]],
        }
        self.assertEqual("under_split", alr.outcome(row, ["A; B; C"]))
        self.assertEqual("over_split", alr.outcome(row, ["A", "B", "C", "D"]))
        self.assertEqual("boundary_mismatch", alr.outcome(row, ["A", "B; C"]))

    def test_candidate_loader_admits_only_unambiguous_accepted_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "candidate.jsonl"
            rows = [
                {"id": "auto", "status": "auto", "footnote_text": "Auto"},
                {"id": "dup-1", "status": "accepted", "footnote_text": "Same"},
                {"id": "dup-2", "status": "accepted", "footnote_text": " Same "},
                {"id": "kept", "status": "accepted", "footnote_text": "Unique"},
            ]
            source.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            loaded = linking.load_gold(source)
        self.assertEqual(["unique"], list(loaded))
        self.assertEqual("kept", loaded["unique"]["id"])

    def test_score_requires_exact_prediction_id_bijection(self) -> None:
        expected = {
            "one": {
                "id": "one",
                "footnote_text": "A",
                "expected_verbatim_parts": ["A"],
            }
        }
        with self.assertRaisesRegex(ValueError, "exact bijection"):
            linking.score(expected, {"footnotes": []})

        result = linking.score(
            expected,
            {
                "footnotes": [
                    {
                        "id": "one",
                        "parts": [{"verbatim": "A"}],
                    }
                ]
            },
        )
        self.assertEqual(1.0, result["exact_rate"])
        self.assertNotIn("exact_accuracy", result)


if __name__ == "__main__":
    unittest.main()
