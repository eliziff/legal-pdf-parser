import tempfile
import unittest
from pathlib import Path

from freeze_benchmark_splits import OUTPUT, validate_benchmark_paths


class BenchmarkTruthTest(unittest.TestCase):
    def test_frozen_corpus_contains_only_accepted_truth(self):
        paths = [Path(line) for line in (OUTPUT / "benchmark-153.lst").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(validate_benchmark_paths(paths), {"true_gold": 123, "manually_vetted_silver": 30})

    def test_unvetted_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, "no accepted provenance"):
                validate_benchmark_paths([Path(folder) / "page.xml"])


if __name__ == "__main__":
    unittest.main()
