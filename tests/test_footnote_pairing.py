from __future__ import annotations

import hashlib
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf import footnote_pairing_support  # noqa: E402
from legalpdf.footnote_pairing_support import heading_text_plausible  # noqa: E402


class BundledPairerInputs(unittest.TestCase):
    def test_frozen_mcgill_reporter_inventory(self) -> None:
        payload = (
            ROOT / "src" / "legalpdf" / "data" / "mcgill_reporters.json"
        ).read_bytes()
        self.assertEqual(30965, len(payload))
        self.assertEqual(
            "946e7554e8e9134d9b148d244d825e999080dd900c666cc4cf43235fa5ec9e2f",
            hashlib.sha256(payload).hexdigest(),
        )
        abbreviations = json.loads(payload)
        self.assertEqual(2110, len(abbreviations))
        for abbreviation in abbreviations:
            pattern = footnote_pairing_support._reporter_abbreviation_regex(
                abbreviation
            )
            self.assertIsNotNone(re.fullmatch(pattern, abbreviation), abbreviation)

    def test_heading_plausibility_matches_text_fidelity_vectors(self) -> None:
        expected = {
            "Legal Principles": True,
            "Background and Overview": True,
            "Member's Right To Fair Treatment": True,
            "D. Tax Cas. 1088.": False,
            "Introduction 1 Ont Liquor Licence App Trib Dec 2 Analysis": False,
        }
        self.assertEqual(
            expected,
            {text: heading_text_plausible(text) for text in expected},
        )

    def test_single_number_heading_skips_reporter_regex(self) -> None:
        with patch.object(
            footnote_pairing_support,
            "_mcgill_reporter_citation_re",
            side_effect=AssertionError("reporter regex should stay lazy"),
        ):
            self.assertTrue(heading_text_plausible("Background 2026 Overview"))


if __name__ == "__main__":
    unittest.main()
