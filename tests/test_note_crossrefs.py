"""Vendored note cross-references: parity vs the Text-Fidelity checkout,
pattern vectors, and deterministic resolution over paired footnotes."""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from legalpdf.model import Footnote  # noqa: E402
from legalpdf.note_crossrefs import (  # noqa: E402
    CROSSREF_PATTERN,
    crossref_kind,
    resolve_note_crossrefs,
    shortform_before,
)

SENTINEL = "# --- byte-equal payload below; do not edit (see header) ---\n"
REGION_START = "CROSSREF_TAXONOMY = "
REGION_END = "\n\ndef _norm"
TFP_ROOT = Path(
    os.environ.get(
        "TEXT_FIDELITY_ROOT",
        ROOT.parent / "Text-Fidelity-Project",
    )
)


def note(pair_id: str, label: str, body: str, *, restart: int = 1) -> Footnote:
    return Footnote(
        pair_id=pair_id,
        label=label,
        occurrence=1,
        restart_sequence=restart,
        reference_page=1,
        body_pages=[1],
        reference_line_id=None,
        body_line_ids=[],
        body=body,
        sentence_proposition="",
        passage_since_prior_note="",
        confidence=0.9,
        provenance="test",
    )


class VendoredParity(unittest.TestCase):
    def test_payload_matches_checkout_region(self) -> None:
        source_path = TFP_ROOT / "tools" / "footnotes" / "note_crossrefs.py"
        if not source_path.is_file():
            self.skipTest("Text-Fidelity checkout not present on this machine")
        source = source_path.read_text(encoding="utf-8")
        region = source[source.index(REGION_START) : source.index(REGION_END)]
        if not region.endswith("\n"):
            region += "\n"
        vendored = (ROOT / "src" / "legalpdf" / "note_crossrefs.py").read_text(
            encoding="utf-8"
        )
        payload = vendored[vendored.index(SENTINEL) + len(SENTINEL) :]
        self.assertEqual(region, payload, "vendored crossref region drifted")


class PatternVectors(unittest.TestCase):
    def test_supported_forms_match(self) -> None:
        cases = {
            "See Smith, supra note 3, at 12.": ("supra", "3"),
            "Jones, infra note 41.": ("infra", "41"),
            "op. cit., note 44": ("op_cit", "44"),
            "see also footnote 7": ("see_footnote", "7"),
            "supra, footnote 12": ("supra", "12"),
        }
        for text, expected in cases.items():
            match = CROSSREF_PATTERN.search(text)
            self.assertIsNotNone(match, text)
            self.assertEqual(expected, (crossref_kind(match), match.group("num")), text)

    def test_journal_abbreviations_and_n_dot_do_not_match(self) -> None:
        for text in ("74 R. du N. 383", "supra n. 4", "décret n. 89-222"):
            self.assertIsNone(CROSSREF_PATTERN.search(text), text)

    def test_shortform_capture(self) -> None:
        text = "See Carosella, supra note 1."
        match = CROSSREF_PATTERN.search(text)
        self.assertIn("Carosella", shortform_before(text, match.start()))


class Resolution(unittest.TestCase):
    def test_resolves_to_unique_target(self) -> None:
        notes = [
            note("fn-1", "1", "R v Smith, 2001 SCC 1."),
            note("fn-2", "2", "Smith, supra note 1, at para 12."),
        ]
        records = resolve_note_crossrefs(notes)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertEqual("fn-2", record["source_pair_id"])
        self.assertEqual("fn-1", record["target_pair_id"])
        self.assertTrue(record["resolved"])
        self.assertEqual("supra", record["kind"])

    def test_unresolved_number_is_a_pairing_witness(self) -> None:
        notes = [note("fn-1", "1", "See op. cit., note 9.")]
        records = resolve_note_crossrefs(notes)
        self.assertEqual(1, len(records))
        self.assertFalse(records[0]["resolved"])
        self.assertEqual("", records[0]["target_pair_id"])

    def test_restarted_numbering_scopes_the_target(self) -> None:
        notes = [
            note("fn-1", "3", "First chapter note.", restart=1),
            note("fn-2", "3", "Second chapter note.", restart=2),
            note("fn-3", "4", "Smith, supra note 3.", restart=2),
        ]
        records = resolve_note_crossrefs(notes)
        self.assertEqual(1, len(records))
        record = records[0]
        self.assertTrue(record["resolved"])
        self.assertEqual("fn-2", record["target_pair_id"])
        self.assertEqual(2, record["target_count"])

    def test_ambiguous_target_stays_unaddressed(self) -> None:
        notes = [
            note("fn-1", "3", "First chapter note.", restart=1),
            note("fn-2", "3", "Second chapter note.", restart=1),
            note("fn-3", "4", "Smith, supra note 3.", restart=1),
        ]
        records = resolve_note_crossrefs(notes)
        record = records[0]
        self.assertTrue(record["resolved"])
        self.assertEqual("", record["target_pair_id"])
        self.assertEqual(2, record["target_count"])


if __name__ == "__main__":
    unittest.main()
