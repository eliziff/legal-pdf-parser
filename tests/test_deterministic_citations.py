"""Vectors for the deterministic citation splitter.

The module is a port of ALR-Quote-Verifier's deterministic_splitter
(read-only oracle); these vectors are taken from the oracle's own test
suite so the port's behavior stays pinned to it.
"""

import unittest

from legalpdf.deterministic_citations import (
    extract_fields,
    split_footnote_recall_first,
)


class CorporateSuffixGuard(unittest.TestCase):
    def test_keeps_corporate_case_name_before_uppercase_v(self):
        # Oracle vector: tests/test_deterministic_splitter.py,
        # test_free_keeps_corporate_case_name_before_uppercase_v.
        # "Ltd. V." is a corporate name meeting a versus, not a
        # sentence boundary; the split used to break the case name.
        text = (
            "1068490 Ontario Ltd. V. Marlin Center Mobile Homes Inc. and "
            "Howard Geisler, 2001 CarswellOnt 4564, at para. 21 "
            "(Book of Authorities TAB 17)"
        )

        result = split_footnote_recall_first(text)

        self.assertEqual([part.text for part in result.parts], [text])
        fields = extract_fields(result.parts[0])
        self.assertEqual(fields.kind, "case")
        self.assertTrue(fields.bare_citation.startswith("2001 CarswellOnt 4564"))

    def test_parallel_reporters_stay_inside_one_citation(self):
        text = "Groia v Law Society, 2018 SCC 27, [2018] 1 SCR 772 at paras 64–67."
        result = split_footnote_recall_first(text)
        self.assertEqual(len(result.parts), 1)


if __name__ == "__main__":
    unittest.main()
