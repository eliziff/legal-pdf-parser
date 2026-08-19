from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from legalpdf.docx_linking import (
    PKG_REL_NS,
    R_NS,
    W_NS,
    _validate_response,
    apply_docx_links,
    assess_route,
    deterministic_intents,
    plan_docx_links,
)


def _fixture(path: Path, footnote: str) -> None:
    document = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="{W_NS}"><w:body><w:p><w:r><w:t>Proposition</w:t></w:r>
<w:r><w:footnoteReference w:id="2"/></w:r></w:p></w:body></w:document>"""
    notes = f"""<?xml version="1.0" encoding="UTF-8"?>
<w:footnotes xmlns:w="{W_NS}" xmlns:r="{R_NS}">
<w:footnote w:id="-1" w:type="separator"><w:p/></w:footnote>
<w:footnote w:id="2"><w:p><w:r><w:footnoteRef/></w:r>
<w:r><w:rPr><w:i/></w:rPr><w:t>{footnote}</w:t></w:r></w:p></w:footnote>
</w:footnotes>"""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", document)
        archive.writestr("word/footnotes.xml", notes)
        archive.writestr(
            "[Content_Types].xml",
            """<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/footnotes.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footnotes+xml"/>
</Types>""",
        )


class DocxLinkingTests(unittest.TestCase):
    def test_ultra_economy_gate_ports_complete_split(self) -> None:
        text = (
            "Criminal Code, RSC 1985, c C-46, s 7; "
            "R v Example, 2024 SCC 1"
        )
        intents = deterministic_intents("2", text)
        self.assertEqual([part["kind"] for part in intents or []], ["statute", "case"])
        self.assertEqual((intents or [])[0]["locator"], "7")
        self.assertNotIn("link", (intents or [])[0])

    def test_auto_route_skips_scan_when_it_saves_no_model_tokens(self) -> None:
        notes = [{"id": "2", "text": "A difficult prose footnote.", "proposition": ""}]
        assessment = assess_route(notes)
        self.assertEqual(assessment["recommended_strategy"], "direct")
        self.assertEqual(assessment["estimated_token_savings"], 0)

    def test_worker_contract_rejects_urls(self) -> None:
        records = [{"id": "2", "text": "R v Example, 2024 SCC 1", "proposition": ""}]
        response = {
            "results": [
                {
                    "id": "2",
                    "parts": [
                        {
                            "verbatim": records[0]["text"],
                            "corrected": records[0]["text"],
                            "kind": "case",
                            "pinpoint_fragments": [],
                            "page_pinpoints": [],
                            "short_form": "Example",
                            "bare_citation": "2024 SCC 1",
                            "citation_with_style": records[0]["text"],
                            "support_quote": "https://example.test",
                        }
                    ],
                }
            ]
        }
        with self.assertRaisesRegex(ValueError, "URL"):
            _validate_response(response, records)

    def test_worker_terminal_punctuation_is_snapped_to_source(self) -> None:
        records = [
            {
                "id": "2",
                "text": "R v Example, 2024 SCC 1.",
                "proposition": "",
            }
        ]
        citation = "R v Example, 2024 SCC 1"
        response = {
            "results": [
                {
                    "id": "2",
                    "parts": [
                        {
                            "verbatim": citation,
                            "corrected": citation,
                            "kind": "case",
                            "pinpoint_fragments": [],
                            "page_pinpoints": [],
                            "short_form": "Example",
                            "bare_citation": "2024 SCC 1",
                            "citation_with_style": citation,
                            "support_quote": "",
                        }
                    ],
                }
            ]
        }
        result = _validate_response(response, records)
        self.assertEqual(result["2"][0]["verbatim"], records[0]["text"])

    def test_provider_url_is_applied_without_changing_footnote_text(self) -> None:
        text = (
            "Criminal Code, RSC 1985, c C-46, s 7; "
            "R v Example, 2024 SCC 1"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.docx"
            output = root / "linked.docx"
            _fixture(source, text)
            plan = plan_docx_links(source, strategy="hybrid")
            self.assertEqual(plan["telemetry"]["codex_batches"], 0)
            links = {
                "2:1": "https://laws.example.test/code#sec7",
                "2:2": "https://cases.example.test/example",
            }
            result = apply_docx_links(source, plan, links, output)
            self.assertEqual(result["linked_parts"], 2)
            with zipfile.ZipFile(output) as archive:
                footnotes = ET.fromstring(archive.read("word/footnotes.xml"))
                rendered = "".join(
                    node.text or ""
                    for node in footnotes.iter(f"{{{W_NS}}}t")
                )
                self.assertEqual(rendered, text)
                relationships = ET.fromstring(
                    archive.read("word/_rels/footnotes.xml.rels")
                )
            targets = {
                node.get("Target")
                for node in relationships.findall(f"{{{PKG_REL_NS}}}Relationship")
            }
            self.assertEqual(targets, set(links.values()))


if __name__ == "__main__":
    unittest.main()
