"""Check source-reviewed SEC complaint facts, independent of extractor snapshots."""
import json
import re
import sys
from pathlib import Path

SOURCE = "8c4578992501e4d3739e07e81b81e52b88d93e429afe6e1cefedec0585615027"
HEADINGS = [
    "I. INTRODUCTION", "II. DEFENDANT", "III. RELEVANT ENTITY",
    "IV. JURISDICTION AND VENUE", "V. FACTS", "VI. CLAIMS FOR RELIEF",
    "VII. RELIEF REQUESTED", "VIII. DEMAND FOR JURY TRIAL",
]
NOTE_GOLD = {
    "0ed0e1e6b3f28fd6c8c3a57647925b873549e528eb70e9d072277bb3f4bd147a": [("7", "Policy, September 2010 at 4, 12, online (PDF)")],
    "af11dde972f508fde9ebb63bbc855202500589f25801d8b4b06977ec757bd858": [("7", "Robert Fife and Steven Chase"), ("7", "COM0000067")],
    "e5293c3f7d294ebe88759effe749f26c52eaa770177a5747efd4201239c816a0": [("4", "This includes staff of"), ("4", "Commodity Futures Trading Commission")],
    "0950c3b8cd3be84ee72f8d4b5002be0075bd52fafed3e65d454c87fe339c8489": [("*", "witnesses also appearing at hearing week 5")],
}


def check_note_gold(structure):
    source = structure["text"].encode("utf-16-le")
    for label, phrase in NOTE_GOLD[structure["source_sha256"]]:
        bodies = [" ".join(source[n["range"]["start"]*2:n["range"]["end"]*2].decode("utf-16-le").split())
                  for n in structure["nodes"] if n["kind"] == "footnote" and n["label"] == label]
        assert any(phrase in body for body in bodies), f"Source footnote {label} lost: {phrase}"


def check(structure):
    assert structure["source_sha256"] == SOURCE, "Wrong source document"
    assert not any(node["kind"] == "footnote" for node in structure["nodes"]), \
        "The nine-page complaint has no footnotes"
    source = structure["text"].encode("utf-16-le")

    def text(node):
        span = node["range"]
        return source[span["start"] * 2:span["end"] * 2].decode("utf-16-le")

    headings = [node for node in structure["nodes"] if node["kind"] == "heading"]
    for expected in HEADINGS:
        matches = [node for node in headings if " ".join(text(node).split()) == expected]
        assert len(matches) == 1, f"Missing or duplicated source heading: {expected}"
    paragraph_nodes = [node for node in structure["nodes"] if node["kind"] == "paragraph"]
    assert len(paragraph_nodes) == 34, "Expected exactly 34 numbered paragraphs"
    paragraphs = {node["label"]: node for node in paragraph_nodes}
    assert set(paragraphs) == {f"par{number}" for number in range(1, 35)}, "Expected paragraphs 1 through 34"
    for label, paragraph in paragraphs.items():
        span = paragraph["range"]
        assert not any(span["start"] < heading["range"]["start"] < span["end"] for heading in headings), \
            f"{label} contains a following heading"
    assert "2018. The Board consisted of four investors" in " ".join(text(paragraphs["par14"]).split())
    assert text(paragraphs["par14"]).strip().endswith("ensued."), "Paragraph 14 is truncated"
    assert not any(node["kind"] == "section" and re.match(r"15\s+U\.?S\.?C", text(node))
                   for node in structure["nodes"]), "A statute citation became a section"


def check_bce(structure):
    source = structure["text"].encode("utf-16-le")
    for label, phrase in [("89", "Refer to Appendix C for sources of all headline data"),
                          ("90", "In addition, there are some small localised cable networks in regional population centres and in Canberra.")]:
        matches = [node for node in structure["nodes"] if node["kind"] == "footnote" and node["label"] == label]
        assert len(matches) == 1, f"BCE page 63 must retain footnote {label}"
        span = matches[0]["range"]
        text = source[span["start"] * 2:span["end"] * 2].decode("utf-16-le")
        assert phrase in " ".join(text.split()), f"BCE footnote {label} lost its source body"


def check_westport(structure):
    # Visible stamp on source page 84, stored as an annotation appearance.
    assert "U.S. MAGISTRATE JUDGE" in " ".join(structure["text"].split()), \
        "Westport's visible judicial stamp must be extracted"


if __name__ == "__main__":
    raw = Path(sys.argv[1]).read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig").splitlines()]
    checks = {SOURCE: check,
              "40941dbae64682c658781dd5ebb49e1dd4ea7c46306e0aa57f2abfcd52a52c85": check_bce,
              "44d79a58446882328a18dcca31cb1bb29d96487c6624a27e9de65297767da972": check_westport}
    checks.update({sha: check_note_gold for sha in NOTE_GOLD})
    matched = 0
    for sha, verify in checks.items():
        matching = [row for row in rows if row["structure"]["source_sha256"] == sha]
        if not matching:
            continue
        assert len(matching) == 1, "Source-reviewed document must occur once"
        verify(matching[0]["structure"])
        matched += 1
    assert matched, "No source-reviewed documents in this result"
    print(f"Independent PDF source gold: {matched} documents")
    if len(sys.argv) > 2:
        expected = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))["documents"]
        actual = [{"path": row["path"], "source_sha256": row["structure"]["source_sha256"],
                   "product_sha256": row["product_sha256"]} for row in rows]
        assert actual == expected, "Frozen PDF products changed; review source evidence before updating regression receipts"
        print(f"Frozen PDF regression products: {len(actual)} documents")
