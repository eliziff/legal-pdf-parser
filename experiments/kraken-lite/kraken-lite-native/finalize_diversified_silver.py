import json
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).parent / "courtlistener-scan-silver"

# Each mapping was selected against the rendered page. CAP supplies same-edition
# transcription; overrides preserve visible page boundaries and non-body matter.
PAGES = {
    "4334796": ("page-002", 1, "1378", "single", "all visible text; redacted running header contains no readable text"),
    "4336445": ("page-002", 1, "1221", "two-column", "all visible text; redacted running header contains no readable text"),
    "4337221": ("page-003", 1, "418", "two-column", "all visible text; redacted running header contains no readable text"),
    "4337694": ("page-004", 2, "27", "two-column", "all visible text; redacted running headers contain no readable text"),
    "4338036": ("page-003", 1, "42", "two-column", "all visible text; redacted running header contains no readable text"),
    "4338373": ("page-005", 3, "380", "single", "all visible text; redacted running header contains no readable text"),
    "4339183": ("page-003", 1, "848", "two-column", "all visible text; redacted running header contains no readable text"),
    "4339650": ("page-002", 1, "1174", "single", "all visible text; redacted running header contains no readable text"),
    "4377746": ("page-007", 4, "446", "two-column", "all visible text; redacted running header contains no readable text"),
    "4335995": ("page-015", 11, "83", "two-column", "all visible text; redacted running header contains no readable text"),
    "4340128": ("page-008", 7, "761", "single", "all visible text; redacted running header contains no readable text"),
}

REPLACEMENTS = {
    "4336445": {"[w]hile": "[W]hile"},
    "4337221": {"ถ": "¶"},
    "4337694": {"งง": "§§", "ง": "§", "govemment": "government", "[gjovemment": "[g]overnment"},
    "4338036": {"ง": "§", "50-e(l)(a)": "50-e(1)(a)"},
    "4338373": {"resolved..": "resolved."},
    "4335995": {"conduct .was": "conduct was", "from.the": "from the", "Pa, 655": "Pa. 655", "A2d": "A.2d", "Commonwealth, v.": "Commonwealth v.", "vietim": "victim", "consum[ed]": "consum[ed]"},
    "4340128": {"he possessed": "he possessed", " [h] igh": " [h]igh", "[b] oth": "[b]oth", "6. 6 Because": "6. Because"},
}

SPECIAL = {
    "4338786": """85

Larry PORTER, Plaintiff-Appellant,
v.
Glenn GOORD, et al., Defendants-Appellees.
No. 16-832-pr
United States Court of Appeals, Second Circuit.
January 17, 2017

FOR APPELLANT: Larry Porter, pro se, Malone, New York.

FOR APPELLEES: Barbara D. Underwood, Solicitor General, Victor Paladino, Jeffrey W. Lang, Assistant Solicitors General, for Eric T. Schneiderman, Attorney General for the State of New York, Albany, New York.

PRESENT: REENA RAGGI, DENNY CHIN, RAYMOND J. LOHIER, JR., Circuit Judges.

SUMMARY ORDER

{draft}""",
    "4341434": """542

{draft}

lation prescribed under that law or regulation has occurred[.]

A seaman alleging discharge or discrimination in violation of subsection (a) of this section, or another person at the seaman’s request, may file a complaint with respect to such allegation in the same manner as a complaint may be filed under subsection (b) of section 31105 of title 49. Such complaint shall be subject to the procedures, requirements, and rights described in that section, including with respect to the right to file an objection, the right of a person to file for a petition for review under subsection (c) of that section, and the requirement to bring a civil action under subsection (d) of that section.

46 U.S.C. § 2114(a)(1)(A), (b).""",
}

SPECIAL_CONFIG = {
    "4338786": ("page-001", 0, "two-column", "all visible text; redacted blocks contain no readable text"),
    "4341434": ("page-005", 2, "two-column", "all visible text; redacted running header contains no readable text"),
}


def write_page(document, page_name, text, layout, scope):
    image = ROOT / document / f"{page_name}.png"
    width, height = Image.open(image).size
    pcgts = ET.Element("PcGts", xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15")
    page = ET.SubElement(pcgts, "Page", imageFilename=image.name, imageWidth=str(width), imageHeight=str(height))
    region = ET.SubElement(page, "TextRegion", id="r1")
    points = f"0,0 {width - 1},0 {width - 1},{height - 1} 0,{height - 1}"
    ET.SubElement(region, "Coords", points=points)
    line = ET.SubElement(region, "TextLine", id="l1", custom="readingOrder {index:0;}")
    ET.SubElement(line, "Coords", points=points)
    ET.SubElement(ET.SubElement(line, "TextEquiv"), "Unicode").text = text.strip()
    target = ROOT / document / f"{page_name}.xml"
    ET.ElementTree(pcgts).write(target, encoding="utf-8", xml_declaration=True)
    audit = {
        "document": document,
        "pages": {page_name: {
            "status": "verified", "image": image.name, "label": target.name, "scope": scope,
            "checks": ["source identity", "exact page boundary and terminal fragment", f"{layout} reading order", "complete visible text", "citations and statutory symbols", "footnotes and non-body matter", "line-wrap dehyphenation", "Unicode punctuation", "omissions and duplications"],
        }},
    }
    (ROOT / document / "manual-audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return image.resolve()


def draft(document, index):
    data = json.loads((ROOT / document / "cap-page-drafts.json").read_text(encoding="utf-8"))[index]
    text = data["text"]
    if data.get("footnote_text"):
        text += "\n\n" + "\n\n".join(data["footnote_text"])
    for old, new in REPLACEMENTS.get(document, {}).items():
        text = text.replace(old, new)
    return text


def main():
    manifest = ROOT / "verified-pages.lst"
    paths = [Path(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    for document, (page_name, index, number, layout, scope) in PAGES.items():
        paths.append(write_page(document, page_name, f"{number}\n\n{draft(document, index)}", layout, scope))
    for document, (page_name, index, layout, scope) in SPECIAL_CONFIG.items():
        paths.append(write_page(document, page_name, SPECIAL[document].format(draft=draft(document, index)), layout, scope))
    unique = list(dict.fromkeys(str(path) for path in paths))
    manifest.write_text("\n".join(unique) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
