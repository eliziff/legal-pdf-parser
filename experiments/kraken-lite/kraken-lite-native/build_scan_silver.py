import difflib
import json
import re
import sqlite3
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image
import numpy as np


ROOT = Path(__file__).parent / "court-scan-corpus"
OUT = Path(__file__).parent / "scan-silver"
DB = Path.home() / "AppData/Local/OpenLegalProducts/LegalData/providers/a2aj/a2aj-cases-fulltext.sqlite"
TESSERACT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
CASES = ("SCC-1970-SCR-638", "SCC-1976-2-SCR-475", "SCC-1977-2-SCR-400", "SCC-1989-2-SCR-778")
STARTS = {
    "SCC-1970-SCR-638": "Her Majesty the Queen",
    "SCC-1976-2-SCR-475": "Baxter Student Housing Ltd",
    "SCC-1977-2-SCR-400": "J A Madill",
    "SCC-1989-2-SCR-778": "Falk Bros Industries Ltd",
}
NS = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"


def words(text):
    return re.findall(r"[\w’'-]+", text, re.UNICODE)


def norm(word):
    return word.casefold().replace("’", "'")


def decision_content(text):
    parts = re.split(r"\bDecision Content\b", text, maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        raise RuntimeError("provider text has no Decision Content boundary")
    return parts[1].strip()


def visible_start(text, phrase):
    match = re.search(r"\W+".join(map(re.escape, phrase.split())), text, re.IGNORECASE)
    if not match:
        raise RuntimeError(f"missing visible start anchor: {phrase}")
    return text[match.start():]


def english_body(page):
    gray = np.asarray(page.convert("L"))
    left = round(page.width * .01)
    low, high = round(page.width * .45), round(page.width * .55)
    columns = np.count_nonzero(gray[round(page.height * .08):round(page.height * .92), low:high] < 180, axis=0)
    smooth = np.convolve(columns, np.ones(9), mode="same")
    right = low + int(np.argmin(smooth[4:-4])) + 4
    darkness = np.count_nonzero(gray[:round(page.height * .14), left:right] < 100, axis=1)
    rules = np.flatnonzero(darkness > (right - left) * .35)
    top = int(rules[-1] + 8) if rules.size else round(page.height * .06)
    return page.crop((left, top, right, round(page.height * .925))).convert("L")


def atomic(path, text):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def pagexml(path, image, text):
    text = "".join(character for character in text if character in "\t\n\r" or ord(character) >= 32)
    root = ET.Element("PcGts", xmlns=NS)
    page = ET.SubElement(root, "Page", imageFilename=path.with_suffix(".png").name, imageWidth=str(image.width), imageHeight=str(image.height))
    region = ET.SubElement(page, "TextRegion", id="r1")
    ET.SubElement(region, "Coords", points=f"0,0 {image.width-1},0 {image.width-1},{image.height-1} 0,{image.height-1}")
    line = ET.SubElement(region, "TextLine", id="l1", custom="readingOrder {index:0;}")
    ET.SubElement(line, "Coords", points=f"0,0 {image.width-1},0 {image.width-1},{image.height-1} 0,{image.height-1}")
    ET.SubElement(ET.SubElement(line, "TextEquiv"), "Unicode").text = text
    tmp = path.with_suffix(".tmp")
    ET.ElementTree(root).write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.replace(path)


def align_pages(clean, observed):
    target_matches = list(re.finditer(r"[\w’'-]+", clean, re.UNICODE))
    target = [match.group() for match in target_matches]
    source_pages = [words(text) for text in observed]
    source = [word for page in source_pages for word in page]
    matcher = difflib.SequenceMatcher(None, list(map(norm, source)), list(map(norm, target)), autojunk=False)
    mapping = {block.a + i: block.b + i for block in matcher.get_matching_blocks() for i in range(block.size)}
    source_starts, total = [0], 0
    for page in source_pages:
        total += len(page)
        source_starts.append(total)
    boundaries = []
    for boundary in source_starts:
        exact = [value for key, value in mapping.items() if abs(key - boundary) <= 40]
        boundaries.append(round(sum(exact) / len(exact)) if exact else None)
    first = min(mapping.values()) if mapping else 0
    last = max(mapping.values()) + 1 if mapping else len(target)
    boundaries[0], boundaries[-1] = first, last
    for index in range(1, len(boundaries) - 1):
        if boundaries[index] is None:
            fraction = source_starts[index] / max(1, source_starts[-1])
            boundaries[index] = round(first + fraction * (last - first))
    boundaries = [max(boundaries[index - 1] if index else first, value) for index, value in enumerate(boundaries)]
    chunks = []
    for a, b in zip(boundaries, boundaries[1:]):
        start = target_matches[a].start() if a < len(target_matches) else len(clean)
        end = target_matches[b].start() if b < len(target_matches) else len(clean)
        chunks.append(clean[start:end].strip())
    checks = []
    for page_words, chunk in zip(source_pages, chunks):
        chunk_words = words(chunk)
        blocks = difflib.SequenceMatcher(None, list(map(norm, page_words)), list(map(norm, chunk_words)), autojunk=False).get_matching_blocks()
        matched = sum(block.size for block in blocks)
        checks.append({"ocr_tokens": len(page_words), "silver_tokens": len(chunk_words), "matched_tokens": matched,
                       "ocr_coverage": matched / max(1, len(page_words)), "silver_coverage": matched / max(1, len(chunk_words))})
    return chunks, checks


def main():
    OUT.mkdir(exist_ok=True)
    db = sqlite3.connect(DB)
    manifest, audit = [], {
        "label": "unverified automatic alignment candidates",
        "selection": "not benchmark eligible; every page requires scan-level manual correction and verification",
        "documents": [],
    }
    for key in CASES:
        source = ROOT / key
        receipt = json.loads((source / "receipt.json").read_text(encoding="utf-8"))
        row = db.execute("select unofficial_text_en from document where dataset='SCC' and citation_en=? limit 1", (receipt["citation"],)).fetchone()
        if not row:
            raise RuntimeError(f"missing provider text: {key}")
        destination = OUT / key
        destination.mkdir(exist_ok=True)
        images, observed = [], []
        for source_image in sorted(source.glob("page-*.png")):
            with Image.open(source_image) as page:
                crop = english_body(page)
            image = destination / source_image.name
            crop.save(image)
            run = subprocess.run([TESSERACT, image, "stdout", "-l", "eng", "--psm", "3"], capture_output=True, text=True, encoding="utf-8", check=True)
            images.append(crop)
            observed.append(run.stdout)
        clean = decision_content(row[0])
        clean = visible_start(clean, STARTS[key])
        chunks, checks = align_pages(clean, observed)
        for number, (image, text) in enumerate(zip(images, chunks), 1):
            xml = destination / f"page-{number:03}.xml"
            pagexml(xml, image, text)
            manifest.append(str(xml.resolve()))
        audit["documents"].append({"case": key, "citation": receipt["citation"], "pages": len(images), "page_checks": checks})
        print(f"{key}: {len(images)} pages", flush=True)
    db.close()
    audit["pages"] = len(manifest)
    audit["minimum_ocr_coverage"] = min(check["ocr_coverage"] for doc in audit["documents"] for check in doc["page_checks"])
    audit["minimum_silver_coverage"] = min(check["silver_coverage"] for doc in audit["documents"] for check in doc["page_checks"])
    atomic(OUT / "candidate-pages.lst", "".join(path + "\n" for path in manifest))
    atomic(OUT / "audit.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: audit[key] for key in ("pages", "minimum_ocr_coverage", "minimum_silver_coverage")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
