import argparse
import json
import os
import sqlite3
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import fitz
from PIL import Image

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[3]
sys.path.insert(0, str(WORKSPACE / "AuthoritiesHelper"))
from toa_maker import Authority, _download_pdf
from build_scan_silver import TESSERACT, align_pages, atomic, pagexml

DB = Path(os.environ["LOCALAPPDATA"]) / "OpenLegalProducts/LegalData/providers/courtlistener/courtlistener.sqlite"


def selected(db, count):
    rows = db.execute("""
        select c.id, c.case_name, c.date_filed, c.filepath_pdf_harvard,
               max(o.plain_text), group_concat(distinct ci.reporter)
        from cluster c join opinion o on o.cluster_id=c.id
        left join citation ci on ci.cluster_id=c.id
        where c.filepath_pdf_harvard is not null
          and length(coalesce(o.plain_text,'')) between 4000 and 60000
        group by c.id order by c.date_filed, c.id
    """).fetchall()
    return [rows[round(i * (len(rows) - 1) / max(1, count - 1))] for i in range(count)]


def raster_ratio(document):
    full = 0
    for page in document:
        area = page.rect.get_area()
        best = max((rect.get_area() / area for image in page.get_images(full=True)
                    for rect in page.get_image_rects(image[0])), default=0)
        full += best >= .70
    return full / max(1, len(document))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "courtlistener-scan-silver")
    parser.add_argument("--documents", type=int, default=32)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB)
    for number, (cluster, name, date, relative, clean, reporters) in enumerate(selected(db, args.documents), 1):
        case = args.output / str(cluster)
        receipt = case / "receipt.json"
        if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")).get("status") == "ready":
            try:
                for xml in case.glob("page-*.xml"):
                    ET.parse(xml)
            except ET.ParseError:
                print(f"{number}/{args.documents} repair_xml {cluster}", flush=True)
            else:
                print(f"{number}/{args.documents} skip {cluster}", flush=True)
                continue
        case.mkdir(exist_ok=True)
        url = f"https://storage.courtlistener.com/{relative.lstrip('/')}"
        authority = Authority(key=str(cluster), kind="cases", citation=str(cluster), name=name or str(cluster),
                              source_url=url, source_text=clean, tab="CL")
        existing = next(case.glob("*.pdf"), None)
        if existing:
            source = existing
            print(f"{number}/{args.documents} cached {cluster}", flush=True)
        else:
            print(f"{number}/{args.documents} download {cluster}", flush=True)
            try:
                _download_pdf(authority, case, "originals")
            except OSError as error:
                existing = next(case.glob("*.pdf"), None)
                if not existing:
                    atomic(receipt, json.dumps({"status": "source_error", "url": url, "error": str(error)}) + "\n")
                    print(f"{number}/{args.documents} source_error {cluster}", flush=True)
                    continue
                authority.pdf_path = str(existing)
                authority.pdf_origin = "original"
            source = Path(authority.pdf_path)
            if authority.pdf_origin != "original":
                source.unlink(missing_ok=True)
                atomic(receipt, json.dumps({"status": "source_error", "url": url}) + "\n")
                print(f"{number}/{args.documents} source_error {cluster}", flush=True)
                continue
        pdf = fitz.open(source)
        print(f"{number}/{args.documents} validate {cluster} pages={len(pdf)}", flush=True)
        if raster_ratio(pdf) < .8:
            atomic(receipt, json.dumps({"status": "excluded_not_scan", "url": url, "pages": len(pdf)}) + "\n")
            print(f"{number}/{args.documents} not_scan {cluster}", flush=True)
            continue
        images, observed = [], []
        for page_number, page in enumerate(pdf, 1):
            image_path = case / f"page-{page_number:03}.png"
            if not image_path.exists():
                page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY).save(image_path)
            with Image.open(image_path) as loaded:
                image = loaded.copy()
            ocr_path = image_path.with_suffix(".ocr.txt")
            if ocr_path.exists():
                ocr = ocr_path.read_text(encoding="utf-8")
            else:
                run = subprocess.run([TESSERACT, image_path, "stdout", "-l", "eng", "--psm", "3"],
                                     capture_output=True, text=True, encoding="utf-8", check=True)
                ocr = run.stdout
                atomic(ocr_path, ocr)
            images.append(image)
            observed.append(ocr)
            print(f"{number}/{args.documents} {cluster} page={page_number}/{len(pdf)}", flush=True)
        chunks, checks = align_pages(clean, observed)
        for page_number, (image, text) in enumerate(zip(images, chunks), 1):
            pagexml(case / f"page-{page_number:03}.xml", image, text)
        atomic(receipt, json.dumps({"status": "ready", "label": "provisional derivative alignment; not benchmark gold", "cluster": cluster,
               "name": name, "date": date, "reporters": reporters, "url": url, "pages": len(images),
               "page_checks": checks}, ensure_ascii=False, indent=2) + "\n")
        print(f"{number}/{args.documents} ready {cluster} pages={len(images)} min_coverage="
              f"{min(x['silver_coverage'] for x in checks):.1%}", flush=True)
    db.close()
    all_pages, cer_pages = [], []
    for receipt in args.output.glob("*/receipt.json"):
        data = json.loads(receipt.read_text(encoding="utf-8"))
        if data.get("status") != "ready":
            continue
        pages = sorted(receipt.parent.glob("page-*.xml"))
        all_pages.extend(page.resolve() for page in pages)
        cer_pages.extend(page.resolve() for page, check in zip(pages, data["page_checks"])
                         if check["matched_tokens"] >= 100
                         and check["ocr_coverage"] >= .75
                         and check["silver_coverage"] >= .75)
    atomic(args.output / "candidate-pages.lst", "".join(f"{page}\n" for page in all_pages))
    atomic(args.output / "cer-pages.lst", "")
    atomic(args.output / "audit.json", json.dumps({
        "status": "rejected_for_cer_benchmarking",
        "documents": 32,
        "scan_pages": len(all_pages),
        "provisional_candidate_pages": len(all_pages),
        "automatically_aligned_pages_meeting_old_threshold": len(cer_pages),
        "benchmark_pages": 0,
        "findings": [
            "Page boundaries and candidate admission depend on Tesseract OCR token matches, biasing a Tesseract comparison.",
            "CourtListener official-court text and Harvard/CAP reporter scans are different editions; editorial and pagination differences were not reconciled page by page.",
            "The deterministic sample covers only decision years 1996, 2016, and 2017, so it is not temporally broad.",
            "Reading order, footnote placement, omissions, and duplications were visually checked on only six pages, not every candidate page.",
        ],
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"manifest provisional_candidates={len(all_pages)} benchmark_pages=0", flush=True)


if __name__ == "__main__":
    main()
