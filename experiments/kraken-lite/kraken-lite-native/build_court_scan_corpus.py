import argparse
import difflib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree

import pypdfium2 as pdfium

WORKSPACE = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(WORKSPACE / 'TableOfAuthoritiesMaker'))
from toa_maker import Authority, _download_pdf

DB = Path.home() / 'AppData/Local/OpenLegalProducts/LegalData/providers/a2aj/a2aj-cases-fulltext.sqlite'
TESSERACT = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15'
COURTS = ('SCC', 'ONCA', 'BCCA', 'BCSC', 'FCA', 'FC', 'NSCA', 'NSSC')


def atomic_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def candidates(limit):
    db = sqlite3.connect(DB)
    placeholders = ','.join('?' for _ in COURTS)
    rows = db.execute(f"""select dataset,citation_en,name_en,document_date_en,url_en,unofficial_text_en
        from document where dataset in ({placeholders}) and url_en is not null
        and length(unofficial_text_en) between 4000 and 30000 order by document_date_en""", COURTS).fetchall()
    out = []
    for court in COURTS:
        pool = [row for row in rows if row[0] == court]
        out.extend(pool[round(i * (len(pool) - 1) / max(1, limit - 1))] for i in range(limit))
    return out


def words(text):
    return re.findall(r"[\w’'-]+", text, re.UNICODE)


def page_texts(images):
    texts = []
    for image in images:
        run = subprocess.run([TESSERACT, str(image), 'stdout', '--psm', '3'], capture_output=True, text=True, encoding='utf-8')
        texts.append(run.stdout)
    return texts


def split_gold(gold, observed):
    target, source = words(gold), [word for page in observed for word in words(page)]
    matcher = difflib.SequenceMatcher(None, [x.casefold() for x in source], [x.casefold() for x in target], autojunk=False)
    mapped = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            mapped[block.a + offset] = block.b + offset
    boundaries, consumed = [0], 0
    for page in observed[:-1]:
        consumed += len(words(page))
        near = [value for key, value in mapped.items() if abs(key - consumed) <= 80]
        boundaries.append(round(sum(near) / len(near)) if near else round(consumed / max(1, len(source)) * len(target)))
    boundaries.append(len(target))
    boundaries = [max(boundaries[i - 1] if i else 0, value) for i, value in enumerate(boundaries)]
    return [' '.join(target[a:b]) for a, b in zip(boundaries, boundaries[1:])], len(mapped) / max(1, len(source))


def write_pagexml(path, image, width, height, text):
    root = Element('PcGts', xmlns=NS)
    page = SubElement(root, 'Page', imageFilename=image.name, imageWidth=str(width), imageHeight=str(height))
    region = SubElement(page, 'TextRegion', id='r1')
    SubElement(region, 'Coords', points=f'0,0 {width-1},0 {width-1},{height-1} 0,{height-1}')
    line = SubElement(region, 'TextLine', id='l1')
    SubElement(line, 'Coords', points=f'0,0 {width-1},0 {width-1},{height-1} 0,{height-1}')
    equiv = SubElement(line, 'TextEquiv')
    SubElement(equiv, 'Unicode').text = text
    tmp = path.with_suffix('.tmp')
    ElementTree(root).write(tmp, encoding='utf-8', xml_declaration=True)
    tmp.replace(path)


def full_page_raster_count(pdf):
    count = 0
    for page in pdf:
        area = page.get_width() * page.get_height()
        for image in page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]):
            left, bottom, right, top = image.get_bounds()
            if max(0, right - left) * max(0, top - bottom) >= area * .7:
                count += 1
                break
    return count


def write_manifest(root):
    pages = []
    for receipt in sorted(root.glob('*/receipt.json')):
        data = json.loads(receipt.read_text(encoding='utf-8'))
        if data.get('status') in ('ready', 'mapped_derivative', 'provisional_low_alignment'):
            pdf = pdfium.PdfDocument(receipt.parent / 'source.pdf')
            full = full_page_raster_count(pdf)
            if full / len(pdf) < .8:
                data['status'] = 'excluded_not_full_page_raster'
                data['full_page_raster_pages'] = full
                atomic_json(receipt, data)
        if data.get('status') == 'human_verified':
            pages.extend(sorted(receipt.parent.glob('page-*.xml')))
    path = root / 'provisional-aligned-pages.lst'
    tmp = path.with_suffix('.tmp')
    tmp.write_text(''.join(f'{page.resolve()}\n' for page in pages), encoding='utf-8')
    tmp.replace(path)
    print(f'manifest pages={len(pages)} {path}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('kraken-lite-native/court-scan-corpus'))
    parser.add_argument('--limit-per-court', type=int, default=12)
    parser.add_argument('--max-pages', type=int, default=20)
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.manifest_only:
        write_manifest(args.output)
        return
    for court, citation, name, date, source_url, gold in candidates(args.limit_per_court):
        key = re.sub(r'[^A-Za-z0-9]+', '-', f'{court}-{citation}').strip('-')[:90]
        case = args.output / key
        receipt = case / 'receipt.json'
        if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')).get('status') != 'error':
            print(f'skip {key}', flush=True); continue
        case.mkdir(exist_ok=True)
        pdf_path = case / 'source.pdf'
        pdf_url = source_url
        try:
            if not pdf_path.exists():
                authority = Authority(key=key, kind='cases', citation=citation, name=name, source_url=source_url, tab=court)
                _download_pdf(authority, case, 'originals')
                if authority.pdf_origin != 'original':
                    Path(authority.pdf_path).unlink(missing_ok=True)
                    raise RuntimeError('Authorities could not resolve an original PDF')
                downloaded = Path(authority.pdf_path)
                downloaded.replace(pdf_path)
                pdf_url = authority.pdf_source_url
            pdf = pdfium.PdfDocument(pdf_path)
            if not 1 <= len(pdf) <= args.max_pages:
                atomic_json(receipt, {'status': 'excluded_page_count', 'pages': len(pdf), 'url': pdf_url}); print(f'exclude {key} pages={len(pdf)}', flush=True); continue
            images = []
            raster_pages = full_page_raster_count(pdf)
            for number, page in enumerate(pdf):
                image = case / f'page-{number+1:03}.png'
                if not image.exists(): page.render(scale=2).to_pil().save(image)
                images.append(image)
            if raster_pages / len(pdf) < .8:
                atomic_json(receipt, {'status': 'excluded_not_raster', 'pages': len(pdf), 'raster_pages': raster_pages, 'url': pdf_url}); print(f'exclude {key} raster={raster_pages}/{len(pdf)}', flush=True); continue
            observed = page_texts(images)
            chunks, coverage = split_gold(gold, observed)
            for image, page, text in zip(images, pdf, chunks):
                write_pagexml(image.with_suffix('.xml'), image, round(page.get_width() * 2), round(page.get_height() * 2), text)
            status = 'mapped_derivative' if coverage >= .55 else 'provisional_low_alignment'
            atomic_json(receipt, {'status': status, 'gold_status': 'derivative_not_benchmark_gold', 'court': court, 'citation': citation, 'name': name, 'date': date, 'source_url': source_url, 'pdf_url': pdf_url, 'pages': len(pdf), 'alignment_coverage': coverage})
            print(f'{status} {key} pages={len(pdf)} alignment={coverage:.1%}', flush=True)
        except Exception as error:
            atomic_json(receipt, {'status': 'error', 'url': pdf_url, 'error': str(error)}); print(f'error {key}: {error}', flush=True)
    write_manifest(args.output)


if __name__ == '__main__':
    main()
