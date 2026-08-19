import argparse
import json
import re
import urllib.request
from pathlib import Path

import duckdb
import fitz

PARQUET = Path.home() / 'AppData/Local/OpenLegalProducts/LegalData/providers/a2aj/source/laws/LEGISLATION-FED/train.parquet'
OUTPUT = Path('kraken-lite-native/court-scan-corpus')


def atomic(path, text):
    tmp = path.with_suffix('.tmp'); tmp.write_text(text, encoding='utf-8'); tmp.replace(path)


def candidates():
    path = str(PARQUET).replace("'", "''")
    rows = duckdb.connect().execute(f"""select citation_en,name_en,source_url_en,unofficial_text_en
        from read_parquet('{path}') where length(unofficial_text_en) between 5000 and 30000
        and source_url_en like '%/XML/%.xml' order by length(unofficial_text_en)""").fetchall()
    for index in range(0, len(rows), max(1, len(rows) // 30)):
        yield rows[index]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--limit', type=int, default=8); args = parser.parse_args()
    accepted = 0
    for citation, name, xml_url, clean in candidates():
        code = xml_url.rsplit('/', 1)[-1][:-4]
        key = 'LEG-FED-' + re.sub(r'[^A-Za-z0-9.-]+', '-', code)
        folder = OUTPUT / key; receipt = folder / 'receipt.json'
        if receipt.exists():
            data = json.loads(receipt.read_text(encoding='utf-8'))
            if data.get('status') == 'excluded_not_full_page_raster': accepted += 1
            if accepted >= args.limit: break
            continue
        folder.mkdir(exist_ok=True)
        pdf_url = f'https://laws-lois.justice.gc.ca/PDF/{code}.pdf'; pdf_path = folder / 'source.pdf'
        try:
            request = urllib.request.Request(pdf_url, headers={'User-Agent': 'Mozilla/5.0 exact-ocr-benchmark/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.headers.get_content_type() != 'application/pdf': raise ValueError('not a PDF')
                data = response.read()
            tmp = pdf_path.with_suffix('.tmp'); tmp.write_bytes(data); tmp.replace(pdf_path)
            document = fitz.open(pdf_path)
            if not 2 <= len(document) <= 30: raise ValueError(f'page count {len(document)} outside 2..30')
            atomic(folder / 'clean.txt', clean)
            atomic(receipt, json.dumps({'status': 'excluded_not_full_page_raster', 'court': 'LEGISLATION-FED', 'citation': citation, 'name': name, 'source_url': xml_url, 'pdf_url': pdf_url, 'pages': len(document)}, ensure_ascii=False, indent=2))
            accepted += 1; print(f'accepted source {key} pages={len(document)}', flush=True)
            if accepted >= args.limit: break
        except Exception as error:
            atomic(receipt, json.dumps({'status': 'source_error', 'pdf_url': pdf_url, 'error': str(error)}, indent=2)); print(f'skip {key}: {error}', flush=True)
    print(f'sources={accepted}', flush=True)


if __name__ == '__main__': main()
