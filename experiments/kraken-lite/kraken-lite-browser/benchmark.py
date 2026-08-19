import argparse
import json
import re
import subprocess
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from rapidfuzz.distance.Levenshtein import distance as _fast_distance
except ImportError:
    _fast_distance = None

ROOT = Path(__file__).resolve().parent
DEFAULT_LIST = ROOT.parent / 'kraken-lite-student/known-good-input/host_test.lst'
TESSERACT = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
NORMALIZATION = 'nfkc-collapse-not-soft-hyphen-v1'


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def normalized(text):
    text = re.sub(r'[\u00ad\u00ac]\r?\n', '', text)
    text = unicodedata.normalize('NFKC', text).replace('\u00ad', '').replace('¬\n', '')
    text = text.replace('\u00ac', '')
    return re.sub(r'\s+', ' ', text).strip()


def distance(a, b):
    if _fast_distance:
        return _fast_distance(a, b)
    if len(a) > len(b):
        a, b = b, a
    row = list(range(len(a) + 1))
    for i, cb in enumerate(b, 1):
        previous, row[0] = row[0], i
        for j, ca in enumerate(a, 1):
            old = row[j]
            row[j] = min(row[j] + 1, row[j - 1] + 1, previous + (ca != cb))
            previous = old
    return row[-1]


def gold(xml_path):
    root = ET.parse(xml_path).getroot()
    lines = []
    for line in root.iter():
        if not line.tag.endswith('TextLine'):
            continue
        match = re.search(r'readingOrder\s*\{index:(\d+)', line.get('custom', ''))
        text = next((node.text or '' for node in line.iter() if node.tag.endswith('Unicode')), '')
        if text:
            lines.append((int(match.group(1)) if match else 10**9, text))
    return normalized('\n'.join(text for _, text in sorted(lines, key=lambda item: item[0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--url', default='http://127.0.0.1:8770/')
    parser.add_argument('--list', type=Path, default=DEFAULT_LIST)
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-results.json')
    parser.add_argument('--mode', choices=('.7', '.85', '1'), default='.7')
    parser.add_argument('--reuse-tesseract', type=Path)
    args = parser.parse_args()
    xmls = [Path(line.strip()) for line in args.list.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    xmls = xmls[args.offset:]
    if args.limit:
        xmls = xmls[:args.limit]
    images = [path.with_suffix('.png') for path in xmls]
    if missing := [str(path) for path in images if not path.exists()]:
        raise FileNotFoundError(missing[0])

    with tempfile.NamedTemporaryFile('w', suffix='.txt', encoding='utf-8', delete=False) as handle:
        handle.write('\n'.join(str(path) for path in images))
        image_list = handle.name
    browser = subprocess.Popen(
        ['node', str(ROOT / 'benchmark-browser.mjs'), args.url, image_list, args.mode],
        cwd=ROOT.parent, stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
    kraken = []
    for line in browser.stdout:
        item = json.loads(line)
        kraken.append(item)
        print(f"kraken {len(kraken)}/{len(images)} {item['seconds']:.3f}s", flush=True)
    if browser.wait():
        raise RuntimeError('browser benchmark failed')

    reused = json.loads(args.reuse_tesseract.read_text(encoding='utf-8')) if args.reuse_tesseract else None
    reused_pages = {page['image']: page for page in reused['pages']} if reused else {}
    reused_seconds = reused['engines']['tesseract']['seconds'] / len(reused['pages']) if reused else 0
    rows = []
    for index, (xml_path, image, kraken_item) in enumerate(zip(xmls, images, kraken), 1):
        truth = gold(xml_path)
        ktext = normalized(kraken_item['text'])
        if reused:
            key = str(image.resolve())
            prior = reused_pages.get(key)
            if prior is None:
                raise ValueError(f'reused receipt has no unambiguous image identity: {key}')
            seconds = reused_seconds
            raw_ttext = prior.get('tesseract_text', prior.get('tesseract', {}).get('raw_text', prior.get('tesseract', {}).get('text', '')))
            ttext = normalized(raw_ttext)
            tess_edits = distance(truth, ttext)
        else:
            start = time.perf_counter()
            run = subprocess.run([TESSERACT, image, 'stdout', '-l', 'eng', '--psm', '3'], capture_output=True, text=True, encoding='utf-8')
            seconds = time.perf_counter() - start
            if run.returncode:
                raise RuntimeError(run.stderr)
            ttext = normalized(run.stdout)
            tess_edits = distance(truth, ttext)
        row = {
            'image': str(image.resolve()), 'gold_chars': len(truth),
            'kraken': {'seconds': kraken_item['seconds'], 'edits': distance(truth, ktext), 'text': ktext, 'raw_text': kraken_item['text']},
            'tesseract': {'seconds': seconds, 'edits': tess_edits, 'text': ttext},
            'gold': truth,
        }
        rows.append(row)
        if not reused:
            print(f"tesseract {index}/{len(images)} {seconds:.3f}s", flush=True)

    result = {'protocol': {'split': str(args.list), 'pages': len(rows), 'normalization': NORMALIZATION, 'tesseract': '5.4.0 eng --psm 3', 'kraken_url': args.url, 'kraken_mode': args.mode}, 'engines': {}, 'pages': rows}
    chars = sum(row['gold_chars'] for row in rows)
    for engine in ('kraken', 'tesseract'):
        seconds = sum(row[engine]['seconds'] for row in rows)
        edits = sum(row[engine]['edits'] for row in rows)
        result['engines'][engine] = {'cer': edits / chars, 'seconds': seconds, 'pages_per_second': len(rows) / seconds, 'edits': edits, 'gold_chars': chars}
    atomic_json(args.output, result)
    print(json.dumps(result['engines'], indent=2), flush=True)


if __name__ == '__main__':
    main()
