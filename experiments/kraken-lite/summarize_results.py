import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NATIVE = ROOT / 'kraken-lite-native'
BROWSER = ROOT / 'kraken-lite-browser'


def load(path, engine):
    data = json.loads(path.read_text(encoding='utf-8'))
    return data['engines'][engine], data['engines']['tesseract'], sum(row['gold_chars'] for row in data['pages'])


def combined(rows):
    pages = sum(row['pages'] for row in rows)
    chars = sum(row['chars'] for row in rows)
    return {
        'pages': pages,
        'kraken_cer': sum(row['kraken_cer'] * row['chars'] for row in rows) / chars,
        'tesseract_cer': sum(row['tesseract_cer'] * row['chars'] for row in rows) / chars,
        'kraken_pages_per_second': pages / sum(row['pages'] / row['kraken_pages_per_second'] for row in rows),
        'tesseract_pages_per_second': pages / sum(row['pages'] / row['tesseract_pages_per_second'] for row in rows),
    }


def collect(label, root, pattern, engine):
    rows = []
    for stratum, pages in (('diversified', 30), ('validation', 68), ('heldout', 55)):
        path = root / pattern.format(stratum=stratum, pages=pages)
        kraken, tess, chars = load(path, engine)
        rows.append({'stratum': stratum, 'pages': pages, 'chars': chars,
                     'kraken_cer': kraken['cer'], 'tesseract_cer': tess['cer'],
                     'kraken_pages_per_second': kraken['pages_per_second'],
                     'tesseract_pages_per_second': tess['pages_per_second'], 'receipt': str(path.relative_to(ROOT))})
    return {'tier': label, 'strata': rows, 'combined': combined(rows)}


def main():
    report = {
        'protocol': {'normalization': 'nfkc-collapse-not-soft-hyphen-v1', 'pages': 153,
                     'reporting': 'each disjoint stratum plus character-weighted CER and summed-time throughput'},
        'native': [
            collect('quality-big-int8', NATIVE, 'benchmark-native-persistent-{stratum}-{pages}-bigq100-serial.json', 'native'),
            collect('balanced-cascade-070', NATIVE, 'benchmark-native-persistent-{stratum}-{pages}-cascade070-serial.json', 'native'),
            collect('speed-student', NATIVE, 'benchmark-native-persistent-{stratum}-{pages}-student100-serial.json', 'native'),
        ],
        'browser': [collect(f'student-scale-{scale}', BROWSER, f'benchmark-browser-system-{{stratum}}-{{pages}}-small{code}-fair.json', 'kraken')
                    for scale, code in (('1.00', '100'), ('0.85', '085'), ('0.70', '070'), ('0.62', '062'))],
        'blla': collect('geometry-full-stock-blla', NATIVE, 'benchmark-native-{stratum}-{pages}-full-blla-fidelity.json', 'native'),
    }
    target = ROOT / 'benchmark-summary.json'
    target.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
