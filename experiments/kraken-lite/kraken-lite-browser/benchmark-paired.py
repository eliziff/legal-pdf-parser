import argparse
import json
import subprocess
import tempfile
import time
import urllib.request
from urllib.parse import urlsplit
import sys
from pathlib import Path

from benchmark import atomic_json, distance, gold, normalized

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'kraken-lite-native'))
from freeze_benchmark_splits import OUTPUT as SPLITS, validate_benchmark_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--mode', choices=('.7', '.76', '.85', '1'), default='.7')
    parser.add_argument('--lane', choices=('system', 'shared', 'both', 'layout'), default='system')
    parser.add_argument('--reuse-tesseract', type=Path)
    parser.add_argument('--tess-data', choices=('fast', 'best-int'), default='fast')
    parser.add_argument('--tess-psm', choices=('AUTO', 'SINGLE_BLOCK', 'SINGLE_COLUMN', 'SPARSE_TEXT'), default='AUTO')
    parser.add_argument('--tess-output', choices=('text', 'blocks'), default='blocks')
    parser.add_argument('--tess-input', choices=('canvas', 'png'), default='canvas')
    parser.add_argument('--tess-workers', type=int, choices=(1, 2, 3, 4), default=1)
    parser.add_argument('--tess-rounds', type=int, default=1)
    parser.add_argument('--kraken-workers', type=int, choices=range(1, 9), default=1)
    parser.add_argument('--kraken-rounds', type=int, default=1)
    parser.add_argument('--kraken-schedule', choices=('round-robin', 'bytes'), default='round-robin')
    parser.add_argument('--kraken-worker-pool', action='store_true')
    parser.add_argument('--url', default='http://127.0.0.1:8771/index.html')
    parser.add_argument('--list', type=Path, default=SPLITS / 'benchmark-153.lst')
    parser.add_argument('--output', type=Path, default=ROOT / 'benchmark-browser-paired.json')
    args = parser.parse_args()
    xmls = [Path(line) for line in args.list.read_text(encoding='utf-8-sig').splitlines() if line]
    xmls = xmls[args.offset:]
    if args.limit:
        xmls = xmls[:args.limit]
    truth_counts = validate_benchmark_paths(xmls)
    with tempfile.NamedTemporaryFile('w', suffix='.txt', encoding='utf-8', delete=False) as handle:
        handle.write('\n'.join(str(path.with_suffix('.png')) for path in xmls))
        image_list = handle.name
    server = subprocess.Popen(['python', str(ROOT / 'serve.py')], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        parsed_url = urlsplit(args.url)
        server_root = f'{parsed_url.scheme}://{parsed_url.netloc}/'
        for _ in range(40):
            try:
                urllib.request.urlopen(server_root, timeout=1).close()
                break
            except Exception:
                time.sleep(.25)
        else:
            raise RuntimeError('browser server did not start')
        url = args.url
        if args.kraken_workers > 1:
            if not args.reuse_tesseract or args.lane != 'system':
                raise ValueError('Kraken worker-pool probes require --reuse-tesseract and --lane system')
            command = ['node', str(ROOT / ('benchmark-kraken-worker-pool.mjs' if args.kraken_worker_pool else 'benchmark-kraken-pool.mjs')), url, image_list, *([str(args.kraken_rounds), args.mode] if args.kraken_worker_pool else [str(args.kraken_workers), str(args.kraken_rounds), args.kraken_schedule])]
        else:
            command = [
                'node', str(ROOT / 'benchmark-browser-paired.mjs'), url, image_list,
                args.mode, args.lane, str(not args.reuse_tesseract).lower(),
                args.tess_data, args.tess_psm, args.tess_output, args.tess_input,
                str(args.tess_workers), str(args.tess_rounds),
            ]
        run = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        outputs = []
        for line in run.stdout:
            outputs.append(json.loads(line))
            print(f"paired {len(outputs)}/{len(xmls)}", flush=True)
        if run.wait():
            raise RuntimeError('paired browser failed')
    finally:
        server.terminate()
        server.wait(timeout=5)
    prior = json.loads(args.reuse_tesseract.read_text(encoding='utf-8')) if args.reuse_tesseract else None
    prior_pages = {str(Path(row['image']).resolve()): row for row in prior['pages']} if prior else {}
    rows = []
    for xml, output in zip(xmls, outputs, strict=True):
        truth = gold(xml)
        row = {'image': output['image'], 'gold_chars': len(truth), 'gold': truth}
        for engine in ('kraken', 'tesseract'):
            if output[engine]:
                text = normalized(output[engine]['text'])
                row[engine] = {**output[engine], 'edits': distance(truth, text), 'text': text}
        if output['tesseract']:
            row['tesseract_blocks'] = output['tesseract'].get('blocks', [])
        if output['shared']:
            for engine in ('kraken', 'tesseract'):
                if output['shared'][engine]:
                    text = normalized(output['shared'][engine]['text'])
                    row[f'{engine}_lines'] = {'seconds': output['shared'][engine]['seconds'], 'edits': distance(truth, text), 'text': text}
        if output.get('layout'):
            value = output['layout']; value['text'] = normalized(value['text']); value['edits'] = distance(truth, value['text']); row['kraken_layout'] = value
        if prior:
            old = prior_pages.get(str(xml.with_suffix('.png').resolve()))
            if old is None:
                raise ValueError(f'reused receipt has no image: {xml.with_suffix(".png")}')
            key = 'tesseract_lines' if args.lane == 'shared' else 'tesseract'
            row[key] = old[key]
        rows.append(row)
    chars = sum(row['gold_chars'] for row in rows)
    engines = {}
    suffixes = ('', '_lines') if args.lane == 'both' else ('_lines',) if args.lane == 'shared' else ('',)
    for suffix in suffixes:
        for engine in ('kraken', 'tesseract'):
            key, label = f'{engine}{suffix}', f'{engine}_shared_lines' if suffix else engine
            seconds = sum(row[key]['seconds'] for row in rows)
            engines[label] = {'cer': sum(row[key]['edits'] for row in rows) / chars, 'seconds': seconds, 'pages_per_second': len(rows) / seconds}
    if args.lane == 'layout':
        seconds = sum(row['kraken_layout']['seconds'] for row in rows)
        engines = {'kraken_layout': {'cer': sum(row['kraken_layout']['edits'] for row in rows) / chars, 'seconds': seconds, 'pages_per_second': len(rows) / seconds}, 'tesseract': engines['tesseract']}
    result = {'protocol': {'benchmark': str(args.list), 'pages': len(rows), 'truth': truth_counts, 'metrics': ['cer', 'pages_per_second'], 'same_pages_and_pixels': True, 'mode': args.mode, 'model': 'current_best-48px-lstm-channel-int8', 'lane': args.lane, 'kraken_workers': args.kraken_workers, 'kraken_rounds': args.kraken_rounds, 'kraken_schedule': 'dynamic-pages', 'kraken_worker_pool': args.kraken_worker_pool, 'warmup': 'one page per persistent Kraken worker, unscored', 'timing': 'in-page OCR call only; input already decoded and rendered; pooled wall time amortized per page when workers > 1; median wall time when rounds > 1', 'comparison': 'same Chromium browser and page pixels', 'tesseract': {'data': args.tess_data, 'psm': args.tess_psm, 'output': args.tess_output, 'input': args.tess_input, 'workers': args.tess_workers, 'rounds': args.tess_rounds}, 'tesseract_receipt': str(args.reuse_tesseract) if args.reuse_tesseract else 'measured in this run'}, 'engines': engines, 'pages': rows}
    atomic_json(args.output, result)
    print(json.dumps(engines, indent=2))


if __name__ == '__main__':
    main()
