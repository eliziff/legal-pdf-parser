import argparse
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from ocr import NativeOCR, tesseract_text
from freeze_benchmark_splits import OUTPUT as SPLITS, validate_benchmark_paths

spec = importlib.util.spec_from_file_location('browser_benchmark', ROOT / 'kraken-lite-browser/benchmark.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--scale', type=float, default=.7)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--page-batch', type=int, default=10)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--layout-workers', type=int, default=2)
    parser.add_argument('--model', type=Path, default=Path('kraken-lite-student/student-model-packs/student-turbo-extreme-cuda'))
    parser.add_argument('--fallback-model', type=Path)
    parser.add_argument('--fallback-threshold', type=float, default=0)
    parser.add_argument('--fallback-character-threshold', type=float, default=0)
    parser.add_argument('--blla', type=Path)
    parser.add_argument('--layout', choices=('auto', 'components', 'projection', 'rapid', 'tesseract', 'blla', 'blla-fast'), default='auto')
    parser.add_argument('--list', type=Path, default=SPLITS / 'benchmark-153.lst')
    parser.add_argument('--output', type=Path, default=HERE / 'benchmark-results.json')
    parser.add_argument('--reuse-tesseract', type=Path)
    parser.add_argument('--checkpoint', action='store_true', help='resume long runs; omitted for timed probes')
    args = parser.parse_args()
    xmls = [Path(line) for line in args.list.read_text(encoding='utf-8-sig').splitlines() if line]
    xmls = xmls[args.offset:]
    if args.limit:
        xmls = xmls[:args.limit]
    truth_counts = validate_benchmark_paths(xmls)
    images = [path.with_suffix('.png') for path in xmls]
    checkpoint_path = args.output.with_suffix(args.output.suffix + '.partial.json')
    protocol_key = {'offset': args.offset, 'limit': args.limit, 'scale': args.scale, 'batch_size': args.batch_size, 'page_batch': args.page_batch, 'threads': args.threads, 'workers': args.workers, 'layout_workers': args.layout_workers, 'layout': args.layout, 'model': str(args.model), 'fallback_model': str(args.fallback_model) if args.fallback_model else None, 'fallback_threshold': args.fallback_threshold, 'fallback_character_threshold': args.fallback_character_threshold, 'blla': str(args.blla) if args.blla else None, 'normalization': 'nfkc-collapse-not-soft-hyphen-v1'}
    checkpoint = json.loads(checkpoint_path.read_text(encoding='utf-8')) if args.checkpoint and checkpoint_path.exists() else {}
    if checkpoint.get('protocol') != protocol_key: checkpoint = {'protocol': protocol_key, 'native_texts': [], 'native_seconds': 0.0, 'rows': [], 'tesseract_seconds': 0.0}

    def save_checkpoint():
        if args.checkpoint:
            atomic_json(checkpoint_path, checkpoint)
    print('initializing native model', flush=True)
    engine = NativeOCR(model=args.model, fallback_model=args.fallback_model, blla=args.blla, threads=args.threads)
    print('native model ready; warming engines', flush=True)
    if images:
        with Image.open(images[0]) as image:
            warm = image.copy()
        engine.recognize([warm], scale=args.scale, batch_size=args.batch_size, layout=args.layout, workers=args.workers, layout_workers=args.layout_workers, fallback_threshold=args.fallback_threshold, fallback_character_threshold=args.fallback_character_threshold)
        if not args.reuse_tesseract:
            tesseract_text(warm)
    print('warmup complete', flush=True)
    texts, native_seconds = checkpoint['native_texts'], checkpoint['native_seconds']
    for start in range(len(texts), len(images), args.page_batch):
        pages = []
        for path in images[start:start + args.page_batch]:
            with Image.open(path) as image:
                pages.append(image.copy())
        chunk_texts, elapsed = engine.recognize(pages, scale=args.scale, batch_size=args.batch_size, layout=args.layout, workers=args.workers, layout_workers=args.layout_workers, fallback_threshold=args.fallback_threshold, fallback_character_threshold=args.fallback_character_threshold)
        texts.extend(chunk_texts)
        native_seconds += elapsed
        checkpoint.update(native_texts=texts, native_seconds=native_seconds)
        save_checkpoint()
        print(f'native {len(texts)}/{len(images)} {elapsed:.3f}s', flush=True)

    reused = json.loads(args.reuse_tesseract.read_text(encoding='utf-8')) if args.reuse_tesseract else None
    reused_pages = {page['image']: page for page in reused['pages']} if reused else {}
    rows, tess_seconds = checkpoint['rows'], checkpoint['tesseract_seconds']
    for index, (xml_path, image, text) in enumerate(zip(xmls[len(rows):], images[len(rows):], texts[len(rows):]), len(rows) + 1):
        truth, native_text = shared.gold(xml_path), shared.normalized(text)
        if reused:
            key = str(image.resolve())
            prior = reused_pages.get(key)
            if prior is None:
                raise ValueError(f'reused receipt has no unambiguous image identity: {key}')
            tess_text = prior.get('tesseract_text') or prior.get('tesseract', {}).get('text', '')
            tess_edits = shared.distance(truth, shared.normalized(tess_text)) if tess_text else prior.get('tesseract_edits', prior['tesseract']['edits'])
            tess_seconds += prior.get('tesseract', {}).get('seconds', reused['engines']['tesseract']['seconds'] / len(reused['pages']))
        else:
            with Image.open(image) as source:
                page = source.copy()
            started = time.perf_counter()
            raw_tess_text = tesseract_text(page)
            elapsed = time.perf_counter() - started
            tess_seconds += elapsed
            tess_text = shared.normalized(raw_tess_text)
            tess_edits = shared.distance(truth, tess_text)
            print(f'tesseract {index}/{len(images)} {elapsed:.3f}s', flush=True)
        rows.append({'image': str(image.resolve()), 'gold_chars': len(truth), 'native_edits': shared.distance(truth, native_text), 'tesseract_edits': tess_edits, 'gold': truth, 'native_text': native_text, 'tesseract_text': tess_text})
        checkpoint.update(rows=rows, tesseract_seconds=tess_seconds)
        save_checkpoint()
    chars = sum(row['gold_chars'] for row in rows)
    result = {
        'protocol': {'benchmark': str(args.list), 'pages': len(rows), 'truth': truth_counts, 'metrics': ['cer', 'pages_per_second'], 'same_pages_and_pixels': True, 'offset': args.offset, 'scale': args.scale, 'batch_size': args.batch_size, 'page_batch': args.page_batch, 'threads': args.threads, 'workers': args.workers, 'layout_workers': args.layout_workers, 'layout': args.layout, 'model': str(args.model), 'fallback_model': protocol_key['fallback_model'], 'fallback_threshold': args.fallback_threshold, 'fallback_character_threshold': args.fallback_character_threshold, 'blla': protocol_key['blla'], 'normalization': protocol_key['normalization'], 'warmup': 'first page once per persistent engine, unscored', 'timing': 'in-process OCR on already-decoded pixels'},
        'engines': {
            'native': {'cer': sum(row['native_edits'] for row in rows) / chars, 'seconds': native_seconds, 'pages_per_second': len(rows) / native_seconds},
            'tesseract': {'cer': sum(row['tesseract_edits'] for row in rows) / chars, 'seconds': tess_seconds, 'pages_per_second': len(rows) / tess_seconds},
        },
        'pages': rows,
    }
    atomic_json(args.output, result)
    if args.checkpoint:
        checkpoint_path.unlink(missing_ok=True)
    print(json.dumps(result['engines'], indent=2), flush=True)


if __name__ == '__main__':
    main()
