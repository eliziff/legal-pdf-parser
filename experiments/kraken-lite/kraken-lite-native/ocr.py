from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
import sys
import time
from pathlib import Path
from threading import Lock, local

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [
    str(ROOT / 'kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-4-turbo-lite/kraken_lite_runtime-0.1.0-py3-none-any.whl'),
]

from kraken_lite.fast_layout import LineBox, _merge_fragments, crop_line, order_line_boxes
from kraken_lite.geometry import Rectification, rectify_line
from kraken_lite.blla import BLLASegmenter
from kraken_lite.model_pack import ModelPack, OnnxSession
from kraken_lite.recognition import Codec, Recognizer

MODEL = ROOT / 'kraken-lite-student/student-model-packs/student-turbo-extreme-cuda'
BIG = ROOT / 'kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-2-quality/models/recognizer'
BIG_FP32 = ROOT / 'kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-1-fidelity/models/recognizer'
BROWSER_BIG = ROOT / 'kraken-lite-browser/dist/optimized-models-ort/lstm-channel-preprocessed.ort'
STOCK_BLLA = ROOT / 'kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-1-fidelity/models/blla'
TIERS = {
    'fidelity': {'model': BIG_FP32, 'blla': STOCK_BLLA, 'layout': 'blla', 'scale': 1.0},
    'quality': {'model': BROWSER_BIG, 'layout': 'tesseract', 'scale': 1.0},
    'balanced': {'model': MODEL, 'fallback_model': BROWSER_BIG, 'fallback_threshold': .70, 'layout': 'tesseract', 'scale': 1.0},
    'speed': {'model': MODEL, 'layout': 'tesseract', 'scale': 1.0},
}
_RAPID_DETECTORS = {}
_TESS_DLL = None
_TESS_LOCK = Lock()
_TESS_LOCAL = local()
_TESS_OCR_API = None
_TESS_OCR_DLL = None


def clean_model_text(text: str) -> str:
    return re.sub(r'[\u00ad\u00ac]\r?\n', '', text).replace('\u00ad', '').replace('\u00ac', '')


def detect_line_boxes(image: Image.Image) -> list[LineBox]:
    gray = np.asarray(image.convert('L'), dtype=np.uint8)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    height, width = gray.shape
    glyphs = [h for x, y, w, h, area in stats[1:] if area >= 3 and 3 <= h <= height * .05 and w <= width * .2]
    if not glyphs:
        return []
    glyph = float(np.median(glyphs))
    joined = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, round(glyph * 1.5)), 1)))
    _, _, stats, _ = cv2.connectedComponentsWithStats(joined, connectivity=8)
    boxes = [LineBox(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h, area in stats[1:]
             if area >= glyph * glyph and w >= glyph * 1.5 and h >= glyph * .45 and h <= glyph * 3 and w <= width * .98]
    boxes = _merge_fragments(boxes, glyph)
    xpad, ypad = max(4, round(glyph * .75)), max(2, round(glyph * .2))
    padded = [LineBox(max(0, b.left-xpad), max(0, b.top-ypad), min(width, b.right+xpad), min(height, b.bottom+ypad)) for b in boxes]
    typical = float(np.median([box.height for box in padded]))
    normalized = []
    for box in padded:
        if box.height < typical * .6:
            middle = (box.top + box.bottom) // 2
            half = round(typical / 2)
            normalized.append(LineBox(box.left, max(0, middle-half), box.right, min(height, middle+half)))
        elif box.height > typical * 1.3:
            middle = (box.top + box.bottom) // 2
            normalized.extend((LineBox(box.left, box.top, box.right, middle+2), LineBox(box.left, middle-2, box.right, box.bottom)))
        else:
            normalized.append(box)
    return normalized


def projection_line_boxes(image: Image.Image) -> list[LineBox]:
    gray = np.asarray(image.convert('L'), dtype=np.uint8)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    height, width = gray.shape
    rows = np.count_nonzero(ink, axis=1)

    def ranges(threshold, start=0, stop=height):
        active = rows[start:stop] >= threshold
        edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        return [(start + a, start + b) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)) if b-a >= 4]

    bands = ranges(max(4, width // 100))
    sizes = sorted(b-a for a, b in bands if b-a < 100)
    typical = sizes[len(sizes)//2] if sizes else 40
    refined = []
    for top, bottom in bands:
        refined.extend(ranges(max(4, width // 30), top, bottom) if bottom-top > typical*1.65 else [(top, bottom)])
    boxes = []
    for top, bottom in refined:
        if bottom-top < typical*.3:
            continue
        columns = np.flatnonzero(np.count_nonzero(ink[max(0, top-3):min(height, bottom+3)], axis=0))
        if columns.size:
            cuts = np.flatnonzero(np.diff(columns) > width*.08)
            starts = np.r_[0, cuts+1]
            stops = np.r_[cuts, len(columns)-1]
            for start, stop in zip(starts, stops):
                left, right = int(columns[start]), int(columns[stop])
                if right-left >= typical:
                    boxes.append(LineBox(max(0, left-12), max(0, top-6), min(width, right+13), min(height, bottom+6)))
    return boxes


def merge_text_boxes(boxes: list[LineBox]) -> list[LineBox]:
    if not boxes:
        return []
    height = float(np.median([box.height for box in boxes]))
    lines = []
    for box in sorted(boxes, key=lambda item: ((item.top + item.bottom) / 2, item.left)):
        match = next((index for index in range(len(lines)-1, -1, -1) if abs((lines[index].top + lines[index].bottom - box.top - box.bottom) / 2) <= height * .55
                      and box.left - lines[index].right <= height * 2.5), None)
        if match is not None:
            line = lines[match]
            lines[match] = LineBox(min(line.left, box.left), min(line.top, box.top), max(line.right, box.right), max(line.bottom, box.bottom))
        else:
            lines.append(LineBox(box.left, box.top, box.right, box.bottom))
    return order_line_boxes(lines)


def order_tesseract_boxes(boxes: list[LineBox], page_width: int, page_height: int) -> list[LineBox]:
    """Preserves Tesseract order, moving paired column footnotes to the end."""
    if len(boxes) < 16:
        return boxes

    def footer_start(lines):
        ordered = sorted(lines, key=lambda box: box.top)
        if len(ordered) < 8:
            return None
        body_height = float(np.median([box.height for box in ordered[:round(len(ordered) * .65)]]))
        for index in range(round(len(ordered) * .45), len(ordered) - 2):
            gap = ordered[index].top - ordered[index - 1].bottom
            tail_height = float(np.median([box.height for box in ordered[index:]]))
            if ordered[index].top > page_height * .55 and gap >= max(12, body_height * .75) and tail_height <= body_height * .92:
                return ordered[index].top
        return None

    left = [box for box in boxes if (box.left + box.right) / 2 < page_width * .48]
    right = [box for box in boxes if (box.left + box.right) / 2 > page_width * .52]
    if len(left) < 8 or len(right) < 8:
        return boxes
    left_start, right_start = footer_start(left), footer_start(right)
    if left_start is None or right_start is None or abs(left_start - right_start) > page_height * .08:
        return boxes

    def is_footer(box):
        center = (box.left + box.right) / 2
        return center < page_width * .48 and box.top >= left_start or center > page_width * .52 and box.top >= right_start

    body = [box for box in boxes if not is_footer(box)]
    footers = sorted((box for box in boxes if is_footer(box)), key=lambda box: (box.left > page_width / 2, box.top, box.left))
    return body + footers


def rapid_line_boxes(image: Image.Image, limit: int = 960) -> list[LineBox]:
    if limit not in _RAPID_DETECTORS:
        from rapidocr_onnxruntime import RapidOCR
        _RAPID_DETECTORS[limit] = RapidOCR(det_limit_side_len=limit, det_limit_type='max', det_thresh=.3,
                                           det_box_thresh=.5, det_unclip_ratio=1.6).text_det
    from rapidocr_onnxruntime.ch_ppocr_det.text_detect import DetPreProcess
    detector = _RAPID_DETECTORS[limit]
    source = cv2.cvtColor(np.asarray(image.convert('RGB')), cv2.COLOR_RGB2BGR)
    prepared = DetPreProcess(limit, 'max', detector.mean, detector.std)(source)
    prediction = detector.infer(prepared)[0]
    boxes, _ = detector.postprocess_op(prediction, source.shape[:2])
    boxes = detector.filter_tag_det_res(boxes, source.shape[:2])
    if boxes is None:
        return []
    raw = [LineBox(max(0, int(box[:, 0].min())-4), max(0, int(box[:, 1].min())-3),
                   min(image.width, int(box[:, 0].max())+5), min(image.height, int(box[:, 1].max())+4)) for box in boxes]
    return merge_text_boxes(raw)


def tesseract_line_boxes(image: Image.Image) -> list[LineBox]:
    global _TESS_DLL
    import ctypes
    with _TESS_LOCK:
        if _TESS_DLL is None:
            _TESS_DLL = ctypes.CDLL(r'C:\Program Files\Tesseract-OCR\libtesseract-5.dll')
            _TESS_DLL.TessBaseAPICreate.restype = ctypes.c_void_p
            _TESS_DLL.TessBaseAPIInitForAnalysePage.argtypes = [ctypes.c_void_p]
            _TESS_DLL.TessBaseAPIInitForAnalysePage.restype = None
            _TESS_DLL.TessBaseAPISetPageSegMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _TESS_DLL.TessBaseAPISetSourceResolution.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _TESS_DLL.TessBaseAPISetImage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
            _TESS_DLL.TessBaseAPIAnalyseLayout.argtypes = [ctypes.c_void_p]
            _TESS_DLL.TessBaseAPIAnalyseLayout.restype = ctypes.c_void_p
            _TESS_DLL.TessPageIteratorBoundingBox.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
            _TESS_DLL.TessPageIteratorNext.restype = ctypes.c_bool
            _TESS_DLL.TessPageIteratorNext.argtypes = [ctypes.c_void_p, ctypes.c_int]
            _TESS_DLL.TessPageIteratorBoundingBox.restype = ctypes.c_bool
            _TESS_DLL.TessPageIteratorDelete.argtypes = [ctypes.c_void_p]
            _TESS_DLL.TessBaseAPIClear.argtypes = [ctypes.c_void_p]
    api = getattr(_TESS_LOCAL, 'api', None)
    if api is None:
        api = _TESS_LOCAL.api = _TESS_DLL.TessBaseAPICreate()
        _TESS_DLL.TessBaseAPIInitForAnalysePage(api)
        _TESS_DLL.TessBaseAPISetPageSegMode(api, 3)
    gray = np.ascontiguousarray(image.convert('L'), dtype=np.uint8)
    _TESS_DLL.TessBaseAPISetImage(api, gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), image.width, image.height, 1, gray.strides[0])
    _TESS_DLL.TessBaseAPISetSourceResolution(api, 200)
    iterator = _TESS_DLL.TessBaseAPIAnalyseLayout(api)
    if not iterator:
        return []
    boxes = []
    left = ctypes.c_int(); top = ctypes.c_int(); right = ctypes.c_int(); bottom = ctypes.c_int()
    while True:
        if _TESS_DLL.TessPageIteratorBoundingBox(iterator, 2, ctypes.byref(left), ctypes.byref(top), ctypes.byref(right), ctypes.byref(bottom)):
            boxes.append(LineBox(left.value, top.value, right.value, bottom.value))
        if not _TESS_DLL.TessPageIteratorNext(iterator, 2):
            break
    _TESS_DLL.TessPageIteratorDelete(iterator)
    _TESS_DLL.TessBaseAPIClear(api)
    return [LineBox(max(0, box.left-10), max(0, box.top-6), min(image.width, box.right+11), min(image.height, box.bottom+7))
            for box in order_tesseract_boxes(boxes, image.width, image.height)]


def tesseract_text(image: Image.Image) -> str:
    """Runs a persistent in-process Tesseract LSTM session on already-decoded pixels."""
    global _TESS_OCR_API, _TESS_OCR_DLL
    import ctypes
    if _TESS_OCR_API is None:
        _TESS_OCR_DLL = ctypes.CDLL(r'C:\Program Files\Tesseract-OCR\libtesseract-5.dll')
        _TESS_OCR_DLL.TessBaseAPICreate.restype = ctypes.c_void_p
        _TESS_OCR_DLL.TessBaseAPIInit3.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        _TESS_OCR_DLL.TessBaseAPIInit3.restype = ctypes.c_int
        _TESS_OCR_DLL.TessBaseAPISetPageSegMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        _TESS_OCR_DLL.TessBaseAPISetImage.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _TESS_OCR_DLL.TessBaseAPIGetUTF8Text.argtypes = [ctypes.c_void_p]
        _TESS_OCR_DLL.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
        _TESS_OCR_DLL.TessDeleteText.argtypes = [ctypes.c_void_p]
        _TESS_OCR_DLL.TessBaseAPIClear.argtypes = [ctypes.c_void_p]
        _TESS_OCR_API = _TESS_OCR_DLL.TessBaseAPICreate()
        if _TESS_OCR_DLL.TessBaseAPIInit3(_TESS_OCR_API, rb'C:\Program Files\Tesseract-OCR\tessdata', b'eng'):
            raise RuntimeError('Tesseract eng initialization failed')
        _TESS_OCR_DLL.TessBaseAPISetPageSegMode(_TESS_OCR_API, 3)
    gray = np.ascontiguousarray(image.convert('L'), dtype=np.uint8)
    _TESS_OCR_DLL.TessBaseAPISetImage(_TESS_OCR_API, gray.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)), image.width, image.height, 1, gray.strides[0])
    pointer = _TESS_OCR_DLL.TessBaseAPIGetUTF8Text(_TESS_OCR_API)
    try:
        return ctypes.string_at(pointer).decode('utf-8')
    finally:
        _TESS_OCR_DLL.TessDeleteText(pointer)
        _TESS_OCR_DLL.TessBaseAPIClear(_TESS_OCR_API)


def squeeze(line: Rectification, scale: float) -> Rectification:
    return Rectification(
        line.image.resize((max(1, round(line.image.width * scale)), line.image.height), Image.Resampling.BILINEAR),
        line.source_x, line.source_y, line.baseline_row, line.source_length)


def load_recognizer(model: Path, threads: int) -> Recognizer:
    if model.is_dir():
        return Recognizer.from_pack(model, device='cpu', intra_threads=threads)
    template = ModelPack.load(BIG)
    manifest = deepcopy(template.manifest)
    manifest['id'] = 'browser-quality-int8-dynamic'
    manifest['model'].update(file=model.name, dynamic_batch=True, input='image', output='logits',
                             lengths_input='sequence_lengths', lengths_output='output_lengths')
    manifest['input'].update(height=48, padding=16)
    manifest['recognition'].update(cpu_batch_size=32, accelerated_batch_size=32, width_bucket=24)
    return Recognizer(OnnxSession(model, device='cpu', intra_threads=threads), manifest, Codec.from_pack(template))


class NativeOCR:
    def __init__(self, *, model: Path = MODEL, fallback_model: Path | None = None, blla: Path | None = None, threads: int = 0):
        self.recognizer = load_recognizer(model, threads)
        self.fallback = load_recognizer(fallback_model, threads) if fallback_model else None
        self.segmenter = BLLASegmenter.from_pack(blla, device='cpu', intra_threads=threads) if blla else None

    def recognize(self, pages: list[Image.Image], *, scale: float = .7, batch_size: int = 32, layout: str = 'auto', workers: int = 2, layout_workers: int = 2, fallback_threshold: float = 0, fallback_character_threshold: float = 0) -> tuple[list[str], float]:
        started = time.perf_counter()
        if layout == 'tesseract' and layout_workers > 1 and len(pages) > 1:
            with ThreadPoolExecutor(max_workers=min(layout_workers, len(pages))) as pool:
                prepared_boxes = list(pool.map(tesseract_line_boxes, pages))
        else:
            prepared_boxes = None
        groups = []
        for page_index, page in enumerate(pages):
            selected = layout
            if selected in {'blla', 'blla-fast'}:
                if not self.segmenter:
                    raise ValueError('blla layout requires a BLLA model pack')
                segmentation = self.segmenter.segment(page.convert('RGB'), batch_size=1)
                if selected == 'blla':
                    groups.append([rectify_line(page, line.baseline, line.boundary) for line in segmentation.lines])
                else:
                    groups.append([crop_line(page, LineBox(
                        max(0, int(min(x for x, _ in line.boundary))),
                        max(0, int(min(y for _, y in line.boundary))),
                        min(page.width, int(max(x for x, _ in line.boundary)) + 1),
                        min(page.height, int(max(y for _, y in line.boundary)) + 1),
                    )) for line in segmentation.lines if line.boundary])
                continue
            if selected == 'auto':
                gray = np.asarray(page.convert('L'), dtype=np.uint8)
                threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                selected = 'projection' if threshold > 160 else 'components'
            if selected == 'tesseract':
                boxes = prepared_boxes[page_index] if prepared_boxes is not None else tesseract_line_boxes(page)
            else:
                boxes = rapid_line_boxes(page) if selected == 'rapid' else projection_line_boxes(page) if selected == 'projection' else order_line_boxes(detect_line_boxes(page))
            if selected == 'projection':
                boxes.sort(key=lambda box: (box.top, box.left))
            groups.append([crop_line(page, box) for box in boxes])
        flat = [line for group in groups for line in group]
        primary_groups = [[squeeze(line, scale) for line in group] for group in groups] if scale != 1 else groups
        primary = [line for group in primary_groups for line in group]
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                recognized_groups = list(pool.map(lambda group: self.recognizer.recognize_many(group, batch_size=batch_size, width_bucket=24), primary_groups))
            recognized = [item for group in recognized_groups for item in group]
        else:
            recognized = self.recognizer.recognize_many(primary, batch_size=batch_size, width_bucket=24)
        if fallback_threshold > 0 or fallback_character_threshold > 0:
            uncertain = [index for index, item in enumerate(recognized) if item.text and (item.confidence < fallback_threshold or (fallback_character_threshold > 0 and min(character.confidence for character in item.characters) < fallback_character_threshold))]
            replacements = (self.fallback or self.recognizer).recognize_many([flat[index] for index in uncertain], batch_size=batch_size, width_bucket=24)
            for index, replacement in zip(uncertain, replacements):
                recognized[index] = replacement
        output, offset = [], 0
        for group in groups:
            output.append(clean_model_text('\n'.join(item.text for item in recognized[offset:offset + len(group)])))
            offset += len(group)
        return output, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser(description='Fast local OCR for page images')
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--tier', choices=TIERS, default='quality')
    parser.add_argument('--scale', type=float)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--layout-workers', type=int, default=2)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--fallback-model', type=Path)
    parser.add_argument('--fallback-threshold', type=float)
    parser.add_argument('--fallback-character-threshold', type=float, default=0)
    parser.add_argument('--blla', type=Path)
    parser.add_argument('--layout', choices=('auto', 'components', 'projection', 'rapid', 'tesseract', 'blla', 'blla-fast'))
    parser.add_argument('--jsonl', action='store_true')
    args = parser.parse_args()
    preset = TIERS[args.tier]
    for name in ('model', 'fallback_model', 'fallback_threshold', 'blla', 'layout', 'scale'):
        if getattr(args, name) is None:
            setattr(args, name, preset.get(name, 0 if name == 'fallback_threshold' else None))
    engine = NativeOCR(model=args.model, fallback_model=args.fallback_model, blla=args.blla, threads=args.threads)
    for start in range(0, len(args.images), 10):
        paths = args.images[start:start+10]
        pages = []
        for path in paths:
            with Image.open(path) as image:
                pages.append(image.copy())
        texts, seconds = engine.recognize(pages, scale=args.scale, batch_size=args.batch_size, layout=args.layout, workers=args.workers, layout_workers=args.layout_workers, fallback_threshold=args.fallback_threshold, fallback_character_threshold=args.fallback_character_threshold)
        for path, text in zip(paths, texts):
            if args.jsonl:
                print(json.dumps({'path': str(path), 'text': text}, ensure_ascii=False), flush=True)
            else:
                print(text, flush=True)
        print(f'{start+len(paths)}/{len(args.images)} pages in {seconds:.3f}s', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
