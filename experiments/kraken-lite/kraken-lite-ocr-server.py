from __future__ import annotations

import io
import base64
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import fitz
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "kraken-lite-runtime-work/experiments/kraken-lite-runtime/src"))
sys.path.insert(0, str(ROOT / "kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-4-turbo-lite/kraken_lite_runtime-0.1.0-py3-none-any.whl"))
sys.path.insert(0, str(ROOT / "kraken-lite-native"))

from kraken_lite.blla import BLLASegmenter
from ocr import NativeOCR

BLLA = ROOT / "kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-3-fast/models/blla"
RECOGNIZER = ROOT / "kraken-lite-student/student-model-packs/student-turbo-extreme-cuda"
BIG_RECOGNIZER = ROOT / "kraken-lite-runtime-0.1.0-four-tier-deliverables/tier-2-quality/models/recognizer"
LOG = ROOT / "kraken-lite-ocr-server.log"
MAX_UPLOAD = 250 * 1024 * 1024


def log(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def load_pages(payload: bytes, content_type: str) -> list[Image.Image]:
    if content_type.startswith("application/pdf") or payload[:4] == b"%PDF":
        document = fitz.open(stream=payload, filetype="pdf")
        try:
            return [
                Image.open(io.BytesIO(page.get_pixmap(dpi=200, alpha=False).tobytes("png"))).copy()
                for page in document
            ]
        finally:
            document.close()
    with Image.open(io.BytesIO(payload)) as image:
        return [image.copy()]


def page_preview(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, "JPEG", quality=72, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def recognize_pages(pages: list[Image.Image], mode: str) -> tuple[list[str], int]:
    if mode in {"quality", "balanced", "fidelity", "blla-fast"}:
        engine = big_engine(mode in {"fidelity", "blla-fast"})
    else:
        engine = SMALL_ENGINE
        if mode == "robust" and engine.fallback is None:
            engine.fallback = big_engine().recognizer
    scale, layout, fallback = {
        "fidelity": (1.0, "blla", 0.0), "blla-fast": (1.0, "blla-fast", 0.0),
        "quality": (1.0, "tesseract", 0.0),
        "balanced": (0.65, "tesseract", 0.0),
        "robust": (1.0, "auto", 0.8), "fast": (0.85, "auto", 0.0),
        "turbo": (0.7, "auto", 0.0),
    }.get(mode, (0.7, "auto", 0.0))
    texts, _ = engine.recognize(pages, scale=scale, batch_size=16, layout=layout, fallback_threshold=fallback)
    return texts, sum(len(text.splitlines()) for text in texts)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, value: object) -> None:
        body = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.end_headers()

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/kraken-lite-ocr.html"}:
            body = (ROOT / "kraken-lite-ocr.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/health":
            self.send_json(200, {"ok": True})
        else:
            self.send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        started = time.perf_counter()
        route = urlparse(self.path)
        if route.path not in {"/ocr", "/preview"}:
            self.send_json(404, {"error": "Not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size < 1 or size > MAX_UPLOAD:
                raise ValueError("File must be between 1 byte and 250 MB")
            pages = load_pages(self.rfile.read(size), self.headers.get("Content-Type", ""))
            if route.path == "/preview":
                buffer = io.BytesIO()
                pages[0].convert("RGB").save(buffer, "JPEG", quality=82, optimize=True)
                body = buffer.getvalue()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
                return
            query = parse_qs(route.query)
            mode = query.get("mode", ["blla" if query.get("segment", ["1"])[0] != "0" else "line"])[0]
            if query.get("stream", ["0"])[0] == "1":
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Connection", "close")
                self.end_headers()
                total_lines = 0
                for index, page in enumerate(pages, 1):
                    texts, lines = recognize_pages([page], mode)
                    total_lines += lines
                    start_event = {
                        "type": "page_start",
                        "page": index,
                        "pages": len(pages),
                        "lines": lines,
                        "image": page_preview(page),
                    }
                    self.wfile.write((json.dumps(start_event) + "\n").encode("utf-8"))
                    for line_number, text in enumerate(texts[0].splitlines(), 1):
                        line_event = {"type": "line", "page": index, "line": line_number, "text": text}
                        self.wfile.write((json.dumps(line_event, ensure_ascii=False) + "\n").encode("utf-8"))
                    end_event = {"type": "page_done", "page": index, "pages": len(pages), "lines": lines}
                    self.wfile.write((json.dumps(end_event) + "\n").encode("utf-8"))
                    self.wfile.flush()
                elapsed = time.perf_counter() - started
                done = {"type": "done", "pages": len(pages), "lines": total_lines, "seconds": elapsed, "pagesPerSecond": len(pages) / elapsed, "mode": mode}
                self.wfile.write((json.dumps(done) + "\n").encode("utf-8"))
                self.wfile.flush()
                self.close_connection = True
                log(f"ocr-stream pages={len(pages)} lines={total_lines} mode={mode} seconds={elapsed:.3f}")
                return
            texts, lines = recognize_pages(pages, mode)
            elapsed = time.perf_counter() - started
            self.send_json(200, {
                "text": "\n\n".join(texts),
                "pages": len(pages),
                "lines": lines,
                "seconds": elapsed,
                "pagesPerSecond": len(pages) / elapsed,
                "mode": mode,
            })
            log(f"ocr pages={len(pages)} lines={lines} mode={mode} seconds={elapsed:.3f}")
        except Exception as exc:
            log(f"error {type(exc).__name__}: {exc}")
            self.send_json(400, {"error": str(exc)})

    def log_message(self, format: str, *args: object) -> None:
        return


log("loading models")
SMALL_ENGINE = NativeOCR(model=RECOGNIZER)
BIG_ENGINE = NativeOCR(model=BIG_RECOGNIZER, threads=8)


def big_engine(with_blla: bool = False) -> NativeOCR:
    if with_blla and BIG_ENGINE.segmenter is None:
        BIG_ENGINE.segmenter = BLLASegmenter.from_pack(BLLA, device="cpu", intra_threads=0)
    return BIG_ENGINE
log("ready http://127.0.0.1:8766")
ThreadingHTTPServer(("127.0.0.1", 8766), Handler).serve_forever()
