"""Fair cold-run benchmark for the Kraken fine-tune and native Tesseract.

Both engines receive the same already-rendered page images. Timing begins
before engine/worker startup and ends after every OCR result has been written.
The benchmark therefore includes Kraken session initialization and every
Tesseract process launch instead of hiding worker setup behind a warm cache.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


HERE = Path(__file__).resolve().parent
TIERS = ("quality", "balanced", "turbo", "extreme")


def benchmark_helpers():
    path = HERE / "kraken-lite-browser" / "benchmark.py"
    spec = importlib.util.spec_from_file_location("kraken_benchmark_helpers", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def page_paths(list_path: Path) -> tuple[list[Path], list[Path]]:
    xmls = [Path(line) for line in list_path.read_text(encoding="utf-8-sig").splitlines() if line]
    images = [path.with_suffix(".png") for path in xmls]
    for path in (*xmls, *images):
        if not path.is_file():
            raise FileNotFoundError(path)
    return xmls, images


def below_normal_flags() -> int:
    return subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 0


def set_below_normal_priority() -> None:
    if os.name != "nt":
        return
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel.SetPriorityClass.restype = ctypes.c_int
    if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x00004000):
        raise ctypes.WinError(ctypes.get_last_error())


def run_kraken(args, tier: str, output: Path) -> dict:
    command = [
        str(args.kraken_runner),
        "--list", str(args.list),
        "--kraken-model", str(args.model),
        "--kraken-codec", str(args.codec),
        "--onnx-runtime", str(args.runtime),
        "--kraken-tesseract-library", str(args.layout_library),
        "--kraken-tier", tier,
        "--kraken-layout", "tesseract",
        "--kraken-backend", args.kraken_backend,
        "--skip-warmup",
    ]
    if args.kraken_device is not None:
        command.extend(("--kraken-device", args.kraken_device))
    for option, value in (
        ("--kraken-batch-size", args.kraken_batch_size),
        ("--kraken-width-bucket", args.kraken_width_bucket),
        ("--kraken-input-height", args.kraken_input_height),
        ("--kraken-workers", args.kraken_workers),
    ):
        if value is not None:
            command.extend((option, str(value)))
    started = time.perf_counter()
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            creationflags=below_normal_flags(),
        )
        assert process.stdout
        completed = 0
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            completed += 1
            if completed % 25 == 0 or completed == args.pages:
                print(f"  {completed}/{args.pages}", flush=True)
        stderr = process.stderr.read() if process.stderr else ""
        process.wait()
    seconds = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(f"Kraken {tier} failed: {stderr.strip()}")
    if completed != args.pages:
        raise RuntimeError(f"Kraken {tier} returned {completed} pages; expected {args.pages}")
    return {"seconds": seconds, "pages_per_second": args.pages / seconds}


class NativeTesseract:
    """One persistent native Tesseract API per benchmark worker."""

    def __init__(self, executable: Path):
        root = executable.resolve().parent
        self._dll_directory = os.add_dll_directory(str(root))
        self._tesseract = ctypes.CDLL(str(root / "libtesseract-5.dll"))
        leptonica = next(root.glob("libleptonica-*.dll"), None)
        if leptonica is None:
            raise RuntimeError(f"Leptonica DLL not found beside {executable}")
        self._leptonica = ctypes.CDLL(str(leptonica))
        self._tessdata = os.fsencode(root / "tessdata")
        self._local = threading.local()
        self._apis: list[int] = []
        self._lock = threading.Lock()

        self._tesseract.TessBaseAPICreate.restype = ctypes.c_void_p
        self._tesseract.TessBaseAPIInit3.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                                      ctypes.c_char_p]
        self._tesseract.TessBaseAPISetPageSegMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._tesseract.TessBaseAPISetSourceResolution.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._tesseract.TessBaseAPISetImage2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._tesseract.TessBaseAPIRecognize.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self._tesseract.TessBaseAPIRecognize.restype = ctypes.c_int
        self._tesseract.TessBaseAPIGetUTF8Text.argtypes = [ctypes.c_void_p]
        self._tesseract.TessBaseAPIGetUTF8Text.restype = ctypes.c_void_p
        self._tesseract.TessDeleteText.argtypes = [ctypes.c_void_p]
        self._tesseract.TessBaseAPIClear.argtypes = [ctypes.c_void_p]
        self._tesseract.TessBaseAPIDelete.argtypes = [ctypes.c_void_p]
        self._leptonica.pixRead.argtypes = [ctypes.c_char_p]
        self._leptonica.pixRead.restype = ctypes.c_void_p
        self._leptonica.pixDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

    def _api(self) -> int:
        if getattr(self._local, "api", None):
            return self._local.api
        api = self._tesseract.TessBaseAPICreate()
        if not api or self._tesseract.TessBaseAPIInit3(api, self._tessdata, b"eng"):
            raise RuntimeError("native Tesseract English initialization failed")
        self._tesseract.TessBaseAPISetPageSegMode(api, 3)
        self._local.api = api
        with self._lock:
            self._apis.append(api)
        return api

    def page(self, image: Path) -> dict:
        started = time.perf_counter()
        pix = self._leptonica.pixRead(os.fsencode(image.resolve()))
        if not pix:
            raise RuntimeError(f"Tesseract could not decode {image}")
        pointer = ctypes.c_void_p(pix)
        try:
            api = self._api()
            self._tesseract.TessBaseAPISetImage2(api, pix)
            self._tesseract.TessBaseAPISetSourceResolution(api, 200)
            if self._tesseract.TessBaseAPIRecognize(api, None):
                raise RuntimeError(f"Tesseract recognition failed for {image}")
            text = self._tesseract.TessBaseAPIGetUTF8Text(api)
            if not text:
                raise RuntimeError(f"Tesseract returned no text buffer for {image}")
            try:
                decoded = ctypes.string_at(text).decode("utf-8", errors="replace")
            finally:
                self._tesseract.TessDeleteText(text)
            self._tesseract.TessBaseAPIClear(api)
            return {"image": str(image.resolve()), "text": decoded,
                    "process_seconds": time.perf_counter() - started}
        finally:
            self._leptonica.pixDestroy(ctypes.byref(pointer))

    def close(self) -> None:
        for api in self._apis:
            self._tesseract.TessBaseAPIDelete(api)
        self._dll_directory.close()


def run_tesseract(args, images: list[Path], output: Path) -> dict:
    os.environ["OMP_THREAD_LIMIT"] = "1"
    started = time.perf_counter()
    engine = NativeTesseract(args.tesseract)
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        with ThreadPoolExecutor(max_workers=args.tesseract_workers) as pool:
            pending = {pool.submit(engine.page, image): index
                       for index, image in enumerate(images)}
            for completed, future in enumerate(as_completed(pending), 1):
                row = future.result()
                row["index"] = pending[future]
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
                if completed % 25 == 0 or completed == args.pages:
                    print(f"  {completed}/{args.pages}", flush=True)
    engine.close()
    seconds = time.perf_counter() - started
    return {"seconds": seconds, "pages_per_second": args.pages / seconds,
            "persistent_api_sessions": args.tesseract_workers, "omp_threads_per_session": 1}


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return sorted(rows, key=lambda row: row.get("index", len(rows)))


def score(output: Path, xmls: list[Path], summary: dict) -> None:
    helpers = benchmark_helpers()
    truths = [helpers.gold(path) for path in xmls]
    characters = sum(map(len, truths))
    for name in summary["engines"]:
        rows = read_jsonl(output / f"{name}.jsonl")
        if len(rows) != len(truths):
            raise RuntimeError(f"{name} returned {len(rows)} pages; expected {len(truths)}")
        edits = sum(helpers.distance(truth, helpers.normalized(row["text"]))
                    for truth, row in zip(truths, rows))
        summary["engines"][name]["character_error_rate"] = edits / characters
    summary["status"] = "complete"


def main() -> None:
    set_below_normal_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--kraken-runner", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--codec", type=Path)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--layout-library", type=Path)
    parser.add_argument("--tiers", nargs="+", choices=TIERS, default=list(TIERS))
    parser.add_argument("--kraken-backend", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--kraken-device")
    parser.add_argument("--kraken-batch-size", type=int)
    parser.add_argument("--kraken-width-bucket", type=int)
    parser.add_argument("--kraken-input-height", type=int)
    parser.add_argument("--kraken-workers", type=int)
    parser.add_argument("--tesseract", type=Path)
    parser.add_argument("--tesseract-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--hardware")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recognize-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--tesseract-only", action="store_true")
    parser.add_argument("--skip-tesseract", action="store_true")
    args = parser.parse_args()
    xmls, images = page_paths(args.list)
    args.pages = len(images)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.score_only:
        summary = json.loads((args.output / "result.json").read_text(encoding="utf-8-sig"))
        score(args.output, xmls, summary)
        atomic_json(args.output / "result.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
        return
    required = (("tesseract", "hardware") if args.tesseract_only else
                ("kraken_runner", "model", "codec", "runtime", "layout_library",
                 "hardware"))
    if not args.skip_tesseract and not args.tesseract_only:
        required += ("tesseract",)
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required unless --score-only is used")

    version = None
    if not args.skip_tesseract:
        version = subprocess.run(
            [str(args.tesseract), "--version"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
            check=True, creationflags=below_normal_flags(),
        ).stdout.splitlines()[0]
    existing = args.output / "result.json"
    summary = json.loads(existing.read_text(encoding="utf-8-sig")) if args.tesseract_only else {
        "status": "recognized",
        "protocol": {
            "pages": len(images),
            "hardware": args.hardware,
            "system": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "timing": "cold process wall on identical already-rendered pixels; startup and output included",
            "normalization": "nfkc-collapse-not-soft-hyphen-v1",
            "tesseract": version,
        },
        "engines": {},
    }
    if not args.tesseract_only:
        for tier in args.tiers:
            print(f"Kraken {tier}: {len(images)} pages", flush=True)
            summary["engines"][tier] = run_kraken(args, tier, args.output / f"{tier}.jsonl")
            atomic_json(args.output / "result.json", summary)
    if not args.skip_tesseract:
        print(f"Tesseract: {len(images)} pages with {args.tesseract_workers} workers", flush=True)
        summary["engines"]["tesseract"] = run_tesseract(
            args, images, args.output / "tesseract.jsonl"
        )
    if not args.recognize_only:
        score(args.output, xmls, summary)
    atomic_json(args.output / "result.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
