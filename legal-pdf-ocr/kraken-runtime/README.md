# Kraken Lite runtime

The maintained inference package provides BLLA segmentation, original-page line
rectification and compiled Kraken recognition. The compiler creates canonical
BLLA and recognizer model packs; it requires the optional compiler dependencies.
Runtime dependencies are NumPy, Pillow, OpenCV and one selected ONNX Runtime
provider. Full Kraken is used only by the model compiler.

The CLI supports segment, ocr, serve, devices, compile-blla and
compile-recognizer. Production uses the persistent JSONL serve operation with
configured BLLA and recognizer packs. No evaluator or test suite is included in
this restored production package.

## Attribution

The BLLA inference design is based on the Kraken OCR engine, Copyright Benjamin
Kiessling and contributors, licensed under the Apache License 2.0.
Kraken: https://github.com/mittagessen/kraken

The original delivery identifies this implementation as new NumPy, Pillow and
OpenCV code, not a verbatim copy of Kraken source. It reproduces Kraken's
heatmap interpretation, ridge and baseline recovery, polygon construction,
region reading order, curved-line rectification and CTC codec decoding. The
compiler uses an installed Kraken distribution to export supported model graphs.
The package's existing AGPL-3.0-only metadata remains unchanged.

## Maintained production owner

This package is maintained once in legal-pdf-ocr/kraken-runtime. Its source was
restored from the existing stock-BLLA runtime delivery, source commit
`35d62b88318957e683d07b8012f69c28111e3036`. The delivery archives are historical
provenance; production imports the wheel built from this directory.

From the parser root, run `python legal-pdf-ocr/kraken-runtime/build-wheel.py`
with setuptools >=68, wheel and pip already installed. This offline command
builds into the canonical build directory, atomically replaces
`runtime/kraken/runtime.whl`, and removes its own staging files. Fixed ZIP
timestamps make repeated builds reproducible with the same Python/build toolchain.
It does not install dependencies, load models, or execute OCR. Runtime numerical
dependencies and the selected ONNX provider must already be installed in the
configured Python environment.

The JSONL `recognize_lines_batch` operation accepts `images` and corresponding
`pages: [{lines: [{id, baseline, boundary}]}]`. Coordinates refer to each full
original image. It validates geometry and recognizes only these supplied lines,
using the same rectification and character-coordinate mapping as ordinary OCR.
The Rust owner supplies uncovered-ink candidates, estimates horizontal baselines,
and retains only recognized fragments meeting its existing confidence threshold.

Recovery scope: the existing uncovered-ink detector requires at least three
segmented lines and uses horizontal line-height/ink components. It does not
recover an entirely unsegmented page or promise vertical/curved-line recovery.
Recovered output retains `recovered: true`: its boundary comes from uncovered
ink and its baseline is estimated, not a BLLA prediction. Normal provider lines
retain `recovered: false`. All coordinates remain on the original page.

The subprocess starts Python with safe-path mode and puts the configured wheel
first on PYTHONPATH; source from the current directory cannot shadow that wheel.
Startup checks the supplied-line operation, so an old wheel fails immediately
instead of silently omitting fragment recovery.
