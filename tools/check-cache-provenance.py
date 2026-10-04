"""Check stage ownership using a compiled build.rs; no Cargo or parser/model run."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/check-cache-provenance.py <build-script-build executable>")
    executable = Path(sys.argv[1]).resolve(strict=True)
    cases = [
        ("legal-pdf-language/src/docx.rs", (False, False, False, False)),
        ("vendor/pdf-inspector/src/lib.rs", (True, False, False, False)),
        ("vendor/pdf-inspector/src/bin/cli.rs", (False, False, False, False)),
        ("legal-pdf-extraction-processor/src/pdf.rs", (True, False, False, False)),
        ("legal-pdf-ocr/src/ocr.rs", (False, True, False, False)),
        ("rust/native/tesseract_layout.c", (False, True, False, False)),
        ("legal-pdf-structure/src/structure.rs", (False, False, True, False)),
        ("legal-pdf-support/src/projection.rs", (False, False, True, False)),
        ("legal-pdf-support/src/ppdoc.rs", (False, False, False, True)),
        ("legal-pdf-core/src/ocr_contract.rs", (True, True, True, False)),
        ("legal-pdf-core/src/model.rs", (True, False, True, True)),
    ]
    names = ("EXTRACTION", "OCR", "DOCUMENT", "LAYOUT")
    with tempfile.TemporaryDirectory(prefix="legalpdf-cache-provenance-") as directory:
        root = Path(directory)
        # Independently invented bytes, testing real filesystem ownership and hashing.
        for file, _ in cases:
            path = root / file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"original")
        environment = {key: value for key, value in os.environ.items() if not key.startswith("CARGO_FEATURE_")}
        environment.update(CARGO_MANIFEST_DIR=str(root), CARGO_FEATURE_PDF="1",
                           CARGO_FEATURE_OCR="1", CARGO_FEATURE_KRAKEN="1", CARGO_FEATURE_PPDOC_FULL="1")

        def hashes():
            output = subprocess.run([str(executable)], env=environment, capture_output=True,
                                    text=True, check=True, timeout=10).stdout
            values = dict(line.removeprefix("cargo:rustc-env=").split("=", 1)
                          for line in output.splitlines() if line.startswith("cargo:rustc-env="))
            return tuple(values[f"LEGAL_PDF_{name}_SHA256"] for name in names)

        original = hashes()
        for file, expected in cases:
            path = root / file
            path.write_bytes(b"changed")
            assert tuple(a != b for a, b in zip(original, hashes())) == expected, file
            path.write_bytes(b"original")
        environment["CARGO_FEATURE_LANGUAGE"] = "1"
        assert hashes() == original, "DOCX feature must not invalidate PDF stages"
    print(f"Passed {len(cases)} source mutations and DOCX feature isolation; scratch removed.")
    print(f"Build script: {executable}")


if __name__ == "__main__":
    main()
