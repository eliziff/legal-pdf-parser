"""Build the maintained runtime into the parser's canonical deployment directory."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def main() -> None:
    source = Path(__file__).resolve().parent
    parser = source.parents[1]
    build = parser / "build" / "kraken-runtime"
    build.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    # Stable ZIP timestamps. Dependencies are deliberately not downloaded here.
    environment["SOURCE_DATE_EPOCH"] = "1786233600"
    environment["PYTHONHASHSEED"] = "0"
    with tempfile.TemporaryDirectory(prefix="wheel-", dir=build) as temporary:
        staging = Path(temporary)
        package = staging / "package"
        package.mkdir()
        for name in ("pyproject.toml", "README.md"):
            shutil.copy2(source / name, package / name)
        shutil.copytree(source / "src", package / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        wheels = staging / "wheels"
        wheels.mkdir()
        subprocess.run([sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
                        "--no-index", "--wheel-dir", str(wheels), str(package)],
                       check=True, env=environment)
        artifacts = list(wheels.glob("kraken_lite_runtime-*.whl"))
        if len(artifacts) != 1:
            raise RuntimeError("Expected exactly one runtime wheel")
        destination = parser / "runtime" / "kraken" / "runtime.whl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        artifacts[0].replace(destination)
        print(destination)


if __name__ == "__main__":
    main()
