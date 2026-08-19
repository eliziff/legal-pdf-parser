"""Build the browser-proven Tesseract layout core as a small native DLL."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def run(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True, env=dict(os.environ))


def find_cmake(configured: Path | None) -> str:
    if configured is not None:
        return str(configured)
    if found := shutil.which("cmake"):
        return found
    visual_studio = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio"
    candidates = visual_studio.glob(
        "2022/*/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe"
    )
    if found := next(candidates, None):
        return str(found)
    raise SystemExit("cmake was not found; pass --cmake <path>")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="local tesseract.js-core source tree used by the browser build",
    )
    parser.add_argument("--tesseract-source", type=Path)
    parser.add_argument("--leptonica-source", type=Path)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--cmake", type=Path)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    tesseract = (args.tesseract_source or source / "third_party/tesseract").resolve()
    leptonica = (args.leptonica_source or source / "third_party/leptonica").resolve()
    build = args.build.resolve()
    prefix = build / "dep"
    cmake = find_cmake(args.cmake)
    common = [
        cmake,
        "-G",
        "Visual Studio 17 2022",
        "-A",
        "x64",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    ]

    leptonica_library = next((prefix / "lib").glob("leptonica-*.lib"), None)
    if leptonica_library is None:
        run(
            common
            + [
                "-S",
                str(leptonica),
                "-B",
                str(build / "leptonica"),
                "-DBUILD_PROG=OFF",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DSW_BUILD=OFF",
                "-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
            ]
        )
        run(
            [
                cmake,
                "--build",
                str(build / "leptonica"),
                "--config",
                "Release",
                "--target",
                "install",
                "--parallel",
            ]
        )
        leptonica_library = next((prefix / "lib").glob("leptonica-*.lib"))

    tesseract_library = next((prefix / "lib").glob("tesseract*.lib"), None)
    if tesseract_library is None:
        run(
            common
            + [
                "-S",
                str(tesseract),
                "-B",
                str(build / "tesseract"),
                "-DBUILD_TRAINING_TOOLS=OFF",
                "-DBUILD_TESTS=OFF",
                "-DGRAPHICS_DISABLED=ON",
                "-DDISABLED_LEGACY_ENGINE=ON",
                "-DOPENMP_BUILD=OFF",
                "-DBUILD_SHARED_LIBS=OFF",
                "-DSW_BUILD=OFF",
                "-DENABLE_LTO=ON",
                "-DCMAKE_CXX_FLAGS=/DTESSERACT_DISABLE_DEBUG_FONTS",
                f"-DLeptonica_DIR={prefix / 'lib/cmake/leptonica'}",
                f"-DCMAKE_INSTALL_PREFIX={prefix}",
            ]
        )
        run(
            [
                cmake,
                "--build",
                str(build / "tesseract"),
                "--config",
                "Release",
                "--target",
                "install",
                "--parallel",
            ]
        )
        tesseract_library = next((prefix / "lib").glob("tesseract*.lib"))

    wrapper = root / "rust/native/tesseract-layout"
    run(
        common
        + [
            "-S",
            str(wrapper),
            "-B",
            str(build / "wrapper"),
            f"-DTESSERACT_SOURCE={tesseract}",
            f"-DTESSERACT_BUILD={build / 'tesseract'}",
            f"-DTESSERACT_LIBRARY={tesseract_library}",
            f"-DLEPTONICA_SOURCE={leptonica}",
            f"-DLEPTONICA_BUILD={build / 'leptonica'}",
            f"-DLEPTONICA_LIBRARY={leptonica_library}",
        ]
    )
    run(
        [
            cmake,
            "--build",
            str(build / "wrapper"),
            "--config",
            "Release",
            "--parallel",
        ]
    )
    artifact = build / "wrapper/Release/legalpdf_tesseract_layout.dll"
    shutil.copy2(tesseract / "LICENSE", artifact.parent / "TESSERACT-LICENSE.txt")
    shutil.copy2(
        leptonica / "leptonica-license.txt",
        artifact.parent / "LEPTONICA-LICENSE.txt",
    )
    print(f"artifact={artifact} bytes={artifact.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
