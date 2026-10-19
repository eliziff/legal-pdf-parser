"""Package a built Windows OCR executable with the shared models and native runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile
from install_ocr_runtime import destination


def command(owner, *args):
    return subprocess.run(args, cwd=owner, check=True, stdout=subprocess.PIPE, text=True, encoding="utf-8").stdout


def crate_notices(owner):
    tree = command(owner, "cargo", "tree", "--offline", "-p", "legal-pdf-ocr", "--features", "indra",
                   "--target", "x86_64-pc-windows-msvc", "--edges", "normal", "--prefix", "none", "--format", "{p}")
    used = {match.groups() for line in tree.splitlines() if (match := re.match(r"([\w-]+) v([\w.+-]+)", line))}
    metadata = json.loads(command(owner, "cargo", "metadata", "--offline", "--format-version", "1", "--features", "legal-pdf-ocr/indra"))
    notices, first_party = [], []
    for package in metadata["packages"]:
        if (package["name"], package["version"]) not in used:
            continue
        folder = Path(package["manifest_path"]).parent
        if not package["source"]:
            repository = Path(command(folder, "git", "rev-parse", "--show-toplevel").strip())
            relative = folder.relative_to(repository).as_posix()
            if command(repository, "git", "status", "--porcelain", "--", relative).strip():
                raise ValueError(f"uncommitted native dependency: {package['name']}")
            first_party.append({"name": package["name"], "version": package["version"],
                "repository": command(repository, "git", "remote", "get-url", "origin").strip(),
                "revision": command(repository, "git", "rev-parse", "HEAD").strip(), "directory": relative})
            continue
        sources = {p.relative_to(folder).as_posix(): p.read_text(encoding="utf-8")
                   for p in folder.rglob("*") if p.is_file() and
                   p.name.upper().startswith(("LICENSE", "LICENCE", "NOTICE", "COPYING"))}
        if package.get("license_file"):
            path = Path(package["license_file"])
            sources[path.name] = path.read_text(encoding="utf-8")
        for parent in [folder, *list(folder.parents)[:3]]:
            if sources:
                break
            files = sorted({p for pattern in ("LICENSE*", "NOTICE*", "COPYING*") for p in parent.glob(pattern) if p.is_file()})
            if files:
                sources = {p.name: p.read_text(encoding="utf-8") for p in files}
                break
        notices.append({"name": package["name"], "version": package["version"],
                        "source": package["source"], "license": package["license"], "notices": sources})
    return notices, first_party


def main():
    owner = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=owner / "runtime/indra")
    parser.add_argument("--runtime", type=Path, default=owner / "runtime/onnx")
    parser.add_argument("--output", type=Path, default=owner / "dist/legal-ocr-runtime-windows-x64.zip")
    args = parser.parse_args()
    if command(owner, "git", "status", "--porcelain").strip():
        raise ValueError("commit the validated native runtime source before packaging")
    binary = args.binary.read_bytes()
    if any(prefix.encode() in binary for prefix in {str(Path.home()), Path.home().as_posix()}):
        raise ValueError("binary contains machine paths; rebuild with tools/build_ocr_runtime.py")
    manifest = json.loads((args.models / "manifest.json").read_text(encoding="utf-8"))
    files = {"legalocr.exe": binary, "runtime/indra/manifest.json": (args.models / "manifest.json").read_bytes()}
    for name, spec in manifest["assets"].items():
        data = destination(args.models, name).read_bytes()
        if len(data) != spec["bytes"] or hashlib.sha256(data).hexdigest() != spec["sha256"]:
            raise ValueError(f"model asset changed: {name}")
        files["runtime/indra/" + name] = data
    for name in ["orientation.onnx", "orientation.NOTICE.md"]:
        if (args.models / name).is_file():
            files["runtime/indra/" + name] = (args.models / name).read_bytes()
    for name in ["onnxruntime.dll", "onnxruntime_providers_shared.dll", "DirectML.dll"]:
        files["runtime/onnx/" + name] = (args.runtime / name).read_bytes()
    runtime_notices = args.runtime / "notices.json"
    if not runtime_notices.exists():
        runtime_notices = args.runtime / "source.json"
    runtime = runtime_notices.read_bytes()
    json.loads(runtime)
    files["runtime/onnx/notices.json"] = runtime
    notices, sources = crate_notices(owner)
    sysroot = Path(command(owner, "rustc", "--print", "sysroot").strip())
    rust_notices = (sysroot / "share/doc/rust/COPYRIGHT-library.html").read_text(encoding="utf-8")
    details = {"version": manifest["version"], "source": "https://github.com/eliziff/legal-pdf-parser",
               "revision": command(owner, "git", "rev-parse", "HEAD").strip(),
               "files": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()} for name, data in files.items()},
               "native_runtime_notices": "runtime/onnx/notices.json", "rust_dependencies": notices,
               "rust_library_notices": rust_notices, "first_party_sources": sources,
               "application_license": (owner / "LICENSE").read_text(encoding="utf-8")}
    files["runtime-manifest.json"] = (json.dumps(details, indent=2) + "\n").encode()
    files["README.txt"] = ("Run legalocr INPUT.pdf --output recognized.json.\n"
        "Quality tiers: --tier quality|balanced|fast|turbo|extreme (Balanced is default).\n"
        "CPU/GPU and worker counts are selected on this machine; use --backend cpu|directml to override.\n"
        "The first automatic selection has a startup cost; repeated documents share calibration in an embedding process.\n"
        "Point LEGALPDF_ENGINE_ROOT at this folder to use these assets in a native parser host.\n"
        "See runtime/indra/manifest.json for the original Indra model terms, use restrictions and notices.\n"
        "See runtime-manifest.json for source identity, checksums and third-party notices.\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".zip.partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(files.items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, data)
        partial.replace(args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
