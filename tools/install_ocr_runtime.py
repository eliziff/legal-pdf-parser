"""Install the shared Indra model pack and official Windows CPU/DirectML runtime."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import platform
import tarfile
import urllib.request
import zipfile

ORT_URL = "https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.22.0/microsoft.ml.onnxruntime.directml.1.22.0.nupkg"
ORT_SHA = "29f9872d786236b79aa83f94482f3a17c14297e4833768d6d0ed4883ee732e60"
DML_URL = "https://api.nuget.org/v3-flatcontainer/microsoft.ai.directml/1.15.4/microsoft.ai.directml.1.15.4.nupkg"
DML_SHA = "4e7cb7ddce8cf837a7a75dc029209b520ca0101470fcdf275c1f49736a3615b9"
MODELS_URL = "https://github.com/eliziff/legal-browser-ocr/releases/download/v0.2.0/legal-ocr-models.tar.gz"
MODELS_SHA = "b5a97370cf28e500870af634aaaf7d04b2249cbe0cdd6f4c1a4247f691949118"


def read_source(source):
    if source.startswith("https://"):
        with urllib.request.urlopen(source, timeout=120) as response:
            data = response.read(128 * 1024 * 1024 + 1)
        if len(data) > 128 * 1024 * 1024:
            raise ValueError("download exceeds 128 MiB")
        return data
    return Path(source).read_bytes()


def destination(root, relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative or ":" in relative or not path.parts:
        raise ValueError("asset path must stay inside its package")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError("asset path escapes its package")
    return target


def write_asset(path, data, force):
    if path.exists() and path.read_bytes() != data and not force:
        raise ValueError(f"refusing to replace {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def install_models(data, root, force=False):
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        files = {}
        total = 0
        for item in archive.getmembers():
            if item.isdir() and item.name in (".", "./"):
                continue
            destination(root, item.name)
            if item.isdir():
                continue
            if not item.isfile() or item.size > 64 * 1024 * 1024:
                raise ValueError("model archive contains a link or oversized asset")
            total += item.size
            if total > 128 * 1024 * 1024:
                raise ValueError("expanded model archive exceeds 128 MiB")
            name = str(PurePosixPath(item.name))
            if name in files:
                raise ValueError("duplicate model asset")
            files[name] = archive.extractfile(item).read()
    manifest = json.loads(files["manifest.json"])
    if manifest["recognizer"] != "indra" or manifest["input_height"] != 32:
        raise ValueError("expected a height-32 Indra model pack")
    assets = manifest["assets"]
    if set(files) != set(assets) | {"manifest.json"}:
        raise ValueError("model archive inventory does not match its manifest")
    for name, spec in assets.items():
        if len(files[name]) != spec["bytes"] or hashlib.sha256(files[name]).hexdigest() != spec["sha256"]:
            raise ValueError(f"model checksum mismatch: {name}")
    # Validate the complete package before making any changes.
    targets = [(destination(root, name), value) for name, value in files.items()]
    if not force and any(path.exists() and path.read_bytes() != value for path, value in targets):
        raise ValueError("model pack already contains different assets; pass --force")
    for path, value in targets:
        write_asset(path, value, force)


def install_windows_runtime(root, force=False):
    notices = []
    for url, expected, prefix, names in [
        (ORT_URL, ORT_SHA, "runtimes/win-x64/native/", ["onnxruntime.dll", "onnxruntime_providers_shared.dll"]),
        (DML_URL, DML_SHA, "bin/x64-win/", ["DirectML.dll"]),
    ]:
        data = read_source(url)
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("official runtime package checksum mismatch")
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for name in names:
                write_asset(root / name, archive.read(prefix + name), force)
            notices.append({"url": url, "sha256": expected, "notices": {
                name: archive.read(name).decode("utf-8-sig") for name in archive.namelist()
                if "license" in name.lower() or "thirdparty" in name.lower()
            }})
    write_asset(root / "notices.json", (json.dumps(notices, indent=2) + "\n").encode(), force)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=MODELS_URL)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "runtime/indra")
    parser.add_argument("--runtime-output", type=Path)
    parser.add_argument("--models-only", action="store_true", help="use an existing platform ONNX Runtime installation")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.models_only and (platform.system() != "Windows" or platform.machine().lower() not in ("amd64", "x86_64")):
        parser.error("automatic runtime installation currently supports Windows x64; use --models-only with your platform's CPU/CUDA ORT library")
    models = read_source(args.models)
    if args.models == MODELS_URL and hashlib.sha256(models).hexdigest() != MODELS_SHA:
        raise ValueError("released model archive checksum mismatch")
    install_models(models, args.output.resolve(), args.force)
    if not args.models_only:
        install_windows_runtime((args.runtime_output or args.output.parent / "onnx").resolve(), args.force)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
