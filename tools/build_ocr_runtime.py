"""Build the native OCR command in the canonical Cargo target without machine paths."""
import os
from pathlib import Path
import subprocess


def main():
    owner = Path(__file__).resolve().parents[1]
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("CARGO_PROFILE_", "CARGO_TARGET_")) and key not in {
               "RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_INCREMENTAL", "CARGO_BUILD_RUSTFLAGS",
               "CARGO_BUILD_TARGET", "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER"}}
    home = Path.home()
    env["CARGO_BUILD_JOBS"] = "1"
    env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(f"--remap-path-prefix={source}={target}" for source, target in [
        (home, "/home"), (os.environ.get("RUSTUP_HOME", home / ".rustup"), "/rustup"),
        (os.environ.get("CARGO_HOME", home / ".cargo"), "/cargo"), (owner.parent, "/workspace")])
    subprocess.run(["cargo", "build", "--offline", "--locked", "--release", "-p", "legal-pdf-ocr",
                    "--features", "indra", "--bin", "legalocr"], cwd=owner, env=env, check=True)


if __name__ == "__main__":
    main()
