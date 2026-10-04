"""Three-way upstream delta application preserves local Inspector modifications."""
import pathlib
import subprocess
import sys

repo = pathlib.Path(__file__).resolve().parents[1]
upstream = sys.argv[1]
prefix = "vendor/pdf-inspector"
subtree = repo / prefix

def git(*args):
    return subprocess.check_output(["git", "-C", str(repo), *args])

try:
    if git("status", "--porcelain", "--", prefix).strip():
        raise RuntimeError("Vendored Inspector source must be clean before upstream sync")
    record = subtree / ".upstream-pdf-inspector"
    base = next(line[7:] for line in record.read_text().splitlines() if line.startswith("commit "))
    upstream = git("rev-parse", "--verify", upstream + "^{commit}").decode().strip()
    if base != upstream:
        # Guidance belongs to our patchset; preserve it without treating it as upstream source.
        patch = git("diff", "--binary", base, upstream, "--", ".", ":!AGENTS.md")
        if patch:
            subprocess.run(["git", "-C", str(repo), "apply", "--3way", "--directory=" + prefix],
                           input=patch, check=True)
        manifest = git("show", upstream + ":Cargo.toml").decode()
        version = next(line.split('"')[1] for line in manifest.splitlines() if line.startswith("version = "))
        record.write_text("commit " + upstream + "\nversion " + version + "\n")
except Exception as error:
    message = "PDF Inspector patchset sync stopped: " + str(error) + "\n"
    sys.stderr.write(message)
    raise SystemExit(1)
