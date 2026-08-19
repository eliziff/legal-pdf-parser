# Python reference fidelity check

This experiment compares source extraction and deterministic structure replay
between a frozen Python reference checkout and the Rust engine. It exchanges
one bounded JSON value per phase and never creates parser publication bundles.

```powershell
python experiments\python-reference\fidelity.py manifest.json `
  --oracle-root C:\path\to\frozen-reference `
  --rust-binary target\release\legalpdf.exe `
  --output fidelity-report.json
```

The manifest uses `legalpdf.fidelity-manifest.v1` and contains a `cases` array
of `{ "id", "pdf", "sha256" }` objects. Paths are relative to the manifest.
The report is rewritten atomically after every case so interrupted runs retain
usable results.
