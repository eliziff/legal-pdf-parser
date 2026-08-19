# PPDoc postprocess parity

This experiment executes the pinned Text-Fidelity Python predecessor and the
production Rust postprocessor over identical canonical JSON. It compares the
complete final region list and every line assignment as UTF-8 bytes.

```powershell
python check_parity.py `
  --text-fidelity-root "C:\path\to\Text-Fidelity-Project"
```

The deterministic targeted cases exercise every enabled production rule. The
seeded stress cases combine multi-page repeats, sequencing, overlaps, edge
zones, labels, scores, line counts, and text shapes. A mismatch exits nonzero
and reports the first structural delta and both output hashes.
