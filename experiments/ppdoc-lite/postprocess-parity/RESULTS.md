# PPDoc postprocess parity results

The production Rust port is byte-for-byte identical to the pinned
Text-Fidelity Python predecessor on the deterministic differential corpus.

- Text-Fidelity revision: `d8b25257687b3b9aad644dec42cca966b45675ff`
- Seed: `20260814`
- Cases: 206 (6 targeted, 200 deterministic stress)
- Pages: 726
- Input regions: 9,179
- Line assignments: 14,275
- Canonical input SHA-256:
  `f3b3ed25b7ec83732c2f27471323fd8479eca7d8eac7a62e7fc4621f667ae8db`
- Python and Rust canonical output SHA-256:
  `453f4050098f83906499b0ec5b5a2e0692677dfb089c27812dbf128f69c846dc`
- Compared output bytes: 1,410,350
- Result: `byte_identical`

The canonical output contains every final region's label, score, rectangle,
postprocess order, and source raw index, plus the selected label and raw region
for every input line. The harness gives each implementation the same raw
region order; Python and Rust independently perform initial ordering, the
production transform ladder, final ordering, and line assignment.

Run from the repository root:

```powershell
python experiments\ppdoc-lite\postprocess-parity\check_parity.py `
  --text-fidelity-root "C:\path\to\Text-Fidelity-Project" `
  --random-cases 200 --seed 20260814
```

The command first rejects a different Text-Fidelity revision or dirty source
files, then exits nonzero on the first structural or byte difference.
