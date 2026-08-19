# Canonical pairer parity receipt

Date: 2026-07-30

The engine's production pairing call graph was compared directly with
`Text-Fidelity-Project/tools/footnotes/footnote_pairing_v2.py` at commit
`d8b25257687b3b9aad644dec42cca966b45675ff` (file SHA-256
`59dca5470c79731dfc21e61fc235256f33943efd43c2662a4f5cf1292f7bb56c`).

Inputs:

- locked 100-article manifest:
  `67d46671cb5b81396e181c8c83600e8018cbc8be4b8a935de4835b24c35a6ce6`
- gold lines:
  `c2b7620d8378c227e3f369fabb69408e9e06fcc6632608772a363ec969a17fd9`
- verification ledger:
  `a25e8d53f8a108f1c1e0bc0ffe99092edd28acbb23091ebead30aac2926e6ec6`

Result:

- 100/100 article marker lists matched exactly.
- Each implementation emitted 3,169 paired label groups, 105 label-only
  groups, 3,169 reference rows, and 6,443 total marker rows.
- The frozen 779-entry McGill reporter inventory is 7,202 bytes with SHA-256
  `6c5aa7b0b826e0842ff0631de40ee8457cb5e948eb120e7568686489703b57d7`.
- Removing unused upstream benchmark/CLI code reduced the production stage from
  4,424 to 2,570 lines without changing marker output.

This is an implementation-parity receipt, not a human-gold accuracy claim.
