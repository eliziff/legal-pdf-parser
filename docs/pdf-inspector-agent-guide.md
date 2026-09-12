# Maintained PDF Inspector fork

This is the `pdf-inspector` branch of `eliziff/legal-pdf-parser`, maintained for
the legal PDF parser. Firecrawl is upstream; this repository's owner does not
maintain Firecrawl or have responsibility for its private infrastructure.

The canonical source of this file is
[`docs/pdf-inspector-agent-guide.md` on the parser's main branch](https://github.com/eliziff/legal-pdf-parser/blob/main/docs/pdf-inspector-agent-guide.md).
The sync workflow copies it to this branch's `AGENTS.md`. Preserve this fork-owned
guide when merging upstream changes.

## Work and validation

- Preserve the fidelity extraction API, source coordinates, glyph/font evidence,
  painted rules and native/WASM support. Trace their consumers before resolving
  upstream conflicts. Never discard either side's capabilities just to merge.
- Use affected `cargo check` and focused behavior tests during iteration. Batch
  changes before linking integration binaries; do not run concurrent Cargo jobs
  against the same target directory.
- For Rust changes, format touched files and run the applicable crate tests and
  Clippy checks. Use the checked-in fixture tests; do not invent a replacement
  gold result from the implementation being tested.
- Gate the exact parser/Inspector combination with the parser's
  [PDF Inspector preflight workflow](https://github.com/eliziff/legal-pdf-parser/blob/main/.github/workflows/pdf-inspector-gate.yml).
  Its frozen
  [legal corpus manifest](https://github.com/eliziff/legal-pdf-parser/blob/main/experiments/cache-contract-fidelity/manifest.json)
  and pinned public fixture checkout are the maintained integration inputs.
  Compare base and candidate source-document products without changing the
  expected hashes or bypassing fidelity checks. Investigate differences.
- Firecrawl's private `pdf-evals`, internal runners and release procedures are
  not prerequisites for work on this fork. Do not ask the owner for access,
  locations or permission to proceed without them. Use the maintained gates
  above and repair their reproducibility when necessary. An unavailable upstream
  internal resource is not a reason to stop otherwise authorized fork work.
- Keep one maintained `pdf-inspector` branch. Temporary candidate worktrees are
  disposable; publish and pin a validated combination rather than creating
  another permanent engine copy. Report actual validation and unresolved
  regressions accurately.
