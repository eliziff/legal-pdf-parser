# Legal PDF parser agent rules

- This is the maintained `eliziff` fork. Its owner is not responsible for
  Firecrawl's private repositories, corpora, runners or release infrastructure.
  Do not ask the owner to supply them or make them an approval checkpoint.
- PDF Inspector's fork instructions are owned by
  `docs/pdf-inspector-agent-guide.md`. The upstream sync copies that guide to
  the maintained `pdf-inspector` branch's `AGENTS.md`; preserve it during manual
  merges too. Use this fork's frozen corpus and product-parity workflow, not
  upstream's private `pdf-evals`, as the integration gate. Do not weaken the
  gate or regenerate expected results to accommodate a candidate.
- For ordinary Rust edits, run `cargo quick`. Do not run `cargo build`,
  `cargo run`, `cargo test`, or a corpus replay after each change.
- Batch source changes behind metadata checks. Link the executable or test
  harness once, only when a final behavioral gate actually requires it.
- Reuse the most recently linked binary for diagnostics that do not require new
  code. Never start another Cargo command while Cargo or rustc is still active
  for this repository.
- Format only touched Rust files with `rustfmt --edition 2021 --check <files>`;
  `cargo fmt --all` needlessly traverses the entire crate and currently
  overflows rustfmt's stack.
- Keep benchmark output bounded and disposable. Reuse one output directory,
  compare it to frozen hashes, and remove it immediately after recording the
  result.
- Keep any future repair or external-layout adapter provider-neutral: the
  engine validates bounded structural assignments and source identity, while
  the embedding application owns provider and runtime selection.
