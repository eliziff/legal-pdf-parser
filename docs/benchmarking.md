# Benchmarking against OpenDataLoader

The paired harness runs two `pdf2md` binaries through the same local
OpenDataLoader corpus, evaluates both outputs, and reports aggregate and
per-document deltas. This avoids comparing results produced from different
corpus revisions or evaluator versions.

Build a candidate and provide a released or worktree build as the baseline:

```bash
cargo build --release
python3 scripts/bench_opendataloader.py \
  --bench-dir ../opendataloader-bench \
  --baseline ../pdf-inspector-main/target/release/pdf2md \
  --candidate target/release/pdf2md \
  --max-document-regression 0.02 \
  --json-output /tmp/pdf-inspector-benchmark.json
```

Pass `--reference-evaluation path/to/evaluation.json` to report the candidate
delta against another evaluation, and add `--require-reference-lead` to make a
negative reference delta fail the run. By default, the candidate must not
regress the baseline overall score or introduce missing predictions. Use
`--min-overall-delta` to require a specific aggregate gain.

The OpenDataLoader repository is external and keeps its normal
`prediction/pdf-inspector` output. Paired evaluation copies each run into a
temporary directory before evaluating it, so the baseline and candidate cannot
overwrite one another.
