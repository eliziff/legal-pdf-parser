PDF Inspector in `eliziff/legal-pdf-parser` is validated by `.github/workflows/pdf-inspector-gate.yml` against `experiments/cache-contract-fidelity/manifest.json`, without Firecrawl's private `pdf-evals` repository.

Upstream lands only as one single-parent commit per sync, made by `tools/merge-pdf-inspector.sh` on the parser's main branch; `.upstream-pdf-inspector` records the upstream commit incorporated and is the next merge base. Never merge upstream history into this branch.
