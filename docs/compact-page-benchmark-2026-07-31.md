# Compact page publication benchmark — 2026-07-31

## Change

The parser still extracts and analyzes the complete word/span geometry. For
Beaver's compact source contract, `legalpdf parse --compact-pages` now avoids
serializing geometry that Beaver immediately discarded. The standard artifact
profile is unchanged.

The JSONL writer also replaces recursive `dataclasses.asdict()` deep copies
with the standard `json` encoder's shallow dataclass adapter. A direct
five-run serializer comparison on the same parsed document produced identical
bytes and fell from 251 ms to 104 ms per publication.

## Cold-process measurement

- Machine: Intel Core i3-1315U Windows development machine
- Source: Beaver Library PDF, 24 pages, 150 paired footnotes
- Baseline: engine commit `51d51c49`, full page publication
- Candidate: same parser plus `--compact-pages`
- Cache: disabled; new Python process and empty output/cache directory per run
- Runs: seven alternating baseline/candidate pairs

| Variant | Median | Minimum | Maximum | Median output bytes |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 1.4219 s | 1.3427 s | 2.1987 s | 2,856,039 |
| Compact | 1.2182 s | 1.1472 s | 1.9492 s | 401,103 |

The cold median improved by 14.3%, and transient output fell by 86.0%.

## Fidelity gate

After Beaver's existing compact-page normalization, `pages`, `paragraphs`,
`sections`, `footnotes`, `diagnostics`, and `repairs` were byte-for-byte
identical. Engine tests also compare the faster serializer against the legacy
serializer for every artifact collection.

The manifest intentionally changes its parser-code hashes and deterministic
cache key because those fields are implementation receipts. Beaver removes
pairing `created_at` and `elapsed_seconds` telemetry from the compact durable
manifest; otherwise two identical cold parses could not be byte-reproducible.

## Second pass: lazy reporter pattern

The McGill reporter recognizer has 779 reporter alternatives and was being
compiled while checking ordinary headings. Every reporter citation requires
at least two separate digit runs, so headings with fewer than two now fail that
check before the large pattern is compiled. This guard cannot remove a match.

Seven alternating cold compact parses against engine commit `80848aa` improved
the median from 1.3513 s to 1.2472 s (`-7.7%`; baseline range
1.3141–1.9234 s, candidate range 1.2149–1.3002 s). All six structural and
evidence JSONL artifacts were byte-for-byte identical.

## On-demand geometry upgrade

`legalpdf add-geometry` extends a matching compact artifact without repeating
paragraph, section, footnote, diagnostic, or repair derivation. It verifies the
source hash, parser version, OCR configuration, full engine-code identity, and
compact page text, then publishes complete gzip-compressed `Page` records with
a payload hash. The sidecar deliberately duplicates page text, but does not
duplicate or recompute any derived evidence. Loading it recreates the original
in-memory `Page` records exactly.

On the same 24-page/150-footnote PDF, seven alternating cold processes compared
a full no-cache reparse with the incremental geometry path:

| Variant | Median | Minimum | Maximum | Added bytes |
| --- | ---: | ---: | ---: | ---: |
| Full reparse | 1.7850 s | 1.6408 s | 2.7194 s | 2,856,038 |
| Geometry sidecar | 1.7284 s | 1.5683 s | 2.5183 s | 535,790 |

The incremental path was 3.2% faster and wrote 81.2% fewer new bytes. Combined
with the existing 401,103-byte compact source, the fully usable compact-plus-
geometry representation is 67.2% smaller than republishing the full artifact.
Geometry is not retained unless a geometry consumer asks for it.
