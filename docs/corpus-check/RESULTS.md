# Production corpus checker

Status: passing on the complete cached corpus.

`legalpdf-corpus-check` reads each settled extraction cache and calls the final
production document builder once. It uses a bounded three-worker standard
library pool, writes one atomic compact receipt per completed document, retains
those receipts when interrupted, and reports live document, page, error, and
throughput progress. There is no resume protocol, run manifest, binary hash,
self-hash comparison, bootstrap mode, or separate throughput gate.

```powershell
legalpdf-corpus-check --baseline <catalog> `
  --corpus-root <corpus> --out <new-empty-dir> --jobs 3
```

The `legalpdf.corpus-check-baseline.v1` catalog contains only source identity,
relative extraction-cache path, page count, jurisdiction, source family, and
the two known ingestion failures. It supplies identities and denominators; it
does not freeze structure output. Receipts record structure counts and sampled
nodes with text, exact source ranges, parents, locator kinds, and proof rules so
changes can be judged as capability gains or regressions.

## 2026-08-21 full run

- Result: **pass**
- Documents: 748/748
- Pages: 24,707/24,707
- Known trailer failures: 2 documents and 72 pages, exactly accounted
- Processing errors: 0
- Wall time: 66.882 seconds (limit: 180 seconds)
- Throughput: 369.413 pages/second
- Heading-derived sections: 0

Detected structure:

| Kind | Count |
| --- | ---: |
| Heading | 9,719 |
| Legal section | 11,923 |
| Numbered paragraph | 23,988 |
| List | 15,729 |
| List item | 82,705 |
| Footnote | 8,471 |
| Endnote | 57 |

Legal sections by locator kind: 10,286 sections, 762 parts, 456 appendices,
252 clauses, 65 schedules, 43 articles, 40 annexes, and 19 exhibits. The run
also recorded 17,773 abstained runs and 3,860 partially resolved runs rather
than promoting uncertain structure.

Diagnostics were 9,259 `note_pair_unmaterialized`, 17,773
`structure_run_abstained`, 3,860 `structure_run_partially_resolved`, and 24,021
`structure_run_resolved`. Incomplete note materialization remained non-fatal.

## Smell test

The sampled receipts preserve real provision structure in long bylaws and
legal submissions, including parts, appendices, rooted numbered paragraphs,
and nested lists. PDF-only collision evidence now abstains on repeated dotted
contents rows, aligned transcript line-number columns, and back-of-book
page-line concordance rows. Heading and paired-note totals were unchanged by
that suppression, while the false-prone section and paragraph totals fell.

This is a ratchet, not a perfection claim. Dense zoning tables and form-like
comment rows remain the highest-risk documents for occasional false positives,
and the unmaterialized-note diagnostic count remains visible for later work.

The A2AJ path is unchanged by the PDF collision evidence: A2AJ does not emit
those observations. Its focused backend structure suite passed 84/84 tests
after this run.
