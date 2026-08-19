# Real Codex structural-repair benchmark — 2026-07-26

This is a real-document comparison, not the earlier synthetic smoke test.

- Source: 19-page native PDF export of the Koning–Reid–Baker ALR article.
- PDF SHA-256: `e4f750449cafa3b08d2acd90dae222ee586ed63f1cb76a9621469470b99ee79e`
- Gold: structural content extracted from the paired ALR DOCX.
- Gold SHA-256: `69fa5c4b5e099aaffb1ebee5f7208648499bc0a7431fb96fc274c43a645b899f`
- Engine commit: `b2fcc96c187b607b8c4cd5e3c0ba49685e4766d3`
- Codex CLI: `0.145.0`
- All Codex arms used the same PDF bytes, local parse, diagnostics, r=1
  context, IDs-only schema, retry policy, and DOCX evaluator.

## Quality

| Arm | CER | WER | Pairwise order | Adjacent order | Footnote F1 | Body similarity | Sentence proposition | Passage proposition |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | .15760 | .16506 | .96440 | .88732 | .93617 | .95084 | .84937 | .80084 |
| Luna low | .03525 | .03612 | .97845 | .90741 | .93617 | .99676 | .95299 | .88981 |
| Terra low | .03565 | .03585 | .98142 | .93333 | .93617 | .99676 | .95299 | .90128 |
| Sol low | **.03487** | **.03531** | .98115 | **.93651** | .93617 | .99676 | **.95968** | **.90447** |
| Luna xhigh | .03525 | .03612 | .97778 | .92593 | .93617 | .99676 | .95299 | .90128 |

Citation recall stayed at `.91304` and footnote recall at `.88` in every arm.
Codex improved boundaries, ordering, body fidelity, and proposition pairing; it
did not discover the nine missing footnotes.

## Runtime and usage

| Arm | Wall time | Peak RSS | Live calls | Retries | Input tokens | Cached input | Output tokens | Line conservation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Local | 0.92 s | 55.1 MiB | 0 | 0 | 0 | 0 | 0 | n/a |
| Luna low | 158.15 s | 61.7 MiB | 7 | 1 | 186,091 | 42,240 | 3,887 | 100% |
| Terra low | 123.71 s | 54.4 MiB | 6 | 0 | 194,611 | 30,208 | 3,966 | 100% |
| Sol low | 126.53 s | 53.9 MiB | 6 | 0 | 207,787 | 0 | 4,155 | 100% |
| Luna xhigh | 226.41 s | 53.8 MiB | 6 | 0 | 186,121 | 28,160 | 10,285 | 100% |

All six repair scopes were schema-valid and applied in every arm. Luna low
needed one retry. An identical Luna-low replay took `0.44 s` and made zero live
calls, confirming persistent content-addressed repair caching.

## Interpretation

On this document, Terra low and Sol low form the useful frontier. Sol low has
the best reconstruction scores for about 6.8% more input tokens and 2.8 seconds
more wall time than Terra low. Luna xhigh is dominated here: it is slower,
produces substantially more output tokens, and does not improve quality over
the low-effort frontier.

One article is enough to prove the bridge and expose a real quality gain, but
not enough to set a universal default. Run the frozen multi-document manifest
before changing the default arm.

## Experimental-adapter port check

The later ALR experimental-adapter port was replayed locally on this same case.
Footnote count, precision, recall, and F1 were unchanged (66 candidates, 66
matches, `.93617` F1), while mean footnote-body similarity rose from `.95084`
to `.98595`, sentence proposition similarity from `.84937` to `.95898`, passage
similarity from `.80084` to `.92205`, and citation recall from `.91304` to
`1.0`.

The stricter region gate also recovered table and body rows that the old parser
had incorrectly swallowed as footnotes. Those recovered table cells are not
yet ordered ideally in local mode: CER/WER moved to `.19711`/`.19730` and
pairwise/adjacent order to `.90952`/`.80952`. This is a structural-repair case,
not a reason to preserve the false footnote region. The Codex arms above remain
the pre-port comparison and should be rerun on the frozen corpus before using
them as post-port defaults.
