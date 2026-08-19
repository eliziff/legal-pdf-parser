# DOCX citation-linking benchmark — 2026-07-27

## Deterministic ALR gold

The port was compared byte-for-byte with the working splitter, ignoring only
line endings, then run locally against all 405 accepted manual-gold rows.

| Mode | Exact | Under | Over | Boundary | Abstain |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conservative | 144 | 7 | 2 | 2 | 250 |
| Recall-first | 324 | 0 | 66 | 15 | 0 |

Recall-first exact accuracy was 80.0%, with zero under-splits and zero
character-conservation failures. The conservative provider-replacement gate
accepted 42 of 405 rows.

## Live synthetic Codex comparison

The tracked fixture contains 12 invented footnotes and propositions; no user
document text was sent. Four were conservative deterministic completions.

| Model | Route | Exact | Under / over | Input / output tokens | Wall time |
| --- | --- | ---: | ---: | ---: | ---: |
| `gpt-5.2` | direct | unsupported | — | — | 4.3 s |
| `gpt-5.6-sol` | direct | 12/12 | 0 / 0 | 20,104 / 2,649 | 55.0 s |
| `gpt-5.6-terra` | direct | 12/12 | 0 / 0 | 17,903 / 2,981 | 40.9 s |
| `gpt-5.6-sol` | forced hybrid | 12/12 | 0 / 0 | 19,799 / 1,584 | 36.3 s |
| `gpt-5.6-terra` | forced hybrid | 12/12 | 0 / 0 | 17,592 / 1,967 | 39.9 s |

All successful arms conserved every source character. The first Terra direct
run omitted terminal periods from otherwise correct parts and was rejected by
the strict contract. A deterministic source-only punctuation snap was added;
the rerun then scored 12/12. No inferred text is inserted.

The installed ChatGPT-authenticated Codex CLI returned HTTP 400 for
`gpt-5.2`: that model is not supported with this authentication mode.

The preliminary scan estimated only 231 input-token savings for hybrid, below
the configured 512-token threshold, so `auto` correctly selected direct. Forced
hybrid is retained for larger documents where the deterministic share clears
the threshold.

`gpt-5.6-sol` remains the conservative default until the real ALR DOCX arm is
approved and run. Terra is viable on the synthetic contract test, but that is
not enough evidence to replace the strong default.

## Verification

- Legal PDF Parser: 22 tests passed, including the five DOCX-linker tests.
- Mike provider/linker tests: 26 tests across DOCX routing, local tools,
  provider links, and journal articles passed.
- TypeScript production build passed.
- Raw benchmark responses, telemetry, and local summaries remain untracked
  under `legal-pdf-parser/_temp/`.

## Pending exact approval

The five-arm authoritative run is pending permission to send the selected 12
footnotes and their preceding proposition passages from
`1_AMPLEMAN-TREMBLAY_[Download_and_edit_me] (2).docx` to OpenAI-hosted Codex
models. Its gold partitions remain local. No workaround was attempted after
the safety reviewer blocked that transmission.
