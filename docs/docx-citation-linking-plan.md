# DOCX citation-linking bridge

## Objective

When a user asks Mike to link citations in a Word document, the main chat model
must call one bounded workflow. It must not reread every footnote, split
citations itself, or invent source URLs.

The workflow is local-first and independent of the ALR Quote Verifier at
runtime. Mike's existing Supabase/cloud paths remain intact.

## Reused, verified behaviour

The implementation was derived from the working code and benchmark records in
the ALR Quote Verifier:

- `verifier_core/deterministic_splitter.py`: deterministic source splitting
  and field extraction.
- `alr_quote_verifier.py`: high-accuracy, economy, ultra-economy, and free
  routing semantics; sequential supra/ibid identity propagation; strict
  response schema and McGill-oriented split instructions.
- `dev/benchmarks/fast_split_manual_gold.jsonl`: 405 accepted gold records.
- `dev/benchtools/benchmark_fast_splitter.py` and
  `benchmark_deterministic_splitter.py`: exact/over/under/boundary scoring.
- `data/inputs` and `data/samples`: real DOCX corpus.

The ALR configuration used `gpt-5.2`, reasoning effort `none`, and a 16,000
output-token cap. The installed ChatGPT-authenticated Codex CLI does not accept
`gpt-5.2`; its successful strong-model replacement is `gpt-5.6-sol`.
`gpt-5.6-terra` remains a benchmark comparison, not an assumed substitute.

## Implemented flow

1. Extract footnotes and the proposition passage preceding each note from the
   DOCX.
2. Run the copied deterministic splitter locally.
3. Estimate direct and hybrid token use, including measured fixed Codex CLI
   overhead. Use the hybrid route only when the estimated saving exceeds the
   configured threshold; otherwise batch the whole job directly.
4. Send at most 32 footnotes or 45,000 input characters per bounded Codex call,
   with a maximum of 13 calls and 400 footnotes per document.
5. Enforce a strict JSON schema and reject:
   - any URL;
   - missing, duplicate, or extra note IDs;
   - unsupported fields or source types;
   - citation parts that lose, add, overlap, or reorder characters;
   - support quotes not copied exactly from the supplied footnote or
     proposition.
6. Resolve supra/ibid identities deterministically from prior citation history.
7. Hand only compact citation identities, locators, and exact support quotes to
   Mike's provider layer.
8. Resolve through A2AJ, CourtListener (local bulk first when configured),
   TNA/GOV.UK/GovInfo, or `public_endpoint.db` journal records.
9. Construct URLs only from provider-returned evidence:
   - native paragraph, section, or page anchors where supported;
   - verified text fragments otherwise;
   - multiple verified quotes become one multi-text directive URL.
10. Verify the DOCX hash, apply external OOXML hyperlinks without changing the
    citation text, and save a new local Library version.

The Python worker cache is content-addressed by prompt/schema version, input,
model, and effort under the shared legal-data AppData cache. Calls are
ephemeral: durable citation-job caching is useful, while an open-ended Codex
conversation is not.

## Mike integration

Account-free/local chat exposes `library_link_docx_citations`. Its tool
description and system instruction require immediate delegation of DOCX
citation-linking requests. The specialized worker model is controlled by:

- `MIKE_DOCX_LINK_MODEL` (default `gpt-5.6-sol`);
- `MIKE_DOCX_LINK_EFFORT` (default `none`);
- `MIKE_DOCX_LINK_STRATEGY` (`auto`, `direct`, or `hybrid`).

These settings are intentionally separate from Mike's dynamic chat-model
catalog. Local Codex is not shown as a normal assistant model.

Cloud storage and Supabase code are neither removed nor replaced. The local
tool uses Mike's local Library version store; a cloud adapter can feed the same
provider-link map and Python writer without changing either component.

## Equivalence test plan

### Deterministic and contract tests

- Compare the copied splitter to the source file, ignoring line endings.
- Run the 405-row accepted manual split gold.
- Verify exact, tolerant-exact, under-split, over-split, and character-neutral
  metrics.
- Assert that worker output containing any URL is rejected.
- Assert that unsafe provider results do not reach the DOCX writer.
- Round-trip a minimal DOCX and prove that its footnote text is unchanged after
  linking.
- Run Mike provider-link tests for native anchors and atomic multi-text
  directives.

### Model comparison

Use the same fixed 12-record sample and compare:

| Arm | Model | Route | Effort |
| --- | --- | --- | --- |
| Historical request | `gpt-5.2` | direct | none |
| Strong replacement | `gpt-5.6-sol` | direct | none |
| Economy candidate | `gpt-5.6-terra` | direct | none |
| Routing check | `gpt-5.6-sol` | forced hybrid | none |
| Routing check | `gpt-5.6-terra` | forced hybrid | none |

Record schema success, exact/tolerant partitions, character conservation,
under/over-splits, wall time, cache status, and Codex-reported token use.
Select Terra only if it matches Sol's tolerant accuracy and does not increase
under-splits. Default to direct routing when the preliminary scan does not save
tokens after fixed invocation overhead.

A 12-case invented fixture is tracked at
`dev/benchmarks/docx_linking_synthetic.json` for safe live-call smoke tests.
The real ALR DOCX sample remains the authoritative equivalence test, but
sending its selected footnotes and proposition passages to OpenAI requires
explicit approval.

