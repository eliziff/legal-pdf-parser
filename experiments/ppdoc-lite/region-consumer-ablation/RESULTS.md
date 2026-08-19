# Region-consumer ablation

Measured 2026-08-14 on the laptop. This asks a deliberately narrow question:
with identical held-out line text and geometry, do region labels improve the
engine's final line roles?

## Inputs and method

- Held-out detector set: 87 pages from the legal25 test COCO.
- Manual-line join: 48 pages, 20 articles, and 2,650 manually labelled lines
  from the 661-page Text-Fidelity corpus. The other 39 detector pages did not
  have a matching manual-line export in the materialized gold directories.
- Model arm: the real `legal25-ppdocv3-l640-e15-q75-l2-orderfree-openvino-fp32`
  pack, OpenVINO CPU, four threads, threshold 0.10.
- Model inference: 107.5 seconds for all 87 pages (0.81 pages/second), including
  process startup and JSON output.
- The model detections run through the production Rust `postprocess_document`
  and `best_region_index` implementations before replay. The production
  complete-coverage rule accepted 17 of 20 article cases; the other three
  correctly fell back to the no-region path.
- The gold-region arm is an oracle upper bound, not a deployable result.

## Final line-role F1

| Final role | No regions | q75 model regions | Gold regions |
| --- | ---: | ---: | ---: |
| Heading | 0.0741 | 0.0992 | 0.0885 |
| Header | 0.5208 | **0.7500** | 0.8548 |
| Footer | 0.1739 | 0.1600 | 0.7647 |
| Footnote body line | 0.3206 | 0.3107 | 0.3444 |

The demonstrated model-delivered improvement is header classification:
**+0.2292 absolute F1**. On only the 17 complete-coverage articles, header F1
is 0.5556 without regions and 0.7895 with the model (+0.2339). The gold-region
ceiling also shows that correct furniture regions could materially improve
footer classification, but this q75 model does not deliver that improvement.

No other row supports a positive product claim. In particular, the heading
replay has geometry-derived font size but lacks the original font flags, so it
is not a suitable heading-quality qualification.

## Explicit non-results

This experiment does **not** measure footnote pairing. The manual line export
has body-region labels but no gold reference-anchor spans, so a previous proxy
that compared body lines with paired-note output was invalid and was removed.
The actual 1,024-article native-core qualification remains the relevant
footnote result: 0.954846 note-label F1 and 0.944823 reference-page F1.

It also does not measure reading order: the replay begins with manual reading
order, so an ordering comparison would be circular.

## Reproduction

```powershell
python experiments\ppdoc-lite\region-consumer-ablation\benchmark.py --self-test

python experiments\ppdoc-lite\region-consumer-ablation\benchmark.py `
  --coco .tmp\stock-layout-20260814\legal25-test\annotations\instance_test.json `
  --manual-gold-root <text-fidelity-gold-root> `
  --legalpdf target\release\legalpdf.exe `
  --output .tmp\region-consumer-ablation\replay `
  --model-predictions <ppdoc-images-jsonl> `
  --postprocess-runner target\release\ppdoc-postprocess-parity.exe
```

The final summary was deterministic across two consecutive runs. Its SHA-256
was `E512E8F731EB0240EF52083D73DAA9FF2A41C873791E3094A2B355282547EC90`.
Raw replay inputs and outputs are disposable and belong under `.tmp/`.
