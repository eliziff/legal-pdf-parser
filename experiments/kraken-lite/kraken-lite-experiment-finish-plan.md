# Kraken-lite experiment finish plan

## Fixed inputs

- Recognizer teacher: `current_best.safetensors`, SHA-256 `cae8fcf6fa2e81e758143dc49c817feef604e42980022045224ad44910d349fc`.
- Teacher architecture: 5,723,777 parameters; three 256-unit bidirectional LSTM layers; 225 CTC outputs.
- Segmenter: stock Kraken BLLA, SHA-256 `77a638a83c9e535620827a09e410ed36391e9e8e8126d5796a0f15b978186056`.
- Training corpus: the 12 accepted DB-order-enriched PAGE exports under
  `F:\oajd_run_archive\oajd_compute_20260713_verified_cleanup\Text-Fidelity-Project\output\escriptorium_exports` on the desktop PC. Use exactly the `_03`/`_04` runs listed in `kraken-lite-training-data\ordered-export-verification.json`.
- Corpus facts: 662 source PAGE/image pairs and 32,630 eScriptorium DB lines; all 32,630 matched during order enrichment with zero warnings. Materializing the accepted `_05` bundle skips one page without manual lines, leaving 661 pages across 350 articles and 32,553 corrected lines.
- Local export receipt: `kraken-lite-training-data\ordered-export-verification.json`, including each archive's SHA-256.
- Never use `train1_12_full_manual_gold_ordered_db_reading_order_20260629_01`. The project's fail-closed verifier explicitly blacklists that intermediate revision. The accepted materialized revision is `_05`.

Do not copy the roughly 945 MB corpus to the laptop. Train beside the 3080 Ti and copy back only manifests, logs, checkpoints, final models, and benchmark receipts.

## 1. Finish and freeze the current inference runtime

1. Resume the exact h1350 calibrated BLLA validation and reject it if it is not a strict size/speed win at the recorded quality gate.
2. Run the full unit suite and one real-PNG smoke for each intended execution path.
3. Record cold and warm page latency, pages/minute, model bytes, line count, and differential CER on the frozen multi-journal corpus.
4. Ship four independent tiers:
   - Fidelity: FP32 BLLA h1800 + FP32 recognizer, full geometry.
   - Quality: static INT8 BLLA h1500 + LSTM-only INT8 recognizer at full width, full geometry.
   - Fast: static INT8 BLLA h1350 + LSTM-only INT8 recognizer at 0.85 width.
   - Turbo-lite: classical line layout + LSTM-only INT8 recognizer at 0.70 width, coarse line boxes, no character geometry.
5. Keep only strict wins. Do not ship DirectML, FP16, full dynamic quantization, reduced input height, x0.50, or x0.60 experiments unless a new corpus benchmark reverses their measured loss.

## 2. Prepare one defensible recognition split

1. Reuse the historical article-level unseen holdout if its manifest is present.
2. Otherwise create one deterministic, journal-stratified split grouped by article, never by page or line. Freeze it before training.
3. Compile train/eval data with Kraken's existing `ketos compile` path. Preserve PAGE XML reading order and the teacher's 225-symbol codec.
4. Save the split manifest and counts; do not duplicate the image corpus.

## 3. Screen smaller supervised VGSL students

Use the existing Kraken training launcher on the 3080 Ti. Start from ground-truth CTC training; no custom distillation code.

Screen only three students:

- S1: two bidirectional LSTM layers at width 192.
- S2: two bidirectional LSTM layers at width 128.
- S3 extreme: one bidirectional LSTM layer at width 128.

Keep the teacher's convolutional front end and codec for the first screen. Run a one-batch smoke, then the same short fixed-step screen for all three. Promote at most two models to early-stopped full training. Use the launcher's existing TF32/compiled-data preset; increase batch size only after measured GPU-memory headroom.

## 4. Select on deployed CPU performance

For the teacher and every student candidate, measure on the same held-out lines and pages:

- absolute CER and exact-line rate against human ground truth;
- error deltas for footnotes, punctuation, capitals, digits, and non-ASCII characters;
- FP32 and selective LSTM INT8 model size;
- warm CPU lines/second with the runtime's eight single-thread workers;
- end-to-end Turbo-lite seconds/page and pages/minute.

A student survives only if it materially reduces model bytes or warm CPU time. Tentative drift ceilings, fixed before the full run:

- Quality replacement: no more than +0.25 percentage points CER versus the teacher.
- Fast replacement: no more than +1.0 point.
- Turbo-lite replacement: no more than +3.0 points and no obvious collapse on any journal.

## 5. Escalation ladder

Stop after supervised training if a student meets the deployment gate.

Only if all supervised students miss:

1. Add ordinary teacher pseudo-labels from extra unlabeled journal lines and retrain the best student.
2. Try logit distillation only if pseudo-label training is insufficient; it requires a custom Kraken training loss/export path.
3. Try quantization-aware training last. Current post-training LSTM INT8 adds only about 0.10% differential CER, while standard TorchAO QAT support does not directly match Kraken's fused LSTM deployment graph.

## Done when

- The current four-tier runtime is tested, benchmarked, and packaged independently.
- At least S1/S2/S3 have comparable short-screen receipts, or a documented environment failure prevents training.
- Any trained replacement beats its current tier on deployed CPU throughput/size within the fixed CER ceiling.
- The final package has no Kraken, PyTorch, compiler, calibration corpus, or training dependency at inference time.
