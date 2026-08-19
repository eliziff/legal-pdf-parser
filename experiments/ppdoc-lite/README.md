# PPdoc Lite

Standalone local layout inference for the Legal PDF Parser, following
the measured-runtime method used by `../kraken-lite`.

The deployable code is `ppdoc-lite-runtime/`. It has no Paddle, PaddleX,
training, compiler, calibration-corpus, Docker, account, server, or network
dependency. The Text Fidelity project remains the source of the 661-page legal
layout corpus and the promoted teacher; it does not own this runtime.

The original 581/40/40 split is retained only as a legacy regression surface:
39 of 40 validation pages and 37 of 40 benchmark pages share an article with
training. `training/make_generalization_splits.py` instead deterministically
repartitions all 661 pages into 499 training pages, 75 article-disjoint
validation pages, and an 87-page test set holding out three complete journals.
The test annotations remain sealed until model selection is over.

The experiment first proves the incumbent teacher through a direct Rust and
OpenVINO CPU path, then tests standard PTQ and QAT/self-distillation recipes.
A smaller detector is considered only after it demonstrates useful validation
quality; the failed PicoDet-M screen is not a deployable tier. The detailed
evidence and training sequence are in `legal-layout-training-recipe.md`. No
candidate gets a user-facing tier name until it is on the held-out
speed/quality Pareto frontier. Durable measurements and rejected candidates
are recorded in `RESULTS.md`.

Generated models, environments, corpora, and raw run artifacts stay outside
Git. Only source, frozen manifests, concise benchmark receipts, and final model
hashes belong here.
