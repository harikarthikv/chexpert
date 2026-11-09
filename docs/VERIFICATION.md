# Documentation verification

This audit documents the committed [notebook](../chexpert_cv.ipynb) as inspected on 2026-11-09. Notebook cell numbers below are **1-based positions**, including markdown cells; they are not execution counts. Code was inspected without executing training or changing the implementation.

## Available evidence

| Check | Finding |
| --- | --- |
| Notebook cells | 45 total: 24 markdown and 21 code |
| Code execution counts | All 21 are null |
| Saved code outputs | All 21 output lists are empty |
| Experiment artifacts in the original tracked tree | None; only the notebook, README, and requirements file |
| Verified performance measurements | None |

An empty-output notebook does not prove that its author never ran it elsewhere. It means this repository provides no execution evidence from which to verify performance. Narrative claims of completed evaluation are not substitutes for outputs.

## Claim-to-source map

| README claim | Notebook evidence |
| --- | --- |
| CUDA requirement, seed, environment export, output roots | Cell 3 |
| Observation order, dataset discovery, image resolution | Cell 5 |
| Schema check for all 14 observations | Cell 7 |
| Raw label statistics and conditional frontal filtering | Cell 9 |
| Patient IDs, split fractions, fallback held-out set, disjointness assertions | Cell 11 |
| U-Ones, blank-label mapping, RGB conversion, gray-image fallback | Cell 13 |
| Augmentation, normalization, batch size, worker count | Cell 15 |
| Five backbones, pretrained weights, output heads, HF authentication | Cell 17 |
| Optimizer, scheduler, early stopping, metrics, cache behavior, checkpointing | Cell 19 |
| Five-epoch budgets for all models | Cells 21, 23, 25, 27, 29 |
| Validation comparison and architecture selection | Cell 31 |
| Positive-weight calculation and ablation invocation | Cell 33 |
| Checkpoint loading, initialization fallback, freezing | Cell 35 |
| Held-out evaluation, summary/per-label tables, metric export, ROC plots | Cell 37 |
| CNN Grad-CAM with explicit finding targets | Cell 39 |
| Lowest-AUROC finding selection and false-positive/negative gallery | Cell 41 |
| Narrative clinical interpretation, without supporting saved outputs | Cell 42 |
| Artifact inventory | Cell 44 |

Dependency lower bounds come from [requirements.txt](../requirements.txt). Python 3.10.12 comes from notebook metadata, not an executed environment export. All numeric training settings in the README are configuration values, not experimental results.

## Implementation caveats

The documentation follows executable code where it differs from notebook prose. These findings have been documented without code changes.

1. **Weighted-loss cache collision (cells 19 and 33).** The ablation constructs a separate `ablation_key` but does not pass it to training. It calls `train_one_model` using the winning baseline key. When the baseline metrics and checkpoint exist, the function returns cached metrics before constructing the weighted loss. The ablation JSON may therefore contain baseline metrics. It cannot establish a weighted-loss benefit. Without the cache, this call would reuse the baseline checkpoint/metric filenames.
2. **Out-of-memory behavior (cell 19).** The code clears the CUDA cache and continues to the next batch. The markdown promises batch-size reduction and gradient accumulation, which are not implemented. Skipped batches change the effective training exposure.
3. **Learning-rate floor (cell 19).** The scheduler uses a minimum factor of `0.01` applied to the base learning rate. The floor is `1e-6` for CNNs and `3e-7` for transformers, rather than a universal `1e-6` stated in markdown. These are configured floors, not measured final rates.
4. **Missing checkpoints (cells 35 and 39).** Final evaluation and Grad-CAM can proceed with a newly initialized classification head if the checkpoint is absent. Such output would not demonstrate a trained classifier's performance.
5. **Input integrity (cells 9 and 13).** Frontal-only filtering depends on the view column being present. Missing/corrupt images receive gray replacements with unchanged labels. Successful execution alone does not establish valid image ingestion.
6. **Held-out protocol (cells 13 and 37).** Held-out labels are cleaned before training, while model inference on that set occurs in Section 19. There is one held-out evaluation call in the sequential workflow, but no guard against repeated cell execution. The set may be a training-derived fallback rather than the official validation CSV.
7. **Metric conventions (cell 19).** The metric named AUPRC is average precision from scikit-learn. Both AUROC and average precision are marked undefined for any single-class label by this implementation. Their macro averages exclude such labels, while macro F1 includes all labels. The included label set matters when comparing runs.
8. **Timing and selection (cells 19 and 31).** `training_time_seconds` is captured when a new best checkpoint is found, not after the full run finishes. It represents elapsed time up to that improvement. Architecture ranking uses AUROC rounded to four decimals in the comparison table, so close scores can tie after rounding.
9. **Reproducibility and reuse (cells 3 and 19).** cuDNN benchmarking is enabled with deterministic execution disabled. Cached runs are keyed by model name and file presence; dataset/configuration compatibility is not checked. Checkpoints do not save optimizer/scheduler state for interrupted-run resumption.
10. **Qualitative outputs (cells 39–42).** Grad-CAM selects available true-positive and false-positive examples, with representative-case fallback; it does not guarantee the balanced mix of correct/incorrect cases described in prose. Grad-CAM and error-gallery exceptions can be caught and reported without producing figures. Clinical discussion remains unverified narrative in the committed notebook.

## Evidence needed before reporting results

After an authorized GPU run, use the following checks before updating the [README](../README.md):

- Preserve an executed notebook and the environment metadata, split sizes, image-loading warnings, and checkpoint-loading messages.
- Verify `results/metrics.json` against `results/comparison_table.csv` and saved validation outputs for each completed architecture. Report the actual defined-label coverage for macro metrics.
- Verify held-out summary and per-label values against `results/final_holdout_metrics.json`; identify whether evaluation used official validation data or the fallback split.
- Establish that any weighted-loss result came from a separate weighted training run. The current ablation filename alone is insufficient evidence because of the cache collision.
- Use actual measured parameter counts and clearly label timing as time to the selected checkpoint. Do not present it as full-run duration.
- Confirm that checkpoints contain trained weights and that any published figures were produced successfully from those weights.
- Keep validation and held-out results separate; retain undefined metrics rather than replacing them with zero or invented scores.

This documentation task did not run training, install project dependencies, modify notebook cells, or generate experimental results. No runtime reproducibility claim is made.
