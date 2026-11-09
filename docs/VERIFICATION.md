# Documentation verification

Audit date: 2026-11-09. The source is the supplied [executed notebook](../chexpert.ipynb), preserved without editing. Cell numbers below are **1-based positions**, including markdown cells, not execution counts. Documentation uses saved outputs and executable code; training was not rerun.

## Evidence and execution status

| Check | Finding / source |
| --- | --- |
| Notebook structure | 45 cells: 24 markdown, 21 code |
| Execution | 19 code cells have execution counts; cells 41 and 44 are unexecuted |
| Error | Cell 39: `ModuleNotFoundError: No module named 'pytorch_grad_cam'` |
| Run metadata | Python 3.12.13; Papermill start/end on 2026-09-26; `exception: true` |
| GPU output | Cell 3: Tesla T4 ×2 detected, first GPU 14.56 GB, compute capability (7, 5); code uses `cuda:0` |
| Data paths | Cell 5: `/kaggle/input/datasets/ashery/chexpert/train.csv` and `valid.csv` |
| Training | Cells 21, 23, 25, 27, 29: all five models complete five epochs |
| Held-out evaluation | Cell 35 loads ResNet50 epoch 4; cell 37 prints summary and per-label results |
| Figures | Cells 9, 31, 37 contain embedded label-distribution, loss, comparison, AUROC, and ROC images |
| Qualitative analysis | Grad-CAM failed before producing a result; error analysis cell 41 did not execute |
| Standalone artifacts | Runtime JSON, CSV, checkpoint, and PNG files are not included separately in the repository |

The run completed model training and held-out evaluation, but did not finish the entire notebook. Its later conclusion and clinical discussion are markdown, not evidence of completed execution. The attached read-only snapshot is only introductory markdown, and its attached output JSON is empty; the full workspace notebook supplies the results.

## Numeric claims and sources

| README values | Source |
| --- | --- |
| Raw training/validation shapes: (223414, 19), (234, 19) | Cell 7 output |
| Frontal training pool: 191027 images, 64534 patients | Cell 11 output |
| `dev-train`: 171927 images, 58080 patients | Cell 11 output |
| `dev-val`: 19100 images, 6454 patients | Cell 11 output |
| Held-out: 202 images, 200 patients; zero pairwise overlap | Cell 11 output |
| Five comparison rows: AUROC, AP, F1, rounded parameter counts, selected epochs, checkpoint times | Cell 31 output |
| Held-out ResNet50: AUROC 0.8159, AP 0.5359, F1 0.3476, 202 images | Cell 37 summary output |
| All 14 held-out per-label rows and positive counts | Cell 37 per-label output |
| Fracture has 0 positives and undefined AUROC/AP; 13 labels contribute to these macro averages | Cell 37 output and cell 19 code |
| F1 threshold 0.5, macro F1 over all 14 labels | Cell 19 code |
| ResNet50 selected checkpoint epoch 4 | Cells 31 and 35 outputs |
| DenseNet121 epoch-5 AUROC 0.8036, retained epoch 4 | Cell 23 log, cell 31 table, cell 19 tolerance |
| GPU/memory, Python version, execution date | Cell 3 output and notebook metadata |

Values are reproduced at the notebook's displayed precision. Summary values are not recalculated from rounded per-label rows. Raw predictions and full-precision metric JSON files are unavailable, so this audit verifies transcription and code interpretation rather than independently recomputing scores.

## Methodology source map

| Topic | Source cells |
| --- | --- |
| CUDA requirement, seed, directories, environment export | 3 |
| Observation order, discovery, resolver, schema | 5, 7 |
| Raw statistics before cleaning and frontal filtering | 9 |
| Patient grouping, fractions, fallback, disjointness | 11 |
| U-Ones, blanks, RGB, gray-image fallback | 13 |
| Augmentation, normalization, batch size 32, workers 2 | 15 |
| Architectures, pretrained weights, 14-output heads, HF access | 17 |
| Optimizer, learning rates/decay, schedule, AMP, early stopping, metrics, cache | 19 |
| Five-epoch training calls | 21, 23, 25, 27, 29 |
| Selection and comparison plots | 31 |
| Positive weights and cached ablation | 33 |
| Freezing and held-out evaluation | 35, 37 |
| Grad-CAM import failure | 39 |
| Unexecuted error analysis and artifact inventory | 41, 44 |

Declared dependency bounds come from [requirements.txt](../requirements.txt), which remains unchanged. Exact executed package versions cannot be recovered from the printed environment-export confirmation alone.

## Implementation caveats

These findings describe the supplied implementation without changing it.

1. **Weighted-loss cache reuse is observed.** Cell 33 prints `Resumability: Found completed run for 'resnet50'. Loading cached metrics & checkpoint...` and then reports 0.8035 AUROC / 0.3789 F1 for both losses. It passes the baseline key to training; cache lookup precedes loss construction. Its separate `ablation_key` is unused. These are duplicated baseline scores, not a separately trained weighted-loss result.
2. **Checkpoint tolerance and rounded ranking matter.** Cell 19 requires AUROC improvement greater than `1e-4`. DenseNet121's final displayed 0.8036 did not trigger a checkpoint; comparison uses epoch 4 at 0.8035. Cell 31 ranks four-decimal values and selects ResNet50, tied with DenseNet121 at that precision. No statistically significant superiority is established.
3. **Comparison times are time to checkpoint.** `training_time_seconds` is captured on qualifying improvement. Cell 31 reports 5006.5, 4229.0, 5029.3, 4891.2, and 5258.1 seconds in model order. End-of-training logs instead print total durations of 6115.6, 5192.8, 5029.3, 4891.4, and 5258.2 seconds. The README labels comparison times accordingly.
4. **AUPRC means average precision.** Cell 19 uses `average_precision_score`. Both AP and AUROC are marked undefined for single-class targets and excluded from those macro averages. Macro F1 includes all labels. Held-out Fracture is the excluded label.
5. **Grad-CAM is not installed automatically.** The updated notebook imports `pytorch_grad_cam` directly. Its missing-package exception occurs before the plotting try/except. Installing the existing requirements into the active kernel is necessary to attempt this section; this task did not install packages or rerun it.
6. **OOM behavior differs from prose.** Cell 19 clears the CUDA cache and skips the affected batch; it does not reduce batch size or accumulate gradients. Documentation describes code behavior without asserting skipped batches occurred in this run.
7. **Learning-rate floor differs by family.** Minimum multiplier 0.01 corresponds to configured floors `1e-6` for CNNs and `3e-7` for transformers, rather than the universal `1e-6` in notebook prose. These are bounds, not measured final rates.
8. **Input fallback can affect validity.** Frontal filtering depends on the view column. Image failures produce gray replacements with unchanged labels. The saved run shows frontal filtering and three successful resolution checks, which do not independently verify every image.
9. **Held-out protocol is procedural.** This set is frontal official validation, not the challenge test set. Labels are cleaned before training, while inference follows freezing. The sequential workflow has one held-out call and no persistent guard against repeated evaluation.
10. **Missing-checkpoint fallback remains in code.** Evaluation and Grad-CAM can initialize a new head if a checkpoint is absent. The saved held-out run confirms successful trained-checkpoint loading, so that fallback did not produce its reported scores.
11. **Reproducibility is limited.** Seed 42 is set, but cuDNN benchmarking is enabled and deterministic execution disabled. Cache reuse does not verify dataset/configuration compatibility. Checkpoints omit optimizer/scheduler state. No repeated seeds, confidence intervals, or external validation results are supplied.
12. **Preserved notebook prose contains an old filename.** Introductory markdown still refers to the previous notebook name. Repository documentation uses the actual lowercase filename. The replacement notebook is preserved byte-for-byte under the user's no-implementation-edit constraint.

## Publication verification

The update checks README rows against saved comparison and held-out tables, checks links/anchors, removes the obsolete tracked notebook, and preserves the replacement notebook's file hash. Dependencies remain unchanged. No model, dataset, training code, or notebook outputs were edited, and no experiment was rerun.
