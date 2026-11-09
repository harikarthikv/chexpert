# CheXpert-CV

Multi-label chest X-ray classification with PyTorch, comparing three convolutional neural networks (CNNs) and two vision transformers on CheXpert-v1.0-small.

**Recorded result:** ResNet50 was selected on `dev-val` and achieved held-out macro AUROC **0.8159**, average precision **0.5359**, and F1 **0.3476** on 202 frontal images. All five models completed training. The saved run subsequently stopped at a missing Grad-CAM dependency, so explainability and error analysis did not complete. See the [verification notes](docs/VERIFICATION.md) for sources and limitations.

## Contents

- [Project overview](#project-overview)
- [Dataset and preprocessing](#dataset-and-preprocessing)
- [Models and training](#models-and-training)
- [Evaluation and results](#evaluation-and-results)
- [Run the notebook](#run-the-notebook)
- [Repository and generated artifacts](#repository-and-generated-artifacts)
- [Limitations](#limitations)
- [References](#references)

## Project overview

The [executed notebook](chexpert.ipynb) fine-tunes pretrained image classifiers to predict 14 observations independently. A radiograph may contain several findings, so each model produces 14 logits, trained with binary cross-entropy (`BCEWithLogitsLoss`) and converted to probabilities with sigmoid.

The workflow covers dataset inspection, patient-level splitting, training, validation-based model selection, a weighted-loss experiment, held-out evaluation, Grad-CAM, and error analysis. Its purpose is to compare architectures under a shared experimental setup. It is a research project, not a validated clinical diagnostic system.

## Dataset and preprocessing

The implementation expects **CheXpert-v1.0-small**, with image files and a training CSV containing `Path` and all observation columns. A separate validation CSV is optional. Dataset files are not included; obtain access under the dataset provider's terms through the [official CheXpert page](https://stanfordmlgroup.github.io/competitions/chexpert/).

**Targets:** No Finding, Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, and Support Devices.

| Step | Implemented behavior |
| --- | --- |
| Discovery | Scans `/kaggle/input` for compatible CSVs and resolves image paths |
| View selection | Keeps frontal views when `Frontal/Lateral` is present; otherwise retains all rows |
| Raw statistics | Counts positive, negative, uncertain, and blank labels before cleaning |
| Uncertain labels | U-Ones: maps `-1` to `1` for every observation |
| Missing labels | Maps blank/NaN to `0`; preserves positive `1` and negative `0` |
| Images | Converts to RGB and resizes to 224 × 224 pixels |
| Training augmentation | Rotation ±10°, random resized crop with scale 0.9–1.0, brightness/contrast jitter 0.1; no horizontal flip |
| Normalization | ImageNet mean `[0.485, 0.456, 0.406]`, standard deviation `[0.229, 0.224, 0.225]` |
| Validation/held-out transforms | Resize, tensor conversion, and normalization only |

Patient IDs are extracted from image paths. `GroupShuffleSplit` assigns approximately 90% of training patients to `dev-train` and 10% to `dev-val`; image proportions can differ. In the saved run, the official validation CSV was patient-disjoint and became the held-out set. Assertions confirmed zero patient overlap between all three splits. The code also supports carving a held-out split from training data if the official validation CSV is absent or overlaps; that fallback was not used here.

The input CSVs contained 223,414 training rows and 234 validation rows, each with 19 columns. Frontal filtering retained 191,027 training images from 64,534 patients.

| Evaluated split | Frontal images | Patients | Role |
| --- | ---: | ---: | --- |
| `dev-train` | 171,927 | 58,080 | Model fitting |
| `dev-val` | 19,100 | 6,454 | Checkpoint and architecture selection |
| Held-out official validation | 202 | 200 | Final evaluation |

## Models and training

All backbones use a trainable 14-output classification head. The optimizer receives all model parameters for fine-tuning.

| Architecture | Implementation / pretrained weights |
| --- | --- |
| ResNet50 | `torchvision`, `IMAGENET1K_V2` |
| DenseNet121 | `torchvision`, `IMAGENET1K_V1` |
| EfficientNet-B0 | `torchvision`, `IMAGENET1K_V1` |
| ViT-S/16 | `timm`: `vit_small_patch16_224`, pretrained |
| DINOv3 ViT-S/16 | `timm`: `vit_small_patch16_dinov3.lvd1689m`, pretrained |

| Setting | Value in code |
| --- | --- |
| Loss | Unweighted `BCEWithLogitsLoss` for the primary comparison |
| Optimizer | AdamW |
| CNN learning rate / weight decay | `1e-4` / `0.01` |
| Transformer learning rate / weight decay | `3e-5` / `0.05` |
| Schedule | Linear warmup over 5% of planned steps, then cosine decay with a minimum multiplier of `0.01` |
| Batch size / data-loader workers | 32 / 2 |
| Epoch budget | Up to 5 per model |
| Early stopping | Patience 2; AUROC improvement must exceed `1e-4` |
| Checkpoint selection | Best qualifying `dev-val` macro AUROC |
| Precision / device | CUDA automatic mixed precision and gradient scaling; `cuda:0` |
| Seed | 42 for Python, NumPy, and PyTorch |

The secondary experiment passes per-label positive weights, `negative_count / max(positive_count, 1)`, computed from cleaned `dev-train` labels. **In the recorded run it returned cached ResNet50 baseline metrics instead of training with weighted loss.** The identical displayed scores are not an independent ablation result. See [verification notes](docs/VERIFICATION.md#implementation-caveats).

The recorded Kaggle run used Python 3.12.13 (notebook metadata) and detected two Tesla T4 GPUs with 14.56 GB on the first device. The implementation uses only `cuda:0`. All five models ran five epochs; checkpoint selection retained epoch 4 for ResNet50 and DenseNet121 and epoch 5 for the others. The run is dated 2026-09-26 in execution metadata. Exact package versions were written to a runtime environment file that is not included in this repository.

## Evaluation and results

Architecture selection uses `dev-val`. The selected checkpoint is reloaded and frozen before one held-out evaluation call in Section 19. Re-running that cell repeats evaluation; there is no persistent one-time lock. This held-out set is the official validation set or the patient-disjoint fallback described above, not the official CheXpert challenge test set.

| Metric | Definition in the notebook |
| --- | --- |
| Macro AUROC (primary) | Mean per-label area under the ROC curve over labels with defined values |
| Macro AUPRC | Mean per-label **average precision**, computed with `average_precision_score`, over labels with defined values |
| Macro F1 | Mean per-label F1 at probability ≥ 0.5; zero division returns 0 |

For labels containing only one target class, the code records both AUROC and average precision as `"undefined"` and excludes them from those macro averages. F1 still includes all labels.

### Validation comparison

Values below reproduce the saved comparison table, using the selected checkpoints rather than each model's final epoch.

| Model | Macro AUROC | Macro AP (AUPRC) | Macro F1 | Parameters (M) | Selected epoch | Time to checkpoint (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ResNet50 | 0.8035 | 0.4805 | 0.3789 | 23.54 | 4 | 5006.5 |
| DenseNet121 | 0.8035 | 0.4817 | 0.3752 | 6.97 | 4 | 4229.0 |
| EfficientNet-B0 | 0.7969 | 0.4703 | 0.3560 | 4.03 | 5 | 5029.3 |
| ViT-S/16 | 0.7981 | 0.4714 | 0.3613 | 21.67 | 5 | 4891.2 |
| DINOv3 ViT-S/16 | 0.7629 | 0.4282 | 0.3059 | 21.59 | 5 | 5258.1 |

The notebook selected **ResNet50** from the comparison table. ResNet50 and DenseNet121 tie at the displayed four-decimal AUROC precision; this does not establish a clear performance advantage. DenseNet121's final epoch prints AUROC 0.8036, but did not exceed the checkpoint improvement tolerance, so the selected checkpoint remains epoch 4. Times above are the comparison table's elapsed times to the selected checkpoint, not necessarily complete training durations.

### Held-out ResNet50 evaluation

The output confirms that the trained ResNet50 epoch-4 checkpoint loaded before evaluation.

| Model | Images | Macro AUROC | Macro AP (AUPRC) | Macro F1 |
| --- | ---: | ---: | ---: | ---: |
| ResNet50, epoch 4 | 202 | 0.8159 | 0.5359 | 0.3476 |

AUROC and AP average **13 defined labels**; Fracture has no positive cases and is excluded. Macro F1 averages all 14 labels at threshold 0.5. These held-out averages therefore have different label coverage from a fully defined 14-label average.

| Observation | Positive cases | AUROC | AP (AUPRC) | F1 |
| --- | ---: | ---: | ---: | ---: |
| No Finding | 26 | 0.8982 | 0.4707 | 0.5200 |
| Enlarged Cardiomediastinum | 105 | 0.5011 | 0.5584 | 0.0189 |
| Cardiomegaly | 66 | 0.7717 | 0.7031 | 0.2597 |
| Lung Opacity | 117 | 0.9089 | 0.9357 | 0.7923 |
| Lung Lesion | 1 | 0.5274 | 0.0104 | 0.0000 |
| Edema | 42 | 0.9464 | 0.8275 | 0.7391 |
| Consolidation | 32 | 0.8676 | 0.4777 | 0.1143 |
| Pneumonia | 8 | 0.7197 | 0.1075 | 0.0000 |
| Atelectasis | 75 | 0.8018 | 0.6598 | 0.4174 |
| Pneumothorax | 7 | 0.9201 | 0.3904 | 0.3750 |
| Pleural Effusion | 64 | 0.9291 | 0.8571 | 0.7705 |
| Pleural Other | 1 | 0.8806 | 0.0400 | 0.0000 |
| Fracture | 0 | undefined | undefined | 0.0000 |
| Support Devices | 99 | 0.9342 | 0.9281 | 0.8586 |

These values are transcribed from saved notebook outputs, not recomputed from predictions. The small held-out sample and rare findings limit interpretation; no confidence intervals or repeated-seed estimates are reported.

### Execution scope

Saved outputs include label-distribution, training-curve, model-comparison, held-out AUROC, and ROC figures. Grad-CAM then failed at import with `ModuleNotFoundError: No module named 'pytorch_grad_cam'`. Error analysis and the artifact-inventory cell have null execution counts. The clinical discussion is explanatory prose, not observed error analysis. The [verification notes](docs/VERIFICATION.md) map all reported values to source cells.

## Run the notebook

### Recommended: Kaggle with a CUDA GPU

1. Import [chexpert.ipynb](chexpert.ipynb) into a Kaggle notebook.
2. Attach CheXpert-v1.0-small under `/kaggle/input`, including the CSV metadata and images. Keep the image directory layout referenced by the CSVs. Dataset discovery runs automatically.
3. Enable a GPU accelerator and Internet access. The notebook stops if CUDA is unavailable; it uses the first visible GPU, even if multiple GPUs are attached.
4. Install the packages in [requirements.txt](requirements.txt) into the notebook's Python environment before execution. For example, after uploading that file, use `%pip install -r /kaggle/working/requirements.txt` in a setup cell. Ensure `grad-cam` is installed in the active kernel: the updated notebook does not install it automatically, and its absence stopped the saved run at Grad-CAM.
5. For DINOv3, use a `timm` version that recognizes the exact model identifier above. Arrange any required weight access/license acceptance and add a Hugging Face token as a Kaggle Secret named `HF_TOKEN`. The loader also checks the `HF_TOKEN` environment variable. Loading failures stop execution with an error; there is no automatic model substitution.
6. Run cells in order. All five models are enabled in `RUN_MODELS`. Review the [known caveats](docs/VERIFICATION.md#implementation-caveats) when interpreting the weighted-loss experiment, warnings, and final evaluation.
7. Save a Kaggle version with outputs and download the generated artifacts before ending the session.

Dependencies cover PyTorch/torchvision, `timm`, NumPy, pandas, SciPy, scikit-learn, Matplotlib, seaborn, Pillow, `grad-cam` (imported as `pytorch_grad_cam`), and `huggingface_hub`. The [dependency file](requirements.txt) specifies lower bounds, not a locked environment; older allowed `timm` versions may lack DINOv3 support. The saved run successfully loaded the configured DINOv3 model. Package versions from the generated environment JSON are not available here.

### Local CUDA environment

Use a Python environment with CUDA-enabled PyTorch and torchvision matched to your GPU/driver using the [PyTorch installation guide](https://pytorch.org/get-started/locally/). Then, from the cloned repository:

```bash
git clone https://github.com/harikarthikv/chexpert.git
cd chexpert
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab chexpert.ipynb
```

JupyterLab is an additional notebook runner, not a dependency listed by the project. When `/kaggle/input` does not exist, dataset discovery searches the current directory, `./data`, `../data`, and an author-specific Windows path. Place your authorized dataset under `./data` with its CSV/image hierarchy intact and run from the repository directory. Local execution is not verified, and CPU execution is blocked. Outside Kaggle, generated artifacts go under `./chexpert-cv/` relative to the notebook's working directory.

## Repository and generated artifacts

| Tracked file | Purpose |
| --- | --- |
| [chexpert.ipynb](chexpert.ipynb) | Executed experiment with saved training/evaluation outputs and the Grad-CAM error |
| [requirements.txt](requirements.txt) | Declared Python dependency lower bounds |
| [README.md](README.md) | Overview, methodology, evidence status, and run instructions |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | Evidence sources, caveats, and result-publication checks |

On Kaggle, the notebook writes artifacts under `/kaggle/working`. Saved notebook messages confirm exports through held-out evaluation, but the separate runtime files are **not included in this repository**; figure images are embedded in notebook outputs:

| Output location | Contents |
| --- | --- |
| `results/` | Environment metadata, ordered label names, raw label statistics, validation metrics, comparison CSV, weighted-loss JSON, final held-out metrics |
| `checkpoints/` | Best state dictionary per model: `{model_key}_best.pt` |
| `figures/` | Label distribution, loss curves, model comparison, held-out per-label AUROC, and ROC curves were generated; Grad-CAM and the error gallery were not |

The held-out per-label table is displayed in the notebook; its metric dictionaries are saved in the held-out JSON. Raw prediction arrays are excluded from saved metric JSON files. Completed models are skipped when matching metrics and checkpoints already exist; this is cached-run reuse, not resumption of interrupted optimizer state.

## Limitations

- U-Ones applies the same uncertainty policy to every observation and introduces label noise; blank labels are treated as negative.
- Seed setting does not guarantee identical results: cuDNN benchmarking is enabled and deterministic execution is disabled.
- Unreadable images are replaced with gray images while retaining their labels. Review warnings and image paths before accepting results.
- CUDA out-of-memory handling skips the affected training batch. It does not reduce batch size or accumulate gradients, despite the notebook prose.
- If a selected checkpoint is missing, final evaluation can use an initialized classification head. Confirm that trained weights loaded before interpreting metrics.
- Saved outputs support the reported validation comparison and held-out metrics, but do not establish ablation benefits, clinical error patterns, or diagnostic utility. Further caveats are recorded in the [verification notes](docs/VERIFICATION.md).

## References

- Irvin et al. (2019), [CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison](https://arxiv.org/abs/1901.07031).
- [Stanford CheXpert dataset and benchmark](https://stanfordmlgroup.github.io/competitions/chexpert/).
- [PyTorch](https://pytorch.org/), [torchvision](https://docs.pytorch.org/vision/stable/index.html), and [timm](https://huggingface.co/docs/timm/index).
- [DINOv3](https://github.com/facebookresearch/dinov3) and [PyTorch Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam).
