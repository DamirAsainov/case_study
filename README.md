# Case Study: Image Classification for Medical Imaging

Code accompanying the case-study report `Case_Study_Report.docx`.

Three models of increasing capacity are trained and compared on two public
benchmarks from **MedMNIST v2** (Yang et al., *Scientific Data*, 2023):

| Dataset | Modality | Task | Images (train / val / test) |
|---|---|---|---|
| PneumoniaMNIST | Pediatric chest X-ray (Kermany et al., 2018) | normal vs. pneumonia | 4 708 / 524 / 624 |
| DermaMNIST | Dermatoscopy, HAM10000 (Tschandl et al., 2018) | 7 skin-lesion classes | 7 007 / 1 003 / 2 005 |

Models:
1. **PCA + logistic regression**: classical baseline on raw pixels
2. **Simple CNN**: 4 conv blocks trained from scratch
3. **ResNet-18 (TL)**: ImageNet-pretrained, fine-tuned end to end

Evaluation: AUC with 95% bootstrap CI, accuracy, balanced accuracy, macro-F1,
sensitivity/specificity/PPV/NPV (binary), decision-threshold tuning on the validation
set (Youden's J), expected calibration error (ECE) and Grad-CAM explanations.

## Project layout

```
run_experiments.py   entry point: trains everything, writes outputs/
build_report.py      builds Case_Study_Report.docx from outputs/results.json + figures
src/
  data.py            MedMNIST loading, augmentation, class weights
  models.py          SimpleCNN, ResNet18Transfer
  train.py           training loop (AdamW, one-cycle LR, AMP, best-val-AUC checkpoint)
  metrics.py         AUC + bootstrap CI, sens/spec, ECE, Youden threshold
  gradcam.py         Grad-CAM
  plots.py           all figures
outputs/
  results.json       all metrics and training histories
  figures/*.png      figures used in the report
  predictions/*.npz  test-set probabilities per model
  train_log.txt      per-epoch training log of the reported run
```

## How to run

```bash
pip install -r requirements.txt
python run_experiments.py --num-workers 8   # ~3-5 min on a data-centre GPU; downloads data to ./data
python build_report.py --author "Name Surname"   # regenerates the report
```

Options: `--size {28,64,128,224}` (image resolution, default 64), `--epochs-cnn 30`,
`--epochs-resnet 15`, `--seed 42`, `--datasets pneumoniamnist dermamnist`. On CPU it runs too, just slower.
Step-by-step instructions for a remote GPU server: [TRAINING.md](TRAINING.md).
