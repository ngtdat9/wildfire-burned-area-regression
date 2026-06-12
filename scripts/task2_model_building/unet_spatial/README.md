# U-Net Spatial Baseline

## Wildfire Area Regression Project

This folder contains the U-Net spatial baseline implementation for Task 2 of the Wildfire Area Regression project.

The model predicts the next-day cumulative wildfire fire mask from a four-day input sequence. It uses fire-mask, weather, and terrain information from the aligned wildfire dataset prepared in Task 1.

---

## 1. Model Task

The U-Net model uses four previous/current days as input:

```text
X = [X(t-3), X(t-2), X(t-1), X(t)]
```

The prediction target is:

```text
Y = Fire Mask(t+1)
```

Each daily input contains:

```text
1 fire-mask channel
6 weather channels
3 terrain channels
```

Therefore, each day contains 10 channels:

```text
1 + 6 + 3 = 10 channels
```

The four-day sequence is flattened into a 40-channel U-Net input:

```text
4 days × 10 channels = 40 channels
```

Input tensor:

```text
x: [40, 512, 512]
```

Target tensor:

```text
y_fire: [1, 512, 512]
```

The model outputs raw logits:

```text
fire_logits: [1, 512, 512]
```

A sigmoid activation and probability threshold are applied during evaluation and prediction visualization.

---

## 2. Folder Contents

```text
unet_spatial/
├── README.md
├── REPORT.md
├── config.py
├── compute_normalization_stats.py
├── debug_smoke_train.py
├── evaluate_unet.py
├── predict_unet.py
├── train_unet.py
├── unet_dataset.py
├── unet_model.py
└── utils/
    ├── losses.py
    ├── metrics.py
    └── visualization.py
```

### Main files

| File                             | Purpose                                                                                   |
| -------------------------------- | ----------------------------------------------------------------------------------------- |
| `config.py`                      | Central configuration for paths, hyperparameters, run names, and output folders           |
| `unet_dataset.py`                | Dataset loaders for temporal windows, fire-aware crops, fire filtering, and normalization |
| `unet_model.py`                  | U-Net model architecture                                                                  |
| `train_unet.py`                  | Full training script                                                                      |
| `evaluate_unet.py`               | Validation threshold sweep and fixed-threshold evaluation                                 |
| `predict_unet.py`                | Prediction visualization script                                                           |
| `compute_normalization_stats.py` | Computes weather and terrain normalization statistics                                     |
| `debug_smoke_train.py`           | Small debugging/smoke training script                                                     |
| `utils/losses.py`                | BCE + Dice loss                                                                           |
| `utils/metrics.py`               | IoU, precision, recall, F1-score, and accuracy                                            |
| `utils/visualization.py`         | Prediction and overlay visualization utilities                                            |
| `REPORT.md`                      | Project report and handover summary for this U-Net module                                 |

---

## 3. Required Dataset Structure

The scripts expect the following dataset structure:

```text
Dataset/
├── tensors_npy_cache/
│   ├── train/
│   ├── val/
│   └── test/
└── unet_spatial_workspace/
    └── reports/
        └── module14_normalization_stats.json
```

Each fire event folder inside `tensors_npy_cache` should contain:

```text
fire_masks.npy
weather.npy
terrain.npy
```

Expected array shapes:

```text
fire_masks.npy: [N, H, W]
weather.npy:    [N, 6, H, W]
terrain.npy:    [3, H, W]
```

Where:

```text
N = number of days for the fire event
H = image height
W = image width
```

The model uses the official `train`, `val`, and `test` folders. It does not create a new random split.

---

## 4. Environment Setup

### 4.1 Set Dataset Root

The code uses the environment variable:

```text
WILDFIRE_DATA_ROOT
```

This variable should point to the `Dataset` folder.

### Windows PowerShell

```powershell
$env:WILDFIRE_DATA_ROOT="C:\path\to\Dataset"
```

### Linux or macOS

```bash
export WILDFIRE_DATA_ROOT="/path/to/Dataset"
```

### Google Colab

```python
import os
os.environ["WILDFIRE_DATA_ROOT"] = "/content/drive/MyDrive/Wildfire_Area_Regression /Dataset"
```

If the environment variable is not set, the default path in `config.py` is:

```text
/content/drive/MyDrive/Wildfire_Area_Regression /Dataset
```

---

## 5. Python Requirements

The main dependencies are:

```text
python
numpy
pandas
torch
matplotlib
tqdm
```

Install with:

```bash
pip install numpy pandas torch matplotlib tqdm
```

For GPU training, install the correct PyTorch CUDA version for the target machine.

---

## 6. Normalization Statistics

Weather and terrain channels are normalized using train-set statistics.

The expected file is:

```text
Dataset/unet_spatial_workspace/reports/module14_normalization_stats.json
```

If this file already exists, it does not need to be recomputed.

To compute normalization statistics:

```bash
python scripts/task2_model_building/unet_spatial/compute_normalization_stats.py
```

With explicit paths:

```bash
python scripts/task2_model_building/unet_spatial/compute_normalization_stats.py \
  --cache-root "/path/to/Dataset/tensors_npy_cache" \
  --output-json "/path/to/Dataset/unet_spatial_workspace/reports/module14_normalization_stats.json"
```

---

## 7. Full Training

Recommended full training command:

```bash
python scripts/task2_model_building/unet_spatial/train_unet.py --require-cuda
```

The default full-training configuration is defined in `config.py`:

```text
Run name: unet_fireonly_normalized_firefiltered_crop512_base16_full_e10
Epochs: 10
Batch size: 1
Crop size: 512
Base channels: 16
Learning rate: 1e-4
Weight decay: 1e-4
Loss: BCE + Dice
Evaluation threshold during training: 0.65
Checkpoint saving: latest + best only
```

The script saves outputs to:

```text
Dataset/unet_spatial_workspace/unet_fireonly_normalized_firefiltered_crop512_base16_full_e10/
```

Expected output structure:

```text
unet_fireonly_normalized_firefiltered_crop512_base16_full_e10/
├── checkpoints/
│   ├── unet_fireonly_normalized_firefiltered_crop512_base16_full_e10_latest.pt
│   └── unet_fireonly_normalized_firefiltered_crop512_base16_full_e10_best.pt
├── logs/
│   └── full_training_log.csv
├── reports/
└── training_config.json
```

To resume from the latest checkpoint:

```bash
python scripts/task2_model_building/unet_spatial/train_unet.py --require-cuda --resume-latest
```

For a short smoke training run:

```bash
python scripts/task2_model_building/unet_spatial/train_unet.py \
  --limit-train-samples 2 \
  --limit-val-samples 2 \
  --epochs 1
```

---

## 8. Validation Threshold Sweep

Use validation data to select the final probability threshold.

```bash
python scripts/task2_model_building/unet_spatial/evaluate_unet.py \
  --checkpoint "/path/to/best_checkpoint.pt" \
  --output-dir "/path/to/output/eval_val" \
  --split val \
  --mode sweep
```

This produces:

```text
threshold_sweep_val.csv
threshold_sweep_val_summary.json
```

The selected threshold from the completed medium-scale run was:

```text
0.65
```

---

## 9. Test Evaluation

After choosing the threshold using validation only, evaluate on the test split using fixed-threshold mode.

```bash
python scripts/task2_model_building/unet_spatial/evaluate_unet.py \
  --checkpoint "/path/to/best_checkpoint.pt" \
  --output-dir "/path/to/output/eval_test" \
  --split test \
  --mode fixed \
  --threshold 0.65 \
  --save-sample-metrics
```

This produces:

```text
test_summary_threshold0.65.json
test_sample_metrics_threshold0.65.csv
```

The test split should not be used for threshold tuning.

---

## 10. Prediction Visualization

Generate prediction visualizations:

```bash
python scripts/task2_model_building/unet_spatial/predict_unet.py \
  --checkpoint "/path/to/best_checkpoint.pt" \
  --output-dir "/path/to/output/prediction_examples" \
  --split test \
  --threshold 0.65 \
  --num-examples 20
```

To visualize selected sample indices:

```bash
python scripts/task2_model_building/unet_spatial/predict_unet.py \
  --checkpoint "/path/to/best_checkpoint.pt" \
  --output-dir "/path/to/output/selected_prediction_examples" \
  --split test \
  --threshold 0.65 \
  --indices "223,224,221,232,84"
```

Each visualization includes:

```text
1. Input fire mask at current day t
2. Ground-truth fire mask at t+1
3. Predicted probability map
4. Prediction overlay
```

Overlay color meaning:

```text
Yellow = correct overlap
Red    = false positive
Green  = missed fire
Black  = correct background
```

---

## 11. Completed Medium-Scale Experiment

A medium-scale experiment was completed to validate the full U-Net pipeline.

Run name:

```text
unet_fireonly_normalized_firefiltered_crop512_base16_medium_e3
```

Training setup:

```text
Train samples used: 200
Validation samples during training: 50
Epochs: 3
Selected checkpoint: epoch 3
Selected threshold: 0.65
```

Full validation result on 262 filtered validation samples:

```text
IoU:       0.8123
Precision: 0.8916
Recall:    0.9014
F1-score:  0.8964
Accuracy:  0.9069
```

Test result on 260 filtered test samples:

```text
IoU:       0.9057
Precision: 0.9765
Recall:    0.9259
F1-score:  0.9505
Accuracy:  0.9441
```

These results are from filtered fire-aware patch-level evaluation, not full-scene evaluation.

---

## 12. Full Dataset Size

The filtered fire-aware dataset contains:

```text
Train samples: 1310
Validation samples: 262
Test samples: 260
```

The full training run should use all 1310 filtered training samples and validate on all 262 filtered validation samples.

---

## 13. Known Limitations

The U-Net spatial baseline has the following limitations:

```text
1. The completed medium-scale model was trained on a subset of the full training set.
2. Evaluation is filtered fire-aware patch-level evaluation, not full-scene evaluation.
3. The model performs best when current fire is already visible in the input sequence.
4. The model is weaker on early ignition or first-appearance cases.
5. Fire masks are cumulative, so many next-day targets are similar to current-day masks.
```

These limitations should be considered when comparing against XGBoost, ConvLSTM, and Prithvi-based models.

---

## 14. Files to Commit

Commit:

```text
README.md
REPORT.md
config.py
compute_normalization_stats.py
debug_smoke_train.py
evaluate_unet.py
predict_unet.py
train_unet.py
unet_dataset.py
unet_model.py
utils/losses.py
utils/metrics.py
utils/visualization.py
```

Do not commit:

```text
*.pt
*.npy
*.npz
prediction images
large logs
dataset folders
__pycache__
.ipynb_checkpoints
```

---

## 15. Suggested Execution Order

For a complete final run:

```text
1. Set WILDFIRE_DATA_ROOT.
2. Confirm tensors_npy_cache exists.
3. Confirm module14_normalization_stats.json exists.
4. Run train_unet.py for full training.
5. Run evaluate_unet.py on validation in sweep mode.
6. Select the best threshold from validation.
7. Run evaluate_unet.py on test in fixed mode.
8. Run predict_unet.py for final visualizations.
9. Update REPORT.md and comparison tables with full-training results.
```
