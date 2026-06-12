# U-Net Spatial Baseline Report

## Wildfire Area Regression Project

## 1. Objective

This report documents the U-Net spatial baseline developed for Task 2 of the Wildfire Area Regression project.

The objective of this module is to predict the next-day wildfire fire mask using a four-day temporal input sequence. The model is designed as a spatial deep learning baseline that uses fire-mask, weather, and terrain information to estimate the next-day cumulative fire extent.

The U-Net model is one component of the broader Task 2 model-building stage, which includes multiple approaches such as XGBoost, U-Net, ConvLSTM, and Prithvi-based modeling.

---

## 2. Task Definition

The model input is a four-day sequence:

```text
X = [X(t-3), X(t-2), X(t-1), X(t)]
```

Each daily input contains:

```text
1 fire-mask channel
6 weather channels
3 terrain channels
```

Therefore, each day has 10 channels:

```text
1 + 6 + 3 = 10 channels
```

The four-day sequence is flattened into a 40-channel U-Net input:

```text
4 days × 10 channels = 40 channels
```

The model target is the next-day cumulative fire mask:

```text
Y = Fire Mask(t+1)
```

Input tensor shape:

```text
x: [40, 512, 512]
```

Target tensor shape:

```text
y_fire: [1, 512, 512]
```

The current U-Net baseline focuses on single-head fire-mask prediction. Burn severity prediction is not included in this baseline model, but the pipeline can be extended later if severity labels are prepared consistently for the same sample structure.

---

## 3. Data Source and Cache Format

The model uses the memory-mapped `.npy` cache prepared from the aligned wildfire dataset.

Expected cache structure:

```text
Dataset/
└── tensors_npy_cache/
    ├── train/
    ├── val/
    └── test/
```

Each fire event folder contains:

```text
fire_masks.npy
weather.npy
terrain.npy
```

The arrays have the following expected shapes:

```text
fire_masks.npy: [N, H, W]
weather.npy:    [N, 6, H, W]
terrain.npy:    [3, H, W]
```

Where:

```text
N = number of available days for one wildfire event
H = image height
W = image width
```

The model uses the official train, validation, and test folders already prepared from Task 1. No random fire-level split is created inside the U-Net pipeline.

---

## 4. Preprocessing Method

### 4.1 Temporal Windowing

For each wildfire event, the dataset creates temporal windows using four input days and one target day.

Example:

```text
Input:  days 0, 1, 2, 3
Target: day 4
```

Next sample:

```text
Input:  days 1, 2, 3, 4
Target: day 5
```

This produces multiple training samples from each fire event while preserving the original split.

---

### 4.2 Fire-Aware Cropping

Because the original wildfire scenes can be large, the model uses 512×512 spatial crops.

The dataset uses fire-aware crop selection based on the union of:

```text
current fire mask at day t
target fire mask at day t+1
```

For training samples, the crop is selected around fire pixels with controlled randomness. For validation and test samples, the crop is selected deterministically around the median fire-pixel location. This makes validation and test evaluation reproducible.

Samples where both the current fire mask and target fire mask are empty are removed from the filtered dataset. This gives a fire-aware patch-level dataset.

---

### 4.3 Normalization

Fire-mask channels remain binary:

```text
0 = background
1 = fire
```

Weather and terrain channels are normalized using train-set statistics.

The normalization statistics are saved in:

```text
Dataset/unet_spatial_workspace/reports/module14_normalization_stats.json
```

The normalized dataset class loads this file and applies channel-wise normalization to weather and terrain arrays.

---

## 5. Model Architecture

The implemented model is a single-head U-Net for binary fire-mask segmentation.

The architecture contains:

```text
Encoder blocks: enc1, enc2, enc3, enc4
Bottleneck block
Decoder blocks: up4, up3, up2, up1
Output head: fire_head
```

Model input:

```text
[B, 40, 512, 512]
```

Model output:

```text
[B, 1, 512, 512]
```

The output is raw logits. During evaluation and visualization, sigmoid activation is applied and a probability threshold is used to obtain a binary fire prediction.

Model size:

```text
Trainable parameters: 1,947,905
```

---

## 6. Loss Function and Metrics

The training loss combines Binary Cross-Entropy with Dice Loss:

```text
Loss = BCEWithLogitsLoss + Dice Loss
```

This is suitable for binary segmentation where fire pixels can be spatially imbalanced.

The following metrics are used:

```text
IoU
Precision
Recall
F1-score
Accuracy
```

For validation and test reporting, metrics are computed from accumulated pixel-level counts:

```text
TP, FP, FN, TN
```

---

## 7. Completed Medium-Scale Experiment

A medium-scale training run was completed to verify the full U-Net pipeline.

Run name:

```text
unet_fireonly_normalized_firefiltered_crop512_base16_medium_e3
```

Training setup:

```text
Train samples used: 200
Validation samples during training: 50
Epochs: 3
Crop size: 512
Base channels: 16
Batch size: 1
Loss: BCE + Dice
Optimizer: AdamW
```

The selected checkpoint was epoch 3.

A validation threshold sweep was performed, and the selected threshold was:

```text
0.65
```

---

## 8. Validation Result

The selected model was evaluated on the full filtered validation set.

Evaluation type:

```text
Filtered fire-aware patch-level validation
```

Validation samples:

```text
262
```

Validation result at threshold 0.65:

```text
IoU:       0.8123
Precision: 0.8916
Recall:    0.9014
F1-score:  0.8964
Accuracy:  0.9069
```

This validation result shows that the normalized U-Net spatial baseline can predict next-day cumulative fire-mask regions effectively on filtered fire-aware validation patches.

---

## 9. Test Result

The selected model was evaluated on the filtered test set using the validation-selected threshold of 0.65.

Evaluation type:

```text
Filtered fire-aware patch-level test evaluation
```

Test samples:

```text
260
```

Test result at threshold 0.65:

```text
IoU:       0.9057
Precision: 0.9765
Recall:    0.9259
F1-score:  0.9505
Accuracy:  0.9441
```

The test result indicates strong spatial agreement between predicted and ground-truth next-day fire masks for filtered fire-aware test patches.

---

## 10. Visualization Results

Prediction visualizations were generated for validation and test samples.

Each visualization contains:

```text
1. Input fire mask at current day t
2. Ground-truth fire mask at t+1
3. Predicted probability map
4. Prediction overlay
```

Overlay color meaning:

```text
Yellow = correct overlap between prediction and ground truth
Red    = false positive prediction
Green  = missed fire region
Black  = correct background
```

The visualization results show that the model performs strongly when active fire is already visible in the input sequence. In many stable fire-growth cases, the predicted mask closely overlaps with the next-day ground truth.

---

## 11. Observed Strengths

The U-Net spatial baseline performs well in the following cases:

```text
1. Active fire is already visible in the input sequence.
2. Fire shape changes gradually between day t and day t+1.
3. The fire region is spatially stable inside the crop.
4. The next-day cumulative fire mask is similar to the current fire mask with moderate growth.
```

In these cases, the model produces high-overlap predictions with strong IoU and F1-score.

---

## 12. Observed Limitations

The model has several important limitations:

```text
1. The current completed model was trained on a medium subset, not the full training set.
2. Evaluation is filtered fire-aware patch-level evaluation, not full-scene evaluation.
3. The model struggles with early ignition or first-appearance cases where current fire is absent but target fire appears.
4. Some difficult cases show underprediction or overprediction of fire extent.
5. Fire masks are cumulative, so many next-day targets are similar to current-day masks.
```

These limitations should be considered when comparing this U-Net baseline against other Task 2 models such as ConvLSTM, XGBoost, and Prithvi-based approaches.

---

## 13. Full Training Continuation Plan

The implemented pipeline is prepared for full-scale training using all available filtered training samples.

Available filtered dataset size:

```text
Train samples: 1310
Validation samples: 262
Test samples: 260
```

Recommended full training configuration:

```text
Run name: unet_fireonly_normalized_firefiltered_crop512_base16_full_e10
Epochs: 8–10
Batch size: 1
Crop size: 512
Base channels: 16
Optimizer: AdamW
Learning rate: 1e-4
Weight decay: 1e-4
Loss: BCE + Dice
Checkpoint saving: latest + best only
```

The full training script is:

```text
train_unet.py
```

After full training, the following steps should be completed:

```text
1. Run validation threshold sweep using evaluate_unet.py.
2. Select the final threshold using validation only.
3. Evaluate the selected full-training checkpoint on the test split.
4. Generate final validation and test prediction visualizations.
5. Update the final comparison table with U-Net full-training results.
```

---

## 14. Reproducible Execution Files

Main files:

```text
config.py
unet_dataset.py
unet_model.py
train_unet.py
evaluate_unet.py
predict_unet.py
compute_normalization_stats.py
```

Utility files:

```text
utils/losses.py
utils/metrics.py
utils/visualization.py
```

These scripts are designed to run using a dataset root configured by the environment variable:

```text
WILDFIRE_DATA_ROOT
```

Example:

```bash
export WILDFIRE_DATA_ROOT="/path/to/Dataset"
```

Windows PowerShell:

```powershell
$env:WILDFIRE_DATA_ROOT="C:\path\to\Dataset"
```

---

## 15. Summary

The U-Net spatial baseline pipeline has been implemented and validated successfully. The current medium-scale model demonstrates strong validation and test performance on filtered fire-aware patches. The dataset loader, normalization process, model architecture, training loop, evaluation scripts, and visualization pipeline are prepared for full-scale training.

The remaining continuation step is to run the provided full-training script on a stable GPU environment, followed by validation threshold selection and final test evaluation.
