# SIIM-FISABIO-RSNA COVID-19 Detection — Kaggle Competition

## Overview

The [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/competitions/siim-covid19-detection) competition (2021) required both study-level classification (4 classes: negative, typical, indeterminate, atypical pneumonia) and image-level object detection (opacity bounding boxes) from chest X-rays.

Kaggle profile: [illidan7](https://www.kaggle.com/illidan7)

## Approach

### 1. Dual-Task Pipeline

Solved classification and detection as separate tracks, combined in the final submission:
- **Study-level**: Multi-model ensemble of EfficientNet-B7 (Keras), EfficientNetV2-L, Swin Transformer, EfficientNet-B3-NS, plus PyTorch Lightning models (ResNet50, EfficientNet variants)
- **Image-level**: YOLOv5x trained on opacity bounding boxes with Weighted Box Fusion (WBF)

### 2. DICOM Processing

Custom DICOM-to-image pipeline with VOI LUT correction, MONOCHROME1 inversion handling, and configurable resize.

### 3. Multi-Framework Training

Study classifiers trained across PyTorch (via PyTorch Lightning + timm) and TensorFlow/Keras, providing model diversity for ensembles. 5-fold stratified CV. W&B experiment tracking.

### 4. Stacking / Meta-Learning

Generated OOF predictions from 11+ models and trained a LightGBM meta-learner with Focal Loss for class imbalance.

### 5. Ensemble Optimization

Systematically compared averaging, weighted averaging, rank averaging across EfnB7/Swin/EffV2/Eff3. The final 4-model study ensemble + YOLOv5 detection achieved the best result.

## Repository Structure

```
├── eda/
│   └── competition-scouting.ipynb                 # Strategy sketch + key resource links
├── data-prep/
│   └── dicom-to-jpg-conversion.ipynb              # DICOM → JPG with VOI LUT correction
├── training/
│   ├── study-classifier-resnet50.ipynb            # ResNet50 + PyTorch Lightning (v13)
│   ├── study-classifier-efficientnet-b6.ipynb     # EfficientNet-B6-NS, 5-fold (timm)
│   ├── object-detection-yolov5x.ipynb             # YOLOv5x opacity detection
│   └── meta-learner-lgbm-stacking.ipynb           # LightGBM meta-learner + Focal Loss
├── inference/
│   ├── combined-pipeline-frankenstein.ipynb        # First combined study+detection pipeline
│   ├── bronze-lungrush-submission.ipynb           # Bronze medal pipeline
│   ├── bronze-lungrush-stage2.ipynb               # Stage 2 with EffV2-L added
│   └── final-submission-4models.ipynb             # ★ Final: 4-model study + YOLOv5
├── ensemble/
│   ├── ensemble-weight-optimization.ipynb         # mAP comparison across strategies
│   └── oof-predictions-generation.ipynb           # OOF from 11 models for stacking
└── utils/
    └── lightning-model-definitions.py             # Shared PyTorch Lightning modules
```


## Kaggle Notebooks

Key notebooks published on Kaggle ([illidan7](https://www.kaggle.com/illidan7)):

- [SIIM-COVID19-BronzeLungRushStage2 (Study)](https://www.kaggle.com/code/illidan7/siim-covid19-bronzelungrushstage2-study) (2 upvotes)
- [SIIM-COVID19-Illidefont (Full)](https://www.kaggle.com/code/illidan7/siim-covid19-illidefont-full) (2 upvotes)
- [SIIM-COVID19-BronzeLungRush (Experiment)](https://www.kaggle.com/code/illidan7/siim-covid19-bronzelungrush-experiment)
- [SIIM-COVID19-BronzeLungRush (Submit)](https://www.kaggle.com/code/illidan7/siim-covid19-bronzelungrush-submit)
- [SIIM-COVID19-BronzeLungRushStage2 (Image)](https://www.kaggle.com/code/illidan7/siim-covid19-bronzelungrushstage2-image)
- [SIIM-COVID19-BronzeLungRushStage2](https://www.kaggle.com/code/illidan7/siim-covid19-bronzelungrushstage2)
- [SIIM-COVID19-EnsembleOpt](https://www.kaggle.com/code/illidan7/siim-covid19-ensembleopt)
- [SIIM-COVID19-Illidefont-Final (2models)](https://www.kaggle.com/code/illidan7/siim-covid19-illidefont-final-2models)
- [SIIM-COVID19-Illidefont-Final (4models)](https://www.kaggle.com/code/illidan7/siim-covid19-illidefont-final-4models)
- [SIIM-COVID19-Illidefont (Study)](https://www.kaggle.com/code/illidan7/siim-covid19-illidefont-study)
- [SIIM-COVID19-LitModels (Infer)](https://www.kaggle.com/code/illidan7/siim-covid19-litmodels-infer)
- [SIIM-COVID19-Strat1 (Infer)](https://www.kaggle.com/code/illidan7/siim-covid19-strat1-infer)
## Tech Stack

- **Classification**: EfficientNet-B7 (Keras), EfficientNetV2-L/S, Swin Transformer, ResNet50 (timm + PyTorch Lightning)
- **Detection**: YOLOv5x, Weighted Box Fusion (WBF)
- **Meta-Learning**: LightGBM with Focal Loss
- **Medical Imaging**: pydicom, DICOM VOI LUT processing
- **Experiment Tracking**: Weights & Biases
- **Infrastructure**: Kaggle Notebooks (GPU), Google Colab

## Competition

- **Name**: [SIIM-FISABIO-RSNA COVID-19 Detection](https://www.kaggle.com/competitions/siim-covid19-detection)
- **Type**: Classification + Object Detection
- **Metric**: Study-level mAP + Image-level mAP
- **Timeline**: May — August 2021
