# EO-SAR Change Detection using Siamese CNN + GNN

## Project Title & Description

This project implements an AI-based EO-SAR (Electro-Optical + Synthetic Aperture Radar) change detection pipeline for satellite imagery.

The objective is to identify regions of significant change between paired EO and SAR satellite images using a Siamese deep learning architecture combined with Graph Neural Network (GNN) refinement.

The repository contains:

- Dataset preprocessing and dataloaders
- Siamese CNN-based change detection model
- GNN refinement module
- Training and evaluation pipelines
- Metric computation utilities
- Prediction scripts
- Configuration-driven experimentation

The implementation is designed to be modular, reproducible, and easy to extend.

---

# Repository Structure

```bash
.
├── config.yaml
├── data_exploration.py
├── data_preprocessing.py
├── graph_generation.py
├── losses.py
├── metrices.py
├── model.py
├── requirements.txt
├── test.py
└── train.py
```

---

# Requirements

## Python Version

- Python 3.10+

## Dependencies

All dependencies are pinned in `requirements.txt`.

```txt
torch==2.2.0
torchvision==0.17.0
torch-geometric==2.5.1
rasterio==1.3.9
numpy==1.26.4
scikit-learn==1.4.1
opencv-python==4.9.0.80
matplotlib==3.8.3
tqdm==4.66.2
pyyaml==6.0.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Environment Setup

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate environment:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
```

Activate environment:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Dataset Structure

Expected dataset directory structure:

```bash
dataset/
├── train/
│   ├── images/
│   └── masks/
│
├── val/
│   ├── images/
│   └── masks/
│
└── test/
    ├── images/
    └── masks/
```

Update dataset paths inside `config.yaml`.

Example:

```yaml
train_dir: "path/to/train"
val_dir: "path/to/val"
test_dir: "path/to/test"
```

---

# Configuration File

All hyperparameters and experiment settings are stored in `config.yaml`.

The configuration file includes:

- Learning rate
- Batch size
- Number of epochs
- Optimizer
- Loss function
- Weight decay
- Image size
- Scheduler parameters
- Data augmentation settings
- Random seed
- Detection threshold
- GNN settings

Example:

```yaml
seed: 42
image_size: 256
batch_size: 8
epochs: 50
learning_rate: 0.0001
weight_decay: 0.0001
use_gnn: true
```

---

# Model Architecture

The proposed architecture combines Siamese feature extraction with Graph Neural Network refinement for EO-SAR change detection.

## Architecture Pipeline

```text
EO Image ---------------------> Siamese Encoder -----\
                                                       \
                                                        ---> Feature Fusion ---> GNN Refinement ---> Decoder ---> Change Map
                                                       /
SAR Image --------------------> Siamese Encoder -----/
```

## Components

### 1. Siamese Feature Extractor

- Two parallel branches process EO and SAR inputs independently
- Shared-weight CNN encoder extracts deep semantic features
- Captures modality-specific spatial information

### 2. Feature Fusion

- Features from EO and SAR branches are fused
- Difference-aware representations are generated
- Multi-scale contextual information is aggregated

### 3. Graph Neural Network Refinement

- Feature maps are converted into graph representations
- Nodes represent spatial regions/features
- Edges capture neighbourhood relationships
- GNN layers refine contextual feature understanding

### 4. Decoder Head

- Refined graph features are projected back to image space
- Upsampling layers generate dense pixel-wise predictions
- Final sigmoid activation produces binary change maps

## Key Features

- EO-SAR multimodal fusion
- Siamese deep feature learning
- Graph-based contextual reasoning
- Dynamic threshold optimisation
- Weighted focal-style loss for class imbalance handling

---

# Training

To train the model from scratch:

```bash
python train.py --config config.yaml
```

Training pipeline includes:

- Validation after every epoch
- Automatic metric computation
- Dynamic threshold optimisation
- Learning rate scheduling
- Gradient clipping
- Best checkpoint saving

---

# Evaluation

To evaluate on test data:

```bash
python test.py \
    --data_path /path/to/test \
    --weights /path/to/checkpoint.pth
```

For prediction/inference:

```bash
python predict.py \
    --weights /path/to/checkpoint.pth \
    --input /path/to/sample
```

---

# Model Weights

Download the final trained checkpoint from:

```text
Add Google Drive / HuggingFace link here
```

# Results

## Validation Metrics

| Metric | Score |
|---|---|
| Accuracy | Add value |
| Precision | Add value |
| Recall | Add value |
| F1 Score | Add value |
| IoU | Add value |

## Test Metrics

| Metric | Score |
|---|---|
| Accuracy | Add value |
| Precision | Add value |
| Recall | Add value |
| F1 Score | Add value |
| IoU | Add value |

---

# Reproducibility

To ensure reproducibility:
- Random seed is fixed in `config.yaml`
- All package versions are pinned
- Hyperparameters are centrally logged
- Training and evaluation commands are documented

---

# Citation / References

1. Siamese Neural Networks for Change Detection
2. Graph Neural Networks for Remote Sensing
3. U-Net: Convolutional Networks for Biomedical Image Segmentation
4. Focal Loss for Dense Object Detection

## Libraries / Frameworks

- PyTorch
- TorchVision
- PyTorch Geometric
- Rasterio
- OpenCV
- Scikit-learn

# Author
Tanya Mishra  
B.Tech Graduate — Artificial Intelligence & Machine Learning
