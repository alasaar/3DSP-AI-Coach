<div align="center">

# 3DSP-ViTPose: Fine-Tuned Vision Transformer for 3D Shot Posture Estimation

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![Parameters](https://img.shields.io/badge/Parameters-85.9M-orange)](https://huggingface.co/usyd-community/vitpose-base-simple)
[![GPU](https://img.shields.io/badge/GPU-NVIDIA%20Tesla%20T4-76B900?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/tesla-t4/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**45.6% reduction in mean keypoint error** over the COCO-pretrained baseline by
fine-tuning ViTPose-Base (85.9M params) on a mixed 3DSP + CrowdPose corpus.

</div>

---

## Overview

This project fine-tunes **ViTPose-Base-Simple** on the domain-specific
**3D-Shot-Posture Dataset (3DSP)** -- 200 film shots, 20 frames each --
supplemented with **23,937** filtered in-the-wild samples from **CrowdPose**.
A 3x oversampling strategy balances the scarce domain data against the larger
general corpus. After 12 epochs on a Tesla T4, the model achieves a mean L2
error of **3.59 px** (baseline: 6.59 px) and a median of **2.05 px**
(baseline: 3.41 px) across 400 held-out frames and 12 exactly-matched joints.

---

## Architecture

```
INPUT (192 x 256 x 3, ImageNet-normalised)
        |
   Patch Embedding (16x16) --> 192 tokens
        |
   ViT-Base Encoder (12 layers, 12 heads, d=768)
        |
   Reshape --> (B, 768, 12, 16)
        |
   Simple Decoupled Head (4 conv layers + 1x1 conv)
        |
   17 Heatmaps (48 x 64) --> Soft-Argmax (tau=100) --> 17 Keypoints
```

| Component | Detail |
|-----------|--------|
| Backbone | ViT-Base: 12 blocks, 12 heads, d_model = 768 |
| Input | 192 x 256, ImageNet mean/std |
| Output | 17 heatmaps at 48 x 64 |
| Parameters | 85.9M |
| Decoding | Soft-argmax, tau = 100 |

Before fine-tuning, the 1x1 conv head is re-initialised (N(0, 0.01), zero
bias) to remove COCO-layout bias. The 17-keypoint 3DSP layout comprises the
limb joints (hips, knees, ankles: 1-6; shoulders, elbows, wrists: 11-16) plus
midline joints (pelvis, spine, shoulder centre, neck, head: 0, 7, 8, 9, 10),
three of which are derived by spatial averaging during data preparation.

---

## Datasets

| Dataset | Samples | Role |
|---------|---------|------|
| 3DSP (train) | 3,600 frames x 3 oversample = 10,800 | Domain-specific |
| CrowdPose (filtered) | 23,937 | In-the-wild generalisation |
| 3DSP (val) | 400 frames (20 shots) | Held-out evaluation |

CrowdPose 14-keypoint annotations are mapped to the 3DSP 17-keypoint layout;
pelvis, spine and shoulder centre are derived via spatial averaging of visible
source joints. Filtering: iscrowd = 0, >= 8 visible keypoints,
bbox >= 40x60 px. Augmentation: horizontal flip (p = 0.5) with left/right
joint-pair permutation.

---

## Training

| Hyperparameter | Value |
|---------------|-------|
| Optimiser | AdamW (lr = 1e-5, wd = 1e-4) |
| Schedule | 1-epoch linear warmup + cosine decay to 0.1x |
| Epochs / Batch | 12 / 8 |
| Loss | Visibility-masked MSE on Gaussian heatmaps (sigma = 2.0) |
| Gradient clip | 1.0 |
| Hardware | NVIDIA Tesla T4, 15.6 GB |
| Duration | ~4.5 hours |

Best validation error: **3.04 px** at Epoch 3.

---

## Results

**400 validation frames, 12 exact joints (limbs and shoulders):**

| Model | Mean L2 (px) | Median L2 (px) |
|-------|:------------:|:--------------:|
| COCO-pretrained baseline | 6.59 | 3.41 |
| **Fine-tuned (ours)** | **3.59** | **2.05** |
| **Improvement** | **-45.6%** | **-39.9%** |

Per-joint reductions range from 29% (hips) to 53% (left shoulder); no
evaluated joint regresses after fine-tuning.

### Training Curves

![Training Curves](assets/training_curves.png)

*Training loss decreases monotonically over 12 epochs; validation error
reaches its minimum of 3.04 px at epoch 3 (star), after which it plateaus.*

### Per-Joint Error

![Per-Joint Error](assets/per_joint_error.png)

*Mean L2 error per joint over 400 validation frames. The green badges report
the error reduction of the fine-tuned model relative to the COCO-pretrained
baseline for each joint.*

### Qualitative Comparison (GT / Baseline / Fine-tuned)

![Qualitative](assets/qualitative_results.png)

*Person-cropped held-out frames. The mean per-frame error over the 12 exact
joints is printed above each prediction; the fine-tuned model tracks the
ground-truth skeleton closely while the baseline shows clear joint
displacement (e.g. 10.9 px vs 2.0 px in row 2).*

### Predicted Heatmaps

![Heatmaps](assets/heatmap_vis.png)

*Inferno heatmaps for four representative joints (head, left wrist, right
ankle, pelvis), cropped to the annotated person. Each distribution shows a
single sharp, well-localised peak.*

---

## Repository Structure

```
3DSP-ViTPose/
|-- configs/default.py          All hyperparameters, keypoint definitions,
|                               and visualisation palette
|-- data/
|   |-- download.py             Clone 3DSP + download CrowdPose
|   |-- crowdpose_preprocess.py Filter and remap CrowdPose to 17-kp
|   +-- dataset.py              MixedPoseDataset (PyTorch)
|-- models/vitpose.py           Model factory + checkpoint loading
|-- utils/
|   |-- keypoints.py            COCO-to-3DSP remapping
|   +-- heatmap.py              Gaussian targets + soft-argmax
|-- train.py                    Training loop
|-- evaluate.py                 Baseline vs fine-tuned evaluation
|-- visualize.py                Figure generation (reproduces assets/)
|-- assets/                     Pre-generated plots (see above)
|-- scripts/run_all.sh          End-to-end pipeline
+-- requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/alasaar/3DSP-ViTPose.git
cd 3DSP-ViTPose
pip install -r requirements.txt

# Download datasets
python -m data.download

# Train (~4.5 h on T4)
python train.py

# Evaluate
python evaluate.py --checkpoint checkpoints/vitpose_B_mix_best.pt

# Regenerate all figures
python visualize.py --checkpoint checkpoints/vitpose_B_mix_best.pt
```

### Model Weights

The fine-tuned checkpoint (~344 MB) exceeds GitHub's repository file limit and
is hosted in GitHub Releases:

```bash
mkdir -p checkpoints
wget -O checkpoints/vitpose_B_mix_best.pt \
  "https://github.com/alasaar/3DSP-ViTPose/releases/download/v1.0.0/vitpose_B_mix_best.pt"
```