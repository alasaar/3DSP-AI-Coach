<div align="center">

# 3DSP AI Coach: Multimodal Biomechanical Analysis

**Phase 2**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Qwen2.5-VL](https://img.shields.io/badge/Qwen2.5--VL-7B--Instruct-orange)](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## Overview

This repository contains the AI Coaching pipeline built on top of the fine-tuned ViTPose-Base model. It implements a **Hybrid Neuro-Symbolic Architecture** to provide accurate, hallucination-free coaching feedback for athletes.

> **Note on Pose Estimation:** The core pose estimation model, its training pipeline, dataset preparation, and quantitative evaluation (45.6% error reduction) are maintained in the original research repository: **[3DSP-ViTPose](https://github.com/alasaar/3DSP-ViTPose)**. This repository focuses exclusively on the downstream biomechanical grounding and language generation features.

## Architecture

The system is divided into three distinct layers:

1. **Perception Layer (External):** 
   The fine-tuned ViTPose-Base (85.9M params) extracts 17 keypoints per frame. See the [original repo](https://github.com/alasaar/3DSP-ViTPose) for architecture details and training logs.

2. **Grounding Layer (`biomechanics.py`):** 
   A deterministic Python pipeline that smooths trajectories, calculates biomechanical joint angles, segments movement phases (approach, strike, follow-through), and applies rule-based technique checks. This layer outputs a structured JSON report, ensuring mathematical rigor.

3. **Language Layer (`vlm_coach.py`):** 
   Qwen2.5-VL-7B-Instruct (4-bit quantized) reads the structured JSON and a skeleton visualization to generate natural-language coaching feedback. Through strict prompt engineering, the VLM is constrained to *only* discuss pre-verified errors from the Grounding Layer, completely eliminating hallucinations.

## Repository Structure

```text
3DSP-AI-Coach/
|-- configs/, models/, utils/, data/   # Inherited from 3DSP-ViTPose (Perception Layer)
|-- train.py, evaluate.py              # Inherited training/evaluation scripts
|-- biomechanics.py                    # NEW: Angle calculation, phase segmentation, rule checks
|-- vlm_coach.py                       # NEW: Qwen2.5-VL-7B hallucination-proof inference
|-- app.py                             # NEW: Gradio UI for the professor demo
|-- requirements.txt                   # Updated with VLM and UI dependencies
+-- assets/                            # Pre-generated figures and sample skeleton
```

## Installation

```bash
# Clone the repository
git clone https://github.com/alasaar/3DSP-AI-Coach.git
cd 3DSP-AI-Coach

# Create environment and install dependencies
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the Gradio Demo
Launch the interactive web interface for the AI Coach:

```bash
python app.py
```

*Note: The current `app.py` uses mock trajectory data for rapid prototyping. In the final deployment, the `mock_inference()` function is replaced by the real-time ViTPose inference pipeline from the Perception Layer.*

### Run the VLM Coach Directly
To generate a coaching report from a saved JSON report and skeleton image:

```python
from vlm_coach import load_vlm, generate_coaching_report

model, processor = load_vlm()
report = generate_coaching_report(model, processor, "report.json", "skeleton.png")
print(report)
```
