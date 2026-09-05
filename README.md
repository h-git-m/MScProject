# MSc Project
Use of ROV for Autonomous Detection of an Underwater Docking Station Using Sonar Data by Improving a ML Detection Model Generalization Ability
# Autonomous FLS-Based Docking Station Detection — Improving ML Model Cross-Sensor Generalization

<div align="center">

![NTNU](https://img.shields.io/badge/NTNU-Marine%20Cybernetics-00529B?style=for-the-badge)
![MIR](https://img.shields.io/badge/MIR-Erasmus%20Mundus-003399?style=for-the-badge)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Ultralytics-FF6B00?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**MSc Thesis | Department of Marine Technology | Norwegian University of Science and Technology (NTNU)**

*Supervised by Prof. Martin Ludvigsen | Spring 2026*

</div>

---

> **Can a sonar detection model trained on one sensor reliably detect the same object on a completely different sensor — without changing the architecture?**
>
> This thesis answers yes — through deliberate multi-domain dataset construction alone.

---

## Table of Contents

- [Overview](#overview)
- [The Problem: Acoustic Domain Shift](#the-problem-acoustic-domain-shift)
- [Approach](#approach)
- [Hardware and Field Setup](#hardware-and-field-setup)
- [Datasets](#datasets)
- [Model](#model)
- [Results](#results)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [How to Use](#how-to-use)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

---

## Overview

This repository contains the code, dataset documentation, and training pipeline for my MSc thesis:

**"Use of ROV for Autonomous Detection of an Underwater Docking Station Using Sonar Data by Improving a ML Detection Model Generalization Ability"**

The thesis investigates whether a **data-centric, multi-domain training strategy** — combining sonar records from two acoustically distinct FLS sensors into a single balanced dataset — can substantially improve a deep learning detection model's ability to generalise across sonar types, without any architectural modification.

The case study is the autonomous detection of a **seafloor resident docking station** using two operationally deployed forward-looking sonar (FLS) systems at two NTNU Oceanlab sites.

<!-- 📸 SUGGESTED VISUAL: Banner image showing side-by-side sonar frames of the docking station
     from the Oculus M750d (left) and Norbit WBMS (right) — demonstrating the acoustic
     domain shift visually. Caption: "Same docking station. Different sonar. Completely
     different acoustic signature." -->

---

## The Problem: Acoustic Domain Shift

Forward-looking sonar imagery is fundamentally different from optical imagery. Targets appear differently depending on:

- **Operating frequency** (1.2 MHz vs 400 kHz)
- **Beam geometry** and transducer design
- **Deployment depth** and environmental conditions
- **Aspect angle** and sensor-to-target distance

A model trained on one sonar system **consistently degrades** when deployed on a different one — a phenomenon known as **acoustic domain shift**.

<!-- 📸 SUGGESTED VISUAL: Short GIF or 2-panel image showing detection bounding boxes
     on Oculus M750d test data (clean detections) vs Norbit WBMS test data under zero-shot
     baseline (missed detections, false positives). This is the single most compelling
     visual in the entire repository. -->

| Approach | Sensor | mAP@50 | mAP@50:95 |
|---|---|---|---|
| Zero-shot baseline | Norbit WBMS (unseen domain) | 0.518 | 0.194 |
| **Mix-dataset (proposed)** | **Norbit WBMS (unseen domain)** | **0.942** | **0.633** |
| Zero-shot baseline | Oculus M750d (primary domain) | — | — |
| Mix-dataset | Oculus M750d (primary domain) | Comparable | Comparable |

This problem is not unique to this thesis. Zhao et al. (2025) documented it systematically: YOLOv8n and their custom UWS-YOLO model, trained on UATD and tested on MDFLS in a zero-shot protocol, achieved mAP@50 of only 72.1% and 76.4% respectively. The research community has largely responded with **architectural solutions**. This thesis proposes a **data-centric solution**.

---

## Approach

Two training strategies are systematically compared:

### Zero-Shot Approach (Baseline)
Trains YOLOv8n exclusively on **Oculus M750d** records and evaluates directly on **Norbit WBMS** test data — replicating the cross-sensor deployment scenario with no target-domain training instances.

### Mix-Dataset Approach (Proposed)
Trains the same YOLOv8n architecture on a **balanced combination** of Oculus M750d and Norbit WBMS records for the same OOI — exposing the model simultaneously to both acoustic domains during training.

Both approaches are evaluated in:
- **Trained-from-scratch** configuration
- **COCO-pretrained fine-tuned** configuration

<!-- 📸 SUGGESTED VISUAL: Simple diagram showing the two training pipelines side by side:
     Zero-shot (Oculus only → Norbit test, fails) vs Mix-dataset (Oculus + Norbit →
     both tests, succeeds). A clean flowchart works well here. -->

---

## Hardware and Field Setup

### Sonar Systems

| Parameter | Oculus M750d | Norbit WBMS |
|---|---|---|
| **Role** | Primary sonar domain | Target sonar domain |
| **Frequency** | 1.2 MHz | 400 kHz |
| **Carrier** | Blueye X3 ROV | Minerva II working-class ROV |
| **Site** | TBS pier, ~9 m depth | Subsea facility, ~90 m depth |
| **Location** | Trondheimsfjorden nearshore | Trondheimsfjorden floor |

### Objects of Interest (OOI)

| Class | Role in Study |
|---|---|
| **Docking station** | Primary OOI — target of autonomous detection task |
| **Tires** | Secondary class — evaluates shape discrimination capability |
| **Pig Loop Module (PLM)** | Secondary class — evaluates discrimination between morphologically similar rectangular structures |

<!-- 📸 SUGGESTED VISUAL: Photo of the Blueye X3 ROV being deployed from the TBS pier
     (field photo), alongside a screenshot of the Oculus M750d sonar feed showing the
     docking station in frame. Field photos ground the work in physical reality and
     immediately distinguish this from simulation-only research. -->

<!-- 📸 SUGGESTED VISUAL: 2×4 mosaic of Oculus M750d sonar records at close, medium,
     and far range showing the docking station at different aspects — demonstrates the
     intra-class acoustic variability the model must handle. -->

---

## Datasets

Five structured datasets are constructed from field-collected, manually annotated sonar records:

### Training Datasets

| Dataset | Sensors | Docking Station | Tires | PLM | Total Instances |
|---|---|---|---|---|---|
| **Oculus** | Oculus M750d only | 1,081 | 1,081 | — | 2,162 |
| **Mini Oculus** | Oculus M750d only | ~540 | ~540 | — | ~1,080 |
| **Mix** | Oculus M750d + Norbit WBMS | 539 + 539 | 1,081 | 1,078 | ~3,237 |

### Test Datasets (Fixed Held-Out)

| Dataset | Sensor | Docking Station | Tires | PLM |
|---|---|---|---|---|
| **Oculus M750d Test** | Oculus M750d | 565 | 671 | — |
| **Norbit WBMS Test** | Norbit WBMS | 561 | — | 580 |

All images are annotated in **YOLO format** (bounding boxes, class labels). Annotation was performed manually using LabelImg.

> **Note:** Raw sonar data and full annotated datasets are available upon reasonable request for academic purposes. Please open an issue or contact via email.

---

## Model

**YOLOv8n** (Ultralytics) is selected as the detection backbone on the basis of:

- Superior performance-to-model-size trade-off across **six benchmark comparisons** on the UATD and MDFLS public sonar datasets
- Real-time inference capability exceeding AUV-relevant FLS frame rates
- Reduced overfitting risk on inherently scarce sonar training data (~3M parameters)

### Training Configuration

| Parameter | Value |
|---|---|
| Input resolution | 640 × 640 |
| Batch size | 16 |
| Epochs | 100 |
| Optimiser | AdamW (auto-selected) |
| Learning rate | 1 × 10⁻² with cosine annealing |
| Weight decay | 5 × 10⁻⁴ |
| Loss weights | box: 7.5 / cls: 0.5 / dfl: 1.5 |
| Hardware | Tesla P100-PCIE-16GB (NTNU HPC) |
| Pretraining | From scratch + COCO-pretrained (both evaluated) |

---

## Results

<!-- 🎥 SUGGESTED VISUAL: Embed the 1-minute detection results video here.
     Top half: Oculus M750d detections (primary domain).
     Bottom half: Norbit WBMS detections (target domain).
     This is the centrepiece visual of the entire repository — host on YouTube
     and embed with a thumbnail. -->

[![Detection Results Video](https://img.shields.io/badge/▶%20Watch-Detection%20Results%20Video-red?style=for-the-badge&logo=youtube)](YOUR_VIDEO_LINK_HERE)

### Docking Station Detection — Target Domain (Norbit WBMS Test Dataset)

| Model | TP | FP | FN | Precision | Recall | mAP@50 | mAP@50:95 | F1 |
|---|---|---|---|---|---|---|---|---|
| Zero-shot (scratch) | 201 | 201 | 360 | 0.50 | 0.36 | 0.429 | 0.142 | 0.42 |
| Zero-shot (fine-tuned) | 232 | 53 | 329 | 0.81 | 0.41 | 0.518 | 0.194 | 0.55 |
| **Mix-dataset (scratch)** | **489** | **28** | **72** | **0.95** | **0.87** | **0.942** | **0.633** | **0.91** |
| Mix-dataset (fine-tuned) | 470 | 56 | 91 | 0.89 | 0.84 | 0.909 | 0.584 | 0.87 |

### Docking Station Detection — Primary Domain (Oculus M750d Test Dataset)

| Model | TP | FP | FN | mAP@50 | mAP@50:95 |
|---|---|---|---|---|---|
| Zero-shot (fine-tuned) | 493 | 13 | 72 | — | — |
| Mix-dataset (scratch) | 468 | 18 | 97 | Comparable | Comparable |

<!-- 📸 SUGGESTED VISUAL: 2×2 grid of bounding box detection outputs:
     Row 1: Zero-shot model on Oculus M750d test (good) | Zero-shot on Norbit WBMS (poor)
     Row 2: Mix-dataset on Oculus M750d test (good) | Mix-dataset on Norbit WBMS (good)
     This single visual tells the entire story of the thesis. -->

---

## Key Findings

1. **Domain diversification outperforms dataset enlargement.** The Mix-dataset approach achieves a 82% relative mAP@50 gain on the target sonar domain using only half the primary-domain docking station training instances compared to the baseline — confirming that the improvement stems from domain coverage, not data volume.

2. **The fix requires no architectural modification.** The same YOLOv8n model, same hyperparameters, same loss configuration — only the training data composition changes.

3. **State-of-the-art benchmark exceeded.** The achieved mAP@50 of 94.2% substantially exceeds the 76.4% reported by UWS-YOLO (Zhao et al., 2025) in an analogous zero-shot cross-dataset generalization test.

4. **Near-shore data collection is a scalable strategy.** Docking station instances collected at shallow depth with an affordable small ROV (Blueye X3) can serve as a cost-effective, physically realistic source of complementary multi-domain training data — reducing reliance on expensive deep-water working-class ROV deployments.

5. **COCO pretraining provides mixed benefits.** Fine-tuned models suppress false positives more effectively in the primary domain; trained-from-scratch models exhibit lower false positive rates in the target domain under the Mix-dataset approach.

---

## Repository Structure

```
📦 fls-docking-detection
├── 📁 datasets/
│   ├── 📁 oculus/                  # Oculus M750d annotated images (YOLO format)
│   ├── 📁 mini_oculus/             # Reduced Oculus dataset
│   ├── 📁 mix/                     # Combined Oculus + Norbit dataset
│   ├── 📁 test_oculus/             # Fixed Oculus M750d test set
│   └── 📁 test_norbit/             # Fixed Norbit WBMS test set
├── 📁 training/
│   ├── train_scratch.py            # Train YOLOv8n from scratch
│   ├── train_finetune.py           # Fine-tune from COCO-pretrained weights
│   └── configs/
│       ├── oculus.yaml             # Oculus dataset config
│       ├── mini_oculus.yaml        # Mini Oculus dataset config
│       └── mix.yaml                # Mix dataset config
├── 📁 evaluation/
│   ├── evaluate.py                 # Run evaluation on test datasets
│   └── metrics.py                  # TP / FP / FN / mAP / F1 computation
├── 📁 results/
│   ├── 📁 weights/                 # Trained model weights (.pt)
│   ├── 📁 predictions/             # Bounding box detection outputs
│   └── 📁 plots/                   # Precision-Recall curves, confusion matrices
├── 📁 notebooks/
│   ├── dataset_analysis.ipynb      # Class balance and dataset statistics
│   └── results_visualization.ipynb # Detection result visualizations
├── requirements.txt
└── README.md
```

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/fls-docking-detection.git
cd fls-docking-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Datasets

Place your annotated sonar images in `datasets/` following the YOLO directory structure:

```
datasets/mix/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### 4. Train — Zero-Shot Approach

```bash
# From scratch
python training/train_scratch.py --data configs/oculus.yaml --epochs 100

# Fine-tuned from COCO
python training/train_finetune.py --data configs/oculus.yaml --epochs 100 --pretrained
```

### 5. Train — Mix-Dataset Approach

```bash
# From scratch
python training/train_scratch.py --data configs/mix.yaml --epochs 100

# Fine-tuned from COCO
python training/train_finetune.py --data configs/mix.yaml --epochs 100 --pretrained
```

### 6. Evaluate

```bash
python evaluation/evaluate.py \
    --weights results/weights/mix_scratch_best.pt \
    --data configs/test_norbit.yaml \
    --iou-threshold 0.5
```

---

## Dependencies

```
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
matplotlib>=3.7.0
PyYAML>=6.0
tqdm>=4.65.0
```

---

## Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{hassan2026fls,
  author       = {Mahmoud Hussein Abdelrazik Hassan},
  title        = {Use of ROV for Autonomous Detection of an Underwater
                  Docking Station Using Sonar Data by Improving a ML
                  Detection Model Generalization Ability},
  school       = {Norwegian University of Science and Technology (NTNU)},
  year         = {2026},
  type         = {MSc Thesis},
  department   = {Department of Marine Technology},
  address      = {Trondheim, Norway}
}
```

---

## Acknowledgements

This work was carried out at the **AUR-Lab, Department of Marine Technology, NTNU**, under the supervision of **Prof. Martin Ludvigsen**.

Field operations were supported by **Ambjørn Waldum** and **Leonard Günzel** (ROV deployment and crane handling), and by the engineers at **Trondhjem Biological Station (TBS)** (nearshore survey support).

This thesis was completed as part of the **MIR Erasmus Mundus Joint Master's Programme**, in collaboration with **Université de Toulon (UTLN)**, France.

---

## Contact

**Mahmoud Hassan**
MSc Marine Cybernetics — NTNU | MIR Erasmus Mundus

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin)](YOUR_LINKEDIN_URL)
[![Email](https://img.shields.io/badge/Email-Contact-D14836?style=flat&logo=gmail)](mailto:YOUR_EMAIL)

---

<div align="center">
<sub>Built with sonar data, field engineering, and too many early mornings on a Norwegian pier.</sub>
</div>
