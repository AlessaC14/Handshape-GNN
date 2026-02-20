# Handshape-GNN: Graph Neural Networks for Sign Language Handshape Recognition

Official implementation of [**"Improving Handshape Representations for Sign Language Processing: A Graph Neural Network Approach"**] (EMNLP 2025).
https://aclanthology.org/2025.emnlp-main.1483/

[Alessa Carbo](mailto:acarbol1@jh.edu) and [Eric Nalisnick](mailto:nalisnick@jhu.edu)
Johns Hopkins University

## Overview

This repository contains code for recognizing handshapes in American Sign Language (ASL) using a novel dual GNN architecture that separates temporal dynamics from static configurations.

### Key Contributions

1. **Dual GNN Architecture**:
   - **SignGNN**: Captures temporal evolution across signing sequences
   - **HandshapeGNN**: Focuses on static handshape configurations in low-motion frames

2. **Triple-Stream Classifier**: Combines sign embeddings, handshape embeddings, and raw landmarks for robust classification

3. **First Benchmark**: Establishes the first dedicated benchmark for handshape recognition as a standalone task

4. **Results**: Achieves **46% accuracy** across 37 handshape classes (baseline on feedforwar network: 25%)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Sign Sequence                      │
│              (MediaPipe 21-point hand landmarks)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
┌──────────▼────────────┐  ┌──────▼──────────────┐
│   SignGNN (Temporal)  │  │ HandshapeGNN (Static│
│                       │  │                      │
│  • Spatial + Temporal │  │  • Spatial edges only│
│    edges              │  │  • Low-motion frames │
│  • Contrastive loss   │  │  • Contrastive loss  │
│    (sign labels)      │  │    (handshape labels)│
│                       │  │                      │
│  Output: 32-dim       │  │  Output: 32-dim      │
│  sign embeddings      │  │  handshape embeddings│
└──────────┬────────────┘  └──────┬───────────────┘
           │                      │
           └───────────┬──────────┘
                       │
           ┌───────────▼────────────────┐
           │  Triple-Stream Classifier  │
           │                            │
           │  • Sign embeddings (32-dim)│
           │  • Handshape embeddings    │
           │    (32-dim)                │
           │  • Raw landmarks (63-dim)  │
           │                            │
           │  Combined: 3x64 → 64 → 37  │
           └────────────┬───────────────┘
                        │
              ┌─────────▼──────────┐
              │ Handshape Classes   │
              │     (37 total)      │
              └─────────────────────┘
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/handshape-gnn.git
cd handshape-gnn

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Preparation

### Required Datasets

1. **PopSign**: Download from [PopSign Dataset](https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html)
2. **ASL-LEX**: Download from [ASL-LEX Database](https://asl-lex.org/)

### Directory Structure

```
data/
├── popsign/
│   └── train_landmark_files/
│       ├── train.csv
│       └── *.parquet files
└── asl-lex/
    └── ASL-LEX.csv
```

### Merge Datasets

The merged dataset combines PopSign keypoint sequences with ASL-LEX handshape annotations:

```bash
python scripts/prepare_dataset.py \
    --popsign_dir data/popsign/train_landmark_files \
    --asllex_path data/asl-lex/ASL-LEX.csv \
    --output_path data/merged_dataset.json
```

This creates a JSON file with handshape labels mapped to PopSign sequences.

## Training

### 1. Train SignGNN (Temporal Model)

```bash
python train/train_sign_gnn.py \
    --data_root data/popsign/train_landmark_files \
    --output_dir checkpoints/sign_gnn \
    --batch_size 8 \
    --num_epochs 100 \
    --learning_rate 0.0001
```

### 2. Train HandshapeGNN (Static Model)

```bash
python train/train_handshape_gnn.py \
    --json_path data/merged_dataset.json \
    --output_dir checkpoints/handshape_gnn \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 0.0001
```

### 3. Generate Embeddings

After training both GNNs, generate embeddings:

```bash
# Sign embeddings
python scripts/generate_sign_embeddings.py \
    --checkpoint checkpoints/sign_gnn/best_model.pt \
    --data_root data/popsign/train_landmark_files \
    --output embeddings/sign_embeddings.pt

# Handshape embeddings
python scripts/generate_handshape_embeddings.py \
    --checkpoint checkpoints/handshape_gnn/best_model.pt \
    --json_path data/merged_dataset.json \
    --output embeddings/handshape_embeddings.pt
```

### 4. Train Triple-Stream Classifier

```bash
python train/train_triple_classifier.py \
    --sign_embeddings embeddings/sign_embeddings.pt \
    --handshape_embeddings embeddings/handshape_embeddings.pt \
    --landmarks_json data/merged_dataset.json \
    --output_dir checkpoints/triple_classifier \
    --batch_size 32 \
    --num_epochs 500 \
    --learning_rate 0.0001
```



## Evaluation

Evaluate a trained model:

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/triple_classifier/best_model.pt \
    --test_data data/merged_dataset.json \
    --output_dir results/
```

This will generate:
- Confusion matrix
- Per-class precision, recall, F1 scores
- Overall accuracy and macro F1

## Results

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Baseline MLP | 25.40% | 0.24 |
| SignGNN only | 30.01% | 0.26 |
| HandshapeGNN only | 31.00% | 0.26 |
| **Dual GNN (Ours)** | **46.07%** | **0.44** |

### Class Distribution

The dataset contains 37 handshape classes with labels provided by ASL-LEX:
- Most frequent: `open_b` (4,291), `1` (3,872), `5` (2,699)
- Total samples: 34,533 (27,626 train, 6,907 val)

## Repository Structure

```
handshape-gnn/
├── models/                    # Model architectures
│   ├── sign_gnn.py           # SignGNN (temporal)
│   ├── handshape_gnn.py      # HandshapeGNN (static)
│   ├── triple_classifier.py  # Triple-stream classifier
│   └── baseline.py           # Baseline MLP
├── datasets/                  # Dataset classes
│   ├── sign_dataset.py       # PopSign sequences
│   └── handshape_dataset.py  # Merged dataset
├── train/                     # Training scripts
│   ├── train_sign_gnn.py
│   ├── train_handshape_gnn.py
│   ├── train_triple_classifier.py
│   └── train_baseline.py
├── scripts/                   # Utility scripts
│   ├── prepare_dataset.py    # Data preparation
│   ├── generate_sign_embeddings.py
│   ├── generate_handshape_embeddings.py
│   └── evaluate.py
├── utils/                     # Helper functions
│   ├── metrics.py            # Biomechanical metrics
│   └── visualization.py      # Plotting utilities
├── checkpoints/              # Trained models
├── results/                  # Evaluation results
└── requirements.txt
```

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{carbo2025handshape,
  title={Improving Handshape Representations for Sign Language Processing: A Graph Neural Network Approach},
  author={Carbo, Alessa and Nalisnick, Eric},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  pages={29122--29135},
  year={2025}
}
```


## Acknowledgments

- PopSign dataset: https://signdata.cc.gatech.edu/view/datasets/popsign_v1_0/index.html
- ASL-LEX database by Emmorey et al: https://asl-lex.org/
- MediaPipe hand tracking by Google: https://ai.google.dev/edge/mediapipe/solutions/guide

## Contact

For questions or issues, please:
- Open an issue on GitHub
- Email: acarbol1@jh.edu
