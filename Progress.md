# LumoraQ — Progress Log

---

## June 9, 2025

### Project Initialization
- Created fresh GitHub repository with proper structure
- Set up Miniforge conda environment (`lumoraq`) pinned to Python 3.11
- Built `src/` module structure: `dataset.py`, `models.py`, `train.py`, `utils.py`
- Configured `.gitignore`, `environment.yml`, `CONTRIBUTING.md`, `README.md`
- Installed `nbstripout` to prevent output commits
- Set up branch protection on `main`
- Established `main` and `dev` as permanent branches

---

### Experiment 1 — BaselineCNN

**Notebook:** `notebooks/01_baseline_cnn.ipynb`

**Architecture:**
- 2 conv layers (3 → 16 → 32 filters)
- MaxPooling after each conv block
- Fully connected: 8192 → 128 → 10
- No BatchNorm, no Dropout, no augmentation

**Config:**
- Epochs: 15
- Batch size: 32
- Optimizer: Adam (lr=0.001)
- Loss: CrossEntropyLoss
- Seed: 42
- Device: MPS (Apple M4)

**Result:**
- Best Val Acc: **84.10%** (epoch 10/15)


---

### Experiment 2 — StrongCNN

**Notebook:** `notebooks/02_strong_cnn.ipynb`

**Architecture:**
- 4 conv layers across 2 blocks (3 → 32 → 32 → 64 → 64 filters)
- BatchNorm after every conv layer
- MaxPooling after each block
- Dropout (0.4) before classifier
- Fully connected: 16384 → 256 → 10

**Augmentation:**
- RandomHorizontalFlip
- RandomRotation (20°)
- ColorJitter (brightness, contrast, saturation ±0.2)

**Config:**
- Epochs: 15
- Batch size: 32
- Optimizer: Adam (lr=0.0005)
- Loss: CrossEntropyLoss
- Seed: 42
- Device: MPS (Apple M4)

**Result:**
- Best Val Acc: **90.64%** (epoch 14/15)

---

### Experiment 3 — StrongCNN Evaluation

**Notebook:** `notebooks/03_evaluation.ipynb`

**Description:**
Full evaluation of the best StrongCNN checkpoint (`strong_cnn_best.pth`) on the held-out test set (4,050 images, never seen during training).

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **93%** |
| Weighted Precision | 0.93 |
| Weighted Recall | 0.93 |
| Weighted F1 | 0.93 |
| Test Set Size | 4,050 images |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| AnnualCrop | 0.90 | 0.95 | 0.92 |
| Forest | 0.98 | 0.95 | 0.96 |
| HerbaceousVegetation | 0.85 | 0.90 | 0.87 |
| Highway | 0.88 | 0.95 | 0.92 |
| Industrial | 0.96 | 0.98 | 0.97 |
| Pasture | 0.92 | 0.93 | 0.92 |
| PermanentCrop | 0.91 | 0.83 | 0.87 |
| Residential | 0.99 | 0.95 | 0.97 |
| River | 0.94 | 0.88 | 0.91 |
| SeaLake | 0.99 | 0.99 | 0.99 |

**Key findings:**
- SeaLake, Residential, Industrial — near perfect across all metrics
- HerbaceousVegetation and PermanentCrop remain the hardest classes — spectrally similar, frequently confused with each other
- 93% test accuracy sets the classical benchmark for quantum comparison

**Artifacts saved:**
- `results/strong_cnn_results.json` — full classification report
- `results/confusion_matrix.png` — confusion matrix heatmap
- `checkpoints/strong_cnn_best.pth` — best model weights (uploaded to GitHub Release v1.0-classical)

---

### Milestone: v1.0-classical tagged
- Classical baseline complete
- StrongCNN at 93% test accuracy is the benchmark all future quantum models will be measured against

---

## Next Steps

- [ ] `notebooks/04_vqc.ipynb` — Variational Quantum Classifier implementation
- [ ] `src/quantum.py` — PennyLane VQC circuit definition
- [ ] Hybrid model: StrongCNN feature extractor + VQC classifier head
- [ ] Benchmark VQC against 93% classical baseline
- [ ] Tag `v2.0-quantum` once quantum results are in
- [ ] ResNet-style architecture experiment (optional, before quantum phase)