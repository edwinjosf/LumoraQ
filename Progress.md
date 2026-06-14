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

## June 13, 2025

### Quantum Phase — Infrastructure

- Implemented `src/quantum.py` — PennyLane VQC circuit using `AngleEmbedding` and `BasicEntanglerLayers`
- Implemented `HybridModel` in `src/models.py` — StrongCNN pretrained feature extractor + VQC classifier head
- Updated `src/train.py` — added optional optimizer parameter and versioned checkpoint saving
- Updated `src/utils.py` — added `get_quantum_device()` (returns CPU; PennyLane simulator not MPS-compatible)
- Added `get_small_dataloaders()` to `src/dataset.py` for faster quantum iteration
- Made `quantum.py` configurable via `make_quantum_layer(n_qubits, n_layers)` — qubit count controlled per notebook
- Added versioned checkpoint saving to `src/train.py` — prevents results being overwritten between runs

**Key design decisions:**
- Pretrained StrongCNN weights loaded and frozen — only quantum head trains
- Per-sample quantum circuit execution via `torch.stack` loop — required for PennyLane 0.45 compatibility
- `BatchNorm1d` added to VQC pre-layer for stable training
- `diff_method='best'` — PennyLane selects optimal gradient method automatically

---

### Experiment 4 — HybridModel 4-qubit VQC

**Notebook:** `notebooks/04_vqc_4qubit`

**Architecture:**
- Feature extractor: StrongCNN conv blocks (frozen, pretrained at 93%)
- Quantum head: 4 qubits, 2 layers, `BasicEntanglerLayers`
- Pre-layer: Linear(256→4) → BatchNorm1d → Tanh
- Post-layer: Linear(4→10)
- Trainable parameters: ~540 quantum parameters only

**Config:**
- Epochs: 15
- Batch size: 32
- Optimizer: Adam (lr=0.0005), quantum params only
- Loss: CrossEntropyLoss
- Seed: 42
- Device: CPU (quantum simulation)
- Dataset: Full EuroSAT (18,900 train / 4,050 val / 4,050 test)
- Training time: ~1 hour (Apple M4)

**Results:**

| Metric | Value |
|--------|-------|
| Best Val Acc | **64.72%** (epoch 14/15) |
| Test Accuracy | **63.04%** |
| Weighted Precision | 0.60 |
| Weighted Recall | 0.63 |
| Weighted F1 | 0.57 |
| Test Set Size | 4,050 images |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| AnnualCrop | 0.54 | 0.94 | 0.69 |
| Forest | 0.81 | 0.94 | 0.87 |
| HerbaceousVegetation | 0.41 | 0.36 | 0.39 |
| Highway | 0.43 | 0.14 | 0.21 |
| Industrial | 0.59 | 0.91 | 0.71 |
| Pasture | 0.70 | 0.13 | 0.22 |
| PermanentCrop | 0.41 | 0.61 | 0.49 |
| Residential | 0.90 | 0.98 | 0.94 |
| River | 0.29 | 0.03 | 0.05 |
| SeaLake | 0.82 | 0.99 | 0.90 |

**Key findings:**
- 4-qubit bottleneck causes partial class collapse — River (3% recall), Pasture (13% recall), Highway (14% recall)
- Strong classes: AnnualCrop, Forest, Residential, SeaLake — well separated even at 4 qubits
- Pretrained backbone significantly reduces collapse severity vs randomly initialized features
- Still climbing at epoch 15 — not converged

**Artifacts saved:**
- `results/vqc_4q_fulldataset_results_v2.json`
- `results/vqc_4q_fulldataset_confusion_matrix_v2.png`
- `results/vqc_4q_training_curves_v2.png`

---

## June 14, 2025

### Experiment 5 — HybridModel 8-qubit VQC

**Notebook:** `notebooks/05_vqc_8qubit.ipynb`

**Architecture:**
- Feature extractor: StrongCNN conv blocks (frozen, pretrained at 93%)
- Quantum head: 8 qubits, 2 layers, `BasicEntanglerLayers`
- Pre-layer: Linear(256→8) → BatchNorm1d → Tanh
- Post-layer: Linear(8→10)
- Trainable parameters: ~1,086 quantum parameters only

**Config:**
- Epochs: 15
- Batch size: 32
- Optimizer: Adam (lr=0.0005), quantum params only
- Loss: CrossEntropyLoss
- Device: CPU (quantum simulation)
- Dataset: Full EuroSAT (18,900 train / 4,050 val / 4,050 test)
- Training time: ~1.3 hours (Apple M4)

**Results:**

| Metric | Value |
|--------|-------|
| Test Accuracy | **78.35%** |
| Weighted Precision | 0.78 |
| Weighted Recall | 0.78 |
| Weighted F1 | 0.78 |
| Test Set Size | 4,050 images |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|-------|-----------|--------|----|
| AnnualCrop | 0.78 | 0.90 | 0.83 |
| Forest | 0.98 | 0.84 | 0.90 |
| HerbaceousVegetation | 0.65 | 0.72 | 0.68 |
| Highway | 0.64 | 0.42 | 0.51 |
| Industrial | 0.88 | 0.92 | 0.90 |
| Pasture | 0.78 | 0.72 | 0.75 |
| PermanentCrop | 0.65 | 0.61 | 0.63 |
| Residential | 0.92 | 0.99 | 0.96 |
| River | 0.58 | 0.62 | 0.60 |
| SeaLake | 0.91 | 0.97 | 0.94 |

**Key findings:**
- 8-qubit model achieves 78.35% — 15-point improvement over 4-qubit (63.04%)
- No class hits 0% recall — all 10 classes represented, bottleneck collapse resolved
- Strong classes: Residential (0.96 F1), SeaLake (0.94 F1), Forest (0.90 F1), Industrial (0.90 F1)
- Hardest classes remain Highway (0.51 F1) and HerbaceousVegetation (0.68 F1) — consistent with classical model
- Pretrained backbone + 8 qubits closes the gap to classical from 30 points (4-qubit) to 14.65 points
- Still climbing at epoch 15 — not converged, further gains expected with longer training

**Artifacts saved:**
- `results/vqc_8q_fulldataset_results_v2.json`
- `results/vqc_8q_fulldataset_confusion_matrix_v2.png`

---

## Final Results Summary

| Model | Test Accuracy | Trainable Params |
|-------|---------------|-----------------|
| BaselineCNN | 84.10% (val) | ~75K |
| StrongCNN (classical) | **93.00%** | ~1.5M |
| HybridModel 4-qubit VQC | 63.04% | ~540 |
| HybridModel 8-qubit VQC | **78.35%** | ~1,086 |

---

### Milestone: v2.0-quantum tagged
- Quantum phase complete
- Both 4-qubit and 8-qubit hybrid models benchmarked against 93% classical baseline
- Key finding: 8-qubit VQC with pretrained backbone achieves 78.35% — only 14.65 points behind classical with 1,086 parameters vs 1.5 million

---

## Next Steps

- [ ] Longer training runs — neither model converged at epoch 15, further gains expected
- [ ] Test 16-qubit circuit — does the 4→8 qubit improvement trend continue?
- [ ] Alternative ansatz designs — data re-uploading circuits, hardware-efficient ansatze
- [ ] Real IBM Quantum hardware validation (free tier via Qiskit)
- [ ] Fine-tune pretrained backbone jointly with quantum head after initial quantum training
- [ ] Characterize initialization sensitivity — repeat runs with multiple seeds to establish variance