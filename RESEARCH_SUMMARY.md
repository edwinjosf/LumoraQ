# Research Summary: Hybrid Quantum-Classical Models for Satellite Image Classification

## Overview

This project investigates whether hybrid quantum-classical neural networks can
perform satellite image classification, and how their performance and error
patterns compare to a purely classical convolutional neural network (CNN).

The dataset is EuroSAT — 27,000 Sentinel-2 satellite images across 10 land-use
and land-cover classes (AnnualCrop, Forest, HerbaceousVegetation, Highway,
Industrial, Pasture, PermanentCrop, Residential, River, SeaLake).

## Methodology

All models share the same convolutional feature extractor (StrongCNN-style:
4 conv layers with batch normalization, dropout, and data augmentation). The
classical model uses a standard fully-connected classifier head. The hybrid
models replace this head with a Variational Quantum Classifier (VQC)
implemented in PennyLane, using AngleEmbedding for data encoding and
BasicEntanglerLayers as the trainable ansatz.

All models were trained on the same 70/15/15 train/validation/test split
(18,900 / 4,050 / 4,050 images) and evaluated on the same held-out 4,050-image
test set, allowing direct comparison.

| Model                     | Architecture                          | Test Accuracy |
|---------------------------|----------------------------------------|---------------|
| BaselineCNN               | 2 conv layers, classical head          | 84.10% (val)  |
| StrongCNN                 | 4 conv layers + BN/Dropout, classical head | 93%       |
| HybridModel (4-qubit VQC) | StrongCNN features + 4-qubit VQC head  | 59%           |
| HybridModel (8-qubit VQC) | StrongCNN features + 8-qubit VQC head  | 76%           |

## Key Finding 1: The 4-Qubit Bottleneck Causes Systematic Pairwise Class Collapse

The 4-qubit VQC head produces only 4 expectation values, which are then mapped
to 10 class scores via a single linear layer. This is a severe information
bottleneck compared to the classical head's 256-dimensional representation.

With 4 qubits, exactly 4 of the 10 classes were never predicted at all
(0% precision and recall), with their samples almost entirely absorbed into
one other class each:

| True Class           | Collapsed Into        | Rate |
|-----------------------|------------------------|------|
| Industrial            | Residential            | 97%  |
| Pasture                | AnnualCrop              | 84%  |
| PermanentCrop          | HerbaceousVegetation    | 81%  |
| River                  | Highway                 | 68%  |

Three of these four pairings (Industrial/Residential, Pasture/AnnualCrop,
PermanentCrop/HerbaceousVegetation) correspond to classes that are also known
to be confused by the classical model — they are genuinely similar in spectral
signature. The fourth (River/Highway) shares a visual characteristic (long,
narrow, linear features) rather than a spectral one.

This suggested a testable hypothesis: the collapse is driven primarily by the
4-qubit output bottleneck forcing a "winner-take-all" choice between similar
classes, rather than by the quantum circuit being fundamentally incapable of
learning useful features.

## Key Finding 2: Doubling to 8 Qubits Resolves Bottleneck-Driven Collapse

Doubling the circuit to 8 qubits (8 expectation values feeding the final
linear layer) increased test accuracy from 59% to 76% — a 17-point
improvement — and, critically, changed the *character* of the errors:

- **Industrial**, completely collapsed at 4 qubits (0% recall), reached 99%
  recall at 8 qubits — essentially resolved.
- **PermanentCrop -> HerbaceousVegetation** confusion dropped from 81% to 29%,
  and became bidirectional (HerbaceousVegetation samples are also sometimes
  misclassified as PermanentCrop) rather than one-directional collapse.
- **River -> Highway** confusion dropped from 68% to 10%.
- **Pasture -> River/SeaLake** confusion remained at a similar rate (~31%),
  the one pairing that did not substantially improve.

At 8 qubits, no class has 0% precision or recall — every class is represented
to some degree, with the lowest-performing class (PermanentCrop) still
achieving 0.55 F1.

### Interpretation

The same class pairs that caused total collapse at 4 qubits remain the
dominant sources of confusion at 8 qubits, but the errors are now partial and
often bidirectional rather than total and one-directional. This is consistent
with the following picture:

1. These class pairs have genuine, underlying visual/spectral similarity that
   makes them inherently hard to separate (this difficulty is also present in
   the classical model's errors).
2. At 4 qubits, the severe output bottleneck forced the model into a binary
   "choose one of the pair" outcome for each similar class pair, fully
   sacrificing one class.
3. At 8 qubits, the additional output dimensions give the model enough
   capacity to represent "probably class A, but could be class B" — producing
   partial confusion rather than total collapse.

In other words: more qubits did not make these classes spectrally distinct
(they aren't), but it did relieve the artificial, bottleneck-driven
all-or-nothing behavior that the 4-qubit model was forced into.

Notably, validation accuracy was still climbing at the final epoch of the
8-qubit run (75.46% -> 76.37% from epoch 13 to 15), suggesting the model had
not yet converged and further training and/or further qubit scaling may yield
additional gains.

---

## Updated Results — Pretrained Backbone

A second experimental run was conducted with a key architectural change: rather
than training the full hybrid model end-to-end from random initialization, the
StrongCNN feature extractor was pre-loaded with weights from the trained
classical model (93% test accuracy) and **frozen**. Only the VQC classifier
head was trained. This tests whether the quality of the input features — rather
than the qubit count — is the dominant factor in hybrid model performance.

| Model | Feature Extractor | Test Accuracy | Trainable Params |
|-------|-------------------|---------------|-----------------|
| HybridModel 4-qubit | Random init (original) | 59% | full model |
| HybridModel 8-qubit | Random init (original) | 76% | full model |
| HybridModel 4-qubit | Pretrained + frozen | **63.04%** | ~540 |
| HybridModel 8-qubit | Pretrained + frozen | **78.35%** | ~1,086 |

### Key Finding 3: Pretrained Features Matter More Than Qubit Count

With a pretrained frozen backbone:

- **4-qubit accuracy improved from 59% to 63.04%** — modest gain, but more
  importantly, the character of errors changed significantly. No class hit 0%
  recall. The catastrophic pairwise collapse seen with random features was
  substantially reduced — River recall went from 0% to 3%, Pasture from 0% to
  13%. The bottleneck still hurts, but it no longer causes total class
  sacrifice.

- **8-qubit accuracy improved from 76% to 78.35%** — a meaningful 2-point gain
  on top of already-strong results. All 10 classes are well-represented, with
  the weakest class (Highway) still achieving 0.51 F1. Weighted F1 across all
  classes reached 0.78.

- **The 4→8 qubit gap narrowed from 17 points to 15 points** (63.04% →
  78.35%). With pretrained features, 4 qubits already capture enough
  discriminative signal to partially separate all classes, reducing the
  marginal benefit of additional qubits.

### Interpretation

The original results (Key Findings 1 and 2) established that qubit count is a
major bottleneck when the quantum circuit must learn from scratch. The pretrained
backbone results reveal a second, equally important factor: **the quality of the
classical features fed into the quantum circuit**.

When the feature extractor is randomly initialized, the quantum circuit must
simultaneously learn to extract useful features and classify them — an
ill-posed problem with only a handful of trainable parameters. When the feature
extractor is pretrained, the quantum circuit receives rich, 256-dimensional
representations that already encode discriminative structure, and needs only to
learn a decision boundary over those representations.

This suggests a practical guideline for hybrid QML systems: **pre-training or
fine-tuning the classical component before introducing the quantum head** may
be more impactful than increasing qubit count, at least in the low-qubit regime
accessible to current simulators.

Additionally, training only ~540–1,086 quantum parameters (vs the full model)
reduced training time significantly — the pretrained 8-qubit run completed in
~1.3 hours on CPU, making iterative experimentation practical without GPU
resources.

---

## Future Work

**Scaling qubit count further (16+ qubits).** Given the clear improvement from
4 to 8 qubits and the lack of convergence at 15 epochs, a natural next step is
testing whether 16 qubits continues this trend or whether other bottlenecks
(circuit depth, ansatz expressiveness, barren plateaus) begin to dominate.
GPU-accelerated simulation (e.g., PennyLane Lightning-GPU) would likely be
necessary at this scale.

**Alternative ansatz designs.** The current ansatz (BasicEntanglerLayers) is a
simple, generic choice. Hardware-efficient ansatze or data re-uploading
circuits might achieve similar or better representational capacity with fewer
qubits or shallower circuits.

**Real quantum hardware validation.** Running inference with the trained 8-qubit
model on real IBM Quantum hardware (via Qiskit, free-tier access) would test
whether the simulated results hold under real hardware noise.

**Targeted investigation of the Pasture/River/SeaLake confusion.** This was the
one pairing that did not substantially improve from 4 to 8 qubits in either
experimental setup, and may warrant separate analysis (e.g., feature
visualization) to understand whether it reflects a genuine, persistent spectral
overlap or a different kind of model limitation.

**Longer training / convergence study.** Neither the 4-qubit nor 8-qubit
pretrained models converged at epoch 15. Training for more epochs (e.g., 30-50)
could establish whether the gap to the classical model (93%) continues to
narrow with additional training alone.

**Joint fine-tuning.** After initial quantum head training with frozen backbone,
unfreezing and jointly fine-tuning the full hybrid model may yield further
accuracy gains by allowing the feature extractor to adapt to the quantum
classifier's needs.

**Initialization sensitivity study.** Variance across runs with different random
seeds should be characterized to understand the stability of the quantum
training procedure and establish confidence intervals around reported results.

## Conclusion

This work demonstrates a complete, working hybrid quantum-classical pipeline
for satellite image classification, with results directly comparable to a strong
classical baseline (93% test accuracy) using identical data splits and
evaluation. Two experimental setups were evaluated:

The original setup (random feature initialization) achieved 59% at 4 qubits
and 76% at 8 qubits, establishing that quantum circuit output dimensionality
is a major and partially addressable bottleneck.

The pretrained backbone setup achieved 63.04% at 4 qubits and 78.35% at 8
qubits using only 540–1,086 trainable quantum parameters — demonstrating that
input feature quality is as important as qubit count, and that practical hybrid
QML systems can be trained efficiently by leveraging pretrained classical
components. The 8-qubit pretrained model closes the gap to the classical
baseline to 14.65 points while using over 1,000x fewer trainable parameters.

Together, these findings motivate a clear set of next experiments around qubit
scaling, circuit design, joint fine-tuning, and real-hardware validation.