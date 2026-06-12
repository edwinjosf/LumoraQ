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
one pairing that did not improve from 4 to 8 qubits, and may warrant separate
analysis (e.g., feature visualization) to understand whether it reflects a
genuine, persistent spectral overlap or a different kind of model limitation.

**Longer training / convergence study.** Since the 8-qubit model was still
improving at epoch 15, training for more epochs (e.g., 30-50) could establish
whether the gap to the classical model (93%) continues to narrow with
additional training alone, independent of further architectural changes.

## Conclusion

This work demonstrates a complete, working hybrid quantum-classical pipeline
for satellite image classification, with results that are directly comparable
to a strong classical baseline (93% test accuracy) using identical data splits
and evaluation. The 4-qubit hybrid model (59%) and 8-qubit hybrid model (76%)
provide concrete evidence that quantum circuit output dimensionality is a
major, and partially addressable, bottleneck — while also revealing that some
classification difficulty is intrinsic to the dataset's class structure rather
than an artifact of the quantum approach. These findings motivate a clear set
of next experiments around qubit scaling, circuit design, and real-hardware
validation.