# LumoraQ
Quantum Machine Learning for Satellite Image Classification using EuroSAT.

---

## Setup

### 1. Clone
```bash
git clone https://github.com/edwinjosf/LumoraQ.git
cd LumoraQ
```

### 2. Create environment
```bash
conda env create -f environment.yml
conda activate lumoraq
```

> **Windows users:** run these commands in Git Bash or Anaconda Prompt to ensure conda initialises properly.

### 3. Register Jupyter kernel
```bash
python -m ipykernel install --user --name lumoraq --display-name "LumoraQ"
```

### 4. Run
```bash
jupyter notebook notebooks/
```

> The EuroSAT dataset (~90MB) downloads automatically on first run.

---

## Results

| Model       | Accuracy |
|-------------|----------|
| BaselineCNN | 84.10%   |
| StrongCNN   | 93.00%   |
| VQC         | TBD      |

---

## Project Structure

```
LumoraQ/
├── src/            Core modules (dataset, models, training, utils)
├── notebooks/      Experiments (one notebook per experiment)
├── checkpoints/    Saved model weights (gitignored)
├── results/        Metrics and outputs
└── data/           EuroSAT dataset (gitignored, auto-downloads)
```

---

## Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) e