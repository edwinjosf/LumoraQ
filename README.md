# LumoraQ
Quantum Machine Learning for Satellite Image Classification using EuroSAT.

## Setup

### 1. Clone
git clone https://github.com/your-username/LumoraQ.git
cd LumoraQ

### 2. Create environment
conda env create -f environment.yml
conda activate lumoraq

### 3. Register Jupyter kernel
python -m ipykernel install --user --name lumoraq --display-name "LumoraQ"

### 4. Run
jupyter notebook notebooks/

The EuroSAT dataset (~90MB) downloads automatically on first run.

## Results

| Model       | Accuracy |
|-------------|----------|
| BaselineCNN | 80.28%   |
| StrongCNN   | 88.83%   |
| VQC         | TBD      |

## Project Structure

    src/           Core modules (dataset, models, training)
    notebooks/     Experiments (one notebook per experiment)
    checkpoints/   Saved model weights (gitignored)
    results/       Metrics and outputs
    data/          EuroSAT dataset (gitignored, auto-downloads)

## Contributing
See CONTRIBUTING.md
