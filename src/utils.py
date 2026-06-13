import torch
import numpy as np
import random

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    print(f"Seed set: {seed}")

def get_quantum_device():
    # Quantum circuits only run on CPU — MPS/CUDA not supported by PennyLane simulator
    return torch.device('cpu')
