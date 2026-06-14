import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(32 * 16 * 16, 128)
        self.fc2   = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class StrongCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.block2(self.block1(x))
        x = x.view(x.size(0), -1)
        return self.classifier(x)


from src.quantum import VQCClassifier

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'checkpoints', 'strong_cnn_best.pth'
)

class HybridModel(nn.Module):
    def __init__(self, n_qubits=8, n_layers=2, n_classes=10):
        super().__init__()

        # Load pretrained StrongCNN weights
        backbone = StrongCNN(num_classes=n_classes)
        backbone.load_state_dict(
            torch.load(CHECKPOINT_PATH, map_location='cpu')
        )

        # Reuse pretrained conv blocks — keys match exactly
        self.features = nn.Sequential(
            *list(backbone.block1.children()),
            *list(backbone.block2.children())
        )

        self.dropout = backbone.classifier[0]   # Dropout(0.4)
        self.fc      = backbone.classifier[1]   # Linear(16384, 256)

        # Freeze pretrained layers — only quantum head trains
        for param in self.features.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = False

        # Quantum classifier head — n_qubits controlled per notebook
        self.vqc = VQCClassifier(
            in_features=256,
            n_qubits=n_qubits,
            n_layers=n_layers,
            n_classes=n_classes
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc(x)))
        return self.vqc(x)