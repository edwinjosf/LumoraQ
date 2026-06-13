import pennylane as qml
import torch
import torch.nn as nn

N_QUBITS = 8
N_LAYERS = 2

dev = qml.device('default.qubit', wires=N_QUBITS)

@qml.qnode(dev, interface='torch', diff_method='best')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS), rotation='Y')
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

weight_shapes = {'weights': (N_LAYERS, N_QUBITS)}
quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class VQCClassifier(nn.Module):
    def __init__(self, in_features=256, n_qubits=N_QUBITS, n_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh()
        )
        self.qlayer = quantum_layer
        self.post = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = self.pre(x)                          # [B, 4]
        x = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])
        x = self.post(x)                         # [B, 10]
        return x