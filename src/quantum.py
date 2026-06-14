import pennylane as qml
import torch
import torch.nn as nn


def make_quantum_layer(n_qubits=8, n_layers=2):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev, interface='torch', diff_method='best')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
        qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    weight_shapes = {'weights': (n_layers, n_qubits)}
    return qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class VQCClassifier(nn.Module):
    def __init__(self, in_features=256, n_qubits=8, n_layers=2, n_classes=10):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Linear(in_features, n_qubits),
            nn.BatchNorm1d(n_qubits),
            nn.Tanh()
        )
        self.qlayer = make_quantum_layer(n_qubits, n_layers)
        self.post   = nn.Linear(n_qubits, n_classes)

    def forward(self, x):
        x = self.pre(x)
        x = torch.stack([self.qlayer(x[i]) for i in range(x.shape[0])])
        x = self.post(x)
        return x