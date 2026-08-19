import torch
from torch import nn

from src.preprocessing.transform import N_FEATURES


class NeuralNetworkV0(nn.Module):
    """MLP com uma camada oculta de 20 neurônios com ReLU
    (21 features de entrada -> 20 neurônios -> 1 saída)."""

    def __init__(self, in_features: int = N_FEATURES, hidden_units: int = 20) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self.layer_2 = nn.Linear(in_features=hidden_units, out_features=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1_output = self.relu(self.layer_1(x))
        return self.layer_2(layer1_output)
