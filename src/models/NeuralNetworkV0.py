import torch
from torch import nn

class NeuralNetworkV0(nn.Module):
    """MLP com uma camada oculta de 20 neurônios com ReLU
    (23 features de entrada -> 20 neurônios -> 1 saída)."""
    def __init__(self) -> None:
        super().__init__()
        self.layer_1 = nn.Linear(in_features=23, out_features=20) # 23 features de entrada e 20 neurônios 
        self.layer_2 = nn.Linear(in_features=20, out_features=1) # 20 neurônios e 1 saída
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer1_output = self.relu(self.layer_1(x))
        return self.layer_2(layer1_output)