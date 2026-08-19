"""Inferência do classificador de risco.

A probabilidade e o rótulo são produzidos por funções separadas de propósito:
a rede estima risco, o threshold é decisão de negócio. A separação permite que o threshold seja ajustado sem re-treinar o modelo.
"""

import torch
from torch import nn

DEFAULT_THRESHOLD = 0.5


def predict_proba(model: nn.Module, X: torch.Tensor) -> torch.Tensor:
    """Retorna a probabilidade estimada de default para cada registro. """
    model.eval()
    with torch.inference_mode():
        logits = model(X).squeeze(-1)
        return torch.sigmoid(logits)


def predict(
    model: nn.Module,
    X: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> torch.Tensor:
    """Converte a probabilidade em rótulo binário (0/1) segundo o threshold."""
    return (predict_proba(model, X) >= threshold).to(torch.int32)
