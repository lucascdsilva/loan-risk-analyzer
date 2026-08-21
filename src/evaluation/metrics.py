"""Métricas de avaliação do classificador de risco"""

from typing import Dict

import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)

from src.inference.inference import DEFAULT_THRESHOLD


def evaluate_model(
    device: str,
    y: torch.Tensor,
    y_prob: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> Dict[str, object]:
    """Calcula as métricas do classificador de risco e as exibe no console.

    Recebe **probabilidades**, não rótulos: a AUROC precisa do score contínuo
    para ser informativa — sobre 0/1 arredondados ela degenera em acurácia
    balanceada. As demais métricas aplicam o ``threshold`` internamente.

    Returns:
        Dicionário com as métricas escalares, pronto para o ``model_card.json``.
    """
    scalar_metrics = MetricCollection({
        "Accuracy":  BinaryAccuracy(threshold=threshold),
        "Precision": BinaryPrecision(threshold=threshold),
        "Recall":    BinaryRecall(threshold=threshold),
        "F1-Score":  BinaryF1Score(threshold=threshold),
        "AUROC":     BinaryAUROC(),
    }).to(device)

    confusion = BinaryConfusionMatrix(threshold=threshold).to(device)

    y = y.to(torch.int32)
    results = {name: value.item() for name, value in scalar_metrics(y_prob, y).items()}
    matrix = confusion(y_prob, y)

    print("\n--- Desempenho do modelo ---")
    for name, value in results.items():
        print(f"{name}: {value:.4f}")

    (tn, fp), (fn, tp) = matrix.tolist()
    print(f"\n--- Matriz de confusão (threshold {threshold}) ---")
    print(f"{'':>12}{'pred 0':>10}{'pred 1':>10}")
    print(f"{'real 0':>12}{tn:>10}{fp:>10}")
    print(f"{'real 1':>12}{fn:>10}{tp:>10}")

    results["confusion_matrix"] = [[tn, fp], [fn, tp]]
    return results
