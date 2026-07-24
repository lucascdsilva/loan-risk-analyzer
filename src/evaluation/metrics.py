import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)

def evaluate_model(device: str, y: torch.Tensor, y_pred: torch.Tensor) -> None:
    """Avalia o modelo calculando acurácia, precisão, recall e F1-score binários
    usando torchmetrics e exibe os resultados formatados no console."""
    metrics = MetricCollection({
        "Accuracy":  BinaryAccuracy(),
        "Precision": BinaryPrecision(),
        "Recall":    BinaryRecall(),
        "F1-Score":  BinaryF1Score(),
    }).to(device)

    y = y.to(torch.int32)
    metrics_result = metrics(y_pred, y)

    print("\n--- Desempenho do modelo ---")
    for name, value in metrics_result.items():
        print(f"{name}: {value:.4f}")

