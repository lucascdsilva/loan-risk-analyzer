"""Ponto de entrada do loan-risk-analyzer (pipeline ponta a ponta da Entrega 1).

Fluxo: carrega CSV de empréstimos -> codifica variáveis categóricas
-> split treino/teste -> grava resultados no diretório de saída.

Os diretórios vêm de variáveis de ambiente e, no container, correspondem a
volumes montados explicitamente. O script não acessa o sistema de arquivos do
host fora desses volumes.
"""

from __future__ import annotations

import csv
import dataclasses
import sys
import torch
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.loan_loader import load_csv
from src.preprocessing.transform import (
    smote_oversampling,
    scale_dataset,
    encode_features
)
from src.training.train import to_tensor, train_nn
from src.models.model import NeuralNetworkV0
from src.inference.inference import predict
from src.utils.config import Settings, RANDOM_SEED
from src.evaluation.metrics import evaluate_model

def run(settings: Settings) -> int:
    """Executa o pipeline e retorna um código de saída (0 = sucesso)."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_csv(settings.data_path)
    if records.empty:
        print(f"Nenhum registro encontrado em {settings.data_path}", file=sys.stderr)
        return 1

    # Codifica variáveis categóricas. Retorna dataset como vetores NumPy
    X, y, features_names = encode_features(records)

    # Separação de conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)

    # Oversampling devido ao desbalanceamento nos dados
    X_train, y_train = smote_oversampling(X_train, y_train)

    # Normalização dos dados
    X_train, X_test = scale_dataset(X_train, y_train, X_test, y_test)

    # Converte arrays NumPy em tensores PyTorch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = to_tensor(X_train, y_train, X_test, y_test, device)

    # Treina a rede neural
    nn_model = NeuralNetworkV0().to(device)
    train_nn(3000, nn_model, 0.01, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

    # Classifica dados de teste
    test_pred = predict(nn_model, X_test_tensor)

    # Exibe métricas do modelo para o conjunto de testes
    evaluate_model(device, y_test_tensor, test_pred)

    return 0


def _build_summary(records, cleaned, train, test) -> str:
    default_rate = (
        sum(r.loan_status for r in cleaned) / len(cleaned) * 100
        if cleaned else 0.0
    )
    lines = [
        "loan-risk-analyzer — Sumário do Pré-processamento",
        "-" * 50,
        f"{'Registros carregados':<30} {len(records):>10}",
        f"{'Após codificação':<30} {len(cleaned):>10}",
        f"{'Treino':<30} {len(train):>10}",
        f"{'Teste':<30} {len(test):>10}",
        f"{'Taxa de default (%)':<30} {default_rate:>9.1f}%",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, records) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in dataclasses.fields(records[0])])
        for r in records:
            writer.writerow(dataclasses.astuple(r))

def main() -> None:
    """Função principal."""
    sys.exit(run(Settings.from_env()))


if __name__ == "__main__":
    main()