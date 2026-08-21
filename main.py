"""Ponto de entrada do loan-risk-analyzer (pipeline ponta a ponta da Entrega 1).

Fluxo: carrega CSV de empréstimos -> codifica variáveis categóricas
-> split treino/teste -> grava resultados no diretório de saída.

Os diretórios vêm de variáveis de ambiente e, no container, correspondem a
volumes montados explicitamente. O script não acessa o sistema de arquivos do
host fora desses volumes.
"""

from __future__ import annotations

import sys
import torch
from torch import nn
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.data.loan_loader import load_csv
from src.preprocessing.transform import (
    smote_oversampling,
    fit_encoder,
    fit_scaler,
)
from src.training.train import to_tensor, train_nn
from src.models.NeuralNetworkV0 import NeuralNetworkV0
from src.inference.inference import predict_proba
from src.utils.config import Settings, RANDOM_SEED
from src.evaluation.metrics import evaluate_model
from src.export.bundle import export_bundle

EPOCHS = 1500
LEARNING_RATE = 0.01

def run(settings: Settings) -> int:
    """Executa o pipeline e retorna um código de saída (0 = sucesso)."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    settings.models_dir.mkdir(parents=True, exist_ok=True)

    records = load_csv(settings.data_path)
    if records.empty:
        print(f"Nenhum registro encontrado em {settings.data_path}", file=sys.stderr)
        return 1

    # Codifica variáveis categóricas. Retorna dataset como vetores NumPy
    encoder, X, y, feature_names = fit_encoder(records)
    print(f"Features: {X.shape[1]} — {', '.join(feature_names)}")

    # Separação de conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=RANDOM_SEED)

    # Oversampling devido ao desbalanceamento nos dados
    X_train, y_train = smote_oversampling(X_train, y_train)

    # Normalização dos dados
    scaler = fit_scaler(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    # Verifica a disponibilidade de GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Converte arrays NumPy em tensores PyTorch (já envia para a GPU, caso esteja disponível)
    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = to_tensor(X_train, y_train, X_test, y_test, device)

    # Semeia ANTES de construir a rede: fonte de aleatoriedade restante do treino
    torch.manual_seed(RANDOM_SEED)

    # Treina a rede neural
    nn_model = NeuralNetworkV0(in_features=X.shape[1]).to(device)
    train_nn(EPOCHS, nn_model, LEARNING_RATE, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor)

    # Salva o modelo
    save_model(nn_model, settings.models_dir / "neural_network_V0.pth")

    # Estima a probabilidade de default no conjunto de teste
    test_proba = predict_proba(nn_model, X_test_tensor)

    # Exibe métricas do modelo para o conjunto de testes
    metrics = evaluate_model(device, y_test_tensor, test_proba)

    # Empacota modelo + pré-processamento + métricas para o serviço de inferência 
    bundle_dir = export_bundle(
        output_dir=settings.models_dir / "bundle",
        model=nn_model,
        scaler=scaler,
        encoder=encoder,
        records=records,
        feature_names=feature_names,
        metrics=metrics,
        dataset_path=settings.data_path,
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
    )
    print(f"\nBundle de inferência exportado em {bundle_dir}")

    return 0


def save_model(model: nn.Module, path: Path) -> None:
    """Salva os pesos do modelo (state_dict) no caminho especificado."""
    torch.save(model.state_dict(), path)
    print(f"\nModelo salvo em {path}")

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

def main() -> None:
    """Função principal."""
    sys.exit(run(Settings.from_env()))


if __name__ == "__main__":
    main()