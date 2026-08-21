"""Exportação do bundle de inferência
Ele produz um diretório autocontido:

    model.onnx           padronização + rede + sigmoid, entrada (batch, 21)
    preprocessing.json   categorias, escala ordinal e ordem das features
    model_card.json      versões, arquitetura, treino, dataset e métricas
    golden.json          registros crus + probabilidade esperada
    SHA256SUMS           integridade

O `StandardScaler` entra **dentro** do grafo ONNX: é uma transformação afim,
vira dois nós, e assim o serviço não reimplementa normalização.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.preprocessing.transform import (
    CATEGORICAL_COLUMNS,
    DROPPED_COLUMNS,
    EDUCATION_ORDER,
    N_FEATURES,
    ONE_HOT_CATEGORIES,
    TARGET_COLUMN,
)
from src.utils.config import (
    DEFAULT_THRESHOLD,
    MODEL_VERSION,
    PREPROCESSING_VERSION,
    RANDOM_SEED,
)

ONNX_OPSET = 17
GOLDEN_SAMPLE_SIZE = 50


class ExportableModel(nn.Module):
    """Envolve a rede com a padronização e a sigmoid para exportação.

    A entrada é o vetor de 21 features **não normalizado** — a normalização
    acontece aqui dentro. Assim o serviço de inferência só precisa saber
    codificar as categóricas.
    """

    def __init__(self, net: nn.Module, mean: np.ndarray, scale: np.ndarray) -> None:
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("scale", torch.tensor(scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        standardized = (x - self.mean) / self.scale
        return torch.sigmoid(self.net(standardized)).squeeze(-1)


def build_exportable(model: nn.Module, scaler: StandardScaler) -> ExportableModel:
    """Monta o modelo exportável a partir da rede treinada e do scaler."""
    exportable = ExportableModel(
        net=model.to("cpu"), mean=scaler.mean_, scale=scaler.scale_
    )
    return exportable.eval()


def export_onnx(exportable: ExportableModel, path: Path) -> None:
    """Serializa o grafo em ONNX com eixo de batch dinâmico.

    ``external_data=False`` mantém os pesos dentro do próprio ``.onnx``. O
    padrão do exportador é gravá-los num ``model.onnx.data`` ao lado, e um
    artefato que depende de um arquivo irmão pelo nome é frágil de distribuir
    """
    dummy = torch.zeros(1, N_FEATURES, dtype=torch.float32)
    torch.onnx.export(
        exportable,
        (dummy,),
        str(path),
        input_names=["features"],
        output_names=["probability"],
        # `dynamic_shapes` (e não `dynamic_axes`) é a API do exportador dynamo;
        # a chave "x" é o nome do parâmetro de ExportableModel.forward.
        dynamic_shapes={"x": {0: torch.export.Dim("batch")}},
        opset_version=ONNX_OPSET,
        external_data=False,
    )


def build_preprocessing_spec(feature_names: Sequence[str]) -> Dict[str, object]:
    """Descreve a codificação de features de forma declarativa.

    É o suficiente para o serviço reconstruir o vetor de 21 posições sem
    scikit-learn: quais colunas descartar, como mapear a escolaridade, quais
    categorias one-hot em que ordem, e o nome de cada posição final.
    """
    return {
        "preprocessing_version": PREPROCESSING_VERSION,
        "n_features": N_FEATURES,
        "target_column": TARGET_COLUMN,
        "dropped_columns": list(DROPPED_COLUMNS),
        # Mapa por coluna, e não uma chave "education_order" solta: assim o
        # consumidor descobre *qual* coluna é ordinal a partir do próprio
        # contrato, em vez de hardcodar o nome.
        "ordinal_encodings": {"person_education": dict(EDUCATION_ORDER)},
        "categorical_columns": list(CATEGORICAL_COLUMNS),
        "one_hot_categories": {c: list(v) for c, v in ONE_HOT_CATEGORIES.items()},
        "feature_names": list(feature_names),
        # RF-ML-04: categoria fora da lista é erro, nunca bloco de zeros.
        "unknown_category_policy": "reject",
    }


def build_model_card(
    model: nn.Module,
    dataset_path: Path,
    n_rows: int,
    metrics: Mapping[str, object],
    epochs: int,
    learning_rate: float,
) -> Dict[str, object]:
    """Reúne tudo que o Model Registry (ML01) precisa saber desta versão."""
    return {
        "model": type(model).__name__,
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "architecture": {
            "in_features": N_FEATURES,
            "hidden_units": model.layer_1.out_features,
            "activation": "ReLU",
            "output": "sigmoid",
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": "Adam",
            "loss": "BCEWithLogitsLoss",
            "oversampling": "SMOTE (somente treino)",
            "test_size": 0.25,
            "seed": RANDOM_SEED,
        },
        "dataset": {
            "name": dataset_path.name,
            "rows": n_rows,
            "sha256": _sha256(dataset_path),
        },
        "metrics": dict(metrics),
        "default_threshold": DEFAULT_THRESHOLD,
        "export": {"format": "onnx", "opset": ONNX_OPSET, "torch": torch.__version__},
    }


def select_golden_records(dataset: pd.DataFrame) -> pd.DataFrame:
    """Escolhe registros de referência cobrindo todas as categorias.

    Amostragem aleatória não basta: ``person_home_ownership='OTHER'`` aparece
    em 0,26% do dataset e seria perdido na maioria das amostras de 50. Aqui a
    amostra determinística é completada com uma linha para cada par
    (coluna, categoria) que tenha ficado de fora.
    """
    sample = dataset.sample(
        n=min(GOLDEN_SAMPLE_SIZE, len(dataset)), random_state=RANDOM_SEED
    )

    faltantes: List[int] = []
    for column, categories in ONE_HOT_CATEGORIES.items():
        for category in categories:
            if (sample[column] == category).any():
                continue
            candidatos = dataset.index[dataset[column] == category]
            if len(candidatos):
                faltantes.append(candidatos[0])

    for education in EDUCATION_ORDER:
        if (sample["person_education"] == education).any():
            continue
        candidatos = dataset.index[dataset["person_education"] == education]
        if len(candidatos):
            faltantes.append(candidatos[0])

    if faltantes:
        sample = pd.concat([sample, dataset.loc[sorted(set(faltantes))]])

    return sample.reset_index(drop=True)


def build_golden(
    records: pd.DataFrame, probabilities: np.ndarray
) -> Dict[str, object]:
    """Monta o arquivo de referência que trava o contrato entre A e B.

    Cada item traz o payload cru (exatamente como chegaria por HTTP) e a
    probabilidade que **este** bundle produz. O serviço de inferência roda isso
    no CI: se o encoder dele divergir, o teste falha.
    """
    payload_columns = [
        c for c in records.columns
        if c != TARGET_COLUMN and c not in DROPPED_COLUMNS
    ]
    return {
        "model_version": MODEL_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "tolerance": 1e-6,
        "records": [
            {
                "payload": _jsonable(row[payload_columns].to_dict()),
                "default_probability": float(probability),
            }
            for (_, row), probability in zip(records.iterrows(), probabilities)
        ],
    }


def run_onnx(model_path: Path, X: np.ndarray) -> np.ndarray:
    """Executa o grafo exportado"""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    features = np.ascontiguousarray(X, dtype=np.float32)
    return session.run(None, {"features": features})[0]


def export_bundle(
    output_dir: Path,
    model: nn.Module,
    scaler: StandardScaler,
    encoder,
    records: pd.DataFrame,
    feature_names: Sequence[str],
    metrics: Mapping[str, object],
    dataset_path: Path,
    epochs: int,
    learning_rate: float,
) -> Path:
    """Gera o bundle completo. É o único ponto de entrada usado pelo pipeline.

    As probabilidades do golden são calculadas com o **ONNX já exportado**, e
    não com o modelo PyTorch: assim a verificação no serviço de inferência
    compara onnxruntime com onnxruntime, sem herdar a diferença de ~1e-6 que
    existe entre os dois backends em float32.
    """
    from src.preprocessing.transform import transform_records

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.onnx"

    export_onnx(build_exportable(model, scaler), model_path)

    golden_records = select_golden_records(records)
    golden_features = transform_records(encoder, golden_records)
    golden_probabilities = run_onnx(model_path, golden_features)

    _write_json(
        output_dir / "preprocessing.json", build_preprocessing_spec(feature_names)
    )
    _write_json(
        output_dir / "model_card.json",
        build_model_card(
            model, dataset_path, len(records), metrics, epochs, learning_rate
        ),
    )
    _write_json(
        output_dir / "golden.json", build_golden(golden_records, golden_probabilities)
    )

    _write_checksums(output_dir)
    return output_dir


def _write_checksums(output_dir: Path) -> None:
    nomes = sorted(
        p.name for p in output_dir.iterdir() if p.name != "SHA256SUMS" and p.is_file()
    )
    linhas = [f"{_sha256(output_dir / name)}  {name}" for name in nomes]
    (output_dir / "SHA256SUMS").write_text(
        "\n".join(linhas) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Auxiliares
# ---------------------------------------------------------------------------

def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for bloco in iter(lambda: fh.read(65536), b""):
            digest.update(bloco)
    return digest.hexdigest()


def _jsonable(row: Mapping[str, object]) -> Dict[str, object]:
    """Converte escalares NumPy em tipos nativos serializáveis."""
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in row.items()
    }
