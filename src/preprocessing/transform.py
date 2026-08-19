"""Pré-processamento do dataset de risco de crédito.

Cobre o contrato de features: descarte
de colunas fora do contrato, escala ordinal da escolaridade, one-hot com
categorias fixas, normalização e balanceamento do treino. O mesmo codificador
ajustado aqui é persistido no bundle e reusado na inferência.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from src.utils.config import RANDOM_SEED

# Escolaridade tratada como ordinal (quanto maior, mais escolaridade).
EDUCATION_ORDER = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}

# ---------------------------------------------------------------------------
# Contrato de features (v0.4.0) — 21 colunas
# ---------------------------------------------------------------------------

TARGET_COLUMN = "loan_status"

# Descartado: gênero não é usado como variável de decisão de crédito.
DROPPED_COLUMNS: Tuple[str, ...] = ("person_gender",)

# Categorias fixadas explicitamente. Garante as mesmas 21 colunas no
# treino e na inferência de um único registro
ONE_HOT_CATEGORIES: Dict[str, List[str]] = {
    "person_home_ownership": ["MORTGAGE", "OTHER", "OWN", "RENT"],
    "loan_intent": [
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE",
    ],
    "previous_loan_defaults_on_file": ["No", "Yes"],
}

CATEGORICAL_COLUMNS = list(ONE_HOT_CATEGORIES)

N_FEATURES = sum(len(c) for c in ONE_HOT_CATEGORIES.values()) + 9


def build_encoder() -> ColumnTransformer:
    """Cria o codificador de features (ainda não ajustado).

    One-hot nas categóricas com categorias fixas, ``passthrough`` no resto.
    """
    return ColumnTransformer(
        transformers=[(
            "one_hot",
            OneHotEncoder(
                categories=[ONE_HOT_CATEGORIES[c] for c in CATEGORICAL_COLUMNS],
                sparse_output=False,
                handle_unknown="error",
            ),
            CATEGORICAL_COLUMNS,
        )],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )


def prepare_frame(dataset: pd.DataFrame) -> pd.DataFrame:
    """Descarta colunas fora do contrato e aplica a escala ordinal.
    Raises:
        ValueError: se aparecer uma escolaridade fora de ``EDUCATION_ORDER``.
    """
    prepared = dataset.drop(
        columns=[c for c in DROPPED_COLUMNS if c in dataset.columns]
    )

    education = prepared["person_education"].map(EDUCATION_ORDER)
    if education.isna().any():
        desconhecidas = sorted(
            set(prepared.loc[education.isna(), "person_education"])
        )
        raise ValueError(f"Escolaridade desconhecida: {desconhecidas}")
    prepared["person_education"] = education

    return prepared


def fit_encoder(
    dataset: pd.DataFrame,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray, List[str]]:
    """Ajusta o codificador no dataset de treino.

    Args:
        dataset: DataFrame com as colunas do CSV, incluindo ``loan_status``.

    Returns:
        Tupla ``(encoder, X, y, feature_names)``. O ``encoder`` retornado é o
        artefato que precisa ser persistido para que a inferência reproduza
        exatamente esta transformação.
    """
    prepared = prepare_frame(dataset)
    target = prepared[TARGET_COLUMN].to_numpy(dtype=np.int64)
    features_df = prepared.drop(columns=[TARGET_COLUMN])

    encoder = build_encoder()
    features = encoder.fit_transform(features_df).astype(np.float64)

    return encoder, features, target, list(encoder.get_feature_names_out())


def transform_records(
    encoder: ColumnTransformer, dataset: pd.DataFrame
) -> np.ndarray:
    """Aplica um codificador já ajustado a registros novos.

    Diferente de :func:`fit_encoder`, não exige ``loan_status`` — é o caminho
    usado na inferência, onde o alvo é justamente o que se quer estimar.
    """
    prepared = prepare_frame(dataset)
    features_df = prepared.drop(columns=[TARGET_COLUMN], errors="ignore")
    return encoder.transform(features_df).astype(np.float64)


def fit_scaler(X_train: np.ndarray) -> StandardScaler:
    """Ajusta a normalização **somente** no treino, para evitar vazamento. """
    return StandardScaler().fit(X_train)


def smote_oversampling(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Balanceia as classes do treino gerando amostras sintéticas da minoria."""
    smote = SMOTE(random_state=RANDOM_SEED)
    return smote.fit_resample(X, y)
