"""Pré-processamento do dataset de risco de crédito.

Cobre codificação de variáveis categóricas e split treino/teste.
A vetorização com NumPy e a conversão para tensores entram nas etapas seguintes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Escolaridade tratada como ordinal (quanto maior, mais escolaridade).
EDUCATION_ORDER = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}

CATEGORICAL_COLUMNS = [
    "person_gender",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

def encode_features(dataset: pd.DataFrame) -> np.ndarray:
    """Codifica as features categóricas do dataset em valores numéricos.

    Args:
        dataset: DataFrame com os registros de empréstimos.

    Returns:
        Vetor NumPy com features numéricas e categóricas codificadas.
    """
    dataset = dataset.copy()
    # Label encoding considerando a ordem natural das categorias
    dataset["person_education"] = dataset["person_education"].map(EDUCATION_ORDER)
    
    # One-hot encoding das categorias restantes
    encoder = OneHotEncoder(sparse_output=False)
    categorical_encoded = encoder.fit_transform(dataset[CATEGORICAL_COLUMNS])
    # Extrai as features numéricas (todas as colunas exceto as categóricas) e converte para NumPy
    numeric_array = dataset.drop(columns=CATEGORICAL_COLUMNS).to_numpy()
    
    # Retorna a combinação das features numéricas com as categóricas codificadas
    return np.hstack([numeric_array, categorical_encoded])

def clean_dataset(records: np.ndarray) -> np.ndarray:
    """Descarta/ajusta dados faltantes ou inconsistentes."""

def scale_features(features: np.ndarray) -> np.ndarray:
    """Aplica normalização ou padronização às features numéricas."""

    
# def split_data(
#     records: Sequence[CleanedRecord],
#     test_ratio: float = 0.2,
#     seed: int = 42,
# ) -> Tuple[List[CleanedRecord], List[CleanedRecord]]:
#     """Divide os registros em treino e teste de forma determinística.

#     Args:
#         records: registros já pré-processados.
#         test_ratio: fração reservada para teste (0 < ratio < 1).
#         seed: semente para embaralhamento reprodutível.

#     Returns:
#         Tupla (treino, teste).
#     """
#     if not 0.0 < test_ratio < 1.0:
#         raise ValueError("test_ratio deve estar entre 0 e 1.")
#     items = list(records)
#     rng = random.Random(seed)
#     rng.shuffle(items)
#     cut = int(len(items) * (1.0 - test_ratio))
#     return items[:cut], items[cut:]