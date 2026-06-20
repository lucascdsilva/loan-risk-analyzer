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

from src.data.loan_loader import LoanRecord

# Escolaridade tratada como ordinal (quanto maior, mais escolaridade).
EDUCATION_ORDER = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}

def encode_features(record: pd.DataFrame) -> np.ndarray:
    """Codifica um registro do dataset em um vetor numérico.

    Args:
        record: linha do DataFrame representando um empréstimo.

    Returns:
        Vetor NumPy com features numéricas e categóricas codificadas.

    Raises:
        ValueError: se algum valor for inválido ou ausente.
    """


def clean_dataset(records: np.ndarray) -> np.ndarray:
    """Descarta/ajusta dados faltantes ou inconsistentes.""

def scale_features(features: np.ndarray) -> np.ndarray:
    """Aplica normalização ou padronização às features numéricas."""

    
def split_data(
    records: Sequence[CleanedRecord],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[CleanedRecord], List[CleanedRecord]]:
    """Divide os registros em treino e teste de forma determinística.

    Args:
        records: registros já pré-processados.
        test_ratio: fração reservada para teste (0 < ratio < 1).
        seed: semente para embaralhamento reprodutível.

    Returns:
        Tupla (treino, teste).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio deve estar entre 0 e 1.")
    items = list(records)
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = int(len(items) * (1.0 - test_ratio))
    return items[:cut], items[cut:]
