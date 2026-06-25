"""Pré-processamento do dataset de risco de crédito.

Cobre codificação de variáveis categóricas e split treino/teste.
A vetorização com NumPy e a conversão para tensores entram nas etapas seguintes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

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

def encode_features(dataset: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Codifica as features categóricas do dataset em valores numéricos.

    Args:
        dataset: DataFrame com os registros de empréstimos.

    Returns:
        Tupla (features codificadas, targets, nomes das features).
    """
    # Label encoding considerando a ordem natural das categorias
    dataset["person_education"] = dataset["person_education"].map(EDUCATION_ORDER)

    #One Hot Encoding
    transformer = ColumnTransformer(
        transformers = [
            ('one_hot', OneHotEncoder(sparse_output=False), CATEGORICAL_COLUMNS)
        ], remainder='passthrough' # Mantém as outras colunas (como 'Valor') sem alterações
        , verbose_feature_names_out=False
    )
    encoded_dataset = transformer.fit_transform(dataset)

    # Separando features e target
    features = encoded_dataset[:,:-1]
    target = encoded_dataset[:,-1].astype(int)
    feature_names = transformer.get_feature_names_out()
    feature_names = feature_names[:-1]
        
    return features, target, feature_names

def clean_dataset(records: np.ndarray) -> np.ndarray:
    """Descarta/ajusta dados faltantes ou inconsistentes."""

def scale_dataset(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Aplica normalização às features numéricas."""
    # Atenção para vazamento de dados. Só aplicar fit nos dados de treino
    #TODO: Implementar normalização
    return X_train, X_test

    
def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_ratio: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Divide os registros em treino e teste

    Args:
        X: features do dataset.
        y: targets do dataset.
        test_ratio: fração reservada para teste (0 < ratio < 1).
        random_state: semente para embaralhamento reprodutível.

    Returns:
        Tupla (X_train, X_test, y_train, y_test).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio deve estar entre 0 e 1.")

    return train_test_split(X, y, test_size=test_ratio, stratify=y, random_state=random_state)