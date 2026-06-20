"""Pré-processamento do dataset de risco de crédito.

Cobre codificação de variáveis categóricas, vetorização com NumPy (matriz de
features pronta para o modelo) e split treino/teste. A conversão para tensores
entra nas etapas seguintes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

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


@dataclass(frozen=True)
class CleanedRecord:
    """Registro com features numéricas e categóricas já codificadas."""

    person_age: float
    person_income: float
    person_emp_exp: int
    loan_amnt: float
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    gender_female: int      # 1 = feminino, 0 = masculino
    education_level: int    # nível ordinal de escolaridade
    home_ownership: str     # categoria preservada (one-hot nas etapas seguintes)
    loan_intent: str        # finalidade do empréstimo
    previous_default: int   # 1 = "Yes", 0 = "No"
    loan_status: int        # variável alvo: 0 = sem default, 1 = default


def encode_record(record: LoanRecord) -> CleanedRecord:
    """Codifica as variáveis categóricas de um único LoanRecord."""
    return CleanedRecord(
        person_age=record.person_age,
        person_income=record.person_income,
        person_emp_exp=record.person_emp_exp,
        loan_amnt=record.loan_amnt,
        loan_int_rate=record.loan_int_rate,
        loan_percent_income=record.loan_percent_income,
        cb_person_cred_hist_length=record.cb_person_cred_hist_length,
        credit_score=record.credit_score,
        gender_female=1 if record.person_gender.lower() == "female" else 0,
        education_level=EDUCATION_ORDER.get(record.person_education, -1),
        home_ownership=record.person_home_ownership,
        loan_intent=record.loan_intent,
        previous_default=1 if record.previous_loan_defaults_on_file == "Yes" else 0,
        loan_status=record.loan_status,
    )


def clean_dataset(records: Sequence[LoanRecord]) -> List[CleanedRecord]:
    """Codifica todos os registros, descartando linhas com erro de parsing."""
    result: List[CleanedRecord] = []
    for r in records:
        try:
            result.append(encode_record(r))
        except (ValueError, KeyError):
            continue
    return result


# Features numéricas e ordinais já prontas para entrar direto na matriz.
NUMERIC_FEATURES = (
    "person_age", "person_income", "person_emp_exp", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
    "credit_score", "gender_female", "education_level", "previous_default",
)

# Categóricas restantes que recebem codificação one-hot na vetorização.
ONEHOT_FEATURES = ("home_ownership", "loan_intent")


def build_feature_matrix(
    records: Sequence[CleanedRecord],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Vetoriza os registros codificados em uma matriz de features NumPy.

    As features numéricas/ordinais são empilhadas diretamente e as categóricas
    restantes (``home_ownership``, ``loan_intent``) recebem codificação one-hot.

    Args:
        records: registros já passados por :func:`encode_record`.

    Returns:
        Tupla ``(X, y, feature_names)`` onde ``X`` é a matriz
        ``(n_amostras, n_features)`` em ``float64``, ``y`` é o vetor alvo
        ``loan_status`` em ``int64`` e ``feature_names`` são os nomes das
        colunas de ``X`` na ordem.
    """
    items = list(records)
    if not items:
        return (
            np.empty((0, 0), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            [],
        )

    numeric = np.array(
        [[getattr(r, name) for name in NUMERIC_FEATURES] for r in items],
        dtype=np.float64,
    )

    feature_names = list(NUMERIC_FEATURES)
    onehot_blocks: List[np.ndarray] = []
    for field in ONEHOT_FEATURES:
        values = np.array([getattr(r, field) for r in items])
        categories = np.unique(values)
        # Broadcasting: (n, 1) == (1, k) -> matriz one-hot (n, k).
        onehot_blocks.append((values[:, None] == categories[None, :]).astype(np.float64))
        feature_names.extend(f"{field}={c}" for c in categories)

    features = np.hstack([numeric, *onehot_blocks])
    target = np.array([r.loan_status for r in items], dtype=np.int64)
    return features, target, feature_names


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
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    shuffled = [items[i] for i in order]
    cut = int(len(items) * (1.0 - test_ratio))
    return shuffled[:cut], shuffled[cut:]
