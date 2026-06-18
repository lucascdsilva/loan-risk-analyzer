"""Pré-processamento do dataset de risco de crédito.

Cobre codificação de variáveis categóricas e split treino/teste.
A vetorização com NumPy e a conversão para tensores entram nas etapas seguintes.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

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
