"""Carregamento do dataset de risco de crédito (CSV).

Lê o arquivo CSV de empréstimos, valida as colunas esperadas e retorna
registros como dataclasses imutáveis, prontos para o pré-processamento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


EXPECTED_COLUMNS = {
    "person_age", "person_gender", "person_education",
    "person_income", "person_emp_exp", "person_home_ownership",
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
    "previous_loan_defaults_on_file", "loan_status",
}


@dataclass(frozen=True)
class LoanRecord:
    """Uma solicitação de empréstimo com seus atributos brutos."""

    person_age: float
    person_gender: str
    person_education: str
    person_income: float
    person_emp_exp: int
    person_home_ownership: str
    loan_amnt: float
    loan_intent: str
    loan_int_rate: float
    loan_percent_income: float
    cb_person_cred_hist_length: float
    credit_score: int
    previous_loan_defaults_on_file: str   # "Yes" / "No"
    loan_status: int                      # 0 = sem default, 1 = default


def load_csv(path: str | Path) -> List[LoanRecord]:
    """Lê o CSV de empréstimos e retorna a lista de registros.

    Args:
        path: caminho do arquivo CSV.

    Returns:
        Lista de :class:`LoanRecord`.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se colunas obrigatórias estiverem ausentes.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

    table = np.genfromtxt(
        file_path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
        comments=None,
    )

    if table.dtype.names is None:
        return []
    missing = EXPECTED_COLUMNS - set(table.dtype.names)
    if missing:
        raise ValueError(f"Colunas ausentes no CSV: {missing}")

    # np.atleast_1d garante iteração mesmo quando há uma única linha de dados.
    return [_parse_row(row) for row in np.atleast_1d(table)]


def _parse_row(row: dict) -> LoanRecord:
    return LoanRecord(
        person_age=float(row["person_age"]),
        person_gender=row["person_gender"].strip(),
        person_education=row["person_education"].strip(),
        person_income=float(row["person_income"]),
        person_emp_exp=int(float(row["person_emp_exp"])),
        person_home_ownership=row["person_home_ownership"].strip(),
        loan_amnt=float(row["loan_amnt"]),
        loan_intent=row["loan_intent"].strip(),
        loan_int_rate=float(row["loan_int_rate"]),
        loan_percent_income=float(row["loan_percent_income"]),
        cb_person_cred_hist_length=float(row["cb_person_cred_hist_length"]),
        credit_score=int(float(row["credit_score"])),
        previous_loan_defaults_on_file=row["previous_loan_defaults_on_file"].strip(),
        loan_status=int(row["loan_status"]),
    )
