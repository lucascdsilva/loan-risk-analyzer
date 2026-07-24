"""Carregamento do dataset de risco de crédito (CSV).

Lê o arquivo CSV de empréstimos, retorna registros como um DataFrame Pandas.
"""

from __future__ import annotations

import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import List
import numpy as np
import csv
import dataclasses


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


def load_csv(path: str | Path) -> pd.DataFrame:
    """Lê o CSV de empréstimos e retorna os registros em um DataFrame.

    Args:
        path: caminho do arquivo CSV.

    Returns:
        Registros em um DataFrame Pandas

    Raises:
        FileNotFoundError: se o arquivo não existir
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

    return pd.read_csv(path)

def _write_csv(path: Path, records) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in dataclasses.fields(records[0])])
        for r in records:
            writer.writerow(dataclasses.astuple(r))

def _parse_row(row: np.void) -> LoanRecord:
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


    