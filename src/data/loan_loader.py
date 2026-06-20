"""Carregamento do dataset de risco de crédito (CSV).

Lê o arquivo CSV de empréstimos, retorna registros como um DataFrame Pandas.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd

EXPECTED_COLUMNS = {
    "person_age", "person_gender", "person_education",
    "person_income", "person_emp_exp", "person_home_ownership",
    "loan_amnt", "loan_intent", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score",
    "previous_loan_defaults_on_file", "loan_status",
}

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

    dataset = pd.read_csv(path)

    return dataset