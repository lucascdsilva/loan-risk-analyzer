"""Testes do carregamento do dataset de empréstimos."""

import unittest
from pathlib import Path

import pandas as pd

from src.data.loan_loader import EXPECTED_COLUMNS, load_csv

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "loan_data.csv"


class TestLoanLoader(unittest.TestCase):
    def test_load_returns_records(self) -> None:
        dataset = load_csv(FIXTURE)
        self.assertIsInstance(dataset, pd.DataFrame)
        self.assertGreater(len(dataset), 0)

    def test_expected_columns_are_present(self) -> None:
        dataset = load_csv(FIXTURE)
        self.assertTrue(EXPECTED_COLUMNS.issubset(dataset.columns))

    def test_numeric_columns_are_numeric(self) -> None:
        dataset = load_csv(FIXTURE)
        for column in ("person_age", "person_income", "loan_amnt",
                       "loan_int_rate", "credit_score"):
            with self.subTest(column=column):
                self.assertTrue(pd.api.types.is_numeric_dtype(dataset[column]))

    def test_loan_status_binary(self) -> None:
        dataset = load_csv(FIXTURE)
        self.assertTrue(dataset["loan_status"].isin({0, 1}).all())

    def test_previous_default_values(self) -> None:
        dataset = load_csv(FIXTURE)
        valores = dataset["previous_loan_defaults_on_file"]
        self.assertTrue(valores.isin({"Yes", "No"}).all())

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_csv(FIXTURE.parent / "nao_existe.csv")


if __name__ == "__main__":
    unittest.main()
