"""Testes do carregamento do dataset de empréstimos."""

import unittest
from pathlib import Path

from src.data.loan_loader import LoanRecord, load_csv

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "loan_data.csv"


class TestLoanLoader(unittest.TestCase):
    def test_load_returns_records(self) -> None:
        records = load_csv(FIXTURE)
        self.assertGreater(len(records), 0)

    def test_record_types(self) -> None:
        records = load_csv(FIXTURE)
        first = records[0]
        self.assertIsInstance(first, LoanRecord)
        self.assertIsInstance(first.person_age, float)
        self.assertIsInstance(first.loan_status, int)
        self.assertIn(first.loan_status, (0, 1))

    def test_loan_status_binary(self) -> None:
        records = load_csv(FIXTURE)
        statuses = {r.loan_status for r in records}
        self.assertTrue(statuses.issubset({0, 1}))

    def test_previous_default_values(self) -> None:
        records = load_csv(FIXTURE)
        values = {r.previous_loan_defaults_on_file for r in records}
        self.assertTrue(values.issubset({"Yes", "No"}))

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_csv(FIXTURE.parent / "nao_existe.csv")


if __name__ == "__main__":
    unittest.main()
