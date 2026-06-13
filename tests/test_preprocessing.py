"""Testes do pré-processamento de texto e split."""

import unittest
from datetime import datetime

from src.data.ofx_loader import Transaction
from src.preprocessing.transform import clean_transactions, normalize_text, split_data


def _txn(memo: str) -> Transaction:
    return Transaction("x", datetime(2026, 1, 1), -1.0, "DEBIT", memo)


class TestPreprocessing(unittest.TestCase):
    def test_normalize_removes_accents_and_case(self) -> None:
        self.assertEqual(normalize_text("Pró-Labore  SÓCIO"), "pro labore socio")

    def test_normalize_strips_symbols(self) -> None:
        self.assertEqual(normalize_text("TARIFA #123 (mensal)"), "tarifa 123 mensal")

    def test_clean_transactions_applies_to_memo(self) -> None:
        cleaned = clean_transactions([_txn("ÁGUA & ESGOTO")])
        self.assertEqual(cleaned[0].memo, "agua esgoto")

    def test_split_is_deterministic_and_proportional(self) -> None:
        txns = [_txn(f"t{i}") for i in range(10)]
        train1, test1 = split_data(txns, test_ratio=0.2, seed=42)
        train2, test2 = split_data(txns, test_ratio=0.2, seed=42)
        self.assertEqual(len(test1), 2)
        self.assertEqual(len(train1), 8)
        self.assertEqual([t.memo for t in test1], [t.memo for t in test2])

    def test_split_invalid_ratio(self) -> None:
        with self.assertRaises(ValueError):
            split_data([_txn("a")], test_ratio=1.5)


if __name__ == "__main__":
    unittest.main()
