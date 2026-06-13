"""Testes da classificação baseline e do fechamento da DRE."""

import unittest
from datetime import datetime

from src.accounting.chart import build_chart_of_accounts, classify
from src.accounting.dre import build_dre
from src.data.ofx_loader import Transaction
from src.preprocessing.transform import clean_transactions


def _txn(amount: float, memo: str) -> Transaction:
    return Transaction("x", datetime(2026, 1, 1), amount, "DEBIT", memo)


class TestAccounting(unittest.TestCase):
    def test_classify_rent_as_ocupacao(self) -> None:
        cleaned = clean_transactions([_txn(-1200.0, "ALUGUEL LOJA")])
        result = classify(cleaned[0])
        self.assertEqual(result.group, "Despesas Operacionais")
        self.assertEqual(result.account, "Ocupacao")

    def test_dre_closes_with_sum_of_transactions(self) -> None:
        txns = clean_transactions([
            _txn(1500.0, "PIX RECEBIDO"),
            _txn(-800.0, "PAGAMENTO FORNECEDOR"),
            _txn(-200.0, "TARIFA BANCARIA"),
        ])
        classified = build_chart_of_accounts(txns)
        dre = build_dre(classified)
        total_txns = round(sum(t.amount for t in txns), 2)
        self.assertAlmostEqual(dre["Resultado do Periodo"], total_txns)


if __name__ == "__main__":
    unittest.main()
