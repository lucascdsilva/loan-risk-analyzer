"""Testes do carregamento e deduplicação de OFX."""

import unittest
from pathlib import Path

from src.data.ofx_loader import deduplicate, load_transactions

FIXTURE = Path(__file__).resolve().parents[1] / "data" / "input" / "exemplo.ofx"


class TestOfxLoader(unittest.TestCase):
    def test_load_parses_transactions(self) -> None:
        txns = load_transactions(FIXTURE)
        self.assertEqual(len(txns), 6)
        self.assertEqual(txns[0].fitid, "0001")
        self.assertAlmostEqual(txns[0].amount, 1500.00)
        self.assertEqual(txns[0].posted_at.year, 2026)

    def test_deduplicate_removes_repeated_fitid(self) -> None:
        txns = load_transactions(FIXTURE)
        unique = deduplicate(txns)
        # FITID 0001 aparece duas vezes no fixture.
        self.assertEqual(len(unique), 5)

    def test_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_transactions(FIXTURE.parent / "nao_existe.ofx")


if __name__ == "__main__":
    unittest.main()
