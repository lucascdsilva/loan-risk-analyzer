"""Pré-processamento das transações.

Nesta Entrega 1 cobrimos a limpeza/normalização de texto e a separação
treino/teste. A vetorização com NumPy e a conversão para tensores entram nas
etapas seguintes (NumPy / PyTorch).
"""

from __future__ import annotations

import re
import unicodedata
from typing import List, Sequence, Tuple

from src.data.ofx_loader import Transaction

_NON_ALNUM = re.compile(r"[^a-z0-9 ]+")
_MULTISPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normaliza a descrição de uma transação.

    Remove acentuação, coloca em minúsculas e elimina ruído, preservando
    o conteúdo útil para classificação (palavras e números).
    """
    no_accents = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    lowered = no_accents.lower()
    cleaned = _NON_ALNUM.sub(" ", lowered)
    return _MULTISPACE.sub(" ", cleaned).strip()


def clean_transactions(transactions: Sequence[Transaction]) -> List[Transaction]:
    """Aplica a normalização de texto ao memo de cada transação."""
    return [
        Transaction(
            fitid=t.fitid,
            posted_at=t.posted_at,
            amount=t.amount,
            trntype=t.trntype,
            memo=normalize_text(t.memo),
        )
        for t in transactions
    ]


def split_data(
    transactions: Sequence[Transaction],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[Transaction], List[Transaction]]:
    """Divide as transações em treino e teste de forma determinística.

    Args:
        transactions: transações já limpas.
        test_ratio: fração reservada para teste (0 < ratio < 1).
        seed: semente para embaralhamento reprodutível.

    Returns:
        Tupla (treino, teste).
    """
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio deve estar entre 0 e 1.")

    import random

    items = list(transactions)
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = int(len(items) * (1.0 - test_ratio))
    return items[:cut], items[cut:]
