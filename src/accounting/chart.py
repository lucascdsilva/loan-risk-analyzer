"""Plano de contas e classificação baseline por regras.

Na Entrega 1 a classificação é feita por regras simples (baseline heurístico).
Esse baseline serve de referência e de gerador de rótulos iniciais
(bootstrapping) para o classificador neural das próximas etapas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.data.ofx_loader import Transaction

# Mapa: grupo da DRE -> conta -> palavras-chave esperadas no memo normalizado.
RULES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "Receita Bruta": {
        "Vendas/Recebimentos": ("pix recebido", "ted", "deposito", "recebimento", "venda"),
    },
    "Deducoes": {
        "Impostos sobre vendas": ("das", "simples", "iss", "darf"),
    },
    "Custos": {
        "Fornecedores/Mercadorias": ("fornecedor", "compra", "insumo", "mercadoria"),
    },
    "Despesas Operacionais": {
        "Pessoal": ("salario", "pro labore", "pro-labore", "fgts", "folha"),
        "Ocupacao": ("aluguel", "energia", "agua", "internet", "telefone"),
    },
    "Despesas Financeiras": {
        "Tarifas e juros": ("tarifa", "juros", "iof", "anuidade"),
    },
}

UNCLASSIFIED = ("Nao classificado", "A revisar")


@dataclass(frozen=True)
class ClassifiedTransaction:
    """Transação com sua conta e grupo de DRE atribuídos."""

    transaction: Transaction
    group: str
    account: str


def classify(transaction: Transaction) -> ClassifiedTransaction:
    """Classifica uma transação no plano de contas via regras de baseline."""
    memo = transaction.memo
    for group, accounts in RULES.items():
        for account, keywords in accounts.items():
            if any(kw in memo for kw in keywords):
                return ClassifiedTransaction(transaction, group, account)

    # Heurística de fallback: sinal do valor separa receita de despesa.
    if transaction.amount >= 0:
        return ClassifiedTransaction(transaction, "Receita Bruta", "Vendas/Recebimentos")
    return ClassifiedTransaction(transaction, *UNCLASSIFIED)


def build_chart_of_accounts(
    transactions: List[Transaction],
) -> List[ClassifiedTransaction]:
    """Classifica todas as transações, construindo o plano de contas do período."""
    return [classify(t) for t in transactions]
