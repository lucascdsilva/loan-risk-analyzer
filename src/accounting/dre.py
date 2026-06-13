"""Geração da DRE (Demonstração do Resultado do Exercício).

Agrega as transações classificadas por grupo e calcula o resultado do período.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

from src.accounting.chart import ClassifiedTransaction

# Ordem canônica dos grupos na DRE.
DRE_ORDER: tuple[str, ...] = (
    "Receita Bruta",
    "Deducoes",
    "Custos",
    "Despesas Operacionais",
    "Despesas Financeiras",
)


def build_dre(classified: List[ClassifiedTransaction]) -> "OrderedDict[str, float]":
    """Soma os valores por grupo da DRE e adiciona o Resultado do Período.

    Returns:
        Mapa ordenado grupo -> total, com a chave final 'Resultado do Periodo'.
    """
    totals: Dict[str, float] = {group: 0.0 for group in DRE_ORDER}
    for item in classified:
        totals.setdefault(item.group, 0.0)
        totals[item.group] += item.transaction.amount

    result: "OrderedDict[str, float]" = OrderedDict()
    for group in DRE_ORDER:
        result[group] = round(totals.get(group, 0.0), 2)
    result["Resultado do Periodo"] = round(sum(totals.values()), 2)
    return result


def format_dre(dre: "OrderedDict[str, float]") -> str:
    """Formata a DRE como texto simples para o relatório de execução."""
    lines = ["DRE — Demonstracao do Resultado do Exercicio", "-" * 44]
    for group, value in dre.items():
        lines.append(f"{group:<28} {value:>12.2f}")
    return "\n".join(lines)
