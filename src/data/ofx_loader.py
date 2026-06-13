"""Carregamento e parsing de arquivos OFX.

Implementa as funções iniciais da Entrega 1 (load/clean/split) sobre extratos
no formato OFX. O parser usa apenas a biblioteca padrão para o formato
SGML/header do OFX, evitando ampliar a superfície de dependências.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


@dataclass(frozen=True)
class Transaction:
    """Uma transação financeira extraída de um OFX."""

    fitid: str
    posted_at: datetime
    amount: float
    trntype: str
    memo: str


_TAG_RE = re.compile(r"<(?P<tag>[A-Z0-9.]+)>(?P<value>[^<\r\n]*)")
_STMTTRN_RE = re.compile(r"<STMTTRN>(.*?)</STMTTRN>", re.DOTALL)


def _parse_ofx_datetime(raw: str) -> datetime:
    """Converte DTPOSTED do OFX (YYYYMMDD[HHMMSS][.xxx][tz]) em datetime."""
    digits = re.sub(r"[^0-9]", "", raw)[:8]
    return datetime.strptime(digits, "%Y%m%d")


def load_transactions(path: str | Path) -> List[Transaction]:
    """Lê um arquivo OFX e retorna a lista de transações.

    Args:
        path: caminho do arquivo OFX (dentro do volume de entrada).

    Returns:
        Lista de :class:`Transaction`.

    Raises:
        FileNotFoundError: se o arquivo não existir.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Arquivo OFX não encontrado: {file_path}")

    content = file_path.read_text(encoding="latin-1", errors="ignore")
    transactions: List[Transaction] = []

    for block in _STMTTRN_RE.findall(content):
        fields = {m.group("tag"): m.group("value").strip()
                  for m in _TAG_RE.finditer(block)}
        if "TRNAMT" not in fields:
            continue
        transactions.append(
            Transaction(
                fitid=fields.get("FITID", ""),
                posted_at=_parse_ofx_datetime(fields.get("DTPOSTED", "19700101")),
                amount=float(fields["TRNAMT"].replace(",", ".")),
                trntype=fields.get("TRNTYPE", "OTHER"),
                memo=fields.get("MEMO", fields.get("NAME", "")),
            )
        )
    return transactions


def deduplicate(transactions: Iterable[Transaction]) -> List[Transaction]:
    """Remove transações duplicadas com base no FITID.

    Transações sem FITID são mantidas (não há chave confiável de deduplicação).
    """
    seen: set[str] = set()
    result: List[Transaction] = []
    for txn in transactions:
        if txn.fitid and txn.fitid in seen:
            continue
        if txn.fitid:
            seen.add(txn.fitid)
        result.append(txn)
    return result


def load_directory(input_dir: str | Path) -> List[Transaction]:
    """Carrega e deduplica todas as transações dos OFX de um diretório."""
    directory = Path(input_dir)
    all_txns: List[Transaction] = []
    for ofx_file in sorted(directory.glob("*.ofx")):
        all_txns.extend(load_transactions(ofx_file))
    return deduplicate(all_txns)
