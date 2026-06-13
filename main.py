"""Ponto de entrada do AutoDRE (pipeline ponta a ponta da Entrega 1).

Fluxo: carrega OFX do diretório de entrada -> limpa -> classifica (baseline)
-> gera DRE -> escreve resultados no diretório de saída.

Os diretórios vêm de variáveis de ambiente e, no container, correspondem a
volumes montados explicitamente. O script não acessa o sistema de arquivos do
host fora desses volumes.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from src.accounting.chart import build_chart_of_accounts
from src.accounting.dre import build_dre, format_dre
from src.data.ofx_loader import load_directory
from src.preprocessing.transform import clean_transactions
from src.utils.config import Settings


def run(settings: Settings) -> int:
    """Executa o pipeline e retorna um código de saída (0 = sucesso)."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    transactions = load_directory(settings.input_dir)
    if not transactions:
        print(f"Nenhuma transacao encontrada em {settings.input_dir}", file=sys.stderr)
        return 1

    cleaned = clean_transactions(transactions)
    classified = build_chart_of_accounts(cleaned)
    dre = build_dre(classified)

    _write_transactions_csv(settings.output_dir / "transacoes.csv", classified)
    _write_dre_csv(settings.output_dir / "dre.csv", dre)
    (settings.output_dir / "relatorio.txt").write_text(
        format_dre(dre) + f"\n\nTransacoes processadas: {len(classified)}\n",
        encoding="utf-8",
    )

    print(format_dre(dre))
    print(f"\nResultados gravados em: {settings.output_dir}")
    return 0


def _write_transactions_csv(path: Path, classified) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["data", "valor", "tipo", "grupo", "conta", "descricao"])
        for item in classified:
            t = item.transaction
            writer.writerow(
                [t.posted_at.date(), f"{t.amount:.2f}", t.trntype,
                 item.group, item.account, t.memo]
            )


def _write_dre_csv(path: Path, dre) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["grupo", "total"])
        for group, total in dre.items():
            writer.writerow([group, f"{total:.2f}"])


def main() -> None:
    """Função principal."""
    sys.exit(run(Settings.from_env()))


if __name__ == "__main__":
    main()
