"""Ponto de entrada do loan-risk-analyzer (pipeline ponta a ponta da Entrega 1).

Fluxo: carrega CSV de empréstimos -> codifica variáveis categóricas
-> split treino/teste -> grava resultados no diretório de saída.

Os diretórios vêm de variáveis de ambiente e, no container, correspondem a
volumes montados explicitamente. O script não acessa o sistema de arquivos do
host fora desses volumes.
"""

from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

from src.data.loan_loader import load_csv
from src.preprocessing.transform import clean_dataset, split_data
from src.utils.config import Settings


def run(settings: Settings) -> int:
    """Executa o pipeline e retorna um código de saída (0 = sucesso)."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_csv(settings.data_path)
    if not records:
        print(f"Nenhum registro encontrado em {settings.data_path}", file=sys.stderr)
        return 1

    cleaned = clean_dataset(records)
    train, test = split_data(cleaned)

    _write_csv(settings.output_dir / "train.csv", train)
    _write_csv(settings.output_dir / "test.csv", test)

    summary = _build_summary(records, cleaned, train, test)
    (settings.output_dir / "summary.txt").write_text(summary + "\n", encoding="utf-8")

    print(summary)
    print(f"\nResultados gravados em: {settings.output_dir}")
    return 0


def _build_summary(records, cleaned, train, test) -> str:
    default_rate = (
        sum(r.loan_status for r in cleaned) / len(cleaned) * 100
        if cleaned else 0.0
    )
    lines = [
        "loan-risk-analyzer — Sumário do Pré-processamento",
        "-" * 50,
        f"{'Registros carregados':<30} {len(records):>10}",
        f"{'Após codificação':<30} {len(cleaned):>10}",
        f"{'Treino':<30} {len(train):>10}",
        f"{'Teste':<30} {len(test):>10}",
        f"{'Taxa de default (%)':<30} {default_rate:>9.1f}%",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, records) -> None:
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([f.name for f in dataclasses.fields(records[0])])
        for r in records:
            writer.writerow(dataclasses.astuple(r))


def main() -> None:
    """Função principal."""
    sys.exit(run(Settings.from_env()))


if __name__ == "__main__":
    main()
