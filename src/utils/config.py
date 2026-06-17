"""Configuração central do loan-risk-analyzer.

Mantém caminhos e parâmetros em um único ponto, alimentados por variáveis de
ambiente. Dentro do container, DATA_PATH e OUTPUT_DIR apontam para volumes
montados explicitamente — o código nunca acessa caminhos arbitrários do host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

RANDOM_SEED: int = 42


@dataclass(frozen=True)
class Settings:
    """Configurações imutáveis derivadas do ambiente."""

    data_path: Path
    output_dir: Path

    @staticmethod
    def from_env() -> "Settings":
        """Constrói as configurações a partir de variáveis de ambiente.

        No container, os diretórios são montados como volumes dedicados:
        - /data/loan_data.csv (somente leitura) -> dataset de empréstimos
        - /data/output        (escrita)          -> resultados gerados
        """
        data_path = Path(
            os.environ.get("LOANRISK_DATA_PATH", "data/loan_data.csv")
        )
        output_dir = Path(
            os.environ.get("LOANRISK_OUTPUT_DIR", "data/output")
        )
        return Settings(data_path=data_path, output_dir=output_dir)
