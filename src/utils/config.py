"""Configuração central do AutoDRE.

Mantém caminhos e parâmetros em um único ponto, alimentados por variáveis de
ambiente. Dentro do container, INPUT_DIR e OUTPUT_DIR apontam para volumes
montados explicitamente — o código nunca acessa caminhos arbitrários do host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Sementes fixas para reprodutibilidade (RNF02).
RANDOM_SEED: int = 42


@dataclass(frozen=True)
class Settings:
    """Configurações imutáveis derivadas do ambiente."""

    input_dir: Path
    output_dir: Path

    @staticmethod
    def from_env() -> "Settings":
        """Constrói as configurações a partir de variáveis de ambiente.

        No container, os diretórios são montados como volumes dedicados:
        - /data/input  (somente leitura) -> arquivos OFX do usuário
        - /data/output (escrita)         -> resultados gerados
        """
        input_dir = Path(os.environ.get("AUTODRE_INPUT_DIR", "/data/input"))
        output_dir = Path(os.environ.get("AUTODRE_OUTPUT_DIR", "/data/output"))
        return Settings(input_dir=input_dir, output_dir=output_dir)
