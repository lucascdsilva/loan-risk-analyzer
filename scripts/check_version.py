#!/usr/bin/env python3
"""Guarda de versão do release

Falha se a tag que está sendo publicada divergir de ``MODEL_VERSION``.

A constante em ``src/utils/config.py`` é o que entra no ``model_card.json`` e no
``golden.json`` — e é ela que o projeto B expõe em ``GET /v1/model``. Um bundle
publicado sob a tag ``v0.5.0`` mas carimbado ``v0.4.0`` mentiria sobre a própria
versão, e a divergência só apareceria no consumidor.

Uso:
    python scripts/check_version.py v0.4.0
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Executado como script solto (`python scripts/check_version.py`), o sys.path[0]
# é scripts/ — a raiz do projeto precisa entrar à mão para importar src/.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import MODEL_VERSION, PREPROCESSING_VERSION  # noqa: E402

TAG_PATTERN = re.compile(r"^v\d+\.\d+\.\d+$")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"uso: {Path(argv[0]).name} <tag>", file=sys.stderr)
        return 2

    tag = argv[1].strip()

    if not TAG_PATTERN.match(tag):
        print(
            f"tag '{tag}' fora do formato esperado vMAJOR.MINOR.PATCH",
            file=sys.stderr,
        )
        return 1

    if tag != MODEL_VERSION:
        print(
            f"tag '{tag}' diverge de MODEL_VERSION '{MODEL_VERSION}' "
            f"(src/utils/config.py). Atualize a constante antes de taguear.",
            file=sys.stderr,
        )
        return 1

    print(
        f"versão conferida: tag {tag} == MODEL_VERSION "
        f"(preprocessing_version {PREPROCESSING_VERSION})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
