#!/usr/bin/env python3
"""Corpo do GitHub Release, gerado do bundle

As métricas divulgadas saem do ``model_card.json`` do artefato que está sendo
publicado — nunca de um número copiado à mão do CHANGELOG. Se o treino do CI
render diferente do que está documentado, é o release que conta.

Uso:
    python scripts/release_notes.py v0.4.0 > release-notes.md
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Mapping

RAIZ = Path(__file__).resolve().parents[1]
BUNDLE_DIR = Path(os.environ.get("LOANRISK_MODELS_DIR", RAIZ / "data" / "models")) / "bundle"
DIST_DIR = Path(os.environ.get("DIST_DIR", RAIZ / "dist"))

ARQUIVOS_DO_BUNDLE = [
    ("`model.onnx`", "Padronização + rede + sigmoid **dentro do grafo**. Entrada `float32[batch, 21]` crua, saída `float32[batch]` = probabilidade de default"),
    ("`preprocessing.json`", "Categorias one-hot em ordem fixa, escala ordinal de escolaridade e a ordem das 21 features"),
    ("`model_card.json`", "Versões, arquitetura, hiperparâmetros, dataset e métricas"),
    ("`golden.json`", "Registros crus + probabilidade esperada, para o consumidor provar que reproduz este modelo"),
    ("`SHA256SUMS`", "Integridade dos quatro arquivos acima"),
]

ORDEM_METRICAS = ["AUROC", "Accuracy", "Precision", "Recall", "F1-Score"]


def _ler_json(caminho: Path) -> Mapping[str, object]:
    return json.loads(caminho.read_text(encoding="utf-8"))


def _sha256_do_asset(nome: str) -> str:
    """Lê o checksum já calculado por pack_bundle.sh, se existir."""
    arquivo = DIST_DIR / f"{nome}.sha256"
    if not arquivo.is_file():
        return "—"
    return arquivo.read_text(encoding="utf-8").split()[0]


def _repo_url() -> str:
    servidor = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repo = os.environ.get("GITHUB_REPOSITORY", "lucascdsilva/loan-risk-analyzer")
    return f"{servidor}/{repo}"


def montar(versao: str) -> str:
    card = _ler_json(BUNDLE_DIR / "model_card.json")
    spec = _ler_json(BUNDLE_DIR / "preprocessing.json")
    golden = _ler_json(BUNDLE_DIR / "golden.json")

    arquitetura = card["architecture"]
    treino = card["training"]
    dataset = card["dataset"]
    metricas = card["metrics"]
    export = card["export"]
    threshold = card["default_threshold"]

    tarball = f"loan-risk-model-{versao}.tar.gz"
    url_download = f"{_repo_url()}/releases/download/{versao}/{tarball}"

    linhas: list[str] = []
    add = linhas.append

    add(f"**Bundle de inferência `{versao}`** — o artefato que o "
        "`loan-risk-ml-service` consome. Modelo, contrato de pré-processamento, "
        "métricas e registros de referência, empacotados e verificáveis.")
    add("")
    add("Gerado por `.github/workflows/release.yml`: treinado do zero no CI a "
        f"partir da tag `{versao}`, com seed `{treino['seed']}`.")
    add("")

    add("## Artefatos")
    add("")
    add("| Asset | SHA-256 |")
    add("|---|---|")
    add(f"| `{tarball}` | `{_sha256_do_asset(tarball)}` |")
    add("")

    add("## Modelo")
    add("")
    add("| Campo | Valor |")
    add("|---|---|")
    add(f"| `model_version` | `{card['model_version']}` |")
    add(f"| `preprocessing_version` | `{card['preprocessing_version']}` |")
    add(f"| Arquitetura | `{card['model']}` — MLP "
        f"{arquitetura['in_features']} → {arquitetura['hidden_units']} "
        f"({arquitetura['activation']}) → 1 ({arquitetura['output']}) |")
    add(f"| Parâmetros | {arquitetura['parameters']} |")
    add(f"| Features | {spec['n_features']} (categoria desconhecida: "
        f"`{spec['unknown_category_policy']}`) |")
    add(f"| Treino | {treino['epochs']} épocas · {treino['optimizer']} "
        f"lr={treino['learning_rate']} · {treino['loss']} |")
    add(f"| Balanceamento | {treino['oversampling']} |")
    add(f"| Split | test_size={treino['test_size']} · seed={treino['seed']} |")
    add(f"| Threshold sugerido | {threshold} |")
    add(f"| Export | {export['format'].upper()} opset {export['opset']} · "
        f"torch {export['torch']} |")
    add(f"| Treinado em | {card['created_at']} |")
    add("")

    add(f"## Métricas (conjunto de teste, threshold {threshold})")
    add("")
    add("| Métrica | Valor |")
    add("|---|---|")
    for nome in ORDEM_METRICAS:
        if nome in metricas:
            add(f"| {nome} | {metricas[nome]:.4f} |")
    add("")

    if "confusion_matrix" in metricas:
        (tn, fp), (fn, tp) = metricas["confusion_matrix"]
        add("**Matriz de confusão**")
        add("")
        add("| | pred 0 | pred 1 |")
        add("|---|---:|---:|")
        add(f"| **real 0** | {tn} | {fp} |")
        add(f"| **real 1** | {fn} | {tp} |")
        add("")

    add("## Dataset")
    add("")
    add(f"`{dataset['name']}` — {dataset['rows']:,} registros · "
        f"SHA-256 `{dataset['sha256']}`".replace(",", " "))
    add("")

    add("## Conteúdo do bundle")
    add("")
    add("| Arquivo | Conteúdo |")
    add("|---|---|")
    for nome, descricao in ARQUIVOS_DO_BUNDLE:
        add(f"| {nome} | {descricao} |")
    add("")

    add("## Como consumir")
    add("")
    add("```bash")
    add(f"curl -LO {url_download}")
    add(f"curl -LO {url_download}.sha256")
    add(f"sha256sum -c {tarball}.sha256")
    add(f"tar xzf {tarball}")
    add("cd bundle && sha256sum -c SHA256SUMS")
    add("```")
    add("")
    add(f"O `golden.json` traz {len(golden['records'])} registros crus com a "
        f"probabilidade que **este** bundle produz, com tolerância "
        f"`{golden['tolerance']}`. O consumidor roda isso no próprio CI: se o "
        "encoder dele divergir deste contrato, o teste falha.")

    return "\n".join(linhas) + "\n"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"uso: {Path(argv[0]).name} <versao>", file=sys.stderr)
        return 2
    if not BUNDLE_DIR.is_dir():
        print(f"bundle não encontrado em {BUNDLE_DIR}", file=sys.stderr)
        return 1
    sys.stdout.write(montar(argv[1].strip()))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
