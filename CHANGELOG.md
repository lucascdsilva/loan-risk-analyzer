# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento segue [SemVer](https://semver.org/lang/pt-BR/), onde cada
**MINOR** corresponde a uma entrega do roadmap do Projeto Integrador.

| Versão | Entrega | Status |
|--------|---------|--------|
| [v0.1.0](#v010--entrega-1) | Entrega 1 — pré-processamento e base de execução | Concluída |
| [Unreleased](#unreleased) | Etapa 4 — vetorização com NumPy | Em andamento |
| _planejado_ | Etapa 5–6 — classificador neural (PyTorch) | — |
| _planejado_ | Etapa 7 — experimentos e hiperparâmetros | — |
| _planejado_ | Etapas 9–11 — visão, requisitos e arquitetura → **v1.0.0** | — |

## [Unreleased]

### Etapa 4 — Vetorização das features com NumPy

- _Em andamento na branch `feat/numpy-and-build_feature_matrix`._

## [v0.1.0] — Entrega 1

_2026-06-19_

Primeira entrega: funções, modularização, repositório e base de execução
isolada em container.

### Adicionado

- Carregamento e validação dos 45 000 registros de `data/loan_data.csv`
  (`src/data/loan_loader.py`).
- Codificação de variáveis categóricas — gênero → binário, escolaridade →
  ordinal, default anterior → binário — preservando as demais para one-hot
  encoding posterior (`src/preprocessing/transform.py`).
- Divisão determinística em treino (80 %) e teste (20 %), com geração de
  `train.csv`, `test.csv` e `summary.txt` em `data/output/`.
- Execução isolada com Docker: build multi-stage, usuário non-root,
  filesystem raiz somente-leitura e `network_mode: none`
  (`Dockerfile`, `docker-compose.yml`).
- Suporte a Jupyter Notebook para análise exploratória, com dependências de
  desenvolvimento separadas do runtime (`requirements-dev.in`/`.txt`).
- Suíte de testes com `unittest` (16 testes) cobrindo carga de dados e
  pré-processamento.
- Reprodutibilidade de dependências com `requirements.txt` gerado por
  `pip-compile` com hashes fixados (`--require-hashes`) e auditoria de
  vulnerabilidades via `make audit`.
- `Makefile` com atalhos: `build`, `run`, `test`, `lock`, `lock-dev`,
  `audit`, `notebook`, `clean`.

### Notas

- Projeto resultante do pivot para **análise de risco de empréstimos**
  (antes: cálculo de DRE).

[Unreleased]: https://github.com/lucascdsilva/auto-dre/compare/v0.1.0...HEAD
[v0.1.0]: https://github.com/lucascdsilva/auto-dre/releases/tag/v0.1.0
