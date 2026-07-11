# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento segue [SemVer](https://semver.org/lang/pt-BR/), onde cada
**MINOR** corresponde a uma entrega do roadmap do Projeto Integrador.

| Versão | Entrega | Status |
|--------|---------|--------|
| [v0.1.0](#v010--entrega-1) | Entrega 1 — pré-processamento e base de execução | Concluída |
| [v0.2.0](#v020--etapa-4) | Etapa 4 — vetorização com NumPy | Concluída |
| _planejado_ | Etapa 5–6 — classificador neural (PyTorch) | — |
| _planejado_ | Etapa 7 — experimentos e hiperparâmetros | — |
| _planejado_ | Etapas 9–11 — visão, requisitos e arquitetura → **v1.0.0** | — |

## [Unreleased]

_Sem mudanças registradas. Próxima entrega: Etapa 5–6 — classificador neural (PyTorch)._

## [v0.2.0] — Etapa 4

_2026-07-11_

### Vetorização das features com NumPy

Vetorização dos registros pré-processados em uma matriz de features NumPy
pronta para o modelo. A rota baseada em `numpy` passa a ser o padrão do
pipeline; a rota alternativa com `scikit-learn` é mantida como referência.

#### Adicionado

- **`build_feature_matrix`** (`src/preprocessing/transform.py`) — núcleo da
  etapa: converte os registros codificados em `(X, y, feature_names)`, onde
  `X` é a matriz `(n_amostras, n_features)` em `float64` e `y` é o vetor alvo
  `loan_status` em `int64`. As 11 features numéricas/ordinais são empilhadas
  diretamente e as categóricas restantes (`home_ownership`, `loan_intent`)
  recebem one-hot encoding vetorizado.
- **`CleanedRecord`** e **`encode_record`** (`src/preprocessing/transform.py`) —
  registro já codificado (gênero → binário, escolaridade → ordinal, default
  anterior → binário) que alimenta a vetorização.
- **`LoanRecord`** e **`_parse_row`** (`src/data/loan_loader.py`) — registro
  bruto imutável e o parsing linha a linha do CSV, entregando os dados ao
  vetorizador como objetos tipados.
- `scikit-learn` e `pandas` fixados em `requirements.txt` (lock com hashes),
  mantendo `encode_features` (sklearn) como rota alternativa funcional.

#### Corrigido

- `clean_dataset` deixou de ser um stub (retornava `None`) e passa a codificar
  todos os registros, descartando linhas com erro de parsing.
- `split_data` — o corpo referenciava variáveis inexistentes; reescrito para
  divisão determinística sobre os registros com `np.random.default_rng`.

#### Alterado

- `main.py` — o pipeline passa a usar a rota NumPy:
  `load_csv → clean_dataset → split_data → build_feature_matrix`.

#### Detalhes da vetorização (trechos-chave em `build_feature_matrix`)

- **Matriz numérica** — `np.array(..., dtype=np.float64)` empilha as features
  numéricas/ordinais (`NUMERIC_FEATURES`) direto na matriz.
- **One-hot por _broadcasting_** — para cada coluna categórica, `np.unique`
  extrai as categorias e a comparação `(n, 1) == (1, k)` gera a matriz one-hot
  sem laço explícito por elemento:

  ```python
  values = np.array([getattr(r, field) for r in items])
  categories = np.unique(values)
  # Broadcasting: (n, 1) == (1, k) -> matriz one-hot (n, k).
  onehot_blocks.append((values[:, None] == categories[None, :]).astype(np.float64))
  ```

- **Montagem** — `np.hstack([numeric, *onehot_blocks])` concatena os blocos e
  `feature_names` acompanha a ordem das colunas.
- **Resultado no dataset real** — **21 features** (11 numéricas/ordinais +
  4 de `home_ownership` + 6 de `loan_intent`), com `X`/`y` prontos para o
  classificador das próximas etapas.

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

[Unreleased]: https://github.com/lucascdsilva/auto-dre/compare/v0.2.0...HEAD
[v0.2.0]: https://github.com/lucascdsilva/auto-dre/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/lucascdsilva/auto-dre/releases/tag/v0.1.0
