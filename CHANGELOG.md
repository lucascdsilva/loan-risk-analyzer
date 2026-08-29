# Changelog

Todas as mudanças relevantes deste projeto são documentadas aqui.

O formato segue [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o versionamento segue [SemVer](https://semver.org/lang/pt-BR/), onde cada
**MINOR** corresponde a uma entrega do roadmap do Projeto Integrador.

| Versão | Entrega | Status |
|--------|---------|--------|
| [v0.1.0](#v010--entrega-1) | Entrega 1 — pré-processamento e base de execução | Concluída |
| [v0.2.0](#v020--etapa-4) | Etapa 4 — vetorização com NumPy | Concluída |
| [v0.3.0](#v030--etapa-56) | Etapa 5–6 — classificador neural (PyTorch) | Concluída |
| [v0.4.0](#v040--bloco-a) | Bloco A — o modelo como componente publicável | Concluída |
| [v0.4.1](#v041--pipeline-de-release) | Pipeline de release automatizado (Bloco A, item A7) | Concluída |
| [v0.5.0](#v050--requisitos-e-testes-automatizados) | Etapa 9 — requisitos (GR4ML) e suíte de testes automatizados | Concluída |
| _planejado_ | Etapa 7 — experimentos e hiperparâmetros | — |
| _planejado_ | Etapas 10–11 — arquitetura e implantação → **v1.0.0** | — |

## [Unreleased]

_Sem mudanças registradas desde a `v0.5.0`._

## [v0.5.0] — Requisitos e testes automatizados

_2026-08-29_

### O que o projeto deve fazer, escrito e verificável

Até aqui o repositório tinha um pipeline correto sem um enunciado formal do
problema que ele resolve. Esta versão fecha os dois lados: os requisitos passam
a existir como documento **GR4ML** — visão, atores, casos de uso, entidades e
requisitos funcionais, não-funcionais e de ML — e a suíte de **40 testes**
`unittest` é a contraparte executável desse documento. Cada requisito de ML que
este repositório atende tem um teste que o prova, e o CI roda todos eles a cada
push.

#### Adicionado

- **Análise de Requisitos (GR4ML)** — visão do produto, atores (Analista de
  Crédito, Administrador, Serviço de ML), casos de uso UC01–UC03, entidades,
  RF01–RF14, RNF01–RNF07, requisitos de ML (RF-ML-01 a 05 e ML01–ML05),
  arquitetura proposta e MVP. Link no README.
- Seção **Requisitos** no README, com a tabela de **rastreabilidade**
  requisito → módulo que o implementa → teste que o cobre. É o que liga o
  documento de requisitos à suíte automatizada.
- Seção **CI e release** no README, documentando `ci.yml`, `release.yml` e os
  `scripts/` entregues na `v0.4.1` — que existiam no repositório mas nunca
  tinham sido descritos.

#### Alterado

- `MODEL_VERSION` de `v0.4.1` para `v0.5.0` (`src/utils/config.py`).
  `PREPROCESSING_VERSION` permanece `v0.4.0`: o contrato de 21 features não
  mudou, e é exatamente para esse caso que os dois eixos são versionados
  separadamente.
- README — *Status atual*, *Etapas entregues*, *Estrutura* e *Roadmap*
  realinhados ao estado real do repositório. A `v0.4.1` não constava em nenhum
  deles, e a árvore de diretórios omitia `.github/workflows/`, `scripts/` e os
  lockfiles de CI.

#### Corrigido

- **A suíte tem 40 testes, não 38.** `tests/test_evaluation.py` (2 testes)
  entrou na `v0.4.1` sem que o total fosse atualizado, e a tabela de testes do
  README também o omitia. Contagem e tabela corrigidas.

## [v0.4.1] — Pipeline de release

_2026-08-20_

### O release deixa de ser manual

Primeira versão publicada de ponta a ponta pelo CI: a tag dispara o treino do
zero, a validação do bundle e a criação do Release com o artefato anexado —
fecha o item A7 do plano de arquitetura.

#### Adicionado

- **`.github/workflows/ci.yml`** — testes unitários e dependências travadas
  (`--require-hashes`) a cada push/PR.
- **`.github/workflows/release.yml`** — disparado por tag `v*.*.*`: treina,
  valida o bundle, empacota e publica o Release com o `.tar.gz` e o `.sha256`.
- **`scripts/check_version.py`** — guarda que falha se a tag divergir de
  `MODEL_VERSION`, impedindo bundle carimbado com versão errada.
- **`scripts/pack_bundle.sh`** — empacota `data/models/bundle/` e gera o
  checksum do tarball.
- **`scripts/release_notes.py`** — corpo do Release gerado a partir do
  `model_card.json` do artefato publicado, não de números copiados à mão.
- **`requirements-ci.in`/`.txt`** — conjunto de dependências do CI, com hashes.
- **`tests/test_evaluation.py`** — testes de `evaluate_model`.

#### Alterado

- `EPOCHS` de 3000 para 1500 em `main.py` — treino mais curto.
- `DEFAULT_THRESHOLD` de `0.5` para `0.35`, agora definido em um único lugar
  (`src/utils/config.py`): `src/inference/inference.py` importa a constante em
  vez de redefini-la. Antes, o `model_card.json` publicava `0.5` enquanto as
  métricas eram calculadas com `0.35`.
- `MODEL_VERSION` para `v0.4.1`. `PREPROCESSING_VERSION` permanece `v0.4.0`:
  o contrato de 21 features não mudou.

## [v0.4.0] — Bloco A

_2026-08-18_

### O modelo como componente publicável

Este repositório deixa de ser apenas um pipeline de treino e passa a **publicar
um artefato versionado** que os demais projetos consomem: o **contrato de features foi fechado em 21 colunas** e o
pré-processamento virou um objeto **ajustado e persistido**, em vez de uma
função reaplicada de memória na inferência.

#### Adicionado

- **`export_bundle`** (`src/export/bundle.py`) — item A2. Gera
  `data/models/bundle/` (48 KB), autocontido:
  - `model.onnx` — `StandardScaler` + rede + `sigmoid` **dentro do grafo**
    (opset 17, eixo de batch dinâmico, `external_data=False`). O serviço
    consumidor não reimplementa a normalização nem depende de arquivo irmão.
  - `preprocessing.json` — categorias fixas, escala ordinal e ordem das
    21 features, de forma declarativa.
  - `model_card.json` — versões, arquitetura, hiperparâmetros, dataset e
    métricas da versão.
  - `golden.json` — 50 registros crus de referência (completados com uma linha
    para cada par _coluna × categoria_ ausente) e a probabilidade esperada,
    **gerada pelo próprio ONNX** para que a verificação compare `onnxruntime`
    com `onnxruntime`.
  - `SHA256SUMS` — integridade dos quatro arquivos.
- **`build_encoder`, `prepare_frame`, `fit_encoder`, `transform_records`**
  (`src/preprocessing/transform.py`) — item A1: separação `fit`/`transform`,
  `OneHotEncoder` com `categories=` explícitas e `handle_unknown="error"`, sem
  mutar o DataFrame de entrada e sem exigir `loan_status` na inferência.
- **`predict_proba`** (`src/inference/inference.py`) — item A4: a rede estima
  risco, o `threshold` é decisão de negócio. Permite ajustar o corte sem
  re-treinar.
- **AUROC e matriz de confusão** (`src/evaluation/metrics.py`) — item A3.
  `evaluate_model` passa a **retornar** as métricas, alimentando o
  `model_card.json`.
- **`MODEL_VERSION`, `PREPROCESSING_VERSION` e `DEFAULT_THRESHOLD`**
  (`src/utils/config.py`) — o pré-processamento versiona separado do modelo:
  dois modelos treinados sobre a mesma transformação compartilham o valor.
- **`tests/test_feature_contract.py`** (15 testes) — ordem e largura das
  21 features, paridade entre lote e registro único, rejeição de categoria
  desconhecida (RF-ML-04) e ausência de efeito colateral no DataFrame.
- **`tests/test_bundle_export.py`** (14 testes) — paridade ONNX × PyTorch
  (`atol=1e-5`), eixo de batch dinâmico, `model.onnx` autocontido, `SHA256SUMS`
  íntegro, cobertura de categorias no golden e reprodução do bundle publicado.
- Execução do pipeline **completo em container**: `LOANRISK_MODELS_DIR` e
  volume `data/models` no `Dockerfile`/`docker-compose.yml`, com o container
  rodando sob o `uid:gid` do host (injetados pelo `Makefile`) para que os
  artefatos gerados pertençam ao usuário sem precisar de permissão 777.
- `torchmetrics`, `onnx`, `onnxscript` e `onnxruntime` promovidos a
  dependências de **runtime** (`requirements.in`): produzir o bundle é a função
  deste projeto, e mantê-las só em dev fazia o container treinar por minutos e
  quebrar no último passo.

#### Alterado

- **Contrato de features: 23 → 21 colunas.** `person_gender` foi removido —
  gênero não é variável de decisão de crédito (decisão D2, seção 3.3 do plano).
  O contrato anterior fica registrado como referência histórica.
- **`NeuralNetworkV0`** — `in_features` e `hidden_units` parametrizados
  (padrão `N_FEATURES = 21`), em vez das 23 entradas fixas no código.
- **`evaluate_model`** recebe **probabilidades**, não rótulos: sobre 0/1
  arredondados a AUROC degenera em acurácia balanceada.
- **`main.py`** — pipeline: `fit_encoder → train_test_split → SMOTE →
  fit_scaler → treino → persistência → métricas → export_bundle`. `EPOCHS` e
  `LEARNING_RATE` viraram constantes nomeadas.
- **Pesos `.pth` deixaram de ser versionados** (`.gitignore`), revertendo a
  decisão da `v0.3.0`: são regeneráveis, ninguém os lê no código e o artefato
  distribuível passou a ser o bundle.

#### Corrigido

- **O treino não era reproduzível** (item A8). `torch.manual_seed(RANDOM_SEED)`
  rodava *depois* de `NeuralNetworkV0()` já ter sido construído — e a
  inicialização dos pesos era a única fonte de aleatoriedade restante. Duas
  execuções idênticas davam AUROC 0,9675 e 0,9689, enquanto o README afirmava
  "treino determinístico" e o model card gravava `"seed": 42`. Semeando antes da
  construção da rede, `model.onnx`, `golden.json` e `preprocessing.json` saem
  byte a byte idênticos entre execuções (só o `model_card.json` varia, pelo
  `created_at`). Atende RNF04.
- **`squeeze()` colapsava o eixo de batch** quando havia um único registro
  (`src/inference/inference.py`) — corrigido para `squeeze(-1)`.
- **A largura do one-hot dependia do lote.** As categorias eram inferidas com
  `np.unique` sobre os registros recebidos, então um registro isolado produzia
  menos colunas que o treino — justamente o caminho da inferência. As categorias
  agora são fixas no contrato.

#### Removido

- Caminho de registros tipados, substituído pelo caminho pandas +
  `ColumnTransformer`: `LoanRecord`, `_parse_row` e `_write_csv`
  (`src/data/loan_loader.py`); `CleanedRecord`, `encode_record`,
  `clean_dataset`, `build_feature_matrix`, `split_data`, `NUMERIC_FEATURES` e
  `ONEHOT_FEATURES` (`src/preprocessing/transform.py`), além de
  `encode_features` e `scale_dataset`.
- Testes escritos contra essa API: `tests/test_data.py` foi realinhado ao
  `DataFrame` e `tests/test_preprocessing.py` ficou restrito à escala ordinal de
  escolaridade — a única regra dele que não estava coberta pelos testes de
  contrato. Suíte: **38 testes**, todos passando.

## [v0.3.0] — Etapa 5–6

_2026-07-24_

### Classificador neural com PyTorch

Treinamento de um classificador neural para o risco de default, com o pipeline
do notebook exploratório extraído para um pacote `src/` modular e executável
ponta a ponta via `main.py`. Inclui balanceamento de classes, avaliação com
torchmetrics e persistência do modelo treinado.

#### Adicionado

- **`NeuralNetworkV0`** (`src/models/NeuralNetworkV0.py`) — MLP com uma camada
  oculta (23 features → 20 neurônios com ReLU → 1 saída), retornando logits.
- **`train_nn`**, **`to_tensor`** e **`accuracy_fn`** (`src/training/train.py`) —
  loop de treinamento com otimizador `Adam` e `BCEWithLogitsLoss`, conversão de
  arrays NumPy em tensores e código _device-agnostic_ (CPU/GPU).
- **`predict`** (`src/inference/inference.py`) — inferência em modo
  `inference_mode` (logits → sigmoid → predições binárias `int32`).
- **`evaluate_model`** (`src/evaluation/metrics.py`) — avaliação com
  `torchmetrics` (Accuracy, Precision, Recall, F1-Score binários).
- **`smote_oversampling`** e **`scale_dataset`** (`src/preprocessing/transform.py`) —
  balanceamento de classes com `SMOTE` aplicado somente no treino (evita
  vazamento) e normalização com `StandardScaler`.
- **`save_model`** (`main.py`) e **`models_dir`** em `Settings`
  (`src/utils/config.py`) — persistência dos pesos (`state_dict`) em
  `data/models/neural_network_V0.pth`, versionado no repositório.
- Baselines (`DummyClassifier`, k-NN) e redução de dimensionalidade com PCA no
  notebook exploratório (`notebooks/loan-risk-analyzer-lamartine.ipynb`).
- `torch>=2.0` e `imbalanced-learn` promovidos a dependências de **runtime**
  (`requirements.in`); `torchmetrics` adicionado às dependências de
  desenvolvimento (`requirements-dev.in`).

#### Alterado

- `main.py` — o pipeline passa a treinar a rede neural:
  `encode_features → train_test_split → SMOTE → scale → treino PyTorch →
  persistência → avaliação`.
- `src/models/model.py` renomeado para `src/models/NeuralNetworkV0.py`.

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

[Unreleased]: https://github.com/lucascdsilva/auto-dre/compare/v0.5.0...HEAD
[v0.5.0]: https://github.com/lucascdsilva/auto-dre/compare/v0.4.1...v0.5.0
[v0.4.1]: https://github.com/lucascdsilva/auto-dre/compare/v0.4.0...v0.4.1
[v0.4.0]: https://github.com/lucascdsilva/auto-dre/compare/v0.3.0...v0.4.0
[v0.3.0]: https://github.com/lucascdsilva/auto-dre/compare/v0.2.0...v0.3.0
[v0.2.0]: https://github.com/lucascdsilva/auto-dre/compare/v0.1.0...v0.2.0
[v0.1.0]: https://github.com/lucascdsilva/auto-dre/releases/tag/v0.1.0
