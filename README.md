# loan-risk-analyzer

Análise de risco financeiro para **aprovação de empréstimos**: pré-processamento
estruturado dos dados, **classificação por rede neural (PyTorch)** e publicação
de um **bundle de inferência versionado** (ONNX) para consumo por outros
serviços.

Projeto Integrador — *Engenharia de Software para IA e Frameworks*.

## Status atual

**`v0.4.0` — Bloco A concluído.** O pipeline vai do CSV bruto ao classificador
neural treinado, avaliado e **empacotado como artefato distribuível**: o
repositório deixa de ser só um pipeline de treino e passa a publicar um bundle
versionado (`model.onnx` + contrato de pré-processamento + model card + golden)
que os projetos de aplicação consomem. Pendente: publicar o Release `v0.4.0`
com o bundle anexado.

O histórico completo de mudanças está no [CHANGELOG.md](CHANGELOG.md).

### Etapas entregues

| Versão | Etapa | O que entregou | Detalhes |
|--------|-------|----------------|----------|
| [`v0.1.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.1.0) | Entrega 1 | Pré-processamento, modularização, testes e execução isolada em container | [CHANGELOG](CHANGELOG.md#v010--entrega-1) |
| [`v0.2.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.2.0) | Etapa 4 | Vetorização das features com NumPy (`build_feature_matrix`) | [CHANGELOG](CHANGELOG.md#v020--etapa-4) |
| [`v0.3.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.3.0) | Etapa 5–6 | Classificador neural em PyTorch: treino, avaliação e persistência | [CHANGELOG](CHANGELOG.md#v030--etapa-56) |
| `v0.4.0` | Bloco A | Contrato de 21 features, bundle de inferência ONNX, métricas completas e treino reproduzível | [CHANGELOG](CHANGELOG.md#v040--bloco-a) |

## Dataset

`data/loan_data.csv` — 45 000 solicitações de empréstimo com as colunas:

| Feature | Tipo | Descrição |
|---|---|---|
| `person_age` | float | Idade do solicitante |
| ~~`person_gender`~~ | str | Gênero — **descartado** do contrato de features (ver abaixo) |
| `person_education` | str | Escolaridade (`High School` … `Doctorate`) |
| `person_income` | float | Renda anual (USD) |
| `person_emp_exp` | int | Anos de experiência profissional |
| `person_home_ownership` | str | `RENT` / `OWN` / `MORTGAGE` |
| `loan_amnt` | float | Valor solicitado (USD) |
| `loan_intent` | str | Finalidade (`PERSONAL`, `EDUCATION`, `MEDICAL`, …) |
| `loan_int_rate` | float | Taxa de juros (%) |
| `loan_percent_income` | float | Parcela/renda |
| `cb_person_cred_hist_length` | float | Histórico de crédito (anos) |
| `credit_score` | int | Score de crédito |
| `previous_loan_defaults_on_file` | str | Default anterior (`Yes` / `No`) |
| **`loan_status`** | int | **Alvo**: 0 = sem default, 1 = default |

## Pipeline atual

A partir de `data/loan_data.csv`, o `main.py` executa ponta a ponta:

1. **carrega** os registros em um `DataFrame` (`src/data/loan_loader.py`);
2. **ajusta o codificador** e produz `X`, `y` — escolaridade → ordinal, one-hot
   com categorias **fixas** para `home_ownership`, `loan_intent` e
   `previous_loan_defaults_on_file`, `person_gender` descartado
   (`src/preprocessing/transform.py`);
3. **divide** em treino/teste de forma estratificada e determinística;
4. **balanceia** as classes com **SMOTE**, aplicado somente ao treino (evita
   vazamento);
5. **normaliza** as features com `StandardScaler`, ajustado só no treino;
6. **treina** a rede neural `NeuralNetworkV0` — MLP `21 → 20 (ReLU) → 1` — com
   `Adam` e `BCEWithLogitsLoss`, em código *device-agnostic* (CPU/GPU)
   (`src/models/`, `src/training/`);
7. **persiste** os pesos treinados em `data/models/neural_network_V0.pth`
   (arquivo local, **não versionado**);
8. **avalia** no conjunto de teste e imprime **Accuracy, Precision, Recall,
   F1-Score, AUROC** e a **matriz de confusão** (`src/inference/`,
   `src/evaluation/`);
9. **exporta o bundle de inferência** em `data/models/bundle/`
   (`src/export/bundle.py`).

### Contrato de features — 21 colunas

O contrato é **fixo e explícito**: as mesmas colunas, na mesma ordem, no treino
e na inferência de um único registro. Categoria desconhecida é **rejeitada**,
nunca zerada em silêncio.

- 9 numéricas/ordinais + 4 (`person_home_ownership`) + 6 (`loan_intent`) +
  2 (`previous_loan_defaults_on_file`) = **21 features**;
- `person_gender` é **descartado**: gênero não é variável de decisão de crédito;
- `person_education` entra como **escala ordinal** (`High School` = 0 …
  `Doctorate` = 4).

### Bundle de inferência (`data/models/bundle/`, 48 KB)

O artefato distribuível do projeto — o que outros serviços consomem no lugar
dos pesos `.pth`:

| Arquivo | Conteúdo |
|---|---|
| `model.onnx` | `StandardScaler` + rede + `sigmoid` **dentro do grafo**, entrada `(batch, 21)` crua e eixo de batch dinâmico |
| `preprocessing.json` | categorias fixas, escala ordinal e ordem das 21 features |
| `model_card.json` | versões, arquitetura, hiperparâmetros, dataset e métricas |
| `golden.json` | registros crus + probabilidade esperada, para o consumidor verificar que reproduz o modelo |
| `SHA256SUMS` | integridade dos quatro arquivos |

Como a normalização e a sigmoid ficam **dentro** do grafo, o serviço consumidor
só precisa saber codificar as categóricas — não reimplementa normalização nem
depende de arquivos irmãos.

## Estrutura

```
loan-risk-analyzer/
├── data/
│   ├── loan_data.csv     # dataset principal (45 000 registros)
│   ├── models/           # pesos .pth e bundle/ (gerados, não versionados)
│   └── output/           # artefatos de saída
├── src/
│   ├── data/             # loan_loader.py       — carga do CSV
│   ├── preprocessing/    # transform.py         — contrato de features, encoder, SMOTE, scaling
│   ├── models/           # NeuralNetworkV0.py   — arquitetura da MLP
│   ├── training/         # train.py             — loop de treino e tensores
│   ├── inference/        # inference.py         — predict_proba / predict
│   ├── evaluation/       # metrics.py           — métricas (torchmetrics)
│   ├── export/           # bundle.py            — exportação do bundle ONNX
│   └── utils/            # config.py            — caminhos, seed e versões
├── notebooks/            # Jupyter Notebooks (análise exploratória, por integrante)
├── tests/                # unittest (dados, contrato de features, bundle, pré-processamento)
├── docs/                 # SECURITY.md
├── Dockerfile            # build multi-stage, non-root
├── docker-compose.yml    # execução endurecida (sem rede, fs read-only)
├── requirements.in       # dependências de runtime
├── requirements.txt      # versões fixas + hashes (--require-hashes)
├── requirements-dev.in   # dependências de desenvolvimento (Jupyter, matplotlib, seaborn)
├── requirements-dev.txt  # versões fixas + hashes (dev)
├── CHANGELOG.md
├── Makefile
└── main.py
```

## Execução local

```bash
# Criar virtualenv e instalar dependências
python3 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt

# Pipeline completo (encode → split → SMOTE → scale → treino → avaliação → bundle)
LOANRISK_DATA_PATH=data/loan_data.csv \
LOANRISK_OUTPUT_DIR=data/output \
LOANRISK_MODELS_DIR=data/models \
python main.py
```

Ao final, os pesos ficam em `data/models/neural_network_V0.pth`, o bundle em
`data/models/bundle/` e as métricas são impressas no terminal. O Makefile
detecta automaticamente o virtualenv `.venv/`; se ele não existir, usa o
`python3` do sistema.

> `requirements-dev.txt` só é necessário para os notebooks (`matplotlib`,
> `seaborn`, Jupyter). Desde a `v0.4.0`, `torchmetrics` e a cadeia ONNX são
> dependências de **runtime**: produzir o bundle é a função deste projeto.

## Execução isolada com Docker

O `Dockerfile` e o `docker-compose.yml` fornecem um ambiente de execução
**endurecido**: imagem multi-stage, usuário **non-root**, **sem rede**
(`network_mode: none`), **filesystem raiz somente-leitura** e todas as
capabilities do kernel removidas. O container enxerga apenas
`data/loan_data.csv` (somente leitura), `data/output` e `data/models`.

```bash
make build   # constrói a imagem endurecida
make run     # executa o pipeline completo no container isolado
```

Desde a `v0.4.0` o **pipeline completo** — treino, avaliação e exportação do
bundle — roda no container: `data/models` é montado como volume gravável e o
`make run` injeta o `uid:gid` do host, de modo que os artefatos gerados
pertencem ao seu usuário sem precisar de permissão 777.

### Jupyter Notebook

```bash
source .venv/bin/activate
make notebook
```

As dependências de análise (`matplotlib`, `seaborn`, Jupyter) estão em
`requirements-dev.in` / `requirements-dev.txt` e **não** entram na imagem Docker
de runtime.

## Testes

```bash
make test
```

**38 testes** com `unittest`, alinhados ao pipeline pandas/PyTorch:

| Arquivo | Cobre |
|---|---|
| `tests/test_data.py` | carga do CSV: colunas esperadas, tipos e domínio de `loan_status` / `previous_loan_defaults_on_file` |
| `tests/test_feature_contract.py` | ordem e largura das 21 features, paridade entre lote e registro único, rejeição de categoria desconhecida, ausência de efeito colateral no `DataFrame` |
| `tests/test_bundle_export.py` | paridade ONNX × PyTorch (`atol=1e-5`), eixo de batch dinâmico, `model.onnx` autocontido, `SHA256SUMS` íntegro e reprodução do bundle publicado |
| `tests/test_preprocessing.py` | escala ordinal de escolaridade |

> Os dois testes sobre o **bundle publicado** são pulados se
> `data/models/bundle/` não existir — rode `python main.py` antes para
> exercitá-los.

## Reprodutibilidade e dependências

- `requirements.txt` é gerado de `requirements.in` com **hashes fixados**.
  Para atualizar: `make lock`.
- `requirements-dev.txt` é gerado de `requirements-dev.in` com **hashes fixados**.
  Para atualizar: `make lock-dev`.
- Auditoria de vulnerabilidades: `make audit`.
- Semente fixa (`RANDOM_SEED = 42`) para splits e treino determinísticos — a
  semeadura ocorre **antes** da construção da rede, então `model.onnx`,
  `preprocessing.json` e `golden.json` saem byte a byte idênticos entre
  execuções (só o `model_card.json` varia, pelo `created_at`).
- `MODEL_VERSION` e `PREPROCESSING_VERSION` (`src/utils/config.py`) versionam
  modelo e pré-processamento separadamente: dois modelos treinados sobre a mesma
  transformação compartilham o segundo valor.

## Integrantes

**Grupo 16**
- Lucas Carvalho
- Paulo Renato Barbosa
- Stefano Sabino Vivas da Silva
- Pietra Oliveira
- Lamartine Teixeira
- Joao Gabriel de Oliveira Feitosa

## Roadmap

| Etapa | Entrega | Status |
|-------|---------|--------|
| 1 | Pré-processamento, modularização e execução isolada | ✅ Concluída (`v0.1.0`) |
| 4 | Vetorização das features com NumPy | ✅ Concluída (`v0.2.0`) |
| 5–6 | Classificador neural em PyTorch (treino, avaliação, inferência) | ✅ Concluída (`v0.3.0`) |
| Bloco A | Contrato de 21 features e bundle de inferência publicável | ✅ Concluída (`v0.4.0`, release pendente) |
| 7 | Experimentos e comparação de hiperparâmetros | ⏳ Planejada |
| 9–11 | Documentos de visão, requisitos e arquitetura → **v1.0.0** | ⏳ Planejada |
