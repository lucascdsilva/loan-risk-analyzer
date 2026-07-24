# loan-risk-analyzer

Análise de risco financeiro para **aprovação de empréstimos**: pré-processamento
estruturado dos dados, vetorização com NumPy e **classificação por rede neural
(PyTorch)**.

Projeto Integrador — *Engenharia de Software para IA e Frameworks*.

## Status atual

**`v0.3.0` — Etapa 5–6 concluída.** O pipeline vai do CSV bruto ao classificador
neural treinado, avaliado e persistido. Próxima entrega: Etapa 7 (experimentos e
hiperparâmetros).

O histórico completo de mudanças está no [CHANGELOG.md](CHANGELOG.md).

### Etapas entregues

| Versão | Etapa | O que entregou | Detalhes |
|--------|-------|----------------|----------|
| [`v0.1.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.1.0) | Entrega 1 | Pré-processamento, modularização, testes e execução isolada em container | [CHANGELOG](CHANGELOG.md#v010--entrega-1) |
| [`v0.2.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.2.0) | Etapa 4 | Vetorização das features com NumPy (`build_feature_matrix`) | [CHANGELOG](CHANGELOG.md#v020--etapa-4) |
| [`v0.3.0`](https://github.com/lucascdsilva/auto-dre/releases/tag/v0.3.0) | Etapa 5–6 | Classificador neural em PyTorch: treino, avaliação e persistência | [CHANGELOG](CHANGELOG.md#v030--etapa-56) |

## Dataset

`data/loan_data.csv` — 45 000 solicitações de empréstimo com as colunas:

| Feature | Tipo | Descrição |
|---|---|---|
| `person_age` | float | Idade do solicitante |
| `person_gender` | str | Gênero (`female` / `male`) |
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

1. **carrega** os registros (`src/data/loan_loader.py`);
2. **codifica e vetoriza** as features — gênero → binário, escolaridade →
   ordinal, default anterior → binário, e one-hot para `home_ownership` /
   `loan_intent` — produzindo `X`, `y` em NumPy (`src/preprocessing/transform.py`);
3. **divide** em treino/teste de forma estratificada e determinística;
4. **balanceia** as classes com **SMOTE**, aplicado somente ao treino (evita
   vazamento);
5. **normaliza** as features com `StandardScaler`;
6. **treina** a rede neural `NeuralNetworkV0` — MLP `23 → 20 (ReLU) → 1` — com
   `Adam` e `BCEWithLogitsLoss`, em código *device-agnostic* (CPU/GPU)
   (`src/models/`, `src/training/`);
7. **persiste** os pesos treinados em `data/models/neural_network_V0.pth`;
8. **avalia** no conjunto de teste e imprime **Accuracy, Precision, Recall e
   F1-Score** (torchmetrics) no stdout (`src/inference/`, `src/evaluation/`).

## Estrutura

```
loan-risk-analyzer/
├── data/
│   ├── loan_data.csv     # dataset principal (45 000 registros)
│   ├── models/           # pesos treinados (neural_network_V0.pth)
│   └── output/           # artefatos de saída
├── src/
│   ├── data/             # loan_loader.py       — carga do CSV
│   ├── preprocessing/    # transform.py         — encode, vetorização, SMOTE, scaling
│   ├── models/           # NeuralNetworkV0.py   — arquitetura da MLP
│   ├── training/         # train.py             — loop de treino e tensores
│   ├── inference/        # inference.py         — predições
│   ├── evaluation/       # metrics.py           — métricas (torchmetrics)
│   └── utils/            # config.py            — caminhos e seed
├── notebooks/            # Jupyter Notebooks (análise exploratória, por integrante)
├── tests/                # unittest (test_data, test_preprocessing)
├── docs/                 # SECURITY.md
├── Dockerfile            # build multi-stage, non-root
├── docker-compose.yml    # execução endurecida (sem rede, fs read-only)
├── requirements.in       # dependências de runtime
├── requirements.txt      # versões fixas + hashes (--require-hashes)
├── requirements-dev.in   # dependências de desenvolvimento (Jupyter, torchmetrics)
├── requirements-dev.txt  # versões fixas + hashes (dev)
├── CHANGELOG.md
├── Makefile
└── main.py
```

## Execução local (recomendado para o pipeline completo)

O treino da rede usa `torchmetrics`, que faz parte das dependências de
desenvolvimento — por isso o pipeline completo roda localmente com ambos os
arquivos de requisitos instalados.

```bash
# Criar virtualenv e instalar dependências
python3 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt
pip install --require-hashes -r requirements-dev.txt

# Pipeline completo (encode → split → SMOTE → scale → treino PyTorch → avaliação)
LOANRISK_DATA_PATH=data/loan_data.csv \
LOANRISK_OUTPUT_DIR=data/output \
LOANRISK_MODELS_DIR=data/models \
python main.py
```

Ao final, os pesos treinados ficam em `data/models/neural_network_V0.pth` e as
métricas são impressas no terminal. O Makefile detecta automaticamente o
virtualenv `.venv/`; se ele não existir, usa o `python3` do sistema.

## Execução isolada com Docker

O `Dockerfile` e o `docker-compose.yml` fornecem um ambiente de execução
**endurecido** para a etapa de pré-processamento (Entrega 1): imagem multi-stage,
usuário **non-root**, **sem rede** (`network_mode: none`), **filesystem raiz
somente-leitura** e todas as capabilities do kernel removidas. O container
enxerga apenas `data/loan_data.csv` (somente leitura) e `data/output`.

```bash
make build   # constrói a imagem endurecida
make run     # executa no container isolado
```

> Nota: o container instala apenas o runtime mínimo (`requirements.txt`). O
> treino da rede neural depende de `torchmetrics` e de escrita em `data/models/`,
> ainda não montados no compose — por isso o pipeline completo da Etapa 5–6 é
> executado localmente (seção acima).

### Jupyter Notebook

```bash
source .venv/bin/activate
make notebook
```

As dependências de análise (`matplotlib`, `seaborn`, `torchmetrics`) estão em
`requirements-dev.in` / `requirements-dev.txt` e **não** entram na imagem Docker
de runtime.

## Testes

```bash
make test
```

> **Status:** a suíte de `unittest` (`tests/test_data.py`,
> `tests/test_preprocessing.py`) foi escrita para a API de registros tipados da
> Entrega 1 e ainda **não** foi realinhada ao pipeline pandas/PyTorch das Etapas
> 4–6 (`load_csv` agora retorna um `DataFrame`). Atualizar a cobertura de testes
> é um item pendente.

## Reprodutibilidade e dependências

- `requirements.txt` é gerado de `requirements.in` com **hashes fixados**.
  Para atualizar: `make lock`.
- `requirements-dev.txt` é gerado de `requirements-dev.in` com **hashes fixados**.
  Para atualizar: `make lock-dev`.
- Auditoria de vulnerabilidades: `make audit`.
- Semente fixa (`RANDOM_SEED = 42`) para splits e treino determinísticos.

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
| 7 | Experimentos e comparação de hiperparâmetros | ⏳ Planejada |
| 9–11 | Documentos de visão, requisitos e arquitetura → **v1.0.0** | ⏳ Planejada |
