# loan-risk-analyzer

Análise de risco financeiro para **aprovação de empréstimos** com pré-processamento
estruturado de dados e pipeline rumo à classificação por rede neural.

Projeto Integrador — *Engenharia de Software para IA e Frameworks*.
Esta é a **Entrega 1**: funções, modularização, repositório e a base de
execução isolada em container.

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

## O que esta entrega faz

A partir de `data/loan_data.csv`, o pipeline:

1. carrega e valida os registros (`src/data/loan_loader.py`);
2. codifica variáveis categóricas (gênero → binário, escolaridade → ordinal,
   default anterior → binário) e preserva as demais como categoria para
   one-hot encoding nas etapas seguintes (`src/preprocessing/transform.py`);
3. divide determinísticamente em treino (80 %) e teste (20 %);
4. grava `train.csv`, `test.csv` e `summary.txt` em `data/output/`.

> A classificação por **rede neural (PyTorch)** substitui o baseline nas
> próximas etapas.

## Estrutura

```
loan-risk-analyzer/
├── data/
│   ├── loan_data.csv     # dataset principal (45 000 registros)
│   └── output/           # resultados gerados (train.csv, test.csv, summary.txt)
├── src/
│   ├── data/             # loan_loader.py
│   ├── preprocessing/    # transform.py
│   └── utils/            # config.py
├── notebooks/            # Jupyter Notebooks (análise exploratória)
├── tests/                # unittest (test_data, test_preprocessing)
├── Dockerfile            # build multi-stage, non-root
├── docker-compose.yml    # execução endurecida (sem rede, fs read-only)
├── requirements.in       # dependências de runtime
├── requirements.txt      # versões fixas + hashes (--require-hashes)
├── requirements-dev.in   # dependências de desenvolvimento (Jupyter)
├── requirements-dev.txt  # versões fixas + hashes (dev)
├── Makefile
└── main.py
```

## Execução isolada com Docker (recomendado)

```bash
# 1. Construir a imagem
make build

# 2. Executar o pipeline
make run
```

Os resultados aparecem em `data/output/`. O container:

- **não tem rede** (`network_mode: none`);
- roda com **filesystem raiz somente-leitura** e como **usuário non-root**;
- enxerga apenas `data/loan_data.csv` (somente leitura) e `data/output`.

## Execução local (desenvolvimento)

O Makefile detecta automaticamente o virtualenv `.venv/` e o utiliza.
Caso ele não exista, usa o `python3` do sistema.

```bash
# Criar virtualenv e instalar dependências
python3 -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt
pip install --require-hashes -r requirements-dev.txt

# Pipeline principal
LOANRISK_DATA_PATH=data/loan_data.csv LOANRISK_OUTPUT_DIR=data/output python main.py
```

### Jupyter Notebook

Para rodar os notebooks localmente:

```bash
source .venv/bin/activate
make notebook
```

As dependências do Jupyter (`matplotlib`, `seaborn`) estão em `requirements-dev.in` /
`requirements-dev.txt` e **não** entram na imagem Docker (runtime mínimo).

## Testes

```bash
make test
```

Saída esperada: **16 testes, OK**.

## Reprodutibilidade e dependências

- `requirements.txt` é gerado de `requirements.in` com **hashes fixados**.
  Para atualizar: `make lock`.
- `requirements-dev.txt` é gerado de `requirements-dev.in` com **hashes fixados**.
  Para atualizar: `make lock-dev`.
- Auditoria de vulnerabilidades: `make audit`.
- Jupyter Notebook: `make notebook`.

## Roadmap (próximas entregas)

| Etapa | Entrega |
|-------|---------|
| 4 | Vetorização das features com NumPy |
| 5–6 | Classificador neural em PyTorch (treino, validação, inferência) |
| 7 | Experimentos e comparação de hiperparâmetros |
| 9–11 | Documentos de visão, requisitos e arquitetura |
