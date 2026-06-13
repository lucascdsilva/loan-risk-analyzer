# AutoDRE

Análise de extratos **OFX** com construção automática de **plano de contas** e
geração da **DRE** (Demonstração do Resultado do Exercício).

Projeto Integrador — *Engenharia de Software para IA e Frameworks*.
Esta é a **Entrega 1**: funções, modularização, repositório e a base de
execução isolada em container.

## O que esta entrega faz

A partir de arquivos `.ofx` colocados em `data/input/`, o pipeline:

1. carrega e deduplica as transações (`src/data/ofx_loader.py`);
2. normaliza as descrições (`src/preprocessing/transform.py`);
3. classifica cada transação no plano de contas — *baseline* por regras nesta
   etapa (`src/accounting/chart.py`);
4. gera a DRE do período (`src/accounting/dre.py`);
5. grava `transacoes.csv`, `dre.csv` e `relatorio.txt` em `data/output/`.

> A classificação por **rede neural (PyTorch)** substitui o baseline nas
> próximas etapas. O baseline também serve para gerar rótulos iniciais
> (*bootstrapping*) da base de treino.

## Estrutura

```
autodre/
├── data/
│   ├── input/        # OFX de entrada (exemplo.ofx incluso)
│   └── output/       # resultados gerados
├── src/
│   ├── data/         # ofx_loader.py
│   ├── preprocessing/# transform.py
│   ├── accounting/   # chart.py, dre.py
│   └── utils/        # config.py
├── tests/            # unittest (test_data, test_preprocessing, test_accounting)
├── Dockerfile        # build multi-stage, non-root
├── docker-compose.yml# execução endurecida (sem rede, fs read-only)
├── requirements.in   # dependências de alto nível
├── requirements.txt  # versões fixas + hashes (--require-hashes)
├── Makefile
└── main.py
```

## Execução isolada com Docker (recomendado)

A aplicação foi desenhada para rodar **dentro de um container isolado**, de
modo que o script não tenha acesso ao ambiente local da máquina nem à rede.
Detalhes em [`docs/SECURITY.md`](docs/SECURITY.md).

```bash
# 1. Construir a imagem (instala dependências verificando hashes)
make build           # ou: docker compose build

# 2. Colocar seus extratos em data/input/ (*.ofx) e executar
make run             # ou: docker compose run --rm autodre
```

Os resultados aparecem em `data/output/`. O container:

- **não tem rede** (`network_mode: none`);
- roda com **filesystem raiz somente-leitura** e como **usuário non-root**;
- enxerga **apenas** `data/input` (somente leitura) e `data/output`.

## Execução local (desenvolvimento)

```bash
python -m venv .venv && source .venv/bin/activate
pip install --require-hashes -r requirements.txt
AUTODRE_INPUT_DIR=data/input AUTODRE_OUTPUT_DIR=data/output python main.py
```

## Testes

```bash
make test            # ou: python3 -m unittest discover -s tests -v
```

Saída esperada: **10 testes, OK**.

## Reprodutibilidade e dependências

- `requirements.txt` é gerado de `requirements.in` com **hashes fixados**.
  Para atualizar: `make lock`.
- Auditoria de vulnerabilidades das dependências: `make audit`.

## Roadmap (próximas entregas)

| Etapa | Entrega |
|-------|---------|
| 4 | Vetorização das descrições com NumPy |
| 5–6 | Classificador neural em PyTorch (treino, validação, inferência) |
| 7 | Experimentos e comparação de hiperparâmetros |
| 9–11 | Documentos de visão, requisitos e arquitetura |
