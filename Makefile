# Atalhos de uso. Os alvos Docker encapsulam as flags de seguranca.
# Detecta virtualenv automaticamente; fallback para python3 do sistema.
VENV := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -x .venv/bin/pip ]; then echo .venv/bin/pip; else echo pip3; fi)

.PHONY: test lock lock-ci lock-dev lock-all build run audit notebook clean

# Roda a suite de testes localmente (sem container).
test:
	$(VENV) -m unittest discover -s tests -v

# Regenera requirements.txt com hashes a partir de requirements.in.
lock:
	$(VENV) -m piptools compile --generate-hashes --allow-unsafe -o requirements.txt requirements.in

# Constroi a imagem endurecida.
build:
	docker compose build

# Executa o pipeline no container isolado (sem rede, fs read-only).
run:
	LOANRISK_UID=$(shell id -u) LOANRISK_GID=$(shell id -g) docker compose run --rm loan-risk-analyzer

# Auditoria de vulnerabilidades nas dependencias travadas (supply chain).
audit:
	$(VENV) -m pip_audit -r requirements.txt

# Regenera requirements-dev.txt com hashes a partir de requirements-dev.in.
lock-dev:
	$(VENV) -m piptools compile --generate-hashes --allow-unsafe -o requirements-dev.txt requirements-dev.in

# Regenera requirements-ci.txt (torch de CPU) a partir de requirements-ci.in.
# O runner do GitHub nao tem GPU: o stack CUDA custaria ~5 GB por execucao.
lock-ci:
	$(VENV) -m piptools compile --generate-hashes --allow-unsafe -o requirements-ci.txt requirements-ci.in

# Regenera os tres lockfiles de uma vez. requirements-ci.in deriva de
# requirements.in com -r, entao os dois precisam ser travados juntos.
lock-all: lock lock-ci lock-dev

# Inicia o Jupyter Notebook no diretorio notebooks/.
notebook:
	$(VENV) -m jupyter notebook --notebook-dir=notebooks

clean:
	rm -rf data/output/* **/__pycache__
