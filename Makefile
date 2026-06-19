# Atalhos de uso. Os alvos Docker encapsulam as flags de seguranca.
# Detecta virtualenv automaticamente; fallback para python3 do sistema.
VENV := $(shell if [ -x .venv/bin/python ]; then echo .venv/bin/python; else echo python3; fi)
PIP := $(shell if [ -x .venv/bin/pip ]; then echo .venv/bin/pip; else echo pip3; fi)

.PHONY: test lock lock-dev build run audit notebook clean

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
	docker compose run --rm loan-risk-analyzer

# Auditoria de vulnerabilidades nas dependencias travadas (supply chain).
audit:
	$(VENV) -m pip_audit -r requirements.txt

# Regenera requirements-dev.txt com hashes a partir de requirements-dev.in.
lock-dev:
	$(VENV) -m piptools compile --generate-hashes --allow-unsafe -o requirements-dev.txt requirements-dev.in

# Inicia o Jupyter Notebook no diretorio notebooks/.
notebook:
	$(VENV) -m jupyter notebook --notebook-dir=notebooks

clean:
	rm -rf data/output/* **/__pycache__
