# Atalhos de uso. Os alvos Docker encapsulam as flags de seguranca.
.PHONY: test lock build run audit clean

# Roda a suite de testes localmente (sem container).
test:
	python3 -m unittest discover -s tests -v

# Regenera requirements.txt com hashes a partir de requirements.in.
lock:
	python3 -m piptools compile --generate-hashes --allow-unsafe -o requirements.txt requirements.in

# Constroi a imagem endurecida.
build:
	docker compose build

# Executa o pipeline no container isolado (sem rede, fs read-only).
run:
	docker compose run --rm loan-risk-analyzer

# Auditoria de vulnerabilidades nas dependencias travadas (supply chain).
audit:
	python3 -m pip_audit -r requirements.txt

clean:
	rm -rf data/output/* **/__pycache__
