# syntax=docker/dockerfile:1.7
#
# Build multi-stage e endurecido para o AutoDRE.
# Objetivos de seguranca:
#  - imagem final minima (sem toolchain de build);
#  - dependencias instaladas com --require-hashes (defesa de supply chain);
#  - execucao como usuario non-root;
#  - sem segredos ou credenciais embutidos.

# ----------------------------------------------------------------------------
# Estagio 1: builder — instala dependencias em um virtualenv isolado.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Evita prompts e bytecode orfao; falha cedo em erros.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Virtualenv dedicado, copiado integro para a imagem final.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Apenas o requirements primeiro -> melhor cache e instalacao verificada.
COPY requirements.txt .

# --require-hashes: aborta se qualquer artefato divergir do hash fixado.
# --no-deps: nada alem do que foi explicitamente travado e' instalado.
RUN pip install --require-hashes --no-deps -r requirements.txt

# ----------------------------------------------------------------------------
# Estagio 2: runtime — imagem final minima, sem ferramentas de build.
# ----------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    AUTODRE_INPUT_DIR=/data/input \
    AUTODRE_OUTPUT_DIR=/data/output

# Usuario non-root sem shell de login.
RUN groupadd --system app && \
    useradd --system --gid app --no-create-home --shell /usr/sbin/nologin app

# Copia o virtualenv ja resolvido do builder.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Copia apenas o codigo da aplicacao (o .dockerignore exclui o resto).
COPY src/ ./src/
COPY main.py ./

# Pontos de montagem dos volumes de dados.
RUN mkdir -p /data/input /data/output && chown -R app:app /data /app

USER app

# Sem ENTRYPOINT com shell: execucao direta do interpretador.
ENTRYPOINT ["python", "main.py"]
