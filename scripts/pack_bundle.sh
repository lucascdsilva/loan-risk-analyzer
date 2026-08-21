#!/usr/bin/env bash
#
# Empacota o bundle de inferência para publicação
#
# Produz em dist/:
#   loan-risk-model-<versao>.tar.gz         o diretório bundle/ completo
#   loan-risk-model-<versao>.tar.gz.sha256  checksum do tarball
#
# O tarball é determinístico (--sort=name, mtime do commit, gzip -n): metadados
# de arquivo não entram no hash. O conteúdo ainda varia entre execuções por
# causa do created_at do model_card.json
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "uso: $(basename "$0") <versao>   (ex.: v0.4.0)" >&2
    exit 2
fi

VERSION="$1"
MODELS_DIR="${LOANRISK_MODELS_DIR:-data/models}"
BUNDLE_DIR="${MODELS_DIR}/bundle"
DIST_DIR="${DIST_DIR:-dist}"
TARBALL="loan-risk-model-${VERSION}.tar.gz"

ESPERADOS="SHA256SUMS golden.json model.onnx model_card.json preprocessing.json"

if [ ! -d "$BUNDLE_DIR" ]; then
    echo "bundle não encontrado em ${BUNDLE_DIR} — rode 'python main.py' antes" >&2
    exit 1
fi

for arquivo in $ESPERADOS; do
    if [ ! -f "${BUNDLE_DIR}/${arquivo}" ]; then
        echo "arquivo ausente no bundle: ${arquivo}" >&2
        exit 1
    fi
done

# Integridade interna antes de empacotar: se o SHA256SUMS já não fecha, não
# adianta publicar o tarball.
(cd "$BUNDLE_DIR" && sha256sum --check --status SHA256SUMS) || {
    echo "SHA256SUMS do bundle não confere" >&2
    exit 1
}

# Data de referência única para todas as entradas do tar. Sem isso, o mtime de
# cada arquivo (que muda a cada execução) entraria no hash do tarball.
EPOCH="$(git log -1 --format=%ct 2>/dev/null || date +%s)"

mkdir -p "$DIST_DIR"

# gzip separado do tar: `tar -z` grava nome e timestamp no cabeçalho gzip,
# `gzip -n` não.
tar --sort=name --owner=0 --group=0 --numeric-owner \
    --mtime="@${EPOCH}" \
    -cf - -C "$MODELS_DIR" bundle \
  | gzip -n -9 > "${DIST_DIR}/${TARBALL}"

(cd "$DIST_DIR" && sha256sum "$TARBALL" > "${TARBALL}.sha256")

echo "empacotado: ${DIST_DIR}/${TARBALL} ($(du -h "${DIST_DIR}/${TARBALL}" | cut -f1))"
cat "${DIST_DIR}/${TARBALL}.sha256"
