#!/usr/bin/env bash
# Sets up an isolated virtual environment for mini-rag and runs the ingest/query
# pipeline. Nothing is installed globally: everything lives under ./.venv,
# which is created next to this script on first run and reused afterwards.
#
# Usage:
#   ./run.sh corpus/corpus.json
#   ./run.sh corpus/corpus.json --top-k 8
#   ./run.sh corpus/corpus.json --min-similarity 0.5
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"

if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating virtual environment in ${VENV_DIR} ..."
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet -e "${SCRIPT_DIR}"

python -m mini_rag.main "$@"
