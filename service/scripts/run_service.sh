#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SERVICE_DIR}/.." && pwd)"

ENV_FILE="${SERVICE_ENV_FILE:-${SERVICE_DIR}/.env}"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
load_env_file "${ENV_FILE}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-8088}"
PYTHON_BIN="${SERVICE_PYTHON:-python}"

cd "${REPO_ROOT}"
"${PYTHON_BIN}" -m uvicorn service.app.main:app --host "${HOST}" --port "${PORT}"
