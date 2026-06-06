#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SERVICE_DIR}/.." && pwd)"

ENV_FILE="${SERVICE_ENV_FILE:-${SERVICE_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

HOST="${SERVICE_HOST:-0.0.0.0}"
PORT="${SERVICE_PORT:-8088}"

cd "${REPO_ROOT}"
python -m uvicorn service.app.main:app --host "${HOST}" --port "${PORT}"
