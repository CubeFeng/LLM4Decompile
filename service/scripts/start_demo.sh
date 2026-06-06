#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SERVICE_DIR}/.." && pwd)"
LOG_DIR="${SERVICE_LOG_DIR:-${SERVICE_DIR}/logs}"
ENV_FILE="${SERVICE_ENV_FILE:-${SERVICE_DIR}/.env}"

VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-llm4decompile}"
SERVICE_CONDA_ENV="${SERVICE_CONDA_ENV:-llm4decompile-service}"

mkdir -p "${LOG_DIR}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

start_process() {
  local name="$1"
  local conda_env="$2"
  local command="$3"
  local pid_file="${LOG_DIR}/${name}.pid"
  local pgid_file="${LOG_DIR}/${name}.pgid"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ -f "${pid_file}" ]]; then
    local old_pid
    old_pid="$(cat "${pid_file}")"
    if kill -0 "${old_pid}" >/dev/null 2>&1; then
      echo "${name} is already running with PID ${old_pid}"
      echo "Log: ${log_file}"
      return 0
    fi
    rm -f "${pid_file}" "${pgid_file}"
  fi

  echo "Starting ${name} with conda env '${conda_env}'..."
  setsid bash -lc "
    set -euo pipefail
    cd "${REPO_ROOT}"
    exec conda run --no-capture-output -n "${conda_env}" bash -lc "${command}"
  " >"${log_file}" 2>&1 &

  local pid="$!"
  local pgid="${pid}"
  echo "${pid}" >"${pid_file}"
  echo "${pgid}" >"${pgid_file}"
  echo "${name} started with PID ${pid}, PGID ${pgid}"
  echo "Log: ${log_file}"
}

start_process "vllm" "${VLLM_CONDA_ENV}" "./service/scripts/run_vllm.sh"
start_process "service" "${SERVICE_CONDA_ENV}" "./service/scripts/run_service.sh"

echo
echo "Demo services are starting."
echo "Check logs:"
echo "  tail -f ${LOG_DIR}/vllm.log"
echo "  tail -f ${LOG_DIR}/service.log"
echo
echo "Health check after startup:"
echo "  curl http://127.0.0.1:${SERVICE_PORT:-8088}/health"
