#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${SERVICE_LOG_DIR:-${SERVICE_DIR}/logs}"
ENV_FILE="${SERVICE_ENV_FILE:-${SERVICE_DIR}/.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

wait_for_exit() {
  local pid="$1"
  local timeout="${2:-10}"
  local i
  for ((i = 0; i < timeout; i++)); do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

stop_process() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"
  local pgid_file="${LOG_DIR}/${name}.pgid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name} is not running (missing ${pid_file})"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  local pgid
  if [[ -f "${pgid_file}" ]]; then
    pgid="$(cat "${pgid_file}")"
  else
    pgid="${pid}"
  fi

  if kill -0 "${pid}" >/dev/null 2>&1 || kill -0 "-${pgid}" >/dev/null 2>&1; then
    echo "Stopping ${name} process group ${pgid}..."
    kill -- "-${pgid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    if ! wait_for_exit "${pid}" 10; then
      echo "${name} did not exit after SIGTERM, sending SIGKILL to process group ${pgid}..."
      kill -9 -- "-${pgid}" >/dev/null 2>&1 || kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  else
    echo "${name} process ${pid} is not running"
  fi

  rm -f "${pid_file}" "${pgid_file}"
}

kill_port_processes() {
  local port="$1"
  if [[ -z "${port}" ]]; then
    return 0
  fi

  local pids
  pids="$(lsof -ti :"${port}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "Port ${port} is still in use; stopping remaining processes: ${pids}"
  # shellcheck disable=SC2086
  kill ${pids} >/dev/null 2>&1 || true
  sleep 2
  pids="$(lsof -ti :"${port}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Processes on port ${port} did not exit, sending SIGKILL: ${pids}"
    # shellcheck disable=SC2086
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

kill_vllm_leftovers() {
  local pids
  pids="$(pgrep -f "vllm.entrypoints.openai.api_server|${VLLM_MODEL_PATH:-llm4decompile}" 2>/dev/null || true)"
  if [[ -z "${pids}" ]]; then
    return 0
  fi

  echo "Stopping remaining vLLM-related processes: ${pids}"
  # shellcheck disable=SC2086
  kill ${pids} >/dev/null 2>&1 || true
  sleep 2
  pids="$(pgrep -f "vllm.entrypoints.openai.api_server|${VLLM_MODEL_PATH:-llm4decompile}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Remaining vLLM processes did not exit, sending SIGKILL: ${pids}"
    # shellcheck disable=SC2086
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

stop_process "service"
stop_process "vllm"

kill_port_processes "${SERVICE_PORT:-8088}"
kill_port_processes "${VLLM_PORT:-8002}"
kill_vllm_leftovers

echo "Demo services stopped."
