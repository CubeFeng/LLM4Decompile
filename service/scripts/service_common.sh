#!/usr/bin/env bash
# Shared helpers for service_ctl.sh (process + systemd backends).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SERVICE_DIR}/.." && pwd)"
DEPLOY_DIR="${SERVICE_DIR}/deploy/systemd"
ENV_FILE="${SERVICE_ENV_FILE:-${SERVICE_DIR}/.env}"

SYSTEMD_UNIT_DIR="/etc/systemd/system"
TARGET_NAME="llm4decompile.target"
VLLM_UNIT="llm4decompile-vllm.service"
API_UNIT="llm4decompile-api.service"
LOG_DIR="${SERVICE_LOG_DIR:-${SERVICE_DIR}/logs}"

systemd_available() {
  [[ -S /run/systemd/private ]] || return 1
  systemctl is-system-running --wait >/dev/null 2>&1
}

is_wsl() {
  [[ -f /proc/sys/fs/binfmt_misc/WSLInterop ]]
}

systemd_units_installed() {
  [[ -f "${SYSTEMD_UNIT_DIR}/${TARGET_NAME}" ]]
}

systemd_units_active() {
  systemd_available || return 1
  systemctl is-active --quiet "${TARGET_NAME}" 2>/dev/null \
    || systemctl is-active --quiet "${VLLM_UNIT}" 2>/dev/null \
    || systemctl is-active --quiet "${API_UNIT}" 2>/dev/null
}

resolve_backend() {
  local backend="${LLM4DECOMPILE_BACKEND:-auto}"
  case "${backend}" in
    systemd)
      if ! systemd_available; then
        echo "ERROR: LLM4DECOMPILE_BACKEND=systemd but systemd is not PID 1." >&2
        echo "On WSL2, enable systemd in /etc/wsl.conf ([boot] systemd=true) and restart WSL," >&2
        echo "or use: LLM4DECOMPILE_BACKEND=process ./service/scripts/service_ctl.sh start" >&2
        exit 1
      fi
      echo "systemd"
      ;;
    process)
      echo "process"
      ;;
    auto)
      # WSL dev/test: prefer process mode unless user opted into systemd.
      if is_wsl && [[ "${LLM4DECOMPILE_PREFER_SYSTEMD:-0}" != "1" ]]; then
        echo "process"
      elif systemd_available; then
        echo "systemd"
      else
        echo "process"
      fi
      ;;
    *)
      echo "ERROR: unknown LLM4DECOMPILE_BACKEND=${backend} (use auto/systemd/process)" >&2
      exit 1
      ;;
  esac
}

require_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    echo "ERROR: this command must be run as root (use sudo)." >&2
    exit 1
  fi
}

ensure_root() {
  if [[ "${EUID}" -ne 0 ]]; then
    exec sudo -E "$0" "$@"
  fi
}

load_env() {
  if [[ ! -f "${ENV_FILE}" ]]; then
    echo "ERROR: missing ${ENV_FILE}" >&2
    echo "Copy service/.env.example to service/.env and configure it first." >&2
    exit 1
  fi
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
}

resolve_service_user() {
  if [[ -n "${LLM4DECOMPILE_SERVICE_USER:-}" ]]; then
    echo "${LLM4DECOMPILE_SERVICE_USER}"
    return
  fi
  if [[ -n "${SUDO_USER:-}" ]]; then
    echo "${SUDO_USER}"
    return
  fi
  id -un
}

service_user_home() {
  local user
  user="$(resolve_service_user)"
  getent passwd "${user}" | cut -d: -f6
}

resolve_conda_python() {
  local env_name="$1"
  local python_path=""

  if command -v conda >/dev/null 2>&1; then
    python_path="$(conda run -n "${env_name}" which python 2>/dev/null || true)"
    if [[ -n "${python_path}" && -x "${python_path}" ]]; then
      echo "${python_path}"
      return 0
    fi
  fi

  local home_dir
  home_dir="$(service_user_home)"

  local conda_base
  for conda_base in \
    "${CONDA_PREFIX:-}" \
    "${home_dir}/miniconda3" \
    "${home_dir}/anaconda3" \
    "${home_dir}/miniforge3" \
    "${home_dir}/micromamba"; do
    [[ -z "${conda_base}" ]] && continue
    python_path="${conda_base}/envs/${env_name}/bin/python"
    if [[ -x "${python_path}" ]]; then
      echo "${python_path}"
      return 0
    fi
  done

  if [[ -n "${SUDO_USER:-}" ]] && command -v sudo >/dev/null 2>&1; then
    python_path="$(sudo -u "${SUDO_USER}" -H bash -lc "conda run -n '${env_name}' which python" 2>/dev/null || true)"
    if [[ -n "${python_path}" && -x "${python_path}" ]]; then
      echo "${python_path}"
      return 0
    fi
  fi

  return 1
}

resolve_python_bin() {
  local explicit="$1"
  local conda_env="$2"
  local label="$3"

  if [[ -n "${explicit}" ]]; then
    echo "${explicit}"
    return
  fi

  local inferred
  if inferred="$(resolve_conda_python "${conda_env}")"; then
    echo "${inferred}"
    return
  fi

  echo "ERROR: ${label} is not set and conda env '${conda_env}' was not found." >&2
  echo "Set ${label} in ${ENV_FILE} to the absolute Python path." >&2
  exit 1
}

validate_python_bin() {
  local python_bin="$1"
  local label="$2"

  if [[ ! -x "${python_bin}" ]]; then
    echo "ERROR: ${label} is not executable: ${python_bin}" >&2
    exit 1
  fi
}

render_template() {
  local template="$1"
  local output="$2"
  local repo_root="$3"
  local service_user="$4"
  local service_group="$5"
  local vllm_python="$6"
  local service_python="$7"

  sed \
    -e "s|@REPO_ROOT@|${repo_root}|g" \
    -e "s|@SERVICE_USER@|${service_user}|g" \
    -e "s|@SERVICE_GROUP@|${service_group}|g" \
    -e "s|@VLLM_PYTHON@|${vllm_python}|g" \
    -e "s|@SERVICE_PYTHON@|${service_python}|g" \
    "${template}" >"${output}"
}

install_units() {
  load_env

  if [[ ! -f "${REPO_ROOT}/service/app/main.py" ]]; then
    echo "ERROR: expected FastAPI entrypoint at ${REPO_ROOT}/service/app/main.py" >&2
    exit 1
  fi

  local service_user service_group vllm_python service_python
  service_user="$(resolve_service_user)"
  service_group="$(id -gn "${service_user}")"
  vllm_python="$(resolve_python_bin "${VLLM_PYTHON:-}" "${VLLM_CONDA_ENV:-llm4decompile}" "VLLM_PYTHON")"
  service_python="$(resolve_python_bin "${SERVICE_PYTHON:-}" "${SERVICE_CONDA_ENV:-llm4decompile-service}" "SERVICE_PYTHON")"

  validate_python_bin "${vllm_python}" "VLLM_PYTHON"
  validate_python_bin "${service_python}" "SERVICE_PYTHON"

  if ! id "${service_user}" >/dev/null 2>&1; then
    echo "ERROR: service user does not exist: ${service_user}" >&2
    exit 1
  fi

  install_one() {
    local template_name="$1"
    local unit_name="$2"
    local template="${DEPLOY_DIR}/${template_name}"
    local output="${SYSTEMD_UNIT_DIR}/${unit_name}"
    if [[ ! -f "${template}" ]]; then
      echo "ERROR: missing template ${template}" >&2
      exit 1
    fi
    echo "Installing ${unit_name}..."
    render_template \
      "${template}" \
      "${output}" \
      "${REPO_ROOT}" \
      "${service_user}" \
      "${service_group}" \
      "${vllm_python}" \
      "${service_python}"
  }

  install_one "llm4decompile-vllm.service.in" "${VLLM_UNIT}"
  install_one "llm4decompile-api.service.in" "${API_UNIT}"
  install_one "llm4decompile.target.in" "${TARGET_NAME}"

  systemctl daemon-reload
}

start_background_process() {
  local name="$1"
  local env_key="$2"
  local env_val="$3"
  local script_rel="$4"
  local pid_file="${LOG_DIR}/${name}.pid"
  local pgid_file="${LOG_DIR}/${name}.pgid"
  local log_file="${LOG_DIR}/${name}.log"

  mkdir -p "${LOG_DIR}"

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

  echo "Starting ${name}..."
  # shellcheck disable=SC2093
  setsid env "${env_key}=${env_val}" "PYTHONPATH=${REPO_ROOT}:${PYTHONPATH:-}" \
    bash "${REPO_ROOT}/${script_rel}" >"${log_file}" 2>&1 &

  local pid="$!"
  echo "${pid}" >"${pid_file}"
  echo "${pid}" >"${pgid_file}"
  echo "${name} started with PID ${pid}"
  echo "Log: ${log_file}"
}

process_start() {
  load_env
  local vllm_python service_python
  vllm_python="$(resolve_python_bin "${VLLM_PYTHON:-}" "${VLLM_CONDA_ENV:-llm4decompile}" "VLLM_PYTHON")"
  service_python="$(resolve_python_bin "${SERVICE_PYTHON:-}" "${SERVICE_CONDA_ENV:-llm4decompile-service}" "SERVICE_PYTHON")"
  validate_python_bin "${vllm_python}" "VLLM_PYTHON"
  validate_python_bin "${service_python}" "SERVICE_PYTHON"

  start_background_process "vllm" "VLLM_PYTHON" "${vllm_python}" "service/scripts/run_vllm.sh"
  start_background_process "service" "SERVICE_PYTHON" "${service_python}" "service/scripts/run_service.sh"

  echo
  echo "Services are starting in the background (process mode)."
  echo "Check logs:"
  echo "  tail -f ${LOG_DIR}/vllm.log"
  echo "  tail -f ${LOG_DIR}/service.log"
  echo
  echo "Health check (vLLM may take 30s+ to load):"
  echo "  curl http://127.0.0.1:$(service_port)/health"
}

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

stop_background_process() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"
  local pgid_file="${LOG_DIR}/${name}.pgid"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name} is not running (missing ${pid_file})"
    return 0
  fi

  local pid pgid
  pid="$(cat "${pid_file}")"
  if [[ -f "${pgid_file}" ]]; then
    pgid="$(cat "${pgid_file}")"
  else
    pgid="${pid}"
  fi

  if kill -0 "${pid}" >/dev/null 2>&1 || kill -0 "-${pgid}" >/dev/null 2>&1; then
    echo "Stopping ${name} process group ${pgid}..."
    kill -- "-${pgid}" >/dev/null 2>&1 || kill "${pid}" >/dev/null 2>&1 || true
    if ! wait_for_exit "${pid}" 10; then
      echo "${name} did not exit after SIGTERM, sending SIGKILL..."
      kill -9 -- "-${pgid}" >/dev/null 2>&1 || kill -9 "${pid}" >/dev/null 2>&1 || true
    fi
  else
    echo "${name} process ${pid} is not running"
  fi

  rm -f "${pid_file}" "${pgid_file}"
}

kill_port_processes() {
  local port="$1"
  local force="${2:-0}"
  [[ -z "${port}" ]] && return 0
  local pids
  pids="$(lsof -ti :"${port}" 2>/dev/null || true)"
  [[ -z "${pids}" ]] && return 0
  echo "Port ${port} in use by: ${pids}"
  # shellcheck disable=SC2086
  kill ${pids} >/dev/null 2>&1 || true
  sleep 2
  pids="$(lsof -ti :"${port}" 2>/dev/null || true)"
  if [[ -n "${pids}" ]]; then
    echo "Port ${port} still in use; sending SIGKILL: ${pids}"
    # shellcheck disable=SC2086
    kill -9 ${pids} >/dev/null 2>&1 || true
  fi
}

kill_orphan_processes() {
  local pattern pids
  for pattern in \
    "${REPO_ROOT}/service/scripts/run_vllm.sh" \
    "${REPO_ROOT}/service/scripts/run_service.sh" \
    "uvicorn service.app.main:app" \
    "vllm.entrypoints.openai.api_server"; do
    pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
    [[ -z "${pids}" ]] && continue
    echo "Stopping processes matching: ${pattern} (${pids})"
    # shellcheck disable=SC2086
    kill ${pids} >/dev/null 2>&1 || true
  done
  sleep 2
  for pattern in \
    "${REPO_ROOT}/service/scripts/run_vllm.sh" \
    "${REPO_ROOT}/service/scripts/run_service.sh" \
    "uvicorn service.app.main:app" \
    "vllm.entrypoints.openai.api_server"; do
    pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
    [[ -z "${pids}" ]] && continue
    echo "Force killing: ${pattern} (${pids})"
    # shellcheck disable=SC2086
    kill -9 ${pids} >/dev/null 2>&1 || true
  done
}

process_stop() {
  if [[ -f "${ENV_FILE}" ]]; then
    load_env
  fi
  stop_background_process "service"
  stop_background_process "vllm"
  kill_orphan_processes
  kill_port_processes "${SERVICE_PORT:-8088}"
  kill_port_processes "${VLLM_PORT:-8001}"
}

verify_stopped() {
  local service_port="${SERVICE_PORT:-8088}"
  local vllm_port="${VLLM_PORT:-8001}"
  local ok=true

  if lsof -ti :"${service_port}" >/dev/null 2>&1; then
    echo "ERROR: port ${service_port} is still in use." >&2
    ok=false
  fi
  if lsof -ti :"${vllm_port}" >/dev/null 2>&1; then
    echo "ERROR: port ${vllm_port} is still in use." >&2
    ok=false
  fi
  if pgrep -f "${REPO_ROOT}/service/scripts/run_vllm.sh" >/dev/null 2>&1 \
    || pgrep -f "vllm.entrypoints.openai.api_server" >/dev/null 2>&1 \
    || pgrep -f "uvicorn service.app.main:app" >/dev/null 2>&1; then
    echo "ERROR: LLM4Decompile processes are still running." >&2
    ok=false
  fi

  if [[ "${ok}" == "true" ]]; then
    echo "All services stopped."
    return 0
  fi
  echo "Run './service/scripts/service_ctl.sh status' for details." >&2
  return 1
}

systemd_stop_if_active() {
  systemd_units_active || return 0
  echo "Stopping systemd units (${TARGET_NAME})..."
  if [[ "${EUID}" -eq 0 ]]; then
    systemctl stop "${TARGET_NAME}"
    return 0
  fi
  if sudo -n systemctl stop "${TARGET_NAME}" 2>/dev/null; then
    return 0
  fi
  echo "WARNING: systemd units are active but sudo is required to stop them." >&2
  echo "  sudo systemctl stop ${TARGET_NAME}" >&2
  return 1
}

stop_all_services() {
  if [[ -f "${ENV_FILE}" ]]; then
    load_env
  fi
  systemd_stop_if_active || true
  process_stop
  verify_stopped
}

process_status() {
  if [[ -f "${ENV_FILE}" ]]; then
    load_env
  fi
  local name pid_file pid service_port vllm_port
  service_port="${SERVICE_PORT:-8088}"
  vllm_port="${VLLM_PORT:-8001}"

  for name in vllm service; do
    pid_file="${LOG_DIR}/${name}.pid"
    echo "=== ${name} (process) ==="
    if [[ -f "${pid_file}" ]] && kill -0 "$(cat "${pid_file}")" >/dev/null 2>&1; then
      pid="$(cat "${pid_file}")"
      echo "running (PID ${pid})"
      echo "log: ${LOG_DIR}/${name}.log"
    elif [[ "${name}" == "service" ]] && lsof -ti :"${service_port}" >/dev/null 2>&1; then
      echo "running (port ${service_port}, no pid file)"
    elif [[ "${name}" == "vllm" ]] && lsof -ti :"${vllm_port}" >/dev/null 2>&1; then
      echo "running (port ${vllm_port}, no pid file)"
    else
      echo "not running"
    fi
    echo
  done

  if lsof -ti :"${service_port}" >/dev/null 2>&1; then
    echo "Health check:"
    if command -v curl >/dev/null 2>&1; then
      curl -fsS "http://127.0.0.1:${service_port}/health" -w '\n' \
        || echo "Health check failed (service may still be starting)."
    else
      echo "curl not found; run: curl http://127.0.0.1:${service_port}/health"
    fi
  fi
}

show_status() {
  if [[ -f "${ENV_FILE}" ]]; then
    load_env
  fi
  echo "Backend for start: $(resolve_backend)"
  echo

  if systemd_available && systemd_units_installed; then
    echo "=== systemd ==="
    systemctl is-active "${VLLM_UNIT}" "${API_UNIT}" 2>/dev/null \
      | paste - - | awk '{print "vllm: "$1"  api: "$2}' || true
    systemctl status "${TARGET_NAME}" --no-pager 2>/dev/null | sed -n '1,3p' || true
    echo
  fi

  echo "=== process / ports ==="
  process_status
}

process_logs() {
  mkdir -p "${LOG_DIR}"
  touch "${LOG_DIR}/vllm.log" "${LOG_DIR}/service.log"
  exec tail -f "${LOG_DIR}/vllm.log" "${LOG_DIR}/service.log"
}

service_port() {
  if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
  fi
  echo "${SERVICE_PORT:-8088}"
}

print_usage() {
  cat <<EOF
Usage: $(basename "$0") <command>

Commands:
  setup      Install units, enable on boot, and start (first-time)
  install    Install or refresh systemd units only
  start      Start services
  stop       Stop services
  restart    Restart services (after editing service/.env)
  status     Show service status and health check
  logs       Follow journal logs
  uninstall  Remove systemd units

Examples:
  ./service/scripts/service_ctl.sh setup
  ./service/scripts/service_ctl.sh status
  ./service/scripts/service_ctl.sh restart

Note: stop works without sudo (cleans process mode + ports).
Start/setup may auto-elevate with sudo when using systemd on non-WSL hosts.
On WSL, auto mode uses process backend (logs in service/logs/).
Set LLM4DECOMPILE_PREFER_SYSTEMD=1 to force systemd on WSL.
EOF
}
