#!/usr/bin/env bash
set -euo pipefail

# shellcheck source=service_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/service_common.sh"

cmd="${1:-}"

needs_root() {
  case "${cmd}" in
    stop|status|logs|restart) return 1 ;;
  esac
  [[ "$(resolve_backend)" == "systemd" ]] || return 1
  case "${cmd}" in
    -h|--help|help|"") return 1 ;;
    *) return 0 ;;
  esac
}

if needs_root; then
  ensure_root "$@"
fi

BACKEND="$(resolve_backend)"
shift || true

systemctl_cmd() {
  if [[ "${EUID}" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo systemctl "$@"
  fi
}

print_systemd_next_steps() {
  echo
  echo "vLLM model loading logs:"
  echo "  journalctl -u ${VLLM_UNIT} -f"
  echo
  echo "Health check (vLLM may take several minutes to load):"
  echo "  curl http://127.0.0.1:$(service_port)/health"
}

run_systemd_start() {
  echo "Starting systemd units..."
  systemctl_cmd reset-failed "${TARGET_NAME}" "${VLLM_UNIT}" "${API_UNIT}" 2>/dev/null || true
  systemctl_cmd start "${VLLM_UNIT}" "${API_UNIT}" "${TARGET_NAME}"
  echo "Started via systemd."
  echo
  systemctl_cmd --no-pager --lines=0 status "${VLLM_UNIT}" "${API_UNIT}" || true
  print_systemd_next_steps
}

run_systemd_restart() {
  echo "Restarting systemd units..."
  systemctl_cmd restart "${VLLM_UNIT}" "${API_UNIT}"
  print_systemd_next_steps
}

case "${cmd}" in
  setup)
    if [[ "${BACKEND}" == "systemd" ]]; then
      if [[ "${EUID}" -ne 0 ]]; then
        exec sudo -E "$0" setup
      fi
      install_units
      echo "Enabling and starting ${TARGET_NAME}..."
      systemctl_cmd enable "${TARGET_NAME}"
      run_systemd_start
    else
      echo "Backend: process"
      if is_wsl; then
        echo "WSL detected — using process mode (logs in service/logs/)."
        echo "Set LLM4DECOMPILE_PREFER_SYSTEMD=1 to use systemd on WSL."
        echo
      fi
      process_start
    fi
    ;;
  install)
    if [[ "${BACKEND}" == "systemd" ]]; then
      if [[ "${EUID}" -ne 0 ]]; then
        exec sudo -E "$0" install
      fi
      install_units
      echo "Units installed. Start with: ./service/scripts/service_ctl.sh start"
    else
      echo "Backend: process — no unit files to install."
      echo "Use './service/scripts/service_ctl.sh start' to run in the background."
    fi
    ;;
  start)
    if [[ "${BACKEND}" == "systemd" ]]; then
      run_systemd_start
    else
      echo "Backend: process"
      process_start
    fi
    ;;
  stop)
    stop_all_services
    ;;
  restart)
    stop_all_services || true
    if [[ "${BACKEND}" == "systemd" ]]; then
      run_systemd_restart
    else
      process_start
    fi
    ;;
  status)
    show_status
    ;;
  logs)
    if [[ "${BACKEND}" == "systemd" ]] && systemd_units_active; then
      exec journalctl -u "${VLLM_UNIT}" -u "${API_UNIT}" -f
    fi
    process_logs
    ;;
  uninstall)
    stop_all_services || true
    if systemd_units_installed; then
      if [[ "${EUID}" -ne 0 ]]; then
        echo "Removing systemd unit files requires root:"
        echo "  sudo $0 uninstall"
        exit 1
      fi
      for unit in "${TARGET_NAME}" "${API_UNIT}" "${VLLM_UNIT}"; do
        systemctl disable "${unit}" 2>/dev/null || true
        rm -f "${SYSTEMD_UNIT_DIR}/${unit}"
      done
      systemctl daemon-reload
      systemctl reset-failed "${TARGET_NAME}" "${API_UNIT}" "${VLLM_UNIT}" 2>/dev/null || true
    fi
    echo "Uninstall complete."
    ;;
  -h|--help|help|"")
    print_usage
    ;;
  *)
    echo "ERROR: unknown command: ${cmd}" >&2
    echo >&2
    print_usage >&2
    exit 1
    ;;
esac
