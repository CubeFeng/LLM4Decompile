#!/usr/bin/env bash
# Load service/.env. By default it preserves non-empty variables already in the
# environment; pass "1" as the second argument to let explicit non-empty .env
# values override variables injected by systemd unit files.

load_env_file() {
  local path="${1:?env file path required}"
  local override_non_empty="${2:-0}"
  [[ -f "${path}" ]] || return 0

  local raw_line line key value
  while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
    line="${raw_line%%#*}"
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line%"${line##*[![:space:]]}"}"
    [[ -z "${line}" || "${line}" != *"="* ]] && continue

    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ "${value}" =~ ^\".*\"$ ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" =~ ^\'.*\'$ ]]; then
      value="${value:1:${#value}-2}"
    fi

    if [[ "${override_non_empty}" == "1" && -n "${value}" ]]; then
      export "${key}=${value}"
    elif [[ -z "${!key+x}" ]]; then
      export "${key}=${value}"
    elif [[ -z "${!key}" && -n "${value}" ]]; then
      export "${key}=${value}"
    fi
  done <"${path}"
}
