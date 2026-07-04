#!/usr/bin/env bash
# Benchmark serial DecompileSerial.java vs parallel DecompileParallel.java (consumer-stream v2).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/service/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/service/.env"
  set +a
fi

GHIDRA_MAX_CPU="${GHIDRA_MAX_CPU:-$(python3 - <<'PY'
import os
print(max(1, os.cpu_count() or 2))
PY
)}"
export GHIDRA_MAX_CPU

resolve_repo_path() {
  local raw_path="$1"
  if [[ "${raw_path}" = /* ]]; then
    echo "${raw_path}"
    return
  fi
  raw_path="${raw_path#./}"
  echo "${REPO_ROOT}/${raw_path}"
}

resolve_analyze_headless() {
  local candidate=""
  if [[ -n "${GHIDRA_ANALYZE_HEADLESS:-}" ]]; then
    candidate="$(resolve_repo_path "${GHIDRA_ANALYZE_HEADLESS}")"
    if [[ -x "${candidate}" ]]; then
      echo "${candidate}"
      return
    fi
  fi
  for fallback in \
    "${REPO_ROOT}/ghidra/ghidra_12.1.2_PUBLIC/support/analyzeHeadless" \
    "${REPO_ROOT}/ghidra/ghidra_11.1.2_PUBLIC/support/analyzeHeadless" \
    "${REPO_ROOT}/ghidra/ghidra_11.0.3_PUBLIC/support/analyzeHeadless"; do
    if [[ -x "${fallback}" ]]; then
      echo "${fallback}"
      return
    fi
  done
  echo ""
}

ghidra_version_label() {
  local analyze_headless="$1"
  local install_dir props version
  install_dir="$(cd "$(dirname "${analyze_headless}")/.." && pwd)"
  props="${install_dir}/Ghidra/application.properties"
  if [[ -f "${props}" ]]; then
    version="$(grep -E '^application\.version=' "${props}" | head -n1 | cut -d= -f2- | tr -d '[:space:]')"
    if [[ -n "${version}" ]]; then
      echo "${version}"
      return
    fi
  fi
  if [[ "${install_dir}" =~ ghidra_([0-9]+\.[0-9]+\.[0-9]+)_PUBLIC ]]; then
    echo "${BASH_REMATCH[1]}"
    return
  fi
  echo "unknown"
}

ANALYZE_HEADLESS="$(resolve_analyze_headless)"
GHIDRA_VERSION="$(ghidra_version_label "${ANALYZE_HEADLESS}")"
SCRIPT_PATH="${GHIDRA_SCRIPT_PATH:-${REPO_ROOT}/ghidra/postscripts}"
if [[ "${SCRIPT_PATH}" != /* ]]; then
  SCRIPT_PATH="$(resolve_repo_path "${SCRIPT_PATH}")"
fi
BINARY_PATH="${1:-}"
WORK_DIR="${2:-${REPO_ROOT}/service_data/benchmark_ghidra}"
MONITOR_CPU="${MONITOR_CPU:-0}"
SERIAL_POSTSCRIPT="${BENCHMARK_SERIAL_POSTSCRIPT:-DecompileSerial.java}"
PARALLEL_POSTSCRIPT="${BENCHMARK_PARALLEL_POSTSCRIPT:-DecompileParallel.java}"

if [[ -z "${BINARY_PATH}" ]]; then
  echo "Usage: $0 <binary_path> [work_dir]" >&2
  echo "Optional: MONITOR_CPU=1 to run pidstat sampling during parallel case." >&2
  exit 1
fi

if [[ -z "${ANALYZE_HEADLESS}" || ! -x "${ANALYZE_HEADLESS}" ]]; then
  echo "analyzeHeadless not found or not executable." >&2
  echo "Set GHIDRA_ANALYZE_HEADLESS in service/.env or install Ghidra under ${REPO_ROOT}/ghidra/." >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}/${SERIAL_POSTSCRIPT}" ]]; then
  echo "Serial postScript not found: ${SCRIPT_PATH}/${SERIAL_POSTSCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${SCRIPT_PATH}/${PARALLEL_POSTSCRIPT}" ]]; then
  echo "Parallel postScript not found: ${SCRIPT_PATH}/${PARALLEL_POSTSCRIPT}" >&2
  exit 1
fi

chmod +x "${SCRIPT_DIR}/configure_ghidra_cpu.sh"
"${SCRIPT_DIR}/configure_ghidra_cpu.sh"

mkdir -p "${WORK_DIR}"
rm -f "${WORK_DIR}/serial.c" "${WORK_DIR}/parallel.c"
rm -f "${WORK_DIR}/serial.stdout.log" "${WORK_DIR}/serial.stderr.log"
rm -f "${WORK_DIR}/parallel.stdout.log" "${WORK_DIR}/parallel.stderr.log"

count_markers() {
  local file_path="$1"
  python3 - <<PY
from pathlib import Path
text = Path("${file_path}").read_text(encoding="utf-8", errors="ignore")
print(text.count("// Function:"))
PY
}

parse_parallel_mode() {
  local log_path="$1"
  python3 - <<PY
import re
from pathlib import Path
text = Path("${log_path}").read_text(encoding="utf-8", errors="ignore")
match = re.search(r"DECOMPILE_PARALLEL mode=(\\S+)", text)
print(match.group(1) if match else "unknown")
PY
}

parse_decompile_ms() {
  local log_path="$1"
  local prefix="$2"
  python3 - <<PY
import re
from pathlib import Path
text = Path("${log_path}").read_text(encoding="utf-8", errors="ignore")
match = re.search(r"${prefix} timing_ms decompile=(\\d+)", text)
print(match.group(1) if match else "unknown")
PY
}

verify_postscript_log() {
  local label="$1"
  local stdout_log="$2"
  if grep -q "REPORT SCRIPT ERROR:" "${stdout_log}"; then
    echo "${label}: postScript failed. See ${stdout_log}" >&2
    grep "REPORT SCRIPT ERROR:" "${stdout_log}" | head -n3 >&2 || true
    return 1
  fi
  return 0
}

run_case() {
  local label="$1"
  local postscript="$2"
  shift 2
  local extra_args=("$@")
  local project_dir="${WORK_DIR}/${label}"
  local output_path="${WORK_DIR}/${label}.c"
  local stdout_log="${WORK_DIR}/${label}.stdout.log"
  local stderr_log="${WORK_DIR}/${label}.stderr.log"
  rm -rf "${project_dir}"
  rm -f "${output_path}" "${stdout_log}" "${stderr_log}"
  mkdir -p "${project_dir}"

  local monitor_pid=""
  if [[ "${label}" == "parallel" && "${MONITOR_CPU}" == "1" ]]; then
    chmod +x "${SCRIPT_DIR}/monitor_ghidra_cpu.sh"
    "${SCRIPT_DIR}/monitor_ghidra_cpu.sh" 1 600 "${WORK_DIR}/parallel_cpu.txt" &
    monitor_pid="$!"
  fi

  local start end elapsed
  start="$(date +%s.%N)"
  "${ANALYZE_HEADLESS}" \
    "${project_dir}" \
    "bench_${label}" \
    -import "${BINARY_PATH}" \
    -readOnly \
    -scriptPath "${SCRIPT_PATH}" \
    -postScript "${postscript}" \
    "${output_path}" \
    "${extra_args[@]}" \
    -deleteProject \
    -max-cpu "${GHIDRA_MAX_CPU}" \
    >"${stdout_log}" 2>"${stderr_log}"
  end="$(date +%s.%N)"
  elapsed="$(python3 - <<PY
print(f"{float('${end}') - float('${start}'):.3f}")
PY
)"

  if [[ -n "${monitor_pid}" ]]; then
    wait "${monitor_pid}" || true
  fi

  if ! verify_postscript_log "${label}" "${stdout_log}"; then
    return 1
  fi

  if [[ ! -f "${output_path}" ]]; then
    echo "${label}: failed (no output). See ${stderr_log}" >&2
    return 1
  fi

  local markers mode decompile_ms
  markers="$(count_markers "${output_path}")"
  if [[ "${label}" == "parallel" ]]; then
    mode="$(parse_parallel_mode "${stdout_log}")"
    decompile_ms="$(parse_decompile_ms "${stdout_log}" "DECOMPILE_PARALLEL")"
  else
    mode="serial_java"
    decompile_ms="$(parse_decompile_ms "${stdout_log}" "DECOMPILE_SERIAL")"
  fi
  echo "${label},seconds=${elapsed},functions=${markers},mode=${mode},decompile_ms=${decompile_ms},output=${output_path},log=${stdout_log}"
}

CHUNK_THRESHOLD="${GHIDRA_DECOMP_CHUNK_THRESHOLD:-500}"
SINGLE_QUEUE_LIMIT="${GHIDRA_DECOMP_SINGLE_QUEUE_LIMIT:-12000}"
echo "Benchmark binary: ${BINARY_PATH}"
echo "Ghidra version: ${GHIDRA_VERSION}"
echo "analyzeHeadless: ${ANALYZE_HEADLESS}"
echo "GHIDRA_MAX_CPU=${GHIDRA_MAX_CPU}"
echo "GHIDRA_DECOMP_CHUNK_THRESHOLD=${CHUNK_THRESHOLD}"
echo "GHIDRA_DECOMP_SINGLE_QUEUE_LIMIT=${SINGLE_QUEUE_LIMIT}"
echo "Serial postScript: ${SERIAL_POSTSCRIPT} (decompile.py is not used; Jython is unavailable on Ghidra 12+)"
echo

run_case "serial" "${SERIAL_POSTSCRIPT}"
run_case "parallel" "${PARALLEL_POSTSCRIPT}" "${CHUNK_THRESHOLD}" "${GHIDRA_MAX_CPU}" "${SINGLE_QUEUE_LIMIT}"

echo
echo "Compare function marker counts and elapsed seconds above."
echo "Serial log should include: DECOMPILE_SERIAL timing_ms"
echo "Parallel log should include: DECOMPILE_PARALLEL mode=consumer_stream"
if [[ "${MONITOR_CPU}" == "1" ]]; then
  echo "CPU sample: ${WORK_DIR}/parallel_cpu.txt"
fi
