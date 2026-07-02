#!/usr/bin/env bash
# Sample Java/Ghidra CPU usage while analyzeHeadless runs (pidstat).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -f "${REPO_ROOT}/service/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/service/.env"
  set +a
fi

INTERVAL_SECONDS="${1:-1}"
DURATION_SECONDS="${2:-120}"
OUTPUT_PATH="${3:-${REPO_ROOT}/service_data/ghidra_cpu_sample.txt}"

if ! command -v pidstat >/dev/null 2>&1; then
  echo "pidstat not found. Install sysstat: sudo apt install sysstat" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "Waiting for analyzeHeadless Java process (timeout ${DURATION_SECONDS}s)..."
deadline=$((SECONDS + DURATION_SECONDS))
target_pid=""

while [[ "${SECONDS}" -lt "${deadline}" ]]; do
  target_pid="$(pgrep -f 'analyzeHeadless|ghidra\.GHIDRA' | head -n 1 || true)"
  if [[ -n "${target_pid}" ]]; then
    break
  fi
  sleep 0.5
done

if [[ -z "${target_pid}" ]]; then
  echo "No Ghidra Java process found within ${DURATION_SECONDS}s." >&2
  exit 1
fi

echo "Sampling PID ${target_pid} every ${INTERVAL_SECONDS}s (Ctrl+C to stop early)..."
{
  echo "# pid=${target_pid} interval=${INTERVAL_SECONDS}s started=$(date -Is)"
  pidstat -u -h -p "${target_pid}" "${INTERVAL_SECONDS}"
} | tee "${OUTPUT_PATH}"

python3 - "${OUTPUT_PATH}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
usages = []
for line in lines:
    if line.startswith("#") or line.startswith("Linux") or "UID" in line or "Average:" in line:
        continue
    parts = line.split()
    if len(parts) < 8:
        continue
    try:
        usages.append(float(parts[7]))
    except ValueError:
        continue

if not usages:
    print("No pidstat samples captured.")
    raise SystemExit(0)

avg = sum(usages) / len(usages)
peak = max(usages)
print(f"Samples={len(usages)} avg_cpu_percent={avg:.1f} peak_cpu_percent={peak:.1f}")
print("Target during postScript: avg >= 700% on 16T host (7 core-equivalents).")
PY
