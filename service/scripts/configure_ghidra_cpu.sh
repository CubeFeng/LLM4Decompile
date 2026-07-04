#!/usr/bin/env bash
# Align Ghidra launch.properties thread pools with GHIDRA_MAX_CPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

resolve_repo_path() {
  local raw_path="$1"
  if [[ "${raw_path}" = /* ]]; then
    echo "${raw_path}"
    return
  fi
  raw_path="${raw_path#./}"
  echo "${REPO_ROOT}/${raw_path}"
}

resolve_ghidra_install_dir() {
  if [[ -n "${GHIDRA_INSTALL_DIR:-}" ]]; then
    local install_dir
    install_dir="$(resolve_repo_path "${GHIDRA_INSTALL_DIR}")"
    if [[ -d "${install_dir}" ]]; then
      echo "${install_dir}"
      return
    fi
  fi

  local analyze_headless=""
  if [[ -n "${GHIDRA_ANALYZE_HEADLESS:-}" ]]; then
    analyze_headless="$(resolve_repo_path "${GHIDRA_ANALYZE_HEADLESS}")"
    if [[ ! -x "${analyze_headless}" ]]; then
      analyze_headless=""
    fi
  fi

  if [[ -z "${analyze_headless}" ]]; then
    for candidate in \
      "${REPO_ROOT}/ghidra/ghidra_12.1.2_PUBLIC/support/analyzeHeadless" \
      "${REPO_ROOT}/ghidra/ghidra_11.1.2_PUBLIC/support/analyzeHeadless" \
      "${REPO_ROOT}/ghidra/ghidra_11.0.3_PUBLIC/support/analyzeHeadless"; do
      if [[ -x "${candidate}" ]]; then
        analyze_headless="${candidate}"
        break
      fi
    done
  fi

  if [[ -z "${analyze_headless}" ]]; then
    echo "Unable to locate Ghidra install dir." >&2
    echo "Set GHIDRA_INSTALL_DIR or GHIDRA_ANALYZE_HEADLESS to an existing path." >&2
    echo "Checked repo candidates under: ${REPO_ROOT}/ghidra/" >&2
    exit 1
  fi

  local install_dir
  install_dir="$(cd "$(dirname "${analyze_headless}")/.." && pwd)"
  echo "${install_dir}"
}

default_physical_cpu_count() {
  python3 - <<'PY'
import re
import subprocess
from pathlib import Path

try:
    output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
    cores = sockets = None
    for line in output.splitlines():
        if line.strip().startswith("Core(s) per socket:"):
            cores = int(line.split(":", 1)[1].strip())
        elif line.strip().startswith("Socket(s):"):
            sockets = int(line.split(":", 1)[1].strip())
    if cores and sockets:
        print(max(1, cores * sockets))
        raise SystemExit
except Exception:
    pass

try:
    cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^cpu cores\s*:\s*(\d+)\s*$", cpuinfo, re.MULTILINE)
    if match:
        physical_ids = {
            int(value)
            for value in re.findall(r"^physical id\s*:\s*(\d+)\s*$", cpuinfo, re.MULTILINE)
        }
        sockets = len(physical_ids) or 1
        print(max(1, int(match.group(1)) * sockets))
        raise SystemExit
except Exception:
    pass

import os
print(max(1, (os.cpu_count() or 2) // 2))
PY
}

GHIDRA_MAX_CPU="${GHIDRA_MAX_CPU:-$(default_physical_cpu_count)}"

if ! [[ "${GHIDRA_MAX_CPU}" =~ ^[0-9]+$ ]] || [[ "${GHIDRA_MAX_CPU}" -lt 1 ]]; then
  echo "GHIDRA_MAX_CPU must be a positive integer, got: ${GHIDRA_MAX_CPU}" >&2
  exit 1
fi

GHIDRA_INSTALL_DIR="$(resolve_ghidra_install_dir)"
LAUNCH_PROPERTIES="${GHIDRA_INSTALL_DIR}/support/launch.properties"
ANALYZE_HEADLESS="${GHIDRA_INSTALL_DIR}/support/analyzeHeadless"

if [[ ! -f "${LAUNCH_PROPERTIES}" ]]; then
  echo "launch.properties not found: ${LAUNCH_PROPERTIES}" >&2
  exit 1
fi

python3 - "${LAUNCH_PROPERTIES}" "${GHIDRA_MAX_CPU}" <<'PY'
import re
import sys
from pathlib import Path

launch_path = Path(sys.argv[1])
max_cpu = int(sys.argv[2])
lines = launch_path.read_text(encoding="utf-8").splitlines()

filtered = []
for line in lines:
    stripped = line.strip()
    if re.search(r"cpu\.core\.(limit|override)=", stripped):
        continue
    if re.search(r"-XX:\+UseG1GC", stripped):
        continue
    if re.search(r"-XX:MaxGCPauseMillis=", stripped):
        continue
    if re.search(r"-XX:ParallelGCThreads=", stripped):
        continue
    filtered.append(line)

while filtered and filtered[-1].strip() == "":
    filtered.pop()

filtered.append(f"VMARGS=-Dcpu.core.override={max_cpu}")
launch_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
print(f"Updated {launch_path} with cpu.core.override={max_cpu}")
PY

if [[ -f "${ANALYZE_HEADLESS}" ]]; then
  python3 - "${ANALYZE_HEADLESS}" "${GHIDRA_MAX_CPU}" <<'PY'
import re
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
max_cpu = int(sys.argv[2])
compiler_count = max(2, max_cpu // 2)
text = script_path.read_text(encoding="utf-8")

replacements = [
    (
        r'VMARG_LIST="-XX:ParallelGCThreads=\d+ -XX:CICompilerCount=\d+ -Djava\.awt\.headless=true"',
        f'VMARG_LIST="-XX:ParallelGCThreads={max_cpu} -XX:CICompilerCount={compiler_count} -Djava.awt.headless=true"',
    ),
    (
        r'VMARG_LIST="-XX:ParallelGCThreads=\d+ -XX:CICompilerCount=\d+ "',
        f'VMARG_LIST="-XX:ParallelGCThreads={max_cpu} -XX:CICompilerCount={compiler_count} "',
    ),
]

count = 0
for pattern, replacement in replacements:
    text, n = re.subn(pattern, replacement, text, count=1)
    count += n
    if n:
        break

if count == 0:
    raise SystemExit("VMARG_LIST GC thread settings not found in analyzeHeadless")

script_path.write_text(text, encoding="utf-8")
print(
    f"Updated {script_path} with ParallelGCThreads={max_cpu} "
    f"CICompilerCount={compiler_count}"
)
PY
fi

if [[ -n "${GHIDRA_MAXMEM:-}" && -f "${ANALYZE_HEADLESS}" ]]; then
  python3 - "${ANALYZE_HEADLESS}" "${GHIDRA_MAXMEM}" <<'PY'
import re
import sys
from pathlib import Path

script_path = Path(sys.argv[1])
maxmem = sys.argv[2]
text = script_path.read_text(encoding="utf-8")

if re.search(r"^GHIDRA_HEADLESS_MAXMEM=", text, flags=re.MULTILINE):
    print(
        f"Ghidra 12.x: heap size via GHIDRA_MAXMEM env ({maxmem}); "
        "analyzeHeadless reads it at launch (no file patch needed)."
    )
    raise SystemExit(0)

updated, count = re.subn(r"^MAXMEM=.*$", f"MAXMEM={maxmem}", text, count=1, flags=re.MULTILINE)
if count == 0:
    raise SystemExit(
        "MAXMEM= not found in analyzeHeadless and not a Ghidra 12.x script; "
        "set GHIDRA_MAXMEM in service/.env instead."
    )
script_path.write_text(updated, encoding="utf-8")
print(f"Updated {script_path} with MAXMEM={maxmem}")
PY
fi

echo "Ghidra install dir: ${GHIDRA_INSTALL_DIR}"
echo "Configured GHIDRA_MAX_CPU=${GHIDRA_MAX_CPU}"
if [[ -n "${GHIDRA_MAXMEM:-}" ]]; then
  echo "Configured GHIDRA_MAXMEM=${GHIDRA_MAXMEM}"
fi
