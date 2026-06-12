#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SERVICE_DIR}/.." && pwd)"

ENV_FILE="${VLLM_ENV_FILE:-${SERVICE_DIR}/.env}"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

MODEL_PATH="${VLLM_MODEL_PATH:-${REPO_ROOT}/models/llm4decompile-1.3b-v2}"
SERVED_MODEL_NAME="${VLLM_MODEL:-llm4decompile}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8002}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${VLLM_DTYPE:-auto}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-true}"
CUDA_DEVICES="${VLLM_CUDA_VISIBLE_DEVICES:-}"
PYTHON_BIN="${VLLM_PYTHON:-python}"

if [[ "${MODEL_PATH}" != /* ]]; then
  MODEL_PATH="${REPO_ROOT}/${MODEL_PATH#./}"
fi

if [[ -n "${CUDA_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
fi

if ! "${PYTHON_BIN}" -c "import vllm" >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: vLLM is not installed in the Python environment used by this script.

Current Python:
  $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)' 2>/dev/null || echo "${PYTHON_BIN}")

Run this script from the conda environment that has vLLM installed, or set VLLM_PYTHON explicitly:

  conda activate <your-llm4decompile-vllm-env>
  ./service/scripts/run_vllm.sh

or:

  VLLM_PYTHON=/path/to/vllm-env/bin/python ./service/scripts/run_vllm.sh

The FastAPI service can still run in the lightweight llm4decompile-service environment.
EOF
  exit 1
fi

if [[ "${DTYPE}" == "auto" ]]; then
  AUTO_DTYPE="$("${PYTHON_BIN}" - <<'PY'
try:
    import torch
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        print("half" if major < 8 else "auto")
    else:
        print("auto")
except Exception:
    print("auto")
PY
)"
  if [[ "${AUTO_DTYPE}" != "${DTYPE}" ]]; then
    echo "Detected GPU does not support bfloat16 reliably; using dtype=${AUTO_DTYPE} instead of auto."
    DTYPE="${AUTO_DTYPE}"
  fi
fi

if [[ ! -d "${MODEL_PATH}" && ! -f "${MODEL_PATH}" ]]; then
  cat >&2 <<EOF
ERROR: VLLM_MODEL_PATH does not exist:
  ${MODEL_PATH}

Set VLLM_MODEL_PATH in ${ENV_FILE} or export it before running this script.
EOF
  exit 1
fi

if "${PYTHON_BIN}" - <<PY
import socket
host = "127.0.0.1" if "${HOST}" in ("0.0.0.0", "::") else "${HOST}"
port = int("${PORT}")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
    sock.settimeout(0.2)
    raise SystemExit(0 if sock.connect_ex((host, port)) == 0 else 1)
PY
then
  cat >&2 <<EOF
ERROR: Port ${PORT} is already in use.

Stop the existing process on this port, or change both values to the same new port:

  VLLM_PORT=8002
  VLLM_BASE_URL=http://127.0.0.1:8002/v1

If FastAPI still points to 8001 while vLLM runs on another port, model inference will not be used.
EOF
  exit 1
fi

echo "Starting vLLM OpenAI-compatible server"
echo "  Python: $("${PYTHON_BIN}" -c 'import sys; print(sys.executable)')"
echo "  Model path: ${MODEL_PATH}"
echo "  Served model: ${SERVED_MODEL_NAME}"
echo "  Bind: ${HOST}:${PORT}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<not set>}"
echo "  max_model_len: ${MAX_MODEL_LEN}"
echo "  gpu_memory_utilization: ${GPU_MEMORY_UTILIZATION}"
echo "  dtype: ${DTYPE}"
echo "  tensor_parallel_size: ${TENSOR_PARALLEL_SIZE}"
echo "  max_num_seqs: ${MAX_NUM_SEQS}"

ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ "${TRUST_REMOTE_CODE}" == "true" || "${TRUST_REMOTE_CODE}" == "1" ]]; then
  ARGS+=(--trust-remote-code)
fi

"${PYTHON_BIN}" "${ARGS[@]}"
