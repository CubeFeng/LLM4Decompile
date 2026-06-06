# LLM4Decompile 演示版服务启动说明

本目录提供一个演示版 HTTP 服务，将 LLM4Decompile 的 Ghidra 反编译和 vLLM 源码还原能力封装成 API。

推荐部署形态：

```text
终端 1：vLLM OpenAI-compatible Server
  使用安装了 vLLM、torch、CUDA 依赖的 conda 环境
  默认监听 8001

终端 2：LLM4Decompile FastAPI Service
  使用轻量服务环境，只安装 fastapi/uvicorn/pydantic
  默认监听 8088
```

## 0. 前置依赖

除 Python 环境外，还需要在本机准备：

| 依赖 | 说明 |
|------|------|
| Ghidra | 解压到 `ghidra/ghidra_11.0.3_PUBLIC`，或在 `service/.env` 中设置 `GHIDRA_ANALYZE_HEADLESS`、`GHIDRA_POSTSCRIPT` |
| Java | Ghidra headless 运行需要 |
| 模型权重 | 下载到 `models/llm4decompile-1.3b-v2`，或修改 `VLLM_MODEL_PATH` |
| GPU + CUDA | vLLM 推理需要；RTX 20xx 请设置 `VLLM_DTYPE=half` |

`/health` 中 `ghidra_available=false` 通常是 Ghidra/Java 路径未配置；`vllm_available=false` 通常是 vLLM 未启动或端口不一致。

## 1. 准备配置

在**仓库根目录**执行：

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cp service/.env.example service/.env
```

编辑 `service/.env`，日常通常只需要确认这些配置：

```bash
# 按本机 conda 路径填写；在 vLLM 环境中运行 run_vllm.sh 时可留空
VLLM_PYTHON=/path/to/miniconda3/envs/llm4decompile/bin/python
VLLM_MODEL_PATH=./models/llm4decompile-1.3b-v2
VLLM_PORT=8001
VLLM_BASE_URL=http://127.0.0.1:8001/v1
VLLM_MODEL=llm4decompile
VLLM_DTYPE=half
```

注意：

- 如果在安装了 vLLM 的 conda 环境中执行 `run_vllm.sh`，`VLLM_PYTHON` 可以留空。
- 如果不想切换 conda 环境，`VLLM_PYTHON` 应指向安装了 `vllm` 的 Python。
- `VLLM_PORT` 和 `VLLM_BASE_URL` 的端口必须一致。
- RTX 20xx/Turing GPU 不支持 bfloat16，`VLLM_DTYPE` 应设置为 `half`。
- Ampere 及更新 GPU 可以尝试 `VLLM_DTYPE=auto`。

## 2. 环境划分

推荐使用两个 conda 环境。

### vLLM 模型环境

用于启动模型服务：

```bash
conda activate llm4decompile
python -c "import vllm; print(vllm.__version__)"
```

如果该命令失败，说明当前环境没有安装 vLLM。可以使用项目根目录依赖安装：

```bash
pip install -r requirements.txt
```

### FastAPI 服务环境

用于启动轻量 HTTP 服务：

```bash
conda create -n llm4decompile-service python=3.10
conda activate llm4decompile-service
pip install -r service/requirements.txt
```

不要把 `vllm`、`torch` 这类重依赖安装到轻量服务环境，除非你明确想把两套环境合并。

## 3. 启动服务

### 方式 A：一键后台启动（推荐）

在仓库根目录：

```bash
./service/scripts/start_demo.sh
```

日志写入 `service/logs/`。vLLM 首次加载模型可能需要 30s 以上。

健康检查：

```bash
curl http://127.0.0.1:8088/health -w '\n'
```

停止：

```bash
./service/scripts/stop_demo.sh
```

### 方式 B：两个终端分别前台启动

终端 1（vLLM）：

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
conda activate llm4decompile
./service/scripts/run_vllm.sh
```

脚本会自动读取 `service/.env`，并打印实际使用的配置，例如：

```text
Starting vLLM OpenAI-compatible server
  Python: .../envs/llm4decompile/bin/python
  Model path: .../models/llm4decompile-1.3b-v2
  Served model: llm4decompile
  Bind: 0.0.0.0:8001
  dtype: half
```

如果不想切换 conda 环境，也可以直接通过 `VLLM_PYTHON` 指定：

```bash
VLLM_PYTHON=/path/to/miniconda3/envs/llm4decompile/bin/python \
./service/scripts/run_vllm.sh
```

终端 2（FastAPI）：

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
conda activate llm4decompile-service
./service/scripts/run_service.sh
```

默认监听 `http://127.0.0.1:8088`。前台运行时在该终端按 `Ctrl+C` 停止。

## 4. 验证服务

### 健康检查

```bash
curl http://127.0.0.1:8088/health -w '\n'
```

正常时（节选）：

```json
{
  "status": "ok",
  "ghidra_available": true,
  "vllm_available": true,
  "data_dir_writable": true,
  "model": "llm4decompile",
  "busy": false
}
```

任一依赖不可用时 `status` 为 `"degraded"`。若 `vllm_available=false`，说明 FastAPI 访问不到 `VLLM_BASE_URL`。

### 准备测试二进制

```bash
gcc samples/sample.c -lm -o /tmp/llm4decompile_sample
```

### 创建反编译任务

上传接口接收**原始二进制请求体**（非 multipart）。文件名可通过 query `filename` 或请求头 `X-Filename` / `X-Upload-Filename` 指定：

```bash
curl --data-binary @/tmp/llm4decompile_sample \
  "http://127.0.0.1:8088/api/v1/decompile/tasks?filename=sample_elf" \
  -w '\n'
```

可选 query 参数：

- `use_llm=false`：跳过 vLLM，直接使用 Ghidra raw
- `fallback_to_ghidra_raw=false`：vLLM 全部失败时不降级，任务标记为 failed

演示版**同时只处理一个任务**。服务忙时再次提交会返回 **409 Conflict**。

返回示例：

```json
{
  "task_id": "dec_xxx",
  "status": "pending",
  "stage": "queued"
}
```

### 查询任务状态

创建任务后需**轮询**直到完成（复杂二进制可能需要数分钟）：

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx -w '\n'
```

运行中（节选）：

```json
{
  "task_id": "dec_xxx",
  "status": "running",
  "stage": "llm_running",
  "progress": 65,
  "message": "Refining functions with vLLM"
}
```

完成后（节选）：

```json
{
  "task_id": "dec_xxx",
  "status": "completed",
  "stage": "completed",
  "progress": 100,
  "message": "Decompilation completed"
}
```

### 查看结果摘要

任务 `status=completed` 后：

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx/result -w '\n'
```

重点字段（节选）：

```json
{
  "fallback_used": false,
  "stats": {
    "function_count": 28,
    "refined_count": 20,
    "failed_count": 2
  },
  "inference_errors": []
}
```

- `stats.refined_count`：LLM **推理成功**的函数数量，不是二进制里的函数总数。
- `fallback_used=true`：全部函数 LLM 失败且走了 Ghidra raw 降级；此时 `result.zip` 只含 `*_ghidra.*`。
- `fallback_used=false` 时仍可能有少量函数失败，见 `inference_errors`；`result.zip` 仍为 `*_refined.*`。

### 下载结果包

```bash
curl -o /tmp/decompile_result.zip \
  http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx/archive \
  -w '\n'

unzip -l /tmp/decompile_result.zip
```

`result.zip` 只包含最终代码文件（扩展名不限）：

- LLM 优化成功：仅含 `{safe_name}_refined.*`
- LLM 优化失败且启用降级：仅含 `{safe_name}_ghidra.*`

不会同时打包两种输出。完整 manifest 还保存在 `service_data/tasks/{task_id}/manifest.json`。

## 5. 日志说明

后台启动（`start_demo.sh`）时，日志写入 `service/logs/`：

| 文件 | 内容 |
|------|------|
| `service.log` | FastAPI 服务日志，**含 LLM 推理批次与逐函数耗时**（推荐看这个） |
| `vllm.log` | vLLM 引擎底层日志（逐条 HTTP 请求、token 吞吐等） |

### 查看 LLM 推理进度

```bash
tail -f service/logs/service.log
```

任务进入 LLM 阶段后，会看到类似输出：

```text
2025-06-06 00:39:50 INFO [service.app.pipeline] task=dec_xxx entering LLM refinement stage
2025-06-06 00:39:50 INFO [service.app.inference_client] task=dec_xxx LLM inference started: total=28 refine=22 skipped=6
2025-06-06 00:40:44 INFO [service.app.inference_client] task=dec_xxx LLM inference [1/22] finished: main (53.1s, ok)
...
2025-06-06 00:41:06 INFO [service.app.inference_client] task=dec_xxx LLM inference finished: refined=22 success=20 failed=2 duration=76.3s
```

`vllm.log` 里每条 `Received request` 对应一次函数级 HTTP 调用，但不会标注任务边界和整体耗时，排查推理进度请优先看 `service.log`。

## 6. 常见问题

### No module named vllm

说明 `run_vllm.sh` 使用的 Python 环境没有安装 vLLM。

解决方式：

```bash
conda activate llm4decompile
./service/scripts/run_vllm.sh
```

或在 `service/.env` 中设置：

```bash
VLLM_PYTHON=/path/to/miniconda3/envs/llm4decompile/bin/python
```

### Bfloat16 is only supported on GPUs with compute capability at least 8.0

RTX 20xx/Turing GPU 不支持 bfloat16。设置：

```bash
VLLM_DTYPE=half
```

### Port 8001 is already in use

说明端口被占用。先查看占用：

```bash
lsof -i :8001
```

可以执行 `./service/scripts/stop_demo.sh`，或改端口。若改端口，必须同时改：

```bash
VLLM_PORT=8002
VLLM_BASE_URL=http://127.0.0.1:8002/v1
```

### health 中 ghidra_available=false

检查：

- Ghidra 是否已解压到 `ghidra/ghidra_11.0.3_PUBLIC`。
- `GHIDRA_ANALYZE_HEADLESS` 是否可执行。
- `GHIDRA_POSTSCRIPT` 是否指向 `ghidra/decompile.py`。
- Java 是否已安装并在 `PATH` 中。

### health 中 vllm_available=false

检查：

- vLLM 服务是否已启动。
- `VLLM_BASE_URL` 是否和 vLLM 端口一致。
- FastAPI 服务是否重新启动以加载最新 `.env`。

### 提交任务返回 409 Conflict

演示版单任务串行执行。等待当前任务完成，或执行 `./service/scripts/stop_demo.sh` 后重启再试。

### 任务完成但没有 refined 输出

检查任务结果：

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx/result -w '\n'
```

如果 `fallback_used=true`，查看 `inference_errors` 字段。常见原因：

- vLLM 未启动。
- `VLLM_MODEL` 和 vLLM `--served-model-name` 不一致。
- vLLM 端口和 `VLLM_BASE_URL` 不一致。
- 模型显存不足或 dtype 设置错误。

## 7. 停止服务

| 启动方式 | 停止方式 |
|----------|----------|
| `./service/scripts/start_demo.sh` | `./service/scripts/stop_demo.sh` |
| 前台 `run_vllm.sh` + `run_service.sh` | 各终端 `Ctrl+C` |

停止顺序没有强制要求。重新启动 FastAPI 服务后会重新读取 `service/.env`。
