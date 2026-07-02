# LLM4Decompile 演示版服务启动说明

本目录提供 HTTP 服务，将 Ghidra 反编译与 vLLM 源码还原封装为 API。

**推荐启动方式**：在仓库根目录使用 `./service/scripts/service_ctl.sh`（WSL 用 process 模式，Ubuntu 用 systemd）。调试时可前台运行 `run_vllm.sh` + `run_service.sh`。
## 0. 前置依赖

除 Python 环境外，还需要在本机准备：

| 依赖 | 说明 |
|------|------|
| Ghidra | 解压到 `ghidra/ghidra_11.0.3_PUBLIC`，或在 `service/.env` 中设置 `GHIDRA_ANALYZE_HEADLESS`、`GHIDRA_POSTSCRIPT`（默认并行脚本 `DecompileParallel.java`） |
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
- 单个二进制上传上限默认 **500 MiB**（`MAX_BINARY_SIZE_BYTES=524288000`）；与 DeepAudit 对接时需保持一致。

### Ghidra 多核提速（并行反编译）

默认 postScript 为 [`ghidra/postscripts/DecompileParallel.java`](../ghidra/postscripts/DecompileParallel.java)（consumer-stream）：全量函数单队列并行反编译，worker 分片写盘后再按地址 merge。

Auto Analysis 与 postScript 均通过 `-max-cpu` 与 `launch.properties` 的 `cpu.core.override` 对齐线程池。`run_service.sh` 启动前会自动调用 `configure_ghidra_cpu.sh`。

**8 核 16 线程机器 Aggressive 配置**：

| 配置项 | 推荐值 | 说明 |
|--------|--------|------|
| `GHIDRA_POSTSCRIPT` | `./ghidra/postscripts/DecompileParallel.java` | 勿用 `decompile.py`（单线程） |
| `GHIDRA_SCRIPT_PATH` | `./ghidra/postscripts` | 仅含 postScript |
| `GHIDRA_CPU_PROFILE` | `aggressive` | 未显式设置 `GHIDRA_MAX_CPU` 时使用逻辑核数 |
| `GHIDRA_MAX_CPU` | `16` | 逻辑线程数（8C16T） |
| `GHIDRA_MAXMEM` | `16G` | JVM 堆 |
| `GHIDRA_DECOMP_SINGLE_QUEUE_LIMIT` | `12000` | 超过后分段 consumer |
| `GHIDRA_DECOMP_CHUNK_THRESHOLD` | `500` | 分段模式 segment 大小 |

部署前可手动执行一次：

```bash
export GHIDRA_MAX_CPU=16
export GHIDRA_MAXMEM=16G
./service/scripts/configure_ghidra_cpu.sh
```

对比串行与并行：

```bash
GHIDRA_MAX_CPU=16 ./service/scripts/benchmark_ghidra.sh /path/to/binary
```

验收 checklist：

1. `curl http://127.0.0.1:8088/health` → `ghidra_parallel_enabled: true`，`ghidra_cpu_configured: true`
2. `ghidra.stdout.log` 含 `DECOMPILE_PARALLEL mode=consumer_stream` 与 `timing_ms decompile=...`
3. manifest `stats.ghidra_parallel_mode` 为 `consumer_stream`

回退到串行 postScript：

```bash
GHIDRA_POSTSCRIPT=./ghidra/postscripts/decompile.py
```

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

### 方式 A：`service_ctl.sh`（推荐）

WSL 自动使用 process 模式（日志在 `service/logs/`）；Ubuntu 自动使用 systemd（`service_ctl.sh logs`）。详见 [deploy/systemd/README.md](deploy/systemd/README.md)。

```bash
./service/scripts/service_ctl.sh setup      # 首次：安装并启动
./service/scripts/service_ctl.sh start
./service/scripts/service_ctl.sh stop
./service/scripts/service_ctl.sh status
./service/scripts/service_ctl.sh restart    # 修改 service/.env 后
./service/scripts/service_ctl.sh logs
```

vLLM 冷启动约 30s。健康检查：`curl http://127.0.0.1:8088/health`

### 方式 B：前台调试（两个终端）
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

## 5. 日志

| 模式 | 查看方式 |
|------|----------|
| process（WSL 默认） | `tail -f service/logs/service.log`（LLM 进度）、`service/logs/vllm.log` |
| systemd（Ubuntu） | `./service/scripts/service_ctl.sh logs` |
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

可以执行 `./service/scripts/service_ctl.sh stop`，或改端口。若改端口，必须同时改：

```bash
VLLM_PORT=8002
VLLM_BASE_URL=http://127.0.0.1:8002/v1
```

### health 中 ghidra_available=false

检查：

- Ghidra 是否已解压到 `ghidra/ghidra_11.0.3_PUBLIC`。
- `GHIDRA_ANALYZE_HEADLESS` 是否可执行。
- `GHIDRA_POSTSCRIPT` 是否指向 `ghidra/postscripts/DecompileParallel.java`（或回退 `ghidra/decompile.py`）。
- `ghidra_cpu_configured` 是否为 `true`（否则运行 `./service/scripts/configure_ghidra_cpu.sh`）。
- Java 是否已安装并在 `PATH` 中。

### health 中 vllm_available=false

检查：

- vLLM 服务是否已启动。
- `VLLM_BASE_URL` 是否和 vLLM 端口一致。
- FastAPI 服务是否重新启动以加载最新 `.env`。

### 提交任务返回 409 Conflict

演示版单任务串行执行。等待当前任务完成，或执行 `./service/scripts/service_ctl.sh stop` 后重启再试。

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