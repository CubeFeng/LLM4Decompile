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

## 1. 准备配置

在仓库根目录执行：

```bash
cd /home/feng/LLM4Decompile/LLM4Decompile
cp service/.env.example service/.env
```

编辑 `service/.env`，日常通常只需要确认这些配置：

```bash
VLLM_PYTHON=/home/feng/miniconda3/envs/llm4decompile/bin/python
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

## 3. 启动 vLLM 服务

终端 1：

```bash
cd /home/feng/LLM4Decompile/LLM4Decompile
conda activate llm4decompile
./service/scripts/run_vllm.sh
```

脚本会自动读取 `service/.env`，并打印实际使用的配置，例如：

```text
Starting vLLM OpenAI-compatible server
  Python: /home/feng/miniconda3/envs/llm4decompile/bin/python
  Model path: /home/feng/LLM4Decompile/LLM4Decompile/models/llm4decompile-1.3b-v2
  Served model: llm4decompile
  Bind: 0.0.0.0:8001
  dtype: half
```

如果不想切换 conda 环境，也可以直接通过 `VLLM_PYTHON` 指定：

```bash
VLLM_PYTHON=/home/feng/miniconda3/envs/llm4decompile/bin/python \
./service/scripts/run_vllm.sh
```

## 4. 启动 FastAPI 服务

终端 2：

```bash
cd /home/feng/LLM4Decompile/LLM4Decompile
conda activate llm4decompile-service
./service/scripts/run_service.sh
```

默认监听：

```text
http://127.0.0.1:8088
```

## 5. 验证服务

### 健康检查

```bash
curl http://127.0.0.1:8088/health -w '\n'
```

期望看到：

```json
{
  "ghidra_available": true,
  "vllm_available": true,
  "data_dir_writable": true,
  "busy": false
}
```

如果 `vllm_available=false`，说明 FastAPI 服务访问不到 `VLLM_BASE_URL`。

### 准备测试二进制

```bash
gcc samples/sample.c -lm -o /tmp/llm4decompile_sample
```

### 创建反编译任务

当前上传接口接收原始二进制请求体：

```bash
curl --data-binary @/tmp/llm4decompile_sample \
  "http://127.0.0.1:8088/api/v1/decompile/tasks?filename=sample_elf" \
  -w '\n'
```

返回示例：

```json
{
  "task_id": "dec_xxx",
  "status": "pending",
  "stage": "queued"
}
```

### 查询任务状态

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx -w '\n'
```

完成后应看到：

```json
{
  "status": "completed",
  "stage": "completed"
}
```

### 查看结果摘要

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx/result -w '\n'
```

重点检查：

```json
{
  "fallback_used": false,
  "stats": {
    "refined_count": 1
  }
}
```

如果 `fallback_used=true`，说明 vLLM 没有成功生成 refined C，服务使用了 Ghidra raw 兜底。此时查看结果中的 `inference_errors`。

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

不会同时打包两种输出；`manifest.json` 中的 `fallback_used` 字段标识是否走了降级路径。

## 6. 日志说明

后台启动时（`start_demo.sh`），日志分别写入 `service/logs/`：

| 文件 | 内容 |
|------|------|
| `service.log` | FastAPI 服务日志，**含 LLM 推理开始/结束**（推荐看这个） |
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

## 7. 常见问题

### No module named vllm

说明 `run_vllm.sh` 使用的 Python 环境没有安装 vLLM。

解决方式：

```bash
conda activate llm4decompile
./service/scripts/run_vllm.sh
```

或在 `service/.env` 中设置：

```bash
VLLM_PYTHON=/home/feng/miniconda3/envs/llm4decompile/bin/python
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

可以停止旧进程，或改端口。若改端口，必须同时改：

```bash
VLLM_PORT=8002
VLLM_BASE_URL=http://127.0.0.1:8002/v1
```

### health 中 vllm_available=false

检查：

- vLLM 服务是否已启动。
- `VLLM_BASE_URL` 是否和 vLLM 端口一致。
- FastAPI 服务是否重新启动以加载最新 `.env`。

### 任务完成但没有 refined C

检查任务结果：

```bash
curl http://127.0.0.1:8088/api/v1/decompile/tasks/dec_xxx/result -w '\n'
```

如果 `fallback_used=true`，查看 `inference_errors` 字段。常见原因：

- vLLM 未启动。
- `VLLM_MODEL` 和 vLLM `--served-model-name` 不一致。
- vLLM 端口和 `VLLM_BASE_URL` 不一致。
- 模型显存不足或 dtype 设置错误。

## 8. 停止服务

两个服务分别在各自终端中按 `Ctrl+C` 停止。

停止顺序没有强制要求。重新启动 FastAPI 服务后会重新读取 `service/.env`。
