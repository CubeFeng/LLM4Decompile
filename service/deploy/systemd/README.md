# systemd 部署补充说明

主文档见 [`../README.md`](../README.md)。本节仅补充 Ubuntu systemd 与 WSL 差异。

## 命令（与主文档相同）

```bash
./service/scripts/service_ctl.sh setup    # Ubuntu：安装 unit + enable + 启动
./service/scripts/service_ctl.sh install  # 仅刷新 unit 文件
./service/scripts/service_ctl.sh uninstall  # 需 sudo，删除 unit
```

## 日志

| 模式 | 位置 |
|------|------|
| systemd | `service_ctl.sh logs` 或 `journalctl -u llm4decompile-vllm -u llm4decompile-api -f` |
| process（WSL 默认） | `service/logs/vllm.log`、`service/logs/service.log` |

## WSL

`auto` 默认 **process 模式**。强制 systemd：`LLM4DECOMPILE_PREFER_SYSTEMD=1`。

## systemd 单元

| 单元 | 说明 |
|------|------|
| `llm4decompile.target` | 统一启停 |
| `llm4decompile-vllm.service` | vLLM |
| `llm4decompile-api.service` | FastAPI |

模板：`*.service.in`、`llm4decompile.target.in`。模型与 Ghidra 路径仅在 `service/.env` 配置。

## 修改配置后

```bash
./service/scripts/service_ctl.sh restart
```

仓库路径或 Python 路径变更时需重新 `setup` 或 `install`。
