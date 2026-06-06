"""Configuration for the LLM4Decompile demo service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _find_default_ghidra(root_dir: Path) -> str:
    candidates = [
        root_dir / "ghidra" / "ghidra_11.0.3_PUBLIC" / "support" / "analyzeHeadless",
        root_dir / "ghidra" / "ghidra_11.1.2_PUBLIC" / "support" / "analyzeHeadless",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    service_host: str
    service_port: int
    service_data_dir: Path
    ghidra_analyze_headless: Path
    ghidra_postscript: Path
    ghidra_timeout_seconds: int
    ghidra_max_cpu: int
    max_binary_size_bytes: int
    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str | None
    vllm_timeout_seconds: int
    vllm_max_tokens: int
    fallback_to_ghidra_raw: bool

    @property
    def tasks_dir(self) -> Path:
        return self.service_data_dir / "tasks"


def get_settings() -> Settings:
    repo_root = Path(__file__).resolve().parents[2]
    _load_env_file(repo_root / "service" / ".env")
    default_data_dir = repo_root / "service_data"
    default_postscript = repo_root / "ghidra" / "decompile.py"

    return Settings(
        repo_root=repo_root,
        service_host=os.getenv("SERVICE_HOST", "0.0.0.0"),
        service_port=_int_env("SERVICE_PORT", 8088),
        service_data_dir=Path(os.getenv("SERVICE_DATA_DIR", str(default_data_dir))).resolve(),
        ghidra_analyze_headless=Path(
            os.getenv("GHIDRA_ANALYZE_HEADLESS", _find_default_ghidra(repo_root))
        ).resolve(),
        ghidra_postscript=Path(os.getenv("GHIDRA_POSTSCRIPT", str(default_postscript))).resolve(),
        ghidra_timeout_seconds=_int_env("GHIDRA_TIMEOUT_SECONDS", 600),
        ghidra_max_cpu=_int_env("GHIDRA_MAX_CPU", max(1, (os.cpu_count() or 2) // 2)),
        max_binary_size_bytes=_int_env("MAX_BINARY_SIZE_BYTES", 50 * 1024 * 1024),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
        vllm_model=os.getenv("VLLM_MODEL", "llm4decompile"),
        vllm_api_key=os.getenv("VLLM_API_KEY") or None,
        vllm_timeout_seconds=_int_env("VLLM_TIMEOUT_SECONDS", 300),
        vllm_max_tokens=_int_env("VLLM_MAX_TOKENS", 4048),
        fallback_to_ghidra_raw=_bool_env("FALLBACK_TO_GHIDRA_RAW", True),
    )


settings = get_settings()
