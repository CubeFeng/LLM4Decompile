"""Configuration for the LLM4Decompile demo service."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .ghidra_cpu import (
    CPU_PROFILE_AGGRESSIVE,
    derive_ghidra_install_dir,
    logical_cpu_count,
    physical_cpu_count,
    resolve_ghidra_max_cpu,
)


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


DEFAULT_MAX_BINARY_SIZE_BYTES = 500 * 1024 * 1024


def _resolve_max_binary_size_bytes() -> int:
    for name in ("MAX_BINARY_SIZE_BYTES", "DECOMPILE_MAX_BINARY_SIZE_BYTES"):
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return int(value)
    return DEFAULT_MAX_BINARY_SIZE_BYTES


def _find_default_ghidra(root_dir: Path) -> str:
    candidates = [
        root_dir / "ghidra" / "ghidra_12.1.2_PUBLIC" / "support" / "analyzeHeadless",
        root_dir / "ghidra" / "ghidra_11.1.2_PUBLIC" / "support" / "analyzeHeadless",
        root_dir / "ghidra" / "ghidra_11.0.3_PUBLIC" / "support" / "analyzeHeadless",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(candidates[0])


def _resolve_repo_path(repo_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (repo_root / path).resolve()


def _resolve_ghidra_analyze_headless(repo_root: Path) -> Path:
    env_value = os.getenv("GHIDRA_ANALYZE_HEADLESS")
    if env_value:
        candidate = _resolve_repo_path(repo_root, env_value)
        if candidate.exists():
            return candidate
    for candidate in (
        repo_root / "ghidra" / "ghidra_12.1.2_PUBLIC" / "support" / "analyzeHeadless",
        repo_root / "ghidra" / "ghidra_11.1.2_PUBLIC" / "support" / "analyzeHeadless",
        repo_root / "ghidra" / "ghidra_11.0.3_PUBLIC" / "support" / "analyzeHeadless",
    ):
        if candidate.exists():
            return candidate.resolve()
    if env_value:
        return _resolve_repo_path(repo_root, env_value)
    return _resolve_repo_path(repo_root, _find_default_ghidra(repo_root))


@dataclass(frozen=True)
class Settings:
    repo_root: Path
    service_host: str
    service_port: int
    service_data_dir: Path
    ghidra_analyze_headless: Path
    ghidra_postscript: Path
    ghidra_script_path: Path
    ghidra_install_dir: Path
    ghidra_timeout_seconds: int
    ghidra_max_cpu: int
    ghidra_cpu_profile: str
    ghidra_logical_cpus: int
    ghidra_physical_cpus: int
    ghidra_maxmem: str | None
    ghidra_decomp_chunk_threshold: int
    ghidra_decomp_single_queue_limit: int
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
    default_postscript = repo_root / "ghidra" / "postscripts" / "DecompileParallel.java"
    default_script_path = repo_root / "ghidra" / "postscripts"
    ghidra_analyze_headless = _resolve_ghidra_analyze_headless(repo_root)
    ghidra_install_dir_env = os.getenv("GHIDRA_INSTALL_DIR")
    if ghidra_install_dir_env:
        ghidra_install_dir = _resolve_repo_path(repo_root, ghidra_install_dir_env)
    else:
        ghidra_install_dir = derive_ghidra_install_dir(ghidra_analyze_headless)
    ghidra_cpu_profile = os.getenv("GHIDRA_CPU_PROFILE", CPU_PROFILE_AGGRESSIVE).strip().lower()
    ghidra_max_cpu_env = os.getenv("GHIDRA_MAX_CPU")
    if ghidra_max_cpu_env is not None and ghidra_max_cpu_env.strip() != "":
        ghidra_max_cpu = int(ghidra_max_cpu_env)
    else:
        ghidra_max_cpu = resolve_ghidra_max_cpu(ghidra_cpu_profile)
    ghidra_maxmem_env = os.getenv("GHIDRA_MAXMEM")
    ghidra_maxmem = ghidra_maxmem_env.strip() if ghidra_maxmem_env and ghidra_maxmem_env.strip() else None

    postscript_env = os.getenv("GHIDRA_POSTSCRIPT")
    ghidra_postscript = (
        _resolve_repo_path(repo_root, postscript_env)
        if postscript_env
        else default_postscript.resolve()
    )
    script_path_env = os.getenv("GHIDRA_SCRIPT_PATH")
    ghidra_script_path = (
        _resolve_repo_path(repo_root, script_path_env)
        if script_path_env
        else default_script_path.resolve()
    )

    return Settings(
        repo_root=repo_root,
        service_host=os.getenv("SERVICE_HOST", "0.0.0.0"),
        service_port=_int_env("SERVICE_PORT", 8088),
        service_data_dir=_resolve_repo_path(
            repo_root,
            os.getenv("SERVICE_DATA_DIR", str(default_data_dir)),
        ),
        ghidra_analyze_headless=ghidra_analyze_headless,
        ghidra_postscript=ghidra_postscript,
        ghidra_script_path=ghidra_script_path,
        ghidra_install_dir=ghidra_install_dir.resolve(),
        ghidra_timeout_seconds=_int_env("GHIDRA_TIMEOUT_SECONDS", 600),
        ghidra_max_cpu=max(1, ghidra_max_cpu),
        ghidra_cpu_profile=ghidra_cpu_profile,
        ghidra_logical_cpus=logical_cpu_count(),
        ghidra_physical_cpus=physical_cpu_count(),
        ghidra_maxmem=ghidra_maxmem,
        ghidra_decomp_chunk_threshold=_int_env("GHIDRA_DECOMP_CHUNK_THRESHOLD", 500),
        ghidra_decomp_single_queue_limit=_int_env("GHIDRA_DECOMP_SINGLE_QUEUE_LIMIT", 12000),
        max_binary_size_bytes=_resolve_max_binary_size_bytes(),
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1").rstrip("/"),
        vllm_model=os.getenv("VLLM_MODEL", "llm4decompile"),
        vllm_api_key=os.getenv("VLLM_API_KEY") or None,
        vllm_timeout_seconds=_int_env("VLLM_TIMEOUT_SECONDS", 300),
        vllm_max_tokens=_int_env("VLLM_MAX_TOKENS", 4048),
        fallback_to_ghidra_raw=_bool_env("FALLBACK_TO_GHIDRA_RAW", True),
    )


settings = get_settings()
