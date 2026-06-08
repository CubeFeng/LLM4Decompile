"""Task filesystem helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings


SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class TransientJsonReadError(RuntimeError):
    """Raised when a JSON file could not be read after retries."""


@dataclass(frozen=True)
class TaskPaths:
    task_id: str
    root: Path
    status_file: Path
    manifest_file: Path
    archive_file: Path
    input_dir: Path
    logs_dir: Path
    raw_dir: Path
    functions_dir: Path
    refined_dir: Path
    ghidra_project_dir: Path


def safe_filename(filename: str | None) -> str:
    name = Path(filename or "binary").name
    name = SAFE_NAME_RE.sub("_", name).strip("._")
    return name or "binary"


def build_task_paths(settings: Settings, task_id: str) -> TaskPaths:
    root = settings.tasks_dir / task_id
    return TaskPaths(
        task_id=task_id,
        root=root,
        status_file=root / "status.json",
        manifest_file=root / "manifest.json",
        archive_file=root / "result.zip",
        input_dir=root / "input",
        logs_dir=root / "logs",
        raw_dir=root / "raw",
        functions_dir=root / "functions",
        refined_dir=root / "refined",
        ghidra_project_dir=root / "ghidra_project",
    )


def ensure_task_dirs(paths: TaskPaths) -> None:
    for directory in [
        paths.root,
        paths.input_dir,
        paths.logs_dir,
        paths.raw_dir,
        paths.functions_dir,
        paths.refined_dir,
        paths.ghidra_project_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


def save_bytes(data: bytes, destination: Path, max_bytes: int) -> tuple[int, str]:
    size = len(data)
    if size > max_bytes:
        raise ValueError(f"file exceeds max size: {max_bytes} bytes")
    if size == 0:
        raise ValueError("empty upload body")
    hasher = hashlib.sha256()
    hasher.update(data)
    with destination.open("wb") as out:
        out.write(data)
    return size, hasher.hexdigest()


async def save_upload(file: Any, destination: Path, max_bytes: int) -> tuple[int, str]:
    hasher = hashlib.sha256()
    size = 0
    with destination.open("wb") as out:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                out.close()
                destination.unlink(missing_ok=True)
                raise ValueError(f"file exceeds max size: {max_bytes} bytes")
            hasher.update(chunk)
            out.write(chunk)
    return size, hasher.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)


def read_json(
    path: Path,
    *,
    retries: int = 5,
    retry_delay_seconds: float = 0.05,
) -> dict[str, Any]:
    last_error: json.JSONDecodeError | None = None
    for attempt in range(retries):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(retry_delay_seconds)
                continue
    raise TransientJsonReadError(f"failed to read JSON from {path}") from last_error


def _collect_code_outputs(directory: Path, tag: str) -> list[Path]:
    """Collect files whose stem ends with tag (e.g. _ghidra or _refined)."""
    return sorted(
        file_path
        for file_path in directory.glob("*")
        if file_path.is_file() and file_path.stem.endswith(tag)
    )


def make_archive(paths: TaskPaths, *, fallback_used: bool) -> Path:
    """Pack final code into result.zip.

    Prefer refined outputs; include ghidra raw only when LLM refinement failed.
    """
    paths.archive_file.unlink(missing_ok=True)
    if fallback_used:
        code_files = _collect_code_outputs(paths.raw_dir, "_ghidra")
    else:
        code_files = _collect_code_outputs(paths.refined_dir, "_refined")

    with zipfile.ZipFile(paths.archive_file, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in code_files:
            archive.write(file_path, file_path.name)
    return paths.archive_file


def reset_task_dir(paths: TaskPaths) -> None:
    if paths.root.exists():
        shutil.rmtree(paths.root)
    ensure_task_dirs(paths)
