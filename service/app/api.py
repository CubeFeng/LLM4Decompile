"""HTTP API for the demo decompilation service."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from .config import settings
from .ghidra_cpu import (
    ghidra_parallel_warning,
    is_ghidra_cpu_configured,
    is_ghidra_parallel_postscript,
)
from .inference_client import VllmInferenceClient
from .pipeline import DecompilePipeline
from .schemas import HealthResponse, TaskCreateResponse, TaskResultResponse, TaskStatusResponse
from .storage import (
    TransientJsonReadError,
    read_json,
    reset_task_dir,
    safe_filename,
    save_bytes,
)
from .task_store import TaskStore


router = APIRouter()
task_store = TaskStore(settings)

DECOMPILE_TIMEOUT_MIN_SECONDS = 60
DECOMPILE_TIMEOUT_MAX_SECONDS = 360000


def _task_id() -> str:
    return f"dec_{uuid.uuid4().hex[:12]}"


def _check_ghidra() -> bool:
    return (
        settings.ghidra_analyze_headless.exists()
        and settings.ghidra_analyze_headless.is_file()
        and os.access(settings.ghidra_analyze_headless, os.X_OK)
    )


def _check_postscript() -> bool:
    return settings.ghidra_postscript.exists() and settings.ghidra_postscript.is_file()


def _resolve_decompile_timeout(decompile_timeout_seconds: int | None) -> int:
    effective = (
        decompile_timeout_seconds
        if decompile_timeout_seconds is not None
        else settings.decompile_timeout_seconds
    )
    if effective < DECOMPILE_TIMEOUT_MIN_SECONDS or effective > DECOMPILE_TIMEOUT_MAX_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"decompile_timeout_seconds must be between "
                f"{DECOMPILE_TIMEOUT_MIN_SECONDS} and {DECOMPILE_TIMEOUT_MAX_SECONDS}"
            ),
        )
    return effective


def _check_data_dir() -> bool:
    try:
        settings.tasks_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.tasks_dir / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ghidra_available = _check_ghidra() and _check_postscript()
    data_dir_writable = _check_data_dir()
    vllm_available = VllmInferenceClient(settings).health()
    ok = ghidra_available and vllm_available and data_dir_writable
    ghidra_parallel_enabled = is_ghidra_parallel_postscript(settings.ghidra_postscript)
    return HealthResponse(
        status="ok" if ok else "degraded",
        ghidra_available=ghidra_available,
        vllm_available=vllm_available,
        data_dir_writable=data_dir_writable,
        model=settings.vllm_model,
        busy=task_store.busy,
        details={
            "ghidra_analyze_headless": str(settings.ghidra_analyze_headless),
            "ghidra_postscript": str(settings.ghidra_postscript),
            "ghidra_script_path": str(settings.ghidra_script_path),
            "ghidra_install_dir": str(settings.ghidra_install_dir),
            "ghidra_max_cpu": settings.ghidra_max_cpu,
            "ghidra_cpu_profile": settings.ghidra_cpu_profile,
            "ghidra_logical_cpus": settings.ghidra_logical_cpus,
            "ghidra_physical_cpus": settings.ghidra_physical_cpus,
            "ghidra_maxmem": settings.ghidra_maxmem,
            "ghidra_decomp_chunk_threshold": settings.ghidra_decomp_chunk_threshold,
            "ghidra_decomp_single_queue_limit": settings.ghidra_decomp_single_queue_limit,
            "ghidra_parallel_enabled": ghidra_parallel_enabled,
            "ghidra_parallel_warning": ghidra_parallel_warning(settings.ghidra_postscript),
            "ghidra_cpu_configured": is_ghidra_cpu_configured(
                settings.ghidra_install_dir,
                settings.ghidra_max_cpu,
            ),
            "service_data_dir": str(settings.service_data_dir),
            "current_task_id": task_store.current_task_id,
            "max_binary_size_bytes": settings.max_binary_size_bytes,
            "decompile_timeout_seconds": settings.decompile_timeout_seconds,
        },
    )


@router.post("/api/v1/decompile/tasks", response_model=TaskCreateResponse)
async def create_task(
    request: Request,
    background_tasks: BackgroundTasks,
    use_llm: bool = True,
    fallback_to_ghidra_raw: bool = settings.fallback_to_ghidra_raw,
    decompile_timeout_seconds: int | None = None,
) -> TaskCreateResponse:
    if not _check_ghidra() or not _check_postscript():
        raise HTTPException(status_code=503, detail="Ghidra is not configured")

    task_id = _task_id()
    if not task_store.try_acquire(task_id):
        raise HTTPException(
            status_code=409,
            detail={"message": "Service is busy", "current_task_id": task_store.current_task_id},
        )

    effective_timeout = _resolve_decompile_timeout(decompile_timeout_seconds)

    paths = task_store.paths(task_id)
    original_filename = (
        request.query_params.get("filename")
        or request.headers.get("x-filename")
        or request.headers.get("x-upload-filename")
        or "binary"
    )
    safe_name = safe_filename(original_filename)
    input_path = paths.input_dir / safe_name

    try:
        reset_task_dir(paths)
        task_store.create(task_id, message="Saving uploaded binary")
        task_store.update(task_id, stage="saving_input", progress=5, message="Saving uploaded binary")
        body = await request.body()
        size, sha256 = save_bytes(body, input_path, settings.max_binary_size_bytes)
        input_info = {
            "filename": original_filename,
            "safe_name": safe_name,
            "size": size,
            "sha256": sha256,
        }
        task_store.update(
            task_id,
            status="pending",
            stage="queued",
            progress=10,
            message="Task queued",
            extra={
                "input": input_info,
                "decompile_timeout_seconds": effective_timeout,
            },
        )
    except ValueError as exc:
        task_store.release(task_id)
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:
        task_store.release(task_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    pipeline = DecompilePipeline(settings, task_store)
    background_tasks.add_task(
        pipeline.run,
        task_id=task_id,
        paths=paths,
        input_path=input_path,
        safe_name=safe_name,
        input_info=input_info,
        use_llm=use_llm,
        fallback_to_ghidra_raw=fallback_to_ghidra_raw,
        task_timeout_seconds=effective_timeout,
    )
    status = task_store.get(task_id)
    return TaskCreateResponse(task_id=task_id, status=status["status"], stage=status["stage"])


@router.post("/api/v1/decompile/tasks/{task_id}/cancel", response_model=TaskStatusResponse)
def cancel_task(task_id: str) -> TaskStatusResponse:
    paths = task_store.paths(task_id)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")

    try:
        status = task_store.get(task_id)
    except TransientJsonReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Task status temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc

    if status["status"] in {"completed", "failed"}:
        return TaskStatusResponse(**status)

    task_store.cancel(task_id)
    status = task_store.update(
        task_id,
        status="failed",
        stage="failed",
        progress=100,
        message="Task cancelled",
        error="Task cancelled",
    )
    task_store.release(task_id)
    return TaskStatusResponse(**status)


@router.get("/api/v1/decompile/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str) -> TaskStatusResponse:
    paths = task_store.paths(task_id)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        return TaskStatusResponse(**task_store.get(task_id))
    except TransientJsonReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Task status temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc


@router.get("/api/v1/decompile/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_result(task_id: str) -> TaskResultResponse:
    paths = task_store.paths(task_id)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    try:
        status = task_store.get(task_id)
    except TransientJsonReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Task status temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc
    if status["status"] != "completed":
        raise HTTPException(status_code=409, detail="Task is not completed")
    if not paths.manifest_file.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    try:
        manifest = read_json(paths.manifest_file)
    except TransientJsonReadError as exc:
        raise HTTPException(
            status_code=503,
            detail="Task result temporarily unavailable",
            headers={"Retry-After": "1"},
        ) from exc
    manifest["outputs"] = {
        **manifest.get("outputs", {}),
        "archive_url": f"/api/v1/decompile/tasks/{task_id}/archive",
    }
    return TaskResultResponse(**manifest)


@router.get("/api/v1/decompile/tasks/{task_id}/archive")
def download_archive(task_id: str) -> FileResponse:
    paths = task_store.paths(task_id)
    archive_path = Path(paths.archive_file)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    if not archive_path.exists() or not archive_path.is_file():
        raise HTTPException(status_code=404, detail="Archive not found")
    return FileResponse(
        archive_path,
        media_type="application/zip",
        filename=f"{task_id}_decompile_result.zip",
    )
