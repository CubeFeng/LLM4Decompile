"""HTTP API for the demo decompilation service."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse

from .config import settings
from .inference_client import VllmInferenceClient
from .pipeline import DecompilePipeline
from .schemas import HealthResponse, TaskCreateResponse, TaskResultResponse, TaskStatusResponse
from .storage import (
    read_json,
    reset_task_dir,
    safe_filename,
    save_bytes,
)
from .task_store import TaskStore


router = APIRouter()
task_store = TaskStore(settings)


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
            "service_data_dir": str(settings.service_data_dir),
            "current_task_id": task_store.current_task_id,
        },
    )


@router.post("/api/v1/decompile/tasks", response_model=TaskCreateResponse)
async def create_task(
    request: Request,
    background_tasks: BackgroundTasks,
    use_llm: bool = True,
    fallback_to_ghidra_raw: bool = settings.fallback_to_ghidra_raw,
) -> TaskCreateResponse:
    if not _check_ghidra() or not _check_postscript():
        raise HTTPException(status_code=503, detail="Ghidra is not configured")

    task_id = _task_id()
    if not task_store.try_acquire(task_id):
        raise HTTPException(
            status_code=409,
            detail={"message": "Service is busy", "current_task_id": task_store.current_task_id},
        )

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
            extra={"input": input_info},
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
    )
    status = task_store.get(task_id)
    return TaskCreateResponse(task_id=task_id, status=status["status"], stage=status["stage"])


@router.get("/api/v1/decompile/tasks/{task_id}", response_model=TaskStatusResponse)
def get_task(task_id: str) -> TaskStatusResponse:
    paths = task_store.paths(task_id)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    return TaskStatusResponse(**task_store.get(task_id))


@router.get("/api/v1/decompile/tasks/{task_id}/result", response_model=TaskResultResponse)
def get_result(task_id: str) -> TaskResultResponse:
    paths = task_store.paths(task_id)
    if not paths.status_file.exists():
        raise HTTPException(status_code=404, detail="Task not found")
    status = task_store.get(task_id)
    if status["status"] != "completed":
        raise HTTPException(status_code=409, detail="Task is not completed")
    if not paths.manifest_file.exists():
        raise HTTPException(status_code=404, detail="Manifest not found")
    manifest = read_json(paths.manifest_file)
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
