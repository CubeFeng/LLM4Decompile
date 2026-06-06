"""API schemas for the demo service."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["pending", "running", "completed", "failed"]
TaskStage = Literal[
    "queued",
    "saving_input",
    "ghidra_running",
    "preprocessing",
    "llm_running",
    "postprocessing",
    "completed",
    "failed",
]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ghidra_available: bool
    vllm_available: bool
    data_dir_writable: bool
    model: str
    busy: bool
    details: dict[str, Any] = Field(default_factory=dict)


class TaskCreateResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: TaskStage


class TaskStatusResponse(BaseModel):
    task_id: str
    status: TaskStatus
    stage: TaskStage
    progress: int
    message: str
    created_at: str
    updated_at: str
    error: str | None = None


class TaskResultResponse(BaseModel):
    task_id: str
    status: TaskStatus
    input: dict[str, Any]
    outputs: dict[str, Any]
    stats: dict[str, Any]
    fallback_used: bool
    inference_errors: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
