"""Local JSON task store for the demo service."""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any

from .config import Settings
from .schemas import TaskStage, TaskStatus
from .storage import TaskPaths, build_task_paths, read_json, write_json


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TaskStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._runner_lock = threading.Lock()
        self._current_task_id: str | None = None
        self.settings.tasks_dir.mkdir(parents=True, exist_ok=True)

    @property
    def busy(self) -> bool:
        return self._current_task_id is not None

    @property
    def current_task_id(self) -> str | None:
        return self._current_task_id

    def try_acquire(self, task_id: str) -> bool:
        with self._runner_lock:
            if self._current_task_id is not None:
                return False
            self._current_task_id = task_id
            return True

    def release(self, task_id: str) -> None:
        with self._runner_lock:
            if self._current_task_id == task_id:
                self._current_task_id = None

    def paths(self, task_id: str) -> TaskPaths:
        return build_task_paths(self.settings, task_id)

    def create(
        self,
        task_id: str,
        *,
        message: str = "Task queued",
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        created_at = now_iso()
        data: dict[str, Any] = {
            "task_id": task_id,
            "status": "pending",
            "stage": "queued",
            "progress": 0,
            "message": message,
            "created_at": created_at,
            "updated_at": created_at,
            "error": None,
        }
        if extra:
            data.update(extra)
        write_json(self.paths(task_id).status_file, data)
        return data

    def get(self, task_id: str) -> dict[str, Any]:
        return read_json(self.paths(task_id).status_file)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        stage: TaskStage | None = None,
        progress: int | None = None,
        message: str | None = None,
        error: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self.get(task_id)
        if status is not None:
            data["status"] = status
        if stage is not None:
            data["stage"] = stage
        if progress is not None:
            data["progress"] = max(0, min(100, progress))
        if message is not None:
            data["message"] = message
        if error is not None:
            data["error"] = error
        if extra:
            data.update(extra)
        data["updated_at"] = now_iso()
        write_json(self.paths(task_id).status_file, data)
        return data
