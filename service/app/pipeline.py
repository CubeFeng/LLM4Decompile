"""End-to-end decompilation pipeline."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from .config import Settings
from .ghidra_runner import GhidraRunner
from .inference_client import VllmInferenceClient
from .postprocessor import Postprocessor
from .preprocessor import Preprocessor
from .storage import TaskPaths
from .task_store import TaskStore


logger = logging.getLogger(__name__)


class DecompilePipeline:
    def __init__(self, settings: Settings, task_store: TaskStore):
        self.settings = settings
        self.task_store = task_store
        self.ghidra_runner = GhidraRunner(settings, task_store)
        self.preprocessor = Preprocessor()
        self.inference_client = VllmInferenceClient(settings)
        self.postprocessor = Postprocessor()

    @staticmethod
    def _remaining_seconds(deadline: float) -> float:
        return deadline - time.monotonic()

    def _ensure_time_left(self, deadline: float, task_timeout_seconds: int) -> None:
        if self._remaining_seconds(deadline) <= 0:
            raise RuntimeError(f"Decompilation timed out after {task_timeout_seconds}s")

    def _ghidra_timeout_for_deadline(self, deadline: float) -> int:
        remaining = int(self._remaining_seconds(deadline))
        if remaining <= 0:
            return 1
        return max(1, min(remaining, self.settings.ghidra_timeout_seconds))

    def run(
        self,
        *,
        task_id: str,
        paths: TaskPaths,
        input_path: Path,
        safe_name: str,
        input_info: dict[str, Any],
        use_llm: bool,
        fallback_to_ghidra_raw: bool,
        task_timeout_seconds: int,
    ) -> None:
        started_at = time.monotonic()
        deadline = started_at + task_timeout_seconds
        raw_code_path: Path | None = None
        functions_path: Path | None = None
        functions = []
        inferences = None
        ghidra_duration_seconds = 0.0
        ghidra_max_cpu = self.settings.ghidra_max_cpu
        ghidra_postscript = str(self.settings.ghidra_postscript)
        ghidra_parallel_enabled = False
        ghidra_function_count: int | None = None
        ghidra_parallel_mode: str | None = None
        ghidra_decompile_ms: int | None = None
        ghidra_merge_ms: int | None = None
        fallback_used = False

        try:
            self._ensure_time_left(deadline, task_timeout_seconds)
            self.task_store.update(
                task_id,
                status="running",
                stage="ghidra_running",
                progress=15,
                message="Running Ghidra headless decompiler",
            )
            ghidra_result = self.ghidra_runner.run(
                input_path,
                safe_name,
                paths,
                ghidra_timeout_seconds=self._ghidra_timeout_for_deadline(deadline),
            )
            raw_code_path = ghidra_result.raw_code_path
            ghidra_duration_seconds = ghidra_result.duration_seconds
            ghidra_max_cpu = ghidra_result.ghidra_max_cpu
            ghidra_postscript = ghidra_result.ghidra_postscript
            ghidra_parallel_enabled = ghidra_result.ghidra_parallel_enabled
            ghidra_function_count = ghidra_result.function_count
            ghidra_parallel_mode = ghidra_result.parallel_mode
            ghidra_decompile_ms = ghidra_result.decompile_ms
            ghidra_merge_ms = ghidra_result.merge_ms
            self._ensure_time_left(deadline, task_timeout_seconds)

            self.task_store.update(
                task_id,
                stage="preprocessing",
                progress=45,
                message="Splitting Ghidra output into functions",
            )
            preprocess_result = self.preprocessor.process(raw_code_path, safe_name, paths)
            functions_path = preprocess_result.functions_path
            functions = preprocess_result.functions
            self._ensure_time_left(deadline, task_timeout_seconds)

            if use_llm:
                self.task_store.update(
                    task_id,
                    stage="llm_running",
                    progress=65,
                    message="Refining functions with vLLM",
                )
                logger.info("task=%s entering LLM refinement stage", task_id)
                inferences = self.inference_client.infer_functions(
                    functions,
                    task_id=task_id,
                    deadline=deadline,
                    task_timeout_seconds=task_timeout_seconds,
                )
                self._ensure_time_left(deadline, task_timeout_seconds)
                if any(item.success for item in inferences):
                    fallback_used = False
                elif fallback_to_ghidra_raw:
                    fallback_used = True
                else:
                    raise RuntimeError("vLLM did not refine any function")
            else:
                fallback_used = True

            self._ensure_time_left(deadline, task_timeout_seconds)
            self.task_store.update(
                task_id,
                stage="postprocessing",
                progress=90,
                message="Writing manifest and archive",
            )
            self.postprocessor.build_outputs(
                paths=paths,
                safe_name=safe_name,
                input_info=input_info,
                raw_code_path=raw_code_path,
                functions_path=functions_path,
                functions=functions,
                inferences=inferences,
                started_at=started_at,
                ghidra_duration_seconds=ghidra_duration_seconds,
                ghidra_max_cpu=ghidra_max_cpu,
                ghidra_postscript=ghidra_postscript,
                ghidra_parallel_enabled=ghidra_parallel_enabled,
                ghidra_function_count=ghidra_function_count,
                ghidra_parallel_mode=ghidra_parallel_mode,
                ghidra_decompile_ms=ghidra_decompile_ms,
                ghidra_merge_ms=ghidra_merge_ms,
                fallback_used=fallback_used,
            )
            self.task_store.update(
                task_id,
                status="completed",
                stage="completed",
                progress=100,
                message="Decompilation completed",
            )
        except Exception as exc:
            if self.task_store.is_cancelled(task_id):
                return

            current = self.task_store.get(task_id)
            if current.get("status") in {"completed", "failed"}:
                return

            is_timeout = "timed out" in str(exc).lower()
            if (
                not is_timeout
                and raw_code_path is not None
                and functions_path is not None
                and self.settings.fallback_to_ghidra_raw
            ):
                try:
                    self.postprocessor.build_outputs(
                        paths=paths,
                        safe_name=safe_name,
                        input_info=input_info,
                        raw_code_path=raw_code_path,
                        functions_path=functions_path,
                        functions=functions,
                        inferences=None,
                        started_at=started_at,
                        ghidra_duration_seconds=ghidra_duration_seconds,
                        ghidra_max_cpu=ghidra_max_cpu,
                        ghidra_postscript=ghidra_postscript,
                        ghidra_parallel_enabled=ghidra_parallel_enabled,
                        ghidra_function_count=ghidra_function_count,
                        ghidra_parallel_mode=ghidra_parallel_mode,
                        ghidra_decompile_ms=ghidra_decompile_ms,
                        ghidra_merge_ms=ghidra_merge_ms,
                        fallback_used=True,
                        error=None,
                    )
                    self.task_store.update(
                        task_id,
                        status="completed",
                        stage="completed",
                        progress=100,
                        message=f"Completed with Ghidra raw fallback: {exc}",
                    )
                    return
                except Exception:
                    pass

            self.task_store.update(
                task_id,
                status="failed",
                stage="failed",
                progress=100,
                message="Decompilation failed",
                error=str(exc),
            )
        finally:
            self.task_store.release(task_id)
