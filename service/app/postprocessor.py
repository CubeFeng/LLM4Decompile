"""Create final outputs and manifest."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .inference_client import FunctionInference
from .preprocessor import FunctionRecord
from .storage import TaskPaths, make_archive, write_json


@dataclass(frozen=True)
class PostprocessResult:
    refined_path: Path
    manifest_path: Path
    archive_path: Path
    manifest: dict[str, Any]


class Postprocessor:
    def build_outputs(
        self,
        *,
        paths: TaskPaths,
        safe_name: str,
        input_info: dict[str, Any],
        raw_code_path: Path,
        functions_path: Path,
        functions: list[FunctionRecord],
        inferences: list[FunctionInference] | None,
        started_at: float,
        ghidra_duration_seconds: float,
        ghidra_max_cpu: int,
        ghidra_postscript: str,
        ghidra_parallel_enabled: bool,
        ghidra_function_count: int | None = None,
        ghidra_parallel_mode: str | None = None,
        ghidra_decompile_ms: int | None = None,
        ghidra_merge_ms: int | None = None,
        fallback_used: bool,
        error: str | None = None,
    ) -> PostprocessResult:
        refined_path = paths.refined_dir / f"{safe_name}_refined.c"
        refined_path.write_text(
            self._render_refined(raw_code_path, functions, inferences, fallback_used),
            encoding="utf-8",
        )

        success_count = sum(1 for item in inferences or [] if item.success)
        failed_count = sum(1 for item in inferences or [] if not item.success)
        inference_errors = [
            {
                "index": item.index,
                "name": item.name,
                "error": item.error,
            }
            for item in inferences or []
            if not item.success and item.error
        ]
        manifest = {
            "task_id": paths.task_id,
            "status": "completed" if error is None else "failed",
            "input": input_info,
            "outputs": {
                "raw_code": str(raw_code_path.relative_to(paths.root)),
                "functions": str(functions_path.relative_to(paths.root)),
                "refined_code": str(refined_path.relative_to(paths.root)),
                "archive": str(paths.archive_file.relative_to(paths.root)),
            },
            "stats": {
                "function_count": len(functions),
                "refined_count": success_count,
                "failed_count": failed_count,
                "ghidra_duration_seconds": ghidra_duration_seconds,
                "ghidra_max_cpu": ghidra_max_cpu,
                "ghidra_postscript": ghidra_postscript,
                "ghidra_parallel_enabled": ghidra_parallel_enabled,
                "ghidra_function_count": ghidra_function_count,
                "ghidra_parallel_mode": ghidra_parallel_mode,
                "ghidra_decompile_ms": ghidra_decompile_ms,
                "ghidra_merge_ms": ghidra_merge_ms,
                "duration_seconds": round(time.monotonic() - started_at, 3),
            },
            "fallback_used": fallback_used,
            "inference_errors": inference_errors,
            "error": error,
        }
        write_json(paths.manifest_file, manifest)
        archive_path = make_archive(paths, fallback_used=fallback_used)
        return PostprocessResult(
            refined_path=refined_path,
            manifest_path=paths.manifest_file,
            archive_path=archive_path,
            manifest=manifest,
        )

    def _render_refined(
        self,
        raw_code_path: Path,
        functions: list[FunctionRecord],
        inferences: list[FunctionInference] | None,
        fallback_used: bool,
    ) -> str:
        if fallback_used or inferences is None:
            return raw_code_path.read_text(encoding="utf-8", errors="ignore")

        by_index = {item.index: item for item in inferences}
        rendered = []
        for function in functions:
            inference = by_index.get(function.index)
            if inference and inference.success:
                rendered.append(f"// Function: {function.name}\n{inference.output.strip()}")
            else:
                reason = inference.error if inference else "No inference result"
                rendered.append(
                    f"// Function: {function.name}\n"
                    f"// vLLM refinement failed: {reason}\n"
                    f"{function.code.strip()}"
                )
        return "\n\n".join(rendered) + "\n"
