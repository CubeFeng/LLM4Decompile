"""Ghidra headless runner."""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .ghidra_cpu import (
    ghidra_parallel_warning,
    is_ghidra_parallel_postscript,
    logical_cpu_count,
    physical_cpu_count,
)
from .storage import TaskPaths


_MODE_RE = re.compile(r"DECOMPILE_PARALLEL mode=(\S+)")
_FUNCTIONS_RE = re.compile(r"DECOMPILE_PARALLEL mode=\S+ functions=(\d+)")
_DECOMPILE_MS_RE = re.compile(r"DECOMPILE_PARALLEL timing_ms decompile=(\d+)")
_MERGE_MS_RE = re.compile(r"DECOMPILE_PARALLEL timing_ms decompile=\d+ merge=(\d+)")


@dataclass(frozen=True)
class GhidraResult:
    raw_code_path: Path
    duration_seconds: float
    ghidra_max_cpu: int
    ghidra_postscript: str
    ghidra_parallel_enabled: bool
    function_count: int | None = None
    parallel_mode: str | None = None
    decompile_ms: int | None = None
    merge_ms: int | None = None


class GhidraRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, binary_path: Path, safe_name: str, paths: TaskPaths) -> GhidraResult:
        output_path = paths.raw_dir / f"{safe_name}_ghidra.c"
        stdout_path = paths.logs_dir / "ghidra.stdout.log"
        stderr_path = paths.logs_dir / "ghidra.stderr.log"
        project_name = f"{paths.task_id}_ghidra"
        postscript_name = self.settings.ghidra_postscript.name
        postscript_args = [str(output_path)]
        if postscript_name.endswith(".java"):
            postscript_args.extend(
                [
                    str(self.settings.ghidra_decomp_chunk_threshold),
                    str(max(1, self.settings.ghidra_max_cpu)),
                    str(max(1, self.settings.ghidra_decomp_single_queue_limit)),
                ]
            )

        command = [
            str(self.settings.ghidra_analyze_headless),
            str(paths.ghidra_project_dir),
            project_name,
            "-import",
            str(binary_path),
            "-scriptPath",
            str(self.settings.ghidra_script_path),
            "-postScript",
            postscript_name,
            *postscript_args,
            "-deleteProject",
            "-max-cpu",
            str(max(1, self.settings.ghidra_max_cpu)),
        ]

        started = time.monotonic()
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
            stderr_file.write(
                "GhidraRunner: "
                f"postscript={postscript_name} "
                f"max_cpu={self.settings.ghidra_max_cpu} "
                f"cpu_profile={self.settings.ghidra_cpu_profile} "
                f"logical={logical_cpu_count()} "
                f"physical={physical_cpu_count()} "
                f"chunk_threshold={self.settings.ghidra_decomp_chunk_threshold} "
                f"single_queue_limit={self.settings.ghidra_decomp_single_queue_limit}\n"
            )
            warning = ghidra_parallel_warning(self.settings.ghidra_postscript)
            if warning:
                stderr_file.write(f"WARNING: {warning}\n")
            stderr_file.flush()

            try:
                subprocess.run(
                    command,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    text=True,
                    check=True,
                    timeout=self.settings.ghidra_timeout_seconds,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"Ghidra timed out after {self.settings.ghidra_timeout_seconds}s") from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Ghidra failed with exit code {exc.returncode}") from exc

        if not output_path.exists():
            raise RuntimeError("Ghidra completed but did not produce a raw output file")

        metrics = self._parse_parallel_metrics(stdout_path)
        return GhidraResult(
            raw_code_path=output_path,
            duration_seconds=round(time.monotonic() - started, 3),
            ghidra_max_cpu=self.settings.ghidra_max_cpu,
            ghidra_postscript=str(self.settings.ghidra_postscript),
            ghidra_parallel_enabled=is_ghidra_parallel_postscript(self.settings.ghidra_postscript),
            function_count=metrics.get("function_count"),
            parallel_mode=metrics.get("parallel_mode"),
            decompile_ms=metrics.get("decompile_ms"),
            merge_ms=metrics.get("merge_ms"),
        )

    def _parse_parallel_metrics(self, stdout_path: Path) -> dict[str, int | str | None]:
        try:
            content = stdout_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return {}

        metrics: dict[str, int | str | None] = {}
        mode_match = _MODE_RE.search(content)
        if mode_match:
            metrics["parallel_mode"] = mode_match.group(1)

        functions_match = _FUNCTIONS_RE.search(content)
        if functions_match:
            metrics["function_count"] = int(functions_match.group(1))

        decompile_match = _DECOMPILE_MS_RE.search(content)
        if decompile_match:
            metrics["decompile_ms"] = int(decompile_match.group(1))

        merge_match = _MERGE_MS_RE.search(content)
        if merge_match:
            metrics["merge_ms"] = int(merge_match.group(1))

        return metrics
