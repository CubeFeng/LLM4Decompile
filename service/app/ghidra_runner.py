"""Ghidra headless runner."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .storage import TaskPaths


@dataclass(frozen=True)
class GhidraResult:
    raw_code_path: Path
    duration_seconds: float


class GhidraRunner:
    def __init__(self, settings: Settings):
        self.settings = settings

    def run(self, binary_path: Path, safe_name: str, paths: TaskPaths) -> GhidraResult:
        output_path = paths.raw_dir / f"{safe_name}_ghidra.c"
        stdout_path = paths.logs_dir / "ghidra.stdout.log"
        stderr_path = paths.logs_dir / "ghidra.stderr.log"
        project_name = f"{paths.task_id}_ghidra"
        command = [
            str(self.settings.ghidra_analyze_headless),
            str(paths.ghidra_project_dir),
            project_name,
            "-import",
            str(binary_path),
            "-postScript",
            str(self.settings.ghidra_postscript),
            str(output_path),
            "-deleteProject",
            "-max-cpu",
            str(max(1, self.settings.ghidra_max_cpu)),
        ]

        started = time.monotonic()
        with stdout_path.open("w", encoding="utf-8", errors="replace") as stdout_file, stderr_path.open(
            "w", encoding="utf-8", errors="replace"
        ) as stderr_file:
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

        return GhidraResult(
            raw_code_path=output_path,
            duration_seconds=round(time.monotonic() - started, 3),
        )
