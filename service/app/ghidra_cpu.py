"""Ghidra CPU configuration helpers."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


_CPU_OVERRIDE_RE = re.compile(r"^\s*VMARGS=-Dcpu\.core\.override=(\d+)\s*$", re.MULTILINE)
_CPU_LIMIT_RE = re.compile(r"^\s*VMARGS=-Dcpu\.core\.limit=(\d+)\s*$", re.MULTILINE)

PARALLEL_POSTSCRIPT = "DecompileParallel.java"
SERIAL_POSTSCRIPT = "decompile.py"
CPU_PROFILE_AGGRESSIVE = "aggressive"
CPU_PROFILE_BALANCED = "balanced"


def logical_cpu_count() -> int:
    """Logical CPU count (includes hyper-threading)."""
    return max(1, os.cpu_count() or 2)


def resolve_ghidra_max_cpu(profile: str) -> int:
    """Resolve Ghidra thread count from CPU profile when GHIDRA_MAX_CPU is unset."""
    normalized = profile.strip().lower()
    if normalized == CPU_PROFILE_AGGRESSIVE:
        return logical_cpu_count()
    return physical_cpu_count()


def derive_ghidra_install_dir(analyze_headless: Path) -> Path:
    """Return Ghidra install root from analyzeHeadless path."""
    return analyze_headless.resolve().parent.parent


def physical_cpu_count() -> int:
    """Best-effort physical core count for Ghidra thread pools."""
    try:
        output = subprocess.check_output(["lscpu"], text=True, stderr=subprocess.DEVNULL)
        cores_per_socket = None
        sockets = None
        for line in output.splitlines():
            if line.strip().startswith("Core(s) per socket:"):
                cores_per_socket = int(line.split(":", 1)[1].strip())
            elif line.strip().startswith("Socket(s):"):
                sockets = int(line.split(":", 1)[1].strip())
        if cores_per_socket is not None and sockets is not None:
            return max(1, cores_per_socket * sockets)
    except (OSError, subprocess.CalledProcessError, ValueError):
        pass

    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="ignore")
        match = re.search(r"^cpu cores\s*:\s*(\d+)\s*$", cpuinfo, re.MULTILINE)
        if match:
            physical_ids = {
                int(value)
                for value in re.findall(r"^physical id\s*:\s*(\d+)\s*$", cpuinfo, re.MULTILINE)
            }
            sockets = len(physical_ids) or 1
            return max(1, int(match.group(1)) * sockets)
    except OSError:
        pass

    logical = os.cpu_count() or 2
    return max(1, logical // 2)


def is_ghidra_parallel_postscript(postscript: Path) -> bool:
    return postscript.name == PARALLEL_POSTSCRIPT and postscript.suffix.lower() == ".java"


def ghidra_parallel_warning(postscript: Path) -> str | None:
    if postscript.name == SERIAL_POSTSCRIPT:
        return (
            "GHIDRA_POSTSCRIPT uses serial decompile.py; "
            "switch to DecompileParallel.java for multi-core decompilation."
        )
    if not is_ghidra_parallel_postscript(postscript):
        return (
            f"GHIDRA_POSTSCRIPT is {postscript.name}; "
            f"expected {PARALLEL_POSTSCRIPT} for parallel decompilation."
        )
    return None


def is_ghidra_cpu_configured(install_dir: Path, expected_cpu: int) -> bool:
    launch_properties = install_dir / "support" / "launch.properties"
    if not launch_properties.is_file():
        return False

    content = launch_properties.read_text(encoding="utf-8", errors="ignore")
    override_match = _CPU_OVERRIDE_RE.search(content)
    if override_match:
        return int(override_match.group(1)) == expected_cpu

    limit_match = _CPU_LIMIT_RE.search(content)
    if limit_match:
        return int(limit_match.group(1)) == expected_cpu

    return False
