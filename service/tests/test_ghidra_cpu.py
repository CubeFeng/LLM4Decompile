"""Tests for Ghidra CPU configuration helpers."""

from pathlib import Path

from app.ghidra_cpu import (
    derive_ghidra_install_dir,
    ghidra_parallel_warning,
    is_ghidra_cpu_configured,
    is_ghidra_parallel_postscript,
)


def test_derive_ghidra_install_dir() -> None:
    analyze_headless = Path("/opt/ghidra/ghidra_11.0.3_PUBLIC/support/analyzeHeadless")
    assert derive_ghidra_install_dir(analyze_headless) == Path(
        "/opt/ghidra/ghidra_11.0.3_PUBLIC"
    )


def test_is_ghidra_cpu_configured_with_override(tmp_path: Path) -> None:
    support_dir = tmp_path / "support"
    support_dir.mkdir()
    launch_properties = support_dir / "launch.properties"
    launch_properties.write_text(
        "VMARGS=-Xmx2G\nVMARGS=-Dcpu.core.override=8\n",
        encoding="utf-8",
    )

    assert is_ghidra_cpu_configured(tmp_path, 8) is True
    assert is_ghidra_cpu_configured(tmp_path, 15) is False


def test_is_ghidra_cpu_configured_missing_file(tmp_path: Path) -> None:
    assert is_ghidra_cpu_configured(tmp_path, 8) is False


def test_parallel_postscript_detection() -> None:
    parallel = Path("/tmp/ghidra/DecompileParallel.java")
    serial = Path("/tmp/ghidra/decompile.py")
    assert is_ghidra_parallel_postscript(parallel) is True
    assert is_ghidra_parallel_postscript(serial) is False
    assert ghidra_parallel_warning(serial) is not None
    assert ghidra_parallel_warning(parallel) is None
