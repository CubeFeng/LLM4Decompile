from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import _resolve_decompile_timeout
from app.config import Settings
from app.main import app
from app.task_store import TaskStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        repo_root=tmp_path,
        service_host="127.0.0.1",
        service_port=8088,
        service_data_dir=tmp_path / "data",
        ghidra_analyze_headless=tmp_path / "analyzeHeadless",
        ghidra_postscript=tmp_path / "decompile.py",
        ghidra_script_path=tmp_path / "postscripts",
        ghidra_install_dir=tmp_path / "ghidra",
        ghidra_timeout_seconds=600,
        ghidra_max_cpu=2,
        ghidra_cpu_profile="aggressive",
        ghidra_logical_cpus=8,
        ghidra_physical_cpus=4,
        ghidra_maxmem=None,
        ghidra_decomp_chunk_threshold=500,
        ghidra_decomp_single_queue_limit=12000,
        decompile_timeout_seconds=1800,
        max_binary_size_bytes=1024 * 1024,
        vllm_base_url="http://127.0.0.1:8001/v1",
        vllm_model="llm4decompile",
        vllm_api_key=None,
        vllm_timeout_seconds=300,
        vllm_max_tokens=4048,
        fallback_to_ghidra_raw=True,
    )


def test_resolve_decompile_timeout_uses_default(settings: Settings):
    with patch("app.api.settings", settings):
        assert _resolve_decompile_timeout(None) == 1800


def test_resolve_decompile_timeout_uses_request_value(settings: Settings):
    with patch("app.api.settings", settings):
        assert _resolve_decompile_timeout(3600) == 3600


def test_resolve_decompile_timeout_rejects_out_of_range(settings: Settings):
    with patch("app.api.settings", settings):
        with pytest.raises(HTTPException) as exc_info:
            _resolve_decompile_timeout(30)
        assert exc_info.value.status_code == 400


def test_task_store_cancel_kills_registered_process(settings: Settings):
    store = TaskStore(settings)
    proc = MagicMock()
    proc.poll.return_value = None
    proc.pid = 4242

    with patch("app.task_store._kill_process_group", return_value=True) as kill_mock:
        store.register_process("dec_test", proc)
        killed = store.cancel("dec_test")

    assert killed is True
    kill_mock.assert_called_once_with(proc)
    assert store.is_cancelled("dec_test")


def test_cancel_task_endpoint_is_idempotent(tmp_path: Path, settings: Settings):
    settings.service_data_dir.mkdir(parents=True, exist_ok=True)
    settings.ghidra_analyze_headless.write_text("", encoding="utf-8")
    settings.ghidra_analyze_headless.chmod(0o755)
    settings.ghidra_postscript.write_text("", encoding="utf-8")

    with patch("app.api.settings", settings), patch("app.api.task_store", TaskStore(settings)):
        from app import api

        store = api.task_store
        client = TestClient(app)
        task_id = "dec_cancel123"
        paths = store.paths(task_id)
        paths.root.mkdir(parents=True, exist_ok=True)
        store.create(task_id)
        store.update(task_id, status="failed", stage="failed", progress=100, error="done")

        response = client.post(f"/api/v1/decompile/tasks/{task_id}/cancel")
        assert response.status_code == 200
        assert response.json()["status"] == "failed"


def test_ghidra_runner_timeout_kills_process(settings: Settings, tmp_path: Path):
    from app.ghidra_runner import GhidraRunner
    from app.storage import build_task_paths

    settings.ghidra_analyze_headless.write_text("#!/bin/sh\nsleep 60\n", encoding="utf-8")
    settings.ghidra_analyze_headless.chmod(0o755)
    settings.ghidra_postscript.write_text("", encoding="utf-8")

    store = TaskStore(settings)
    runner = GhidraRunner(settings, store)
    paths = build_task_paths(settings, "dec_timeout")
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.logs_dir.mkdir(parents=True, exist_ok=True)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.ghidra_project_dir.mkdir(parents=True, exist_ok=True)
    binary_path = paths.input_dir / "sample.bin"
    binary_path.parent.mkdir(parents=True, exist_ok=True)
    binary_path.write_bytes(b"\x7fELF")

    with pytest.raises(RuntimeError, match="Ghidra timed out after 1s"):
        runner.run(
            binary_path,
            "sample",
            paths,
            ghidra_timeout_seconds=1,
        )

    assert store.busy is False


def test_pipeline_raises_on_task_timeout(settings: Settings, tmp_path: Path):
    from app.pipeline import DecompilePipeline
    from app.storage import build_task_paths

    store = TaskStore(settings)
    pipeline = DecompilePipeline(settings, store)
    paths = build_task_paths(settings, "dec_pipeline_timeout")
    paths.root.mkdir(parents=True, exist_ok=True)
    store.create("dec_pipeline_timeout")
    input_path = paths.input_dir / "sample.bin"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_bytes(b"\x7fELF")

    with patch.object(
        pipeline,
        "_ensure_time_left",
        side_effect=RuntimeError("Decompilation timed out after 1s"),
    ):
        pipeline.run(
            task_id="dec_pipeline_timeout",
            paths=paths,
            input_path=input_path,
            safe_name="sample",
            input_info={"filename": "sample.bin"},
            use_llm=False,
            fallback_to_ghidra_raw=True,
            task_timeout_seconds=1,
        )

    status = store.get("dec_pipeline_timeout")
    assert status["status"] == "failed"
    assert "timed out" in status["error"].lower()
    assert store.busy is False
