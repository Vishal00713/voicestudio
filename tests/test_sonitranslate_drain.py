"""Unit tests for SoniTranslate subprocess stdout draining and process termination."""
import asyncio
import collections
import importlib
import io
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _get_sonitranslate():
    return importlib.import_module("services.sonitranslate")


def test_drain_output_reads_lines_into_ring_buffer():
    sonitranslate = _get_sonitranslate()
    lines = [b"Line 1\n", b"Line 2\n", b"Line 3\n"]
    mock_stdout = io.BytesIO(b"".join(lines))
    mock_proc = MagicMock(spec=subprocess.Popen)
    mock_proc.stdout = mock_stdout

    tail = collections.deque(maxlen=2)
    sonitranslate._drain_output(mock_proc, tail)

    assert list(tail) == ["Line 2", "Line 3"]


def test_stop_terminates_and_cleans_up():
    sonitranslate = _get_sonitranslate()

    async def _run():
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.stdout = MagicMock()
        mock_proc.wait = MagicMock(return_value=0)

        sonitranslate._proc = mock_proc
        result = await sonitranslate.stop()

        assert result["stopped"] is True
        assert sonitranslate._proc is None
        mock_proc.terminate.assert_called_once()
        mock_proc.stdout.close.assert_called_once()

    asyncio.run(_run())


def test_start_cleans_up_on_early_exit_with_real_is_running(monkeypatch):
    sonitranslate = _get_sonitranslate()

    async def _run():
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=1)
        home = str(Path.home())
        mock_proc.stdout = io.BytesIO(f"Failed at {home}/models/checkpoint\n".encode())

        monkeypatch.setattr(sonitranslate, "is_installed", lambda: True)
        monkeypatch.setattr(sonitranslate, "is_venv_ready", lambda: False)
        monkeypatch.setattr(sonitranslate.subprocess, "Popen", lambda *args, **kwargs: mock_proc)

        # Stub httpx so test never hits a real server on port 7860
        try:
            import httpx
            monkeypatch.setattr(httpx, "get", lambda *a, **kw: MagicMock(status_code=503))
        except ImportError:
            pass
        mock_httpx = MagicMock()
        mock_httpx.get.return_value = MagicMock(status_code=503)
        monkeypatch.setitem(sys.modules, "httpx", mock_httpx)

        # Do NOT mock is_running(); let real is_running() run, which checks poll() and resets _proc
        with pytest.raises(RuntimeError) as exc_info:
            await sonitranslate.start()

        msg = str(exc_info.value)
        assert "SoniTranslate exited early" in msg
        # Verify home directory path was redacted
        assert home not in msg
        assert "~/models/checkpoint" in msg
        assert sonitranslate._proc is None

    asyncio.run(_run())


def test_stop_interrupts_start(monkeypatch):
    sonitranslate = _get_sonitranslate()

    async def _run():
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.stdout = io.BytesIO(b"Startup line 1\n")
        mock_proc.wait = MagicMock(return_value=0)

        monkeypatch.setattr(sonitranslate, "is_installed", lambda: True)
        monkeypatch.setattr(sonitranslate, "is_venv_ready", lambda: False)
        monkeypatch.setattr(sonitranslate.subprocess, "Popen", lambda *args, **kwargs: mock_proc)
        monkeypatch.setattr(sonitranslate, "is_running", lambda: False)

        start_task = asyncio.create_task(sonitranslate.start())

        # Give start() time to acquire lock, spawn proc, and enter readiness sleep
        await asyncio.sleep(0.1)

        # Call stop() while startup is waiting for readiness
        stop_result = await sonitranslate.stop()
        assert stop_result["stopped"] is True
        assert sonitranslate._proc is None

        with pytest.raises(RuntimeError) as exc_info:
            await start_task

        assert "interrupted by stop request" in str(exc_info.value)

    asyncio.run(_run())


def test_overlapping_starts_do_not_spawn_duplicate_child(monkeypatch):
    sonitranslate = _get_sonitranslate()

    async def _run():
        spawn_count = 0
        mock_proc = MagicMock(spec=subprocess.Popen)
        mock_proc.poll = MagicMock(return_value=None)
        mock_proc.stdout = io.BytesIO(b"Startup line 1\n")
        mock_proc.wait = MagicMock(return_value=0)

        def mock_popen(*args, **kwargs):
            nonlocal spawn_count
            spawn_count += 1
            return mock_proc

        monkeypatch.setattr(sonitranslate, "is_installed", lambda: True)
        monkeypatch.setattr(sonitranslate, "is_venv_ready", lambda: False)
        monkeypatch.setattr(sonitranslate.subprocess, "Popen", mock_popen)

        running_checks = 0
        def mock_is_running():
            nonlocal running_checks
            running_checks += 1
            return running_checks > 2

        monkeypatch.setattr(sonitranslate, "is_running", mock_is_running)

        res1, res2 = await asyncio.gather(sonitranslate.start(), sonitranslate.start())
        assert res1["started"] is True
        assert res2["started"] is True
        assert spawn_count == 1

        await sonitranslate.stop()

    asyncio.run(_run())


def test_stop_cancels_startup_task_without_cancelled_error():
    sonitranslate = _get_sonitranslate()

    async def _run():
        started = asyncio.Event()

        async def slow_start():
            started.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(slow_start())
        sonitranslate._startup_task = task

        await started.wait()
        res = await sonitranslate.stop()
        assert res["stopped"] is True
        assert task.cancelled() or task.done()

    asyncio.run(_run())



