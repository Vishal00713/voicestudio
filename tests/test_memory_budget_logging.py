"""Unit tests verifying exception logging in memory_budget.py and tts_stream.py."""
import logging
from unittest.mock import MagicMock, patch
import pytest


def test_available_memory_logs_psutil_exception(caplog, monkeypatch):
    import sys
    from services import memory_budget

    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.side_effect = RuntimeError("psutil failed to read /proc/meminfo")
    monkeypatch.setitem(sys.modules, "psutil", mock_psutil)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.memory_budget"):
        result = memory_budget.available_memory()
        assert "ram_available_gb" not in result
        assert any(
            "System RAM query failed" in record.message
            and "psutil failed to read /proc/meminfo" in record.message
            for record in caplog.records
        )


def test_available_memory_logs_torch_cuda_exception(caplog, monkeypatch):
    import sys
    from services import memory_budget

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.mem_get_info.side_effect = RuntimeError("cuda driver query failed")
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.memory_budget"):
        result = memory_budget.available_memory()
        assert "vram_free_gb" not in result
        assert any(
            "GPU VRAM query failed" in record.message
            and "cuda driver query failed" in record.message
            for record in caplog.records
        )


def test_available_memory_returns_values_on_success(monkeypatch):
    import sys
    from services import memory_budget

    mock_psutil = MagicMock()
    mock_vm = MagicMock()
    mock_vm.available = 8 * (1024 ** 3)
    mock_vm.total = 16 * (1024 ** 3)
    mock_psutil.virtual_memory.return_value = mock_vm
    monkeypatch.setitem(sys.modules, "psutil", mock_psutil)

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.mem_get_info.return_value = (4 * (1024 ** 3), 8 * (1024 ** 3))
    monkeypatch.setitem(sys.modules, "torch", mock_torch)

    result = memory_budget.available_memory()
    assert result["ram_available_gb"] == 8.0
    assert result["ram_total_gb"] == 16.0
    assert result["vram_free_gb"] == 4.0
    assert result["vram_total_gb"] == 8.0


def test_available_memory_logs_redacts_sensitive_paths(caplog, monkeypatch):
    import sys
    from services import memory_budget

    mock_psutil = MagicMock()
    mock_psutil.virtual_memory.side_effect = RuntimeError("failed at /Users/developer/secret.txt")
    monkeypatch.setitem(sys.modules, "psutil", mock_psutil)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.memory_budget"):
        result = memory_budget.available_memory()
        assert "ram_available_gb" not in result
        assert any(
            "System RAM query failed" in record.message
            and "/Users/developer" not in record.message
            for record in caplog.records
        )

