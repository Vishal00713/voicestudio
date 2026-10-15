"""Unit tests verifying exception logging and redaction in settings device probe and remote model fetch."""
import logging
from unittest.mock import MagicMock
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/devices/probe.sock", "/Users/audio"),
        (r"C:\Users\audio\devices\probe.sock", r"C:\Users\audio"),
        ("C:/Users/audio/devices/probe.sock", "C:/Users/audio"),
    ],
)
def test_offload_after_generation_device_probe_failure_logs_debug_and_falls_back(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    from api.routers import settings
    from services import model_manager as mm

    def _faulty_get_best_device():
        raise RuntimeError(f"GPU enumeration failed at {raw_path}")

    monkeypatch.setattr(mm, "get_best_device", _faulty_get_best_device)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.api.settings"):
        res = settings.get_offload_after_generation()

    assert res["device"] == "cpu"
    assert any(
        "Failed to probe best device in offload_after_generation_state, falling back to cpu" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/llm/client.sock", "/Users/audio"),
        (r"C:\Users\audio\llm\client.sock", r"C:\Users\audio"),
        ("C:/Users/audio/llm/client.sock", "C:/Users/audio"),
    ],
)
def test_local_models_failure_logs_debug_and_returns_none(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    import sys
    import types
    from api.routers import settings

    mock_openai = types.ModuleType("openai")
    test_token = "sk-test-secret-token-12345"

    class MockClient:
        def __init__(self, **kwargs):
            pass

        @property
        def models(self):
            mock_models = MagicMock()
            mock_models.list.side_effect = RuntimeError(
                f"remote connection reset for key {test_token} at {raw_path}"
            )
            return mock_models

    mock_openai.OpenAI = MockClient
    monkeypatch.setitem(sys.modules, "openai", mock_openai)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.api.settings"):
        res = settings._local_models("http://127.0.0.1:11434/v1", test_token)

    assert res is None
    assert any(
        "Failed to list local models: " in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text
    assert test_token not in caplog.text
