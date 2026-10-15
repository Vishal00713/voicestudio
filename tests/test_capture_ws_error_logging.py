"""Unit tests verifying exception logging and path redaction in capture_ws error/send handlers."""
import asyncio
import logging
from unittest.mock import AsyncMock, patch
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/ws/stream.sock", "/Users/audio"),
        (r"C:\Users\audio\ws\stream.sock", r"C:\Users\audio"),
        ("C:/Users/audio/ws/stream.sock", "C:/Users/audio"),
    ],
)
@pytest.mark.parametrize("fail_on", ["send", "close"])
def test_capture_ws_asr_missing_preflight_logs_debug_and_scrubs_path(
    monkeypatch, caplog, raw_path, forbidden_prefix, fail_on
):
    from api.routers import capture_ws

    ws = AsyncMock()
    ws.url.path = "/ws/transcribe"
    ws.headers = {}
    ws.client.host = "127.0.0.1"
    ws.query_params = {"model": "test-model"}

    err = RuntimeError(f"preflight {fail_on} failed at {raw_path}")
    if fail_on == "send":
        ws.send_json.side_effect = err
    else:
        ws.close.side_effect = err

    monkeypatch.setattr(
        "services.asr_backend.asr_model_missing_error",
        lambda purpose, sherpa_model_id: {"kind": "asr_model_missing", "repo_id": "test/model"},
    )
    monkeypatch.setattr(
        "services.asr_backend.asr_model_missing_detail",
        lambda err: "Missing ASR model",
    )

    with caplog.at_level(logging.DEBUG, logger="omnivoice.capture_ws"):
        asyncio.run(capture_ws.ws_transcribe(ws))

    assert any(
        "Failed to send asr_model_missing error frame" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/ws/licence.sock", "/Users/audio"),
        (r"C:\Users\audio\ws\licence.sock", r"C:\Users\audio"),
        ("C:/Users/audio/ws/licence.sock", "C:/Users/audio"),
    ],
)
@pytest.mark.parametrize("fail_on", ["send", "close"])
def test_capture_ws_licence_preflight_logs_debug_and_scrubs_path(
    monkeypatch, caplog, raw_path, forbidden_prefix, fail_on
):
    from api.routers import capture_ws

    ws = AsyncMock()
    ws.url.path = "/ws/transcribe"
    ws.headers = {}
    ws.client.host = "127.0.0.1"
    ws.query_params = {}

    err = RuntimeError(f"preflight {fail_on} failed at {raw_path}")
    if fail_on == "send":
        ws.send_json.side_effect = err
    else:
        ws.close.side_effect = err

    monkeypatch.setattr(
        "services.asr_backend.asr_model_missing_error",
        lambda purpose, sherpa_model_id: None,
    )
    monkeypatch.setattr(
        capture_ws,
        "_dictation_licence_error",
        lambda spec: {"code": "model_licence_required", "repo_id": "gated/model"},
    )

    with caplog.at_level(logging.DEBUG, logger="omnivoice.capture_ws"):
        asyncio.run(capture_ws.ws_transcribe(ws))

    assert any(
        "Failed to send licence error frame" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/ws/stream.sock", "/Users/audio"),
        (r"C:\Users\audio\ws\stream.sock", r"C:\Users\audio"),
        ("C:/Users/audio/ws/stream.sock", "C:/Users/audio"),
    ],
)
def test_capture_ws_safe_send_logs_debug_and_returns_false_via_handler(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    from api.routers import capture_ws

    ws = AsyncMock()
    ws.url.path = "/ws/transcribe"
    ws.headers = {}
    ws.client.host = "127.0.0.1"
    ws.query_params = {}
    ws.receive.return_value = {"type": "websocket.receive", "text": "EOF"}
    ws.send_json.side_effect = RuntimeError(f"client connection reset at {raw_path}")

    monkeypatch.setattr(
        "services.asr_backend.asr_model_missing_error",
        lambda purpose, sherpa_model_id: None,
    )
    monkeypatch.setattr(
        capture_ws,
        "_dictation_licence_error",
        lambda spec: None,
    )
    monkeypatch.setattr(
        capture_ws,
        "_select_sherpa_spec",
        lambda websocket: None,
    )

    with caplog.at_level(logging.DEBUG, logger="omnivoice.capture_ws"):
        asyncio.run(capture_ws.ws_transcribe(ws))

    assert any(
        "WebSocket send failed (client disconnected)" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/ws/sherpa.onnx", "/Users/audio"),
        (r"C:\Users\audio\ws\sherpa.onnx", r"C:\Users\audio"),
        ("C:/Users/audio/ws/sherpa.onnx", "C:/Users/audio"),
    ],
)
def test_capture_ws_sherpa_probe_exception_logs_debug_and_falls_back(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    import sys
    import types
    from api.routers import capture_ws

    mock_sherpa = types.ModuleType("services.sherpa_dictation")

    def _faulty_is_sherpa(model):
        raise RuntimeError(f"failed to read sherpa model spec at {raw_path}")

    mock_sherpa.is_sherpa_model = _faulty_is_sherpa
    monkeypatch.setitem(sys.modules, "services.sherpa_dictation", mock_sherpa)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.capture_ws"):
        # Without pcm or aec, if sherpa probe fails, returns None
        rate = capture_ws._requested_pcm_sample_rate({"model": f"model-at-{raw_path}"})

    assert rate is None
    assert any(
        "Sherpa dictation probe failed for model=" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text

