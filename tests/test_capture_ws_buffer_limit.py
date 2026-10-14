"""Unit tests for WebSocket audio streaming buffer quota and memory bounds."""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# Gracefully stub third-party framework dependencies if running in hermetic test env
for mod_name in ("fastapi", "starlette", "starlette.requests", "starlette.websockets", "starlette.status"):
    if mod_name not in sys.modules:
        try:
            __import__(mod_name)
        except ModuleNotFoundError:
            m = MagicMock()
            if mod_name == "fastapi":
                class _WebSocketDisconnect(Exception):
                    pass
                m.WebSocketDisconnect = _WebSocketDisconnect
                m.APIRouter = lambda *a, **kw: MagicMock(
                    websocket=lambda *wa, **wkw: lambda fn: fn,
                    get=lambda *ga, **gkw: lambda fn: fn,
                    post=lambda *pa, **pkw: lambda fn: fn,
                )
            sys.modules[mod_name] = m

import pytest


def test_buffer_limit_constant_defaults_and_overrides(monkeypatch):
    import importlib.util

    spec = importlib.util.find_spec("api.routers.capture_ws")
    assert spec is not None and spec.loader is not None

    # Test environment variable override in isolated module
    monkeypatch.setenv("OMNIVOICE_MAX_STREAM_BUFFER_BYTES", "1048576")
    fresh_capture_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fresh_capture_ws)
    assert fresh_capture_ws.MAX_STREAM_BUFFER_BYTES == 1048576

    # Test default fallback without environment variable
    monkeypatch.delenv("OMNIVOICE_MAX_STREAM_BUFFER_BYTES", raising=False)
    default_capture_ws = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(default_capture_ws)
    assert default_capture_ws.MAX_STREAM_BUFFER_BYTES == 25 * 1024 * 1024


def test_ws_transcribe_enforces_buffer_limit(monkeypatch):
    from api.routers import capture_ws

    async def _run():
        monkeypatch.setattr(capture_ws, "MAX_STREAM_BUFFER_BYTES", 500)

        mock_ws = MagicMock()
        mock_ws.url.path = "/ws/transcribe"
        mock_ws.headers = {}
        mock_ws.client.host = "127.0.0.1"
        mock_ws.query_params = {}
        mock_ws.accept = AsyncMock()
        mock_ws.close = AsyncMock()

        sent_messages = []

        async def _mock_send_json(payload):
            sent_messages.append(payload)

        mock_ws.send_json = AsyncMock(side_effect=_mock_send_json)

        messages = [
            {"type": "websocket.receive", "bytes": b"A" * 300},
            {"type": "websocket.receive", "bytes": b"B" * 300},
        ]
        msg_idx = 0

        async def _mock_receive():
            nonlocal msg_idx
            if msg_idx < len(messages):
                msg = messages[msg_idx]
                msg_idx += 1
                return msg
            await asyncio.sleep(10)
            return {"type": "websocket.disconnect"}

        mock_ws.receive = AsyncMock(side_effect=_mock_receive)

        monkeypatch.setattr(capture_ws, "_transcribe_buffer", AsyncMock(return_value=""))
        monkeypatch.setattr(
            capture_ws,
            "_transcribe_buffer_full",
            AsyncMock(return_value={"text": "", "segments": []}),
        )

        await capture_ws.ws_transcribe(mock_ws)

        error_frames = [msg for msg in sent_messages if msg.get("type") == "error"]
        assert len(error_frames) >= 1
        assert "Audio stream exceeded maximum buffer limit" in error_frames[0].get("error", "")
        assert mock_ws.receive.await_count == 2
        assert not any(msg.get("type") == "final" for msg in sent_messages)

    asyncio.run(_run())


def test_sherpa_offline_rejects_oversized_frame(monkeypatch):
    from api.routers import capture_ws

    async def _run():
        monkeypatch.setattr(capture_ws, "MAX_STREAM_BUFFER_BYTES", 500)
        monkeypatch.setattr(capture_ws, "_sherpa_session", AsyncMock(return_value=(16000, None)))
        monkeypatch.setattr(capture_ws, "_sherpa_load_with_status", AsyncMock(return_value=True))

        mock_backend = MagicMock()
        mock_backend.id = "mock_model"
        mock_backend._decode_offline = MagicMock(return_value="decoded text")
        monkeypatch.setattr("services.asr_backend.get_sherpa_dictation_backend", lambda mid: mock_backend)

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()

        # Frame 1: 1000 bytes (exceeds 500 byte limit)
        # Frame 2: EOF
        frames = [
            ("data", b"\x00\x01" * 500),
            ("eof", b""),
        ]
        frame_idx = 0

        async def _mock_recv(ws, aec):
            nonlocal frame_idx
            if frame_idx < len(frames):
                f = frames[frame_idx]
                frame_idx += 1
                return f
            return ("eof", b"")

        monkeypatch.setattr(capture_ws, "_recv_pcm_frame", _mock_recv)

        spec = MagicMock()
        spec.id = "mock_model"

        await capture_ws._run_sherpa_offline(mock_ws, spec)

        mock_ws.close.assert_awaited_once()
        sent_messages = [call.args[0] for call in mock_ws.send_json.await_args_list]
        assert any(msg.get("type") == "error" and "limit exceeded" in msg.get("error", "") for msg in sent_messages)
        # Assert oversized frame is not decoded or included in final transcript
        mock_backend._decode_offline.assert_not_called()
        assert not any(msg.get("final_kind") == "summary" for msg in sent_messages)

    asyncio.run(_run())


def test_sherpa_offline_preserves_inflight_commit_on_eof(monkeypatch):
    from api.routers import capture_ws

    async def _run():
        monkeypatch.setattr(capture_ws, "MAX_STREAM_BUFFER_BYTES", 50000)
        monkeypatch.setattr(capture_ws, "SHERPA_OFFLINE_PARTIAL_S", 0.01)
        monkeypatch.setattr(capture_ws, "SHERPA_OFFLINE_SILENCE_S", 0.01)
        monkeypatch.setattr(capture_ws, "_sherpa_session", AsyncMock(return_value=(16000, None)))
        monkeypatch.setattr(capture_ws, "_sherpa_load_with_status", AsyncMock(return_value=True))

        mock_backend = MagicMock()
        mock_backend.id = "mock_model"
        
        # Simulate decode delay in background thread
        def delayed_decode(samples, sr):
            import time
            time.sleep(0.05)
            return "committed speech"

        mock_backend._decode_offline = MagicMock(side_effect=delayed_decode)
        monkeypatch.setattr("services.asr_backend.get_sherpa_dictation_backend", lambda mid: mock_backend)

        mock_ws = MagicMock()
        mock_ws.send_json = AsyncMock()
        mock_ws.close = AsyncMock()

        # Frame 1: Loud speech audio (3000 bytes)
        # Frame 2: Trailing silence audio (1000 bytes)
        # Frame 3: EOF after short sleep
        loud_pcm = b"\x00\x40" * 1500
        sil_pcm = b"\x00\x00" * 500
        
        frames = [
            ("data", loud_pcm),
            ("data", sil_pcm),
        ]
        frame_idx = 0

        async def _mock_recv(ws, aec):
            nonlocal frame_idx
            if frame_idx < len(frames):
                f = frames[frame_idx]
                frame_idx += 1
                return f
            await asyncio.sleep(0.02)
            return ("eof", b"")

        monkeypatch.setattr(capture_ws, "_recv_pcm_frame", _mock_recv)

        spec = MagicMock()
        spec.id = "mock_model"

        await capture_ws._run_sherpa_offline(mock_ws, spec)

        sent_messages = [call.args[0] for call in mock_ws.send_json.await_args_list]
        summaries = [msg for msg in sent_messages if msg.get("final_kind") == "summary"]
        assert len(summaries) == 1
        assert "committed speech" in summaries[0].get("text", "").lower()

    asyncio.run(_run())


