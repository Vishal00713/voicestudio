"""Unit tests verifying exception logging and redaction in dub_core.py word extraction."""
from __future__ import annotations

import asyncio
import logging
import struct
import wave
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def _make_wav(path: Path, seconds: float = 1.0, sr: int = 16000) -> None:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *([0] * n)))


def test_words_from_whisper_exception_logs_warning_and_scrubs_paths(
    tmp_path, monkeypatch, caplog, asr_model_installed
):
    import sys
    mock_torch = MagicMock()
    monkeypatch.setitem(sys.modules, "torch", mock_torch)
    monkeypatch.setitem(sys.modules, "torchaudio", MagicMock())
    from api.routers import dub_core as dc

    job_id = "t_words_fail"
    audio = tmp_path / "audio.wav"
    _make_wav(audio, seconds=1.0)
    dc._dub_jobs[job_id] = {
        "audio_path": str(audio),
        "vocals_path": None,
        "scene_cuts": [],
        "duration": 1.0,
    }

    fake_model = MagicMock()
    fake_model._asr_pipe = MagicMock()

    async def _ok_model():
        return fake_model

    class _FakeASR:
        id = "fake"
        def ensure_loaded(self):
            pass
        def transcribe(self, path, *, word_timestamps=True):
            return {
                "chunks": [{"text": "hello world", "timestamp": (0.0, 1.0)}],
                "segments": [{"text": "hello world", "start": 0.0, "end": 1.0}],
                "language": "en",
            }
        def unload(self):
            pass

    monkeypatch.setattr(dc, "get_model", _ok_model)
    monkeypatch.setattr(
        "services.asr_backend.get_active_asr_backend",
        lambda *a, **k: _FakeASR(),
    )
    monkeypatch.setattr(dc, "offload_tts_for_asr", lambda *a, **k: None)
    monkeypatch.setattr(dc, "restore_tts_after_asr", lambda *a, **k: None)
    monkeypatch.setattr(
        dc,
        "get_diarization_pipeline",
        lambda *a, **k: (None, None) if k.get("return_error") else None,
    )

    # Inject exception with private user path into _words_from_whisper
    def _faulty_words_from_whisper(part):
        raise RuntimeError("word alignment crashed at /Users/developer/secret_audio.wav")

    monkeypatch.setattr(dc, "_words_from_whisper", _faulty_words_from_whisper)

    async def _collect():
        resp = await dc.dub_transcribe_stream(job_id)
        parts = []
        async for chunk in resp.body_iterator:
            parts.append(chunk.decode() if isinstance(chunk, (bytes, bytearray)) else str(chunk))
        return "".join(parts)

    with caplog.at_level(logging.WARNING):
        try:
            body = asyncio.run(_collect())
        finally:
            dc._dub_jobs.pop(job_id, None)

    # Verify warning was logged
    assert any("word extraction skipped for chunk 0" in r.message for r in caplog.records)
    # Verify sensitive path was redacted
    assert "/Users/developer" not in caplog.text
    # Verify stream completed cleanly with final event and no error
    assert "event: segments" in body
    assert "event: final" in body
    assert "event: error" not in body
