"""Unit tests verifying exception logging and redaction in longform_render.py SegmentCache.store."""
import logging
import sys
import types
from unittest.mock import MagicMock
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/developer/cache/segment.wav", "/Users/developer"),
        (r"C:\Users\developer\cache\segment.wav", r"C:\Users\developer"),
        ("C:/Users/developer/cache/segment.wav", "C:/Users/developer"),
    ],
)
def test_segment_cache_store_exception_logs_debug_and_scrubs_path(
    tmp_path, monkeypatch, caplog, raw_path, forbidden_prefix
):
    orig_audio_io = sys.modules.get("services.audio_io")
    orig_services = sys.modules.get("services")
    orig_services_audio_io = getattr(orig_services, "audio_io", None) if orig_services else None

    # Patch a temporary services.audio_io module containing only the failing atomic_save_wav
    mock_audio_io = types.ModuleType("services.audio_io")

    def _faulty_save(*args, **kwargs):
        raise RuntimeError(f"disk write failure at {raw_path}")

    mock_audio_io.atomic_save_wav = _faulty_save
    monkeypatch.setitem(sys.modules, "services.audio_io", mock_audio_io)
    if orig_services is not None:
        setattr(orig_services, "audio_io", mock_audio_io)

    try:
        from services import longform_render

        cache = longform_render.SegmentCache(
            str(tmp_path),
            sample_rate=24000,
            engine_id="test_engine",
        )

        span = MagicMock()
        span.text = "Hello world"
        span.voice_id = "speaker_1"
        span.speed = None

        audio = MagicMock()

        with caplog.at_level(logging.DEBUG, logger="omnivoice.longform_render"):
            # Must not raise
            cache.store(span, audio)

        # Verify debug message logged and path redacted
        assert any("Failed to cache longform span" in r.message for r in caplog.records)
        assert forbidden_prefix not in caplog.text
        assert "~" in caplog.text
    finally:
        if orig_audio_io is not None:
            sys.modules["services.audio_io"] = orig_audio_io
        else:
            sys.modules.pop("services.audio_io", None)

        if orig_services is not None:
            if orig_services_audio_io is not None:
                setattr(orig_services, "audio_io", orig_services_audio_io)
            elif hasattr(orig_services, "audio_io"):
                delattr(orig_services, "audio_io")

