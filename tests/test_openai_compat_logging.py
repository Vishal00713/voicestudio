"""Unit tests verifying exception logging and redaction in openai_compat handlers."""
import logging
from unittest.mock import MagicMock, patch
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/db/profiles.sqlite", "/Users/audio"),
        (r"C:\Users\audio\db\profiles.sqlite", r"C:\Users\audio"),
        ("C:/Users/audio/db/profiles.sqlite", "C:/Users/audio"),
    ],
)
def test_resolve_voice_profile_exception_logs_warning_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import openai_compat

    with patch("core.db.db_conn", side_effect=RuntimeError(f"db access error at {raw_path}")), \
         caplog.at_level(logging.WARNING, logger="omnivoice.openai_compat"):
        profile = openai_compat._resolve_voice_profile("custom-voice")

    assert profile is None
    assert any(
        "Voice profiles could not be read; treating voice as an engine preset" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/services/tts_backend.py", "/Users/audio"),
        (r"C:\Users\audio\services\tts_backend.py", r"C:\Users\audio"),
        ("C:/Users/audio/services/tts_backend.py", "C:/Users/audio"),
    ],
)
def test_known_engine_ids_exception_logs_debug_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import openai_compat

    with patch("services.tts_backend.list_backends", side_effect=RuntimeError(f"backend error at {raw_path}")), \
         caplog.at_level(logging.DEBUG, logger="omnivoice.openai_compat"):
        engines = openai_compat._known_engine_ids()

    assert "omnivoice" in engines
    assert any(
        "Failed to list TTS backends, falling back to defaults" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/services/asr_backend.py", "/Users/audio"),
        (r"C:\Users\audio\services\asr_backend.py", r"C:\Users\audio"),
        ("C:/Users/audio/services/asr_backend.py", "C:/Users/audio"),
    ],
)
def test_is_asr_engine_id_exception_logs_debug_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import openai_compat

    mock_reg = MagicMock()
    mock_reg.__contains__.side_effect = RuntimeError(f"registry error at {raw_path}")

    with patch("services.asr_backend._REGISTRY", mock_reg), \
         caplog.at_level(logging.DEBUG, logger="omnivoice.openai_compat"):
        res = openai_compat._is_asr_engine_id("test-model")

    assert res is False
    assert any(
        "Failed to query ASR engine registry for test-model" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/db/voices.sqlite", "/Users/audio"),
        (r"C:\Users\audio\db\voices.sqlite", r"C:\Users\audio"),
        ("C:/Users/audio/db/voices.sqlite", "C:/Users/audio"),
    ],
)
def test_list_voices_db_exception_logs_warning_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import openai_compat

    with patch("core.db.db_conn", side_effect=RuntimeError(f"db lock failure at {raw_path}")), \
         caplog.at_level(logging.WARNING, logger="omnivoice.openai_compat"):
        result = openai_compat.list_voices()

    assert isinstance(result, dict)
    assert "voices" in result
    assert any(
        "Voice profiles could not be loaded; returning built-in aliases only" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text
