"""Unit tests for exception logging in model_lifecycle.py."""
import asyncio
import logging
import sys
from unittest.mock import MagicMock, patch

import pytest


def test_tts_vram_mb_logs_exception_and_returns_zero(caplog):
    import services.model_lifecycle as ml
    import services.model_manager as mm

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = RuntimeError("CUDA driver crashed")

    with patch.object(mm, "_lazy_torch", return_value=mock_torch):
        with caplog.at_level(logging.DEBUG, logger="omnivoice.model_lifecycle"):
            result = ml._tts_vram_mb()
            assert result == 0.0
            assert any(
                "Failed to query in-process TTS VRAM" in record.message
                and "CUDA driver crashed" in record.message
                for record in caplog.records
            )


def test_active_tts_id_logs_exception_and_returns_none(caplog):
    import services.model_lifecycle as ml

    with patch.dict(
        sys.modules,
        {"services.tts_backend": MagicMock(active_backend_id=MagicMock(side_effect=Exception("tts backend init error")))},
    ):
        with caplog.at_level(logging.DEBUG, logger="omnivoice.model_lifecycle"):
            result = ml._active_tts_id()
            assert result is None
            assert any(
                "Failed to resolve active TTS engine id" in record.message
                and "tts backend init error" in record.message
                for record in caplog.records
            )


def test_list_loaded_handles_model_parameters_exception(caplog, monkeypatch):
    import services.model_lifecycle as ml
    import services.model_manager as mm

    class ExplodingModel:
        def parameters(self):
            raise RuntimeError("Corrupted weights tensor")

    monkeypatch.setattr(mm, "model", ExplodingModel())
    monkeypatch.setattr(mm, "_diar_pipeline", None)
    monkeypatch.setattr(mm, "resolve_omnivoice_checkpoint", lambda: "test-ckpt")
    monkeypatch.setattr(ml, "get_best_device", lambda: "cpu")
    monkeypatch.setattr(ml, "_tts_vram_mb", lambda: 0.0)
    monkeypatch.setattr(ml, "_active_tts_id", lambda: None)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.model_lifecycle"):
        result = ml.list_loaded()
        assert any(entry["id"] == "tts" and entry["device"] == "cpu" for entry in result["models"])
        assert any(
            "Failed to determine TTS model device from parameters" in record.message
            and "Corrupted weights tensor" in record.message
            for record in caplog.records
        )


def test_list_loaded_handles_memory_budget_exception(caplog, monkeypatch):
    import services.model_lifecycle as ml
    import services.model_manager as mm
    import services.memory_budget as mb

    monkeypatch.setattr(mm, "model", None)
    monkeypatch.setattr(mm, "_diar_pipeline", None)
    monkeypatch.setattr(mb, "available_memory", MagicMock(side_effect=RuntimeError("cgroups read failure")))

    with caplog.at_level(logging.WARNING, logger="omnivoice.model_lifecycle"):
        result = ml.list_loaded()
        assert "models" in result
        assert any(
            "Failed to collect system memory budget" in record.message
            and "cgroups read failure" in record.message
            for record in caplog.records
        )


def test_unload_all_handles_optional_engine_enumeration_exception(caplog, monkeypatch):
    import services.model_lifecycle as ml

    def fail_list_loaded():
        raise RuntimeError("Subprocess manager unavailable")

    monkeypatch.setattr(ml, "list_loaded", fail_list_loaded)

    with caplog.at_level(logging.WARNING, logger="omnivoice.model_lifecycle"):
        res = asyncio.run(ml.unload_all())
        assert isinstance(res, dict)
        assert any(
            "Could not enumerate optional engines during global unload: RuntimeError" in record.message
            for record in caplog.records
        )
