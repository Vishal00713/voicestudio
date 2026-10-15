"""Unit tests verifying exception logging and path redaction in audiobook progress reporting."""
import logging
import sys
import types
from unittest.mock import MagicMock
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/narrator/books/chapter1.txt", "/Users/narrator"),
        (r"C:\Users\narrator\books\chapter1.txt", r"C:\Users\narrator"),
        ("C:/Users/narrator/books/chapter1.txt", "C:/Users/narrator"),
    ],
)
def test_audiobook_progress_report_exception_logs_debug_and_scrubs_path(
    monkeypatch, caplog, raw_path, forbidden_prefix
):
    orig_mm = sys.modules.get("services.model_manager")
    orig_services = sys.modules.get("services")
    orig_services_mm = getattr(orig_services, "model_manager", None) if orig_services else None

    mock_mm = types.ModuleType("services.model_manager")

    def _faulty_report():
        raise RuntimeError(f"progress connection dropped at {raw_path}")

    mock_mm.report_generate_progress = _faulty_report
    monkeypatch.setitem(sys.modules, "services.model_manager", mock_mm)
    if orig_services is not None:
        setattr(orig_services, "model_manager", mock_mm)

    try:
        from services.audiobook import _note_chunk_done

        with caplog.at_level(logging.DEBUG, logger="omnivoice.audiobook"):
            # Must not raise
            _note_chunk_done()

        # Verify debug message logged and path redacted
        assert any(
            "Failed to report audiobook chunk progress" in r.message
            for r in caplog.records
        )
        assert forbidden_prefix not in caplog.text
        assert "~" in caplog.text
    finally:
        if orig_mm is not None:
            sys.modules["services.model_manager"] = orig_mm
        else:
            sys.modules.pop("services.model_manager", None)

        if orig_services is not None:
            if orig_services_mm is not None:
                setattr(orig_services, "model_manager", orig_services_mm)
            elif hasattr(orig_services, "model_manager"):
                delattr(orig_services, "model_manager")
