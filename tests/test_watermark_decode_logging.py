"""Unit tests verifying exception logging and redaction in watermark message decoding."""
import logging
from unittest.mock import MagicMock
import pytest


def test_watermark_message_decode_exception_logs_debug_and_scrubs_path(monkeypatch, caplog):
    import sys
    orig_torch = sys.modules.get("torch")
    orig_watermark = sys.modules.get("services.watermark")
    orig_services = sys.modules.get("services")
    orig_services_wm = getattr(orig_services, "watermark", None) if orig_services else None

    try:
        if orig_torch is None:
            mock_torch = MagicMock()
            mock_torch.no_grad.return_value = lambda f: f
            mock_waveform = MagicMock()
            mock_waveform.dim.return_value = 2
            mock_waveform.unsqueeze.return_value = mock_waveform
            mock_torch.Tensor = MagicMock
            monkeypatch.setitem(sys.modules, "torch", mock_torch)
        else:
            mock_waveform = orig_torch.zeros((1, 16000))

        from services import watermark

        monkeypatch.setattr(watermark, "_check_available", lambda: True)

        faulty_decoded_msg = MagicMock()
        faulty_decoded_msg.squeeze.side_effect = RuntimeError(
            "failed to squeeze message at /Users/tester/secret/audioseal.pt"
        )

        mock_detector = MagicMock()
        mock_detector.detect_watermark.return_value = (0.9, faulty_decoded_msg)
        monkeypatch.setattr(watermark, "_get_detector", lambda: mock_detector)
        monkeypatch.setattr(watermark, "_iter_chunks", lambda audio, sr: [audio])
        monkeypatch.setattr(watermark, "_audioseal_resample", lambda seg, in_sr, out_sr: seg)

        with caplog.at_level(logging.DEBUG, logger="omnivoice.watermark"):
            res = watermark.detect_watermark(mock_waveform, sample_rate=16000)

        # Verify detection still succeeds for confidence
        assert res["is_watermarked"] is True
        assert res["confidence"] == 0.9
        assert res["message_bits"] == ""
        assert res["is_omnivoice"] is False

        # Verify debug message logged and path redacted
        assert any("Failed to decode watermark message bits" in r.message for r in caplog.records)
        assert "/Users/tester" not in caplog.text
    finally:
        if orig_torch is None:
            sys.modules.pop("services.watermark", None)
            if orig_watermark is not None:
                sys.modules["services.watermark"] = orig_watermark
            if orig_services is not None:
                if orig_services_wm is not None:
                    setattr(orig_services, "watermark", orig_services_wm)
                elif hasattr(orig_services, "watermark"):
                    delattr(orig_services, "watermark")

