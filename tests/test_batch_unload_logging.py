"""Unit tests verifying exception logging and redaction during batch transcription backend unload."""
import asyncio
import logging
from unittest.mock import MagicMock
import pytest


def test_batch_transcribe_unload_exception_logs_warning_and_scrubs_path(tmp_path, monkeypatch, caplog):
    async def _test():
        import sys
        monkeypatch.setitem(sys.modules, "torch", MagicMock())
        monkeypatch.setitem(sys.modules, "torchaudio", MagicMock())

        from api.routers import batch

        job_id = "test_batch_unload"
        video_path = tmp_path / "input.mp4"
        video_path.write_bytes(b"dummy video")

        batch._jobs[job_id] = {
            "status": "processing",
            "video_path": str(video_path),
            "langs": ["en"],
            "stage": "pending",
            "progress": 0,
            "filename": "input.mp4",
            "translation_provider": "argos",
        }

        # Mock extract step helpers
        monkeypatch.setattr("services.ffmpeg_utils.validate_media_source", lambda path: None)
        monkeypatch.setattr("services.ffmpeg_utils.require_audio_stream", lambda path: None)
        monkeypatch.setattr("services.ffmpeg_utils.find_ffmpeg", lambda: "ffmpeg")

        # Mock subprocess.run for audio extraction and duration probing
        def _mock_subprocess_run(*args, **kwargs):
            mock_res = MagicMock()
            mock_res.stderr = b"Duration: 00:00:01.00"
            return mock_res

        monkeypatch.setattr("subprocess.run", _mock_subprocess_run)

        # Mock ASR backend with unload raising an exception containing a sensitive path
        fake_backend = MagicMock()
        fake_backend.transcribe.return_value = {
            "text": "hello",
            "language": "en",
            "segments": [{"text": "hello", "start": 0.0, "end": 1.0}],
        }
        fake_backend.unload.side_effect = RuntimeError("failed to release VRAM at /Users/tester/cache/whisper.bin")

        monkeypatch.setattr("services.asr_backend.load_active_asr_backend", lambda: fake_backend)

        # Make run_transcribe_guarded execute the callable directly
        async def _mock_run_transcribe_guarded(pool, fn, what="ASR"):
            return fn()

        monkeypatch.setattr("services.asr_backend.run_transcribe_guarded", _mock_run_transcribe_guarded)

        # Stop before step 3 (translation) to isolate transcription unload
        original_set_progress = batch._set_progress

        def _intercept_progress(job, stage, pct, **kwargs):
            original_set_progress(job, stage, pct, **kwargs)
            if stage == "transcribe" and pct == 100:
                job["status"] = "cancelled"

        monkeypatch.setattr(batch, "_set_progress", _intercept_progress)

        with caplog.at_level(logging.WARNING):
            try:
                await batch._run_batch_pipeline(job_id, batch._jobs[job_id])
            finally:
                batch._jobs.pop(job_id, None)

        # Verify warning was logged
        assert any(
            "Failed to unload ASR backend after batch transcription" in r.message
            for r in caplog.records
        )
        # Verify path was scrubbed
        assert "/Users/tester" not in caplog.text

    asyncio.run(_test())
