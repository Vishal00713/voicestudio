"""Unit tests verifying exception logging and redaction in dub_export handlers."""
import asyncio
import logging
from unittest.mock import AsyncMock, patch
import pytest


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/db/jobs.sqlite", "/Users/audio"),
        (r"C:\Users\audio\db\jobs.sqlite", r"C:\Users\audio"),
        ("C:/Users/audio/db/jobs.sqlite", "C:/Users/audio"),
    ],
)
def test_stream_task_events_since_exception_logs_debug_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import dub_export

    with patch("core.job_store.get", return_value={"status": "done"}), \
         patch.dict(dub_export.task_manager.active_tasks, {"test-task": {"status": "done"}}, clear=True), \
         patch("core.job_store.events_since", side_effect=RuntimeError(f"db failure at {raw_path}")):
        async def run():
            resp = await dub_export.stream_task("test-task", after_seq=0)
            async for _ in resp.body_iterator:
                pass

        with caplog.at_level(logging.DEBUG, logger="omnivoice.api"):
            asyncio.run(run())

    assert any(
        "Failed to replay persisted events for task" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/qc/events.log", "/Users/audio"),
        (r"C:\Users\audio\qc\events.log", r"C:\Users\audio"),
        ("C:/Users/audio/qc/events.log", "C:/Users/audio"),
    ],
)
def test_dub_qc_pass_append_event_exception_logs_debug_and_scrubs_path(
    caplog, raw_path, forbidden_prefix
):
    from api.routers import dub_export

    with patch.object(dub_export, "_job_dir_or_400"), \
         patch.object(dub_export, "_safe_lang_or_400", side_effect=lambda l: l), \
         patch.object(dub_export, "_get_job", return_value={"dubbed_tracks": {"en": {"path": "dubbed.wav"}}, "segments": [{"id": 0, "text": "hello"}]}), \
         patch.object(dub_export, "_dub_artifact", return_value="dummy.wav"), \
         patch.object(dub_export, "_segments_for_lang", return_value=[{"id": 0, "text": "hello"}]), \
         patch("services.asr_backend.asr_model_missing_error", return_value=None), \
         patch("services.asr_backend.run_transcribe_guarded", new=AsyncMock(return_value=([], "whisper"))), \
         patch("services.dub_qc.score_dub", return_value=[]), \
         patch("services.dub_pipeline.put_job"), \
         patch("services.dub_pipeline.save_job"), \
         patch("core.job_store.append_event", side_effect=RuntimeError(f"disk failure at {raw_path}")):
        async def run():
            res = await dub_export.dub_qc_pass("job-123", lang="en")
            assert res["engine"] == "whisper"

        with caplog.at_level(logging.DEBUG, logger="omnivoice.api"):
            asyncio.run(run())

    assert any(
        "QC event append failed for job" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


@pytest.mark.parametrize(
    "raw_path, forbidden_prefix",
    [
        ("/Users/audio/cache/onsets.json", "/Users/audio"),
        (r"C:\Users\audio\cache\onsets.json", r"C:\Users\audio"),
        ("C:/Users/audio/cache/onsets.json", "C:/Users/audio"),
    ],
)
def test_onsets_cache_read_exception_logs_debug_and_scrubs_path(
    monkeypatch, caplog, raw_path, forbidden_prefix, tmp_path
):
    import os
    from api.routers import dub_export

    job_dir = tmp_path / "job-123"
    job_dir.mkdir()
    src_file = job_dir / "vocals.wav"
    src_file.write_bytes(b"dummy wav data")

    cache_file = job_dir / "onsets.json"
    cache_file.write_text("{corrupt json", encoding="utf-8")
    # Set cache mtime >= src mtime
    os.utime(cache_file, (src_file.stat().st_mtime + 10, src_file.stat().st_mtime + 10))

    monkeypatch.setattr(dub_export, "DUB_DIR", str(tmp_path))
    monkeypatch.setattr(dub_export, "_job_dir_or_400", lambda jid: str(job_dir))
    monkeypatch.setattr(dub_export, "_get_job", lambda jid: {"vocals_path": "vocals.wav"})
    monkeypatch.setattr(dub_export, "_optional_dub_artifact", lambda p, jid: str(src_file))
    monkeypatch.setattr(dub_export, "_compute_timeline_sync", lambda path: ([0.1], [0.5]))

    # Inject path in ValueError representation to test redaction
    def _faulty_json_load(f):
        raise ValueError(f"malformed onsets syntax at {raw_path}")

    monkeypatch.setattr("json.load", _faulty_json_load)

    with caplog.at_level(logging.DEBUG, logger="omnivoice.api"):
        res = asyncio.run(dub_export.dub_get_onsets("job-123"))

    assert res["onsets"] == [0.1]
    assert any(
        "onsets cache read failed, recomputing" in r.message
        for r in caplog.records
    )
    assert forbidden_prefix not in caplog.text
    assert "~" in caplog.text


def test_dub_download_early_failure_after_subtitle_creation_cleans_up_subtitle_file(
    tmp_path, monkeypatch
):
    import os
    from api.routers import dub_export

    job_dir = tmp_path / "job-early-fail"
    exports_dir = job_dir / "exports"
    exports_dir.mkdir(parents=True)
    vid_file = job_dir / "source.mp4"
    vid_file.write_bytes(b"dummy video")
    track_file = job_dir / "track.wav"
    track_file.write_bytes(b"dummy audio")

    job = {
        "video_path": str(vid_file),
        "input_type": "video",
        "dubbed_tracks": {"es": {"path": str(track_file), "language": "Spanish"}},
        "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hola"}],
    }

    created_subs = []
    real_write_burn_srt = dub_export._write_burn_srt

    def mock_write_burn_srt(*args, **kwargs):
        path = real_write_burn_srt(*args, **kwargs)
        created_subs.append(path)
        assert os.path.exists(path)
        return path

    monkeypatch.setattr(dub_export, "_job_dir_or_400", lambda jid: str(job_dir))
    monkeypatch.setattr(dub_export, "_get_job", lambda jid: job)
    monkeypatch.setattr(dub_export, "_dub_artifact", lambda p, j, **kw: p)
    monkeypatch.setattr(dub_export, "_write_burn_srt", mock_write_burn_srt)

    async def _failing_preserved_bg(*args, **kwargs):
        raise RuntimeError("early background mix failure")

    monkeypatch.setattr(dub_export, "_preserved_background", _failing_preserved_bg)

    with pytest.raises(RuntimeError, match="early background mix failure"):
        asyncio.run(dub_export.dub_download("job-early-fail", burn_subs=True, include_tracks="", default_track=""))

    assert len(created_subs) == 1
    assert not os.path.exists(created_subs[0]), f"Subtitle {created_subs[0]} was left behind after early failure!"
