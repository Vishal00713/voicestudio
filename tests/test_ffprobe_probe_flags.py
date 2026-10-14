"""Unit tests verifying ffprobe command invocations safely delimit file paths with '--'."""
import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def test_probe_audio_stream_delimits_path_with_double_dash(monkeypatch):
    from services import ffmpeg_utils

    captured_cmds = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.stdout = b"0\n"
        return mock_proc

    monkeypatch.setattr(ffmpeg_utils, "find_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(ffmpeg_utils.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ffmpeg_utils.subprocess, "run", mock_run)

    path = "-flag_like_path.wav"
    result = ffmpeg_utils._probe_audio_stream(path)

    assert result is True
    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    dash_idx = cmd.index("--")
    assert dash_idx != -1
    assert cmd[dash_idx + 1] == path


def test_probe_duration_delimits_path_with_double_dash(monkeypatch):
    from services import ffmpeg_utils

    captured_args = []

    async def mock_spawn(*args, **kwargs):
        captured_args.append(list(args))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"12.34\n", b""))
        return mock_proc

    monkeypatch.setattr(ffmpeg_utils, "find_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(ffmpeg_utils, "resolve_within", lambda root, p: MagicMock(is_file=lambda: True, __str__=lambda self: p))
    monkeypatch.setattr(ffmpeg_utils, "spawn_subprocess", mock_spawn)

    async def _run():
        path = "-lead_dash.mp3"
        duration = await ffmpeg_utils.probe_duration(path, allowed_root="/tmp")
        assert duration == 12.34
        assert len(captured_args) == 1
        cmd = captured_args[0]
        dash_idx = cmd.index("--")
        assert cmd[dash_idx + 1] == path

    asyncio.run(_run())


def test_probe_frame_rates_delimits_path_with_double_dash(monkeypatch):
    from services import ffmpeg_utils

    captured_args = []

    async def mock_spawn(*args, **kwargs):
        captured_args.append(list(args))
        mock_proc = MagicMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"30/1,30/1\n", b""))
        return mock_proc

    monkeypatch.setattr(ffmpeg_utils, "find_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(ffmpeg_utils.os.path, "isfile", lambda p: True)
    monkeypatch.setattr(ffmpeg_utils, "spawn_subprocess", mock_spawn)

    async def _run():
        path = "-video_file.mp4"
        rates = await ffmpeg_utils.probe_frame_rates(path)
        assert rates == ("30/1", "30/1")
        assert len(captured_args) == 1
        cmd = captured_args[0]
        dash_idx = cmd.index("--")
        assert cmd[dash_idx + 1] == path

    asyncio.run(_run())


def test_dub_pipeline_probe_codecs_delimits_path_with_double_dash(monkeypatch):
    from services import dub_pipeline

    captured_cmds = []

    def mock_run(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        mock_proc = MagicMock()
        mock_proc.stdout = "h264\n"
        return mock_proc

    monkeypatch.setattr(dub_pipeline, "find_ffprobe", lambda: "/usr/bin/ffprobe")
    monkeypatch.setattr(dub_pipeline.subprocess, "run", mock_run)

    path = "-leading_dash_video.mp4"
    vcodec, acodec = dub_pipeline._probe_codecs(path)

    assert vcodec == "h264"
    assert len(captured_cmds) == 2
    for cmd in captured_cmds:
        dash_idx = cmd.index("--")
        assert cmd[dash_idx + 1] == path
