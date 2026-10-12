"""Music (/music/*) — text → instrumental pieces, downloaded only on request.

MusicGen is stubbed: no GPU, no weights, no network.
"""
import asyncio
import importlib
import os

import pytest
import torch
from fastapi import HTTPException

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")


def _music():
    return importlib.import_module("services.music_gen")


def _router():
    return importlib.import_module("api.routers.music")


def test_status_reports_not_ready_until_downloaded(monkeypatch):
    music = _music()
    monkeypatch.setattr(music, "model_downloaded", lambda: False)
    status = music.status()
    assert status["ready"] is False
    assert status["license"] == "CC-BY-NC-4.0"
    assert status["max_seconds"] == 30.0


def test_download_starts_once_and_only_on_request(monkeypatch):
    music = _music()
    runs = []
    monkeypatch.setattr(music.threading, "Thread", lambda target, **_: type(
        "T", (), {"start": lambda self: runs.append(target)})())
    monkeypatch.setattr(music, "model_downloaded", lambda: False)
    monkeypatch.setitem(music._install, "stage", "idle")
    music.status()
    assert runs == []
    music.start_install()
    music.start_install()
    assert len(runs) == 1
    assert music.status()["stage"] == "downloading"
    monkeypatch.setitem(music._install, "stage", "idle")


@pytest.mark.parametrize("seconds, expected", [(1, 5.0), (12, 12.0), (90, 30.0)])
def test_length_is_clamped_to_the_training_window(seconds, expected):
    assert _music().clamp_seconds(seconds) == expected


def test_generate_refuses_with_download_cta_when_not_ready(monkeypatch):
    music, router = _music(), _router()
    monkeypatch.setattr(music, "model_downloaded", lambda: False)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(router.music_generate(prompt="lofi", seconds=10, variations=1, seed=1))
    assert caught.value.status_code == 409
    assert caught.value.detail["error"] == "music_not_installed"


def test_stereo_pieces_are_marked_per_channel_and_saved(monkeypatch):
    music, router = _music(), _router()
    monkeypatch.setattr(music, "model_downloaded", lambda: True)
    calls = {}

    def fake_generate(prompt, seconds, variations, seed):
        calls.update(prompt=prompt, seconds=seconds, variations=variations, seed=seed)
        return [torch.full((2, 32000), 0.1) for _ in range(variations)], 32000

    monkeypatch.setattr(music, "generate", fake_generate)
    marked = []

    async def fake_mark(waveform, sample_rate, *, context, force=False):
        marked.append((tuple(waveform.shape), sample_rate, context))
        return waveform

    monkeypatch.setattr(importlib.import_module("services.watermark"), "mark_synthetic_async", fake_mark)
    result = asyncio.run(router.music_generate(prompt=" calm piano ", seconds=12, variations=2, seed=9))

    assert calls == {"prompt": "calm piano", "seconds": 12.0, "variations": 2, "seed": 9}
    # AudioSeal is mono: every channel of every piece is marked on its own.
    assert marked == [((1, 32000), 32000, "music.finalize")] * 4
    assert [take["duration_s"] for take in result["takes"]] == [1.0, 1.0]

    from core import db as core_db
    with core_db.db_conn() as conn:
        modes = [row[0] for row in conn.execute(
            "SELECT mode FROM generation_history WHERE text = ?", ("calm piano",))]
    assert modes == ["music"] * 2
