"""Sound effects (/sfx/*) — text → SFX, installed only on explicit user action.

The diffusion model is stubbed: no GPU, no weights, no network.
"""
import asyncio
import importlib
import os

import pytest
import torch
from fastapi import HTTPException

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")


def _sfx():
    return importlib.import_module("services.sound_effects")


def _router():
    return importlib.import_module("api.routers.sound_effects")


def test_status_reports_not_ready_without_runtime_or_model(monkeypatch):
    sfx = _sfx()
    monkeypatch.setattr(sfx, "runtime_installed", lambda: False)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: False)
    status = sfx.status()
    assert status["ready"] is False
    assert status["stage"] == "idle"
    assert status["license"] == "CC-BY-NC-SA-4.0"
    assert status["download_bytes"] > 1_000_000_000


def test_install_never_starts_on_its_own(monkeypatch):
    """Local-first: nothing downloads until the user asks (start_install)."""
    sfx = _sfx()
    started = []
    monkeypatch.setattr(sfx, "_run_install", lambda: started.append(True))
    monkeypatch.setattr(sfx, "runtime_installed", lambda: False)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: False)
    sfx.status()
    assert started == []


def test_start_install_runs_once_while_busy(monkeypatch):
    sfx = _sfx()
    runs = []
    monkeypatch.setattr(sfx.threading, "Thread", lambda target, **_: type(
        "T", (), {"start": lambda self: runs.append(target)})())
    monkeypatch.setattr(sfx, "runtime_installed", lambda: False)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: False)
    monkeypatch.setitem(sfx._install, "stage", "idle")
    sfx.start_install()
    sfx.start_install()
    assert len(runs) == 1
    assert sfx.status()["stage"] == "installing"
    monkeypatch.setitem(sfx._install, "stage", "idle")


def test_failed_install_is_reported_not_raised(monkeypatch):
    sfx = _sfx()
    monkeypatch.setattr(sfx, "runtime_installed", lambda: False)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: False)

    def boom():
        raise RuntimeError("Installing diffusers failed: no network")

    monkeypatch.setattr(sfx, "_install_runtime", boom)
    sfx._run_install()
    status = sfx.status()
    assert status["stage"] == "error"
    assert "no network" in status["error"]
    monkeypatch.setitem(sfx._install, "stage", "idle")
    monkeypatch.setitem(sfx._install, "error", None)


@pytest.mark.parametrize("seconds, expected", [(0.1, 1.0), (5, 5.0), (99, 20.0)])
def test_length_is_clamped(seconds, expected):
    assert _sfx().clamp_seconds(seconds) == expected


def test_generate_refuses_with_install_cta_when_not_ready(monkeypatch):
    sfx, router = _sfx(), _router()
    monkeypatch.setattr(sfx, "runtime_installed", lambda: False)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: False)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(router.sfx_generate(prompt="rain", seconds=5, variations=2, seed=1))
    assert caught.value.status_code == 409
    assert caught.value.detail["error"] == "sfx_not_installed"


def test_generate_saves_every_variation_as_a_marked_take(monkeypatch):
    sfx, router = _sfx(), _router()
    monkeypatch.setattr(sfx, "runtime_installed", lambda: True)
    monkeypatch.setattr(sfx, "model_downloaded", lambda: True)
    calls = {}

    def fake_generate(prompt, seconds, variations, seed, steps=100):
        calls.update(prompt=prompt, seconds=seconds, variations=variations, seed=seed)
        return [torch.full((1, 16000), 0.1 * (i + 1)) for i in range(variations)]

    monkeypatch.setattr(sfx, "generate", fake_generate)
    marked = []

    async def fake_mark(waveform, sample_rate, *, context, force=False):
        marked.append((context, sample_rate))
        return waveform

    monkeypatch.setattr(importlib.import_module("services.watermark"), "mark_synthetic_async", fake_mark)
    result = asyncio.run(router.sfx_generate(prompt="  door creak ", seconds=50, variations=3, seed=7))

    assert calls == {"prompt": "door creak", "seconds": 20.0, "variations": 3, "seed": 7}
    assert len(result["takes"]) == 3
    assert marked == [("sfx.finalize", 16000)] * 3
    assert all(take["audio_url"].startswith("/audio/") for take in result["takes"])
    assert result["seed"] == 7

    from core import db as core_db
    with core_db.db_conn() as conn:
        modes = [row[0] for row in conn.execute(
            "SELECT mode FROM generation_history WHERE text = ?", ("door creak",))]
    assert modes == ["sfx"] * 3


def test_takes_are_ranked_by_clap_without_librosa():
    """diffusers ranks AudioLDM2 takes only when librosa is installed, and the
    app runtime has none; the bound scorer resamples with torchaudio instead."""
    import types

    sfx = _sfx()
    seen = {}

    class Inputs(dict):
        def to(self, _device):
            return self

    class FeatureExtractor:
        sampling_rate = 48000

        def __call__(self, audios, return_tensors, sampling_rate):
            seen["lengths"] = [len(a) for a in audios]
            return types.SimpleNamespace(input_features=torch.zeros(len(audios), 1))

    pipe = types.SimpleNamespace(
        tokenizer=lambda text, **_: Inputs(text=text),
        feature_extractor=FeatureExtractor(),
        vocoder=types.SimpleNamespace(config=types.SimpleNamespace(sampling_rate=16000)),
        # CLAP's similarity of the prompt to takes 0, 1, 2: take 2 matches best.
        text_encoder=lambda **_: types.SimpleNamespace(logits_per_text=torch.tensor([[0.1, 0.3, 0.9]])),
    )
    audio = torch.stack([torch.full((16000,), float(i)) for i in range(3)])
    ranked = sfx._score_waveforms(pipe, "rain", audio, 3, "cpu", torch.float32)
    assert [int(clip[0]) for clip in ranked] == [2, 1, 0]
    assert seen["lengths"] == [48000] * 3, "resampled to CLAP's 48 kHz"
