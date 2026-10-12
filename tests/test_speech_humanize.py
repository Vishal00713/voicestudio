"""Sound more human on the OpenAI-compatible ``POST /v1/audio/speech``.

Every project on the owner's PC reaches VoiceStudio through this route, so the
same pass /generate offers is available here as two extension fields. Off by
default: a request without them is unchanged.
"""
import importlib
import io
import math

import pytest
import soundfile as sf
import torch

SR = 24000


def _take() -> torch.Tensor:
    """speech, 0.6 s of exact digital silence, speech."""
    t = torch.arange(SR) / SR
    tone = torch.sin(2 * math.pi * 220 * t) * 0.5
    return torch.cat([tone, torch.zeros(int(0.6 * SR)), tone]).unsqueeze(0)


class _Backend:
    sample_rate = SR
    applies_own_mastering = True

    def __init__(self):
        self.calls = []

    def generate(self, text, **kw):
        self.calls.append((text, kw))
        return _take()


def test_run_tts_is_unchanged_without_humanize(monkeypatch):
    oc = importlib.import_module("api.routers.openai_compat")
    wav, sr = oc._run_tts(_Backend(), "नमस्ते", {"seed": 3})
    gap = wav[..., SR + SR // 10: SR + SR // 2]
    assert sr == SR and float(gap.abs().max()) == 0.0, "pauses stay digital silence"


def test_run_tts_humanizes_with_the_profile_room(tmp_path):
    oc = importlib.import_module("api.routers.openai_compat")
    from services.humanize import HumanizeSettings

    ref = tmp_path / "ref.wav"
    sf.write(str(ref), _take().squeeze(0).numpy() + 0.002 * torch.randn(SR * 2 + int(0.6 * SR)).numpy(), SR)
    backend = _Backend()
    wav, _ = oc._run_tts(backend, "नमस्ते", {"seed": 3, "ref_audio": str(ref)},
                         HumanizeSettings(strength=0.6))
    gap = wav[..., SR + SR // 10: SR + SR // 2]
    assert float(gap.abs().max()) > 0.0, "pauses carry a room tone"
    again, _ = oc._run_tts(backend, "नमस्ते", {"seed": 3, "ref_audio": str(ref)},
                           HumanizeSettings(strength=0.6))
    assert torch.equal(wav, again), "a pinned seed renders the same take"


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app

    tts = importlib.import_module("services.tts_backend")

    class _Fake(tts.TTSBackend):
        id = "fake-speech-humanize"
        display_name = "Fake (test)"
        gpu_compat = ("cpu",)
        applies_own_mastering = True

        @property
        def sample_rate(self) -> int:
            return SR

        @property
        def supported_languages(self) -> list[str]:
            return ["multi"]

        @classmethod
        def is_available(cls):
            return True, "ready"

        def generate(self, text, **kw):
            SPOKEN.append(text)
            return _take()

    SPOKEN.clear()
    monkeypatch.setitem(tts._REGISTRY, _Fake.id, _Fake)
    return TestClient(app, client=("127.0.0.1", 50000))


SPOKEN: list = []


@pytest.mark.parametrize("humanize, silent_gap", [(False, True), (True, False)])
def test_speech_route_honours_humanize(client, humanize, silent_gap):
    body = {"model": "fake-speech-humanize", "input": "नमस्ते दुनिया", "response_format": "wav",
            "language": "hi", "seed": 5}
    if humanize:
        body.update(humanize=True, humanize_strength=0.6)
    resp = client.post("/v1/audio/speech", json=body)
    assert resp.status_code == 200, resp.text[:300]
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    gap = audio[int(sr * 1.1): int(sr * 1.5)]
    assert bool(abs(gap).max() < 1e-6) is silent_gap


def test_speech_route_rejects_an_out_of_range_strength(client):
    resp = client.post("/v1/audio/speech", json={
        "model": "fake-speech-humanize", "input": "नमस्ते", "humanize": True, "humanize_strength": 2,
    })
    assert resp.status_code in (400, 422)


def test_speech_route_stitches_pause_markers_into_real_silence(client):
    """``[pause 500ms]`` is a pause, not text: each side renders on its own and
    exactly half a second of silence joins them, as on /generate."""
    resp = client.post("/v1/audio/speech", json={
        "model": "fake-speech-humanize", "input": "पहली बात। [pause 500ms] दूसरी बात।",
        "response_format": "wav", "language": "hi",
    })
    assert resp.status_code == 200, resp.text[:300]
    assert SPOKEN == ["पहली बात।", "दूसरी बात।"]
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    one = _take().shape[-1]
    assert audio.shape[0] == 2 * one + sr // 2
    assert float(abs(audio[one: one + sr // 2]).max()) == 0.0


def test_speech_route_never_reads_a_cue_aloud(client):
    resp = client.post("/v1/audio/speech", json={
        "model": "fake-speech-humanize", "input": "[excited] हमने कर दिखाया!",
        "response_format": "wav", "language": "hi",
    })
    assert resp.status_code == 200, resp.text[:300]
    assert SPOKEN == ["हमने कर दिखाया!"]
    assert resp.headers.get("X-OmniVoice-Expression-Dropped") == "[excited]"
