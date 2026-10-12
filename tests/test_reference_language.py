"""Reference-language notice (``services/reference_language.py``).

A clone copies the reference's accent. When the reference transcript is in a
script the requested language never uses, ``/generate`` says so in a header
so the UI can tell the user why the take sounds foreign.
"""
import pytest
import torch

from services.reference_language import (
    HEADER,
    dominant_script,
    notice_headers,
    script_mismatch,
)

EN = "Welcome aboard. I was just a three-second clip a moment ago."
HI = "नमस्ते और इस छोटे से वॉइस डेमो में आपका स्वागत है।"
UR = "آپ کا اس مختصر آواز کے ڈیمو میں خوش آمدید ہے"
JA = "こんにちは。この短い音声デモへようこそ。"
ZH = "大家好，欢迎来到这个声音示范。"


@pytest.mark.parametrize("text, script", [
    (EN, "latin"),
    (HI, "devanagari"),
    (UR, "arabic"),
    (JA, "kana"),
    (ZH, "cjk"),
    ("Привет, как дела у тебя сегодня?", "cyrillic"),
    ("hi there", None),                       # too few letters to say
    ("Hello नमस्ते Hello नमस्ते", None),        # no clear leader
    ("", None),
    (None, None),
    ("1234 5678 !!!", None),
])
def test_dominant_script(text, script):
    assert dominant_script(text) == script


@pytest.mark.parametrize("ref, language, found", [
    (EN, "Hindi", "latin"),                   # today's bug: English sample, Hindi take
    (EN, "hi", "latin"),
    (EN, "hi-IN", "latin"),
    (HI, "English", "devanagari"),
    (HI, "Hindi", None),
    (UR, "Hindi", None),                      # Hindi/Urdu: one spoken language
    (HI, "Urdu", None),
    (EN, "Spanish", None),                    # same script: cannot tell, so silent
    (HI, "Marathi", None),
    (JA, "Japanese", None),
    (ZH, "Japanese", None),
    (EN, "Japanese", "latin"),
    (HI, "Bengali", "devanagari"),
    (EN, "Auto", None),                       # no language to compare with
    (EN, None, None),
    (EN, "Klingon", None),                    # unknown language: silent
    ("", "Hindi", None),                      # no transcript: silent
])
def test_script_mismatch(ref, language, found):
    assert script_mismatch(ref, language) == found


def test_notice_headers_shape_and_never_raises(monkeypatch):
    assert notice_headers(EN, "Hindi") == {HEADER: "latin"}
    assert notice_headers(HI, "Hindi") == {}
    import services.reference_language as rl

    monkeypatch.setattr(rl, "script_mismatch", lambda *a: 1 / 0)
    assert rl.notice_headers(EN, "Hindi") == {}


# ── /generate carries the notice ────────────────────────────────────────────

def _make_fake_engine():
    import importlib

    tts = importlib.import_module("services.tts_backend")

    class _FakeEngine(tts.TTSBackend):
        id = "fake-ref-lang-engine"
        display_name = "Fake Reference Language Engine (test)"
        gpu_compat = ("cpu",)

        @property
        def sample_rate(self) -> int:
            return 24000

        @property
        def supported_languages(self) -> list[str]:
            return ["multi"]

        @classmethod
        def is_available(cls):
            return True, "ready"

        def generate(self, text, **kw) -> torch.Tensor:
            t = torch.arange(24000) / 24000
            return (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)

    return tts, _FakeEngine


@pytest.fixture()
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from main import app

    tts, fake = _make_fake_engine()
    monkeypatch.setitem(tts._REGISTRY, fake.id, fake)
    monkeypatch.delenv("OMNIVOICE_TTS_BACKEND", raising=False)
    return TestClient(app, client=("127.0.0.1", 50000))


def _wav_bytes() -> bytes:
    import io

    import soundfile as sf

    buf = io.BytesIO()
    t = torch.arange(24000 * 3) / 24000
    sf.write(buf, (0.3 * torch.sin(2 * torch.pi * 180 * t)).numpy(), 24000, format="WAV")
    return buf.getvalue()


@pytest.mark.parametrize("ref_text, language, expected", [
    (EN, "Hindi", "latin"),
    (HI, "Hindi", None),
])
def test_generate_sends_the_notice_only_on_a_mismatch(client, ref_text, language, expected):
    resp = client.post(
        "/generate",
        data={"text": "नमस्ते दुनिया", "language": language, "ref_text": ref_text,
              "engine": "fake-ref-lang-engine", "effect_preset": "raw"},
        files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text[:300]
    assert resp.headers.get(HEADER) == expected
