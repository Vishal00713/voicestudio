"""Runaway-take guard (``services/runaway_guard.py``).

Thresholds are pinned to the measured Hindi eval run they came from: a slow
but clean narrator take at 2.09x its expected length must pass untouched, a
take at 2.56x (the shortest runaway seen) must be re-rendered.
"""
import importlib

import pytest
import torch


class _Guard:
    """``services.runaway_guard`` looked up on every use, so a test that
    reloads ``services`` cannot leave this file holding a stale module."""

    def __getattr__(self, name):
        return getattr(importlib.import_module("services.runaway_guard"), name)


rg = _Guard()

SR = 24000


def _secs(s: float) -> torch.Tensor:
    return torch.zeros(1, int(SR * s))


def test_expected_seconds_uses_the_language_rate_and_ignores_markup():
    hindi = "तुम सच में कल सुबह इतनी जल्दी उठोगे?"
    assert rg.expected_seconds(hindi, "Hindi") == pytest.approx(len(hindi) / 17.0, rel=0.02)
    tagged = "[pause 300ms] " + hindi + " [laughter]"
    assert rg.expected_seconds(tagged, "hi") == pytest.approx(rg.expected_seconds(hindi, "hi"), rel=0.02)
    assert rg.expected_seconds("(calm, warm) " + hindi, "Hindi") == pytest.approx(
        rg.expected_seconds(hindi, "Hindi"), rel=0.02)
    assert rg.expected_seconds(hindi, "Hindi", speed=0.5) == pytest.approx(
        2 * rg.expected_seconds(hindi, "Hindi"), rel=0.02)


@pytest.mark.parametrize("ratio, expected_s, runaway", [
    (2.09, 3.5, False),     # the slow, clean documentary take
    (2.56, 3.9, True),      # the shortest runaway in the eval
    (12.6, 2.1, True),
    (1.0, 3.0, False),
    (2.5, 1.0, False),      # only 1.5 s over: a short line read slowly
])
def test_is_runaway_thresholds(ratio, expected_s, runaway):
    assert rg.is_runaway(ratio * expected_s, expected_s) is runaway


class Renders:
    def __init__(self, *lengths):
        self.lengths = list(lengths)
        self.attempts = []

    def __call__(self, attempt):
        self.attempts.append(attempt)
        return _secs(self.lengths[len(self.attempts) - 1])


TEXT = "यह एक सामान्य वाक्य है जिसे लगभग तीन सेकंड में पढ़ा जाता है।"


def test_a_normal_take_is_rendered_once():
    r = Renders(3.5)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert r.attempts == [0] and out.shape[-1] == int(SR * 3.5)


def test_a_runaway_take_is_rendered_again_and_the_retry_kept():
    r = Renders(20.0, 3.6)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert r.attempts == [0, 1] and out.shape[-1] == int(SR * 3.6)


def test_a_second_retry_rescues_a_stubborn_line():
    # The eval's date line ran on with two seeds and came out right with the third.
    r = Renders(13.3, 16.0, 3.4)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert r.attempts == [0, 1, 2] and out.shape[-1] == int(SR * 3.4)


def test_every_attempt_running_on_keeps_the_shortest():
    r = Renders(20.0, 12.0, 15.0)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert len(r.attempts) == 1 + rg.MAX_RETRIES
    assert out.shape[-1] == int(SR * 12.0)


def test_a_retry_that_lost_words_is_never_kept():
    r = Renders(20.0, 0.5, 0.6)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert out.shape[-1] == int(SR * 20.0)
    r = Renders(20.0, 0.5, 3.5)
    out = rg.render_guarded(r, TEXT, sample_rate=SR, language="Hindi")
    assert out.shape[-1] == int(SR * 3.5)


def test_retry_seed():
    assert rg.retry_seed(None) is None
    assert rg.retry_seed(7) == 7 + rg.RETRY_SEED_OFFSET
    assert rg.retry_seed(7, 2) == 7 + 2 * rg.RETRY_SEED_OFFSET
    assert 0 <= rg.retry_seed(2**31 - 1) < 2**31


def test_voxcpm2_sidecar_re_renders_a_runaway_with_another_seed(monkeypatch):
    from engines.voxcpm2_subprocess import VoxCPM2SubprocessBackend

    # Patch the class the adapter really inherits from: other test modules
    # re-import services.subprocess_backend, so a fresh import can be a
    # different class object than the adapter's base.
    SubprocessBackend = VoxCPM2SubprocessBackend.__bases__[0]
    calls = []
    lengths = iter([25.0, 3.4])

    def fake_generate(self, text, **kw):
        calls.append(kw.get("seed"))
        return torch.full((1, int(48000 * next(lengths))), 0.1)

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    backend = VoxCPM2SubprocessBackend()
    out = backend.generate(TEXT, language="Hindi", seed=11)
    assert calls == [11, rg.retry_seed(11)]
    assert out.shape[-1] == int(48000 * 3.4)


def test_voxcpm2_sidecar_normal_take_is_one_call(monkeypatch):
    from engines.voxcpm2_subprocess import VoxCPM2SubprocessBackend

    SubprocessBackend = VoxCPM2SubprocessBackend.__bases__[0]
    calls = []

    def fake_generate(self, text, **kw):
        calls.append(kw.get("seed"))
        return torch.full((1, int(48000 * 3.4)), 0.1)

    monkeypatch.setattr(SubprocessBackend, "generate", fake_generate)
    VoxCPM2SubprocessBackend().generate(TEXT, language="Hindi", seed=11)
    assert calls == [11]
