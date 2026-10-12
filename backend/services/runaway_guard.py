"""Catch a take that keeps talking after its text has ended, and render it again.

Autoregressive engines decide for themselves when to stop, and sometimes they
do not: on the Hindi eval set (``scripts/eval_hindi.py``, VoxCPM2, 25 lines) a
one-line question came back 26.7 s long, the line followed by twenty seconds
of invented syllables, and two more lines ran on the same way. Measured there,
good takes ran up to 2.1x the language's expected reading time (a slow
documentary narrator) and runaway ones from 2.6x to 12.6x.

A take longer than ``RUNAWAY_RATIO`` times its expected length, and at least
``MIN_EXCESS_S`` seconds over it, is rendered again with another seed, up to
``MAX_RETRIES`` times. The first retry that is neither a runaway nor
suspiciously short is kept; when none is, the shortest runaway wins (a
too-short retry has probably lost words and is never kept). On that eval set
one retry rescued two of the three run-on lines and a second retry the third.
Only a runaway is retried, so a normal take costs nothing. Expected reading
time comes from ``speech_rate`` (per-language characters per second); pure
arithmetic, identical on every platform.
"""
from __future__ import annotations

import logging
import re
from typing import Callable, Optional

logger = logging.getLogger("omnivoice.runaway_guard")

RUNAWAY_RATIO = 2.3
MIN_EXCESS_S = 2.5
MAX_RETRIES = 2
#: A retry shorter than this share of the expected time has probably lost words.
MIN_COMPLETE_RATIO = 0.5
RETRY_SEED_OFFSET = 7919

_TAGS = re.compile(r"\[[^\]]*\]")
_LEADING_CONTROL = re.compile(r"^\s*\([^()]*\)")


def expected_seconds(text: str, language: Optional[str] = None,
                     speed: Optional[float] = None) -> float:
    """How long a natural reading of ``text`` takes, in seconds."""
    from services.speech_rate import expected_duration  # noqa: PLC0415
    from services.text_normalization import _plain_lang_code  # noqa: PLC0415

    spoken = " ".join(_TAGS.sub(" ", _LEADING_CONTROL.sub("", text or "")).split())
    seconds = expected_duration(spoken, _plain_lang_code(language) or "en")
    try:
        rate = float(speed) if speed is not None else 1.0
    except (TypeError, ValueError):
        rate = 1.0
    return seconds / max(0.25, rate)


def is_runaway(seconds: float, expected: float) -> bool:
    return expected > 0 and seconds > max(RUNAWAY_RATIO * expected, expected + MIN_EXCESS_S)


def retry_seed(seed: Optional[int], attempt: int = 1) -> Optional[int]:
    """The seed for retry ``attempt``; None stays None (the engine draws one)."""
    if seed is None:
        return None
    return (int(seed) + attempt * RETRY_SEED_OFFSET) & 0x7FFF_FFFF


def render_guarded(render: Callable[[int], "object"], text: str, *, sample_rate: int,
                   language: Optional[str] = None, speed: Optional[float] = None,
                   label: str = "") -> "object":
    """``render(attempt)`` for attempt 0, then for 1, 2, … while the takes run
    away. Returns the take to keep (a tensor, last axis = samples)."""
    first = render(0)
    expected = expected_seconds(text, language, speed)
    first_s = first.shape[-1] / sample_rate
    if not is_runaway(first_s, expected):
        return first
    runaways = [(first_s, first)]
    lengths = [round(first_s, 1)]
    for attempt in range(1, MAX_RETRIES + 1):
        take = render(attempt)
        secs = take.shape[-1] / sample_rate
        lengths.append(round(secs, 1))
        if not is_runaway(secs, expected):
            if secs >= MIN_COMPLETE_RATIO * expected:
                logger.warning("%s take ran on past its text (about %.1f s of reading); "
                               "re-rendered: %s s, kept the last", label or "TTS", expected, lengths)
                return take
            continue  # too short: it probably lost words
        runaways.append((secs, take))
    logger.warning("%s take kept running on past its text (about %.1f s of reading) "
                   "in every attempt: %s s; kept the shortest", label or "TTS", expected, lengths)
    return min(runaways, key=lambda pair: pair[0])[1]
