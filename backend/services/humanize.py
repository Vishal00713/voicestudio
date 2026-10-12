"""Humanize — take the synthetic tells out of a clean render.

A model's output is *too* clean. Three things a listener hears as "AI"
without being able to name them, and what this module does about each:

* **Digital silence.** Every ``[pause]`` and every gap the chunker stitches
  is a run of exact zeros. No microphone has ever recorded that. A low pink
  noise bed runs under the whole take, so a pause sounds like the same room
  as the words around it (:func:`add_room_tone`).
* **Nobody breathes.** A person inhales before a sentence. Short, quiet,
  band-limited noise bursts are placed at the end of silences long enough to
  hold one, timed to finish just before the next onset. Only some gaps get
  one — a breath at every pause is its own tell (:func:`add_breaths`).
* **Metronome pacing.** Every chunk is rendered on its own and joined with
  whatever lead-in and tail the engine padded it with, so each sentence
  boundary has the same shape. :func:`prepare_chunks` trims that padding,
  sizes the gap by the punctuation that ended the chunk (comma, full stop,
  ellipsis), jitters it, and lets each chunk's tempo and pitch drift a
  little, the way one speaker never delivers two sentences quite alike.

When the take was cloned from a reference clip, the room and the breaths
come from that clip (:func:`room_from_reference`): the bed takes the
reference's own noise spectrum and its level below the voice, and the breaths
are the speaker's real inhales cut out of the reference. A clone then sounds
like the same microphone in the same room, breathing like the same person.
Without a usable reference (voice design, a studio-clean clip with digital
silence, no pauses) the synthetic pink bed and synthesized inhales stand in.

Pure torch on the tensor's own device, seeded from the take's seed so a
pinned seed renders the same breaths in the same places. No model, no
network, identical on every platform. Explicit ``[pause Nms]`` lengths are
never changed — those are the user's direction. Everything here is opt-in:
nothing runs unless the request asks for it (``humanize=true``).

A take with no speech in it (a dead render, or pause markers only) is
returned untouched: dressing silence up as a room would hide the failure
from the dead-render guards and let ``normalize_audio`` amplify the bed to
full-scale hiss.
"""
from __future__ import annotations

import logging
import math
import os
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import torch

logger = logging.getLogger("omnivoice.humanize")

DEFAULT_STRENGTH = 0.6

#: A silence must be at least this long to hold a breath and still leave a
#: beat between the breath and the next word.
MIN_BREATH_GAP_MS = 350
#: Frames whose peak sits this far below the clip's peak count as silent.
SILENCE_BELOW_PEAK_DB = 40.0
#: Same floor as ``audio_dsp.normalize_audio``: a clip that never rises above
#: it has no speech, and is left alone.
SPEECH_FLOOR_DBFS = -50.0

_FRAME_MS = 5
_BREATH_LEAD_MS = 60      # a breath ends this long before the next onset
_BREATH_MIN_S = 0.18
_BREATH_MAX_S = 0.42
_EDGE_FADE_MS = 8
_PINK_BLOCK = 1 << 15     # ~1.4 s at 24 kHz; FFT-shaped with 75 % Hann overlap

#: Latin, CJK and the Devanagari danda / double danda (Hindi, Marathi, Nepali).
_SENTENCE_END = ".!?。！？।॥"
_CLAUSE_END = ",;:、，；："
_TRAIL_OFF = "…—"
_CLOSERS = "\"')]}»”’"

_SPEECH_FLOOR = 10.0 ** (SPEECH_FLOOR_DBFS / 20.0)


@dataclass(frozen=True)
class HumanizeSettings:
    """What the pass does, resolved from one 0–1 ``strength``.

    ``strength`` scales the breath level and probability, the tempo drift and
    the gap jitter together; the room tone rises from -66 dBFS RMS at 0 to
    -58 dBFS at 1. The three switches exist for tests and power users — the
    request surface is just the strength.
    """

    strength: float = DEFAULT_STRENGTH
    breaths: bool = True
    drift: bool = True
    gaps: bool = True

    @staticmethod
    def from_request(enabled: object, strength: object = None) -> Optional["HumanizeSettings"]:
        """``None`` unless ``enabled`` is truthy; ``strength`` clamped to 0–1,
        falling back to the default when it is missing or not a number."""
        if not enabled:
            return None
        try:
            value = DEFAULT_STRENGTH if strength is None else float(strength)
        except (TypeError, ValueError):
            value = DEFAULT_STRENGTH
        if not math.isfinite(value):
            value = DEFAULT_STRENGTH
        return HumanizeSettings(strength=min(1.0, max(0.0, value)))

    @property
    def room_tone_db(self) -> float:
        return -66.0 + 8.0 * self.strength


def make_rng(seed: Optional[int], salt: int = 0) -> torch.Generator:
    """A CPU generator derived from the take's seed. Two call sites that must
    not share a stream (the join and the whole-take pass) use different salts."""
    rng = torch.Generator(device="cpu")
    if seed is None:
        base = int(torch.randint(0, 2**31 - 1, (1,)).item())
    else:
        base = int(seed)
    rng.manual_seed((base * 1_000_003 + salt * 7919) & 0x7FFF_FFFF_FFFF_FFFF)
    return rng


def _uniform(rng: torch.Generator) -> float:
    return float(torch.rand(1, generator=rng).item())


def _mono_level(audio: torch.Tensor) -> torch.Tensor:
    level = audio.abs()
    if level.dim() > 1:
        level = level.reshape(-1, level.shape[-1]).amax(dim=0)
    return level


def has_speech(audio: torch.Tensor) -> bool:
    """False for an empty clip or one that never rises above the speech floor."""
    if audio is None or audio.numel() == 0:
        return False
    return bool(audio.abs().max().item() > _SPEECH_FLOOR)


# ── Room tone ───────────────────────────────────────────────────────────────

def _interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of ``fp`` (sampled at ascending ``xp``) at ``x``."""
    idx = torch.searchsorted(xp, x).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    t = ((x - x0) / (x1 - x0).clamp(min=1e-12)).clamp(0.0, 1.0)
    return y0 + (y1 - y0) * t


def pink_noise(n: int, sample_rate: int, rng: torch.Generator) -> torch.Tensor:
    """``n`` samples of unit-RMS pink (1/f) noise, high-passed below 30 Hz."""
    return shaped_noise(n, sample_rate, rng)


def shaped_noise(n: int, sample_rate: int, rng: torch.Generator,
                 psd: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``n`` samples of unit-RMS noise with the power spectrum ``psd``.

    ``psd`` is a power spectrum over ``rfftfreq(_ROOM_NFFT)`` bins (what
    :func:`room_from_reference` measures); without one the shape is pink
    (1/f). Everything below 30 Hz is removed either way.

    Shaped block by block in the frequency domain and overlap-added under a
    Hann window at 75 % overlap, where the squared windows sum to a constant:
    the bed's level stays flat across block seams instead of swelling once a
    second. Memory is one output buffer, not a whole-take FFT.
    """
    if n <= 0:
        return torch.zeros(0)
    block = min(_PINK_BLOCK, max(64, 1 << (n - 1).bit_length()))
    hop = block // 4
    window = torch.hann_window(block, periodic=True)
    freqs = torch.fft.rfftfreq(block, d=1.0 / sample_rate)
    if psd is None:
        weight = (30.0 / freqs.clamp(min=30.0)).sqrt()
    else:
        src = torch.fft.rfftfreq(_ROOM_NFFT, d=1.0 / sample_rate)
        weight = _interp(freqs, src, psd.to(torch.float32).clamp(min=0)).sqrt()
    weight = torch.where(freqs < 30.0, torch.zeros_like(weight), weight)
    # Pre-roll by (block - hop): the first sample handed back already has all
    # four windows under it, instead of the ramp-up a bare overlap-add starts with.
    lead = block - hop
    total = n + lead
    out = torch.zeros(total + block)
    pos = 0
    while pos < total:
        white = torch.randn(block, generator=rng)
        shaped = torch.fft.irfft(torch.fft.rfft(white) * weight, n=block)
        out[pos:pos + block] += shaped * window
        pos += hop
    bed = out[lead:lead + n]
    rms = bed.pow(2).mean().sqrt()
    return bed / rms if rms > 0 else bed


def add_room_tone(audio: torch.Tensor, sample_rate: int, level_db: float,
                  rng: torch.Generator, psd: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mix a noise bed at ``level_db`` dBFS RMS under the whole clip — pink,
    or shaped like ``psd`` when the reference's room was measured."""
    n = audio.shape[-1]
    if n == 0:
        return audio
    bed = shaped_noise(n, sample_rate, rng, psd) * (10.0 ** (level_db / 20.0))
    return audio + bed.to(device=audio.device, dtype=audio.dtype)


# ── Reference room ──────────────────────────────────────────────────────────

_ROOM_NFFT = 2048
_ANALYSIS_FRAME_MS = 20
#: The reference's room has to sit at least this far under its voice to be
#: worth copying; a noisier clip would plant hiss under every pause.
_MIN_ROOM_GAP_DB = 25.0
#: The copied room never comes closer to the take's voice than this.
_ROOM_GAP_RANGE_DB = (32.0, 72.0)
#: Below this RMS a "room" is digital silence or dither, not a microphone.
_ROOM_FLOOR_RMS = 10.0 ** (-90.0 / 20.0)
_MAX_REFERENCE_S = 60
_MAX_BREATH_SAMPLES = 8


@dataclass(frozen=True)
class RoomProfile:
    """What a reference clip's room and breathing sound like, measured at
    the take's sample rate.

    ``noise_psd`` is the room's power spectrum over ``rfftfreq(_ROOM_NFFT)``
    bins and ``noise_gap_db`` how far it sits under the voice; both are None
    when the clip has no usable room (too noisy, or digital silence). Each of
    ``breaths`` is ``(snippet, level_db)``: a peak-normalized inhale and its
    peak relative to the reference voice's peak.
    """

    sample_rate: int
    noise_psd: Optional[torch.Tensor] = None
    noise_gap_db: Optional[float] = None
    breaths: tuple = field(default_factory=tuple)

    @property
    def has_room(self) -> bool:
        return self.noise_psd is not None and self.noise_gap_db is not None


def _frame_db(x: torch.Tensor, frame: int) -> torch.Tensor:
    n = x.shape[-1] // frame * frame
    rms = x[:n].view(-1, frame).pow(2).mean(dim=1).sqrt()
    return 20.0 * torch.log10(rms.clamp(min=1e-10))


def _speech_db(x: torch.Tensor, sample_rate: int) -> float:
    """The voice's working level: the 95th percentile of 20 ms frame RMS."""
    frame = max(1, sample_rate * _ANALYSIS_FRAME_MS // 1000)
    db = _frame_db(x, frame)
    return float(torch.quantile(db, 0.95)) if db.numel() else -200.0


def _welch_psd(x: torch.Tensor) -> torch.Tensor:
    window = torch.hann_window(_ROOM_NFFT, periodic=True)
    spec = torch.stft(x, n_fft=_ROOM_NFFT, hop_length=_ROOM_NFFT // 2, window=window,
                      center=False, return_complex=True)
    return spec.abs().pow(2).mean(dim=-1)


def _voiced(seg: torch.Tensor, sample_rate: int) -> bool:
    """True when a segment carries a pitch (70-400 Hz): a vowel or a hum,
    not an inhale. Normalized autocorrelation on 40 ms windows."""
    win = max(64, sample_rate * 40 // 1000)
    lo, hi = sample_rate // 400, sample_rate // 70
    scores = []
    for start in range(0, max(1, seg.shape[-1] - win + 1), win // 2):
        w = seg[start:start + win]
        if w.shape[-1] < hi + 8:
            break
        w = w - w.mean()
        energy = float((w * w).sum())
        if energy <= 0:
            continue
        ac = torch.stack([(w[:-lag] * w[lag:]).sum() for lag in range(lo, hi)])
        scores.append(float(ac.max()) / energy)
    return bool(scores) and sorted(scores)[len(scores) // 2] > 0.45


def _centroid_hz(seg: torch.Tensor, sample_rate: int) -> float:
    mag = torch.fft.rfft(seg * torch.hann_window(seg.shape[-1], periodic=False)).abs()
    freqs = torch.fft.rfftfreq(seg.shape[-1], d=1.0 / sample_rate)
    total = float(mag.sum())
    return float((mag * freqs).sum()) / total if total > 0 else 0.0


def analyze_reference(audio: torch.Tensor, sample_rate: int) -> RoomProfile:
    """Measure a reference clip's room and pick out its inhales.

    Room: the quietest 20 ms frames (within 3 dB of the 10th percentile and
    at least 25 dB under the voice), averaged into a power spectrum.

    Breaths: runs of 140-700 ms that sit between the room and the voice
    (6 dB over the floor, 18 dB under the voice), follow a pause, lead into
    speech within 300 ms, carry no pitch and have their energy centred
    between 500 Hz and 3.5 kHz. That excludes vowels and hums (pitched), a
    word-initial "s" or "sh" (energy above 3.5 kHz), "h" (under 140 ms) and
    clicks.
    """
    x = audio.detach().to("cpu", torch.float32)
    if x.dim() > 1:
        x = x.reshape(-1, x.shape[-1]).mean(dim=0)
    x = x[: sample_rate * _MAX_REFERENCE_S]
    frame = max(1, sample_rate * _ANALYSIS_FRAME_MS // 1000)
    db = _frame_db(x, frame)
    if db.numel() < 20:
        return RoomProfile(sample_rate)
    speech = float(torch.quantile(db, 0.95))
    floor = float(torch.quantile(db, 0.10))

    psd = gap = None
    quiet = (db <= floor + 3.0) & (db <= speech - _MIN_ROOM_GAP_DB)
    if int(quiet.sum()) * frame >= _ROOM_NFFT * 2:
        frames = x[: db.numel() * frame].view(-1, frame)[quiet].reshape(-1)
        rms = float(frames.pow(2).mean().sqrt())
        if rms > _ROOM_FLOOR_RMS:
            psd = _welch_psd(frames)
            gap = speech - 20.0 * math.log10(rms)

    breaths = []
    peak = float(x.abs().max())
    cand = (db >= floor + 6.0) & (db <= speech - 18.0)
    loud = db >= speech - 15.0
    i, nf = 0, db.numel()
    while i < nf and len(breaths) < _MAX_BREATH_SAMPLES:
        if not bool(cand[i]):
            i += 1
            continue
        j = i
        while j < nf and bool(cand[j]):
            j += 1
        run_s = (j - i) * _ANALYSIS_FRAME_MS / 1000
        after_pause = i == 0 or bool((db[max(0, i - 3):i] <= floor + 6.0).any())
        into_speech = bool(loud[j:j + 15].any())
        if 0.14 <= run_s <= 0.7 and after_pause and into_speech and peak > 0:
            seg = x[i * frame:j * frame]
            if not _voiced(seg, sample_rate) and 500.0 <= _centroid_hz(seg, sample_rate) <= 3500.0:
                snippet = fade_edges(seg, sample_rate, ms=15)
                top = float(snippet.abs().max())
                if top > 0:
                    breaths.append((snippet / top, 20.0 * math.log10(top / peak)))
        i = j
    return RoomProfile(sample_rate, psd, gap, tuple(breaths))


_ROOM_CACHE: "OrderedDict[tuple, RoomProfile]" = OrderedDict()
_ROOM_CACHE_LOCK = threading.Lock()


def room_from_reference(path: Optional[str], sample_rate: int) -> Optional[RoomProfile]:
    """The :class:`RoomProfile` of the reference file at ``path``, or None.

    Cached by path, size and modification time, so a streamed or chunked take
    measures its reference once. Never raises: an unreadable reference just
    means the synthetic room and breaths are used.
    """
    if not path:
        return None
    try:
        st = os.stat(path)
        key = (os.path.abspath(path), st.st_size, st.st_mtime_ns, int(sample_rate))
    except OSError:
        return None
    with _ROOM_CACHE_LOCK:
        hit = _ROOM_CACHE.get(key)
        if hit is not None:
            _ROOM_CACHE.move_to_end(key)
            return hit
    try:
        from services.audio_io import load_audio  # noqa: PLC0415

        wav, sr = load_audio(path)
        wav = wav.to(torch.float32)
        if wav.dim() > 1:
            wav = wav.mean(dim=0)
        wav = wav[: sr * _MAX_REFERENCE_S]
        if sr != sample_rate:
            import torchaudio.functional as af  # noqa: PLC0415

            wav = af.resample(wav.unsqueeze(0), orig_freq=sr, new_freq=sample_rate).squeeze(0)
        room = analyze_reference(wav, sample_rate)
    except Exception:  # noqa: BLE001 — a reference we cannot read just means no match
        logger.debug("humanize: could not measure the reference room", exc_info=True)
        return None
    with _ROOM_CACHE_LOCK:
        _ROOM_CACHE[key] = room
        while len(_ROOM_CACHE) > 16:
            _ROOM_CACHE.popitem(last=False)
    return room


# ── Breaths ─────────────────────────────────────────────────────────────────

def silent_runs(audio: torch.Tensor, sample_rate: int, *,
                below_peak_db: float = SILENCE_BELOW_PEAK_DB) -> list[tuple[int, int]]:
    """``[(start, end)]`` sample ranges (end exclusive) where the clip sits
    ``below_peak_db`` under its own peak, measured on 5 ms frames."""
    n = audio.shape[-1]
    if n == 0:
        return []
    level = _mono_level(audio)
    peak = level.max()
    if peak <= 0:
        return [(0, n)]
    frame = max(1, sample_rate * _FRAME_MS // 1000)
    pad = (-n) % frame
    if pad:
        level = torch.nn.functional.pad(level, (0, pad))
    peaks = level.view(-1, frame).amax(dim=1)
    quiet = (peaks <= peak * 10.0 ** (-below_peak_db / 20.0)).to(torch.int8)
    edges = torch.diff(quiet, prepend=quiet.new_zeros(1), append=quiet.new_zeros(1))
    starts = torch.nonzero(edges == 1).flatten().tolist()
    ends = torch.nonzero(edges == -1).flatten().tolist()
    return [(s * frame, min(n, e * frame)) for s, e in zip(starts, ends)]


def synth_breath(sample_rate: int, duration_s: float, rng: torch.Generator) -> torch.Tensor:
    """One inhale: noise band-limited around 1.4 kHz under a slow-rise,
    faster-fall envelope. Peak-normalized to 1.0; the caller sets the level."""
    n = max(16, int(sample_rate * duration_s))
    white = torch.randn(n, generator=rng)
    freqs = torch.fft.rfftfreq(n, d=1.0 / sample_rate)
    octaves = torch.log2(freqs.clamp(min=1.0))
    mask = torch.exp(-0.5 * ((octaves - math.log2(1400.0)) / 1.1) ** 2)
    mask = torch.where(freqs < 150.0, torch.zeros_like(mask), mask)
    shaped = torch.fft.irfft(torch.fft.rfft(white) * mask, n=n)
    t = torch.linspace(0.0, 1.0, n)
    rise = 0.6
    env = torch.where(
        t < rise,
        0.5 - 0.5 * torch.cos(math.pi * t / rise),
        0.5 + 0.5 * torch.cos(math.pi * (t - rise) / (1.0 - rise)),
    ) ** 1.5
    breath = shaped * env
    peak = breath.abs().max()
    return breath / peak if peak > 0 else breath


def add_breaths(audio: torch.Tensor, sample_rate: int, strength: float,
                rng: torch.Generator, samples: Sequence = ()) -> tuple[torch.Tensor, int]:
    """Place breaths before speech that follows a long enough silence.

    Returns the mixed clip and how many breaths were placed. The trailing
    silence never gets one — nobody inhales after the last word — and each
    eligible gap is taken with a probability that rises with ``strength``.

    ``samples`` are the speaker's own inhales from the reference
    (:attr:`RoomProfile.breaths`): each placement picks one and keeps its
    level relative to the voice, nudged by ``strength`` (-4 dB at 0, +4 dB
    at 1). Without samples a synthesized inhale peaks 38 dB (strength 0) to
    28 dB (strength 1) under the clip's peak, where a close-mic breath lands
    in a real recording.
    """
    n = audio.shape[-1]
    peak = float(audio.abs().max().item()) if n else 0.0
    if peak <= 0:
        return audio, 0
    min_gap = sample_rate * MIN_BREATH_GAP_MS // 1000
    lead = sample_rate * _BREATH_LEAD_MS // 1000
    level = peak * 10.0 ** ((-38.0 + 10.0 * strength) / 20.0)
    nudge = 10.0 ** ((8.0 * strength - 4.0) / 20.0)
    chance = 0.45 + 0.5 * strength
    out = audio.clone()
    placed = 0
    for start, end in silent_runs(audio, sample_rate):
        if end >= n or end - start < min_gap:
            continue
        if _uniform(rng) > chance:
            continue
        if samples:
            snippet, rel_db = samples[int(_uniform(rng) * len(samples)) % len(samples)]
            breath = snippet * (peak * 10.0 ** (rel_db / 20.0) * nudge)
        else:
            gap_s = (end - start) / sample_rate
            duration = min(_BREATH_MAX_S, max(_BREATH_MIN_S, 0.45 * gap_s)) * (0.85 + 0.3 * _uniform(rng))
            breath = synth_breath(sample_rate, duration, rng) * (level * (0.8 + 0.4 * _uniform(rng)))
        b_end = end - lead
        b_start = max(start, b_end - breath.shape[-1])
        if b_end - b_start < 16:
            continue
        breath = breath[-(b_end - b_start):].to(device=out.device, dtype=out.dtype)
        out[..., b_start:b_end] += breath
        placed += 1
    return out, placed


# ── Per-chunk pacing ────────────────────────────────────────────────────────

def drift_ratio(rng: torch.Generator, strength: float) -> float:
    """A tempo/pitch factor near 1: ±0.4 % at strength 0, ±1.6 % at 1."""
    spread = 0.004 + 0.012 * strength
    return 1.0 + (2.0 * _uniform(rng) - 1.0) * spread


def vary_tempo(audio: torch.Tensor, sample_rate: int, ratio: float) -> torch.Tensor:
    """Resample by ``ratio`` (>1 = slower and lower, like a tape running
    slow). Windowed-sinc through torchaudio; linear interpolation if that is
    unavailable. The sample rate is unchanged — only the clip's length is."""
    n = audio.shape[-1]
    if n < 16 or abs(ratio - 1.0) < 1e-4:
        return audio
    squeeze = audio.dim() == 1
    x = audio.unsqueeze(0) if squeeze else audio.reshape(-1, n)
    work = x.to(torch.float32)
    try:
        import torchaudio.functional as af  # noqa: PLC0415 — optional at runtime
        orig = 10_000
        new = max(1, int(round(orig * ratio)))
        y = af.resample(work, orig_freq=orig, new_freq=new)
    except Exception:  # noqa: BLE001 — a missing/failed resampler must not fail the take
        y = torch.nn.functional.interpolate(
            work.unsqueeze(0), size=max(16, int(round(n * ratio))), mode="linear",
            align_corners=False,
        ).squeeze(0)
    y = y.to(dtype=audio.dtype)
    if squeeze:
        return y.squeeze(0)
    return y.reshape(*audio.shape[:-1], y.shape[-1])


def fade_edges(audio: torch.Tensor, sample_rate: int, ms: int = _EDGE_FADE_MS) -> torch.Tensor:
    """Linear fade-in/out over ``ms`` so a trimmed chunk joins without a click."""
    n = audio.shape[-1]
    k = min(n // 2, sample_rate * ms // 1000)
    if k <= 0:
        return audio
    out = audio.clone()
    ramp = torch.linspace(0.0, 1.0, k, dtype=out.dtype, device=out.device)
    out[..., :k] *= ramp
    out[..., n - k:] *= ramp.flip(0)
    return out


def boundary_gap_ms(prev_text: str, rng: torch.Generator, strength: float) -> int:
    """How long to wait after ``prev_text`` before the next chunk starts.

    Sized by what ended the chunk — a paragraph break, a trailing-off ellipsis
    or dash, a sentence end, a clause break, or nothing — then jittered by
    ±10 % (strength 0) to ±35 % (strength 1) so no two boundaries match.
    """
    text = prev_text or ""
    body = text.rstrip()
    trailing = text[len(body):]
    body = body.rstrip(_CLOSERS)
    last = body[-1:] if body else ""
    if "\n\n" in trailing:
        base = 850
    elif body.endswith("...") or body.endswith("--") or (last and last in _TRAIL_OFF):
        base = 650
    elif last and last in _SENTENCE_END:
        base = 420
    elif last and last in _CLAUSE_END:
        base = 220
    else:
        base = 160
    jitter = 0.10 + 0.25 * strength
    return max(0, int(round(base * (1.0 + (2.0 * _uniform(rng) - 1.0) * jitter))))


def prepare_span(audio: torch.Tensor, sample_rate: int, settings: HumanizeSettings,
                 rng: torch.Generator) -> torch.Tensor:
    """Trim the engine's own padding, fade the edges and drift the tempo of
    one rendered span. Used for both chunk joins and ``[pause]`` spans."""
    if audio is None or audio.shape[-1] == 0:
        return audio
    from services.chunked_tts import trim_edge_silence  # noqa: PLC0415

    x = fade_edges(trim_edge_silence(audio, sample_rate), sample_rate)
    if settings.drift:
        x = vary_tempo(x, sample_rate, drift_ratio(rng, settings.strength))
    return x


def prepare_chunks(parts: Sequence[Optional[torch.Tensor]], texts: Optional[Sequence[str]],
                   sample_rate: int, settings: HumanizeSettings,
                   rng: torch.Generator) -> list[Optional[torch.Tensor]]:
    """Humanize each rendered chunk ahead of a hard join.

    Every chunk but the last carries its punctuation-sized gap as trailing
    silence, so the caller joins with ``crossfade_ms=0`` and the list keeps
    one entry per chunk — ``None``/empty entries pass through untouched, so
    dropped-chunk reporting in the join still sees them (#1330).
    """
    out: list[Optional[torch.Tensor]] = []
    last = len(parts) - 1
    for i, part in enumerate(parts):
        if part is None or part.shape[-1] == 0:
            out.append(part)
            continue
        x = prepare_span(part, sample_rate, settings, rng)
        if settings.gaps and i < last:
            text = texts[i] if texts is not None and i < len(texts) else ""
            gap = sample_rate * boundary_gap_ms(text, rng, settings.strength) // 1000
            if gap > 0:
                x = torch.cat([x, torch.zeros(*x.shape[:-1], gap, dtype=x.dtype, device=x.device)], dim=-1)
        out.append(x)
    return out


def span_preparer(sample_rate: int, settings: HumanizeSettings,
                  rng: torch.Generator) -> Callable[[torch.Tensor, int], torch.Tensor]:
    """``prepare(audio, index)`` for ``_render_with_pauses``: spans are
    trimmed, faded and drifted; the ``[pause]`` silences between them stay
    exactly as long as the user wrote."""
    def prepare(audio: torch.Tensor, _index: int) -> torch.Tensor:
        return prepare_span(audio, sample_rate, settings, rng)
    return prepare


# ── Whole take ──────────────────────────────────────────────────────────────

def humanize_audio(audio: torch.Tensor, sample_rate: int, settings: HumanizeSettings,
                   rng: torch.Generator, room: Optional[RoomProfile] = None) -> torch.Tensor:
    """Breaths into the long silences, then the room-tone bed under it all.

    With a ``room`` measured from the reference clip the breaths are the
    speaker's own and the bed copies the reference's noise spectrum at the
    same distance under the voice (nudged by strength: 5 dB quieter at 0,
    4 dB louder at 1). Runs on the joined take before mastering so the bed
    and the breaths get the same EQ and compression as the voice. A clip
    with no speech is returned as-is (see the module docstring).
    """
    if not torch.is_tensor(audio) or not has_speech(audio):
        return audio
    work = audio if audio.is_floating_point() else audio.to(torch.float32)
    if room is not None and room.sample_rate != sample_rate:
        room = None
    if settings.breaths:
        work, _ = add_breaths(work, sample_rate, settings.strength, rng,
                              samples=room.breaths if room is not None else ())
    if room is not None and room.has_room:
        lo, hi = _ROOM_GAP_RANGE_DB
        gap = min(hi, max(lo, room.noise_gap_db + 8.0 * (0.6 - settings.strength)))
        mono = work.reshape(-1, work.shape[-1]).mean(dim=0).detach().to("cpu", torch.float32)
        level_db = _speech_db(mono, sample_rate) - gap
        work = add_room_tone(work, sample_rate, level_db, rng, psd=room.noise_psd)
    else:
        work = add_room_tone(work, sample_rate, settings.room_tone_db, rng)
    return work
