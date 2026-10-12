"""Hindi realism eval: render a fixed Hindi set through a running backend and
measure what a listener notices — mispronounced words, dead-silent pauses.

Each line of ``scripts/eval/hindi_eval_set.json`` is rendered by ``POST
/generate`` on a running VoiceStudio backend (the app, or ``bun run dev:api``),
then transcribed back with the locally installed faster-whisper model. The
report gives, per line and per category:

* **CER** — character error rate of the transcript against the text the
  engine was asked to say *after* Hindi normalization (so "₹15" is scored as
  "पंद्रह रुपये"). Spaces, punctuation, nukta and chandrabindu/anusvara
  spelling variants are ignored: they are ASR spelling choices, not
  pronunciation errors. Lower is better; it measures intelligibility and
  pronunciation, not naturalness.
* **longest_zero_ms** — the longest run of exact digital silence. Anything
  above a few milliseconds is a gap no microphone would record.
* **pause_floor_dbfs** — RMS of the quiet frames, i.e. what the pauses sound
  like; ``null`` when the take has no pause to measure.
* **runaway** — the take is far longer than its text takes to read (see
  ``services/runaway_guard.py``): the engine kept talking past the line.

Speaker similarity and MOS predictors need extra model downloads and are not
part of this script. Compare runs by ear as well: the WAVs are kept.

Run with the app's own Python runtime (it has faster-whisper and torch), with
the backend already running::

    python scripts/eval_hindi.py --output eval-runs/doc-humanize \\
        --profile-id <voice id> --humanize

    python scripts/eval_hindi.py --output eval-runs/design-plain \\
        --instruct "male, middle-aged, very low pitch"

Offline: the only network call is to the local backend. The output directory
must not exist yet.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SET = ROOT / "scripts" / "eval" / "hindi_eval_set.json"
DEFAULT_ASR = "deepdml/faster-whisper-large-v3-turbo-ct2"

_TAG_RE = re.compile(r"\[[^\]]*\]")
_PUNCT_RE = re.compile(r"[\s।॥.,!?;:'\"“”‘’…\-–—()]+")


def _multipart(fields: dict[str, str]) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'
        )
    parts.append(f"--{boundary}--\r\n")
    return "".join(parts).encode("utf-8"), f"multipart/form-data; boundary={boundary}"


def render(base_url: str, fields: dict[str, str], timeout: float) -> tuple[bytes, float, str]:
    """One take: ``(wav bytes, seconds, history id)``."""
    body, ctype = _multipart(fields)
    req = urllib.request.Request(f"{base_url.rstrip('/')}/generate", data=body,
                                 headers={"Content-Type": ctype}, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
            take_id = resp.headers.get("X-Audio-Id", "")
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode("utf-8", "replace")
        raise RuntimeError(f"/generate returned {exc.code}: {detail}") from exc
    if data[:4] != b"RIFF":
        raise RuntimeError(f"/generate did not return a WAV: {data[:200]!r}")
    return data, time.perf_counter() - t0, take_id


def forget(base_url: str, take_id: str) -> None:
    """Remove an eval take from the app's history; the WAV copy here is kept."""
    if not take_id:
        return
    req = urllib.request.Request(f"{base_url.rstrip('/')}/history/{take_id}", method="DELETE")
    try:
        urllib.request.urlopen(req, timeout=30).close()
    except urllib.error.URLError as exc:
        print(f"  could not remove take {take_id} from history: {exc}", file=sys.stderr)


_DIGIT_WORDS = ["शून्य", "एक", "दो", "तीन", "चार", "पाँच", "छह", "सात", "आठ", "नौ"]


def spell_digits(text: str) -> str:
    """Digits Whisper wrote for a line spoken digit by digit ("9-8-7-6-5"),
    which normalization leaves alone, as the Hindi words the voice said."""
    return re.sub(r"\d", lambda m: f" {_DIGIT_WORDS[int(m.group(0))]} ", text)


def scoring_form(text: str) -> str:
    """Comparable characters only: no tags, punctuation, spaces or spelling variants."""
    s = unicodedata.normalize("NFC", _TAG_RE.sub(" ", text))
    s = s.replace("़", "")            # nukta: ज़ / ज
    s = s.replace("ँ", "ं")      # chandrabindu → anusvara: पाँच / पांच
    return _PUNCT_RE.sub("", s).lower()


def cer(ref: str, hyp: str) -> float:
    """Levenshtein distance over characters, divided by the reference length."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i]
        for j, hc in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / len(ref)


def silence_stats(path: Path) -> dict:
    import numpy as np
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    zero = (x == 0).astype(np.int8)
    longest = run = 0
    for z in zero:
        run = run + 1 if z else 0
        longest = max(longest, run)
    frame = max(1, sr // 20)
    n = len(x) // frame * frame
    frames = x[:n].reshape(-1, frame)
    peaks = np.abs(frames).max(axis=1)
    quiet = frames[peaks < peaks.max() * 10 ** (-40 / 20)] if n else frames[:0]
    if not quiet.size:
        floor = None                      # no pause in the take
    else:
        rms = float(np.sqrt(np.mean(quiet ** 2)))
        floor = round(20 * math.log10(rms), 1) if rms > 0 else -200.0  # -200: dead silence
    return {
        "duration_s": round(len(x) / sr, 2),
        "longest_zero_ms": round(longest / sr * 1000, 1),
        "pause_floor_dbfs": floor,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True,
                    help="new directory for WAVs and the report (an existing one with --score-only)")
    ap.add_argument("--base-url", default=f"http://127.0.0.1:{os.environ.get('OMNIVOICE_PORT', '3900')}")
    ap.add_argument("--set", default=str(DEFAULT_SET), help="eval set JSON")
    voice = ap.add_mutually_exclusive_group()
    voice.add_argument("--profile-id", help="saved voice profile to use")
    voice.add_argument("--instruct", help="voice-design instruct, e.g. 'female, young adult, high pitch'")
    ap.add_argument("--engine", help="TTS engine id; default is the app's active engine")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-step", type=int, default=32)
    ap.add_argument("--humanize", action="store_true", help="turn on Sound more human")
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--asr-model", default=DEFAULT_ASR, help="installed faster-whisper model")
    ap.add_argument("--device", default="cpu",
                    help="ASR device. cpu works everywhere; cuda needs the cuDNN build "
                         "your faster-whisper/ctranslate2 expects, and a mismatch aborts "
                         "the process without a Python error")
    ap.add_argument("--only", nargs="*", help="item ids or categories to run")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds to wait for one take; a busy GPU can need many minutes")
    ap.add_argument("--score-only", action="store_true",
                    help="render nothing: score the WAVs already in --output again")
    ap.add_argument("--client-normalize", action="store_true",
                    help="normalize the text with this checkout before sending it, to score "
                         "normalization changes against a backend still running older code")
    ap.add_argument("--keep-history", action="store_true",
                    help="leave the eval takes in the app's history (removed by default)")
    args = ap.parse_args()
    if not args.score_only and not (args.profile_id or args.instruct):
        ap.error("one of --profile-id or --instruct is required unless --score-only")

    sys.path.insert(0, str(ROOT / "backend"))
    from services.runaway_guard import expected_seconds, is_runaway  # noqa: PLC0415
    from services.text_normalization import normalize_text  # noqa: PLC0415

    spec = json.loads(Path(args.set).read_text(encoding="utf-8"))
    language = spec.get("language", "Hindi")
    items = spec["items"]
    if args.only:
        wanted = set(args.only)
        items = [it for it in items if it["id"] in wanted or it["category"] in wanted]
    out = Path(args.output).resolve()
    if args.score_only:
        if not out.is_dir():
            ap.error(f"--score-only needs an existing --output directory: {out}")
    else:
        out.mkdir(parents=True, exist_ok=False)

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from faster_whisper import WhisperModel  # noqa: PLC0415

    device = args.device
    asr = WhisperModel(args.asr_model, device=device,
                       compute_type="float16" if device == "cuda" else "int8")

    rows, failed = [], []
    for it in items:
        path = out / f"{it['id']}.wav"
        secs = None
        if args.score_only:
            if not path.is_file():
                continue
        else:
            try:
                secs = render_item(args, it, language, normalize_text, path)
            except Exception as exc:  # noqa: BLE001 — record and keep going
                failed.append({"id": it["id"], "error": str(exc)[:300]})
                print(f"{it['id']:8s} FAILED: {str(exc)[:160]}", flush=True)
                continue
        segments, _ = asr.transcribe(str(path), language="hi", beam_size=5)

        hyp_raw = " ".join(s.text.strip() for s in segments).strip()
        spoken = normalize_text(it["text"], language)
        ref = scoring_form(spoken)
        hyp = scoring_form(spell_digits(normalize_text(hyp_raw, language)))
        row = {
            "id": it["id"], "category": it["category"], "text": it["text"],
            "spoken": spoken, "transcript": hyp_raw,
            "cer": round(cer(ref, hyp), 4), "render_s": None if secs is None else round(secs, 2),
            **silence_stats(path),
        }
        row["expected_s"] = round(expected_seconds(spoken, language), 2)
        row["runaway"] = is_runaway(row["duration_s"], row["expected_s"])
        rows.append(row)
        floor = "n/a" if row["pause_floor_dbfs"] is None else f"{row['pause_floor_dbfs']} dBFS"
        print(f"{row['id']:8s} CER {row['cer']:.3f}  zero {row['longest_zero_ms']:7.1f} ms  "
              f"floor {floor:>11}  {row['duration_s']:5.1f}s"
              f"{'  RUNAWAY' if row['runaway'] else ''}", flush=True)

    by_cat: dict[str, list[float]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r["cer"])
    summary = {
        "voice": args.profile_id or args.instruct, "engine": args.engine or "active",
        "seed": args.seed, "num_step": args.num_step,
        "client_normalize": args.client_normalize,
        "humanize": args.strength if args.humanize else None,
        "cer_mean": round(statistics.mean(r["cer"] for r in rows), 4) if rows else None,
        "cer_by_category": {k: round(statistics.mean(v), 4) for k, v in sorted(by_cat.items())},
        "longest_zero_ms_max": max((r["longest_zero_ms"] for r in rows), default=None),
        "runaway": [r["id"] for r in rows if r["runaway"]],
        "failed": failed,
    }
    (out / "results.json").write_text(
        json.dumps({"summary": summary, "items": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    lines = ["| Category | CER |", "|---|---:|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in summary["cer_by_category"].items()]
    if summary["cer_mean"] is not None:
        lines.append(f"| **all** | **{summary['cer_mean']:.3f}** |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"longest exact-zero run across the set: {summary['longest_zero_ms_max']} ms")
    print(f"runaway takes: {', '.join(summary['runaway']) or 'none'}")
    if failed:
        print(f"failed renders: {', '.join(f['id'] for f in failed)}")
    return 1 if failed else 0


def render_item(args, it: dict, language: str, normalize_text, path: Path) -> float:
    """Render one eval line into ``path``; returns the render time."""
    sent = normalize_text(it["text"], language) if args.client_normalize else it["text"]
    fields = {"text": sent, "language": language, "seed": str(args.seed),
              "num_step": str(args.num_step)}
    if args.profile_id:
        fields["profile_id"] = args.profile_id
    else:
        fields["instruct"] = args.instruct
    if args.engine:
        fields["engine"] = args.engine
    if args.humanize:
        fields.update(humanize="true", humanize_strength=str(args.strength))
    wav, secs, take_id = render(args.base_url, fields, args.timeout)
    path.write_bytes(wav)
    if not args.keep_history:
        forget(args.base_url, take_id)
    return secs


if __name__ == "__main__":
    sys.exit(main())
