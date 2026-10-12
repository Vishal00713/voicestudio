"""Text → music (ElevenLabs Music parity), fully local.

Model: MusicGen stereo-medium (``facebook/musicgen-stereo-medium``,
CC-BY-NC-4.0 — non-commercial), which ``transformers`` already ships, so the
only thing to fetch is the weights — and only on an explicit user action
(``start_install``), as the local-first rule requires. The download is
limited to the safetensors + configs (the repo also carries ``.bin`` copies).

Generation runs on the shared GPU pool; output is 32 kHz stereo.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from services.tool_models import ResidentModel

logger = logging.getLogger("omnivoice.music")

MUSIC_REPO = "facebook/musicgen-stereo-medium"
MUSIC_LICENSE = "CC-BY-NC-4.0"
MUSIC_ALLOW_PATTERNS = [
    "config.json", "generation_config.json", "model.safetensors",
    "preprocessor_config.json", "special_tokens_map.json", "spiece.model",
    "tokenizer.json", "tokenizer_config.json",
]
MUSIC_DOWNLOAD_BYTES = 4_080_000_000
MIN_SECONDS = 5.0
MAX_SECONDS = 30.0  # the model's training window
MAX_VARIATIONS = 3
GUIDANCE_SCALE = 3.0

_state_lock = threading.Lock()
_install: dict[str, Any] = {"stage": "idle", "error": None}


def model_downloaded() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MUSIC_REPO, allow_patterns=MUSIC_ALLOW_PATTERNS, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — anything short of a full snapshot is "not yet"
        return False


def status() -> dict[str, Any]:
    with _state_lock:
        install = dict(_install)
    ready = model_downloaded()
    if ready and install["stage"] in ("idle", "done"):
        install["stage"] = "ready"
    return {
        "ready": ready,
        "stage": install["stage"],
        "error": install["error"],
        "model": MUSIC_REPO,
        "license": MUSIC_LICENSE,
        "download_bytes": MUSIC_DOWNLOAD_BYTES,
        "min_seconds": MIN_SECONDS,
        "max_seconds": MAX_SECONDS,
        "max_variations": MAX_VARIATIONS,
    }


def _run_install() -> None:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(MUSIC_REPO, allow_patterns=MUSIC_ALLOW_PATTERNS)
        with _state_lock:
            _install.update(stage="done", error=None)
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never raised
        logger.exception("Music model download failed")
        with _state_lock:
            _install.update(stage="error", error=str(exc))


def start_install() -> dict[str, Any]:
    """Begin the user-requested download in the background (idempotent)."""
    with _state_lock:
        busy = _install["stage"] == "downloading"
        if not busy:
            _install.update(stage="downloading", error=None)
    if not busy:
        threading.Thread(target=_run_install, name="music-install", daemon=True).start()
    return status()


def _load():
    import torch
    from transformers import AutoProcessor, MusicgenForConditionalGeneration

    from services.model_loading import exclusive_model_load

    cuda = torch.cuda.is_available()
    with exclusive_model_load("Music"):
        processor = AutoProcessor.from_pretrained(MUSIC_REPO, local_files_only=True)
        model = MusicgenForConditionalGeneration.from_pretrained(
            MUSIC_REPO,
            dtype=torch.float16 if cuda else torch.float32,
            local_files_only=True,
        ).to("cuda" if cuda else "cpu").eval()
    return model, processor


#: Freed by the single-engine policy when speech or the other tool needs the GPU.
MODEL = ResidentModel("music", _load)


def clamp_seconds(seconds: float) -> float:
    return max(MIN_SECONDS, min(MAX_SECONDS, float(seconds)))


def generate(prompt: str, seconds: float, variations: int, seed: int) -> tuple[list, int]:
    """Render ``variations`` pieces; returns ``([C, N] float tensors, sample_rate)``."""
    with MODEL.use() as (model, processor):
        return _render(model, processor, prompt, seconds, variations, seed)


def _render(model, processor, prompt, seconds, variations, seed):
    import torch

    count = max(1, min(MAX_VARIATIONS, int(variations)))
    inputs = processor(text=[prompt] * count, padding=True, return_tensors="pt").to(model.device)
    frame_rate = int(model.config.audio_encoder.frame_rate)
    # MusicGen's sampler takes no generator, so the global RNG carries the seed.
    # manual_seed seeds every device (CPU and CUDA); the tool render lock in
    # services.tool_models keeps the other tool from reseeding it mid-render.
    torch.manual_seed(int(seed))
    with torch.inference_mode():
        audio = model.generate(
            **inputs,
            do_sample=True,
            guidance_scale=GUIDANCE_SCALE,
            max_new_tokens=int(clamp_seconds(seconds) * frame_rate),
        )
    pieces = []
    for piece in audio.float().cpu():
        peak = float(piece.abs().max()) if piece.numel() else 0.0
        if peak > 0.99:
            piece = piece * (0.99 / peak)
        pieces.append(piece)
    return pieces, int(model.config.audio_encoder.sampling_rate)
