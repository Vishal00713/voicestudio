"""Music — text → music clips (ElevenLabs Music parity).

``GET /music/status`` reports readiness, ``POST /music/install`` starts the
explicit, user-requested model download, and ``POST /music/generate`` renders
1–3 stereo pieces. AudioSeal is a mono model, so each channel is marked on its
own through the ``mark_synthetic`` chokepoint before the shared
``_finalize_generation`` tail (WAV in OUTPUTS_DIR, history row mode ``music``).
"""

from __future__ import annotations

import functools
import random
import time

import torch
from fastapi import APIRouter, Form, HTTPException

from services import music_gen

router = APIRouter()


@router.get("/music/status")
def music_status():
    from services.license_guard import tool_blocked

    return {**music_gen.status(), "commercial_blocked": tool_blocked("music")}


@router.post("/music/install")
def music_install():
    from services.license_guard import check_tool

    check_tool("music")  # Commercial-safe mode: no multi-GB download of weights it would refuse
    music_gen.start_install()
    return music_status()


async def _mark_channels(piece, sample_rate):
    from services.watermark import mark_synthetic_async

    channels = [
        await mark_synthetic_async(channel.unsqueeze(0), sample_rate, context="music.finalize")
        for channel in piece
    ]
    return torch.cat(channels, dim=0)


@router.post("/music/generate")
async def music_generate(
    prompt: str = Form(..., min_length=1, max_length=500),
    seconds: float = Form(15.0),
    variations: int = Form(1, ge=1, le=music_gen.MAX_VARIATIONS),
    seed: int | None = Form(None),
):
    from services.license_guard import check_tool

    check_tool("music")  # Commercial-safe mode (opt-in): MusicGen weights are CC-BY-NC
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Describe the music to generate.")
    if not music_gen.status()["ready"]:
        raise HTTPException(status_code=409, detail={
            "error": "music_not_installed",
            "message": "Music needs a one-time model download.",
            "download_bytes": music_gen.MUSIC_DOWNLOAD_BYTES,
        })

    from api.routers.generation import _finalize_generation
    from services.model_manager import (
        GpuJobTimeoutError, GpuPoolBusyError, run_on_gpu_pool_guarded,
    )

    from services.engine_memory import evict_for_tool

    # A speech engine or the other tool may still hold the GPU this model needs.
    await evict_for_tool(music_gen.MODEL.tool_id)
    used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    seconds = music_gen.clamp_seconds(seconds)
    start_time = time.time()
    try:
        pieces, sample_rate = await run_on_gpu_pool_guarded(
            functools.partial(music_gen.generate, prompt, seconds, variations, used_seed),
            what="Music",
            # First-run load + ~1 s of audio per second of GPU time, with headroom.
            timeout=300.0 + 10.0 * seconds * variations,
        )
    except GpuPoolBusyError as e:
        raise HTTPException(status_code=503, detail=str(e), headers={
            "Retry-After": str(e.retry_after), "X-OmniVoice-Retryable": "true",
        }) from e
    except GpuJobTimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e), headers={
            "Retry-After": "30", "X-OmniVoice-Retryable": "true",
        }) from e

    takes = []
    for index, piece in enumerate(pieces):
        piece = await _mark_channels(piece, sample_rate)
        _, meta = await _finalize_generation(
            piece, sample_rate, text=prompt, history_mode="music",
            ref_audio_path=None, language=None, instruct=None,
            resolved_profile_id=None, used_seed=used_seed + index,
            start_time=start_time, already_marked=True,
        )
        takes.append({
            "id": meta["id"],
            "audio_url": f"/audio/{meta['filename']}",
            "duration_s": meta["duration"],
        })
    return {
        "prompt": prompt,
        "seed": used_seed,
        "gen_time_s": round(time.time() - start_time, 2),
        "takes": takes,
    }
