"""Sound effects — text → SFX clips (ElevenLabs Sound Effects parity).

``GET /sfx/status`` reports whether the model is ready, ``POST /sfx/install``
starts the explicit, user-requested install (diffusers + AudioLDM2 weights),
and ``POST /sfx/generate`` renders 1–4 variations. Every clip goes through
the ``mark_synthetic`` provenance chokepoint and the shared
``_finalize_generation`` tail (WAV in OUTPUTS_DIR, history row, retention),
so it is served from ``/audio`` and listed in history like any other take.
"""

from __future__ import annotations

import functools
import random
import time

from fastapi import APIRouter, Form, HTTPException

from services import sound_effects as sfx

router = APIRouter()


@router.get("/sfx/status")
def sfx_status():
    from services.license_guard import tool_blocked

    return {**sfx.status(), "commercial_blocked": tool_blocked("sfx")}


@router.post("/sfx/install")
def sfx_install():
    from services.license_guard import check_tool

    check_tool("sfx")  # Commercial-safe mode: no multi-GB download of weights it would refuse
    sfx.start_install()
    return sfx_status()


@router.post("/sfx/generate")
async def sfx_generate(
    prompt: str = Form(..., min_length=1, max_length=500),
    seconds: float = Form(5.0),
    variations: int = Form(4, ge=1, le=sfx.MAX_VARIATIONS),
    seed: int | None = Form(None),
):
    from services.license_guard import check_tool

    check_tool("sfx")  # Commercial-safe mode (opt-in): AudioLDM2 weights are CC-BY-NC-SA
    prompt = prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Describe the sound to generate.")
    if not sfx.status()["ready"]:
        raise HTTPException(status_code=409, detail={
            "error": "sfx_not_installed",
            "message": "Sound effects need a one-time model install.",
            "download_bytes": sfx.SFX_DOWNLOAD_BYTES,
        })

    from api.routers.generation import _finalize_generation
    from services.model_manager import (
        GpuJobTimeoutError, GpuPoolBusyError, run_on_gpu_pool_guarded,
    )
    from services.watermark import mark_synthetic_async

    from services.engine_memory import evict_for_tool

    # A speech engine or the other tool may still hold the GPU this model needs.
    await evict_for_tool(sfx.MODEL.tool_id)
    used_seed = seed if seed is not None else random.randint(0, 2**31 - 1)
    seconds = sfx.clamp_seconds(seconds)
    start_time = time.time()
    try:
        clips = await run_on_gpu_pool_guarded(
            functools.partial(sfx.generate, prompt, seconds, variations, used_seed),
            what="Sound effects",
            # Load (first run) + 100 diffusion steps per batch, with headroom.
            timeout=300.0 + 20.0 * variations,
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
    for index, clip in enumerate(clips):
        clip = await mark_synthetic_async(clip, sfx.SFX_SAMPLE_RATE, context="sfx.finalize")
        _, meta = await _finalize_generation(
            clip, sfx.SFX_SAMPLE_RATE, text=prompt, history_mode="sfx",
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
