"""Text → sound effects (ElevenLabs "Sound Effects" parity), fully local.

Model: AudioLDM2 (``cvssp/audioldm2``, CC-BY-NC-SA-4.0 — non-commercial),
run through ``diffusers``. Neither ships with the app, so both arrive only on
an explicit user action (``start_install``), as the local-first rule
requires:

* ``diffusers`` is pure Python and installs ``--no-deps`` into its own folder
  under the data dir (``sfx_libs``) — the app's runtime venv is never
  modified, and deleting that folder uninstalls it.
* the weights download into the shared Hugging Face cache (safetensors and
  configs only — the repo also carries duplicate ``.bin`` weights).

Generation runs on the shared GPU pool like every other model.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import threading
import types
from typing import Any

from core.config import DATA_DIR
from services.tool_models import ResidentModel

logger = logging.getLogger("omnivoice.sfx")

SFX_REPO = "cvssp/audioldm2"
SFX_LICENSE = "CC-BY-NC-SA-4.0"
SFX_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model"]
SFX_DOWNLOAD_BYTES = 4_480_000_000
SFX_SAMPLE_RATE = 16_000
DIFFUSERS_SPEC = "diffusers==0.40.0"
LIBS_DIR = os.path.join(DATA_DIR, "sfx_libs")

MIN_SECONDS = 1.0
MAX_SECONDS = 20.0
MAX_VARIATIONS = 4
DEFAULT_STEPS = 100
NEGATIVE_PROMPT = "low quality, distorted, noise, muffled"

_state_lock = threading.Lock()
_install: dict[str, Any] = {"stage": "idle", "error": None}


def _use_libs_dir() -> None:
    # Appended, never prepended: the runtime's own packages always win.
    if os.path.isdir(LIBS_DIR) and LIBS_DIR not in sys.path:
        sys.path.append(LIBS_DIR)


def runtime_installed() -> bool:
    _use_libs_dir()
    try:
        import importlib.util

        return importlib.util.find_spec("diffusers") is not None
    except Exception:  # noqa: BLE001 — a broken partial install reads as missing
        return False


def model_downloaded() -> bool:
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(SFX_REPO, allow_patterns=SFX_ALLOW_PATTERNS, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001 — anything short of a full snapshot is "not yet"
        return False


def status() -> dict[str, Any]:
    with _state_lock:
        install = dict(_install)
    ready = runtime_installed() and model_downloaded()
    if ready and install["stage"] in ("idle", "done"):
        install["stage"] = "ready"
    return {
        "ready": ready,
        "stage": install["stage"],
        "error": install["error"],
        "model": SFX_REPO,
        "license": SFX_LICENSE,
        "download_bytes": SFX_DOWNLOAD_BYTES,
        "sample_rate": SFX_SAMPLE_RATE,
        "max_seconds": MAX_SECONDS,
        "max_variations": MAX_VARIATIONS,
    }


def _locate_uv() -> str | None:
    from services.sidecar_install import _locate_uv as locate

    return locate()


def _install_runtime() -> None:
    os.makedirs(LIBS_DIR, exist_ok=True)
    uv = _locate_uv()
    if uv:
        cmd = [uv, "pip", "install", "--python", sys.executable, "--target", LIBS_DIR,
               "--no-deps", DIFFUSERS_SPEC]
    else:
        cmd = [sys.executable, "-m", "pip", "install", "--target", LIBS_DIR,
               "--no-deps", DIFFUSERS_SPEC]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
        raise RuntimeError("Installing diffusers failed: " + " ".join(tail))
    import importlib

    importlib.invalidate_caches()
    if not runtime_installed():
        raise RuntimeError("diffusers installed but cannot be imported")


def _download_model() -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(SFX_REPO, allow_patterns=SFX_ALLOW_PATTERNS)


def _run_install() -> None:
    try:
        if not runtime_installed():
            _set_stage("installing")
            _install_runtime()
        if not model_downloaded():
            _set_stage("downloading")
            _download_model()
        _set_stage("done")
    except Exception as exc:  # noqa: BLE001 — surfaced to the UI, never raised
        logger.exception("Sound effects install failed")
        with _state_lock:
            _install.update(stage="error", error=str(exc))


def _set_stage(stage: str) -> None:
    with _state_lock:
        _install.update(stage=stage, error=None)


def start_install() -> dict[str, Any]:
    """Begin the user-requested install in the background (idempotent)."""
    with _state_lock:
        busy = _install["stage"] in ("installing", "downloading")
        if not busy:
            _install.update(stage="installing", error=None)
    if not busy:
        threading.Thread(target=_run_install, name="sfx-install", daemon=True).start()
    return status()


def _load_pipeline():
    _use_libs_dir()
    import torch
    from diffusers import AudioLDM2Pipeline

    from services.model_loading import exclusive_model_load

    cuda = torch.cuda.is_available()
    with exclusive_model_load("Sound effects"):
        pipe = AudioLDM2Pipeline.from_pretrained(
            SFX_REPO,
            torch_dtype=torch.float16 if cuda else torch.float32,
            local_files_only=True,
        )
        pipe.generate_language_model = types.MethodType(_generate_language_model, pipe)
        pipe.score_waveforms = types.MethodType(_score_waveforms, pipe)
        return pipe.to("cuda" if cuda else "cpu")


#: Freed by the single-engine policy when speech or the other tool needs the GPU.
MODEL = ResidentModel("sfx", _load_pipeline)


def _generate_language_model(self, inputs_embeds=None, max_new_tokens=8, **model_kwargs):
    """AudioLDM2's GPT-2 prior without the KV cache.

    diffusers' loop leans on ``GPT2Model._update_model_kwargs_for_generation``,
    a ``GenerationMixin`` internal the base ``GPT2Model`` no longer carries in
    transformers 5. The prior only emits ``max_new_tokens`` (8) hidden states,
    so re-running the causal model over the whole sequence each step is cheap
    and gives the same states the cached loop would.
    """
    import torch

    lm = self.language_model
    max_new_tokens = max_new_tokens or lm.config.max_new_tokens
    attention_mask = model_kwargs.get("attention_mask")
    for _ in range(max_new_tokens):
        output = lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        next_hidden = output.hidden_states[-1][:, -1:, :]
        inputs_embeds = torch.cat([inputs_embeds, next_hidden], dim=1)
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=1
            )
    return inputs_embeds[:, -max_new_tokens:, :]


def _score_waveforms(self, text, audio, num_waveforms_per_prompt, device, dtype):
    """AudioLDM2's CLAP ranking of the takes, best match first, without librosa.

    diffusers resamples the takes to CLAP's rate with librosa and, when it is
    missing (it is not in the app runtime, and diffusers is installed with
    ``--no-deps``), returns them unranked. torchaudio, which the runtime
    ships, does the same resample; the scoring is diffusers' own.
    """
    import torch
    import torchaudio

    inputs = self.tokenizer(text, return_tensors="pt", padding=True)
    target_sr = self.feature_extractor.sampling_rate
    resampled = torchaudio.functional.resample(
        torch.as_tensor(audio, dtype=torch.float32), self.vocoder.config.sampling_rate, target_sr,
    )
    inputs["input_features"] = self.feature_extractor(
        list(resampled.numpy()), return_tensors="pt", sampling_rate=target_sr,
    ).input_features.type(dtype)
    inputs = inputs.to(device)
    logits_per_text = self.text_encoder(**inputs).logits_per_text
    indices = torch.argsort(logits_per_text, dim=1, descending=True)[:, :num_waveforms_per_prompt]
    return torch.index_select(audio, 0, indices.reshape(-1).cpu())


def clamp_seconds(seconds: float) -> float:
    return max(MIN_SECONDS, min(MAX_SECONDS, float(seconds)))


def generate(prompt: str, seconds: float, variations: int, seed: int,
             steps: int = DEFAULT_STEPS) -> list:
    """Render ``variations`` clips for ``prompt``; returns ``[1, N]`` float tensors."""
    with MODEL.use() as pipe:
        return _render(pipe, prompt, seconds, variations, seed, steps)


def _render(pipe, prompt, seconds, variations, seed, steps) -> list:
    import torch

    device = pipe.device if hasattr(pipe, "device") else "cpu"
    generator = torch.Generator(device=str(device)).manual_seed(int(seed))
    out = pipe(
        prompt,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=int(steps),
        audio_length_in_s=clamp_seconds(seconds),
        num_waveforms_per_prompt=max(1, min(MAX_VARIATIONS, int(variations))),
        generator=generator,
    )
    clips = []
    for waveform in out.audios:
        tensor = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        peak = float(tensor.abs().max()) if tensor.numel() else 0.0
        if peak > 0.99:
            tensor = tensor * (0.99 / peak)
        clips.append(tensor)
    return clips
