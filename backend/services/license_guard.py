"""Commercial-safe mode: which engine's weights may be used for monetized output, and a switch that enforces it.

Most TTS / music / sound-effect weights are not licensed like the code that runs them: OmniVoice's code is
Apache-2.0 but its pre-trained model is CC-BY-NC (training data), MusicGen is CC-BY-NC-4.0, AudioLDM2 is
CC-BY-NC-SA-4.0. A creator who posts to monetized channels needs to know that before a render ships.

* ``ENGINE_LICENSES`` / ``TOOL_LICENSES`` — the weights' licence per engine/tool, with a status:
  ``commercial`` (permissive), ``conditional`` (commercial use allowed under the licence's conditions — read
  them), ``non_commercial`` and ``unknown`` (depends on the model the user picks, or not reviewed).
* ``commercial_only()`` — opt-in switch (Settings, or ``OMNIVOICE_COMMERCIAL_ONLY=1``); default off, so nothing
  changes for anyone who does not turn it on.
* ``check_engine()`` / ``check_tool()`` — raise ``CommercialUseBlocked`` while the switch is on and the weights
  are ``non_commercial`` or ``unknown`` (fail-safe: an unreviewed engine is not assumed to be safe).

Statuses are the maintainers' reading of each model card on the date in ``REVIEWED``; this is guidance, not
legal advice — the linked licence is authoritative.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

REVIEWED = "2026-10-07"
PREF_KEY = "commercial_only"
ENV_VAR = "OMNIVOICE_COMMERCIAL_ONLY"

COMMERCIAL = "commercial"
CONDITIONAL = "conditional"
NON_COMMERCIAL = "non_commercial"
UNKNOWN = "unknown"
ALLOWED_WHEN_COMMERCIAL = (COMMERCIAL, CONDITIONAL)


@dataclass(frozen=True)
class LicenseInfo:
    status: str
    weights: str  # the weights' licence name
    source: str  # where the licence is stated
    note: str = ""


_OMNIVOICE = LicenseInfo(NON_COMMERCIAL, "CC-BY-NC-4.0", "https://huggingface.co/k2-fsa/OmniVoice",
                         "Code is Apache-2.0; the pre-trained model is CC-BY-NC because of its training data.")

ENGINE_LICENSES: dict[str, LicenseInfo] = {
    "omnivoice": _OMNIVOICE,
    "omnivoice-subprocess": _OMNIVOICE,
    "omnivoice-gguf": LicenseInfo(NON_COMMERCIAL, "CC-BY-NC-4.0 (quantised OmniVoice)",
                                  "https://huggingface.co/k2-fsa/OmniVoice", "Same weights, quantised."),
    "voxcpm2": LicenseInfo(COMMERCIAL, "Apache-2.0", "https://huggingface.co/openbmb/VoxCPM2"),
    "cosyvoice": LicenseInfo(COMMERCIAL, "Apache-2.0", "https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512"),
    "moss-tts-nano": LicenseInfo(COMMERCIAL, "Apache-2.0", "https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M"),
    "kittentts": LicenseInfo(COMMERCIAL, "Apache-2.0", "https://huggingface.co/KittenML/kitten-tts-mini-0.8"),
    # Next three: licence as stated in this repo's install notes (services/tts_backend.py _INSTALL_HINTS).
    "moss-tts-v15": LicenseInfo(COMMERCIAL, "Apache-2.0", "_INSTALL_HINTS['moss-tts-v15']"),
    "dots-tts": LicenseInfo(COMMERCIAL, "Apache-2.0", "_INSTALL_HINTS['dots-tts']"),
    "confucius4-tts": LicenseInfo(COMMERCIAL, "Apache-2.0", "_INSTALL_HINTS['confucius4-tts']"),
    "indextts2": LicenseInfo(CONDITIONAL, "bilibili Model Use License", "https://huggingface.co/IndexTeam/IndexTTS-2",
                             "Free commercial use below 100M monthly users / RMB 1B yearly revenue; outputs may not "
                             "be used to improve other AI models."),
    "supertonic3": LicenseInfo(CONDITIONAL, "OpenRAIL-M", "_INSTALL_HINTS['supertonic3']",
                               "Commercial use allowed with the licence's use-based restrictions."),
    "pockettts": LicenseInfo(CONDITIONAL, "CC-BY-4.0 weights", "_INSTALL_HINTS['pockettts']",
                             "Commercial use allowed with attribution; the weights are HF-gated, review the terms."),
    "audiocpp": LicenseInfo(NON_COMMERCIAL, "Research / non-commercial", "config/models.yaml (audio.cpp bundle note)",
                            "Breeze-TTS-2 weights are research / non-commercial only."),
    "gpt-sovits": LicenseInfo(UNKNOWN, "Depends on the trained voice model", "https://github.com/RVC-Boss/GPT-SoVITS",
                              "Code is MIT; the licence of the voice you load decides."),
    "mlx-audio": LicenseInfo(UNKNOWN, "Depends on the selected model", "https://github.com/Blaizzy/mlx-audio",
                             "Curated models carry different licences."),
    "sherpa-onnx": LicenseInfo(UNKNOWN, "Depends on the selected model", "https://github.com/k2-fsa/sherpa-onnx"),
}

TOOL_LICENSES: dict[str, LicenseInfo] = {
    "music": LicenseInfo(NON_COMMERCIAL, "CC-BY-NC-4.0 (MusicGen)", "https://huggingface.co/facebook/musicgen-stereo-medium"),
    "sfx": LicenseInfo(NON_COMMERCIAL, "CC-BY-NC-SA-4.0 (AudioLDM2)", "https://huggingface.co/cvssp/audioldm2"),
}

_UNREVIEWED = LicenseInfo(UNKNOWN, "Not reviewed", "", "This engine's licence has not been reviewed yet.")


class CommercialUseBlocked(Exception):
    """Commercial-safe mode is on and the requested engine/tool is not cleared for commercial use."""

    def __init__(self, message: str, *, target: str, info: LicenseInfo):
        super().__init__(message)
        self.target = target
        self.info = info

    def detail(self) -> dict:
        """The refusal as the client reads it on every transport (HTTP 403 body, WebSocket error frame)."""
        return {"code": "commercial_use_blocked", "message": str(self), "target": self.target,
                "status": self.info.status, "weights": self.info.weights, "source": self.info.source}


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def commercial_only() -> bool:
    from core import prefs

    return _truthy(prefs.resolve(PREF_KEY, env=ENV_VAR, default=False))


def set_commercial_only(enabled: bool) -> None:
    from core import prefs

    prefs.set_(PREF_KEY, bool(enabled))


def license_for(engine_id: str) -> LicenseInfo:
    return ENGINE_LICENSES.get(engine_id, _UNREVIEWED)


def commercial_engines() -> list[str]:
    return sorted(e for e, info in ENGINE_LICENSES.items() if info.status in ALLOWED_WHEN_COMMERCIAL)


def _blocked(target: str, info: LicenseInfo, what: str) -> CommercialUseBlocked:
    return CommercialUseBlocked(
        f"Commercial-safe mode is on: {what} '{target}' uses {info.weights} weights "
        f"({info.status.replace('_', '-')}). Pick a commercial-safe engine "
        f"({', '.join(commercial_engines())}) or turn the mode off (POST /engines/commercial-only, "
        f"or unset {ENV_VAR}).",
        target=target, info=info)


def check_engine(engine_id: str) -> None:
    if not commercial_only():
        return
    info = license_for(engine_id)
    if info.status not in ALLOWED_WHEN_COMMERCIAL:
        raise _blocked(engine_id, info, "TTS engine")


def check_tool(tool: str) -> None:
    if not commercial_only():
        return
    info = TOOL_LICENSES.get(tool, _UNREVIEWED)
    if info.status not in ALLOWED_WHEN_COMMERCIAL:
        raise _blocked(tool, info, "tool")


def tool_blocked(tool: str) -> bool:
    """True when Commercial-safe mode would refuse ``tool``: its status says so,
    so the app offers no download for a model it could not use."""
    try:
        check_tool(tool)
    except CommercialUseBlocked:
        return True
    return False


def payload() -> dict:
    """Everything the UI / API needs in one call."""
    import os

    from core import prefs

    return {
        "commercial_only": commercial_only(),
        "env_pinned": os.environ.get(ENV_VAR) is not None,
        "reviewed": REVIEWED,
        "engines": {k: asdict(v) for k, v in ENGINE_LICENSES.items()},
        "tools": {k: asdict(v) for k, v in TOOL_LICENSES.items()},
        "pref_saved": prefs.get(PREF_KEY) is not None,
    }
