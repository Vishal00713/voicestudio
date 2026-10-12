"""ElevenLabs-style audio tags → what the active engine can really perform.

Scripts written for ElevenLabs v3 carry inline cues such as ``[laughs]``,
``[sighs]``, ``[whispers]`` or ``[excited]``. Every engine receives text
verbatim, so an unknown tag is *spoken aloud*. This module lowers a closed set
of those cues before synthesis (Spec 01, docs/specs/01-expressive-tts.md):

* reactions become the engine's native non-verbal token where one exists
  (``[laughs]`` → ``[laughter]`` on VoiceStudio and CosyVoice 3);
* pause cues become the engine-agnostic ``[pause Nms]`` marker;
* ``[whispers]`` opening the script becomes the whisper style on engines that
  take an instruct (the whole take — no engine here can switch style
  mid-line, so a mid-line cue is dropped and reported);
* everything else in the set is removed and reported, so degradation is
  visible (``X-OmniVoice-Expression-Dropped``) instead of silent or spoken.

The set is closed: a bracket span outside it (``[pause 1s]``, ``[laughter]``,
``[voice:Ana]``, CMU phonemes, ``[[…]]`` overrides) is never touched.
Pure text work — identical on every platform, no model, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# cue → intent. Native tags ([laughter], [sigh], [pause …]) are deliberately
# absent: they already reach the engine in the form it understands.
_REACTIONS = {
    "laughs": "laugh", "laugh": "laugh", "laughing": "laugh",
    "chuckles": "laugh", "chuckle": "laugh", "giggles": "laugh",
    "giggle": "laugh", "laughs harder": "laugh", "starts laughing": "laugh",
    "laughs softly": "laugh", "snickers": "laugh",
    "sighs": "sigh", "sighing": "sigh", "exhales": "sigh",
    "exhales sharply": "sigh", "sighs heavily": "sigh",
    "inhales": "breath", "inhales deeply": "breath", "breathes": "breath",
    "breathing": "breath", "gasps": "breath",
}
_PAUSES = {"short pause": 300, "pause briefly": 300, "long pause": 1000}
_WHISPER = {"whispers", "whispering", "whispered", "quietly"}
# Delivery cues no shipped engine can render mid-line. Removed, then reported.
_UNSUPPORTED = {
    "excited", "excitedly", "happy", "happily", "sad", "sadly", "angry",
    "angrily", "curious", "curiously", "sarcastic", "sarcastically",
    "nervous", "nervously", "cheerfully", "cheerful", "crying", "cries",
    "sobbing", "shouts", "shouting", "yelling", "mischievously", "calm",
    "calmly", "serious", "seriously", "surprised", "annoyed", "thoughtful",
    "thoughtfully", "frustrated", "impressed", "amazed", "warmly",
    "playfully", "dramatically", "softly", "tired", "sleepy", "scared",
    "fearful", "confident", "confidently", "hesitant", "hesitates",
    "sorrowful", "regretful", "rushed", "slowly", "drawn out", "emphasized",
    "stammers", "clears throat", "coughs", "sniffs", "applause", "gulps",
}

# Engine family → intent → native token.
_NATIVE = {
    "omnivoice": {"laugh": "[laughter]", "sigh": "[sigh]"},
    "cosyvoice": {"laugh": "[laughter]", "breath": "[breath]"},
}

_TAG_RE = re.compile(r"(?<!\[)\[([A-Za-z][A-Za-z \-']{1,30})\](?!\])")


@dataclass
class LoweredText:
    text: str
    whisper: bool = False
    dropped: list[str] = field(default_factory=list)


def _family(engine_id: str | None, instruct_vocabulary: str) -> str | None:
    engine = (engine_id or "omnivoice").lower()
    if engine.startswith("cosyvoice"):
        return "cosyvoice"
    if instruct_vocabulary == "tags" or engine.startswith("omnivoice"):
        return "omnivoice"
    return None


def lower_audio_tags(
    text: str,
    engine_id: str | None,
    instruct_vocabulary: str = "tags",
    takes_instruct: bool = True,
) -> LoweredText:
    """Rewrite ElevenLabs-style cues in ``text`` for ``engine_id``."""
    if "[" not in text:
        return LoweredText(text)
    native = _NATIVE.get(_family(engine_id, instruct_vocabulary) or "", {})
    result = LoweredText(text)

    def replace(match: re.Match[str]) -> str:
        cue = " ".join(match.group(1).lower().split())
        if cue in _PAUSES:
            return f"[pause {_PAUSES[cue]}ms]"
        intent = _REACTIONS.get(cue)
        if intent:
            token = native.get(intent)
            if token:
                return token
        elif cue in _WHISPER:
            # Style covers the whole take, so only a cue that opens the script
            # means "whisper all of it"; mid-line it would whisper the lines
            # before it too, which is worse than leaving it out.
            opens_script = not _TAG_RE.sub("", text[: match.start()]).strip()
            if takes_instruct and opens_script:
                result.whisper = True
                return ""
        elif cue not in _UNSUPPORTED:
            return match.group(0)
        if match.group(0) not in result.dropped:
            result.dropped.append(match.group(0))
        return ""

    lowered = _TAG_RE.sub(replace, text)
    if lowered != text:
        # Removing a cue leaves doubled spaces or a space before punctuation.
        lowered = re.sub(r"[ \t]{2,}", " ", lowered)
        lowered = re.sub(r" +([,.!?;:।])", r"\1", lowered).strip()
    result.text = lowered
    return result


def with_whisper(instruct: str | None, instruct_vocabulary: str) -> str:
    """Add the whisper style to ``instruct`` once, in the engine's vocabulary."""
    current = (instruct or "").strip()
    if "whisper" in current.lower():
        return current
    style = "whisper" if instruct_vocabulary == "tags" else "whispering softly"
    return f"{current}, {style}" if current else style
