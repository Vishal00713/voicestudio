"""Does the clone reference speak the language being generated?

Zero-shot cloning copies the delivery of the reference clip, accent included:
a Hindi take cloned from an English clip comes out with an English accent,
and nothing in the take says why. The reference transcript (saved with the
profile, typed by the user, or transcribed for this request) shows which
script the speaker used; the request names the language. When the two
disagree, ``/generate`` answers with ``X-OmniVoice-Reference-Script`` and the
UI tells the user to use a sample in the same language.

Script, not language: languages that share a script (English and Spanish,
Hindi and Marathi) cannot be told apart from text without a language-ID
model, and a false alarm is worse than a missed one. So only a script
disagreement counts, only when the transcript has enough letters for one
script to clearly dominate, and pairs that are one spoken language in two
scripts (Hindi/Urdu, Punjabi, Serbian) count as agreeing.

Pure text work, no model, identical on every platform.
"""
from __future__ import annotations

from typing import Optional

HEADER = "X-OmniVoice-Reference-Script"

#: Fewer letters than this and the transcript says nothing reliable.
MIN_LETTERS = 8
#: The leading script must hold at least this share of the letters.
DOMINANCE = 0.6

_RANGES = (
    ("latin", ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF))),
    ("greek", ((0x0370, 0x03FF),)),
    ("cyrillic", ((0x0400, 0x04FF),)),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF))),
    ("devanagari", ((0x0900, 0x097F),)),
    ("bengali", ((0x0980, 0x09FF),)),
    ("gurmukhi", ((0x0A00, 0x0A7F),)),
    ("gujarati", ((0x0A80, 0x0AFF),)),
    ("tamil", ((0x0B80, 0x0BFF),)),
    ("telugu", ((0x0C00, 0x0C7F),)),
    ("kannada", ((0x0C80, 0x0CFF),)),
    ("malayalam", ((0x0D00, 0x0D7F),)),
    ("thai", ((0x0E00, 0x0E7F),)),
    ("hangul", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
    ("kana", ((0x3040, 0x30FF),)),
    ("cjk", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF))),
)

_LATIN = {"latin"}
_EXPECTED = {
    **{code: _LATIN for code in (
        "en", "es", "fr", "de", "it", "pt", "nl", "pl", "tr", "id", "vi", "sv", "da",
        "no", "nb", "nn", "fi", "ro", "hu", "cs", "sk", "sl", "hr", "bs", "lt", "lv",
        "et", "sw", "ms", "tl", "fil", "az", "ca", "eu", "gl", "ga", "cy", "is", "sq",
        "af", "uz", "ha", "yo", "ig", "zu", "xh", "so", "mt", "lb",
    )},
    "hi": {"devanagari", "arabic"},   # Hindi and Urdu: one spoken language
    "ur": {"arabic", "devanagari"},
    "mr": {"devanagari"},
    "ne": {"devanagari"},
    "bn": {"bengali"},
    "as": {"bengali"},
    "pa": {"gurmukhi", "arabic"},
    "gu": {"gujarati"},
    "ta": {"tamil"},
    "te": {"telugu"},
    "kn": {"kannada"},
    "ml": {"malayalam"},
    "ar": {"arabic"},
    "fa": {"arabic"},
    "he": {"hebrew"},
    "el": {"greek"},
    "ru": {"cyrillic"},
    "uk": {"cyrillic"},
    "bg": {"cyrillic"},
    "mk": {"cyrillic"},
    "be": {"cyrillic"},
    "kk": {"cyrillic"},
    "kz": {"cyrillic"},
    "sr": {"cyrillic", "latin"},
    "mn": {"cyrillic"},
    "th": {"thai"},
    "ko": {"hangul"},
    "ja": {"kana", "cjk"},
    "zh": {"cjk"},
}

# Display names the shared resolver (text_normalization) does not map.
_NAMES = {
    "bengali": "bn", "bangla": "bn", "tamil": "ta", "telugu": "te", "kannada": "kn",
    "gujarati": "gu", "punjabi": "pa", "marathi": "mr", "nepali": "ne", "urdu": "ur",
    "assamese": "as", "arabic": "ar", "greek": "el", "thai": "th", "bulgarian": "bg",
    "macedonian": "mk", "belarusian": "be", "mongolian": "mn", "hebrew": "he",
    "catalan": "ca", "basque": "eu", "galician": "gl", "irish": "ga", "welsh": "cy",
    "icelandic": "is", "albanian": "sq", "afrikaans": "af", "swahili": "sw",
    "malay": "ms", "tagalog": "tl", "filipino": "fil", "croatian": "hr",
    "bosnian": "bs", "slovak": "sk", "estonian": "et",
}


def _script_of(ch: str) -> Optional[str]:
    cp = ord(ch)
    for name, spans in _RANGES:
        for lo, hi in spans:
            if lo <= cp <= hi:
                return name
    return None


def dominant_script(text: Optional[str]) -> Optional[str]:
    """The script most of ``text``'s letters are written in, or None when the
    text is too short or mixed for one script to clearly lead."""
    counts: dict[str, int] = {}
    total = 0
    for ch in text or "":
        if not ch.isalpha():
            continue
        script = _script_of(ch)
        if script is None:
            continue
        counts[script] = counts.get(script, 0) + 1
        total += 1
    if total < MIN_LETTERS:
        return None
    script, count = max(counts.items(), key=lambda kv: kv[1])
    return script if count / total >= DOMINANCE else None


def _code(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    name = str(language).strip().lower()
    if not name or name == "auto":
        return None
    if name in _NAMES:
        return _NAMES[name]
    from services.text_normalization import _plain_lang_code  # noqa: PLC0415

    return _plain_lang_code(language)


def script_mismatch(ref_text: Optional[str], language: Optional[str]) -> Optional[str]:
    """The reference transcript's script when it cannot be ``language``'s, else None."""
    expected = _EXPECTED.get(_code(language) or "")
    if not expected:
        return None
    found = dominant_script(ref_text)
    if found is None or found in expected:
        return None
    return found


def notice_headers(ref_text: Optional[str], language: Optional[str]) -> dict[str, str]:
    """``{HEADER: script}`` when the reference and the take disagree, else ``{}``.
    Never raises: a notice must not be able to fail a generation."""
    try:
        found = script_mismatch(ref_text, language)
    except Exception:  # noqa: BLE001
        return {}
    return {HEADER: found} if found else {}
