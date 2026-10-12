"""Roman-script text to the language's own script (Hinglish to Devanagari).

Every TTS engine reads ``kya haal hai`` as English: English vowels, English
stress, the wrong voice for a Hindi take. Hindi speakers type Roman script
all the time, so the Script box offers to rewrite it in Devanagari before
synthesis. A language model does the rewrite (the ``transliteration`` LLM
skill, routed like every other LLM call): casual Roman Hindi has no spelling
standard ("nahi", "nhi", "nahin" are one word), which rules cannot resolve
and a language model does well.

Guarded so a model cannot quietly change the script:

* bracket spans (``[pause 300ms]``, ``[laughter]``, ``[voice:Name]``,
  ``[[…]]``) are masked before the call and must all come back exactly once;
* line breaks must come back where they were;
* the answer must be mostly in the target script and of a plausible length.

Anything else raises :class:`TransliterationError`, which the route turns
into an error the UI shows. The user's text is never replaced by a rewrite
that failed these checks.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

from services.reference_language import _script_of

logger = logging.getLogger("omnivoice.transliteration")

SKILL_ID = "transliteration"
MAX_CHARS = 20_000
_CHUNK_CHARS = 1_200

#: Language code to (English name, target script id, script name for the prompt).
_TARGETS: dict[str, tuple[str, str, str]] = {
    "hi": ("Hindi", "devanagari", "Devanagari"),
    "mr": ("Marathi", "devanagari", "Devanagari"),
    "ne": ("Nepali", "devanagari", "Devanagari"),
    "bn": ("Bengali", "bengali", "Bengali"),
    "pa": ("Punjabi", "gurmukhi", "Gurmukhi"),
    "gu": ("Gujarati", "gujarati", "Gujarati"),
    "ta": ("Tamil", "tamil", "Tamil"),
    "te": ("Telugu", "telugu", "Telugu"),
    "kn": ("Kannada", "kannada", "Kannada"),
    "ml": ("Malayalam", "malayalam", "Malayalam"),
    "ur": ("Urdu", "arabic", "Urdu (Perso-Arabic)"),
    "ar": ("Arabic", "arabic", "Arabic"),
    "fa": ("Persian", "arabic", "Persian (Perso-Arabic)"),
    "ru": ("Russian", "cyrillic", "Cyrillic"),
    "uk": ("Ukrainian", "cyrillic", "Cyrillic"),
    "el": ("Greek", "greek", "Greek"),
    "th": ("Thai", "thai", "Thai"),
}

_PROMPT = """\
You convert {language} written in Roman (Latin) letters into {language} written \
in the {script} script, the way a native writer would type it.

Rules:
- Change the script, never the words. Every Roman word becomes the same word \
in {script}, in the same place: never a synonym, never a rephrasing, never a \
fixed-up sentence. For Hindi, "kya haal hai" is "क्या हाल है" (not "कैसा हाल है").
- Casual spellings ("nahi", "nhi", "nahin") are the same word: write its \
standard spelling.
- English words used inside the {language} text (meeting, phone, plan) are \
written in {script} as {language} speakers write them. Brand and product names \
may stay in Latin letters.
- Keep digits, punctuation, emoji and every line break exactly where they are.
- Tokens like {marker} are placeholders: copy each one unchanged, in place.
- Do not translate, explain, add or remove anything.

Reply with only the converted text."""

_BRACKET_RE = re.compile(r"\[\[[^\]\n]*\]\]|\[[^\[\]\n]{0,128}\]")
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


class TransliterationError(Exception):
    """``code``: ``unsupported_language``, ``llm_unavailable``, ``too_long``,
    ``llm_failed`` or ``bad_output``."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Result:
    text: str
    changed: bool


def target_for(language: Optional[str]) -> Optional[tuple[str, str, str]]:
    """``(name, script id, script name)`` for a language with its own script, else None."""
    from services.reference_language import _code  # noqa: PLC0415

    return _TARGETS.get(_code(language) or "")


def _letters(text: str) -> list[str]:
    return [c for c in _BRACKET_RE.sub(" ", text) if c.isalpha()]


def script_share(text: str, script: str) -> float:
    """Share of ``text``'s letters (outside bracket tags) written in ``script``."""
    letters = _letters(text)
    if not letters:
        return 0.0
    return sum(1 for c in letters if _script_of(c) == script) / len(letters)


def needs_conversion(text: str, language: Optional[str]) -> bool:
    """True when ``text`` is mostly Roman letters but ``language`` has its own script."""
    if target_for(language) is None:
        return False
    letters = _letters(text)
    return len(letters) >= 4 and script_share(text, "latin") >= 0.6


def _mask(text: str) -> tuple[str, list[str]]:
    spans: list[str] = []

    def keep(m: re.Match) -> str:
        spans.append(m.group(0))
        return f"⟦{len(spans) - 1}⟧"

    return _BRACKET_RE.sub(keep, text), spans


def _unmask(text: str, spans: list[str]) -> str:
    for i, span in enumerate(spans):
        marker = f"⟦{i}⟧"
        if text.count(marker) != 1:
            raise TransliterationError("bad_output", "a [tag] in the script was lost or duplicated")
        text = text.replace(marker, span)
    if re.search("⟦\\d+⟧", text):
        raise TransliterationError("bad_output", "the model invented a placeholder")
    return text


def _chunks(text: str) -> list[str]:
    """Whole lines, packed up to ``_CHUNK_CHARS`` (a longer line is its own chunk)."""
    out: list[str] = []
    cur: Optional[str] = None  # None, not "": an empty line is still a line
    for line in text.split("\n"):
        if cur is None:
            cur = line
        elif len(cur) + 1 + len(line) > _CHUNK_CHARS:
            out.append(cur)
            cur = line
        else:
            cur = f"{cur}\n{line}"
    out.append(cur or "")
    return out


def _clean(answer: str) -> str:
    s = _FENCE_RE.sub("", (answer or "").strip())
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'“”":
        s = s[1:-1]
    return s


def _words(text: str) -> int:
    return len([w for w in _BRACKET_RE.sub(" ", text).split() if any(c.isalnum() for c in w)])


def _check(source: str, answer: str, script: str) -> None:
    if answer.count("\n") != source.count("\n"):
        raise TransliterationError("bad_output", "the model changed the line breaks")
    # A transliteration keeps one word per word. A rewrite that grew or shrank
    # by more than a few words translated, summarised or rephrased instead.
    src_words, out_words = _words(source), _words(answer)
    if abs(out_words - src_words) > max(2, src_words // 5):
        raise TransliterationError("bad_output", "the rewrite changed the words, not just the script")
    src_letters, out_letters = len(_letters(source)), len(_letters(answer))
    if src_letters and not (0.35 <= out_letters / src_letters <= 2.5):
        raise TransliterationError("bad_output", "the rewrite is not the same length of text")
    # Names and acronyms (CEO, YouTube) may stay in Latin letters, so they do
    # not count against the target script, but only names the source itself
    # capitalised: a reply that capitalises "kya haal hai" is not a name.
    names = set(_NAME_RE.findall(source))
    unnamed = _NAME_RE.sub(lambda m: " " if m.group(0) in names else m.group(0), answer)
    if src_letters >= 4 and len(_letters(unnamed)) >= 2 and script_share(unnamed, script) < 0.6:
        raise TransliterationError("bad_output", "the rewrite is not in the target script")
    if src_letters >= 4 and _letters(answer) and script_share(answer, script) == 0:
        raise TransliterationError("bad_output", "the rewrite has no letters of the target script")


_NAME_RE = re.compile(r"(?<![A-Za-z])[A-Z][A-Za-z0-9'&.-]*")


#: A lowercase Roman word: romanized text the model left unconverted. Acronyms
#: (CEO) and brand names (YouTube, iPhone) start with a capital and may stay.
_LEFTOVER_RE = re.compile(r"(?<![A-Za-z])[a-z][a-z']+(?![A-Za-z])")


def _leftovers(text: str) -> list[str]:
    return _LEFTOVER_RE.findall(_BRACKET_RE.sub(" ", re.sub("\u27e6\\d+\u27e7", " ", text)))


def _convert_words(llm, words, language: str, script_name: str, script: str) -> dict[str, str]:
    """Romanized words to the target script, one model call; only clean one-word answers."""
    unique = list(dict.fromkeys(words))
    system = (f"Write each Romanized {language} word in the {script_name} script, the way a "
              f"native writer spells it. One word per line, in the same order, nothing else.")
    lines = [ln.strip() for ln in _ask(llm, system, "\n".join(unique)).splitlines() if ln.strip()]
    if len(lines) != len(unique):
        return {}
    out = {}
    for word, native in zip(unique, lines):
        # Models sometimes answer "humne → हमने" or "humne: हमने"; keep the answer side.
        native = re.split(r"\s*(?:→|->|=>|=|:)\s*", native)[-1]
        native = native.strip(" .,:;-–—\"'")
        if len(native.split()) == 1 and _letters(native) and script_share(native, script) == 1.0:
            out[word] = native
    return out


def _replace_words(text: str, mapping: dict[str, str]) -> str:
    for word, native in mapping.items():
        text = re.sub(rf"(?<![A-Za-z]){re.escape(word)}(?![A-Za-z])", native, text)
    return text


def _ask(llm, system: str, user: str) -> str:
    try:
        return _clean(llm.chat(system=system, user=user))
    except TransliterationError:
        raise
    except Exception as exc:  # noqa: BLE001 — provider errors surface as one code
        logger.warning("transliteration LLM call failed: %s", exc)
        raise TransliterationError("llm_failed", "the LLM call failed") from exc


def transliterate(text: str, language: Optional[str], *, llm=None) -> Result:
    """Rewrite Roman-letter ``text`` in ``language``'s own script.

    Text already in the target script (or with too few Roman letters to
    matter) comes back unchanged without calling the model.
    """
    target = target_for(language)
    if target is None:
        raise TransliterationError("unsupported_language",
                                   "this language is written in Latin letters already")
    if len(text) > MAX_CHARS:
        raise TransliterationError("too_long", f"scripts over {MAX_CHARS} characters are not converted")
    if not needs_conversion(text, language):
        return Result(text=text, changed=False)
    name, script, script_name = target

    if llm is None:
        from services import llm_skills  # noqa: PLC0415
        from services.llm_backend import OffBackend  # noqa: PLC0415

        llm = llm_skills.skill_backend(SKILL_ID)
        if isinstance(llm, OffBackend):
            raise TransliterationError("llm_unavailable", "no LLM is set up for script conversion")

    masked, spans = _mask(text)
    system = _PROMPT.format(language=name, script=script_name, marker="⟦0⟧")
    pieces = []
    for chunk in _chunks(masked):
        if not chunk.strip() or not needs_conversion(chunk, language):
            pieces.append(chunk)
            continue
        lead = chunk[: len(chunk) - len(chunk.lstrip())]
        trail = chunk[len(chunk.rstrip()):]
        answer = f"{lead}{_ask(llm, system, chunk).strip()}{trail}"
        _check(chunk, answer, script)
        if _leftovers(answer):
            # A model sometimes leaves a word in Roman letters ("humne 10 हज़ार
            # से…"). One more pass over its own answer converts the rest; it is
            # kept only if it passes the same checks and leaves fewer behind.
            left = ", ".join(sorted(set(_leftovers(answer))))
            follow_up = f"{system}\n\nThese words are still in Roman letters; write them in the target script too: {left}."
            second = f"{lead}{_ask(llm, follow_up, answer).strip()}{trail}"
            try:
                _check(chunk, second, script)
            except TransliterationError:
                second = answer
            if len(_leftovers(second)) < len(_leftovers(answer)):
                answer = second
        if _leftovers(answer):
            # Still there: ask for just those words and put each one back in
            # its place. A reply that is not one word in the target script is
            # not used, so the worst case is the word staying in Roman.
            answer = _replace_words(answer, _convert_words(llm, _leftovers(answer), name, script_name, script))
        pieces.append(answer)
    converted = _unmask("\n".join(pieces), spans)
    if converted.count("\n") != text.count("\n"):
        raise TransliterationError("bad_output", "the conversion changed the line breaks")
    return Result(text=converted, changed=converted != text)
