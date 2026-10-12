"""Roman-script to native-script conversion (``services/transliteration.py``).

The LLM is faked: these tests pin the guards around it — what is sent, what
must come back, and that a bad rewrite never replaces the user's text.
"""
import pytest

from services import transliteration as tl
from services.transliteration import TransliterationError, needs_conversion, transliterate


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, *, system, user, **_):
        self.calls.append((system, user))
        return self.reply(user) if callable(self.reply) else self.reply


def _devanagari_for(user: str) -> str:
    """A faithful fake: maps a few Roman words, keeps placeholders and breaks."""
    words = {"kya": "क्या", "haal": "हाल", "hai": "है", "aap": "आप", "sab": "सब",
             "ka": "का", "meeting": "मीटिंग", "kal": "कल", "thi": "थी", "nahi": "नहीं"}
    out_lines = []
    for line in user.split("\n"):
        out_lines.append(" ".join(words.get(w, w) for w in line.split(" ")))
    return "\n".join(out_lines)


@pytest.mark.parametrize("text, language, expected", [
    ("kya haal hai", "Hindi", True),
    ("kya haal hai", "hi", True),
    ("क्या हाल है", "Hindi", False),
    ("kya haal hai", "English", False),
    ("kya haal hai", "Auto", False),
    ("ok", "Hindi", False),                         # too few letters
    ("[pause 300ms] है", "Hindi", False),           # tags do not count
])
def test_needs_conversion(text, language, expected):
    assert needs_conversion(text, language) is expected


def test_converts_and_keeps_tags_and_line_breaks():
    llm = FakeLLM(_devanagari_for)
    res = transliterate("kya haal hai [pause 300ms] aap\nsab ka [laughter]", "Hindi", llm=llm)
    assert res.changed
    assert res.text == "क्या हाल है [pause 300ms] आप\nसब का [laughter]"
    system, user = llm.calls[0]
    assert "Hindi" in system and "Devanagari" in system
    assert "[pause" not in user and "⟦0⟧" in user, "tags travel as placeholders"


def test_text_already_in_script_does_not_call_the_model():
    llm = FakeLLM(lambda u: pytest.fail("the LLM must not be called"))
    res = transliterate("क्या हाल है आप सब का", "Hindi", llm=llm)
    assert res == tl.Result(text="क्या हाल है आप सब का", changed=False)


@pytest.mark.parametrize("reply, why", [
    ("kya haal hai aap", "still Roman letters"),
    ("क्या हाल है आप", "dropped the placeholder for [pause 300ms]"),
    ("क्या हाल है ⟦0⟧ ⟦0⟧ आप", "duplicated the placeholder"),
    ("क्या हाल है ⟦0⟧\nआप", "added a line break"),
    ("क्या ⟦0⟧ " + "हाल " * 40, "far longer than the source"),
    ("आप सभी लोग कैसे हैं, क्या सब ठीक चल रहा है ⟦0⟧ आप", "rephrased, not transliterated"),
])
def test_bad_rewrites_are_refused(reply, why):
    with pytest.raises(TransliterationError) as err:
        transliterate("kya haal hai [pause 300ms] aap", "Hindi", llm=FakeLLM(reply))
    assert err.value.code == "bad_output", why


def test_code_fences_and_quotes_are_stripped():
    res = transliterate("kya haal hai", "Hindi", llm=FakeLLM('```\n"क्या हाल है"\n```'))
    assert res.text == "क्या हाल है"


def test_long_text_goes_in_line_aligned_chunks():
    line = "kya haal hai aap sab ka " * 20
    text = "\n".join([line.strip()] * 6)
    llm = FakeLLM(_devanagari_for)
    res = transliterate(text, "Hindi", llm=llm)
    assert len(llm.calls) > 1
    assert res.text.count("\n") == 5
    assert "kya" not in res.text


def test_unsupported_language_and_too_long():
    with pytest.raises(TransliterationError) as err:
        transliterate("kya haal hai", "English", llm=FakeLLM("x"))
    assert err.value.code == "unsupported_language"
    with pytest.raises(TransliterationError) as err:
        transliterate("a" * (tl.MAX_CHARS + 1), "Hindi", llm=FakeLLM("x"))
    assert err.value.code == "too_long"


def test_provider_failure_is_one_error_code():
    class Boom:
        def chat(self, **_):
            raise RuntimeError("401 from provider")

    with pytest.raises(TransliterationError) as err:
        transliterate("kya haal hai", "Hindi", llm=Boom())
    assert err.value.code == "llm_failed"


def test_no_llm_configured(monkeypatch):
    from services import llm_skills
    from services.llm_backend import OffBackend

    monkeypatch.setattr(llm_skills, "skill_backend", lambda *_a, **_k: OffBackend())
    with pytest.raises(TransliterationError) as err:
        transliterate("kya haal hai", "Hindi")
    assert err.value.code == "llm_unavailable"


# ── route ───────────────────────────────────────────────────────────────────

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from main import app

    return TestClient(app, client=("127.0.0.1", 50000))


def test_route_returns_the_conversion(client, monkeypatch):
    monkeypatch.setattr(tl, "transliterate",
                        lambda text, language: tl.Result(text="क्या हाल है", changed=True))
    resp = client.post("/tools/transliterate", json={"text": "kya haal hai", "language": "Hindi"})
    assert resp.status_code == 200
    assert resp.json() == {"text": "क्या हाल है", "changed": True}


@pytest.mark.parametrize("code, status", [
    ("llm_unavailable", 503), ("unsupported_language", 422), ("too_long", 413),
    ("bad_output", 502), ("llm_failed", 502),
])
def test_route_maps_errors_to_statuses(client, monkeypatch, code, status):
    def fail(text, language):
        raise TransliterationError(code, "nope")

    monkeypatch.setattr(tl, "transliterate", fail)
    resp = client.post("/tools/transliterate", json={"text": "kya haal hai", "language": "Hindi"})
    assert resp.status_code == status
    assert resp.json()["detail"]["code"] == code


def test_a_word_left_in_roman_gets_one_more_pass():
    replies = iter(["humne दस हज़ार से शुरू किया", "हमने दस हज़ार से शुरू किया"])
    llm = FakeLLM(lambda user: next(replies))
    res = transliterate("humne das hazaar se shuru kiya", "Hindi", llm=llm)
    assert res.text == "हमने दस हज़ार से शुरू किया"
    assert len(llm.calls) == 2 and llm.calls[1][1] == "humne दस हज़ार से शुरू किया"


def test_a_follow_up_pass_that_breaks_the_checks_is_ignored():
    rephrase = "हमने बहुत पहले दस हज़ार रुपये से अपना काम शुरू किया था"
    replies = iter(["humne दस हज़ार से शुरू किया", rephrase, "humne → हमने"])
    res = transliterate("humne das hazaar se shuru kiya", "Hindi", llm=FakeLLM(lambda user: next(replies)))
    assert res.text == "हमने दस हज़ार से शुरू किया", "the rephrase is dropped, the word fixed on its own"


def test_capitalised_names_left_in_latin_do_not_trigger_a_second_pass():
    llm = FakeLLM(lambda user: "कल CEO ने YouTube पर बताया")
    transliterate("kal CEO ne YouTube par bataya", "Hindi", llm=llm)
    assert len(llm.calls) == 1


def test_a_word_that_survives_both_passes_is_converted_on_its_own():
    replies = iter(["humne दस हज़ार से शुरू किया", "humne दस हज़ार से शुरू किया", "हमने"])
    llm = FakeLLM(lambda user: next(replies))
    res = transliterate("humne das hazaar se shuru kiya", "Hindi", llm=llm)
    assert res.text == "हमने दस हज़ार से शुरू किया"
    assert llm.calls[2][1] == "humne"


def test_a_word_reply_that_is_not_one_native_word_is_not_used():
    replies = iter(["humne दस हज़ार से शुरू किया", "humne दस हज़ार से शुरू किया", "हम ने तो"])
    res = transliterate("humne das hazaar se shuru kiya", "Hindi", llm=FakeLLM(lambda user: next(replies)))
    assert res.text == "humne दस हज़ार से शुरू किया"


@pytest.mark.parametrize("text", ["\n\nkya haal hai", "kya haal hai\n\n", "kya haal\n\n\nhai aap sab"])
def test_blank_lines_survive_conversion(text):
    res = transliterate(text, "Hindi", llm=FakeLLM(_devanagari_for))
    assert res.text.count("\n") == text.count("\n")
    assert res.text.strip("\n").startswith("क्या")


def test_long_text_keeps_its_leading_blank_lines_across_chunks():
    text = "\n\n" + "\n".join(["kya haal hai aap sab"] * 200)
    res = transliterate(text, "Hindi", llm=FakeLLM(_devanagari_for))
    assert res.text.startswith("\n\nक्या") and res.text.count("\n") == text.count("\n")


@pytest.mark.parametrize("reply", ["Kya Haal Hai Aap", "KYA HAAL HAI AAP"])
def test_a_capitalised_roman_reply_is_not_a_conversion(reply):
    with pytest.raises(TransliterationError) as err:
        transliterate("kya haal hai aap", "Hindi", llm=FakeLLM(reply))
    assert err.value.code == "bad_output"


def test_names_the_source_capitalised_may_stay_in_latin():
    res = transliterate("Kya CEO ne YouTube par haal bataya", "Hindi",
                        llm=FakeLLM("Kya CEO ने YouTube पर हाल बताया"))
    assert res.text == "Kya CEO ने YouTube पर हाल बताया"
