"""ElevenLabs-style audio tags lower to what each engine can perform (Spec 01)."""

import pytest

from services.audio_tags import lower_audio_tags, with_whisper


def test_plain_text_is_untouched():
    assert lower_audio_tags("Hello there.", "omnivoice").text == "Hello there."


@pytest.mark.parametrize(
    "cue, native",
    [("[laughs]", "[laughter]"), ("[Chuckles]", "[laughter]"), ("[sighs]", "[sigh]")],
)
def test_reactions_become_the_native_omnivoice_token(cue, native):
    result = lower_audio_tags(f"Well {cue} okay.", "omnivoice")
    assert result.text == f"Well {native} okay."
    assert result.dropped == []


def test_cosyvoice_gets_its_own_breath_token():
    result = lower_audio_tags("Run [inhales deeply] faster.", "cosyvoice3", "freeform")
    assert result.text == "Run [breath] faster."


def test_reaction_without_a_native_token_is_dropped_and_reported():
    result = lower_audio_tags("Run [inhales] faster.", "omnivoice")
    assert result.text == "Run faster."
    assert result.dropped == ["[inhales]"]


def test_pause_cues_become_engine_agnostic_markers():
    result = lower_audio_tags("Wait [short pause] now [long pause] go.", "kittentts", "freeform")
    assert result.text == "Wait [pause 300ms] now [pause 1000ms] go."


def test_unsupported_delivery_cues_are_removed_not_spoken():
    result = lower_audio_tags("[excited] We did it! [sarcastic] Great.", "omnivoice")
    assert result.text == "We did it! Great."
    assert result.dropped == ["[excited]", "[sarcastic]"]


def test_whisper_becomes_an_instruct_style_where_the_engine_takes_one():
    result = lower_audio_tags("[whispers] Don't tell anyone.", "omnivoice")
    assert result.text == "Don't tell anyone."
    assert result.whisper is True
    assert result.dropped == []


def test_whisper_after_other_opening_cues_still_covers_the_take():
    result = lower_audio_tags("[excited] [whispers] Psst.", "omnivoice")
    assert result.whisper is True
    assert result.text == "Psst."


def test_mid_script_whisper_is_dropped_instead_of_whispering_everything():
    result = lower_audio_tags("Hello friends! [whispers] This is a secret.", "omnivoice")
    assert result.whisper is False
    assert result.text == "Hello friends! This is a secret."
    assert result.dropped == ["[whispers]"]


def test_whisper_is_reported_where_no_instruct_is_read():
    result = lower_audio_tags("[whispers] Hi.", "kittentts", "freeform", takes_instruct=False)
    assert result.whisper is False
    assert result.dropped == ["[whispers]"]


@pytest.mark.parametrize(
    "text",
    [
        "Native [laughter] and [sigh] stay.",
        "Pause [pause 500ms] stays.",
        "Speaker [voice:Ana] stays.",
        "Phonemes [B EY1 S] stay.",
        "Override [[Nuh-VAD-uh]] stays.",
        "Unknown [banana] stays.",
    ],
)
def test_spans_outside_the_closed_set_pass_through(text):
    assert lower_audio_tags(text, "omnivoice").text == text


def test_removed_cue_does_not_leave_a_space_before_punctuation():
    assert lower_audio_tags("नमस्ते [happily]।", "omnivoice").text == "नमस्ते।"


def test_with_whisper_adds_the_style_once():
    assert with_whisper("female, young adult", "tags") == "female, young adult, whisper"
    assert with_whisper("female, whisper", "tags") == "female, whisper"
    assert with_whisper(None, "tags") == "whisper"
    assert with_whisper("", "freeform") == "whispering softly"
