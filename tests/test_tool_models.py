"""Music and sound-effect models give the GPU back to speech (and to each other).

Before this, MusicGen and AudioLDM2 stayed resident after a render: the
single-engine eviction only knew the TTS caches, so a card that fits one model
at a time ran later speech requests out of memory until the backend restarted.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import threading

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")

import pytest


def _tm():
    """services.tool_models at run time (other tests may reload ``services``)."""
    return importlib.import_module("services.tool_models")


def _em():
    return importlib.import_module("services.engine_memory")


@pytest.fixture
def registry(monkeypatch):
    monkeypatch.setattr(_tm(), "_REGISTRY", {})
    monkeypatch.setattr(importlib.import_module("services.model_manager"),
                        "release_device_cache", lambda **_: None)
    return _tm()._REGISTRY


def _slot(tool_id):
    loads = []

    def load():
        loads.append(1)
        return object()

    return _tm().ResidentModel(tool_id, load), loads


def test_an_idle_model_is_unloaded_and_reloads_on_next_use(registry):
    slot, loads = _slot("music")
    with slot.use():
        pass
    assert slot.unload() is True and not slot.loaded
    assert slot.unload() is False, "nothing left to free"
    with slot.use():
        pass
    assert len(loads) == 2


def test_a_model_mid_render_is_not_unloaded(registry):
    slot, _ = _slot("music")
    with slot.use():
        assert slot.unload() is False
        assert _tm().unload_all() == []
        assert slot.loaded
    assert _tm().unload_all() == ["music"]


def test_unload_all_keeps_the_tool_about_to_render(registry):
    music, _ = _slot("music")
    sfx, _ = _slot("sfx")
    music.get()
    sfx.get()
    assert _tm().unload_all(keep="sfx") == ["music"]
    assert sfx.loaded and not music.loaded


def test_a_render_racing_an_unload_keeps_its_model(registry):
    slot, _ = _slot("music")
    slot.get()
    entered, release = threading.Event(), threading.Event()

    def render():
        with slot.use() as model:
            entered.set()
            release.wait(5)
            assert model is not None

    worker = threading.Thread(target=render)
    worker.start()
    entered.wait(5)
    assert slot.unload() is False
    release.set()
    worker.join(5)


def test_a_tool_waits_for_the_other_and_loads_only_after_freeing_it(registry):
    music, _ = _slot("music")
    sfx, _ = _slot("sfx")
    entered, release = threading.Event(), threading.Event()
    seen = {}

    def render_music():
        with music.use():
            entered.set()
            release.wait(5)

    def render_sfx():
        with sfx.use():
            seen["music_loaded"] = music.loaded

    first = threading.Thread(target=render_music)
    first.start()
    entered.wait(5)
    second = threading.Thread(target=render_sfx)
    second.start()
    second.join(0.2)
    assert second.is_alive(), "sound effects must wait for the music render"
    release.set()
    first.join(5)
    second.join(5)
    assert seen == {"music_loaded": False}, "music is freed before sound effects load"


def _no_tts(monkeypatch):
    async def none(*_a, **_k):
        return []

    monkeypatch.setattr(_em(), "_evict_tts", none)


def test_speech_eviction_frees_idle_tool_models(registry, monkeypatch):
    _no_tts(monkeypatch)
    monkeypatch.setattr(importlib.import_module("services.tts_backend"), "get_backend_class",
                        lambda _id: object)
    music, _ = _slot("music")
    music.get()
    assert asyncio.run(_em().evict_other_tts_engines("voxcpm2")) == ["music"]
    assert not music.loaded


def test_a_tool_render_frees_speech_and_the_other_tool(registry, monkeypatch):
    seen = {}

    async def tts(keep_cls, keep_id):
        seen["args"] = (keep_cls, keep_id)
        return ["voxcpm2"]

    monkeypatch.setattr(_em(), "_evict_tts", tts)
    music, _ = _slot("music")
    sfx, _ = _slot("sfx")
    music.get()
    sfx.get()
    assert asyncio.run(_em().evict_for_tool("sfx")) == ["voxcpm2", "music"]
    assert seen["args"] == (None, "sfx"), "no TTS engine is kept for a tool render"
    assert sfx.loaded


def test_tool_eviction_respects_the_policy_switch(registry, monkeypatch):
    monkeypatch.setenv("OMNIVOICE_SINGLE_ENGINE_RESIDENT", "0")
    music, _ = _slot("music")
    music.get()
    assert asyncio.run(_em().evict_for_tool("sfx")) == []
    assert music.loaded


def test_both_tools_register_a_resident_model():
    from services import music_gen, sound_effects

    assert music_gen.MODEL.tool_id == "music"
    assert sound_effects.MODEL.tool_id == "sfx"
    assert _tm()._REGISTRY["music"] is music_gen.MODEL
    assert _tm()._REGISTRY["sfx"] is sound_effects.MODEL
