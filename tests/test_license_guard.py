"""Commercial-safe mode (services/license_guard.py): weights licences per engine/tool and the opt-in switch that
refuses non-commercial or unreviewed weights on every generation path."""
import asyncio
import importlib
import os

import pytest

os.environ.setdefault("OMNIVOICE_MODEL", "test")
os.environ.setdefault("OMNIVOICE_DISABLE_FILE_LOG", "1")


@pytest.fixture
def guard(tmp_path, monkeypatch):
    from core import prefs

    monkeypatch.setattr(prefs, "_PREFS_PATH", str(tmp_path / "prefs.json"))
    monkeypatch.delenv("OMNIVOICE_COMMERCIAL_ONLY", raising=False)
    return importlib.import_module("services.license_guard")


def test_every_tts_engine_has_a_reviewed_licence(guard):
    """A new engine must get a licence entry before it ships — otherwise Commercial-safe mode would treat it
    as unknown and users would not see why."""
    tts = importlib.import_module("services.tts_backend")
    engine_ids = set(tts._INSTALL_HINTS) | set(dict(tts._REGISTRY.items()) if hasattr(tts._REGISTRY, "items") else {})
    missing = sorted(e for e in engine_ids if e not in guard.ENGINE_LICENSES)
    assert not missing, f"add these engines to services/license_guard.ENGINE_LICENSES: {missing}"
    for engine_id, info in guard.ENGINE_LICENSES.items():
        assert info.status in (guard.COMMERCIAL, guard.CONDITIONAL, guard.NON_COMMERCIAL, guard.UNKNOWN), engine_id
        assert info.weights and info.source, engine_id


def test_off_by_default_changes_nothing(guard):
    assert guard.commercial_only() is False
    guard.check_engine("omnivoice")
    guard.check_tool("music")


def test_on_refuses_non_commercial_and_unknown_allows_permissive_and_conditional(guard):
    guard.set_commercial_only(True)
    assert guard.commercial_only() is True
    for refused in ("omnivoice", "omnivoice-subprocess", "omnivoice-gguf", "audiocpp", "gpt-sovits",
                    "engine-nobody-reviewed"):
        with pytest.raises(guard.CommercialUseBlocked) as info:
            guard.check_engine(refused)
        assert info.value.target == refused
        assert "voxcpm2" in str(info.value)  # the message names the commercial-safe alternatives
    for allowed in ("voxcpm2", "cosyvoice", "kittentts", "moss-tts-nano", "indextts2", "pockettts"):
        guard.check_engine(allowed)
    for tool in ("music", "sfx"):
        with pytest.raises(guard.CommercialUseBlocked):
            guard.check_tool(tool)


def test_env_pin_wins_over_the_saved_choice(guard, monkeypatch):
    guard.set_commercial_only(True)
    monkeypatch.setenv("OMNIVOICE_COMMERCIAL_ONLY", "0")
    assert guard.commercial_only() is False
    monkeypatch.setenv("OMNIVOICE_COMMERCIAL_ONLY", "1")
    guard.set_commercial_only(False)
    assert guard.commercial_only() is True
    assert guard.payload()["env_pinned"] is True


def test_payload_lists_engines_and_tools(guard):
    data = guard.payload()
    assert data["commercial_only"] is False and data["reviewed"]
    assert data["engines"]["omnivoice"]["status"] == "non_commercial"
    assert data["engines"]["voxcpm2"] == {"status": "commercial", "weights": "Apache-2.0",
                                          "source": "https://huggingface.co/openbmb/VoxCPM2", "note": ""}
    assert data["tools"]["music"]["status"] == "non_commercial"


def test_shared_generation_gate_refuses_before_loading_anything(guard, monkeypatch):
    """batch, dubbing, previews and voice conversion all resolve through resolve_generation_backend."""
    tts = importlib.import_module("services.tts_backend")
    monkeypatch.setattr(tts, "active_backend_id", lambda: "omnivoice")
    monkeypatch.setattr(tts, "get_backend_class", lambda _id: pytest.fail("must refuse before loading"))
    guard.set_commercial_only(True)
    with pytest.raises(guard.CommercialUseBlocked):
        asyncio.run(tts.resolve_generation_backend())


def test_openai_speech_refuses_in_openai_error_shape(guard, monkeypatch):
    oa = importlib.import_module("api.routers.openai_compat")
    tts = importlib.import_module("services.tts_backend")
    monkeypatch.setattr(tts, "active_backend_id", lambda: "omnivoice")
    guard.set_commercial_only(True)
    for model in ("tts-1", "omnivoice"):  # OpenAI alias → active engine, and the engine id itself
        with pytest.raises(oa.OpenAIError) as info:
            oa._resolve_engine(model)
        assert info.value.status_code == 403 and info.value.code == "commercial_use_blocked"


def test_music_and_sfx_routes_refuse_before_install_checks(guard):
    music = importlib.import_module("api.routers.music")
    sfx = importlib.import_module("api.routers.sound_effects")
    guard.set_commercial_only(True)
    with pytest.raises(guard.CommercialUseBlocked):
        asyncio.run(music.music_generate(prompt="calm piano", seconds=5.0, variations=1, seed=None))
    with pytest.raises(guard.CommercialUseBlocked):
        asyncio.run(sfx.sfx_generate(prompt="rain", seconds=2.0, variations=1, seed=None))


def test_blocked_error_becomes_a_403_with_a_code(guard):
    import main
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    handler = main.app.exception_handlers[guard.CommercialUseBlocked]
    app = FastAPI()
    app.add_exception_handler(guard.CommercialUseBlocked, handler)

    @app.get("/x")
    def boom():
        guard.set_commercial_only(True)
        guard.check_engine("omnivoice")

    body = TestClient(app).get("/x")
    assert body.status_code == 403
    detail = body.json()["detail"]
    assert detail["code"] == "commercial_use_blocked" and detail["target"] == "omnivoice"
    assert detail["status"] == "non_commercial" and "CC-BY-NC" in detail["weights"]


def test_websocket_stream_refusal_carries_the_code(guard):
    """/ws/tts sends the same refusal fields as the HTTP 403, so a stream client can tell it apart."""
    from fastapi.testclient import TestClient
    from main import app

    guard.set_commercial_only(True)
    with TestClient(app, client=("127.0.0.1", 50000)).websocket_connect("/ws/tts") as ws:
        ws.send_json({"text": "hello", "engine": "omnivoice"})
        frame = ws.receive_json()
    assert frame["type"] == "error" and frame["code"] == "commercial_use_blocked"
    assert frame["target"] == "omnivoice" and frame["status"] == "non_commercial"


def test_tool_installs_are_refused_and_status_says_so(guard, monkeypatch):
    """No multi-GB download of weights Commercial-safe mode would refuse to run."""
    music = importlib.import_module("api.routers.music")
    sfx = importlib.import_module("api.routers.sound_effects")
    started = []
    monkeypatch.setattr(music.music_gen, "start_install", lambda: started.append("music"))
    monkeypatch.setattr(sfx.sfx, "start_install", lambda: started.append("sfx"))
    monkeypatch.setattr(music.music_gen, "model_downloaded", lambda: False)
    monkeypatch.setattr(sfx.sfx, "model_downloaded", lambda: False)
    assert music.music_status()["commercial_blocked"] is False
    assert sfx.sfx_status()["commercial_blocked"] is False

    guard.set_commercial_only(True)
    assert music.music_status()["commercial_blocked"] is True
    assert sfx.sfx_status()["commercial_blocked"] is True
    with pytest.raises(guard.CommercialUseBlocked):
        music.music_install()
    with pytest.raises(guard.CommercialUseBlocked):
        sfx.sfx_install()
    assert started == []
