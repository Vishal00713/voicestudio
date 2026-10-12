# Commercial-safe mode

Many voice, music and sound-effect models ship code under one licence and **weights** under another. OmniVoice's
code is Apache-2.0, but its pre-trained model is CC-BY-NC (because of its training data); MusicGen is
CC-BY-NC-4.0 and AudioLDM2 is CC-BY-NC-SA-4.0. If you publish to monetized channels, output from those weights is
not cleared for that use.

Commercial-safe mode is an **opt-in** switch. It is off by default and changes nothing until you turn it on.

## What it does when on

Every generation path checks the weights' licence before any work starts:

- `POST /generate`, the live stream (`/ws/tts`), dubbing, batch, previews and voice conversion
- `POST /v1/audio/speech` (OpenAI-compatible; refusals use OpenAI's error shape with
  `code: "commercial_use_blocked"`)
- Tools → Music (`/music/generate`) and Sound effects (`/sfx/generate`), and their one-time model
  downloads (`/music/install`, `/sfx/install`): `/music/status` and `/sfx/status` report
  `commercial_blocked: true` and the app explains why instead of offering the download

Allowed: engines whose weights are `commercial` (e.g. Apache-2.0: VoxCPM2, CosyVoice 3, KittenTTS, MOSS-TTS-Nano)
or `conditional` (commercial use allowed under the licence's conditions — IndexTTS-2, Supertonic 3, PocketTTS;
read them). Refused with HTTP 403: `non_commercial` weights (OmniVoice in every form, audio.cpp, MusicGen,
AudioLDM2) and `unknown` (GPT-SoVITS, MLX-Audio and sherpa-onnx, whose licence depends on the model you load,
plus any engine not reviewed yet — fail-safe).

The 403 body: `{"detail": {"code": "commercial_use_blocked", "message", "target", "status", "weights",
"source"}}`. A `/ws/tts` error frame carries the same fields next to `type: "error"` and `detail`.

## Turning it on

- API: `POST /engines/commercial-only {"enabled": true}` (admin action), read state with `GET /engines/licenses`.
- Environment: `OMNIVOICE_COMMERCIAL_ONLY=1` (wins over the saved choice — useful for an unattended content
  pipeline). `GET /engines/licenses` reports `env_pinned`.
- `GET /engines/tts` adds `weights_license` to every engine.

## The licence table

`backend/services/license_guard.py` holds one entry per engine and tool with the licence name, where it is stated,
and the date it was reviewed. A test fails when an engine is added without an entry. Statuses are the maintainers'
reading of each model card; they are guidance, not legal advice — the linked licence is authoritative.
