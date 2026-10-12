# Music

**Tools → Music** turns a description of style, mood and instruments into an
instrumental piece (5–30 s, stereo, 32 kHz), rendered on this computer — the
local counterpart of ElevenLabs Music. Each run gives 1–3 versions.

## Download (once)

The feature is off until you click **Download music model**, which fetches
**MusicGen stereo-medium** (`facebook/musicgen-stereo-medium`, ~4.1 GB of
safetensors) into the shared Hugging Face model cache. MusicGen ships with
`transformers`, so nothing else is installed. The model is licensed
**CC-BY-NC-4.0: non-commercial use only**.

## Use

- Describe the music **in English**: genre, mood, instruments, tempo —
  *"Upbeat Bollywood dance track with dhol, tabla and sitar"*.
- Instrumental only: MusicGen does not sing lyrics.
- Output is watermarked per channel (AudioSeal is a mono model), saved to the
  outputs folder and recorded in history with mode `music`.

On an RTX 5080 two 12-second versions render in ~35 s and one 8-second
version in ~8 s once the model is loaded (~4 GB VRAM in fp16).

## API

| Endpoint | Purpose |
|---|---|
| `GET /music/status` | `ready`, download `stage`, licence, size |
| `POST /music/install` | Start the model download in the background (idempotent) |
| `POST /music/generate` | Form fields `prompt`, `seconds` (5–30), `variations` (1–3), optional `seed` → `{takes: [{id, audio_url, duration_s}]}`; `409 music_not_installed` until downloaded |

Cold loads of this model, the sound-effects model and the speech model are
serialised (`services/model_loading.py`): loading two transformers models at
once could leave weights on the `meta` device.

Implementation: `backend/services/music_gen.py`, `backend/api/routers/music.py`,
`electron/src/renderer/src/features/tools/audio-generator.tsx`.
