# Sound effects

**Tools → Sound effects** turns a text description into sound effects —
rain, footsteps, a door creak, a cinematic whoosh — rendered on this computer.
Each run gives 1–4 takes to choose from, like ElevenLabs Sound Effects.

## Install (once)

The feature is off until you click **Install sound effects**. That one click:

1. installs `diffusers` (pure Python, `--no-deps`) into its own folder,
   `<data dir>/sfx_libs` — the app's Python runtime is not modified, and
   deleting that folder removes it;
2. downloads the **AudioLDM2** model (`cvssp/audioldm2`, ~4.5 GB of
   safetensors) into the shared Hugging Face model cache.

Nothing is downloaded before that click. AudioLDM2 is licensed
**CC-BY-NC-SA-4.0: non-commercial use only** — check the licence before using
the output commercially.

## Use

- Describe the sound **in English** (the model's text encoder is English).
  Concrete, physical descriptions work best: *"Footsteps on gravel"*,
  *"A creaking wooden door slowly opening"*.
- Length: 1–20 seconds. Takes: 1–4 per run; they are ranked by how well they
  match the description, best first.
- Output is 16 kHz mono WAV, watermarked like every other generated clip
  (`mark_synthetic`), saved to the outputs folder and recorded in history with
  mode `sfx`.

On an RTX 5080 the first run takes ~30 s (model load); after that four
5-second takes render in ~5 s. The model shares the GPU with the speech
engine (about 3 GB extra VRAM in fp16) and also runs on CPU, slowly.

## API

| Endpoint | Purpose |
|---|---|
| `GET /sfx/status` | `ready`, install `stage` (`idle`/`installing`/`downloading`/`ready`/`error`), licence, download size |
| `POST /sfx/install` | Start the install in the background (idempotent) |
| `POST /sfx/generate` | Form fields `prompt`, `seconds` (1–20), `variations` (1–4), optional `seed` → `{takes: [{id, audio_url, duration_s}]}`; `409 sfx_not_installed` until installed |

Implementation: `backend/services/sound_effects.py`,
`backend/api/routers/sound_effects.py`,
`electron/src/renderer/src/features/tools/audio-generator.tsx`.
