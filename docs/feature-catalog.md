# Features and engines

Engine availability depends on installed models, hardware, and configured providers.

## Features

- **Voice Cloning** - [Audio quality, export precision, and local checks](audio-quality.md)
- **Voice Design** - three candidate voices per description to compare and keep
- **Audio Tags** - [ElevenLabs-style `[laughs]` / `[whispers]` / `[short pause]` cues](expressive-speech.md#elevenlabs-style-tags)
- **Sound Effects** - [Text → SFX with AudioLDM2, installed on request](sound-effects.md)
- **Music** - [Text → instrumental music with MusicGen, downloaded on request](music.md)
- **Video Dubbing**
- **Dictation Widget**
- **Vocal Isolation**
- **Speaker Diarization**
- **Batch Queue**
- **MCP Server**
- **AI Watermark**
- **Local-first**
- **GPU Auto-Detect**
- **Remote Model Downloads**
- **Extensible**

## Speech generation

- **VoiceStudio** (default, powered by k2-fsa/OmniVoice)
- omnivoice-subprocess — [Guide](engines/omnivoice-subprocess.md)
- CosyVoice 3 — [Guide](engines/cosyvoice.md)
- KittenTTS
- MLX-Audio
- VoxCPM2
- MOSS-TTS-Nano
- gpt-sovits
- sherpa-onnx
- **IndexTTS 2.5** ⚡ — [Guide](engines/indextts.md)
- omnivoice-gguf
- supertonic3
- **MOSS-TTS-v1.5** — [Guide](engines/moss-tts-v15.md)
- **dots.tts** — [Guide](engines/dots-tts.md)
- **Confucius4-TTS** — [Guide](engines/confucius4-tts.md)
- pockettts
- audiocpp — [Guide](engines/audio-cpp.md)

## Transcription

- **WhisperX** (default)
- Faster-Whisper
- MLX Whisper
- PyTorch Whisper
- Parakeet TDT
- Parakeet TDT v3 (MLX)
- Moonshine
- FunASR
- **sherpa-onnx** (live dictation)
- **OpenAI-compatible** ⚠️ configured server
