"""SoniTranslate sidecar integration.

Manages an isolated SoniTranslate instance that runs as a Gradio service
on port 7860. VoiceStudio calls it via `gradio_client` for full-pipeline
video dubbing with access to Edge TTS, Piper, Coqui XTTS, and RVC.
"""

import asyncio
import collections
import logging
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from services.ffmpeg_utils import spawn_subprocess
from core.logging_utils import log_safe
from core.path_security import UnsafePath, resolve_within, safe_filename

logger = logging.getLogger("omnivoice.sonitranslate")

# Default install location — inside the VoiceStudio project tree
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SONI_DIR = _PROJECT_ROOT / "engines" / "sonitranslate"
SONI_VENV = SONI_DIR / ".venv"
SONI_PORT = 7860
SONI_URL = f"http://127.0.0.1:{SONI_PORT}"

# Track subprocess and log drainer
_proc: Optional[subprocess.Popen] = None
_output_tail: collections.deque[str] = collections.deque(maxlen=100)
_drain_thread: Optional[threading.Thread] = None
_lifecycle_lock: Optional[asyncio.Lock] = None
_startup_task: Optional[asyncio.Task] = None


def _get_lifecycle_lock() -> asyncio.Lock:
    global _lifecycle_lock
    if _lifecycle_lock is None:
        _lifecycle_lock = asyncio.Lock()
    return _lifecycle_lock


def _redact_home_paths(text: str) -> str:
    try:
        home = str(Path.home())
        if home and home != "/":
            return text.replace(home, "~")
    except Exception:
        pass
    return text


def _wait_and_reap(proc: subprocess.Popen, drain_thread: Optional[threading.Thread]) -> None:
    """Reap child process and join output reader in worker thread off event loop."""
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                pass
    if proc.stdout:
        try:
            proc.stdout.close()
        except Exception:
            pass
    if drain_thread and drain_thread.is_alive():
        drain_thread.join(timeout=2)


def _drain_output(proc: subprocess.Popen, tail: collections.deque[str]) -> None:
    """Continuously read stdout/stderr from proc into a bounded ring buffer to prevent pipe deadlock."""
    try:
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, b""):
            text = line.decode("utf-8", errors="replace").rstrip()
            tail.append(text)
    except Exception:
        pass
    finally:
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass


def _venv_bin(name: str):
    """Path to an executable inside the SoniTranslate venv, cross-platform.
    Windows venvs put executables in Scripts\\ (with a .exe suffix); POSIX uses
    bin/. (Matches engines/indextts/bootstrap.py's _venv_python_path.)"""
    if sys.platform == "win32":
        return SONI_VENV / "Scripts" / f"{name}.exe"
    return SONI_VENV / "bin" / name


def is_installed() -> bool:
    """Check if SoniTranslate is cloned and has its entry point."""
    return (SONI_DIR / "app_rvc.py").is_file()


def is_venv_ready() -> bool:
    """Check if the SoniTranslate virtualenv exists with key deps."""
    pip = _venv_bin("pip")
    return pip.is_file()


def is_running() -> bool:
    """Check if the Gradio server is responding."""
    global _proc
    if _proc is not None and _proc.poll() is not None:
        _proc = None
    try:
        import httpx
        r = httpx.get(f"{SONI_URL}/info", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


def status() -> dict:
    """Return full status object for the frontend."""
    installed = is_installed()
    return {
        "installed": installed,
        "venv_ready": is_venv_ready() if installed else False,
        "running": is_running() if installed else False,
        "path": str(SONI_DIR),
        "url": SONI_URL,
    }


async def install(progress_callback=None) -> dict:
    """Clone SoniTranslate and set up its virtualenv.

    This is a heavy operation (~15GB with models). Runs in background.
    """
    if not is_installed():
        logger.info("Cloning SoniTranslate...")
        if progress_callback:
            progress_callback("Cloning SoniTranslate repository...")
        proc = await spawn_subprocess(
            "git", "clone", "--depth", "1",
            "https://github.com/R3gm/SoniTranslate.git",
            str(SONI_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Clone failed: {stderr.decode()}")

    # Create venv if needed
    if not is_venv_ready():
        logger.info("Creating SoniTranslate virtualenv...")
        if progress_callback:
            progress_callback("Creating virtualenv...")

        python = sys.executable
        proc = await spawn_subprocess(
            python, "-m", "venv", str(SONI_VENV),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

        # Install base requirements
        pip = str(_venv_bin("pip"))
        if progress_callback:
            progress_callback("Installing base requirements (this may take a while)...")

        proc = await spawn_subprocess(
            pip, "install", "-r", str(SONI_DIR / "requirements_base.txt"),
            cwd=str(SONI_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("Base requirements failed: %s", stderr.decode()[-500:])

        # Install extra requirements
        if progress_callback:
            progress_callback("Installing extra requirements...")
        proc = await spawn_subprocess(
            pip, "install", "-r", str(SONI_DIR / "requirements_extra.txt"),
            cwd=str(SONI_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()

    return status()


async def _start_internal() -> dict:
    global _proc, _output_tail, _drain_thread
    if not is_installed():
        raise RuntimeError("SoniTranslate not installed. Call /engines/sonitranslate/install first.")

    python = str(_venv_bin("python")) if is_venv_ready() else sys.executable

    from services import engine_env, token_resolver
    resolved = token_resolver.resolve()
    env = engine_env.build_engine_env()
    from services.hf_auth import env_allows_token
    if resolved and resolved.token and env_allows_token(env):
        env["HF_TOKEN"] = resolved.token
        env["YOUR_HF_TOKEN"] = resolved.token

    _output_tail.clear()
    logger.info("Starting SoniTranslate on port %d...", SONI_PORT)
    proc = subprocess.Popen(
        [python, "app_rvc.py"],
        cwd=str(SONI_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    _proc = proc
    drain_thread = threading.Thread(
        target=_drain_output,
        args=(proc, _output_tail),
        daemon=True,
        name="sonitranslate-stdout-drain",
    )
    _drain_thread = drain_thread
    drain_thread.start()

    try:
        for _ in range(60):
            await asyncio.sleep(0.5)
            if _proc is not proc:
                raise RuntimeError("SoniTranslate startup was interrupted by stop request")
            if is_running():
                logger.info("SoniTranslate started successfully")
                return {"started": True, **status()}
            if proc.poll() is not None:
                recent_logs = _redact_home_paths("\n".join(_output_tail))
                raise RuntimeError(f"SoniTranslate exited early: {recent_logs}")

        recent_logs = _redact_home_paths("\n".join(_output_tail))
        raise RuntimeError(f"SoniTranslate failed to start within 30s: {recent_logs}")
    except asyncio.CancelledError:
        if _proc is proc:
            _proc = None
            _drain_thread = None
        await asyncio.to_thread(_wait_and_reap, proc, drain_thread)
        raise RuntimeError("SoniTranslate startup was interrupted by stop request")
    except Exception:
        if _proc is proc:
            _proc = None
            _drain_thread = None
        await asyncio.to_thread(_wait_and_reap, proc, drain_thread)
        raise


async def start() -> dict:
    """Start the SoniTranslate Gradio server as a subprocess."""
    global _startup_task

    async with _get_lifecycle_lock():
        if is_running():
            return {"started": False, "reason": "already_running", **status()}

        if _startup_task is not None and not _startup_task.done():
            task = _startup_task
        else:
            task = asyncio.create_task(_start_internal())
            _startup_task = task

    return await asyncio.shield(task)


async def stop() -> dict:
    """Stop the SoniTranslate subprocess."""
    async with _get_lifecycle_lock():
        global _proc, _drain_thread, _startup_task
        task = _startup_task
        proc = _proc
        drain_thread = _drain_thread
        _proc = None
        _drain_thread = None
        _startup_task = None

        if proc is None and (task is None or task.done()):
            return {"stopped": False, "reason": "not_running"}

        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        if proc is None and _proc is not None:
            proc = _proc
            drain_thread = _drain_thread
            _proc = None
            _drain_thread = None

        if proc is not None:
            await asyncio.to_thread(_wait_and_reap, proc, drain_thread)

        logger.info("SoniTranslate stopped")
        return {"stopped": True}


async def dub_video(
    video_path: str,
    target_language: str = "Spanish (es)",
    source_language: str = "Automatic detection",
    tts_voice: str = "es-ES-AlvaroNeural-Male",
    max_speakers: int = 1,
    output_dir: Optional[str] = None,
) -> dict:
    """Run the full SoniTranslate dubbing pipeline on a video file.

    Returns the path to the dubbed output file.
    """
    if not is_running():
        await start()

    try:
        from gradio_client import Client, handle_file
    except ImportError:
        raise RuntimeError(
            "gradio_client not installed. Run: pip install gradio_client"
        )

    logger.info("Submitting dub job to SoniTranslate (target=%s)", log_safe(target_language))

    client = Client(SONI_URL)

    # Phase 1 AUTH-01: resolve from the 3-source cascade. Empty-string
    # fallback preserves SoniTranslate's library-side behaviour when no
    # token is available (it skips diarization there too).
    from services import token_resolver
    _resolved_for_soni = token_resolver.resolve()
    _hf_token_for_soni = _resolved_for_soni.token if _resolved_for_soni else ""
    from services.hf_auth import env_allows_token
    if not env_allows_token(os.environ):
        _hf_token_for_soni = ""  # a mirror never receives the HF token

    # The main function is `batch_multilingual_media_conversion`
    # which is exposed as the first API endpoint
    result = client.predict(
        handle_file(video_path),  # media_file
        "",                       # link_media
        "",                       # directory_input
        _hf_token_for_soni,       # YOUR_HF_TOKEN
        False,                    # preview
        "large-v3",               # transcriber_model
        4,                        # batch_size
        "auto",                   # compute_type
        source_language,          # origin_language
        target_language,          # target_language
        1,                        # min_speakers
        max_speakers,             # max_speakers
        tts_voice,                # tts_voice00
        tts_voice,                # tts_voice01 (fallback same)
        tts_voice,                # tts_voice02
        tts_voice,                # tts_voice03
        tts_voice,                # tts_voice04
        tts_voice,                # tts_voice05
        tts_voice,                # tts_voice06
        tts_voice,                # tts_voice07
        tts_voice,                # tts_voice08
        tts_voice,                # tts_voice09
        tts_voice,                # tts_voice10
        tts_voice,                # tts_voice11
        "",                       # video_output_name
        "Adjusting volumes and mixing audio",  # mix_method_audio
        2.1,                      # max_accelerate_audio
        False,                    # acceleration_rate_regulation
        0.25,                     # volume_original_audio
        1.80,                     # volume_translated_audio
        "srt",                    # output_format_subtitle
        False,                    # get_translated_text
        False,                    # get_video_from_text_json
        "{}",                     # text_json
        False,                    # avoid_overlap
        False,                    # vocal_refinement
        True,                     # literalize_numbers
        15,                       # segment_duration_limit
        "pyannote_3.1",           # diarization_model
        "google_translator_batch",  # translate_process
        None,                     # subtitle_file
        "video (mp4)",            # output_type
        False,                    # voiceless_track
        False,                    # voice_imitation
        3,                        # voice_imitation_max_segments
        False,                    # voice_imitation_vocals_dereverb
        True,                     # voice_imitation_remove_previous
        "freevc",                 # voice_imitation_method
        True,                     # dereverb_automatic_xtts
        "sentence",               # text_segmentation_scale
        "",                       # divide_text_segments_by
        True,                     # soft_subtitles_to_video
        False,                    # burn_subtitles_to_video
        True,                     # enable_cache
        False,                    # custom_voices
        1,                        # custom_voices_workers
        False,                    # is_gui
        api_name="/batch_multilingual_media_conversion",
    )

    # Result is the output file path(s)
    if isinstance(result, list):
        output_file = result[0] if result else None
    else:
        output_file = result

    if output_file and output_dir:
        output_root = Path(output_dir).expanduser()
        if not output_root.is_absolute() or not output_root.is_dir():
            raise ValueError("output_dir must be an existing absolute directory")
        try:
            output_name = safe_filename(os.path.basename(output_file))
            dest = resolve_within(output_root, output_name)
        except UnsafePath as exc:
            raise ValueError("SoniTranslate returned an invalid output filename") from exc
        shutil.copy2(output_file, str(dest))
        output_file = str(dest)

    logger.info("SoniTranslate dub completed")
    return {
        "output_file": output_file,
        "target_language": target_language,
        "source_language": source_language,
    }
"""Backend service wrapper for SoniTranslate sidecar."""
