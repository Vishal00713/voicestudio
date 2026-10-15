"""Unit tests verifying stderr drain thread cleanup across subprocess sidecar lifecycle."""
import collections
import io
import sys
import threading
import time
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def mock_torch_if_missing(monkeypatch):
    if "torch" not in sys.modules:
        mock_t = MagicMock()
        mock_t.cuda.is_available.return_value = False
        monkeypatch.setitem(sys.modules, "torch", mock_t)


def _get_dummy_sidecar_class():
    from services.subprocess_backend import SubprocessBackend

    class DummySidecar(SubprocessBackend):
        id = "dummy"
        display_name = "Dummy Sidecar"

        @classmethod
        def venv_python(cls):
            return "/bin/python3"

        @classmethod
        def sidecar_script(cls):
            from pathlib import Path
            return Path("/tmp/dummy.py")

        @property
        def sample_rate(self):
            return 24000

        @property
        def supported_languages(self):
            return ["en"]

        @classmethod
        def is_available(cls):
            return True, "ok"

    return DummySidecar


def test_join_stderr_thread_closes_stderr_and_joins_alive_thread():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()
    mock_stderr = MagicMock()
    mock_proc = MagicMock()
    mock_proc.stderr = mock_stderr

    started_event = threading.Event()
    stop_event = threading.Event()

    def fake_drain():
        started_event.set()
        stop_event.wait(timeout=2.0)

    thread = threading.Thread(target=fake_drain, daemon=True)
    backend._stderr_thread = thread
    thread.start()
    started_event.wait(timeout=1.0)
    assert thread.is_alive()

    # Trigger stop and join
    stop_event.set()
    backend._join_stderr_thread(mock_proc, timeout=1.0)

    mock_stderr.close.assert_called_once()
    assert not thread.is_alive()
    assert backend._stderr_thread is None


def test_force_kill_cleans_up_stderr_thread():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()
    mock_stderr = MagicMock()
    mock_proc = MagicMock()
    mock_proc.stderr = mock_stderr
    mock_proc.kill = MagicMock()
    mock_proc.wait = MagicMock(return_value=0)

    backend._proc = mock_proc

    mock_thread = MagicMock()
    mock_thread.is_alive.side_effect = [True, False]
    backend._stderr_thread = mock_thread

    backend._force_kill()

    assert backend._proc is None
    assert backend._stderr_thread is None
    mock_stderr.close.assert_called_once()
    mock_thread.join.assert_called_once()


def test_shutdown_cleans_up_stderr_thread():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()
    mock_stderr = MagicMock()
    mock_proc = MagicMock()
    mock_proc.stderr = mock_stderr
    mock_proc.wait = MagicMock(return_value=0)

    backend._proc = mock_proc

    mock_thread = MagicMock()
    mock_thread.is_alive.side_effect = [True, False]
    backend._stderr_thread = mock_thread

    backend.shutdown()

    assert backend._proc is None
    assert backend._stderr_thread is None
    mock_stderr.close.assert_called_once()
    mock_thread.join.assert_called_once()


def test_join_stderr_thread_retains_reference_on_timeout():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()
    mock_stderr = MagicMock()
    mock_proc = MagicMock()
    mock_proc.stderr = mock_stderr

    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True
    backend._stderr_thread = mock_thread

    backend._join_stderr_thread(mock_proc, timeout=0.1)

    mock_thread.join.assert_called_once_with(timeout=0.1)
    # The reference is retained and stderr is not closed prematurely
    assert backend._stderr_thread is mock_thread
    mock_stderr.close.assert_not_called()


def test_join_stderr_thread_with_real_pipe_writer_open_retains_alive_reader():
    import os
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()

    r_fd, w_fd = os.pipe()
    r_file = os.fdopen(r_fd, "rb")

    mock_proc = MagicMock()
    mock_proc.stderr = r_file

    entered_readline = threading.Event()
    real_readline = r_file.readline

    def signaling_readline(*args, **kwargs):
        entered_readline.set()
        return real_readline(*args, **kwargs)

    r_file.readline = signaling_readline

    tail = collections.deque()
    thread = threading.Thread(
        target=backend._drain_stderr,
        args=(mock_proc, tail),
        daemon=True,
    )
    backend._stderr_thread = thread
    thread.start()

    assert entered_readline.wait(timeout=1.0)

    try:
        # With writer still open and no data written, readline is blocked on Linux.
        # Reader is confirmed inside readline; short join timeout (0.05s) will time out without thread exit.
        backend._join_stderr_thread(mock_proc, timeout=0.05)

        # Thread is still alive because writer is open; _stderr_thread must NOT be cleared.
        assert thread.is_alive()
        assert backend._stderr_thread is thread
    finally:
        # Close writer to unblock read and allow clean thread exit
        os.close(w_fd)
        thread.join(timeout=1.0)

    assert not thread.is_alive()
    # Calling join now clears it
    backend._join_stderr_thread(mock_proc, timeout=0.1)
    assert backend._stderr_thread is None


def test_drain_stderr_closes_stderr_in_finally():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()
    tail = collections.deque()
    mock_stderr = MagicMock()
    mock_stderr.readline = MagicMock(return_value=b"")
    mock_proc = MagicMock()
    mock_proc.stderr = mock_stderr

    backend._drain_stderr(mock_proc, tail)

    mock_stderr.close.assert_called_once()


def test_spawn_reuses_healthy_running_sidecar_with_active_reader():
    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Process is running and healthy
    backend._proc = mock_proc

    mock_thread = MagicMock()
    mock_thread.is_alive.return_value = True  # Reader is actively draining
    backend._stderr_thread = mock_thread

    # _retry_timeout_cleanup succeeds without interfering with active process/reader
    assert backend._retry_timeout_cleanup() is True
    assert backend._stderr_thread is mock_thread

    # _spawn returns immediately without raising or killing
    backend._spawn()
    assert backend._proc is mock_proc
    assert backend._stderr_thread is mock_thread


def test_spawn_rejects_reuse_when_process_is_quarantined():
    import subprocess as sp

    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None  # Running child
    mock_proc.wait.side_effect = sp.TimeoutExpired(cmd="dummy", timeout=2)
    backend._proc = mock_proc
    with backend._timeout_quarantine_lock:
        backend._timeout_quarantine.append(mock_proc)

    with pytest.raises(RuntimeError, match="still stopping after a timeout"):
        backend._spawn()


def test_shutdown_quarantines_unreaped_child():
    import subprocess as sp

    DummySidecar = _get_dummy_sidecar_class()
    backend = DummySidecar()

    mock_proc = MagicMock()
    mock_proc.poll.return_value = None
    mock_proc.wait.side_effect = sp.TimeoutExpired(cmd="dummy", timeout=1)
    backend._proc = mock_proc

    backend.shutdown()

    assert backend._proc is None
    with backend._timeout_quarantine_lock:
        assert mock_proc in backend._timeout_quarantine

