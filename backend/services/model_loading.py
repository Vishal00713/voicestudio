"""One cold model load at a time, shared with the speech engine.

transformers builds a model on the ``meta`` device and then fills in the
weights. When another load runs at the same moment — the speech model's
startup preload is the usual one — a model can come out with tensors still on
``meta`` ("Cannot copy out of meta tensor" on ``.to()``), and that worker keeps
failing afterwards. The speech engine already serialises its own cold loads on
``model_manager._model_load_thread_lock`` (#1417); optional generators take
the same lock, with a deadline so a wedged load fails loudly instead of
hanging the request.
"""

from __future__ import annotations

from contextlib import contextmanager

LOAD_WAIT_S = 600.0


@contextmanager
def exclusive_model_load(what: str, timeout: float = LOAD_WAIT_S):
    from services.model_manager import _model_load_thread_lock

    if not _model_load_thread_lock.acquire(timeout=timeout):
        raise RuntimeError(
            f"{what}: another model is still loading after {int(timeout)} s. "
            "Try again once it finishes."
        )
    try:
        yield
    finally:
        _model_load_thread_lock.release()
