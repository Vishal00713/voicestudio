"""Tool models (music, sound effects) that hand their memory back when idle.

The single-engine policy (``services.engine_memory``) frees resident TTS models
before another one loads. MusicGen and AudioLDM2 live outside the TTS caches,
so without this they would stay on the GPU after a render and a later speech
request on a card that fits only one of them would run out of memory.

Each tool keeps its model in a :class:`ResidentModel`. ``unload()`` drops it
unless a render is running; ``unload_all(keep=...)`` is what the eviction
paths call.
"""

from __future__ import annotations

import gc
import logging
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger("omnivoice.tool_models")

_REGISTRY: dict[str, "ResidentModel"] = {}

#: One tool render at a time. Held across load and render, so the model a
#: render needs is loaded only after every other tool model is gone: the
#: route's eviction cannot free a model that is mid-render, but this can wait
#: for it. Torch's global RNG is seeded per render, which this also keeps
#: from being reseeded by the other tool in between.
_RENDER_LOCK = threading.Lock()


class ResidentModel:
    def __init__(self, tool_id: str, loader: Callable[[], Any]):
        self.tool_id = tool_id
        self._loader = loader
        self._value: Any = None
        self._load_lock = threading.Lock()
        self._use_lock = threading.Lock()
        self._in_use = 0
        _REGISTRY[tool_id] = self

    @property
    def loaded(self) -> bool:
        return self._value is not None

    def get(self) -> Any:
        with self._load_lock:
            if self._value is None:
                self._value = self._loader()
            return self._value

    @contextmanager
    def use(self) -> Iterator[Any]:
        """The loaded model, kept resident until the block ends.

        Waits for any other tool's render, and frees the other tool models
        before loading this one."""
        with _RENDER_LOCK:
            with self._use_lock:
                self._in_use += 1
            try:
                if not self.loaded:
                    unload_all(keep=self.tool_id)
                yield self.get()
            finally:
                with self._use_lock:
                    self._in_use -= 1

    def unload(self) -> bool:
        """Drop the model unless a render holds it; True when memory was freed."""
        with self._load_lock:
            with self._use_lock:
                if self._value is None or self._in_use:
                    return False
            self._value = None
        gc.collect()
        try:
            from services.model_manager import release_device_cache

            release_device_cache()
        except Exception:  # noqa: BLE001 — freeing memory must never raise
            logger.debug("releasing the device cache after %s failed", self.tool_id)
        return True


def unload_all(keep: Optional[str] = None) -> list[str]:
    """Unload every idle tool model except ``keep``; returns the tool ids freed."""
    freed = []
    for tool_id, slot in list(_REGISTRY.items()):
        if tool_id == keep:
            continue
        try:
            if slot.unload():
                freed.append(tool_id)
        except Exception:  # noqa: BLE001 — a stuck unload must not block a render
            logger.warning("unloading %s failed", tool_id, exc_info=True)
    return freed
