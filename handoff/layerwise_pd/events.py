"""CPU event implementations for the slot-fence contract."""

from __future__ import annotations

import threading


class ThreadingEvent:
    def __init__(self) -> None:
        self._event = threading.Event()

    def record(self) -> None:
        self._event.set()

    def wait(self, timeout_s: float | None = None) -> bool:
        return self._event.wait(timeout_s)

    def reset(self) -> None:
        self._event.clear()


class NoopEvent:
    def record(self) -> None:
        return None

    def wait(self, timeout_s: float | None = None) -> bool:
        del timeout_s
        return True
