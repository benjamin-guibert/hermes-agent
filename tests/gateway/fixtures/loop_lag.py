"""Measure how late the running event loop serves a timer: a stalled loop shows as lag."""
import asyncio


class LoopLag:
    """The worst lateness of a 50 ms timer chain on the running loop (no sleeps in the test)."""

    def __init__(self, period: float = 0.05):
        self.period, self.worst, self.ticks = period, 0.0, 0
        self.loop = asyncio.get_running_loop()
        self._due = self.loop.time() + period
        self._handle = self.loop.call_at(self._due, self._tick)

    def _tick(self):
        now = self.loop.time()
        self.worst = max(self.worst, now - self._due)
        self.ticks += 1
        self._due = now + self.period
        self._handle = self.loop.call_at(self._due, self._tick)

    def stop(self) -> float:
        """The worst lateness, the pending tick's included (a stall just before stop counts)."""
        self._handle.cancel()
        self.worst = max(self.worst, self.loop.time() - self._due)
        return self.worst
