"""Wake sources — when the autonomic driver polls next (M8 §3.2.1).

The driver's correctness rides on the cursor-poll (the poll is the source of
truth, design note §2); a *wake source* only decides *when* the next poll fires.
This module ships the always-present floor (`TimerWake`, a backoff timer); the
SSE doorbell (`SseWake` — wake the instant a relevant event arrives, poll
immediately) plugs into the same `WakeSource` seam and is added alongside the M6
streaming surface.

`Backoff` lets an active experiment poll fast and an idle one back off: the driver
calls `progress()` after a productive poll to reset to the floor, and `wait()`
grows the interval when nothing new arrives.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import httpx

from auspexai_tenant.http_signing import Rfc9421Auth
from auspexai_tenant.signing import TenantKey


class Backoff:
    """Exponential backoff clamped to ``[start, max]``. `next()` returns the
    current delay then grows it; `reset()` returns to `start`."""

    def __init__(self, start: float = 30.0, max: float = 900.0, factor: float = 2.0) -> None:
        if start <= 0 or max < start or factor < 1:
            raise ValueError("require 0 < start <= max and factor >= 1")
        self.start = start
        self.max = max
        self.factor = factor
        self._current = start

    def reset(self) -> None:
        self._current = self.start

    def next(self) -> float:
        delay = self._current
        self._current = min(self.max, self._current * self.factor)
        return delay


@runtime_checkable
class WakeSource(Protocol):
    """Decides when the driver polls next. `wait()` blocks until the next poll is
    due (a timer floor elapsed — or, for `SseWake`, a relevant event arrived,
    whichever first). `progress()` is called after a productive poll (reset any
    backoff); `close()` releases resources (an SSE connection)."""

    def wait(self) -> None: ...

    def progress(self) -> None: ...

    def close(self) -> None: ...


class TimerWake:
    """Backoff-timer wake — the correct, always-present floor. `sleep` is injected
    so tests run without real time."""

    def __init__(
        self, backoff: Backoff | None = None, *, sleep: Callable[[float], None] = time.sleep
    ) -> None:
        self._backoff = backoff or Backoff()
        self._sleep = sleep

    def wait(self) -> None:
        self._sleep(self._backoff.next())

    def progress(self) -> None:
        self._backoff.reset()

    def close(self) -> None:  # nothing to release
        pass


# ---- SSE doorbell (the M8 decision-B wake source) --------------------------

# Default coordinator event types that should wake the driver to poll now.
DOORBELL_EVENTS = frozenset({"unit.progress", "receipt.issued", "experiment.status"})

# Yielded by a line source when there's no data right now (a read-timeout): lets
# `SseWake.wait` check its floor deadline instead of blocking indefinitely.
IDLE: Any = object()

# A line source is an iterator of SSE text lines, or `IDLE` when momentarily idle.
LineSource = Iterator[Any]


@dataclass
class SseEvent:
    """A parsed SSE frame (the `data` is the raw JSON text; the driver only needs
    the `type` to decide whether to wake, so we don't parse the body)."""

    type: str
    data: str
    id: str | None = None


class _SseFrameParser:
    """Accumulates SSE field lines into events. Blank line ends a frame; `:`
    comments (the coordinator's `: connected` / `: ping` keepalives) are ignored."""

    def __init__(self) -> None:
        self._type: str | None = None
        self._data: list[str] = []
        self._id: str | None = None

    def feed_line(self, line: str) -> SseEvent | None:
        if line == "":
            if self._type is None and not self._data:
                return None  # blank between keepalives — nothing to emit
            event = SseEvent(self._type or "message", "\n".join(self._data), self._id)
            self._type, self._data, self._id = None, [], None
            return event
        if line.startswith(":"):
            return None  # comment / keepalive
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            self._type = value
        elif field == "data":
            self._data.append(value)
        elif field == "id":
            self._id = value
        return None


class SseWake:
    """The SSE doorbell: subscribe to the experiment's authenticated event stream
    and wake the driver the instant a relevant event arrives — poll immediately
    instead of waiting out the timer floor (design note §3.2.1, decision B).

    **The doorbell is a hint, never the truth.** A dropped/disconnected stream is
    harmless: `wait` returns when the **floor** elapses (so the driver polls
    anyway), reconnecting underneath; the cursor-poll remains the source of truth.

    `open_stream()` returns a fresh `LineSource` (an iterator of SSE lines, or
    `IDLE` when momentarily idle, raising / ending on disconnect). It's injected so
    the wake *logic* is unit-tested deterministically; `sse_line_source` is the
    httpx-backed production adapter."""

    def __init__(
        self,
        open_stream: Callable[[], LineSource],
        *,
        floor: Backoff | None = None,
        event_types: frozenset[str] = DOORBELL_EVENTS,
        reconnect_backoff: Backoff | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._open_stream = open_stream
        self._floor = floor or Backoff()
        self._event_types = frozenset(event_types)
        self._reconnect = reconnect_backoff or Backoff(start=1.0, max=30.0)
        self._clock = clock
        self._sleep = sleep
        self._stream: LineSource | None = None
        self._parser = _SseFrameParser()

    def wait(self) -> None:
        """Block until a relevant event arrives or the floor elapses, whichever
        first. Reconnects (within the floor) if the stream drops."""
        deadline = self._clock() + self._floor.next()
        while self._clock() < deadline:
            if self._stream is None and not self._connect(deadline):
                return  # couldn't (re)connect before the floor — let the driver poll
            try:
                item = next(self._stream)  # type: ignore[arg-type]
            except StopIteration:
                self._stream = None
                if not self._reconnect_pause(deadline):
                    return
                continue
            except Exception:  # any stream error → reconnect within the floor
                self._close_stream()
                if not self._reconnect_pause(deadline):
                    return
                continue
            if item is IDLE:
                continue  # no data; loop re-checks the deadline
            event = self._parser.feed_line(item)
            if event is not None and event.type in self._event_types:
                return  # woke on a relevant event → poll now

    def progress(self) -> None:
        self._floor.reset()

    def close(self) -> None:
        self._close_stream()

    # internals ----------------------------------------------------------------

    def _connect(self, deadline: float) -> bool:
        try:
            self._stream = iter(self._open_stream())
            self._parser = _SseFrameParser()
            self._reconnect.reset()
            return True
        except Exception:  # connect failure → retry within the floor
            self._stream = None
            return self._reconnect_pause(deadline)

    def _reconnect_pause(self, deadline: float) -> bool:
        """Sleep a reconnect backoff, clamped to the floor; False if the floor
        elapsed (give up reconnecting this wait — the driver polls anyway)."""
        remaining = deadline - self._clock()
        if remaining <= 0:
            return False
        self._sleep(min(self._reconnect.next(), remaining))
        return self._clock() < deadline

    def _close_stream(self) -> None:
        stream, self._stream = self._stream, None
        closer = getattr(stream, "close", None)
        if callable(closer):
            closer()


def sse_line_source(
    coordinator_url: str,
    key: TenantKey,
    experiment_id: str,
    *,
    client: httpx.Client | None = None,
    read_timeout: float = 20.0,
) -> Callable[[], LineSource]:
    """Production `open_stream` for `SseWake`: an RFC 9421-signed GET to the
    experiment's `text/event-stream`, yielding SSE lines (and `IDLE` on a read
    timeout so `wait` can honor its floor). The driver holds the tenant key, so it
    streams the authenticated endpoint directly (no proxy — that's only for
    browsers, §5). The streaming-timeout path is validated live, not over
    MockTransport (which can't simulate it — same constraint as the coordinator's
    SSE tests)."""
    auth = Rfc9421Auth(key)
    url = f"{coordinator_url.rstrip('/')}/api/v0/experiments/{experiment_id}/events"

    def _open() -> LineSource:
        timeout = httpx.Timeout(None, read=read_timeout)
        c = client or httpx.Client(timeout=timeout)
        with c.stream("GET", url, auth=auth, headers={"Accept": "text/event-stream"}) as resp:
            resp.raise_for_status()
            lines = resp.iter_lines()
            while True:
                try:
                    yield next(lines)
                except httpx.ReadTimeout:
                    yield IDLE
                except StopIteration:
                    return

    return _open
