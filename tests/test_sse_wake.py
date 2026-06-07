"""Tests for the SSE doorbell (`SseWake`) and frame parser (M8 §3.2.1, decision B).

The doorbell *logic* is what matters — wake on a relevant event, fall back to the
floor on idle, reconnect on drop, give up to the poll if it can't reconnect. All
of it is exercised deterministically with a fake line source + a fake clock; the
real httpx streaming-timeout path is validated live (see `sse_line_source`).
"""

from __future__ import annotations

import itertools

from auspexai_tenant.wake import (
    IDLE,
    Backoff,
    SseWake,
    WakeSource,
    _SseFrameParser,
)

# ---- frame parser ----------------------------------------------------------


def _feed(parser: _SseFrameParser, *lines: str):
    out = []
    for line in lines:
        ev = parser.feed_line(line)
        if ev is not None:
            out.append(ev)
    return out


def test_parser_emits_event_on_blank_line() -> None:
    events = _feed(_SseFrameParser(), "id: 7", "event: unit.progress", "data: {}", "")
    assert len(events) == 1
    assert events[0].type == "unit.progress"
    assert events[0].id == "7"
    assert events[0].data == "{}"


def test_parser_ignores_comments() -> None:
    assert _feed(_SseFrameParser(), ": connected", "", ": ping", "") == []


def test_parser_multiline_data() -> None:
    (event,) = _feed(_SseFrameParser(), "event: x", "data: a", "data: b", "")
    assert event.data == "a\nb"


def test_parser_strips_one_leading_space() -> None:
    (event,) = _feed(_SseFrameParser(), "event:unit.progress", "data:{}", "")
    assert event.type == "unit.progress"  # no space after colon also works


# ---- fake time + sources ---------------------------------------------------


class FakeTime:
    """A clock that advances `step` per call, plus a sleep that advances by the
    slept duration — so `SseWake.wait` deadlines are reached deterministically."""

    def __init__(self, step: float = 1.0) -> None:
        self.now = 0.0
        self.step = step

    def clock(self) -> float:
        self.now += self.step
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _const_source(items):
    """open_stream that yields `items` then IDLE forever (one persistent stream)."""

    def factory():
        def gen():
            yield from items
            yield from itertools.repeat(IDLE)

        return gen()

    return factory


_FRAME = ["id: 1", "event: unit.progress", "data: {}", ""]


def _wake(open_stream, ft: FakeTime, *, floor=30.0, **kw) -> SseWake:
    return SseWake(
        open_stream,
        floor=Backoff(start=floor, max=floor),
        clock=ft.clock,
        sleep=ft.sleep,
        **kw,
    )


# ---- wake behavior ---------------------------------------------------------


def test_wakes_on_relevant_event() -> None:
    ft = FakeTime()
    _wake(_const_source(_FRAME), ft).wait()
    assert ft.now < 31  # returned well before the floor deadline


def test_ignores_irrelevant_event_then_times_out() -> None:
    ft = FakeTime()
    frame = ["event: some.other", "data: {}", ""]
    _wake(_const_source(frame), ft).wait()
    assert ft.now >= 31  # not woken; ran out the floor


def test_idle_only_times_out_at_floor() -> None:
    ft = FakeTime()
    _wake(_const_source([]), ft).wait()  # IDLE forever
    assert ft.now >= 31


def test_reconnects_after_drop_then_wakes() -> None:
    ft = FakeTime()
    scripts = iter(
        [
            [ConnectionError("dropped")],  # first stream: errors immediately
            _FRAME,  # second stream: delivers the event
        ]
    )

    def open_stream():
        script = next(scripts)

        def gen():
            for item in script:
                if isinstance(item, BaseException):
                    raise item
                yield item
            yield from itertools.repeat(IDLE)

        return gen()

    _wake(open_stream, ft, reconnect_backoff=Backoff(start=1.0, max=1.0)).wait()
    assert ft.now < 31  # reconnected and woke before the floor


def test_gives_up_to_poll_when_cannot_reconnect() -> None:
    ft = FakeTime()

    def open_stream():
        def gen():
            raise ConnectionError("down")
            yield  # unreachable; makes this a generator

        return gen()

    # Should return (not hang) once the floor elapses despite never connecting.
    _wake(open_stream, ft, floor=5.0, reconnect_backoff=Backoff(start=1.0, max=1.0)).wait()
    assert ft.now >= 5


def test_progress_resets_floor() -> None:
    ft = FakeTime()
    w = _wake(_const_source([]), ft, floor=4.0)
    w.wait()  # times out at ~4
    first = ft.now
    w.progress()  # reset floor
    w.wait()  # another ~4
    assert ft.now - first >= 4


def test_close_invokes_stream_close() -> None:
    closed = {"n": 0}

    class ClosableSource:
        def __iter__(self):
            return self

        def __next__(self):
            return IDLE

        def close(self):
            closed["n"] += 1

    ft = FakeTime()
    w = _wake(lambda: ClosableSource(), ft, floor=3.0)
    w.wait()  # opens the stream
    w.close()
    assert closed["n"] == 1


def test_sse_wake_satisfies_protocol() -> None:
    assert isinstance(_wake(_const_source([]), FakeTime()), WakeSource)
