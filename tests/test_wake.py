"""Tests for the wake sources (M8 §3.2.1) — Backoff + TimerWake."""

from __future__ import annotations

import pytest

from auspexai_tenant.wake import Backoff, TimerWake, WakeSource


def test_backoff_grows_and_clamps() -> None:
    b = Backoff(start=1.0, max=8.0, factor=2.0)
    assert [b.next() for _ in range(5)] == [1.0, 2.0, 4.0, 8.0, 8.0]  # clamped at max


def test_backoff_reset() -> None:
    b = Backoff(start=1.0, max=8.0, factor=2.0)
    b.next()
    b.next()
    b.reset()
    assert b.next() == 1.0


def test_backoff_validates() -> None:
    with pytest.raises(ValueError):
        Backoff(start=0)
    with pytest.raises(ValueError):
        Backoff(start=10, max=5)
    with pytest.raises(ValueError):
        Backoff(factor=0.5)


def test_timerwake_sleeps_backoff_intervals() -> None:
    slept: list[float] = []
    w = TimerWake(Backoff(start=1.0, max=4.0, factor=2.0), sleep=slept.append)
    w.wait()
    w.wait()
    w.wait()
    assert slept == [1.0, 2.0, 4.0]


def test_timerwake_progress_resets() -> None:
    slept: list[float] = []
    w = TimerWake(Backoff(start=1.0, max=4.0, factor=2.0), sleep=slept.append)
    w.wait()  # 1.0
    w.wait()  # 2.0
    w.progress()  # productive poll → reset
    w.wait()  # back to 1.0
    assert slept == [1.0, 2.0, 1.0]


def test_timerwake_satisfies_protocol() -> None:
    assert isinstance(TimerWake(), WakeSource)
