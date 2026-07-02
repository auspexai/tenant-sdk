"""D18 — transient-error retry for coordinator HTTP (`retry.call_with_retry`)."""

from __future__ import annotations

import httpx
import pytest

from auspexai_tenant.retry import call_with_retry


def _recorder():
    slept: list[float] = []
    return slept, (lambda d: slept.append(d))


def test_absorbs_transient_transport_error_then_succeeds():
    """The incident's failure: a truncated chunked read (RemoteProtocolError, a
    TransportError). Two transient failures then a 200 → returns the 200, having
    backed off twice. The driver loop never sees the blip."""
    calls = []

    def fn():
        calls.append(1)
        if len(calls) <= 2:
            raise httpx.RemoteProtocolError("peer closed connection (incomplete chunked read)")
        return httpx.Response(200)

    slept, sleep = _recorder()
    resp = call_with_retry(fn, sleep=sleep, rand=lambda: 0.5)
    assert resp.status_code == 200
    assert len(calls) == 3
    assert len(slept) == 2  # backed off before each retry, not after the success


def test_raises_last_transport_error_after_budget():
    """A sustained outage (every attempt fails at the transport layer) re-raises the
    last error after the budget — the CLI then surfaces it loudly (no silent orphan)."""

    def fn():
        raise httpx.ConnectError("connection refused")

    slept, sleep = _recorder()
    with pytest.raises(httpx.ConnectError):
        call_with_retry(fn, attempts=4, sleep=sleep, rand=lambda: 0.5)
    assert len(slept) == 3  # slept between the 4 attempts, not after the final raise


def test_retries_transient_5xx_then_succeeds():
    """A 502/503/504 edge page is transient → retried; a following 200 is returned."""
    codes = [503, 200]

    def fn():
        return httpx.Response(codes.pop(0))

    slept, sleep = _recorder()
    resp = call_with_retry(fn, sleep=sleep, rand=lambda: 0.5)
    assert resp.status_code == 200
    assert len(slept) == 1


def test_passes_4xx_through_immediately():
    """A 4xx (e.g. a semantic 409) is NOT transient — returned at once, no retry, so
    the caller's normal status handling runs unchanged."""
    calls = []

    def fn():
        calls.append(1)
        return httpx.Response(409)

    slept, sleep = _recorder()
    resp = call_with_retry(fn, sleep=sleep)
    assert resp.status_code == 409
    assert len(calls) == 1 and slept == []


def test_returns_lingering_5xx_after_budget():
    """If a 5xx persists through every attempt, the final response is RETURNED (not
    raised) so the caller raises its own CoordinatorError with the real status."""

    def fn():
        return httpx.Response(503)

    slept, sleep = _recorder()
    resp = call_with_retry(fn, attempts=3, sleep=sleep, rand=lambda: 0.5)
    assert resp.status_code == 503
    assert len(slept) == 2


def test_backoff_is_bounded_and_jittered():
    """Exponential base*2^i, clamped to max_delay, times jitter in [0.5, 1.5)."""

    def fn():
        raise httpx.ReadTimeout("timeout")

    slept, sleep = _recorder()
    with pytest.raises(httpx.ReadTimeout):
        # rand=0.0 -> jitter factor 0.5 (the low end); base 1.0, max 4.0
        call_with_retry(
            fn, attempts=5, base_delay=1.0, max_delay=4.0, sleep=sleep, rand=lambda: 0.0
        )
    # delays before retries: 1,2,4,4 (clamped) -> x0.5 jitter -> 0.5,1,2,2
    assert slept == [0.5, 1.0, 2.0, 2.0]


def test_rejects_zero_attempts():
    with pytest.raises(ValueError):
        call_with_retry(lambda: httpx.Response(200), attempts=0)
