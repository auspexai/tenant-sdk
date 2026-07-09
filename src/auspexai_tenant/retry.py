"""D18 — transient-error retry for coordinator HTTP.

The long-running driver loop (`driver.run_until`) polls the coordinator over a
Cloudflare tunnel where transient failures are NORMAL, not exceptional: 502/503/504
edge pages, the Cloudflare origin-down 520-524/530 pages (the tunnel/coordinator
momentarily unreachable — a reconnect or a local WAN blip), read timeouts, and
truncated chunked reads (`httpx.RemoteProtocolError`, a `TransportError`). Before
D18 a single blip raised straight out of the poll and killed the driver — it exited
without aborting or a resume hint, orphaning an `approved` run that the dashboard
still showed as healthy (the C16 incident's third finding; and again 2026-07-09
when a 502 outlasted the then-too-short budget).

`call_with_retry` wraps ONE HTTP call: it retries the TRANSIENT class with bounded,
jittered exponential backoff and re-raises only after the budget is spent — at which
point the CLI surfaces it loudly with a resume/abort choice (never a silent orphan).
Non-transient responses (2xx, 4xx, auth failures, the semantic 409s) pass straight
through, so the caller's normal status handling is unchanged. Retry is safe: reads
are idempotent, and the coordinator's submit idempotency (a re-sent submit that
already landed returns 409 `UnitsAlreadySubmitted`, treated as success) makes the
POST path safe too.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import httpx

# Cloudflare / edge transients a retry usually rides out — a 4xx is a real client
# error the caller must see immediately, but these are all "the tunnel/origin is
# momentarily unreachable, try again":
#   502/503/504 — standard bad-gateway / unavailable / gateway-timeout.
#   520-524, 530 — Cloudflare ORIGIN-DOWN codes: the edge is up but cloudflared /
#     the coordinator behind it is unreachable (a tunnel reconnect, a local WAN
#     blip). 530 is the one paired with the "Error 1033" tunnel page — the exact
#     failure that killed a real overnight driver (2026-07-09, exp-omJ9jjXw) when
#     it was NOT retried. These MUST be ridden out, not treated as fatal.
_TRANSIENT_STATUS = frozenset({502, 503, 504, 520, 521, 522, 523, 524, 530})
# Budget sized to ride out a typical tunnel reconnect / short WAN blip (seconds to
# a couple minutes) on an UNATTENDED overnight driver, while still giving up
# (resumably, from the journal) on a genuine multi-hour outage. Backoff sum with
# the defaults below ≈ 1-2.7 min (was ~5-11 s at 5 attempts / 8 s cap — far too
# short; a driver died on a 502 that outlasted it).
_DEFAULT_ATTEMPTS = 9


def call_with_retry(
    fn: Callable[[], httpx.Response],
    *,
    attempts: int = _DEFAULT_ATTEMPTS,
    base_delay: float = 0.5,
    max_delay: float = 45.0,
    sleep: Callable[[float], None] = time.sleep,
    rand: Callable[[], float] = random.random,
) -> httpx.Response:
    """Call `fn` (which performs one HTTP request), retrying TRANSIENT failures —
    `httpx.TransportError` (connect/read timeout, connection reset, truncated
    chunked read) and a transient status (`_TRANSIENT_STATUS`: 502/503/504 plus
    the Cloudflare origin-down 520-524/530) — with bounded, jittered exponential
    backoff.

    Returns the response for a success, a non-transient status, or the final
    attempt (a lingering transient 5xx is returned so the caller raises its normal
    `CoordinatorError`). Raises the last `httpx.TransportError` if every attempt
    failed at the transport layer. `sleep`/`rand` are injectable for tests.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    last_exc: httpx.TransportError | None = None
    for i in range(attempts):
        is_last = i == attempts - 1
        try:
            resp = fn()
        except httpx.TransportError as exc:
            last_exc = exc
            if is_last:
                raise
        else:
            if is_last or resp.status_code not in _TRANSIENT_STATUS:
                return resp
        # backoff before the next attempt: base*2^i clamped, times jitter in [0.5, 1.5)
        delay = min(max_delay, base_delay * (2**i))
        sleep(delay * (0.5 + rand()))
    # Unreachable: the final iteration always returns or raises. Present for the
    # type-checker and as a defensive backstop.
    raise last_exc if last_exc is not None else RuntimeError("call_with_retry exhausted")
