"""Live raw-content collector (D20) — the tenant-side durable sink for
`[capture] raw` runs.

When an experiment declares `[capture] raw`, the executor emits a `raw_response`
and the coordinator parks it in an in-memory, TTL'd transit buffer — raw NEVER
rests on coordinator infrastructure (the §7 containment guarantee) — and serves
it live via `GET /experiments/{id}/raw-content` (R3-gated, audited). Nothing
collects it unless the researcher's driver polls DURING the run. This collector is
that poller, run as a sidecar for the whole life of the driver process:

  - polls every `interval_s` (default 180s, well under the coordinator's 3600s
    TTL, so each buffered item gets many collection windows and at most ~one
    interval of raw is at risk if the coordinator restarts);
  - writes each NEW result to a durable local JSONL (append + fsync) so collected
    raw survives a driver/Mac crash — durability lives tenant-side precisely
    because the coordinator's buffer is deliberately ephemeral;
  - dedupes by `result_id` (reads are non-destructive and overlap across polls)
    and REBUILDS the seen-set from the JSONL on restart, so a resumed run neither
    re-writes nor loses anything;
  - treats every failure (coordinator down, tunnel 5xx, network partition) as
    transient — back off and resume, NEVER crash the run. Collection is auxiliary
    to the scientific result (the drift scalar does not depend on raw);
  - a 403 (research standing < R3) is permanent for this credential: log once and
    stop polling (the run itself is unaffected).

What it cannot recover: raw produced during a coordinator RESTART (the in-memory
buffer is wiped) or across a gap longer than the TTL. That is the honest
live-collection limit of a never-at-rest design; the mitigation is a short poll
interval, and unrecoverable gaps are the coordinator's to avoid.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from auspexai_tenant.client import CoordinatorError

if TYPE_CHECKING:
    from auspexai_tenant.experiment import Experiment

log = logging.getLogger("auspexai_tenant.raw_collector")

DEFAULT_INTERVAL_S = 180.0  # << the coordinator's 3600s RawTransitBuffer TTL
_COORDINATOR_TTL_S = 3600.0  # for the start/finish log lines only
_FAILURE_BACKOFF_S = 15.0  # first retry after a transient failure
_FAILURE_BACKOFF_MAX_S = 120.0  # recover within ~2 min of the coordinator returning


class RawContentCollector:
    """Durable sidecar poller for `GET /experiments/{id}/raw-content`. Construct
    with the run's `Experiment` and a sink path, `start()` it before the run loop,
    and `stop_and_drain()` it in a `finally` so the tail is captured on every exit
    path (converge / Ctrl-C / coordinator error). Poll/persist logic is in
    `poll_once`, which is pure enough to unit-test without a thread."""

    def __init__(
        self,
        experiment: Experiment,
        sink_path: str | os.PathLike[str],
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        final_polls: int = 1,
        log_: logging.Logger | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._exp = experiment
        self._sink = Path(sink_path)
        self._interval = max(1.0, float(interval_s))
        self._final_polls = max(0, int(final_polls))
        self._log = log_ or log
        # An injectable sleep for tests; production uses the stop event as an
        # interruptible timer so `stop_and_drain` wakes the poller immediately.
        self._sleep = sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._seen: set[str] = set()
        self._written = 0
        self._poll_cycles = 0
        self._transient_failures = 0
        self._disabled_reason: str | None = None
        self._preexisting = self._load_seen()

    # ---- public surface ---------------------------------------------------

    @property
    def written(self) -> int:
        return self._written

    @property
    def disabled_reason(self) -> str | None:
        return self._disabled_reason

    @property
    def sink_path(self) -> Path:
        return self._sink

    def start(self) -> None:
        """Spawn the background poller (a daemon thread — never blocks process exit)."""
        self._log.info(
            "raw-content: collecting to %s every %.0fs (coordinator TTL %.0fs); "
            "%d already on disk.",
            self._sink,
            self._interval,
            _COORDINATOR_TTL_S,
            self._preexisting,
        )
        self._thread = threading.Thread(
            target=self._run_loop, name="raw-content-collector", daemon=True
        )
        self._thread.start()

    def stop_and_drain(self) -> int:
        """Signal stop, join the poller, then do a bounded FINAL drain so the last
        rounds' raw (buffered since the thread's last poll) is captured before the
        TTL evicts it. Returns the total items written. NEVER raises — cleanup must
        not turn a finished run into a crash."""
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=self._interval + 10.0)
        # The final drain is the important poll: everything settled after the
        # thread's last cycle is still in the buffer (post-finalize, the coordinator
        # keeps serving until the TTL). Poll until a pass yields nothing new.
        for _ in range(self._final_polls + 1):
            try:
                if self.poll_once() == 0:
                    break
            except Exception:  # cleanup must never raise into the run
                break
        self._log.info(
            "raw-content: done — %d output(s) collected to %s (%d poll cycle(s))%s.",
            self._written,
            self._sink,
            self._poll_cycles,
            f"; DISABLED: {self._disabled_reason}" if self._disabled_reason else "",
        )
        return self._written

    def poll_once(self) -> int:
        """Collect + durably persist every not-yet-seen raw item; return the count
        newly written. NEVER raises: a transient coordinator/network failure is
        counted, logged, and returns 0 (the caller backs off); a 403 permanently
        disables collection (standing < R3)."""
        if self._disabled_reason is not None:
            return 0
        try:
            items = self._exp.collect_raw_content()
        except CoordinatorError as e:
            if e.status_code == 403:
                self._disabled_reason = "research standing < R3"
                self._log.warning(
                    "raw-content: collection DISABLED — %s (HTTP 403). The run continues; "
                    "raw output will NOT be collected.",
                    self._disabled_reason,
                )
                return 0
            self._transient_failures += 1
            self._log.warning(
                "raw-content: poll failed (HTTP %s) — will retry; %d consecutive failure(s).",
                e.status_code,
                self._transient_failures,
            )
            return 0
        except Exception as e:  # transport down, partition, torn JSON — all transient
            self._transient_failures += 1
            self._log.warning(
                "raw-content: poll failed (%s: %s) — will retry; %d consecutive failure(s).",
                type(e).__name__,
                e,
                self._transient_failures,
            )
            return 0

        now_wall = time.time()
        with self._lock:
            fresh: list[dict[str, Any]] = []
            for result_id, item in items.items():
                if result_id in self._seen:
                    continue
                text = item.get("raw") if isinstance(item, dict) else item
                if text is None:
                    continue
                sig = item if isinstance(item, dict) else {}
                fresh.append(
                    {
                        "result_id": result_id,
                        "experiment_id": self._exp.experiment_id,
                        "collected_at": now_wall,
                        "raw": text,
                        # Free integrity anchor: this must equal the response_sha256 in
                        # the already-signed feature payload (executor emits both from
                        # the same text) — lets a later step tie raw to the signed result.
                        "raw_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        # AUD-26 detached worker signature + pubkey, stored verbatim for
                        # independent authenticity verification.
                        "raw_signature": sig.get("raw_signature"),
                        "worker_pubkey": sig.get("worker_pubkey"),
                    }
                )
                self._seen.add(result_id)
            if not fresh:
                self._transient_failures = 0
                return 0
            try:
                self._append(fresh)
            except OSError as e:
                # Roll the seen-set back so the next poll retries the write; better a
                # duplicate-free retry than silently dropping collected content.
                for rec in fresh:
                    self._seen.discard(rec["result_id"])
                self._log.error(
                    "raw-content: could not write %d item(s) to %s: %s — will retry.",
                    len(fresh),
                    self._sink,
                    e,
                )
                return 0
            self._written += len(fresh)
        self._transient_failures = 0
        return len(fresh)

    # ---- internals --------------------------------------------------------

    def _run_loop(self) -> None:
        backoff = _FAILURE_BACKOFF_S
        while not self._stop.is_set():
            self.poll_once()
            self._poll_cycles += 1
            if self._disabled_reason is not None:
                return  # 403 — nothing more this credential can collect
            if self._transient_failures:
                delay = min(backoff, _FAILURE_BACKOFF_MAX_S)
                backoff = min(backoff * 2, _FAILURE_BACKOFF_MAX_S)
            else:
                delay = self._interval
                backoff = _FAILURE_BACKOFF_S
            self._wait(delay)

    def _wait(self, seconds: float) -> None:
        if self._sleep is not None:
            self._sleep(seconds)  # test hook
        else:
            self._stop.wait(seconds)  # interruptible: stop_and_drain wakes it at once

    def _load_seen(self) -> int:
        """Restart recovery: rebuild the seen-set from an existing sink so a resumed
        run neither re-writes nor loses already-collected raw. Tolerates a truncated
        final line (a crash mid-append)."""
        if not self._sink.exists():
            return 0
        n = 0
        try:
            with self._sink.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # torn last line from a crash — skip
                    result_id = rec.get("result_id")
                    if result_id:
                        self._seen.add(result_id)
                        n += 1
        except OSError as e:
            self._log.warning("raw-content: could not read existing sink %s: %s", self._sink, e)
        return n

    def _append(self, records: list[dict[str, Any]]) -> None:
        """Append records durably (flush + fsync) so a crash cannot lose collected raw."""
        self._sink.parent.mkdir(parents=True, exist_ok=True)
        # If a prior crash left the file ending mid-line (no trailing newline),
        # start on a fresh line so the torn fragment stays its own (skippable) line
        # instead of being concatenated with — and corrupting — the next record.
        need_newline = False
        if self._sink.exists() and self._sink.stat().st_size > 0:
            with self._sink.open("rb") as fh:
                fh.seek(-1, os.SEEK_END)
                need_newline = fh.read(1) != b"\n"
        with self._sink.open("a", encoding="utf-8") as fh:
            if need_newline:
                fh.write("\n")
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
