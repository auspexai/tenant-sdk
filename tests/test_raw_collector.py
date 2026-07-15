"""Tests for the D20 live raw-content collector (`RawContentCollector`).

The collector is a durable, crash-tolerant sidecar over `Experiment.collect_raw_content`.
These drive `poll_once` (pure enough to test without a thread) for the collection,
dedup, persistence, restart-recovery, 403-disable, transient-retry, and
write-failure-rollback behaviors, plus one start/stop_and_drain lifecycle test that
exercises the background thread deterministically via an injected sleep.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from auspexai_tenant.client import CoordinatorError
from auspexai_tenant.raw_collector import RawContentCollector


class FakeExperiment:
    """Stands in for `Experiment`: `collect_raw_content` returns queued responses.
    A response is either a `{result_id: {...}}` dict (returned) or an Exception
    (raised). The last response repeats once the queue is exhausted."""

    def __init__(self, responses: list, experiment_id: str = "exp-fake") -> None:
        self.experiment_id = experiment_id
        self._responses = list(responses)
        self.calls = 0

    def collect_raw_content(self) -> dict:
        self.calls += 1
        resp = self._responses[0] if len(self._responses) == 1 else self._responses.pop(0)
        if isinstance(resp, Exception):
            raise resp
        return resp


def _item(text: str, sig: str | None = None, pk: str | None = None) -> dict:
    return {"raw": text, "raw_signature": sig, "worker_pubkey": pk}


def _read(sink: Path) -> list[dict]:
    """Parseable records only — mirrors the collector, which skips a torn line
    rather than rewriting history."""
    out: list[dict] = []
    for line in sink.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def test_collects_persists_and_dedups(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    exp = FakeExperiment(
        [
            {"r1": _item("alpha", "sigA", "pkA"), "r2": _item("beta")},
            # second poll: r2 repeats (non-destructive reads overlap) + r3 is new.
            {"r2": _item("beta"), "r3": _item("gamma")},
        ]
    )
    c = RawContentCollector(exp, sink)

    assert c.poll_once() == 2  # r1, r2
    assert c.poll_once() == 1  # only r3 is fresh
    assert c.written == 3

    recs = _read(sink)
    assert [r["result_id"] for r in recs] == ["r1", "r2", "r3"]
    r1 = recs[0]
    assert r1["raw"] == "alpha"
    assert r1["raw_signature"] == "sigA" and r1["worker_pubkey"] == "pkA"
    # The free integrity anchor: raw_sha256 == sha256(raw).
    assert r1["raw_sha256"] == hashlib.sha256(b"alpha").hexdigest()
    assert r1["experiment_id"] == "exp-fake"


def test_restart_recovery_rebuilds_seen_set(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    RawContentCollector(FakeExperiment([{"r1": _item("a"), "r2": _item("b")}]), sink).poll_once()
    assert len(_read(sink)) == 2

    # A NEW collector (a resumed run) must not re-write already-collected ids.
    exp2 = FakeExperiment([{"r1": _item("a"), "r2": _item("b"), "r3": _item("c")}])
    c2 = RawContentCollector(exp2, sink)
    assert c2.poll_once() == 1  # only r3
    assert [r["result_id"] for r in _read(sink)] == ["r1", "r2", "r3"]


def test_restart_recovery_tolerates_torn_final_line(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    sink.write_text(
        json.dumps({"result_id": "r1", "raw": "a"}) + "\n" + '{"result_id": "r2", "raw": "b'
    )  # second line truncated mid-write (a crash)
    c = RawContentCollector(FakeExperiment([{"r2": _item("b")}]), sink)
    # r1 is seen (parsed); r2's torn line did NOT register → r2 is (re)collected.
    assert c.poll_once() == 1
    assert [r["result_id"] for r in _read(sink)][-1] == "r2"


def test_403_disables_collection_and_run_continues(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    exp = FakeExperiment([CoordinatorError(403, "needs R3")])
    c = RawContentCollector(exp, sink)
    assert c.poll_once() == 0
    assert c.disabled_reason is not None
    # Once disabled, further polls are no-ops (no repeated 403 spam, no calls).
    calls_after = exp.calls
    assert c.poll_once() == 0
    assert exp.calls == calls_after
    assert not sink.exists()


def test_transient_failure_is_retried_not_fatal(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    exp = FakeExperiment(
        [
            CoordinatorError(502, "bad gateway"),  # tunnel wobble
            RuntimeError("connection reset"),  # transport error
            {"r1": _item("recovered")},  # coordinator back
        ]
    )
    c = RawContentCollector(exp, sink)
    assert c.poll_once() == 0  # 502
    assert c.disabled_reason is None
    assert c.poll_once() == 0  # transport error
    assert c.disabled_reason is None
    assert c.poll_once() == 1  # recovered
    assert _read(sink)[0]["raw"] == "recovered"


def test_write_failure_rolls_back_seen_so_retry_succeeds(tmp_path: Path, monkeypatch) -> None:
    sink = tmp_path / "raw_content.jsonl"
    exp = FakeExperiment([{"r1": _item("a")}])  # same content re-served on retry
    c = RawContentCollector(exp, sink)

    calls = {"n": 0}
    real_append = c._append

    def flaky_append(records):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("disk full")
        return real_append(records)

    monkeypatch.setattr(c, "_append", flaky_append)
    assert c.poll_once() == 0  # write failed → nothing recorded, seen rolled back
    assert c.written == 0
    assert not sink.exists()
    # The same result_id must not be considered already-collected → retry writes it.
    assert c.poll_once() == 1
    assert [r["result_id"] for r in _read(sink)] == ["r1"]


def test_start_and_drain_lifecycle_captures_tail(tmp_path: Path) -> None:
    sink = tmp_path / "raw_content.jsonl"
    # The thread's first poll returns r1; the FINAL drain (after stop) returns r2 —
    # proving the tail is captured after the loop ends.
    exp = FakeExperiment([{"r1": _item("first")}, {"r1": _item("first"), "r2": _item("tail")}])
    c = RawContentCollector(exp, sink, interval_s=1.0)

    # Deterministic loop: the injected sleep stops the collector after the 1st poll,
    # so the thread runs exactly one cycle then exits; stop_and_drain does the rest.
    c._sleep = lambda _s: c._stop.set()
    c.start()
    total = c.stop_and_drain()

    assert total == 2
    assert [r["result_id"] for r in _read(sink)] == ["r1", "r2"]
