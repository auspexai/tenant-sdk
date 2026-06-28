"""Tests for the autonomic control loop, `run_until` (M8 §3.2).

Driven by a controllable `FakeExperiment` (a duck-typed Experiment) so the
round/result flow is deterministic: convergence, max_rounds, exhaustion, the
stall policies, the idempotency contract, dedup, and — the load-bearing one —
crash-resume with no double-counting.
"""

from __future__ import annotations

import pytest

from auspexai_tenant.aggregate import Counter
from auspexai_tenant.driver import (
    AbortAfter,
    FinalizeOnPartial,
    StallContext,
    WaitForever,
    run_until,
)
from auspexai_tenant.experiment import ResultPage, SubmitResult, Unit, UnitsAlreadySubmittedError
from auspexai_tenant.journal import RunJournal
from auspexai_tenant.wake import TimerWake

# ---- test doubles ----------------------------------------------------------


class FakeExperiment:
    """A duck-typed Experiment: accepts submits, makes consensus results available
    (immediately, unless the unit_id matches a blocked prefix), serves them
    paginated, and records finalize/abort. `unblock` releases withheld units."""

    def __init__(self, *, page_size: int = 1000, blocked_prefixes: tuple[str, ...] = ()) -> None:
        self.experiment_id = "exp-fake"
        self.page_size = page_size
        self.submitted_rounds: list[tuple[int | None, list[str]]] = []
        self._results: list[dict] = []
        self._seen: set[str] = set()
        self._blocked = list(blocked_prefixes)
        self._withheld: list[Unit] = []
        self.finalized = False
        self.aborted = False

    def _blocked_id(self, unit_id: str) -> bool:
        return any(unit_id.startswith(p) for p in self._blocked)

    def _emit(self, unit: Unit) -> None:
        i = len(self._results)
        self._results.append(
            {
                "unit_id": unit.unit_id,
                "result_id": f"res-{unit.unit_id}",
                "completed_at": f"2026-01-01T00:00:{i:02d}+00:00",
                "payload": dict(unit.payload),
            }
        )

    def submit_units(self, units, *, round=None) -> SubmitResult:
        ids = [u.unit_id for u in units]
        new = [u for u in units if u.unit_id not in self._seen]
        if not new:  # everything was already submitted (crash-resume re-submit)
            raise UnitsAlreadySubmittedError(409, "already submitted")
        self.submitted_rounds.append((round, ids))
        for u in new:
            self._seen.add(u.unit_id)
            (self._withheld.append(u) if self._blocked_id(u.unit_id) else self._emit(u))
        return SubmitResult(ids, len(new))

    def unblock(self, prefix: str) -> None:
        self._blocked = [p for p in self._blocked if p != prefix]
        keep: list[Unit] = []
        for u in self._withheld:
            keep.append(u) if self._blocked_id(u.unit_id) else self._emit(u)
        self._withheld = keep

    def results_page(self, *, cursor=None, include="consensus") -> ResultPage:
        start = int(cursor) if cursor else 0
        chunk = self._results[start : start + self.page_size]
        end = start + len(chunk)
        full = len(chunk) == self.page_size
        next_cursor = str(end) if (full and end < len(self._results)) else None
        return ResultPage(results=list(chunk), next_cursor=next_cursor)

    def progress(self):
        return None

    def finalize(self) -> dict:
        self.finalized = True
        return {}

    def abort(self) -> dict:
        self.aborted = True
        return {}

    def attestation(self, *, checkpoint=False):
        return None


class _CrashError(Exception):
    pass


class CrashWake:
    """A wake source that raises after `crash_after` waits — simulates the driver
    process dying mid-run."""

    def __init__(self, crash_after: int = 1) -> None:
        self.calls = 0
        self.crash_after = crash_after

    def wait(self) -> None:
        self.calls += 1
        if self.calls >= self.crash_after:
            raise _CrashError()

    def progress(self) -> None:
        pass

    def close(self) -> None:
        pass


def _wake() -> TimerWake:
    return TimerWake(sleep=lambda _d: None)  # no real sleeping in tests


def _batch_of(n: int):
    def next_batch(_agg, rnd: int):
        return [Unit(f"r{rnd}u{i}", {"value": 1}) for i in range(n)]

    return next_batch


# ---- happy paths -----------------------------------------------------------


def test_converges(tmp_path) -> None:
    agg = Counter()
    result = run_until(
        FakeExperiment(),
        condition=lambda a: a.total >= 5,
        next_batch=_batch_of(2),
        reduce=agg,
        journal=tmp_path / "j",
        wake=_wake(),
    )
    assert result.outcome == "converged"
    assert result.aggregate["total"] == 6  # rounds 0,1,2 → 2 each, condition at total 6
    assert result.rounds == 3


def test_finalize_called_on_convergence(tmp_path) -> None:
    fake = FakeExperiment()
    run_until(
        fake,
        condition=lambda a: a.total >= 2,
        next_batch=_batch_of(2),
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
    )
    assert fake.finalized is True


def test_max_rounds(tmp_path) -> None:
    result = run_until(
        FakeExperiment(),
        condition=lambda a: False,  # never converges
        next_batch=_batch_of(1),
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        max_rounds=3,
    )
    assert result.outcome == "max_rounds"
    assert result.rounds == 3
    assert result.aggregate["total"] == 3


def test_duration_cap_finalizes_like_max_rounds(tmp_path) -> None:
    # A run that never converges, with an injected clock that jumps past the
    # deadline after the first round → outcome "time_capped" AND the experiment
    # finalizes (same side-effect the max_rounds path asserts), not an abort.
    fake = FakeExperiment()
    # First reads (deadline compute + round-0 gate) are before the cap; the next
    # gate check (round-1 boundary) is past it. monotonic() is read a few times
    # per round (deadline calc, gate, stall bookkeeping), so feed a long ramp.
    ticks = iter([0.0, 0.0, 0.0, 0.0] + [100.0] * 50)

    result = run_until(
        fake,
        condition=lambda a: False,  # never converges
        next_batch=_batch_of(1),
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        max_rounds=999,  # would otherwise run far past the cap
        duration_cap_seconds=10.0,
        monotonic=lambda: next(ticks),
    )
    assert result.outcome == "time_capped"
    assert fake.finalized is True  # FINALIZED, like max_rounds
    assert fake.aborted is False  # NOT aborted
    assert result.rounds >= 1  # at least one round completed before the cap
    assert result.aggregate["total"] == result.rounds  # 1 unit/round folded


def test_duration_cap_none_runs_to_max_rounds(tmp_path) -> None:
    # With no cap the loop is unchanged: it runs to max_rounds.
    result = run_until(
        FakeExperiment(),
        condition=lambda a: False,
        next_batch=_batch_of(1),
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        max_rounds=3,
        duration_cap_seconds=None,
    )
    assert result.outcome == "max_rounds"
    assert result.rounds == 3


def test_exhausted_when_next_batch_returns_none(tmp_path) -> None:
    def next_batch(_agg, rnd):
        return [Unit(f"r{rnd}", {})] if rnd < 2 else None

    result = run_until(
        FakeExperiment(),
        condition=lambda a: False,
        next_batch=next_batch,
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
    )
    assert result.outcome == "exhausted"
    assert result.rounds == 2


# ---- stall policies --------------------------------------------------------


def test_finalize_on_partial(tmp_path) -> None:
    # 2 units/round, but the second never reaches consensus → stall at 50% agreed.
    fake = FakeExperiment(blocked_prefixes=("r0u1",))

    def next_batch(_agg, rnd):
        return [Unit("r0u0", {}), Unit("r0u1", {})] if rnd == 0 else None

    result = run_until(
        fake,
        condition=lambda a: False,
        next_batch=next_batch,
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        stall=FinalizeOnPartial(min_fraction=0.5, after_polls=2),
    )
    assert result.outcome == "finalized_partial"
    assert fake.finalized is True
    assert result.aggregate["total"] == 1  # the one agreed unit


def test_abort_after(tmp_path) -> None:
    fake = FakeExperiment(blocked_prefixes=("r0",))
    ticks = iter([5.0 * i for i in range(1, 50)])

    result = run_until(
        fake,
        condition=lambda a: False,
        next_batch=lambda _a, rnd: [Unit("r0u0", {})] if rnd == 0 else None,
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        stall=AbortAfter(seconds=10),
        monotonic=lambda: next(ticks),
    )
    assert result.outcome == "aborted"
    assert fake.aborted is True
    assert fake.finalized is False


def test_wait_forever_is_default_and_keeps_waiting(tmp_path) -> None:
    # A blocked unit under the default policy never terminates → the driver keeps
    # polling; CrashWake stands in for "still running" by cutting the loop off.
    fake = FakeExperiment(blocked_prefixes=("r0",))
    with pytest.raises(_CrashError):
        run_until(
            fake,
            condition=lambda a: False,
            next_batch=lambda _a, rnd: [Unit("r0u0", {})],
            reduce=Counter(),
            journal=tmp_path / "j",
            wake=CrashWake(crash_after=5),  # WaitForever would otherwise loop forever
        )
    assert fake.finalized is False  # never auto-terminated


# ---- idempotency + dedup ---------------------------------------------------


def test_submit_already_submitted_is_tolerated(tmp_path) -> None:
    fake = FakeExperiment()
    fake.submit_units([Unit("r0u0", {})])  # pre-seed so the driver's submit 409s
    result = run_until(
        fake,
        condition=lambda a: a.total >= 1,
        next_batch=lambda _a, rnd: [Unit("r0u0", {})],
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
    )
    assert result.outcome == "converged"
    assert result.aggregate["total"] == 1  # folded once despite the 409


def test_results_are_not_double_folded_on_repoll(tmp_path) -> None:
    # cursor stays None (single partial page) so each poll re-serves all results;
    # dedup-by-unit_id must fold each exactly once. A blocked 2nd unit forces re-polls.
    fake = FakeExperiment(blocked_prefixes=("keep-blocked",))
    agg = Counter()
    run_until(
        fake,
        condition=lambda a: a.total >= 1,
        next_batch=lambda _a, rnd: [Unit("done-0", {}), Unit("keep-blocked-0", {})],
        reduce=agg,
        journal=tmp_path / "j",
        wake=_wake(),
        max_rounds=1,
    )
    # max_rounds=1 with condition met at total>=1 after folding done-0 once.
    assert agg.total == 1


def test_pagination_drains_all_pages(tmp_path) -> None:
    fake = FakeExperiment(page_size=2)  # force multi-page drains
    result = run_until(
        fake,
        condition=lambda a: a.total >= 5,
        next_batch=_batch_of(5),  # 5 units in round 0 → 3 pages of size 2
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
    )
    assert result.outcome == "converged"
    assert result.aggregate["total"] == 5


# ---- crash-resume (the load-bearing one) -----------------------------------


def test_resume_after_crash_no_double_count(tmp_path) -> None:
    journal_path = tmp_path / "run.journal"
    # One persistent coordinator state across the crash; round 1 withheld at first.
    fake = FakeExperiment(blocked_prefixes=("r1",))

    def next_batch(_agg, rnd):
        return [Unit(f"r{rnd}u0", {}), Unit(f"r{rnd}u1", {})]

    # First run: round 0 completes; round 1 submitted but withheld → stall → crash.
    with pytest.raises(_CrashError):
        run_until(
            fake,
            condition=lambda a: a.total >= 4,
            next_batch=next_batch,
            reduce=Counter(),
            journal=journal_path,
            wake=CrashWake(crash_after=1),
        )
    kinds = [r["kind"] for r in RunJournal(journal_path).records()]
    assert kinds == ["submit", "round_done", "submit"]  # r0 done, r1 in-flight

    # Coordinator now produces round 1's results; resume with a FRESH aggregate.
    fake.unblock("r1")
    resumed = run_until(
        fake,
        condition=lambda a: a.total >= 4,
        next_batch=next_batch,
        reduce=Counter(),  # fresh — must be restored from the journal, not refolded
        journal=journal_path,
        wake=_wake(),
    )
    assert resumed.outcome == "converged"
    assert resumed.aggregate["total"] == 4  # NOT 6 — round 0 not double-counted
    assert fake.finalized is True


def test_resume_already_finalized_returns_restored_aggregate(tmp_path) -> None:
    journal_path = tmp_path / "run.journal"
    run_until(
        FakeExperiment(),
        condition=lambda a: a.total >= 4,
        next_batch=_batch_of(2),
        reduce=Counter(),
        journal=journal_path,
        wake=_wake(),
    )
    # Re-invoke against the finalized journal: no new work, aggregate restored.
    fake2 = FakeExperiment()
    again = run_until(
        fake2,
        condition=lambda a: a.total >= 4,
        next_batch=_batch_of(2),
        reduce=Counter(),
        journal=journal_path,
        wake=_wake(),
    )
    assert again.outcome == "already_finalized"
    assert again.aggregate["total"] == 4
    assert fake2.submitted_rounds == []  # no new submissions on a finalized run


# ---- stall policy units ----------------------------------------------------


def test_wait_forever_policy() -> None:
    ctx = StallContext(0, 2, 1, 1, 99, 9999.0, None)
    assert WaitForever()(ctx).value == "wait"


def test_finalize_on_partial_policy_thresholds() -> None:
    policy = FinalizeOnPartial(min_fraction=0.5, after_polls=2)
    assert policy(StallContext(0, 2, 1, 1, 1, 0.0, None)).value == "wait"  # too few polls
    assert policy(StallContext(0, 2, 1, 1, 2, 0.0, None)).value == "finalize_partial"
    assert policy(StallContext(0, 4, 1, 3, 5, 0.0, None)).value == "wait"  # only 25% agreed


def test_external_terminal_stops_the_driver(tmp_path) -> None:
    """D14 (server→client coupling): when the experiment is aborted/completed
    out-of-band, a stalled driver detects the terminal status and exits cleanly
    instead of polling a dead run forever (WaitForever)."""
    from auspexai_tenant.experiment import Progress

    class AbortedMidRun(FakeExperiment):
        def __init__(self) -> None:
            super().__init__(blocked_prefixes=("r0",))  # units never fold → the driver stalls

        def progress(self):
            # A UI/ops abort flipped the server-side status to terminal.
            return Progress(
                status="aborted",
                reduce_ready=False,
                submissions_finalized=False,
                total_work_units=2,
                work_unit_counts={},
                completions_total=0,
                replication_target_total=None,
                active_contributor_count=None,
                network_active_workers=None,
            )

    exp = AbortedMidRun()
    result = run_until(
        exp,
        condition=lambda a: a.total >= 100,  # never converges on its own
        next_batch=_batch_of(2),
        reduce=Counter(),
        journal=tmp_path / "j",
        wake=_wake(),
        stall=WaitForever(),  # without the fix this would loop forever
    )
    assert result.outcome == "externally_aborted"
    assert not exp.finalized  # never finalized an already-terminal experiment


def test_abort_after_policy_threshold() -> None:
    policy = AbortAfter(seconds=10)
    assert policy(StallContext(0, 1, 0, 1, 1, 9.0, None)).value == "wait"
    assert policy(StallContext(0, 1, 0, 1, 1, 10.0, None)).value == "abort"
