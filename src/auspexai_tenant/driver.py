"""The autonomic control loop — `run_until` (M8 §3.2).

Composes the pieces into the headless driver that *runs* an adaptive experiment:
submit a batch → poll agreed results (the **poll is the source of truth**, §2) →
fold them into a running aggregate → test a termination condition → generate the
next batch or finalize. Resumable across crashes via the run-journal (§3.4) and
idempotent submit; bounded-memory via the incremental aggregate (§3.3).

Loop shape is **round-synchronous**: each round submits a batch, waits for that
batch to reach consensus (folding as results arrive), then the tenant's
`condition`/`next_batch` decide convergence or the next batch — the idiom for
adaptive sampling / refine-until-tolerance. A stuck unit (never reaching quorum)
is handled by the tenant `StallPolicy` (§3.5 / §10.3), not silently.

Correctness notes:
  - **Fold idempotency by unit_id.** Each consensus unit has exactly one result;
    the driver dedupes folds against a `folded` set (bounded by the experiment's
    max_units, reconstructed from the journal on resume). This makes folding
    immune to the coordinator emitting `next_cursor` only on full pages — the
    cursor is a fetch optimization, the `folded` set is the correctness guarantee.
  - **Round-atomic checkpointing.** `{cursor, aggregate}` is journaled at round
    boundaries; resume re-runs the in-flight round from its boundary (§3.4).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from auspexai_tenant.aggregate import RunningAggregate
from auspexai_tenant.attestation import ResultSetAttestation
from auspexai_tenant.client import CoordinatorError
from auspexai_tenant.experiment import (
    Experiment,
    LifecycleConflictError,
    Unit,
    UnitsAlreadySubmittedError,
)
from auspexai_tenant.journal import RunJournal, resume_state
from auspexai_tenant.wake import TimerWake, WakeSource

# Tenant-supplied callables.
Condition = Callable[[RunningAggregate], bool]
NextBatch = Callable[[RunningAggregate, int], Sequence[Unit] | None]


class StallAction(Enum):
    """What the driver does when a round stops making progress (§3.5)."""

    WAIT = "wait"
    FINALIZE_PARTIAL = "finalize_partial"
    ABORT = "abort"


@dataclass(frozen=True)
class StallContext:
    """Passed to a `StallPolicy` when a poll yields no new agreed results."""

    round: int
    submitted: int  # units submitted this round
    agreed: int  # units of this round folded so far
    pending: int  # units of this round still awaiting consensus
    stalled_polls: int  # consecutive polls with no new agreed results
    stalled_seconds: float  # wall-clock since progress last seen
    progress: Any  # latest Experiment.progress() snapshot, or None


class StallPolicy(Protocol):
    """Decides, on a stall, whether to keep waiting, finalize over the partial
    set, or abort. Science-specific — see the named policies below (§10.3)."""

    def __call__(self, ctx: StallContext) -> StallAction: ...


class WaitForever:
    """Default: keep polling (capacity-collapse is recoverable; the dashboard is
    the human override). A run never silently changes its result or dies."""

    def __call__(self, ctx: StallContext) -> StallAction:
        return StallAction.WAIT


class FinalizeOnPartial:
    """Finalize over consensus-so-far once at least `min_fraction` of the round's
    units have agreed and the stall has persisted `after_polls` polls (anchored by
    the checkpoint attestation). For statistical aggregates that tolerate gaps."""

    def __init__(self, min_fraction: float = 0.9, *, after_polls: int = 3) -> None:
        self.min_fraction = min_fraction
        self.after_polls = after_polls

    def __call__(self, ctx: StallContext) -> StallAction:
        if ctx.stalled_polls < self.after_polls:
            return StallAction.WAIT
        fraction = ctx.agreed / ctx.submitted if ctx.submitted else 0.0
        return StallAction.FINALIZE_PARTIAL if fraction >= self.min_fraction else StallAction.WAIT


class AbortAfter:
    """Abort once the stall has lasted `seconds`. Conservative bound for runs that
    must not linger."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def __call__(self, ctx: StallContext) -> StallAction:
        return StallAction.ABORT if ctx.stalled_seconds >= self.seconds else StallAction.WAIT


@dataclass(frozen=True)
class RunResult:
    """Outcome of a `run_until` call.

    `outcome` ∈ converged | max_rounds | time_capped | exhausted | finalized_partial |
    aborted | already_finalized | already_aborted. `aggregate` is `reduce.finalize()`;
    `attestation` is the result-set attestation when it was retrievable (None if
    the experiment hadn't finished auto-completing — fetch later via
    `Experiment.attestation()`)."""

    outcome: str
    rounds: int  # completed rounds
    aggregate: Any
    attestation: ResultSetAttestation | None


@dataclass
class DriverSpec:
    """A tenant's loop configuration for `auspexai-tenant experiment run` — the
    factory named on the CLI (`module:attr`) returns one of these. Keeps the
    science (condition / adaptive generator / aggregator) in tenant code while the
    CLI wires the signed `Experiment`, journal, and wake source.

    The factory is called with the **resolved `ExperimentConfig`** —
    `def build(cfg): ...` — so the `[driver]` knobs and the active `--profile`
    arrive as an argument, resolved once by the CLI. A factory reads what it needs
    from `cfg.driver`; it must NOT re-load the config from disk."""

    condition: Condition
    next_batch: NextBatch
    reduce: RunningAggregate
    wake: WakeSource | None = None
    stall: StallPolicy | None = None
    max_rounds: int = 50
    # Wall-clock budget (seconds). When the loop next checks (after the condition
    # and max_rounds gates) the cap is reached, the run finalizes at its current
    # state — exactly like max_rounds, not an abort. The CLI `--duration-cap`
    # flag overrides this when set.
    duration_cap_seconds: float | None = None


def run_until(
    experiment: Experiment,
    *,
    condition: Condition,
    next_batch: NextBatch,
    reduce: RunningAggregate,
    journal: str | Path | RunJournal,
    wake: WakeSource | None = None,
    max_rounds: int = 50,
    stall: StallPolicy | None = None,
    include: str = "consensus",
    fetch_attestation: bool = True,
    duration_cap_seconds: float | None = None,
    monotonic: Callable[[], float] = time.monotonic,
) -> RunResult:
    """Drive `experiment` until `condition(reduce)` holds, `next_batch` is
    exhausted, `max_rounds` is hit, the `duration_cap_seconds` wall-clock budget
    is reached, or a `StallPolicy` ends it. See the module docstring for the loop
    shape and correctness model.

    `next_batch(reduce, round)` returns the round's units (or None/empty to stop);
    `condition(reduce)` is the convergence predicate over the running aggregate;
    `reduce` is a `RunningAggregate` (incremental + checkpointable); `journal` is a
    path or `RunJournal` for crash-resume; `wake` defaults to `TimerWake`; `stall`
    defaults to `WaitForever`.

    `duration_cap_seconds` is an optional wall-clock budget for the whole run: a
    `deadline` is fixed at `monotonic() + duration_cap_seconds` on entry, and the
    loop checks it at each round boundary (after the condition and max_rounds
    gates). When the cap is reached the run FINALIZES at its current state with
    outcome `"time_capped"` — exactly like `max_rounds`, never an abort — so an
    adaptive / overnight run stops cleanly within a time budget instead of running
    to `max_rounds` or forever. The cap is checked between rounds, so an
    in-flight round always completes (or stalls per `stall`) before the cap fires."""
    jrnl = journal if isinstance(journal, RunJournal) else RunJournal(journal)
    wake = wake or TimerWake()
    stall = stall or WaitForever()

    state = resume_state(jrnl.records())
    if state.aggregate is not None:
        reduce.restore(state.aggregate)
    if state.terminal == "aborted":
        return RunResult("already_aborted", state.current_round, reduce.finalize(), None)
    if state.terminal == "finalized":
        att = _try_attestation(experiment, checkpoint=False) if fetch_attestation else None
        return RunResult("already_finalized", state.current_round, reduce.finalize(), att)

    cursor = state.cursor
    folded: set[str] = set(state.folded_units)
    pinned = state.pinned_units
    round_ = state.current_round
    completed_rounds = round_
    outcome = ""
    deadline = monotonic() + duration_cap_seconds if duration_cap_seconds is not None else None

    while True:
        if condition(reduce):
            outcome = "converged"
            break
        if round_ >= max_rounds:
            outcome = "max_rounds"
            break
        if deadline is not None and monotonic() >= deadline:
            outcome = "time_capped"
            break

        # --- ensure the current round's batch is on the coordinator ---
        if pinned is not None:
            units = [Unit(u["unit_id"], u["payload"]) for u in pinned]
            pinned = None
        else:
            generated = next_batch(reduce, round_)
            if not generated:
                outcome = "exhausted"
                break
            units = list(generated)
            jrnl.record_submit(
                round_, [{"unit_id": u.unit_id, "payload": u.payload} for u in units]
            )
        _submit_idempotent(experiment, units, round_)

        # --- poll-fold this round until complete / converged / stalled ---
        pending = {u.unit_id for u in units} - folded
        stalled_polls = 0
        stalled_since = monotonic()
        round_complete = True
        while pending:
            progressed, cursor = _drain(experiment, reduce, cursor, pending, folded, include)
            if not pending:
                break  # round complete — every unit folded (the normal path)
            if condition(reduce):
                round_complete = False  # early convergence, before the round finished
                break
            if progressed:
                wake.progress()
                stalled_polls = 0
                stalled_since = monotonic()
            else:
                stalled_polls += 1
                action = stall(
                    StallContext(
                        round=round_,
                        submitted=len(units),
                        agreed=len(units) - len(pending),
                        pending=len(pending),
                        stalled_polls=stalled_polls,
                        stalled_seconds=monotonic() - stalled_since,
                        progress=_safe_progress(experiment),
                    )
                )
                if action is StallAction.ABORT:
                    outcome = "aborted"
                    round_complete = False
                    break
                if action is StallAction.FINALIZE_PARTIAL:
                    outcome = "finalized_partial"
                    round_complete = False
                    break
            wake.wait()

        if outcome in ("aborted", "finalized_partial"):
            break

        # round complete (or converged mid-round) → journal the boundary checkpoint
        jrnl.record_round_done(round_, cursor, reduce.checkpoint())
        if round_complete:
            round_ += 1
            completed_rounds = round_
        # loop; the top re-checks condition (catches mid-round convergence) + max_rounds

    return _terminate(
        outcome, round_, completed_rounds, cursor, reduce, jrnl, experiment, fetch_attestation
    )


# ---- internals -------------------------------------------------------------


def _submit_idempotent(experiment: Experiment, units: list[Unit], round_: int) -> None:
    """Submit; treat `UnitsAlreadySubmittedError` as success — on resume the round's
    pinned units may already be on the coordinator (§3.4 idempotency contract)."""
    try:
        experiment.submit_units(units, round=round_)
    except UnitsAlreadySubmittedError:
        pass


def _drain(
    experiment: Experiment,
    reduce: RunningAggregate,
    cursor: str | None,
    pending: set[str],
    folded: set[str],
    include: str,
) -> tuple[bool, str | None]:
    """Fold every not-yet-folded consensus result currently available, advancing
    the cursor through full pages. Dedupes by unit_id (the correctness guarantee);
    returns (progressed, cursor) where `progressed` is True iff a *new* unit folded."""
    progressed = False
    while True:
        page = experiment.results_page(cursor=cursor, include=include)
        for result in page.results:
            unit_id = result.get("unit_id")
            if unit_id in folded:
                continue
            reduce.fold(result)
            folded.add(unit_id)
            pending.discard(unit_id)
            progressed = True
        if not page.next_cursor:
            return progressed, cursor
        cursor = page.next_cursor


def _safe_progress(experiment: Experiment) -> Any:
    try:
        return experiment.progress()
    except CoordinatorError:
        return None


def _try_attestation(experiment: Experiment, *, checkpoint: bool) -> ResultSetAttestation | None:
    """Fetch the attestation, tolerating 'not ready' — after a full finalize the
    experiment auto-completes asynchronously, so the final attestation may 425
    briefly; the tenant can fetch it later via `Experiment.attestation()`."""
    try:
        return experiment.attestation(checkpoint=checkpoint)
    except CoordinatorError:
        return None


def _terminate(
    outcome: str,
    round_: int,
    completed_rounds: int,
    cursor: str | None,
    reduce: RunningAggregate,
    jrnl: RunJournal,
    experiment: Experiment,
    fetch_attestation: bool,
) -> RunResult:
    if outcome == "aborted":
        _safe_action(experiment.abort)
        jrnl.record_aborted()
        return RunResult("aborted", completed_rounds, reduce.finalize(), None)
    if outcome == "finalized_partial":
        # the partial-round folds weren't journaled at a boundary — persist now so
        # resume restores the exact aggregate behind the published partial result.
        jrnl.record_round_done(round_, cursor, reduce.checkpoint())
    _safe_action(experiment.finalize)
    jrnl.record_finalized()
    att = (
        _try_attestation(experiment, checkpoint=(outcome == "finalized_partial"))
        if fetch_attestation
        else None
    )
    return RunResult(outcome, completed_rounds, reduce.finalize(), att)


def _safe_action(action: Callable[[], Any]) -> None:
    """Run a lifecycle action, tolerating a conflict (already terminal)."""
    try:
        action()
    except LifecycleConflictError:
        pass
