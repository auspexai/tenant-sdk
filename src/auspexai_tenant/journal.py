"""Durable append-only run-journal for the autonomic driver (M8 §3.4).

A week-long control loop *will* crash and restart; correctness must not depend on
the process staying alive. The journal is the whole recovery story (paired with
idempotent submit): an append-only JSONL log of the loop's lifecycle events,
tenant-side, fsync'd so a record is durable before the driver acts on it.

**Round-atomic checkpointing (the load-bearing simplification, §3.4 / §10.1):**
the `{cursor, aggregate}` checkpoint is written *only at round boundaries*, not
mid-round. So resume is always from a clean boundary — the restored aggregate and
cursor reflect "everything through round R-1 folded," and the in-flight round R is
simply re-run from that boundary (re-submit its *pinned* unit_ids, re-drain its
results from the boundary cursor). The aggregate never double-counts because the
boundary checkpoint predates round R's results, and re-submit is idempotent
(`UnitsAlreadySubmittedError` treated as success). Mid-round work is cheap to
redo; correctness is trivial to reason about.

Record kinds (each one JSON object per line):
  - ``submit``     {round, units}            — written *before* the POST. Carries the
                                               full units (unit_id + payload), not just
                                               ids, because resume re-submits this round
                                               and `next_batch` may be non-deterministic
                                               (so payloads can't be regenerated). Past
                                               rounds' units are dead weight after their
                                               `round_done`; compaction is a future
                                               optimization, harmless to keep for P1.
  - ``round_done`` {round, cursor, aggregate} — round complete: the durable checkpoint
  - ``finalized``  {}                         — the loop called finalize()
  - ``aborted``    {}                         — the loop called abort()
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KIND_SUBMIT = "submit"
KIND_ROUND_DONE = "round_done"
KIND_FINALIZED = "finalized"
KIND_ABORTED = "aborted"


class RunJournal:
    """Append-only JSONL journal at `path`. `append` fsyncs each record; `records`
    replays the log. Dumb storage — the resume *semantics* are `resume_state`."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def append(self, record: Mapping[str, Any]) -> None:
        line = json.dumps(dict(record), separators=(",", ":"), sort_keys=True)
        # The journal owns its file: ensure the parent dir exists before the first
        # write. The default path is `runs/<label>/run.journal` (the shared per-run
        # layout) whose dir nothing else creates on the launch path — without this,
        # the very first record_submit (written before the POST) crashes the driver
        # with FileNotFoundError and no units ever reach the coordinator.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped:
                out.append(json.loads(stripped))
        return out

    def exists(self) -> bool:
        return self.path.exists()

    # convenience writers (keep the record vocabulary in one place) -----------

    def record_submit(self, round_: int, units: list[Mapping[str, Any]]) -> None:
        """Pin a round's units (each ``{"unit_id", "payload"}``) before the POST."""
        self.append(
            {
                "kind": KIND_SUBMIT,
                "round": round_,
                "units": [{"unit_id": u["unit_id"], "payload": u["payload"]} for u in units],
            }
        )

    def record_round_done(
        self, round_: int, cursor: str | None, aggregate: Mapping[str, Any]
    ) -> None:
        self.append(
            {
                "kind": KIND_ROUND_DONE,
                "round": round_,
                "cursor": cursor,
                "aggregate": dict(aggregate),
            }
        )

    def record_finalized(self) -> None:
        self.append({"kind": KIND_FINALIZED})

    def record_aborted(self) -> None:
        self.append({"kind": KIND_ABORTED})


@dataclass
class ResumeState:
    """The loop's restart point, reconstructed from the journal (`resume_state`)."""

    # "finalized" / "aborted" if the loop already terminated; None to continue.
    terminal: str | None = None
    # Cursor + aggregate state as of the last completed round (the boundary).
    cursor: str | None = None
    aggregate: dict[str, Any] | None = None
    # The round to (re)run next.
    current_round: int = 0
    # Set iff `current_round` was already submitted (a crash between submit and its
    # round_done): re-submit *these* pinned units (each {"unit_id","payload"})
    # rather than regenerating the batch (which next_batch may do non-deterministically).
    pinned_units: list[dict[str, Any]] | None = None
    # unit_ids already folded into the restored aggregate (all units of completed
    # rounds). The driver dedupes folds against this so re-fetched tail results
    # (the coordinator only emits next_cursor on full pages) are never double-counted.
    folded_units: set[str] = field(default_factory=set)
    # True if the journal had any records (i.e. this is a resume, not a fresh run).
    resumed: bool = field(default=False)


def resume_state(records: list[dict[str, Any]]) -> ResumeState:
    """Reconstruct the restart point from a replayed journal. Round-atomic: the
    last `round_done` gives the boundary (cursor + aggregate); the next round is
    one past it; if that round was already `submit`-ted it's re-run from its pinned
    unit_ids."""
    state = ResumeState(resumed=bool(records))
    last_done = -1
    submits: dict[int, list[dict[str, Any]]] = {}
    for rec in records:
        kind = rec.get("kind")
        if kind == KIND_SUBMIT:
            submits[int(rec["round"])] = list(rec["units"])
        elif kind == KIND_ROUND_DONE:
            r = int(rec["round"])
            if r > last_done:
                last_done = r
                state.cursor = rec.get("cursor")
                state.aggregate = rec.get("aggregate")
        elif kind == KIND_FINALIZED:
            state.terminal = "finalized"
        elif kind == KIND_ABORTED:
            state.terminal = "aborted"
    state.current_round = last_done + 1
    state.pinned_units = submits.get(state.current_round)
    # Units folded into the restored boundary aggregate = all units of rounds that
    # have a round_done (each round completes only when all its units are folded).
    state.folded_units = {
        u["unit_id"] for r, units in submits.items() if r <= last_done for u in units
    }
    return state
