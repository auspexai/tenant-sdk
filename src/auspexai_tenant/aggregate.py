"""Incremental result aggregators for the autonomic driver (M8 §3.3).

The control loop folds *agreed consensus results* into a running aggregate that
feeds the termination `condition` and becomes the published statistic. Per the
design note (§3.3, §10.1) the live aggregator must be **incremental** — fold one
result at a time in O(state) memory, never re-reduce the whole set each poll — and
**checkpointable** so a crashed driver resumes exactly-once from the run-journal.
Implementations here are also **commutative** (order-independent), which P1 does
not strictly require for a single cursor-ordered fold but which is mandatory for
the future parallel / round.progress / Phase-2 coordinator-side reduces — so the
batteries-included aggregators are written that way from day one.

`fold` receives one consensus result item (the `api/results` `ResultItem` wire
shape — `{unit_id, payload, result_hash, receipt_id, ...}`); each aggregator is
constructed with a small extractor that pulls its value out of `result["payload"]`
(the tenant's opaque executor output), keeping the aggregators science-agnostic.
"""

from __future__ import annotations

import bisect
import itertools
import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

# A consensus result item — the coordinator's `ResultItem` shape (see
# `platform .../api/results.py`); opaque to these aggregators beyond the extractor.
ConsensusResult = Mapping[str, Any]


@runtime_checkable
class RunningAggregate(Protocol):
    """The incremental-fold contract the driver's `reduce` must satisfy (§3.3).

    `fold` absorbs one agreed result; `checkpoint`/`restore` round-trip the running
    state through the run-journal (JSON-serializable); `finalize` returns the
    published statistic. Should be commutative — see the module docstring.
    """

    def fold(self, result: ConsensusResult) -> None: ...

    def checkpoint(self) -> dict[str, Any]: ...

    def restore(self, state: Mapping[str, Any]) -> None: ...

    def finalize(self) -> Any: ...


class Counter:
    """Counts folded results, optionally bucketed by a key derived from each
    result (a categorical tally). Commutative and exact."""

    def __init__(self, *, bucket: Callable[[ConsensusResult], Hashable] | None = None) -> None:
        self._bucket = bucket
        self._total = 0
        self._counts: dict[str, int] = {}

    @property
    def total(self) -> int:
        return self._total

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def fold(self, result: ConsensusResult) -> None:
        self._total += 1
        if self._bucket is not None:
            label = str(self._bucket(result))
            self._counts[label] = self._counts.get(label, 0) + 1

    def checkpoint(self) -> dict[str, Any]:
        return {"total": self._total, "counts": dict(self._counts)}

    def restore(self, state: Mapping[str, Any]) -> None:
        self._total = int(state.get("total", 0))
        self._counts = {str(k): int(v) for k, v in (state.get("counts") or {}).items()}

    def finalize(self) -> dict[str, Any]:
        out: dict[str, Any] = {"total": self._total}
        if self._bucket is not None:
            out["counts"] = dict(self._counts)
        return out


class Mean:
    """Running count / mean / population variance of a numeric value extracted
    from each result. Kept as count + sum + sum-of-squares: exactly commutative
    and trivially checkpointable. `mean`/`variance`/`std` are live properties for
    use in a termination `condition`."""

    def __init__(self, value: Callable[[ConsensusResult], float]) -> None:
        self._value = value
        self._count = 0
        self._sum = 0.0
        self._sumsq = 0.0

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._sum / self._count if self._count else 0.0

    @property
    def variance(self) -> float:
        if self._count == 0:
            return 0.0
        m = self.mean
        return max(0.0, self._sumsq / self._count - m * m)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def fold(self, result: ConsensusResult) -> None:
        x = float(self._value(result))
        self._count += 1
        self._sum += x
        self._sumsq += x * x

    def checkpoint(self) -> dict[str, Any]:
        return {"count": self._count, "sum": self._sum, "sumsq": self._sumsq}

    def restore(self, state: Mapping[str, Any]) -> None:
        self._count = int(state.get("count", 0))
        self._sum = float(state.get("sum", 0.0))
        self._sumsq = float(state.get("sumsq", 0.0))

    def finalize(self) -> dict[str, Any]:
        return {
            "count": self._count,
            "mean": self.mean,
            "variance": self.variance,
            "std": self.std,
        }


class Histogram:
    """Counts a numeric value into fixed half-open bins ``[edges[i], edges[i+1])``,
    with underflow (< edges[0]) and overflow (>= edges[-1]) tallies. Commutative.
    `edges` must be strictly ascending; the histogram has ``len(edges) - 1`` bins."""

    def __init__(self, edges: Sequence[float], value: Callable[[ConsensusResult], float]) -> None:
        if len(edges) < 2:
            raise ValueError("histogram needs at least 2 edges (1 bin)")
        if any(b <= a for a, b in itertools.pairwise(edges)):
            raise ValueError("histogram edges must be strictly ascending")
        self._edges = [float(e) for e in edges]
        self._value = value
        self._counts = [0] * (len(self._edges) - 1)
        self._underflow = 0
        self._overflow = 0

    @property
    def total(self) -> int:
        return sum(self._counts) + self._underflow + self._overflow

    @property
    def counts(self) -> list[int]:
        return list(self._counts)

    def fold(self, result: ConsensusResult) -> None:
        x = float(self._value(result))
        idx = bisect.bisect_right(self._edges, x) - 1
        if idx < 0:
            self._underflow += 1
        elif idx >= len(self._counts):
            self._overflow += 1
        else:
            self._counts[idx] += 1

    def checkpoint(self) -> dict[str, Any]:
        return {
            "counts": list(self._counts),
            "underflow": self._underflow,
            "overflow": self._overflow,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        counts = list(state.get("counts") or [])
        if len(counts) != len(self._counts):
            raise ValueError(
                f"checkpoint has {len(counts)} bins but this histogram has "
                f"{len(self._counts)} — edges must match the checkpointed histogram"
            )
        self._counts = [int(c) for c in counts]
        self._underflow = int(state.get("underflow", 0))
        self._overflow = int(state.get("overflow", 0))

    def finalize(self) -> dict[str, Any]:
        return {
            "edges": list(self._edges),
            "counts": list(self._counts),
            "underflow": self._underflow,
            "overflow": self._overflow,
            "total": self.total,
        }
