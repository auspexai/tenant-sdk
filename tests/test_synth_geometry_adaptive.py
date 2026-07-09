"""Drives the synth-geometry adaptive example end-to-end offline (M8 §3.2).

Loads the example's `build()` DriverSpec and runs `run_until` against a fake
coordinator that emits a geometry metric per unit — proving the example wires the
autonomic loop correctly (convergence on a stable metric, the running mean as the
published aggregate). The real-coordinator/real-executor path is `prove_consensus.py`.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from auspexai_tenant.driver import run_until
from auspexai_tenant.experiment import ResultPage, SubmitResult, UnitsAlreadySubmittedError
from auspexai_tenant.wake import TimerWake

_DRIVER_PATH = Path(__file__).parent.parent / "examples" / "synth_geometry" / "adaptive_driver.py"


def _load_build():
    spec = importlib.util.spec_from_file_location("adaptive_driver", _DRIVER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeometryFake:
    """A fake coordinator that emits `metric` as each unit's consensus geometry
    result, so the running mean is exactly `metric` and the std is 0 (converges
    as soon as MIN_UNITS estimates have agreed)."""

    def __init__(self, metric: float) -> None:
        self.experiment_id = "exp-geom"
        self._metric = metric
        self._results: list[dict] = []
        self._seen: set[str] = set()
        self.finalized = False

    def submit_units(self, units, *, round=None) -> SubmitResult:
        new = [u for u in units if u.unit_id not in self._seen]
        if not new:
            raise UnitsAlreadySubmittedError(409, "already submitted")
        for u in new:
            self._seen.add(u.unit_id)
            i = len(self._results)
            self._results.append(
                {
                    "unit_id": u.unit_id,
                    "result_id": f"res-{u.unit_id}",
                    "completed_at": f"2026-01-01T00:00:{i:02d}+00:00",
                    "payload": {"mean_abs_cosine_separation": self._metric},
                }
            )
        return SubmitResult([u.unit_id for u in new], len(new))

    def results_page(self, *, cursor=None, include="consensus") -> ResultPage:
        return ResultPage(results=list(self._results), next_cursor=None)

    def progress(self):
        return None

    def finalize(self) -> dict:
        self.finalized = True
        return {}

    def abort(self) -> dict:
        return {}

    def driver_heartbeat(self, status, *, reason=None, round=None, run_id=None) -> None:
        pass

    def attestation(self, *, checkpoint=False):
        return None


def test_adaptive_example_converges(tmp_path) -> None:
    build = _load_build().build
    spec = build()
    fake = GeometryFake(metric=0.42)
    result = run_until(
        fake,
        condition=spec.condition,
        next_batch=spec.next_batch,
        reduce=spec.reduce,
        journal=tmp_path / "geom.journal",
        wake=TimerWake(sleep=lambda _d: None),
        max_rounds=spec.max_rounds,
    )
    assert result.outcome == "converged"
    assert result.aggregate["mean"] == pytest.approx(0.42)
    assert result.aggregate["count"] >= 12  # MIN_UNITS
    assert fake.finalized is True


def test_build_returns_a_usable_spec() -> None:
    spec = _load_build().build()
    assert spec.max_rounds == 25
    # next_batch yields fresh, uniquely-id'd seeded units per round
    batch0 = spec.next_batch(spec.reduce, 0)
    batch1 = spec.next_batch(spec.reduce, 1)
    ids0 = {u.unit_id for u in batch0}
    ids1 = {u.unit_id for u in batch1}
    assert len(ids0) == len(batch0)  # unique within a round
    assert ids0.isdisjoint(ids1)  # disjoint across rounds
    assert all("seed" in u.payload for u in batch0)
