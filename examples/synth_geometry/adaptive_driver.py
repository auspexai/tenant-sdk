#!/usr/bin/env python3
"""Adaptive `run_until` driver for the synth-geometry tenant (M8 §3.2 example).

The static example (`make_workunits.py` + `reduce_experiment.py`) submits a fixed
batch once. This is the **autonomic** version: keep drawing batches of seeded
geometry estimates and folding their agreed metrics into a running mean, until the
estimate **stabilizes** (population std below a tolerance) or `max_rounds`. It's
the archetype-C forcing function exercising the whole control loop — submit →
poll agreed results → fold → test condition → next batch or finalize.

`build()` returns a `DriverSpec`; the CLI wires the signed `Experiment`, journal,
and wake source. Run it headless against a live coordinator (the experiment must
already be submitted + approved, with no work-units — the driver supplies them):

    auspexai-tenant experiment run <coordinator-experiment-id> \\
        --driver adaptive_driver:build \\
        --coordinator https://coord.auspexai.network --key <key> \\
        --journal synth-geometry.journal --doorbell

The same `build()` is what `tests/test_synth_geometry_adaptive.py` drives offline.
"""

from __future__ import annotations

from typing import Any

from auspexai_tenant import DriverSpec, Mean, Unit

METRIC_KEY = "mean_abs_cosine_separation"
N_SAMPLES = 2000  # Monte-Carlo samples per estimate (executor.py)
BATCH = 4  # units (fresh seeds) per round
MIN_UNITS = 12  # don't declare convergence before this many agreed estimates
STD_TOLERANCE = 0.02  # converge once the running std drops below this
MAX_ROUNDS = 25  # client guard (coordinator max_units is the hard backstop)


def _metric(result: dict[str, Any]) -> float:
    """Pull the geometry metric out of a consensus result payload."""
    return float(result["payload"][METRIC_KEY])


def _converged(agg: Mean) -> bool:
    """Stable once we have enough estimates and their spread is small."""
    return agg.count >= MIN_UNITS and agg.std < STD_TOLERANCE


def _next_batch(_agg: Mean, rnd: int) -> list[Unit]:
    """Each round draws BATCH fresh seeded estimates. Seeds are unique per
    (round, index) so every unit is distinct work (the #33 seed lives in the
    payload, shared across a unit's replicas)."""
    return [
        Unit(f"geom-r{rnd}-u{i}", {"seed": rnd * 1000 + i, "n_samples": N_SAMPLES})
        for i in range(BATCH)
    ]


def build() -> DriverSpec:
    """The DriverSpec factory named on `auspexai-tenant experiment run --driver`."""
    return DriverSpec(
        condition=_converged,
        next_batch=_next_batch,
        reduce=Mean(_metric),
        max_rounds=MAX_ROUNDS,
    )
