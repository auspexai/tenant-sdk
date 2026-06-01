#!/usr/bin/env python3
"""Experiment-level cross-unit reduce (§9 #34, Phase-1 tenant-side).

The per-unit consensus gives one agreed metric per unit. The *research output*
is a single global statistic over all units — here a "geometry fingerprint" of
the model: the grand mean +/- spread of the per-unit separation metric, plus a
coarse histogram. This runs **on the researcher's machine, model-blind to the
coordinator** (the Phase-1 answer in ``aggregate_reduction_and_control_loop_design.md``
§6.2); the coordinator only attests the input set.

Usable two ways:
  - import ``reduce_metrics(values)`` (what the proof + a real driver call);
  - CLI: ``python reduce_experiment.py <dir-of-result-*.json>`` to reduce a
    directory of consensus result payloads.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

METRIC_KEY = "mean_abs_cosine_separation"


def reduce_metrics(values: list[float], *, bins: int = 8) -> dict[str, Any]:
    """Fold per-unit metrics into one global geometry fingerprint."""
    if not values:
        raise ValueError("no metrics to reduce")
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    lo, hi = min(values), max(values)
    # Coarse histogram over [lo, hi].
    width = (hi - lo) or 1.0
    counts = [0] * bins
    for v in values:
        idx = min(bins - 1, int((v - lo) / width * bins))
        counts[idx] += 1
    return {
        "unit_count": n,
        "grand_mean": mean,
        "stdev": math.sqrt(var),
        "min": lo,
        "max": hi,
        "histogram": counts,
    }


def _load_dir(d: Path) -> list[float]:
    values: list[float] = []
    for path in sorted(d.glob("result_*.json")):
        payload = json.loads(path.read_text())
        # Accept either a raw payload or a {payload: {...}} envelope.
        payload = payload.get("payload", payload)
        values.append(float(payload[METRIC_KEY]))
    return values


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if len(argv) != 1:
        print("usage: reduce_experiment.py <dir-of-result-*.json>", file=sys.stderr)
        return 2
    fingerprint = reduce_metrics(_load_dir(Path(argv[0])))
    print(json.dumps(fingerprint, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
