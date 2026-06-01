#!/usr/bin/env python3
"""Archetype-C synthetic tenant — static weights-only geometric analysis.

The §9 #30-33 genericity-pass forcing function (sibling to ``synth_tenant``'s
integer doubler, but deliberately *maximally different*): no runtime input, the
**model is the subject**, the output is a **floating-point geometric metric**,
and consensus rides the **#33 determinism contract** rather than trivial integer
equality. See ``Documentation/AuspexAI/v0.1.0/archetype_c_tenant_design.md``.

Each work unit pins a PRNG ``seed`` in its payload. The executor loads a weight
matrix ``W`` from ``--models`` (the BYOM hook — the volunteer supplies the
weights; ``local_weights_required: true`` in the manifest declares the need) and
estimates a geometric separation statistic over ``W`` by seeded Monte-Carlo:
the mean absolute cosine similarity between ``n_samples`` random row pairs — a
neutral stand-in for the Ramsauer-Δ / causal-inner-product family.

Determinism contract (#33), the reason this passes exact-hash quorum unchanged:
  1. the seed lives in the (identical-across-replicas) payload  -> same draws;
  2. the computation is a deterministic function of (W, seed);
  3. the float metric is **canonical-quantized** before return, so honest
     hardware noise below the quantum collapses to one canonical value.

Pure standard library (no numpy) so it runs wherever the doubler runs.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path
from typing import Any

from auspexai_tenant.determinism import canonicalize_floats
from auspexai_tenant.executor import ExecutorHarness
from auspexai_tenant.workunits import WorkUnit

MODEL_FILENAME = "W.json"
QUANTIZE_PLACES = 6


def _load_weights(models_dir: Path) -> list[list[float]]:
    path = models_dir / MODEL_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"weight matrix {MODEL_FILENAME!r} not present in --models dir "
            f"({models_dir}); this tenant declares local_weights_required"
        )
    rows = json.loads(path.read_text())
    if not rows or not isinstance(rows, list):
        raise ValueError("weight matrix must be a non-empty list of rows")
    return rows


def _cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)


def run_one(unit: WorkUnit, models_dir: Path) -> dict[str, Any]:
    """Estimate a geometric separation statistic over the model's weights."""
    seed = unit.payload["seed"]
    n_samples = unit.payload["n_samples"]
    if not isinstance(seed, int) or not isinstance(n_samples, int):
        raise TypeError("payload.seed and payload.n_samples must be ints")

    weights = _load_weights(models_dir)
    n_rows = len(weights)
    if n_rows < 2:
        raise ValueError("need at least 2 rows to sample a pair")

    rng = random.Random(seed)
    acc = 0.0
    for _ in range(n_samples):
        i = rng.randrange(n_rows)
        j = rng.randrange(n_rows - 1)
        if j >= i:  # draw a distinct second index without rejection looping
            j += 1
        acc += abs(_cosine(weights[i], weights[j]))
    metric = acc / n_samples

    # #33: quantize the whole output payload so honest float noise can't break
    # exact-hash quorum. (This synthetic tenant is already bit-exact on CPU; the
    # quantization is the contract a real GPU-noisy tenant depends on.)
    return canonicalize_floats(
        {
            "mean_abs_cosine_separation": metric,
            "n_samples": n_samples,
            "n_rows": n_rows,
            "seed": seed,
        },
        places=QUANTIZE_PLACES,
    )


if __name__ == "__main__":
    sys.exit(ExecutorHarness(run_one).main())
